"""Exit management with multiple exit reasons"""

import logging
from typing import Optional, Callable
from datetime import datetime

from bot.execution.position import Position, ExitReason
from bot.execution.trailing_stop import TrailingStopManager
from bot.execution.order_manager import OrderManager
from bot.execution.models import OrderSide, OrderPurpose


logger = logging.getLogger("trading_bot.execution.exit")


class ExitManager:
    """
    Manages position exits with multiple exit reasons.

    Exit Reasons:
        - TP: Take profit hit
        - SL: Stop loss hit (initial or trailing)
        - TRAIL: Trailing stop hit
        - KILL_SWITCH: Emergency exit (risk limits)
        - MANUAL: User-initiated close
        - TIMEOUT: Max holding period exceeded

    Responsibilities:
        1. Monitor positions for exit conditions
        2. Update trailing stops on candle closes
        3. Execute exit orders (market orders)
        4. Track exit reasons for analytics
        5. Handle emergency exits (kill switch)
    """

    def __init__(
        self,
        order_manager: OrderManager,
        trailing_manager: Optional[TrailingStopManager] = None
    ):
        """
        Initialize ExitManager.

        Args:
            order_manager: OrderManager for placing exit orders
            trailing_manager: TrailingStopManager (creates new if None)
        """
        self.order_manager = order_manager
        self.trailing_manager = trailing_manager or TrailingStopManager()

        # Callbacks for exit events
        self._exit_callbacks = []

        logger.info("ExitManager initialized")

    def check_and_exit(
        self,
        position: Position,
        current_price: float,
        atr: Optional[float] = None,
        force_exit: bool = False,
        exit_reason: Optional[ExitReason] = None
    ) -> Optional[ExitReason]:
        """
        Check exit conditions and execute if met.

        Args:
            position: Position to check
            current_price: Current market price
            atr: Current ATR (for trailing updates)
            force_exit: Force exit regardless of conditions
            exit_reason: Exit reason if force_exit=True

        Returns:
            ExitReason if exit executed, None otherwise
        """
        if not position.is_open:
            return None

        # Force exit (kill switch, manual, etc.)
        if force_exit:
            reason = exit_reason or ExitReason.MANUAL
            logger.warning(
                f"Force exit triggered for {position.position_id} ({position.symbol}): "
                f"{reason.value}"
            )
            self._execute_exit(position, current_price, reason)
            return reason

        # Update trailing stop if ATR available
        if atr is not None and position.trailing_enabled:
            updated_stop = self.trailing_manager.update_trailing_stop(
                position=position,
                current_price=current_price,
                atr=atr
            )

            if updated_stop:
                # Update stop order on exchange
                self._update_stop_order(position, updated_stop)

        # Check TP hit
        if self.trailing_manager.check_tp_hit(position, current_price):
            logger.info(
                f"Position {position.position_id} ({position.symbol}): "
                f"TP hit at {current_price:.2f}"
            )
            self._execute_exit(position, current_price, ExitReason.TP)
            return ExitReason.TP

        # Check stop hit
        if self.trailing_manager.check_stop_hit(position, current_price):
            # Determine if trailing stop or initial stop
            if position.trailing_enabled and position.stop_price > position.initial_stop_price:
                reason = ExitReason.TRAIL
                logger.info(
                    f"Position {position.position_id} ({position.symbol}): "
                    f"Trailing stop hit at {current_price:.2f}"
                )
            else:
                reason = ExitReason.SL
                logger.info(
                    f"Position {position.position_id} ({position.symbol}): "
                    f"Stop loss hit at {current_price:.2f}"
                )

            self._execute_exit(position, current_price, reason)
            return reason

        # No exit condition met
        return None

    def manual_exit(
        self,
        position: Position,
        current_price: float
    ) -> None:
        """
        Manually exit position.

        Args:
            position: Position to exit
            current_price: Current market price
        """
        logger.info(
            f"Manual exit requested for {position.position_id} ({position.symbol})"
        )
        self._execute_exit(position, current_price, ExitReason.MANUAL)

    def kill_switch_exit(
        self,
        position: Position,
        current_price: float
    ) -> None:
        """
        Emergency exit (kill switch activation).

        Args:
            position: Position to exit
            current_price: Current market price
        """
        logger.warning(
            f"KILL SWITCH: Emergency exit for {position.position_id} ({position.symbol})"
        )
        self._execute_exit(position, current_price, ExitReason.KILL_SWITCH)

    def timeout_exit(
        self,
        position: Position,
        current_price: float,
        max_holding_seconds: float
    ) -> Optional[ExitReason]:
        """
        Exit position if max holding time exceeded.

        Args:
            position: Position to check
            current_price: Current market price
            max_holding_seconds: Max holding time in seconds

        Returns:
            ExitReason.TIMEOUT if exited, None otherwise
        """
        holding_time = position.holding_time_seconds

        if holding_time >= max_holding_seconds:
            logger.info(
                f"Position {position.position_id} ({position.symbol}): "
                f"Timeout exit after {holding_time:.0f}s (max: {max_holding_seconds:.0f}s)"
            )
            self._execute_exit(position, current_price, ExitReason.TIMEOUT)
            return ExitReason.TIMEOUT

        return None

    def update_trailing_on_candle_close(
        self,
        position: Position,
        close_price: float,
        atr: float
    ) -> Optional[float]:
        """
        Update trailing stop on 5m candle close.

        Args:
            position: Position to update
            close_price: Candle close price
            atr: Current ATR

        Returns:
            New stop price if updated, None otherwise
        """
        if not position.is_open:
            return None

        new_stop = self.trailing_manager.update_trailing_stop(
            position=position,
            current_price=close_price,
            atr=atr
        )

        if new_stop:
            self._update_stop_order(position, new_stop)

        return new_stop

    def _execute_exit(
        self,
        position: Position,
        exit_price: float,
        exit_reason: ExitReason
    ) -> None:
        """
        Execute position exit.

        Args:
            position: Position to exit
            exit_price: Exit price
            exit_reason: Exit reason
        """
        # Place market exit order
        try:
            exit_side = OrderSide.SHORT if position.is_long else OrderSide.LONG

            client_order_id = f"{position.position_id}_exit_{exit_reason.value}"

            exit_order = self.order_manager.place_market_order(
                symbol=position.symbol,
                side=exit_side,
                quantity=position.quantity,
                purpose=OrderPurpose.EXIT,
                client_order_id=client_order_id,
                position_id=position.position_id,
                reduce_only=True
            )

            logger.info(
                f"Exit order placed for {position.position_id}: "
                f"{exit_order.client_order_id} ({exit_reason.value})"
            )

            # Cancel stop/TP orders
            self._cancel_exit_orders(position)

            # Close position (assume market order fills immediately)
            position.close_position(
                exit_price=exit_price,
                exit_reason=exit_reason,
                fees_paid=0.0  # Will be updated from actual fill
            )

            # Trigger callbacks
            self._trigger_exit_callbacks(position, exit_reason)

        except Exception as e:
            logger.error(
                f"Failed to execute exit for {position.position_id}: {e}",
                exc_info=True
            )
            raise

    def _update_stop_order(self, position: Position, new_stop_price: float) -> None:
        """
        Update stop order on exchange.

        Args:
            position: Position
            new_stop_price: New stop price
        """
        try:
            # Cancel old stop order
            old_stop_order = self.order_manager.get_order(position.stop_order_id)
            if old_stop_order and old_stop_order.is_open:
                self.order_manager.cancel_order(old_stop_order, reason="Trailing update")

            # Place new stop order
            stop_side = OrderSide.SHORT if position.is_long else OrderSide.LONG

            new_stop_order_id = f"{position.position_id}_stop_trail_{int(datetime.utcnow().timestamp())}"

            new_stop_order = self.order_manager.place_stop_order(
                symbol=position.symbol,
                side=stop_side,
                quantity=position.quantity,
                stop_price=new_stop_price,
                client_order_id=new_stop_order_id,
                position_id=position.position_id,
                reduce_only=True
            )

            # Update position tracking
            position.stop_order_id = new_stop_order.client_order_id

            logger.debug(
                f"Stop order updated for {position.position_id}: {new_stop_price:.2f}"
            )

        except Exception as e:
            logger.error(
                f"Failed to update stop order for {position.position_id}: {e}",
                exc_info=True
            )

    def _cancel_exit_orders(self, position: Position) -> None:
        """
        Cancel stop and TP orders.

        Args:
            position: Position
        """
        # Cancel stop order
        if position.stop_order_id:
            stop_order = self.order_manager.get_order(position.stop_order_id)
            if stop_order and stop_order.is_open:
                try:
                    self.order_manager.cancel_order(stop_order, reason="Position closed")
                except Exception as e:
                    logger.error(f"Failed to cancel stop order: {e}")

        # Cancel TP order
        if position.tp_order_id:
            tp_order = self.order_manager.get_order(position.tp_order_id)
            if tp_order and tp_order.is_open:
                try:
                    self.order_manager.cancel_order(tp_order, reason="Position closed")
                except Exception as e:
                    logger.error(f"Failed to cancel TP order: {e}")

    def register_exit_callback(
        self,
        callback: Callable[[Position, ExitReason], None]
    ) -> None:
        """
        Register callback for position exits.

        Args:
            callback: Callback function receiving (position, exit_reason)
        """
        self._exit_callbacks.append(callback)
        logger.debug("Exit callback registered")

    def _trigger_exit_callbacks(
        self,
        position: Position,
        exit_reason: ExitReason
    ) -> None:
        """
        Trigger exit callbacks.

        Args:
            position: Position that exited
            exit_reason: Exit reason
        """
        for callback in self._exit_callbacks:
            try:
                callback(position, exit_reason)
            except Exception as e:
                logger.error(f"Error in exit callback: {e}", exc_info=True)

    def get_exit_summary(self, position: Position) -> dict:
        """
        Get exit summary for closed position.

        Args:
            position: Closed position

        Returns:
            Exit summary dictionary
        """
        if position.is_open:
            return {"error": "Position still open"}

        return {
            "position_id": position.position_id,
            "symbol": position.symbol,
            "side": position.side.value,
            "entry_price": position.entry_price,
            "exit_price": position.exit_price,
            "exit_reason": position.exit_reason.value if position.exit_reason else None,
            "pnl_usd": position.realized_pnl_usd,
            "pnl_r": position.realized_pnl_usd / position.risk_amount_usd if position.risk_amount_usd > 0 else 0,
            "holding_time_seconds": position.holding_time_seconds,
            "trailing_enabled": position.trailing_enabled,
            "initial_stop": position.initial_stop_price,
            "final_stop": position.stop_price,
            "fees_paid": position.fees_paid_usd,
            "funding_paid": position.funding_paid_usd
        }
