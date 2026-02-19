"""ATR-based trailing stop management"""

import logging
from typing import Optional

from bot.execution.position import Position
from bot.execution.models import OrderSide


logger = logging.getLogger("trading_bot.execution.trailing")


class TrailingStopManager:
    """
    ATR-based trailing stop manager.

    Manages trailing stops that:
    1. Enable after position reaches trail_after_r profit (e.g., 1.0R)
    2. Trail at atr_trail_mult * ATR distance from current price
    3. Only move in favorable direction (never worsen stop)
    4. Update on 5m candle closes or significant price moves

    Example:
        - Position: LONG @ $50,000, stop @ $49,500 (1% stop, $500 risk)
        - trail_after_r = 1.0 (enable trailing after 1R profit = $500)
        - atr_trail_mult = 2.0, ATR = $300
        - Trail distance = 2.0 * $300 = $600

        When price reaches $50,500:
        - PnL = $500 = 1.0R → Enable trailing
        - New stop = $50,500 - $600 = $49,900
        - Stop moved from $49,500 → $49,900 (locked in $400 profit)

        When price reaches $51,000:
        - New stop = $51,000 - $600 = $50,400
        - Stop moved from $49,900 → $50,400 (locked in $900 profit)
    """

    def __init__(self):
        """Initialize trailing stop manager"""
        logger.info("TrailingStopManager initialized")

    def update_trailing_stop(
        self,
        position: Position,
        current_price: float,
        atr: float
    ) -> Optional[float]:
        """
        Update trailing stop if conditions met.

        Args:
            position: Position to update
            current_price: Current market price
            atr: Current ATR value

        Returns:
            New stop price if updated, None if no update

        Logic:
            1. Check if trailing should be enabled (pnl_r >= trail_after_r)
            2. Calculate new stop price based on ATR
            3. Only update if new stop is better than current stop
            4. Update position.stop_price and position.trailing_enabled
        """
        # Update unrealized PnL
        position.update_unrealized_pnl(current_price)

        # Update highest/lowest price seen
        position.update_highest_price_seen(current_price)

        # Check if trailing should be enabled
        if not position.trailing_enabled:
            if position.should_enable_trailing():
                position.trailing_enabled = True
                logger.info(
                    f"Position {position.position_id} ({position.symbol}): "
                    f"Trailing enabled at {position.pnl_r:.2f}R profit"
                )
            else:
                # Not profitable enough yet
                return None

        # Calculate new trailing stop
        new_stop = self._calculate_trailing_stop(
            position=position,
            current_price=current_price,
            atr=atr
        )

        # Check if new stop is better than current stop
        if self._should_update_stop(position, new_stop):
            old_stop = position.stop_price
            position.stop_price = new_stop

            logger.info(
                f"Position {position.position_id} ({position.symbol}): "
                f"Trailing stop updated {old_stop:.2f} → {new_stop:.2f} "
                f"(price: {current_price:.2f}, ATR: {atr:.2f})"
            )

            return new_stop

        return None

    def _calculate_trailing_stop(
        self,
        position: Position,
        current_price: float,
        atr: float
    ) -> float:
        """
        Calculate new trailing stop price.

        Args:
            position: Position
            current_price: Current price
            atr: Current ATR

        Returns:
            New stop price
        """
        # Calculate trail distance
        trail_distance = atr * position.atr_trail_mult

        if position.is_long:
            # LONG: Stop trails below current price
            new_stop = current_price - trail_distance
        else:
            # SHORT: Stop trails above current price
            new_stop = current_price + trail_distance

        return new_stop

    def _should_update_stop(self, position: Position, new_stop: float) -> bool:
        """
        Check if stop should be updated.

        Args:
            position: Position
            new_stop: Proposed new stop price

        Returns:
            True if stop should be updated

        Logic:
            - LONG: Only move stop UP (new_stop > current_stop)
            - SHORT: Only move stop DOWN (new_stop < current_stop)
        """
        current_stop = position.stop_price

        if position.is_long:
            # LONG: Only move stop up (never down)
            return new_stop > current_stop
        else:
            # SHORT: Only move stop down (never up)
            return new_stop < current_stop

    def check_stop_hit(self, position: Position, current_price: float) -> bool:
        """
        Check if stop loss has been hit.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            True if stop has been hit
        """
        if position.is_long:
            # LONG: Stop hit if price <= stop_price
            return current_price <= position.stop_price
        else:
            # SHORT: Stop hit if price >= stop_price
            return current_price >= position.stop_price

    def check_tp_hit(self, position: Position, current_price: float) -> bool:
        """
        Check if take profit has been hit.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            True if TP has been hit
        """
        if position.tp_price is None:
            return False

        if position.is_long:
            # LONG: TP hit if price >= tp_price
            return current_price >= position.tp_price
        else:
            # SHORT: TP hit if price <= tp_price
            return current_price <= position.tp_price

    def get_stop_distance_r(self, position: Position, current_price: float) -> float:
        """
        Get distance to stop in R multiples.

        Args:
            position: Position
            current_price: Current price

        Returns:
            Distance to stop in R (negative if stop would cause loss)
        """
        # Update PnL
        position.update_unrealized_pnl(current_price)

        # Calculate what PnL would be if stopped out now
        price_diff_to_stop = position.stop_price - position.entry_price

        if position.is_short:
            price_diff_to_stop = -price_diff_to_stop

        pnl_at_stop = price_diff_to_stop * position.quantity

        # Convert to R
        if position.risk_amount_usd == 0:
            return 0.0

        return pnl_at_stop / position.risk_amount_usd

    def get_tp_distance_r(self, position: Position) -> Optional[float]:
        """
        Get distance to TP in R multiples.

        Args:
            position: Position

        Returns:
            Distance to TP in R, or None if no TP set
        """
        if position.tp_price is None:
            return None

        # Calculate PnL at TP
        price_diff_to_tp = position.tp_price - position.entry_price

        if position.is_short:
            price_diff_to_tp = -price_diff_to_tp

        pnl_at_tp = price_diff_to_tp * position.quantity

        # Convert to R
        if position.risk_amount_usd == 0:
            return 0.0

        return pnl_at_tp / position.risk_amount_usd

    def should_update_on_price_move(
        self,
        position: Position,
        price_move_pct: float,
        threshold_pct: float = 0.005  # 0.5% default
    ) -> bool:
        """
        Check if trailing should update based on price move.

        Args:
            position: Position
            price_move_pct: Price move percentage since last update
            threshold_pct: Threshold for significant move (default: 0.5%)

        Returns:
            True if should update trailing stop
        """
        if not position.trailing_enabled:
            return False

        return abs(price_move_pct) >= threshold_pct
