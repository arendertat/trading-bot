"""Startup reconciliation: rebuild internal state from exchange"""

import logging
from datetime import datetime
from typing import List, Optional

from bot.execution.models import Order, OrderStatus, OrderSide, OrderType, OrderPurpose
from bot.execution.position import Position, PositionStatus
from bot.execution.order_lifecycle import OrderLifecycle
from bot.state.state_manager import StateManager


logger = logging.getLogger("trading_bot.state.reconciler")

# Client order ID suffixes that identify stop/TP orders
_EXIT_ORDER_SUFFIXES = ("_stop", "_tp", "_trail_", "_exit_")


class ReconciliationResult:
    """
    Summary of a reconciliation run.

    Attributes:
        positions_restored: Positions rebuilt from exchange
        orders_linked: Stop/TP orders linked to positions
        orphan_orders_cancelled: Orders with no matching position, cancelled
        unknown_orders_skipped: Orders that couldn't be classified
        errors: Any errors encountered (non-fatal)
        timestamp: When reconciliation ran
    """

    def __init__(self) -> None:
        self.positions_restored: int = 0
        self.orders_linked: int = 0
        self.orphan_orders_cancelled: int = 0
        self.unknown_orders_skipped: int = 0
        self.errors: List[str] = []
        self.timestamp: datetime = datetime.utcnow()

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "positions_restored": self.positions_restored,
            "orders_linked": self.orders_linked,
            "orphan_orders_cancelled": self.orphan_orders_cancelled,
            "unknown_orders_skipped": self.unknown_orders_skipped,
            "errors": self.errors,
        }

    def __str__(self) -> str:
        return (
            f"ReconciliationResult("
            f"positions={self.positions_restored}, "
            f"orders_linked={self.orders_linked}, "
            f"orphans_cancelled={self.orphan_orders_cancelled}, "
            f"errors={len(self.errors)})"
        )


class Reconciler:
    """
    Startup reconciler — rebuilds internal state from exchange data.

    Steps performed on reconcile():
        1. Fetch open positions from exchange
        2. Rebuild Position objects and add to StateManager
        3. Fetch all open orders from exchange
        4. Link stop/TP orders to their positions
        5. Cancel orphan orders (no associated position)
        6. Log reconciliation summary

    Idempotency:
        Safe to call multiple times; positions/orders already in
        StateManager are updated (upserted), not duplicated.
    """

    def __init__(
        self,
        exchange_client,
        state_manager: StateManager,
        lifecycle: Optional[OrderLifecycle] = None,
    ) -> None:
        """
        Initialise Reconciler.

        Args:
            exchange_client: BinanceFuturesClient (or compatible mock)
            state_manager: StateManager to populate
            lifecycle: OrderLifecycle (creates new if None)
        """
        self.client = exchange_client
        self.state = state_manager
        self.lifecycle = lifecycle or OrderLifecycle()
        logger.info("Reconciler initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconcile(self, symbols: Optional[List[str]] = None) -> ReconciliationResult:
        """
        Perform startup reconciliation.

        Args:
            symbols: Symbols to reconcile (None = all)

        Returns:
            ReconciliationResult summary
        """
        result = ReconciliationResult()
        logger.info("Starting reconciliation…")

        # Step 1 & 2: positions
        self._reconcile_positions(result, symbols)

        # Step 3-5: orders
        self._reconcile_orders(result, symbols)

        logger.info(f"Reconciliation complete: {result}")
        return result

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def _reconcile_positions(
        self,
        result: ReconciliationResult,
        symbols: Optional[List[str]],
    ) -> None:
        """Fetch open positions and rebuild in StateManager."""
        try:
            raw_positions = self.client.fetch_positions(symbols=symbols)
        except Exception as e:
            msg = f"Failed to fetch positions: {e}"
            logger.error(msg)
            result.errors.append(msg)
            return

        for raw in raw_positions:
            try:
                position = self._position_from_exchange(raw)
                if position is None:
                    continue   # Zero-size position, skip
                self.state.update_position(position)
                result.positions_restored += 1
                logger.info(
                    f"Position restored: {position.position_id} "
                    f"({position.symbol} {position.side.value} "
                    f"qty={position.quantity} @ {position.entry_price})"
                )
            except Exception as e:
                msg = f"Error restoring position {raw}: {e}"
                logger.error(msg)
                result.errors.append(msg)

    def _position_from_exchange(self, raw: dict) -> Optional[Position]:
        """
        Build a Position from raw exchange position data.

        Args:
            raw: Exchange position dict (ccxt format)

        Returns:
            Position or None if zero-size
        """
        qty = abs(float(raw.get("contracts", 0) or raw.get("positionAmt", 0) or 0))
        if qty == 0:
            return None

        symbol: str = raw.get("symbol", "")
        side_str: str = raw.get("side", "").upper()
        side = OrderSide.LONG if side_str in ("LONG", "BUY") else OrderSide.SHORT
        entry_price = float(raw.get("entryPrice", 0) or raw.get("entry_price", 0))
        leverage = float(raw.get("leverage", 1) or 1)
        notional = entry_price * qty
        margin = notional / leverage

        # Build deterministic position ID from symbol + side + entry_price
        # (exchange doesn't have a position UUID, so we derive one)
        position_id = f"{symbol}_{side.value}_{int(entry_price)}_{int(datetime.utcnow().timestamp())}"

        position = Position(
            position_id=position_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=qty,
            notional_usd=notional,
            leverage=leverage,
            margin_usd=margin,
            stop_price=0.0,          # Will be updated when stop order found
            entry_time=datetime.utcnow(),
            risk_amount_usd=0.0,     # Will be updated when stop order linked
            initial_stop_price=0.0,
            trail_after_r=1.0,       # Conservative defaults
            atr_trail_mult=2.0,
            entry_order_id="unknown_recovered",
            stop_order_id="unknown_recovered",
            metadata={"recovered": True},
        )
        return position

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def _reconcile_orders(
        self,
        result: ReconciliationResult,
        symbols: Optional[List[str]],
    ) -> None:
        """Fetch open orders, link to positions, cancel orphans."""
        try:
            if symbols:
                raw_orders = []
                for sym in symbols:
                    raw_orders.extend(self.client.fetch_open_orders(sym))
            else:
                raw_orders = self.client.fetch_open_orders()
        except Exception as e:
            msg = f"Failed to fetch open orders: {e}"
            logger.error(msg)
            result.errors.append(msg)
            return

        for raw in raw_orders:
            try:
                order = self._order_from_exchange(raw)
                self.state.add_order(order)

                if self._is_exit_order(order):
                    linked = self._link_order_to_position(order)
                    if linked:
                        result.orders_linked += 1
                    else:
                        # Exit order with no matching position → cancel
                        self._cancel_orphan(order, result)
                else:
                    # Entry or unknown order with no position → cancel
                    pos = self.state.get_open_positions(order.symbol)
                    if not pos:
                        self._cancel_orphan(order, result)
                    else:
                        result.orders_linked += 1

            except Exception as e:
                msg = f"Error processing order {raw}: {e}"
                logger.error(msg)
                result.errors.append(msg)
                result.unknown_orders_skipped += 1

    def _order_from_exchange(self, raw: dict) -> Order:
        """Build an Order from raw exchange order data."""
        side_str = raw.get("side", "").upper()
        side = OrderSide.LONG if side_str == "BUY" else OrderSide.SHORT

        type_str = raw.get("type", "").upper()
        if "STOP_MARKET" in type_str:
            order_type = OrderType.STOP_MARKET
        elif "TAKE_PROFIT" in type_str:
            order_type = OrderType.TAKE_PROFIT_MARKET
        elif "MARKET" in type_str:
            order_type = OrderType.MARKET
        else:
            order_type = OrderType.LIMIT

        client_order_id: str = raw.get("clientOrderId") or raw.get("client_order_id", "")
        purpose = (
            OrderPurpose.STOP if "stop" in client_order_id.lower()
            else OrderPurpose.TAKE_PROFIT if "_tp" in client_order_id.lower()
            else OrderPurpose.ENTRY
        )

        order = Order(
            symbol=raw.get("symbol", ""),
            side=side,
            order_type=order_type,
            purpose=purpose,
            quantity=float(raw.get("origQty", 0) or raw.get("amount", 0) or 0),
            price=float(raw.get("price", 0) or 0) or None,
            stop_price=float(raw.get("stopPrice", 0) or 0) or None,
            client_order_id=client_order_id,
            exchange_order_id=raw.get("orderId") or raw.get("id"),
            filled_quantity=float(raw.get("executedQty", 0) or 0),
            timestamp_created=datetime.utcnow(),
            timestamp_submitted=datetime.utcnow(),
        )

        # Mark as OPEN in lifecycle
        order.status = OrderStatus.OPEN
        return order

    def _is_exit_order(self, order: Order) -> bool:
        """Return True if order is a stop or TP order."""
        if order.order_type in (OrderType.STOP_MARKET, OrderType.TAKE_PROFIT_MARKET):
            return True
        cid = order.client_order_id.lower()
        return any(cid.endswith(sfx) or sfx in cid for sfx in _EXIT_ORDER_SUFFIXES)

    def _link_order_to_position(self, order: Order) -> bool:
        """
        Link exit order to its position using client_order_id convention.

        Convention: client_order_id = "<position_id>_stop" or "<position_id>_tp"

        Returns:
            True if linked successfully
        """
        cid = order.client_order_id
        # Derive position_id by stripping known suffixes
        position_id = None
        for sfx in ("_stop", "_tp"):
            if sfx in cid:
                position_id = cid.split(sfx)[0]
                break

        if position_id is None:
            return False

        position = self.state.get_position(position_id)
        if position is None:
            return False

        # Link order to position
        order.position_id = position_id
        self.state.update_order(order)

        # Update position's stop/TP order IDs
        if order.purpose == OrderPurpose.STOP or order.order_type == OrderType.STOP_MARKET:
            position.stop_order_id = order.client_order_id
            if order.stop_price:
                position.stop_price = order.stop_price
                position.initial_stop_price = order.stop_price
        elif order.purpose == OrderPurpose.TAKE_PROFIT or order.order_type == OrderType.TAKE_PROFIT_MARKET:
            position.tp_order_id = order.client_order_id
            position.tp_price = order.stop_price

        self.state.update_position(position)
        logger.debug(f"Order {order.client_order_id} linked to position {position_id}")
        return True

    def _cancel_orphan(self, order: Order, result: ReconciliationResult) -> None:
        """Cancel an orphan order and update result."""
        try:
            self.client.cancel_order(
                symbol=order.symbol,
                order_id=order.exchange_order_id,
            )
            order.status = OrderStatus.CANCELED
            self.state.update_order(order)
            result.orphan_orders_cancelled += 1
            logger.warning(
                f"Orphan order cancelled: {order.client_order_id} ({order.symbol})"
            )
        except Exception as e:
            msg = f"Failed to cancel orphan order {order.client_order_id}: {e}"
            logger.error(msg)
            result.errors.append(msg)
