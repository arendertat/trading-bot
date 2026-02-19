"""In-memory state manager for positions and orders"""

import logging
from datetime import datetime
from threading import RLock
from typing import Dict, List, Optional

from bot.execution.position import Position, PositionStatus
from bot.execution.models import Order, OrderStatus


logger = logging.getLogger("trading_bot.state.manager")


class StateManager:
    """
    Thread-safe in-memory state manager.

    Tracks:
    - Open and closed positions
    - Active orders linked to positions
    - Daily/weekly PnL aggregates (reset on schedule)

    All mutations are protected by a single RLock so callers
    don't need to coordinate locking themselves.
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # position_id → Position
        self._positions: Dict[str, Position] = {}

        # client_order_id → Order
        self._orders: Dict[str, Order] = {}

        logger.info("StateManager initialised")

    # ------------------------------------------------------------------
    # Position API
    # ------------------------------------------------------------------

    def add_position(self, position: Position) -> None:
        """
        Add position to state.

        Args:
            position: Position to track

        Raises:
            ValueError: If position_id already exists
        """
        with self._lock:
            if position.position_id in self._positions:
                raise ValueError(
                    f"Position {position.position_id} already exists in state"
                )
            self._positions[position.position_id] = position
            logger.info(
                f"Position added: {position.position_id} "
                f"({position.symbol} {position.side.value} @ {position.entry_price})"
            )

    def update_position(self, position: Position) -> None:
        """
        Upsert position (add if missing, replace if present).

        Args:
            position: Updated position
        """
        with self._lock:
            self._positions[position.position_id] = position
            logger.debug(f"Position updated: {position.position_id}")

    def remove_position(self, position_id: str) -> Optional[Position]:
        """
        Remove position from state.

        Args:
            position_id: Position ID to remove

        Returns:
            Removed position, or None if not found
        """
        with self._lock:
            pos = self._positions.pop(position_id, None)
            if pos:
                logger.info(f"Position removed: {position_id}")
            return pos

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        with self._lock:
            return self._positions.get(position_id)

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get all open positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open positions
        """
        with self._lock:
            positions = [
                p for p in self._positions.values()
                if p.status == PositionStatus.OPEN
            ]
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions

    def get_all_positions(self) -> List[Position]:
        """Get all positions (open + closed)."""
        with self._lock:
            return list(self._positions.values())

    def open_position_count(self) -> int:
        """Return number of open positions."""
        return len(self.get_open_positions())

    def has_open_position(self, symbol: str) -> bool:
        """Check if there is an open position for symbol."""
        return len(self.get_open_positions(symbol)) > 0

    # ------------------------------------------------------------------
    # Order API
    # ------------------------------------------------------------------

    def add_order(self, order: Order) -> None:
        """
        Add order to state.

        Args:
            order: Order to track
        """
        with self._lock:
            self._orders[order.client_order_id] = order
            logger.debug(f"Order added to state: {order.client_order_id}")

    def update_order(self, order: Order) -> None:
        """Upsert order."""
        with self._lock:
            self._orders[order.client_order_id] = order

    def get_order(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID."""
        with self._lock:
            return self._orders.get(client_order_id)

    def get_orders_for_position(self, position_id: str) -> List[Order]:
        """Get all orders linked to a position."""
        with self._lock:
            return [
                o for o in self._orders.values()
                if o.position_id == position_id
            ]

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        with self._lock:
            orders = [
                o for o in self._orders.values()
                if o.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)
            ]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    def total_open_notional(self) -> float:
        """Sum of notional USD across all open positions."""
        return sum(p.notional_usd for p in self.get_open_positions())

    def total_open_risk_usd(self) -> float:
        """Sum of risk_amount_usd across all open positions."""
        return sum(p.risk_amount_usd for p in self.get_open_positions())

    def total_unrealized_pnl(self) -> float:
        """Sum of unrealized PnL across all open positions."""
        return sum(p.unrealized_pnl_usd for p in self.get_open_positions())

    # ------------------------------------------------------------------
    # Reset / snapshot
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all state (used in tests / after full reconciliation)."""
        with self._lock:
            self._positions.clear()
            self._orders.clear()
        logger.warning("StateManager cleared")

    def snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot for debugging."""
        with self._lock:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "open_positions": len(self.get_open_positions()),
                "total_positions": len(self._positions),
                "total_orders": len(self._orders),
                "open_orders": len(self.get_open_orders()),
                "total_open_notional_usd": self.total_open_notional(),
                "total_open_risk_usd": self.total_open_risk_usd(),
                "total_unrealized_pnl": self.total_unrealized_pnl(),
            }
