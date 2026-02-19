"""Order lifecycle state machine and transitions"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any

from bot.execution.models import Order, OrderStatus, FillEvent


logger = logging.getLogger("trading_bot.execution.lifecycle")


class OrderLifecycle:
    """
    Order lifecycle state machine.

    Manages order state transitions and validates state changes.

    State Flow:
        NEW → SUBMITTED → OPEN → (FILLED | PARTIALLY_FILLED → FILLED | CANCELED)
        NEW → SUBMITTED → REJECTED
        OPEN → EXPIRED → CANCELED

    Terminal States:
        - FILLED
        - CANCELED
        - REJECTED
        - EXPIRED (transitions to CANCELED)
    """

    # Valid state transitions
    VALID_TRANSITIONS: Dict[OrderStatus, set[OrderStatus]] = {
        OrderStatus.NEW: {OrderStatus.SUBMITTED, OrderStatus.REJECTED},
        OrderStatus.SUBMITTED: {OrderStatus.OPEN, OrderStatus.REJECTED},
        OrderStatus.OPEN: {
            OrderStatus.FILLED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.CANCELED,
            OrderStatus.EXPIRED
        },
        OrderStatus.PARTIALLY_FILLED: {
            OrderStatus.FILLED,
            OrderStatus.CANCELED
        },
        OrderStatus.EXPIRED: {OrderStatus.CANCELED},
        # Terminal states (no transitions)
        OrderStatus.FILLED: set(),
        OrderStatus.CANCELED: set(),
        OrderStatus.REJECTED: set()
    }

    def __init__(self):
        """Initialize order lifecycle manager"""
        self.state_change_callbacks: Dict[OrderStatus, list[Callable]] = {}
        logger.info("OrderLifecycle initialized")

    def validate_transition(self, order: Order, new_status: OrderStatus) -> bool:
        """
        Validate if status transition is allowed.

        Args:
            order: Order to validate
            new_status: Target status

        Returns:
            True if transition is valid

        Raises:
            ValueError: If transition is invalid
        """
        current_status = order.status

        # Allow idempotent transitions (same status)
        if current_status == new_status:
            logger.debug(f"Order {order.client_order_id}: Idempotent transition to {new_status}")
            return True

        # Check if transition is valid
        valid_next_states = self.VALID_TRANSITIONS.get(current_status, set())

        if new_status not in valid_next_states:
            raise ValueError(
                f"Invalid order status transition: {current_status.value} → {new_status.value}. "
                f"Valid transitions from {current_status.value}: {[s.value for s in valid_next_states]}"
            )

        return True

    def transition_to(
        self,
        order: Order,
        new_status: OrderStatus,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Order:
        """
        Transition order to new status.

        Args:
            order: Order to transition
            new_status: Target status
            metadata: Optional metadata to update

        Returns:
            Updated order

        Raises:
            ValueError: If transition is invalid
        """
        # Validate transition
        self.validate_transition(order, new_status)

        # Log transition
        logger.info(
            f"Order {order.client_order_id} ({order.symbol}): "
            f"{order.status.value} → {new_status.value}"
        )

        # Update order
        old_status = order.status
        order.status = new_status
        order.timestamp_updated = datetime.utcnow()

        # Update metadata if provided
        if metadata:
            order.metadata.update(metadata)

        # Trigger callbacks
        self._trigger_callbacks(order, old_status, new_status)

        return order

    def submit_order(self, order: Order, exchange_order_id: str) -> Order:
        """
        Mark order as submitted to exchange.

        Args:
            order: Order to submit
            exchange_order_id: Exchange-assigned order ID

        Returns:
            Updated order
        """
        order.exchange_order_id = exchange_order_id
        order.timestamp_submitted = datetime.utcnow()
        return self.transition_to(order, OrderStatus.SUBMITTED)

    def mark_open(self, order: Order) -> Order:
        """
        Mark order as open on exchange.

        Args:
            order: Order to mark open

        Returns:
            Updated order
        """
        return self.transition_to(order, OrderStatus.OPEN)

    def mark_rejected(self, order: Order, reason: str) -> Order:
        """
        Mark order as rejected by exchange.

        Args:
            order: Order to reject
            reason: Rejection reason

        Returns:
            Updated order
        """
        return self.transition_to(
            order,
            OrderStatus.REJECTED,
            metadata={"rejection_reason": reason}
        )

    def apply_fill(
        self,
        order: Order,
        fill_event: FillEvent
    ) -> Order:
        """
        Apply fill event to order.

        Args:
            order: Order to fill
            fill_event: Fill event details

        Returns:
            Updated order
        """
        # Update fill data
        order.filled_quantity = fill_event.cumulative_filled
        order.fees_paid += fill_event.fee

        # Calculate average fill price
        if order.avg_fill_price is None:
            order.avg_fill_price = fill_event.price
        else:
            # Weighted average
            total_filled = fill_event.cumulative_filled
            new_fill_qty = fill_event.quantity
            old_fill_qty = total_filled - new_fill_qty

            order.avg_fill_price = (
                (order.avg_fill_price * old_fill_qty + fill_event.price * new_fill_qty)
                / total_filled
            )

        # Determine new status
        if order.filled_quantity >= order.quantity:
            # Fully filled
            new_status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            # Partially filled
            new_status = OrderStatus.PARTIALLY_FILLED
        else:
            # No fill yet (shouldn't happen with valid fill event)
            logger.warning(f"Order {order.client_order_id}: Fill event with zero cumulative fill")
            return order

        logger.info(
            f"Order {order.client_order_id}: Fill {fill_event.quantity:.6f} @ {fill_event.price:.2f}. "
            f"Total filled: {order.filled_quantity:.6f}/{order.quantity:.6f} "
            f"({order.fill_percentage:.1%})"
        )

        return self.transition_to(order, new_status)

    def mark_canceled(self, order: Order, reason: str = "Manual") -> Order:
        """
        Mark order as canceled.

        Args:
            order: Order to cancel
            reason: Cancellation reason

        Returns:
            Updated order
        """
        return self.transition_to(
            order,
            OrderStatus.CANCELED,
            metadata={"cancellation_reason": reason}
        )

    def check_ttl_expiry(self, order: Order) -> bool:
        """
        Check if order TTL has expired.

        Args:
            order: Order to check

        Returns:
            True if order has expired
        """
        if order.ttl_seconds is None:
            return False

        if order.timestamp_submitted is None:
            return False

        elapsed = (datetime.utcnow() - order.timestamp_submitted).total_seconds()
        expired = elapsed >= order.ttl_seconds

        if expired:
            logger.warning(
                f"Order {order.client_order_id}: TTL expired "
                f"({elapsed:.1f}s >= {order.ttl_seconds}s)"
            )

        return expired

    def mark_expired(self, order: Order) -> Order:
        """
        Mark order as expired (TTL exceeded).

        Args:
            order: Order to expire

        Returns:
            Updated order
        """
        # First transition to EXPIRED
        order = self.transition_to(order, OrderStatus.EXPIRED)

        # Then transition to CANCELED
        return self.transition_to(
            order,
            OrderStatus.CANCELED,
            metadata={"expiry_reason": "TTL exceeded"}
        )

    def register_callback(
        self,
        status: OrderStatus,
        callback: Callable[[Order], None]
    ) -> None:
        """
        Register callback for status transition.

        Args:
            status: Status to trigger callback
            callback: Callback function receiving order
        """
        if status not in self.state_change_callbacks:
            self.state_change_callbacks[status] = []

        self.state_change_callbacks[status].append(callback)
        logger.debug(f"Registered callback for status: {status.value}")

    def _trigger_callbacks(
        self,
        order: Order,
        old_status: OrderStatus,
        new_status: OrderStatus
    ) -> None:
        """
        Trigger callbacks for status change.

        Args:
            order: Order that changed
            old_status: Previous status
            new_status: New status
        """
        callbacks = self.state_change_callbacks.get(new_status, [])

        for callback in callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(
                    f"Error in callback for {new_status.value}: {e}",
                    exc_info=True
                )

    def is_terminal_state(self, order: Order) -> bool:
        """
        Check if order is in terminal state.

        Args:
            order: Order to check

        Returns:
            True if order is in terminal state (no more transitions)
        """
        return order.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED
        ]

    def can_cancel(self, order: Order) -> bool:
        """
        Check if order can be canceled.

        Args:
            order: Order to check

        Returns:
            True if order can be canceled
        """
        return order.status in [
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.EXPIRED
        ]

    def can_modify(self, order: Order) -> bool:
        """
        Check if order can be modified (price/quantity).

        Args:
            order: Order to check

        Returns:
            True if order can be modified
        """
        # Only open orders (not partially filled) can be modified
        return order.status == OrderStatus.OPEN and order.filled_quantity == 0
