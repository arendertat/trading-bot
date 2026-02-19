"""Unit tests for order lifecycle state machine"""

import pytest
from datetime import datetime, timedelta

from bot.execution.models import (
    Order,
    OrderStatus,
    OrderSide,
    OrderType,
    OrderPurpose,
    FillEvent
)
from bot.execution.order_lifecycle import OrderLifecycle


class TestOrderLifecycle:
    """Test order lifecycle state machine"""

    @pytest.fixture
    def lifecycle(self):
        """Create order lifecycle manager"""
        return OrderLifecycle()

    @pytest.fixture
    def sample_order(self):
        """Create sample order for testing"""
        return Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=0.01,
            price=50000.0,
            client_order_id="test_order_001"
        )

    def test_valid_transition_new_to_submitted(self, lifecycle, sample_order):
        """Test NEW → SUBMITTED transition"""
        assert sample_order.status == OrderStatus.NEW

        updated = lifecycle.submit_order(sample_order, "EXCH123")

        assert updated.status == OrderStatus.SUBMITTED
        assert updated.exchange_order_id == "EXCH123"
        assert updated.timestamp_submitted is not None

    def test_valid_transition_submitted_to_open(self, lifecycle, sample_order):
        """Test SUBMITTED → OPEN transition"""
        sample_order = lifecycle.submit_order(sample_order, "EXCH123")

        updated = lifecycle.mark_open(sample_order)

        assert updated.status == OrderStatus.OPEN

    def test_valid_transition_open_to_filled(self, lifecycle, sample_order):
        """Test OPEN → FILLED transition"""
        sample_order = lifecycle.submit_order(sample_order, "EXCH123")
        sample_order = lifecycle.mark_open(sample_order)

        # Create fill event
        fill_event = FillEvent(
            order_id=sample_order.client_order_id,
            exchange_order_id=sample_order.exchange_order_id,
            symbol=sample_order.symbol,
            side=sample_order.side,
            quantity=0.01,
            price=50000.0,
            fee=0.0,
            timestamp=datetime.utcnow(),
            is_partial=False,
            cumulative_filled=0.01
        )

        updated = lifecycle.apply_fill(sample_order, fill_event)

        assert updated.status == OrderStatus.FILLED
        assert updated.filled_quantity == 0.01
        assert updated.avg_fill_price == 50000.0

    def test_partial_fill_transition(self, lifecycle, sample_order):
        """Test OPEN → PARTIALLY_FILLED transition"""
        sample_order = lifecycle.submit_order(sample_order, "EXCH123")
        sample_order = lifecycle.mark_open(sample_order)

        # Partial fill (50%)
        fill_event = FillEvent(
            order_id=sample_order.client_order_id,
            exchange_order_id=sample_order.exchange_order_id,
            symbol=sample_order.symbol,
            side=sample_order.side,
            quantity=0.005,
            price=50000.0,
            fee=0.0,
            timestamp=datetime.utcnow(),
            is_partial=True,
            cumulative_filled=0.005
        )

        updated = lifecycle.apply_fill(sample_order, fill_event)

        assert updated.status == OrderStatus.PARTIALLY_FILLED
        assert updated.filled_quantity == 0.005
        assert updated.fill_percentage == 0.5

    def test_partial_to_filled_transition(self, lifecycle, sample_order):
        """Test PARTIALLY_FILLED → FILLED transition"""
        sample_order = lifecycle.submit_order(sample_order, "EXCH123")
        sample_order = lifecycle.mark_open(sample_order)

        # First partial fill
        fill_event1 = FillEvent(
            order_id=sample_order.client_order_id,
            exchange_order_id=sample_order.exchange_order_id,
            symbol=sample_order.symbol,
            side=sample_order.side,
            quantity=0.005,
            price=50000.0,
            fee=0.0,
            timestamp=datetime.utcnow(),
            is_partial=True,
            cumulative_filled=0.005
        )
        sample_order = lifecycle.apply_fill(sample_order, fill_event1)
        assert sample_order.status == OrderStatus.PARTIALLY_FILLED

        # Second fill completes the order
        fill_event2 = FillEvent(
            order_id=sample_order.client_order_id,
            exchange_order_id=sample_order.exchange_order_id,
            symbol=sample_order.symbol,
            side=sample_order.side,
            quantity=0.005,
            price=50100.0,
            fee=0.0,
            timestamp=datetime.utcnow(),
            is_partial=False,
            cumulative_filled=0.01
        )
        sample_order = lifecycle.apply_fill(sample_order, fill_event2)

        assert sample_order.status == OrderStatus.FILLED
        assert sample_order.filled_quantity == 0.01
        # Check weighted average price
        expected_avg = (50000.0 * 0.005 + 50100.0 * 0.005) / 0.01
        assert sample_order.avg_fill_price == pytest.approx(expected_avg)

    def test_invalid_transition_new_to_filled(self, lifecycle, sample_order):
        """Test invalid NEW → FILLED transition raises error"""
        with pytest.raises(ValueError, match="Invalid order status transition"):
            lifecycle.transition_to(sample_order, OrderStatus.FILLED)

    def test_invalid_transition_filled_to_open(self, lifecycle, sample_order):
        """Test invalid FILLED → OPEN transition (terminal state)"""
        # Get to FILLED state
        sample_order = lifecycle.submit_order(sample_order, "EXCH123")
        sample_order = lifecycle.mark_open(sample_order)

        fill_event = FillEvent(
            order_id=sample_order.client_order_id,
            exchange_order_id=sample_order.exchange_order_id,
            symbol=sample_order.symbol,
            side=sample_order.side,
            quantity=0.01,
            price=50000.0,
            fee=0.0,
            timestamp=datetime.utcnow(),
            cumulative_filled=0.01
        )
        sample_order = lifecycle.apply_fill(sample_order, fill_event)

        # Try to transition back to OPEN
        with pytest.raises(ValueError, match="Invalid order status transition"):
            lifecycle.transition_to(sample_order, OrderStatus.OPEN)

    def test_cancel_order_from_open(self, lifecycle, sample_order):
        """Test canceling order from OPEN state"""
        sample_order = lifecycle.submit_order(sample_order, "EXCH123")
        sample_order = lifecycle.mark_open(sample_order)

        updated = lifecycle.mark_canceled(sample_order, reason="Manual cancel")

        assert updated.status == OrderStatus.CANCELED
        assert updated.metadata["cancellation_reason"] == "Manual cancel"

    def test_reject_order_from_submitted(self, lifecycle, sample_order):
        """Test rejecting order from SUBMITTED state"""
        sample_order = lifecycle.submit_order(sample_order, "EXCH123")

        updated = lifecycle.mark_rejected(sample_order, reason="Insufficient margin")

        assert updated.status == OrderStatus.REJECTED
        assert updated.metadata["rejection_reason"] == "Insufficient margin"

    def test_ttl_expiry_detection(self, lifecycle, sample_order):
        """Test TTL expiry detection"""
        sample_order.ttl_seconds = 30
        sample_order.timestamp_submitted = datetime.utcnow() - timedelta(seconds=35)

        expired = lifecycle.check_ttl_expiry(sample_order)

        assert expired is True

    def test_ttl_not_expired(self, lifecycle, sample_order):
        """Test TTL not expired"""
        sample_order.ttl_seconds = 30
        sample_order.timestamp_submitted = datetime.utcnow() - timedelta(seconds=15)

        expired = lifecycle.check_ttl_expiry(sample_order)

        assert expired is False

    def test_ttl_expiry_without_ttl(self, lifecycle, sample_order):
        """Test order without TTL never expires"""
        sample_order.ttl_seconds = None
        sample_order.timestamp_submitted = datetime.utcnow() - timedelta(hours=1)

        expired = lifecycle.check_ttl_expiry(sample_order)

        assert expired is False

    def test_mark_expired_transitions_to_canceled(self, lifecycle, sample_order):
        """Test marking order as expired transitions to CANCELED"""
        sample_order = lifecycle.submit_order(sample_order, "EXCH123")
        sample_order = lifecycle.mark_open(sample_order)

        # Mark expired
        updated = lifecycle.mark_expired(sample_order)

        # Should end up in CANCELED state (EXPIRED is intermediate)
        assert updated.status == OrderStatus.CANCELED
        assert updated.metadata.get("expiry_reason") == "TTL exceeded"

    def test_idempotent_transition(self, lifecycle, sample_order):
        """Test idempotent transition (same status)"""
        sample_order = lifecycle.submit_order(sample_order, "EXCH123")
        sample_order = lifecycle.mark_open(sample_order)

        # Try to mark open again
        updated = lifecycle.mark_open(sample_order)

        # Should succeed without error
        assert updated.status == OrderStatus.OPEN

    def test_is_terminal_state(self, lifecycle, sample_order):
        """Test terminal state detection"""
        # NEW is not terminal
        assert lifecycle.is_terminal_state(sample_order) is False

        # OPEN is not terminal
        sample_order = lifecycle.submit_order(sample_order, "EXCH123")
        sample_order = lifecycle.mark_open(sample_order)
        assert lifecycle.is_terminal_state(sample_order) is False

        # FILLED is terminal
        fill_event = FillEvent(
            order_id=sample_order.client_order_id,
            exchange_order_id=sample_order.exchange_order_id,
            symbol=sample_order.symbol,
            side=sample_order.side,
            quantity=0.01,
            price=50000.0,
            fee=0.0,
            timestamp=datetime.utcnow(),
            cumulative_filled=0.01
        )
        sample_order = lifecycle.apply_fill(sample_order, fill_event)
        assert lifecycle.is_terminal_state(sample_order) is True

    def test_can_cancel_states(self, lifecycle, sample_order):
        """Test which states allow cancellation"""
        # NEW cannot be canceled (not on exchange yet)
        assert lifecycle.can_cancel(sample_order) is False

        # OPEN can be canceled
        sample_order = lifecycle.submit_order(sample_order, "EXCH123")
        sample_order = lifecycle.mark_open(sample_order)
        assert lifecycle.can_cancel(sample_order) is True

        # PARTIALLY_FILLED can be canceled
        fill_event = FillEvent(
            order_id=sample_order.client_order_id,
            exchange_order_id=sample_order.exchange_order_id,
            symbol=sample_order.symbol,
            side=sample_order.side,
            quantity=0.005,
            price=50000.0,
            fee=0.0,
            timestamp=datetime.utcnow(),
            is_partial=True,
            cumulative_filled=0.005
        )
        sample_order = lifecycle.apply_fill(sample_order, fill_event)
        assert lifecycle.can_cancel(sample_order) is True

        # Complete the fill
        fill_event2 = FillEvent(
            order_id=sample_order.client_order_id,
            exchange_order_id=sample_order.exchange_order_id,
            symbol=sample_order.symbol,
            side=sample_order.side,
            quantity=0.005,
            price=50000.0,
            fee=0.0,
            timestamp=datetime.utcnow(),
            cumulative_filled=0.01
        )
        sample_order = lifecycle.apply_fill(sample_order, fill_event2)

        # FILLED cannot be canceled
        assert lifecycle.can_cancel(sample_order) is False

    def test_can_modify_states(self, lifecycle, sample_order):
        """Test which states allow modification"""
        # NEW cannot be modified (not on exchange)
        assert lifecycle.can_modify(sample_order) is False

        # OPEN can be modified (if no fills)
        sample_order = lifecycle.submit_order(sample_order, "EXCH123")
        sample_order = lifecycle.mark_open(sample_order)
        assert lifecycle.can_modify(sample_order) is True

        # PARTIALLY_FILLED cannot be modified
        fill_event = FillEvent(
            order_id=sample_order.client_order_id,
            exchange_order_id=sample_order.exchange_order_id,
            symbol=sample_order.symbol,
            side=sample_order.side,
            quantity=0.005,
            price=50000.0,
            fee=0.0,
            timestamp=datetime.utcnow(),
            is_partial=True,
            cumulative_filled=0.005
        )
        sample_order = lifecycle.apply_fill(sample_order, fill_event)
        assert lifecycle.can_modify(sample_order) is False

    def test_state_change_callback(self, lifecycle, sample_order):
        """Test callback triggers on state change"""
        callback_triggered = []

        def on_filled(order: Order):
            callback_triggered.append(order.client_order_id)

        # Register callback
        lifecycle.register_callback(OrderStatus.FILLED, on_filled)

        # Trigger state change
        sample_order = lifecycle.submit_order(sample_order, "EXCH123")
        sample_order = lifecycle.mark_open(sample_order)

        fill_event = FillEvent(
            order_id=sample_order.client_order_id,
            exchange_order_id=sample_order.exchange_order_id,
            symbol=sample_order.symbol,
            side=sample_order.side,
            quantity=0.01,
            price=50000.0,
            fee=0.0,
            timestamp=datetime.utcnow(),
            cumulative_filled=0.01
        )
        sample_order = lifecycle.apply_fill(sample_order, fill_event)

        # Callback should have been triggered
        assert "test_order_001" in callback_triggered

    def test_order_properties(self, sample_order):
        """Test order convenience properties"""
        assert sample_order.is_entry_order is True
        assert sample_order.is_exit_order is False
        assert sample_order.is_open is False
        assert sample_order.is_filled is False
        assert sample_order.is_closed is False
        assert sample_order.remaining_quantity == 0.01
        assert sample_order.fill_percentage == 0.0

    def test_order_to_dict(self, sample_order):
        """Test order serialization to dict"""
        order_dict = sample_order.to_dict()

        assert order_dict["symbol"] == "BTCUSDT"
        assert order_dict["side"] == "LONG"
        assert order_dict["order_type"] == "LIMIT"
        assert order_dict["quantity"] == 0.01
        assert order_dict["price"] == 50000.0
        assert order_dict["client_order_id"] == "test_order_001"

    def test_fill_event_to_dict(self):
        """Test fill event serialization to dict"""
        fill_event = FillEvent(
            order_id="test_order_001",
            exchange_order_id="EXCH123",
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            quantity=0.01,
            price=50000.0,
            fee=0.5,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            is_partial=False,
            cumulative_filled=0.01
        )

        fill_dict = fill_event.to_dict()

        assert fill_dict["order_id"] == "test_order_001"
        assert fill_dict["quantity"] == 0.01
        assert fill_dict["price"] == 50000.0
        assert fill_dict["fee"] == 0.5
        assert fill_dict["is_partial"] is False


class TestOrderModels:
    """Test order model classes"""

    def test_order_creation_with_defaults(self):
        """Test order creation with default values"""
        order = Order(
            symbol="ETHUSDT",
            side=OrderSide.SHORT,
            order_type=OrderType.MARKET,
            purpose=OrderPurpose.EXIT,
            quantity=1.0,
            client_order_id="exit_001"
        )

        assert order.status == OrderStatus.NEW
        assert order.filled_quantity == 0.0
        assert order.retry_count == 0
        assert order.timestamp_created is not None
        assert len(order.metadata) == 0

    def test_order_purpose_detection(self):
        """Test order purpose detection properties"""
        entry_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=0.01,
            client_order_id="entry_001"
        )

        stop_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            order_type=OrderType.STOP_MARKET,
            purpose=OrderPurpose.STOP,
            quantity=0.01,
            client_order_id="stop_001"
        )

        assert entry_order.is_entry_order is True
        assert entry_order.is_exit_order is False

        assert stop_order.is_entry_order is False
        assert stop_order.is_exit_order is True

    def test_remaining_quantity_calculation(self):
        """Test remaining quantity calculation"""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=1.0,
            client_order_id="test_001"
        )

        assert order.remaining_quantity == 1.0

        order.filled_quantity = 0.6
        assert order.remaining_quantity == pytest.approx(0.4)

        order.filled_quantity = 1.0
        assert order.remaining_quantity == 0.0

    def test_fill_percentage_calculation(self):
        """Test fill percentage calculation"""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=1.0,
            client_order_id="test_001"
        )

        assert order.fill_percentage == 0.0

        order.filled_quantity = 0.25
        assert order.fill_percentage == pytest.approx(0.25)

        order.filled_quantity = 1.0
        assert order.fill_percentage == pytest.approx(1.0)
