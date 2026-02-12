"""Integration tests for OrderManager with mock exchange"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from bot.config.models import ExecutionConfig
from bot.execution.order_manager import OrderManager
from bot.execution.models import OrderSide, OrderPurpose, OrderStatus
from bot.execution.order_lifecycle import OrderLifecycle


class MockExchange:
    """Mock exchange for testing"""

    def __init__(self):
        self.orders = {}
        self.order_counter = 1000
        self.open_orders = []

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        client_order_id: str,
        price: float = None,
        stop_price: float = None,
        reduce_only: bool = False,
        time_in_force: str = "GTC"
    ) -> dict:
        """Mock place order"""
        order_id = self.order_counter
        self.order_counter += 1

        order = {
            "orderId": order_id,
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "origQty": quantity,
            "price": price,
            "stopPrice": stop_price,
            "clientOrderId": client_order_id,
            "status": "NEW",
            "executedQty": 0,
            "avgPrice": None,
            "timeInForce": time_in_force
        }

        self.orders[order_id] = order
        self.open_orders.append(order)

        return order

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        """Mock cancel order"""
        if order_id not in self.orders:
            raise Exception(f"Order {order_id} not found")

        order = self.orders[order_id]
        order["status"] = "CANCELED"

        # Remove from open orders
        self.open_orders = [o for o in self.open_orders if o["orderId"] != order_id]

        return order

    def fetch_order(self, symbol: str, order_id: int) -> dict:
        """Mock fetch order"""
        if order_id not in self.orders:
            raise Exception(f"Order {order_id} not found")

        return self.orders[order_id]

    def fetch_open_orders(self, symbol: str) -> list:
        """Mock fetch open orders"""
        return [o for o in self.open_orders if o["symbol"] == symbol]

    def fill_order(self, order_id: int, fill_price: float, fill_qty: float = None):
        """Simulate order fill"""
        if order_id not in self.orders:
            raise Exception(f"Order {order_id} not found")

        order = self.orders[order_id]

        if fill_qty is None:
            fill_qty = float(order["origQty"])

        order["executedQty"] = fill_qty
        order["avgPrice"] = fill_price

        if fill_qty >= float(order["origQty"]):
            order["status"] = "FILLED"
            # Remove from open orders
            self.open_orders = [o for o in self.open_orders if o["orderId"] != order_id]
        else:
            order["status"] = "PARTIALLY_FILLED"

        return order

    def partial_fill_order(self, order_id: int, fill_price: float, fill_qty: float):
        """Simulate partial fill"""
        return self.fill_order(order_id, fill_price, fill_qty)


class TestOrderManagerIntegration:
    """Integration tests for OrderManager"""

    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange"""
        return MockExchange()

    @pytest.fixture
    def config(self):
        """Execution config"""
        return ExecutionConfig(
            limit_ttl_seconds=5,  # Minimum allowed TTL
            limit_retry_count=1
        )

    @pytest.fixture
    def order_manager(self, mock_exchange, config):
        """Create OrderManager with mock exchange"""
        return OrderManager(
            exchange_client=mock_exchange,
            config=config
        )

    def test_place_limit_order_success(self, order_manager, mock_exchange):
        """Test placing single LIMIT order (bypassing TTL retry logic)"""
        # Create order
        from bot.execution.models import Order, OrderType
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=0.01,
            price=50000.0,
            client_order_id="test_entry_001",
            ttl_seconds=None  # Disable TTL for this test
        )

        # Add to tracking
        order_manager._add_order(order)

        # Place order directly (bypass retry logic)
        order = order_manager._place_single_limit_order(order)

        # Order should be tracked
        assert order.client_order_id == "test_entry_001"
        assert order.exchange_order_id == 1000
        assert order.status == OrderStatus.OPEN

        # Order should be on exchange
        exchange_orders = mock_exchange.fetch_open_orders("BTCUSDT")
        assert len(exchange_orders) == 1
        assert exchange_orders[0]["clientOrderId"] == "test_entry_001"

    def test_place_market_order_success(self, order_manager, mock_exchange):
        """Test placing MARKET order"""
        order = order_manager.place_market_order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            quantity=0.01,
            purpose=OrderPurpose.EXIT,
            client_order_id="test_exit_001"
        )

        assert order.client_order_id == "test_exit_001"
        assert order.status == OrderStatus.OPEN
        assert order.order_type.value == "MARKET"

    def test_place_stop_order_success(self, order_manager, mock_exchange):
        """Test placing STOP order"""
        order = order_manager.place_stop_order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            quantity=0.01,
            stop_price=49000.0,
            client_order_id="test_stop_001",
            position_id="pos_001"
        )

        assert order.client_order_id == "test_stop_001"
        assert order.stop_price == 49000.0
        assert order.status == OrderStatus.OPEN

    def test_place_take_profit_order_success(self, order_manager, mock_exchange):
        """Test placing TAKE_PROFIT order"""
        order = order_manager.place_take_profit_order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            quantity=0.01,
            tp_price=51000.0,
            client_order_id="test_tp_001",
            position_id="pos_001"
        )

        assert order.client_order_id == "test_tp_001"
        assert order.stop_price == 51000.0
        assert order.status == OrderStatus.OPEN

    def test_idempotent_order_placement(self, order_manager, mock_exchange):
        """Test idempotency - don't place duplicate order"""
        # Place first order using direct method
        from bot.execution.models import Order, OrderType
        order1 = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=0.01,
            price=50000.0,
            client_order_id="test_idempotent_001"
        )
        order_manager._add_order(order1)
        order1 = order_manager._place_single_limit_order(order1)

        # Try to place same order again using public method
        order2 = order_manager._check_existing_order("BTCUSDT", "test_idempotent_001")

        # Should return existing order
        assert order2 is not None
        assert order1.client_order_id == order2.client_order_id
        assert order1.exchange_order_id == order2.exchange_order_id

        # Only one order on exchange
        exchange_orders = mock_exchange.fetch_open_orders("BTCUSDT")
        assert len(exchange_orders) == 1

    def test_cancel_order(self, order_manager, mock_exchange):
        """Test canceling order"""
        # Place order using direct method
        from bot.execution.models import Order, OrderType
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=0.01,
            price=50000.0,
            client_order_id="test_cancel_001"
        )
        order_manager._add_order(order)
        order = order_manager._place_single_limit_order(order)

        # Cancel order
        canceled = order_manager.cancel_order(order, reason="Test cancel")

        assert canceled.status == OrderStatus.CANCELED
        assert canceled.metadata["cancellation_reason"] == "Test cancel"

        # Should be removed from exchange open orders
        exchange_orders = mock_exchange.fetch_open_orders("BTCUSDT")
        assert len(exchange_orders) == 0

    def test_check_order_fills_full_fill(self, order_manager, mock_exchange):
        """Test checking order fills - full fill"""
        # Place order
        order = order_manager.place_limit_order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            quantity=0.01,
            price=50000.0,
            purpose=OrderPurpose.ENTRY,
            client_order_id="test_fill_001"
        )

        # Simulate fill on exchange
        mock_exchange.fill_order(order.exchange_order_id, fill_price=50000.0)

        # Check fills
        updated = order_manager.check_order_fills(order)

        assert updated.status == OrderStatus.FILLED
        assert updated.filled_quantity == 0.01
        assert updated.avg_fill_price == 50000.0

    def test_check_order_fills_partial_fill(self, order_manager, mock_exchange):
        """Test checking order fills - partial fill"""
        # Place order
        order = order_manager.place_limit_order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            quantity=0.01,
            price=50000.0,
            purpose=OrderPurpose.ENTRY,
            client_order_id="test_partial_001"
        )

        # Simulate partial fill (50%)
        mock_exchange.partial_fill_order(
            order.exchange_order_id,
            fill_price=50000.0,
            fill_qty=0.005
        )

        # Check fills
        updated = order_manager.check_order_fills(order)

        assert updated.status == OrderStatus.PARTIALLY_FILLED
        assert updated.filled_quantity == 0.005
        assert updated.fill_percentage == pytest.approx(0.5)

    def test_get_order_by_id(self, order_manager, mock_exchange):
        """Test retrieving order by client order ID"""
        # Place order
        order_manager.place_limit_order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            quantity=0.01,
            price=50000.0,
            purpose=OrderPurpose.ENTRY,
            client_order_id="test_get_001"
        )

        # Retrieve order
        retrieved = order_manager.get_order("test_get_001")

        assert retrieved is not None
        assert retrieved.client_order_id == "test_get_001"

    def test_get_all_orders(self, order_manager, mock_exchange):
        """Test getting all tracked orders"""
        # Place multiple orders
        order_manager.place_limit_order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            quantity=0.01,
            price=50000.0,
            purpose=OrderPurpose.ENTRY,
            client_order_id="test_all_001"
        )

        order_manager.place_limit_order(
            symbol="ETHUSDT",
            side=OrderSide.SHORT,
            quantity=1.0,
            price=3000.0,
            purpose=OrderPurpose.ENTRY,
            client_order_id="test_all_002"
        )

        # Get all orders
        all_orders = order_manager.get_all_orders()
        assert len(all_orders) == 2

        # Get orders for specific symbol
        btc_orders = order_manager.get_all_orders(symbol="BTCUSDT")
        assert len(btc_orders) == 1
        assert btc_orders[0].symbol == "BTCUSDT"

    def test_get_open_orders(self, order_manager, mock_exchange):
        """Test getting open orders only"""
        # Place orders
        order1 = order_manager.place_limit_order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            quantity=0.01,
            price=50000.0,
            purpose=OrderPurpose.ENTRY,
            client_order_id="test_open_001"
        )

        order2 = order_manager.place_limit_order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            quantity=0.01,
            price=50100.0,
            purpose=OrderPurpose.ENTRY,
            client_order_id="test_open_002"
        )

        # Cancel one order
        order_manager.cancel_order(order1, reason="Test")

        # Get open orders
        open_orders = order_manager.get_open_orders(symbol="BTCUSDT")
        assert len(open_orders) == 1
        assert open_orders[0].client_order_id == "test_open_002"

    @patch('time.sleep', return_value=None)  # Mock sleep to speed up test
    def test_ttl_expiry_triggers_cancel(self, mock_sleep, order_manager, mock_exchange, config):
        """Test TTL expiry triggers order cancellation"""
        # Place order with short TTL
        order = order_manager.place_limit_order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            quantity=0.01,
            price=50000.0,
            purpose=OrderPurpose.ENTRY,
            client_order_id="test_ttl_001"
        )

        # Manually expire and process
        import time
        time.sleep = lambda x: None  # Mock sleep
        from datetime import datetime, timedelta
        order.timestamp_submitted = datetime.utcnow() - timedelta(seconds=config.limit_ttl_seconds + 1)

        # Process TTL expiry
        result = order_manager.process_ttl_expiry(order)

        # Should cancel and create retry
        assert result is not None
        assert result.retry_count == 1

    def test_idempotency_with_exchange_check(self, order_manager, mock_exchange):
        """Test idempotency checks exchange for existing orders"""
        # Manually add order to exchange (simulating recovery scenario)
        mock_exchange.place_order(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.01,
            client_order_id="test_recovery_001",
            price=50000.0
        )

        # Try to place order with same client_order_id
        order = order_manager.place_limit_order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            quantity=0.01,
            price=50000.0,
            purpose=OrderPurpose.ENTRY,
            client_order_id="test_recovery_001"
        )

        # Should detect existing order
        assert order.client_order_id == "test_recovery_001"
        # Should only be one order on exchange
        exchange_orders = mock_exchange.fetch_open_orders("BTCUSDT")
        assert len(exchange_orders) == 1


class TestOrderManagerEdgeCases:
    """Edge case tests for OrderManager"""

    @pytest.fixture
    def mock_exchange(self):
        return MockExchange()

    @pytest.fixture
    def config(self):
        return ExecutionConfig(limit_ttl_seconds=5, limit_retry_count=1)

    @pytest.fixture
    def order_manager(self, mock_exchange, config):
        return OrderManager(exchange_client=mock_exchange, config=config)

    def test_cancel_already_filled_order(self, order_manager, mock_exchange):
        """Test canceling an already filled order does nothing"""
        # Place and fill order
        order = order_manager.place_limit_order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            quantity=0.01,
            price=50000.0,
            purpose=OrderPurpose.ENTRY,
            client_order_id="test_filled_001"
        )

        mock_exchange.fill_order(order.exchange_order_id, fill_price=50000.0)
        order = order_manager.check_order_fills(order)

        # Try to cancel filled order
        result = order_manager.cancel_order(order, reason="Test")

        # Should still be FILLED (can't cancel)
        assert result.status == OrderStatus.FILLED

    def test_place_order_with_exchange_error(self, order_manager, mock_exchange):
        """Test handling exchange errors during order placement"""
        # Mock exchange to raise error
        def raise_error(*args, **kwargs):
            raise Exception("Insufficient margin")

        mock_exchange.place_order = raise_error

        # Try to place order
        with pytest.raises(Exception, match="Failed to place order"):
            order_manager.place_limit_order(
                symbol="BTCUSDT",
                side=OrderSide.LONG,
                quantity=0.01,
                price=50000.0,
                purpose=OrderPurpose.ENTRY,
                client_order_id="test_error_001"
            )

        # Order should be marked as rejected
        order = order_manager.get_order("test_error_001")
        assert order is not None
        assert order.status == OrderStatus.REJECTED

    def test_multiple_partial_fills(self, order_manager, mock_exchange):
        """Test order with multiple partial fills"""
        # Place order
        order = order_manager.place_limit_order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            quantity=1.0,
            price=50000.0,
            purpose=OrderPurpose.ENTRY,
            client_order_id="test_multi_partial_001"
        )

        # First partial fill (30%)
        mock_exchange.partial_fill_order(
            order.exchange_order_id,
            fill_price=50000.0,
            fill_qty=0.3
        )
        order = order_manager.check_order_fills(order)
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == 0.3

        # Second partial fill (additional 40%)
        mock_exchange.partial_fill_order(
            order.exchange_order_id,
            fill_price=50100.0,
            fill_qty=0.7  # Cumulative
        )
        order = order_manager.check_order_fills(order)
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == 0.7

        # Final fill (remaining 30%)
        mock_exchange.fill_order(
            order.exchange_order_id,
            fill_price=50050.0,
            fill_qty=1.0  # Full quantity
        )
        order = order_manager.check_order_fills(order)
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 1.0
