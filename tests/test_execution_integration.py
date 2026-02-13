"""
Execution Integration Tests - Task 12

Tests complete order lifecycle with mock exchange responses.

Scenarios:
    1. LIMIT order placed → filled → stop/TP orders placed
    2. LIMIT order placed → TTL expires → cancelled → retry → filled
    3. LIMIT order partially filled → partial position created
    4. Trailing stop enabled → profit exceeds 1.0R → stop updates
    5. Kill switch activated → emergency market close
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from bot.config.models import ExecutionConfig
from bot.execution.models import Order, OrderStatus, OrderSide, OrderType, OrderPurpose, FillEvent
from bot.execution.order_lifecycle import OrderLifecycle
from bot.execution.order_manager import OrderManager
from bot.execution.position import Position, ExitReason, PositionStatus
from bot.execution.trailing_stop import TrailingStopManager
from bot.execution.exit_manager import ExitManager


# ---------------------------------------------------------------------------
# Shared mock exchange
# ---------------------------------------------------------------------------

class MockExchange:
    """
    Mock exchange for integration tests.

    Supports:
    - place_order, cancel_order, fetch_order, fetch_open_orders
    - Simulated fills (full and partial)
    - Order state tracking
    """

    def __init__(self):
        self._orders: dict = {}
        self._open_orders: list = []
        self._counter: int = 5000

    # ---- Order placement --------------------------------------------------

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
        order_id = self._counter
        self._counter += 1

        record = {
            "orderId": order_id,
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "origQty": str(quantity),
            "executedQty": "0",
            "price": str(price) if price else "0",
            "stopPrice": str(stop_price) if stop_price else "0",
            "avgPrice": "0",
            "clientOrderId": client_order_id,
            "status": "NEW",
            "reduceOnly": reduce_only,
        }
        self._orders[order_id] = record
        self._open_orders.append(record)
        return record

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        if order_id not in self._orders:
            raise Exception(f"Order {order_id} not found")
        record = self._orders[order_id]
        record["status"] = "CANCELED"
        self._open_orders = [o for o in self._open_orders if o["orderId"] != order_id]
        return record

    def fetch_order(self, symbol: str, order_id: int) -> dict:
        if order_id not in self._orders:
            raise Exception(f"Order {order_id} not found")
        return self._orders[order_id]

    def fetch_open_orders(self, symbol: str) -> list:
        return [o for o in self._open_orders if o["symbol"] == symbol]

    # ---- Helpers ----------------------------------------------------------

    def simulate_fill(self, order_id: int, fill_price: float, fill_qty: float = None):
        """Fully or partially fill an order."""
        record = self._orders[order_id]
        orig_qty = float(record["origQty"])
        qty = fill_qty if fill_qty is not None else orig_qty

        record["executedQty"] = str(qty)
        record["avgPrice"] = str(fill_price)

        if qty >= orig_qty:
            record["status"] = "FILLED"
            self._open_orders = [o for o in self._open_orders if o["orderId"] != order_id]
        else:
            record["status"] = "PARTIALLY_FILLED"

        return record

    def order_count(self, symbol: str = None) -> int:
        if symbol:
            return len([o for o in self._open_orders if o["symbol"] == symbol])
        return len(self._open_orders)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def exchange():
    return MockExchange()


@pytest.fixture
def config():
    return ExecutionConfig(
        limit_ttl_seconds=5,
        limit_retry_count=1,
        maker_fee_pct=0.0002,
        taker_fee_pct=0.0004,
    )


@pytest.fixture
def order_manager(exchange, config):
    return OrderManager(exchange_client=exchange, config=config)


@pytest.fixture
def trailing_manager():
    return TrailingStopManager()


@pytest.fixture
def exit_manager(order_manager, trailing_manager):
    return ExitManager(order_manager=order_manager, trailing_manager=trailing_manager)


def make_long_position(
    position_id: str = "pos_001",
    symbol: str = "BTCUSDT",
    entry_price: float = 50000.0,
    quantity: float = 0.1,
    stop_price: float = 49500.0,
    tp_price: float = 50750.0,
    trail_after_r: float = 1.0,
    atr_trail_mult: float = 2.0,
    stop_order_id: str = "stop_001",
    tp_order_id: str = "tp_001",
) -> Position:
    """Helper to create a LONG position for tests."""
    risk = abs(entry_price - stop_price) * quantity   # $500 * 0.1 = $50
    return Position(
        position_id=position_id,
        symbol=symbol,
        side=OrderSide.LONG,
        entry_price=entry_price,
        quantity=quantity,
        notional_usd=entry_price * quantity,
        leverage=1.0,
        margin_usd=entry_price * quantity,
        stop_price=stop_price,
        tp_price=tp_price,
        entry_time=datetime.utcnow(),
        risk_amount_usd=risk,
        initial_stop_price=stop_price,
        trail_after_r=trail_after_r,
        atr_trail_mult=atr_trail_mult,
        entry_order_id="entry_001",
        stop_order_id=stop_order_id,
        tp_order_id=tp_order_id,
    )


def make_short_position(
    position_id: str = "pos_002",
    symbol: str = "ETHUSDT",
    entry_price: float = 3000.0,
    quantity: float = 1.0,
    stop_price: float = 3030.0,
    tp_price: float = 2955.0,
    trail_after_r: float = 1.0,
    atr_trail_mult: float = 2.0,
    stop_order_id: str = "stop_002",
    tp_order_id: str = "tp_002",
) -> Position:
    """Helper to create a SHORT position for tests."""
    risk = abs(stop_price - entry_price) * quantity   # $30 * 1 = $30
    return Position(
        position_id=position_id,
        symbol=symbol,
        side=OrderSide.SHORT,
        entry_price=entry_price,
        quantity=quantity,
        notional_usd=entry_price * quantity,
        leverage=1.0,
        margin_usd=entry_price * quantity,
        stop_price=stop_price,
        tp_price=tp_price,
        entry_time=datetime.utcnow(),
        risk_amount_usd=risk,
        initial_stop_price=stop_price,
        trail_after_r=trail_after_r,
        atr_trail_mult=atr_trail_mult,
        entry_order_id="entry_002",
        stop_order_id=stop_order_id,
        tp_order_id=tp_order_id,
    )


# ===========================================================================
# SCENARIO 1: LIMIT order → filled → stop/TP orders placed
# ===========================================================================

class TestScenario1LimitOrderFillThenExitOrders:
    """
    Scenario 1: LIMIT entry → fill → stop and TP orders placed.

    Flow:
        1. Place LIMIT entry order
        2. Simulate fill
        3. Verify position created with correct stop/TP
        4. Verify stop and TP orders are on exchange
    """

    def test_limit_entry_placed_correctly(self, order_manager, exchange):
        """Entry LIMIT order reaches exchange with correct params."""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=0.1,
            price=50000.0,
            client_order_id="S1_entry_001",
        )
        order_manager._add_order(order)
        order = order_manager._place_single_limit_order(order)

        assert order.status == OrderStatus.OPEN
        assert order.exchange_order_id is not None

        ex_orders = exchange.fetch_open_orders("BTCUSDT")
        assert len(ex_orders) == 1
        assert ex_orders[0]["clientOrderId"] == "S1_entry_001"
        assert ex_orders[0]["side"] == "BUY"

    def test_entry_fill_updates_order_status(self, order_manager, exchange):
        """After fill, order status becomes FILLED."""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=0.1,
            price=50000.0,
            client_order_id="S1_fill_001",
        )
        order_manager._add_order(order)
        order = order_manager._place_single_limit_order(order)

        # Simulate fill on exchange
        exchange.simulate_fill(order.exchange_order_id, fill_price=50000.0)

        # Check fills
        order = order_manager.check_order_fills(order)

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == pytest.approx(0.1)
        assert order.avg_fill_price == pytest.approx(50000.0)

    def test_stop_order_placed_after_fill(self, order_manager, exchange):
        """Stop order is placed immediately after entry fill."""
        # Entry filled
        entry_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=0.1,
            price=50000.0,
            client_order_id="S1_entry_stop",
        )
        order_manager._add_order(entry_order)
        entry_order = order_manager._place_single_limit_order(entry_order)
        exchange.simulate_fill(entry_order.exchange_order_id, fill_price=50000.0)
        entry_order = order_manager.check_order_fills(entry_order)
        assert entry_order.status == OrderStatus.FILLED

        # Place stop order
        stop_order = order_manager.place_stop_order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,       # Opposite side to close LONG
            quantity=0.1,
            stop_price=49500.0,
            client_order_id="S1_stop_001",
            position_id="pos_s1",
        )

        assert stop_order.status == OrderStatus.OPEN
        assert stop_order.stop_price == pytest.approx(49500.0)

        # Both entry (filled) and stop should be tracked
        all_orders = order_manager.get_all_orders("BTCUSDT")
        assert len(all_orders) == 2

    def test_tp_order_placed_after_fill(self, order_manager, exchange):
        """TP order is placed immediately after entry fill."""
        entry_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=0.1,
            price=50000.0,
            client_order_id="S1_entry_tp",
        )
        order_manager._add_order(entry_order)
        entry_order = order_manager._place_single_limit_order(entry_order)
        exchange.simulate_fill(entry_order.exchange_order_id, fill_price=50000.0)
        entry_order = order_manager.check_order_fills(entry_order)

        # Place TP order
        tp_order = order_manager.place_take_profit_order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            quantity=0.1,
            tp_price=50750.0,
            client_order_id="S1_tp_001",
            position_id="pos_s1",
        )

        assert tp_order.status == OrderStatus.OPEN
        assert tp_order.stop_price == pytest.approx(50750.0)    # TP stored in stop_price

    def test_full_entry_to_stop_tp_flow(self, order_manager, exchange):
        """Full scenario: entry → fill → stop order → TP order → both open."""
        # 1. Place entry
        entry = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=0.1,
            price=50000.0,
            client_order_id="S1_full_entry",
        )
        order_manager._add_order(entry)
        entry = order_manager._place_single_limit_order(entry)
        assert entry.status == OrderStatus.OPEN

        # 2. Fill entry
        exchange.simulate_fill(entry.exchange_order_id, fill_price=50000.0)
        entry = order_manager.check_order_fills(entry)
        assert entry.status == OrderStatus.FILLED

        # 3. Place stop
        stop = order_manager.place_stop_order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            quantity=0.1,
            stop_price=49500.0,
            client_order_id="S1_full_stop",
            position_id="S1_pos",
        )
        assert stop.status == OrderStatus.OPEN

        # 4. Place TP
        tp = order_manager.place_take_profit_order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            quantity=0.1,
            tp_price=50750.0,
            client_order_id="S1_full_tp",
            position_id="S1_pos",
        )
        assert tp.status == OrderStatus.OPEN

        # 5. Verify exchange state: entry filled (not open), stop+TP open
        open_on_exchange = exchange.fetch_open_orders("BTCUSDT")
        assert len(open_on_exchange) == 2
        client_ids = {o["clientOrderId"] for o in open_on_exchange}
        assert "S1_full_stop" in client_ids
        assert "S1_full_tp" in client_ids


# ===========================================================================
# SCENARIO 2: LIMIT order TTL → cancel → retry → filled
# ===========================================================================

class TestScenario2TTLAndRetry:
    """
    Scenario 2: LIMIT order TTL expires → cancelled → retry → filled.

    Flow:
        1. Place LIMIT order
        2. TTL expires (simulated)
        3. Order cancelled automatically
        4. Retry order placed
        5. Retry order filled
    """

    def test_ttl_expiry_detected(self, order_manager, config):
        """Order marked as expired when TTL elapsed."""
        lifecycle = OrderLifecycle()
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=0.1,
            price=50000.0,
            client_order_id="S2_ttl_001",
            ttl_seconds=30,
        )
        # Manually back-date submission time
        order.timestamp_submitted = datetime.utcnow() - timedelta(seconds=35)

        assert lifecycle.check_ttl_expiry(order) is True

    def test_ttl_not_expired_within_window(self, order_manager, config):
        """Order not expired while within TTL window."""
        lifecycle = OrderLifecycle()
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=0.1,
            price=50000.0,
            client_order_id="S2_ttl_002",
            ttl_seconds=30,
        )
        order.timestamp_submitted = datetime.utcnow() - timedelta(seconds=10)

        assert lifecycle.check_ttl_expiry(order) is False

    def test_cancel_on_ttl_removes_from_exchange(self, order_manager, exchange):
        """Cancelling expired order removes it from exchange open orders."""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=0.1,
            price=50000.0,
            client_order_id="S2_cancel_001",
        )
        order_manager._add_order(order)
        order = order_manager._place_single_limit_order(order)
        assert exchange.order_count("BTCUSDT") == 1

        order_manager.cancel_order(order, reason="TTL expired")

        assert order.status == OrderStatus.CANCELED
        assert exchange.order_count("BTCUSDT") == 0

    def test_retry_order_has_incremented_retry_count(self, order_manager, exchange):
        """Retry order carries retry_count = 1."""
        # Place original order
        original = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=0.1,
            price=50000.0,
            client_order_id="S2_retry_orig",
        )
        order_manager._add_order(original)
        original = order_manager._place_single_limit_order(original)

        # Cancel (simulate TTL)
        order_manager.cancel_order(original, reason="TTL expired")

        # Create retry manually (mirrors process_ttl_expiry logic)
        retry = Order(
            symbol=original.symbol,
            side=original.side,
            order_type=original.order_type,
            purpose=original.purpose,
            quantity=original.quantity,
            price=original.price,
            client_order_id=f"{original.client_order_id}_retry1",
            ttl_seconds=original.ttl_seconds,
            retry_count=1,
        )
        order_manager._add_order(retry)
        retry = order_manager._place_single_limit_order(retry)

        assert retry.retry_count == 1
        assert retry.status == OrderStatus.OPEN

    def test_retry_order_fills_successfully(self, order_manager, exchange):
        """Retry order fills as expected."""
        retry = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=0.1,
            price=50000.0,
            client_order_id="S2_retry_fill",
            retry_count=1,
        )
        order_manager._add_order(retry)
        retry = order_manager._place_single_limit_order(retry)

        # Simulate fill on retry
        exchange.simulate_fill(retry.exchange_order_id, fill_price=50010.0)
        retry = order_manager.check_order_fills(retry)

        assert retry.status == OrderStatus.FILLED
        assert retry.avg_fill_price == pytest.approx(50010.0)


# ===========================================================================
# SCENARIO 3: Partial fill → partial position
# ===========================================================================

class TestScenario3PartialFill:
    """
    Scenario 3: LIMIT order partially filled → partial position created.

    Flow:
        1. Place LIMIT order
        2. 50% partial fill arrives
        3. Order status → PARTIALLY_FILLED
        4. Position created from filled portion
        5. Remaining quantity cancelled
    """

    def test_partial_fill_status(self, order_manager, exchange):
        """Order transitions to PARTIALLY_FILLED."""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=1.0,
            price=50000.0,
            client_order_id="S3_partial_001",
        )
        order_manager._add_order(order)
        order = order_manager._place_single_limit_order(order)

        # 40% fill
        exchange.simulate_fill(order.exchange_order_id, fill_price=50000.0, fill_qty=0.4)
        order = order_manager.check_order_fills(order)

        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == pytest.approx(0.4)
        assert order.fill_percentage == pytest.approx(0.4)
        assert order.remaining_quantity == pytest.approx(0.6)

    def test_position_created_from_partial_fill(self, order_manager, exchange):
        """Position uses filled quantity and avg fill price."""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=1.0,
            price=50000.0,
            client_order_id="S3_partial_pos",
        )
        order_manager._add_order(order)
        order = order_manager._place_single_limit_order(order)

        exchange.simulate_fill(order.exchange_order_id, fill_price=50050.0, fill_qty=0.5)
        order = order_manager.check_order_fills(order)

        # Create position from filled portion
        filled_qty = order.filled_quantity
        avg_price = order.avg_fill_price

        position = Position(
            position_id="S3_pos",
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            entry_price=avg_price,
            quantity=filled_qty,
            notional_usd=avg_price * filled_qty,
            leverage=1.0,
            margin_usd=avg_price * filled_qty,
            stop_price=49500.0,
            entry_time=datetime.utcnow(),
            risk_amount_usd=abs(avg_price - 49500.0) * filled_qty,
            initial_stop_price=49500.0,
            trail_after_r=1.0,
            atr_trail_mult=2.0,
            entry_order_id=order.client_order_id,
            stop_order_id="S3_stop",
        )

        assert position.quantity == pytest.approx(0.5)
        assert position.entry_price == pytest.approx(50050.0)

    def test_remaining_cancelled_after_partial(self, order_manager, exchange):
        """Remaining quantity is cancelled after accepting partial fill."""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=1.0,
            price=50000.0,
            client_order_id="S3_remain_cancel",
        )
        order_manager._add_order(order)
        order = order_manager._place_single_limit_order(order)

        exchange.simulate_fill(order.exchange_order_id, fill_price=50000.0, fill_qty=0.3)
        order = order_manager.check_order_fills(order)
        assert order.status == OrderStatus.PARTIALLY_FILLED

        # Cancel remaining
        order_manager.cancel_order(order, reason="Accept partial fill")

        assert order.status == OrderStatus.CANCELED
        assert exchange.order_count("BTCUSDT") == 0

    def test_multiple_sequential_partial_fills(self, order_manager, exchange):
        """Multiple partial fills accumulate correctly."""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.LIMIT,
            purpose=OrderPurpose.ENTRY,
            quantity=1.0,
            price=50000.0,
            client_order_id="S3_multi_partial",
        )
        order_manager._add_order(order)
        order = order_manager._place_single_limit_order(order)

        # Fill 1: 30%
        exchange.simulate_fill(order.exchange_order_id, fill_price=50000.0, fill_qty=0.3)
        order = order_manager.check_order_fills(order)
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == pytest.approx(0.3)

        # Fill 2: 40% more (cumulative 70%)
        exchange.simulate_fill(order.exchange_order_id, fill_price=50100.0, fill_qty=0.7)
        order = order_manager.check_order_fills(order)
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == pytest.approx(0.7)

        # Fill 3: final 30% (fully filled)
        exchange.simulate_fill(order.exchange_order_id, fill_price=50050.0, fill_qty=1.0)
        order = order_manager.check_order_fills(order)
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == pytest.approx(1.0)


# ===========================================================================
# SCENARIO 4: Trailing stop → 1.0R profit → stop updates
# ===========================================================================

class TestScenario4TrailingStop:
    """
    Scenario 4: Position open → price rises past 1.0R → trailing activates → stop moves up.

    Flow:
        1. Position created with trail_after_r = 1.0
        2. Price moves to exactly 1.0R profit
        3. Trailing activates
        4. Price rises further → stop follows
        5. Price drops → stop does NOT drop
        6. Price drops to stop → TRAIL exit
    """

    def test_trailing_activates_at_1r(self, trailing_manager):
        """Trailing activates when PnL reaches 1.0R."""
        pos = make_long_position(
            entry_price=50000.0,
            quantity=0.1,
            stop_price=49500.0,   # $500 risk on 0.1 BTC = $50 risk (1R)
            trail_after_r=1.0,
            atr_trail_mult=2.0,
        )
        atr = 250.0

        # At entry: no trailing
        result = trailing_manager.update_trailing_stop(pos, 50000.0, atr)
        assert result is None
        assert pos.trailing_enabled is False

        # At 1.0R profit: $50000 + $500 = $50500
        result = trailing_manager.update_trailing_stop(pos, 50500.0, atr)
        assert pos.trailing_enabled is True
        # Expected stop: 50500 - (2.0 * 250) = 50000
        assert result == pytest.approx(50000.0)

    def test_stop_follows_price_up(self, trailing_manager):
        """Stop moves up as price rises."""
        pos = make_long_position(
            entry_price=50000.0,
            quantity=0.1,
            stop_price=49500.0,
            trail_after_r=1.0,
            atr_trail_mult=2.0,
        )
        atr = 250.0

        # Enable trailing
        trailing_manager.update_trailing_stop(pos, 50500.0, atr)
        stop_after_first = pos.stop_price  # 50000

        # Price rises to $51,000
        result = trailing_manager.update_trailing_stop(pos, 51000.0, atr)
        # New stop: 51000 - 500 = 50500
        assert result == pytest.approx(50500.0)
        assert pos.stop_price > stop_after_first

    def test_stop_does_not_drop_on_price_fall(self, trailing_manager):
        """Stop does NOT drop when price falls."""
        pos = make_long_position(
            entry_price=50000.0,
            quantity=0.1,
            stop_price=49500.0,
            trail_after_r=1.0,
            atr_trail_mult=2.0,
        )
        atr = 250.0

        # Enable trailing
        trailing_manager.update_trailing_stop(pos, 51000.0, atr)
        locked_stop = pos.stop_price  # 50500

        # Price falls back to $50,700
        result = trailing_manager.update_trailing_stop(pos, 50700.0, atr)
        assert result is None  # No update
        assert pos.stop_price == pytest.approx(locked_stop)   # Stop unchanged

    def test_trailing_exit_when_stop_hit(self, exit_manager, order_manager, exchange):
        """TRAIL exit triggered when price drops to trailing stop."""
        pos = make_long_position(
            position_id="S4_trail_exit",
            entry_price=50000.0,
            quantity=0.1,
            stop_price=49500.0,
            tp_price=52000.0,
            trail_after_r=1.0,
            atr_trail_mult=2.0,
            stop_order_id="S4_stop",
            tp_order_id="S4_tp",
        )
        # Simulate trailing already active and stop moved up
        pos.trailing_enabled = True
        pos.stop_price = 50500.0          # Trailing stop moved above initial
        pos.update_unrealized_pnl(51500.0)   # Position is profitable

        # Register stop & TP orders so cancel works
        stop_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            order_type=OrderType.STOP_MARKET,
            purpose=OrderPurpose.STOP,
            quantity=0.1,
            stop_price=50500.0,
            client_order_id="S4_stop",
            position_id="S4_trail_exit",
        )
        tp_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            purpose=OrderPurpose.TAKE_PROFIT,
            quantity=0.1,
            stop_price=52000.0,
            client_order_id="S4_tp",
            position_id="S4_trail_exit",
        )
        order_manager._add_order(stop_order)
        order_manager._add_order(tp_order)
        # Place both on exchange so cancel can find them
        order_manager._place_single_limit_order = lambda o, **kw: o  # no-op for this
        stop_order.status = OrderStatus.OPEN
        stop_order.exchange_order_id = 9001
        tp_order.status = OrderStatus.OPEN
        tp_order.exchange_order_id = 9002
        exchange._orders[9001] = {"orderId": 9001, "symbol": "BTCUSDT", "status": "NEW", "clientOrderId": "S4_stop"}
        exchange._orders[9002] = {"orderId": 9002, "symbol": "BTCUSDT", "status": "NEW", "clientOrderId": "S4_tp"}
        exchange._open_orders.extend([exchange._orders[9001], exchange._orders[9002]])

        # Price drops to trailing stop
        current_price = 50500.0
        reason = exit_manager.check_and_exit(pos, current_price, atr=None)

        assert reason == ExitReason.TRAIL
        assert pos.is_closed
        assert pos.exit_reason == ExitReason.TRAIL

    def test_trailing_for_short_position(self, trailing_manager):
        """Trailing stop moves down for SHORT position."""
        pos = make_short_position(
            entry_price=3000.0,
            quantity=1.0,
            stop_price=3030.0,  # $30 risk = $30 (1R)
            trail_after_r=1.0,
            atr_trail_mult=2.0,
        )
        atr = 10.0

        # Enable trailing: price at 3000 - 30 = 2970 (1R profit)
        result = trailing_manager.update_trailing_stop(pos, 2970.0, atr)
        assert pos.trailing_enabled is True
        # New stop = 2970 + (2 * 10) = 2990
        assert result == pytest.approx(2990.0)
        assert pos.stop_price < 3030.0  # Stop moved down (improved)

        # Price falls further
        result = trailing_manager.update_trailing_stop(pos, 2950.0, atr)
        # New stop = 2950 + 20 = 2970
        assert result == pytest.approx(2970.0)

    def test_trailing_candle_close_update(self, exit_manager):
        """update_trailing_on_candle_close updates stop on each 5m close."""
        pos = make_long_position(
            entry_price=50000.0,
            quantity=0.1,
            stop_price=49500.0,
            trail_after_r=1.0,
            atr_trail_mult=2.0,
            stop_order_id="candle_stop",
            tp_order_id="candle_tp",
        )
        pos.trailing_enabled = True
        pos.update_unrealized_pnl(51000.0)
        pos.stop_price = 50500.0  # Stop already trailed up

        # Register stop order so update can cancel old and place new
        stop_ord = Order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            order_type=OrderType.STOP_MARKET,
            purpose=OrderPurpose.STOP,
            quantity=0.1,
            stop_price=50500.0,
            client_order_id="candle_stop",
            position_id=pos.position_id,
        )
        exit_manager.order_manager._add_order(stop_ord)
        stop_ord.status = OrderStatus.OPEN
        stop_ord.exchange_order_id = 9010
        exit_manager.order_manager.client._orders[9010] = {
            "orderId": 9010, "symbol": "BTCUSDT", "status": "NEW", "clientOrderId": "candle_stop"
        }
        exit_manager.order_manager.client._open_orders.append(
            exit_manager.order_manager.client._orders[9010]
        )

        # 5m candle closes at higher price
        atr = 200.0
        new_stop = exit_manager.update_trailing_on_candle_close(pos, close_price=51500.0, atr=atr)

        # Expected new stop = 51500 - (2 * 200) = 51100
        assert new_stop == pytest.approx(51100.0)
        assert pos.stop_price == pytest.approx(51100.0)


# ===========================================================================
# SCENARIO 5: Kill switch → emergency market close
# ===========================================================================

class TestScenario5KillSwitch:
    """
    Scenario 5: Kill switch activated → emergency MARKET close of all positions.

    Flow:
        1. Position is open
        2. Kill switch triggered (risk limit exceeded)
        3. Market exit order placed immediately
        4. Stop/TP orders cancelled
        5. Position closed with KILL_SWITCH exit reason
    """

    def test_kill_switch_closes_long(self, exit_manager, order_manager, exchange):
        """Kill switch closes LONG position with market order."""
        pos = make_long_position(
            position_id="S5_ks_long",
            entry_price=50000.0,
            quantity=0.1,
            stop_order_id="S5_stop_long",
            tp_order_id="S5_tp_long",
        )

        # Pre-register stop/TP orders (minimal state - no exchange needed)
        stop_ord = Order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            order_type=OrderType.STOP_MARKET,
            purpose=OrderPurpose.STOP,
            quantity=0.1,
            stop_price=49500.0,
            client_order_id="S5_stop_long",
            position_id="S5_ks_long",
        )
        tp_ord = Order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            purpose=OrderPurpose.TAKE_PROFIT,
            quantity=0.1,
            stop_price=50750.0,
            client_order_id="S5_tp_long",
            position_id="S5_ks_long",
        )
        order_manager._add_order(stop_ord)
        order_manager._add_order(tp_ord)

        # Activate kill switch
        current_price = 49800.0
        exit_manager.kill_switch_exit(pos, current_price)

        # Position must be closed
        assert pos.is_closed
        assert pos.exit_reason == ExitReason.KILL_SWITCH
        assert pos.exit_price == pytest.approx(current_price)

    def test_kill_switch_closes_short(self, exit_manager, order_manager, exchange):
        """Kill switch closes SHORT position."""
        pos = make_short_position(
            position_id="S5_ks_short",
            stop_order_id="S5_stop_short",
            tp_order_id="S5_tp_short",
        )

        stop_ord = Order(
            symbol="ETHUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.STOP_MARKET,
            purpose=OrderPurpose.STOP,
            quantity=1.0,
            stop_price=3030.0,
            client_order_id="S5_stop_short",
            position_id="S5_ks_short",
        )
        order_manager._add_order(stop_ord)

        current_price = 3010.0
        exit_manager.kill_switch_exit(pos, current_price)

        assert pos.is_closed
        assert pos.exit_reason == ExitReason.KILL_SWITCH
        assert pos.exit_price == pytest.approx(current_price)

    def test_kill_switch_exit_reason_logged(self, exit_manager, order_manager):
        """Exit summary reports KILL_SWITCH reason."""
        pos = make_long_position(
            position_id="S5_log",
            stop_order_id="S5_log_stop",
            tp_order_id="S5_log_tp",
        )
        order_manager._add_order(Order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            order_type=OrderType.STOP_MARKET,
            purpose=OrderPurpose.STOP,
            quantity=0.1,
            stop_price=49500.0,
            client_order_id="S5_log_stop",
            position_id="S5_log",
        ))

        exit_manager.kill_switch_exit(pos, current_price=49900.0)

        summary = exit_manager.get_exit_summary(pos)
        assert summary["exit_reason"] == "KILL_SWITCH"
        assert summary["exit_price"] == pytest.approx(49900.0)

    def test_force_exit_via_check_and_exit(self, exit_manager, order_manager):
        """check_and_exit with force_exit=True triggers kill switch."""
        pos = make_long_position(
            position_id="S5_force",
            stop_order_id="S5_force_stop",
            tp_order_id="S5_force_tp",
        )
        order_manager._add_order(Order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            order_type=OrderType.STOP_MARKET,
            purpose=OrderPurpose.STOP,
            quantity=0.1,
            stop_price=49500.0,
            client_order_id="S5_force_stop",
            position_id="S5_force",
        ))

        reason = exit_manager.check_and_exit(
            pos,
            current_price=50000.0,
            force_exit=True,
            exit_reason=ExitReason.KILL_SWITCH,
        )

        assert reason == ExitReason.KILL_SWITCH
        assert pos.is_closed

    def test_exit_callback_fires_on_kill_switch(self, exit_manager, order_manager):
        """Registered callbacks are invoked on kill switch exit."""
        fired = []

        def on_exit(position: Position, reason: ExitReason):
            fired.append((position.position_id, reason))

        exit_manager.register_exit_callback(on_exit)

        pos = make_long_position(
            position_id="S5_cb",
            stop_order_id="S5_cb_stop",
            tp_order_id="S5_cb_tp",
        )
        order_manager._add_order(Order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            order_type=OrderType.STOP_MARKET,
            purpose=OrderPurpose.STOP,
            quantity=0.1,
            stop_price=49500.0,
            client_order_id="S5_cb_stop",
            position_id="S5_cb",
        ))

        exit_manager.kill_switch_exit(pos, current_price=49950.0)

        assert len(fired) == 1
        assert fired[0] == ("S5_cb", ExitReason.KILL_SWITCH)

    def test_already_closed_position_not_exited_again(self, exit_manager):
        """check_and_exit ignores already-closed positions."""
        pos = make_long_position(position_id="S5_closed")
        pos.close_position(50000.0, ExitReason.TP)

        reason = exit_manager.check_and_exit(pos, current_price=49000.0)

        # Should return None (already closed)
        assert reason is None
        assert pos.exit_reason == ExitReason.TP   # Not changed


# ===========================================================================
# SCENARIO: TP exit
# ===========================================================================

class TestTPAndSLExits:
    """Additional exit tests for TP and SL conditions."""

    def test_tp_hit_exits_long(self, exit_manager, order_manager):
        """TP hit triggers TP exit."""
        pos = make_long_position(
            position_id="tp_long",
            tp_price=50750.0,
            stop_order_id="tp_long_stop",
            tp_order_id="tp_long_tp",
        )
        order_manager._add_order(Order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            order_type=OrderType.STOP_MARKET,
            purpose=OrderPurpose.STOP,
            quantity=0.1,
            stop_price=49500.0,
            client_order_id="tp_long_stop",
            position_id="tp_long",
        ))

        reason = exit_manager.check_and_exit(pos, current_price=50750.0)

        assert reason == ExitReason.TP
        assert pos.is_closed
        assert pos.exit_reason == ExitReason.TP

    def test_sl_hit_exits_long(self, exit_manager, order_manager):
        """SL hit triggers SL exit."""
        pos = make_long_position(
            position_id="sl_long",
            stop_price=49500.0,
            tp_price=50750.0,
            stop_order_id="sl_long_stop",
            tp_order_id="sl_long_tp",
        )
        order_manager._add_order(Order(
            symbol="BTCUSDT",
            side=OrderSide.SHORT,
            order_type=OrderType.STOP_MARKET,
            purpose=OrderPurpose.STOP,
            quantity=0.1,
            stop_price=49500.0,
            client_order_id="sl_long_stop",
            position_id="sl_long",
        ))

        reason = exit_manager.check_and_exit(pos, current_price=49400.0)

        assert reason == ExitReason.SL
        assert pos.is_closed
        assert pos.exit_reason == ExitReason.SL

    def test_tp_hit_exits_short(self, exit_manager, order_manager):
        """TP hit triggers TP exit for SHORT."""
        pos = make_short_position(
            position_id="tp_short",
            tp_price=2955.0,
            stop_order_id="tp_short_stop",
            tp_order_id="tp_short_tp",
        )
        order_manager._add_order(Order(
            symbol="ETHUSDT",
            side=OrderSide.LONG,
            order_type=OrderType.STOP_MARKET,
            purpose=OrderPurpose.STOP,
            quantity=1.0,
            stop_price=3030.0,
            client_order_id="tp_short_stop",
            position_id="tp_short",
        ))

        reason = exit_manager.check_and_exit(pos, current_price=2955.0)

        assert reason == ExitReason.TP
        assert pos.is_closed

    def test_no_exit_when_price_in_range(self, exit_manager):
        """No exit when price is between stop and TP."""
        pos = make_long_position(
            position_id="no_exit",
            stop_price=49500.0,
            tp_price=50750.0,
        )
        pos.update_unrealized_pnl(50300.0)

        reason = exit_manager.check_and_exit(pos, current_price=50300.0)

        assert reason is None
        assert pos.is_open
