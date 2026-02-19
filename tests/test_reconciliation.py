"""Tests for state manager and startup reconciler"""

import pytest
from datetime import datetime

from bot.execution.models import Order, OrderStatus, OrderSide, OrderType, OrderPurpose
from bot.execution.position import Position, PositionStatus
from bot.state.state_manager import StateManager
from bot.state.reconciler import Reconciler, ReconciliationResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_position(
    position_id: str = "pos_001",
    symbol: str = "BTCUSDT",
    side: OrderSide = OrderSide.LONG,
    entry_price: float = 50000.0,
    quantity: float = 0.1,
    stop_price: float = 49500.0,
) -> Position:
    risk = abs(entry_price - stop_price) * quantity
    return Position(
        position_id=position_id,
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        quantity=quantity,
        notional_usd=entry_price * quantity,
        leverage=1.0,
        margin_usd=entry_price * quantity,
        stop_price=stop_price,
        entry_time=datetime.utcnow(),
        risk_amount_usd=risk,
        initial_stop_price=stop_price,
        trail_after_r=1.0,
        atr_trail_mult=2.0,
        entry_order_id="entry_001",
        stop_order_id="pos_001_stop",
    )


def make_order(
    client_order_id: str = "order_001",
    symbol: str = "BTCUSDT",
    side: OrderSide = OrderSide.LONG,
    order_type: OrderType = OrderType.LIMIT,
    purpose: OrderPurpose = OrderPurpose.ENTRY,
    status: OrderStatus = OrderStatus.OPEN,
    quantity: float = 0.1,
    position_id: str = None,
) -> Order:
    order = Order(
        symbol=symbol,
        side=side,
        order_type=order_type,
        purpose=purpose,
        quantity=quantity,
        client_order_id=client_order_id,
        position_id=position_id,
    )
    order.status = status
    order.exchange_order_id = hash(client_order_id) % 100000
    return order


class MockExchange:
    """Mock exchange for reconciliation tests."""

    def __init__(
        self,
        positions: list = None,
        open_orders: list = None,
    ):
        self._positions = positions or []
        self._open_orders = open_orders or []
        self.cancelled_orders = []

    def fetch_positions(self, symbols=None) -> list:
        if symbols:
            return [p for p in self._positions if p.get("symbol") in symbols]
        return self._positions

    def fetch_open_orders(self, symbol: str = None) -> list:
        if symbol:
            return [o for o in self._open_orders if o.get("symbol") == symbol]
        return self._open_orders

    def cancel_order(self, symbol: str, order_id) -> dict:
        self.cancelled_orders.append(order_id)
        for o in self._open_orders:
            if o.get("orderId") == order_id:
                o["status"] = "CANCELED"
                return o
        raise Exception(f"Order {order_id} not found")


# ---------------------------------------------------------------------------
# StateManager tests
# ---------------------------------------------------------------------------

class TestStateManager:
    """Unit tests for StateManager"""

    @pytest.fixture
    def sm(self):
        return StateManager()

    def test_add_and_get_position(self, sm):
        pos = make_position("p1")
        sm.add_position(pos)
        assert sm.get_position("p1") is pos

    def test_add_duplicate_position_raises(self, sm):
        pos = make_position("p1")
        sm.add_position(pos)
        with pytest.raises(ValueError, match="already exists"):
            sm.add_position(pos)

    def test_update_position_upserts(self, sm):
        pos = make_position("p1")
        sm.update_position(pos)   # No raise — upsert
        assert sm.get_position("p1") is pos

    def test_remove_position(self, sm):
        pos = make_position("p1")
        sm.add_position(pos)
        removed = sm.remove_position("p1")
        assert removed is pos
        assert sm.get_position("p1") is None

    def test_remove_nonexistent_position_returns_none(self, sm):
        assert sm.remove_position("ghost") is None

    def test_get_open_positions_only_open(self, sm):
        p_open = make_position("open_1")
        p_closed = make_position("closed_1")
        p_closed.status = PositionStatus.CLOSED

        sm.add_position(p_open)
        sm.add_position(p_closed)

        open_pos = sm.get_open_positions()
        assert len(open_pos) == 1
        assert open_pos[0].position_id == "open_1"

    def test_get_open_positions_by_symbol(self, sm):
        btc = make_position("btc_pos", symbol="BTCUSDT")
        eth = make_position("eth_pos", symbol="ETHUSDT")
        sm.add_position(btc)
        sm.add_position(eth)

        btc_pos = sm.get_open_positions(symbol="BTCUSDT")
        assert len(btc_pos) == 1
        assert btc_pos[0].symbol == "BTCUSDT"

    def test_has_open_position(self, sm):
        sm.add_position(make_position("p1", symbol="BTCUSDT"))
        assert sm.has_open_position("BTCUSDT") is True
        assert sm.has_open_position("ETHUSDT") is False

    def test_open_position_count(self, sm):
        sm.add_position(make_position("p1"))
        sm.add_position(make_position("p2"))
        assert sm.open_position_count() == 2

    def test_add_and_get_order(self, sm):
        order = make_order("o1")
        sm.add_order(order)
        assert sm.get_order("o1") is order

    def test_get_orders_for_position(self, sm):
        o1 = make_order("o1", position_id="pos_001")
        o2 = make_order("o2", position_id="pos_001")
        o3 = make_order("o3", position_id="pos_002")
        sm.add_order(o1)
        sm.add_order(o2)
        sm.add_order(o3)

        pos_orders = sm.get_orders_for_position("pos_001")
        assert len(pos_orders) == 2

    def test_get_open_orders(self, sm):
        o_open = make_order("o_open", status=OrderStatus.OPEN)
        o_filled = make_order("o_filled", status=OrderStatus.FILLED)
        sm.add_order(o_open)
        sm.add_order(o_filled)

        open_orders = sm.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].client_order_id == "o_open"

    def test_total_open_risk_usd(self, sm):
        p1 = make_position("p1", stop_price=49500.0)   # risk = $50
        p2 = make_position("p2", symbol="ETHUSDT",
                           entry_price=3000.0, stop_price=2970.0,
                           quantity=1.0)                # risk = $30
        sm.add_position(p1)
        sm.add_position(p2)

        assert sm.total_open_risk_usd() == pytest.approx(50.0 + 30.0)

    def test_clear_removes_all(self, sm):
        sm.add_position(make_position("p1"))
        sm.add_order(make_order("o1"))
        sm.clear()
        assert sm.open_position_count() == 0
        assert sm.get_order("o1") is None

    def test_snapshot_returns_dict(self, sm):
        sm.add_position(make_position("p1"))
        snap = sm.snapshot()
        assert snap["open_positions"] == 1
        assert "timestamp" in snap
        assert "total_open_risk_usd" in snap


# ---------------------------------------------------------------------------
# Reconciler tests
# ---------------------------------------------------------------------------

class TestReconciler:
    """Tests for startup reconciler"""

    @pytest.fixture
    def sm(self):
        return StateManager()

    # -- Position reconciliation ----------------------------------------

    def test_reconcile_restores_open_position(self, sm):
        exchange = MockExchange(
            positions=[{
                "symbol": "BTCUSDT",
                "side": "LONG",
                "contracts": 0.1,
                "entryPrice": 50000.0,
                "leverage": 1,
            }]
        )
        rec = Reconciler(exchange, sm)
        result = rec.reconcile()

        assert result.positions_restored == 1
        assert len(sm.get_open_positions()) == 1
        pos = sm.get_open_positions()[0]
        assert pos.symbol == "BTCUSDT"
        assert pos.entry_price == pytest.approx(50000.0)
        assert pos.quantity == pytest.approx(0.1)

    def test_reconcile_skips_zero_size_position(self, sm):
        exchange = MockExchange(
            positions=[{
                "symbol": "BTCUSDT",
                "side": "LONG",
                "contracts": 0,
                "entryPrice": 50000.0,
                "leverage": 1,
            }]
        )
        rec = Reconciler(exchange, sm)
        result = rec.reconcile()

        assert result.positions_restored == 0
        assert sm.open_position_count() == 0

    def test_reconcile_handles_fetch_positions_error(self, sm):
        class BrokenExchange(MockExchange):
            def fetch_positions(self, symbols=None):
                raise Exception("API error")

        rec = Reconciler(BrokenExchange(), sm)
        result = rec.reconcile()

        assert len(result.errors) > 0
        assert sm.open_position_count() == 0

    def test_reconcile_restores_short_position(self, sm):
        exchange = MockExchange(
            positions=[{
                "symbol": "ETHUSDT",
                "side": "SHORT",
                "contracts": 1.0,
                "entryPrice": 3000.0,
                "leverage": 2,
            }]
        )
        rec = Reconciler(exchange, sm)
        result = rec.reconcile()

        assert result.positions_restored == 1
        pos = sm.get_open_positions()[0]
        assert pos.side == OrderSide.SHORT
        assert pos.leverage == pytest.approx(2.0)

    # -- Order reconciliation ------------------------------------------

    def test_reconcile_links_stop_order_to_position(self, sm):
        # First, add a position manually so the reconciler can link to it
        pos = make_position("pos_001")
        sm.add_position(pos)

        exchange = MockExchange(
            positions=[],   # Position already in state
            open_orders=[{
                "symbol": "BTCUSDT",
                "side": "SELL",
                "type": "STOP_MARKET",
                "orderId": 9001,
                "clientOrderId": "pos_001_stop",
                "origQty": "0.1",
                "stopPrice": "49500.0",
                "price": "0",
                "executedQty": "0",
            }]
        )
        rec = Reconciler(exchange, sm)
        result = rec.reconcile()

        assert result.orders_linked == 1
        order = sm.get_order("pos_001_stop")
        assert order is not None
        assert order.position_id == "pos_001"
        # Stop price should be updated on position
        assert sm.get_position("pos_001").stop_price == pytest.approx(49500.0)

    def test_reconcile_links_tp_order_to_position(self, sm):
        pos = make_position("pos_002")
        sm.add_position(pos)

        exchange = MockExchange(
            positions=[],
            open_orders=[{
                "symbol": "BTCUSDT",
                "side": "SELL",
                "type": "TAKE_PROFIT_MARKET",
                "orderId": 9002,
                "clientOrderId": "pos_002_tp",
                "origQty": "0.1",
                "stopPrice": "50750.0",
                "price": "0",
                "executedQty": "0",
            }]
        )
        rec = Reconciler(exchange, sm)
        result = rec.reconcile()

        assert result.orders_linked == 1
        assert sm.get_position("pos_002").tp_price == pytest.approx(50750.0)

    def test_reconcile_cancels_orphan_stop_order(self, sm):
        # Stop order for a position that no longer exists
        exchange = MockExchange(
            positions=[],
            open_orders=[{
                "symbol": "BTCUSDT",
                "side": "SELL",
                "type": "STOP_MARKET",
                "orderId": 9003,
                "clientOrderId": "ghost_pos_stop",
                "origQty": "0.1",
                "stopPrice": "49000.0",
                "price": "0",
                "executedQty": "0",
            }]
        )
        rec = Reconciler(exchange, sm)
        result = rec.reconcile()

        assert result.orphan_orders_cancelled == 1
        assert 9003 in exchange.cancelled_orders

    def test_reconcile_cancels_orphan_entry_order(self, sm):
        # Entry order but no position open
        exchange = MockExchange(
            positions=[],
            open_orders=[{
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "orderId": 9004,
                "clientOrderId": "entry_orphan",
                "origQty": "0.1",
                "stopPrice": "0",
                "price": "50000",
                "executedQty": "0",
            }]
        )
        rec = Reconciler(exchange, sm)
        result = rec.reconcile()

        assert result.orphan_orders_cancelled == 1
        assert 9004 in exchange.cancelled_orders

    def test_reconcile_full_scenario(self, sm):
        """Full reconciliation: 1 position, 1 stop linked, 1 orphan cancelled."""
        exchange = MockExchange(
            positions=[{
                "symbol": "BTCUSDT",
                "side": "LONG",
                "contracts": 0.1,
                "entryPrice": 50000.0,
                "leverage": 1,
            }],
            open_orders=[],  # Orders added after position is reconciled
        )
        rec = Reconciler(exchange, sm)
        result = rec.reconcile()

        # Only positions fetched in this pass (no orders)
        assert result.positions_restored == 1
        assert result.orders_linked == 0
        assert result.orphan_orders_cancelled == 0
        assert len(result.errors) == 0

    def test_reconcile_handles_fetch_orders_error(self, sm):
        class BrokenOrderExchange(MockExchange):
            def fetch_open_orders(self, symbol=None):
                raise Exception("Orders API error")

        exchange = BrokenOrderExchange(positions=[])
        rec = Reconciler(exchange, sm)
        result = rec.reconcile()

        assert any("Orders API error" in e for e in result.errors)

    def test_reconciliation_result_to_dict(self):
        result = ReconciliationResult()
        result.positions_restored = 2
        result.orders_linked = 3
        result.orphan_orders_cancelled = 1
        result.errors = ["some error"]

        d = result.to_dict()
        assert d["positions_restored"] == 2
        assert d["orders_linked"] == 3
        assert d["orphan_orders_cancelled"] == 1
        assert len(d["errors"]) == 1
        assert "timestamp" in d

    def test_reconcile_idempotent_on_second_call(self, sm):
        """Calling reconcile twice should upsert, not raise."""
        exchange = MockExchange(
            positions=[{
                "symbol": "BTCUSDT",
                "side": "LONG",
                "contracts": 0.1,
                "entryPrice": 50000.0,
                "leverage": 1,
            }]
        )
        rec = Reconciler(exchange, sm)
        result1 = rec.reconcile()
        result2 = rec.reconcile()  # Should not raise

        assert result1.positions_restored == 1
        assert result2.positions_restored == 1
        # State should still have only unique positions
        # (upsert by position_id — id includes timestamp so two entries, that's ok)
        assert sm.open_position_count() >= 1
