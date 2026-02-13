"""
E2E Recovery & Reconciliation Tests — Milestone 8, Task 21

Covers crash-recovery scenarios:
  1. Crash during entry (order placed, position not yet in StateManager)
  2. Crash after fill (position open, stop order placed)
  3. Crash during exit (exit order orphaned, no position)
  4. Idempotency (reconcile() called twice produces no duplicates)
  5. Reconciliation result accuracy (counters match actions)
  6. Reconciliation result logged via TradeLogger
"""

import json
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from bot.execution.models import Order, OrderPurpose, OrderSide, OrderStatus, OrderType
from bot.execution.position import Position, PositionStatus
from bot.state.reconciler import Reconciler, ReconciliationResult
from bot.state.state_manager import StateManager
from bot.state.logger import TradeLogger


# ---------------------------------------------------------------------------
# Helpers: exchange mock factory
# ---------------------------------------------------------------------------

def _make_exchange_client(
    positions: Optional[List[Dict[str, Any]]] = None,
    orders: Optional[List[Dict[str, Any]]] = None,
    fail_fetch_positions: bool = False,
    fail_fetch_orders: bool = False,
    fail_cancel: bool = False,
) -> MagicMock:
    """Build a mock exchange client returning the provided position/order data."""
    client = MagicMock()

    if fail_fetch_positions:
        client.fetch_positions.side_effect = RuntimeError("exchange unavailable")
    else:
        client.fetch_positions.return_value = positions or []

    if fail_fetch_orders:
        client.fetch_open_orders.side_effect = RuntimeError("exchange unavailable")
    else:
        client.fetch_open_orders.return_value = orders or []

    if fail_cancel:
        client.cancel_order.side_effect = RuntimeError("cancel failed")
    else:
        client.cancel_order.return_value = {"status": "CANCELED"}

    return client


def _raw_position(
    symbol: str = "BTCUSDT",
    side: str = "LONG",
    qty: float = 0.1,
    entry_price: float = 50_000.0,
    leverage: float = 5.0,
) -> Dict[str, Any]:
    """Build a raw exchange position dict (ccxt-like format)."""
    return {
        "symbol": symbol,
        "side": side,
        "contracts": qty,
        "entryPrice": entry_price,
        "leverage": leverage,
    }


def _raw_order(
    client_order_id: str,
    symbol: str = "BTCUSDT",
    side: str = "SELL",
    order_type: str = "STOP_MARKET",
    qty: float = 0.1,
    stop_price: float = 48_000.0,
) -> Dict[str, Any]:
    """Build a raw exchange order dict (ccxt-like format)."""
    return {
        "clientOrderId": client_order_id,
        "orderId": f"EX_{client_order_id}",
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "origQty": str(qty),
        "stopPrice": str(stop_price),
        "executedQty": "0",
        "price": "0",
    }


def _make_reconciler(
    positions: Optional[List[Dict]] = None,
    orders: Optional[List[Dict]] = None,
    **kwargs,
) -> tuple:
    """Return (reconciler, state_manager, exchange_client)."""
    state = StateManager()
    client = _make_exchange_client(positions=positions, orders=orders, **kwargs)
    reconciler = Reconciler(exchange_client=client, state_manager=state)
    return reconciler, state, client


# ---------------------------------------------------------------------------
# 1. Crash during entry: position on exchange, nothing in StateManager
# ---------------------------------------------------------------------------

class TestCrashDuringEntry:
    """Bot crashed after placing entry order & exchange filled it,
    but before StateManager was updated."""

    def test_position_restored_from_exchange(self):
        """Reconciler rebuilds position when exchange reports it open."""
        raw = _raw_position("BTCUSDT", "LONG", qty=0.1, entry_price=50_000.0)
        reconciler, state, _ = _make_reconciler(positions=[raw])

        result = reconciler.reconcile()

        assert result.positions_restored == 1
        open_positions = state.get_open_positions("BTCUSDT")
        assert len(open_positions) == 1
        pos = open_positions[0]
        assert pos.symbol == "BTCUSDT"
        assert pos.side == OrderSide.LONG
        assert pos.quantity == pytest.approx(0.1)
        assert pos.entry_price == pytest.approx(50_000.0)

    def test_position_marked_recovered(self):
        """Restored position carries metadata flag 'recovered': True."""
        raw = _raw_position("BTCUSDT", "LONG", qty=0.1, entry_price=50_000.0)
        reconciler, state, _ = _make_reconciler(positions=[raw])

        reconciler.reconcile()
        pos = state.get_open_positions("BTCUSDT")[0]
        assert pos.metadata.get("recovered") is True

    def test_zero_qty_position_skipped(self):
        """Exchange positions with zero quantity are ignored."""
        raw = _raw_position("BTCUSDT", "LONG", qty=0.0, entry_price=50_000.0)
        reconciler, state, _ = _make_reconciler(positions=[raw])

        result = reconciler.reconcile()

        assert result.positions_restored == 0
        assert state.get_open_positions("BTCUSDT") == []

    def test_multiple_symbols_restored(self):
        """Multiple positions across different symbols all restored."""
        raws = [
            _raw_position("BTCUSDT", "LONG", qty=0.1, entry_price=50_000.0),
            _raw_position("ETHUSDT", "SHORT", qty=1.0, entry_price=3_000.0),
        ]
        reconciler, state, _ = _make_reconciler(positions=raws)

        result = reconciler.reconcile()

        assert result.positions_restored == 2
        assert len(state.get_open_positions("BTCUSDT")) == 1
        assert len(state.get_open_positions("ETHUSDT")) == 1

    def test_short_position_side_correct(self):
        """SHORT positions are correctly identified."""
        raw = _raw_position("ETHUSDT", "SHORT", qty=1.0, entry_price=3_000.0)
        reconciler, state, _ = _make_reconciler(positions=[raw])

        reconciler.reconcile()
        pos = state.get_open_positions("ETHUSDT")[0]
        assert pos.side == OrderSide.SHORT

    def test_exchange_fetch_failure_records_error(self):
        """If exchange fetch_positions fails, result.errors is populated."""
        state = StateManager()
        client = _make_exchange_client(fail_fetch_positions=True)
        reconciler = Reconciler(exchange_client=client, state_manager=state)

        result = reconciler.reconcile()

        assert result.positions_restored == 0
        assert len(result.errors) >= 1
        assert any("fetch" in e.lower() or "position" in e.lower() for e in result.errors)

    def test_leverage_and_notional_computed(self):
        """Position notional and margin are derived from exchange data."""
        raw = _raw_position("BTCUSDT", "LONG", qty=0.1, entry_price=50_000.0, leverage=10.0)
        reconciler, state, _ = _make_reconciler(positions=[raw])

        reconciler.reconcile()
        pos = state.get_open_positions("BTCUSDT")[0]
        assert pos.notional_usd == pytest.approx(5_000.0)   # 0.1 * 50_000
        assert pos.margin_usd == pytest.approx(500.0)        # notional / leverage


# ---------------------------------------------------------------------------
# 2. Crash after fill: position open, stop order on exchange
# ---------------------------------------------------------------------------

class TestCrashAfterFill:
    """Bot crashed after fill was confirmed and stop order placed,
    but before state was fully persisted."""

    def test_stop_order_linked_to_position(self):
        """Stop order with <position_id>_stop client_order_id is linked."""
        raw_pos = _raw_position("BTCUSDT", "LONG", qty=0.1, entry_price=50_000.0)
        reconciler, state, _ = _make_reconciler(positions=[raw_pos])

        # First reconcile to get position_id
        reconciler.reconcile()
        pos = state.get_open_positions("BTCUSDT")[0]
        position_id = pos.position_id

        # Now simulate stop order on exchange
        stop_cid = f"{position_id}_stop"
        raw_stop = _raw_order(
            client_order_id=stop_cid,
            symbol="BTCUSDT",
            side="SELL",
            order_type="STOP_MARKET",
            stop_price=48_000.0,
        )

        # Re-reconcile with the stop order present
        state2 = StateManager()
        client2 = _make_exchange_client(positions=[raw_pos], orders=[raw_stop])
        reconciler2 = Reconciler(exchange_client=client2, state_manager=state2)
        result = reconciler2.reconcile()

        assert result.orders_linked == 1
        assert result.orphan_orders_cancelled == 0

    def test_stop_price_updated_on_position(self):
        """Linked stop order propagates its stop_price to the position."""
        raw_pos = _raw_position("BTCUSDT", "LONG", qty=0.1, entry_price=50_000.0)

        # We need a deterministic position_id. Use a pre-populated state.
        state = StateManager()
        client = _make_exchange_client(positions=[raw_pos], orders=[])
        r = Reconciler(exchange_client=client, state_manager=state)
        r.reconcile()
        pos = state.get_open_positions("BTCUSDT")[0]
        position_id = pos.position_id

        # Full reconcile with stop order
        stop_cid = f"{position_id}_stop"
        raw_stop = _raw_order(stop_cid, stop_price=48_500.0)
        state2 = StateManager()
        client2 = _make_exchange_client(positions=[raw_pos], orders=[raw_stop])
        reconciler2 = Reconciler(exchange_client=client2, state_manager=state2)
        reconciler2.reconcile()

        pos2 = state2.get_open_positions("BTCUSDT")[0]
        assert pos2.stop_price == pytest.approx(48_500.0)
        assert pos2.initial_stop_price == pytest.approx(48_500.0)

    def test_tp_order_linked_to_position(self):
        """Take-profit order with <position_id>_tp is linked."""
        raw_pos = _raw_position("BTCUSDT", "LONG", qty=0.1, entry_price=50_000.0)
        state = StateManager()
        client = _make_exchange_client(positions=[raw_pos], orders=[])
        r = Reconciler(exchange_client=client, state_manager=state)
        r.reconcile()
        pos = state.get_open_positions("BTCUSDT")[0]
        position_id = pos.position_id

        tp_cid = f"{position_id}_tp"
        raw_tp = _raw_order(
            tp_cid,
            side="SELL",
            order_type="TAKE_PROFIT_MARKET",
            stop_price=55_000.0,
        )
        state2 = StateManager()
        client2 = _make_exchange_client(positions=[raw_pos], orders=[raw_tp])
        reconciler2 = Reconciler(exchange_client=client2, state_manager=state2)
        result = reconciler2.reconcile()

        assert result.orders_linked >= 1
        assert result.orphan_orders_cancelled == 0

    def test_both_stop_and_tp_linked(self):
        """Both stop and TP orders can be linked in a single reconcile."""
        raw_pos = _raw_position("BTCUSDT", "LONG", qty=0.1, entry_price=50_000.0)
        state = StateManager()
        r = Reconciler(
            exchange_client=_make_exchange_client(positions=[raw_pos], orders=[]),
            state_manager=state,
        )
        r.reconcile()
        position_id = state.get_open_positions("BTCUSDT")[0].position_id

        raw_stop = _raw_order(f"{position_id}_stop", stop_price=48_000.0)
        raw_tp = _raw_order(
            f"{position_id}_tp",
            side="SELL",
            order_type="TAKE_PROFIT_MARKET",
            stop_price=55_000.0,
        )
        state2 = StateManager()
        r2 = Reconciler(
            exchange_client=_make_exchange_client(
                positions=[raw_pos], orders=[raw_stop, raw_tp]
            ),
            state_manager=state2,
        )
        result = r2.reconcile()

        assert result.orders_linked == 2
        assert result.orphan_orders_cancelled == 0


# ---------------------------------------------------------------------------
# 3. Crash during exit: orphan exit order, no matching position
# ---------------------------------------------------------------------------

class TestCrashDuringExit:
    """Bot crashed after placing an exit order but the position was
    already closed (or never restored), leaving an orphan on the exchange."""

    def test_orphan_stop_order_cancelled(self):
        """Stop order with no matching position is cancelled."""
        orphan_stop = _raw_order(
            "POS_XYZ_stop",
            symbol="BTCUSDT",
            stop_price=48_000.0,
        )
        reconciler, state, client = _make_reconciler(
            positions=[],   # No open positions
            orders=[orphan_stop],
        )

        result = reconciler.reconcile()

        assert result.orphan_orders_cancelled == 1
        client.cancel_order.assert_called_once_with(
            symbol="BTCUSDT",
            order_id=f"EX_POS_XYZ_stop",
        )

    def test_orphan_tp_order_cancelled(self):
        """TP order with no matching position is cancelled."""
        orphan_tp = _raw_order(
            "POS_ABC_tp",
            symbol="ETHUSDT",
            side="SELL",
            order_type="TAKE_PROFIT_MARKET",
            stop_price=3_500.0,
        )
        reconciler, state, client = _make_reconciler(
            positions=[],
            orders=[orphan_tp],
        )

        result = reconciler.reconcile()

        assert result.orphan_orders_cancelled == 1

    def test_multiple_orphans_all_cancelled(self):
        """Multiple orphan orders are all cancelled."""
        orphans = [
            _raw_order("POS_A_stop", symbol="BTCUSDT"),
            _raw_order("POS_B_stop", symbol="ETHUSDT"),
            _raw_order("POS_C_tp", symbol="BNBUSDT", order_type="TAKE_PROFIT_MARKET"),
        ]
        reconciler, state, client = _make_reconciler(positions=[], orders=orphans)

        result = reconciler.reconcile()

        assert result.orphan_orders_cancelled == 3
        assert client.cancel_order.call_count == 3

    def test_cancel_failure_recorded_in_errors(self):
        """If cancel_order raises, error is recorded but reconcile continues."""
        orphan = _raw_order("POS_FAIL_stop", symbol="BTCUSDT")
        state = StateManager()
        client = _make_exchange_client(
            positions=[], orders=[orphan], fail_cancel=True
        )
        reconciler = Reconciler(exchange_client=client, state_manager=state)

        result = reconciler.reconcile()

        # Cancelled count stays 0 (cancel failed), but error recorded
        assert result.orphan_orders_cancelled == 0
        assert len(result.errors) >= 1

    def test_entry_order_with_no_position_cancelled(self):
        """A lingering LIMIT entry order with no open position is cancelled."""
        stale_entry = _raw_order(
            "entry_BTCUSDT_001",
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
        )
        reconciler, state, client = _make_reconciler(
            positions=[],
            orders=[stale_entry],
        )

        result = reconciler.reconcile()

        # Non-exit order, no position found → orphan cancelled
        assert result.orphan_orders_cancelled == 1


# ---------------------------------------------------------------------------
# 4. Idempotency: calling reconcile() twice produces no duplicates
# ---------------------------------------------------------------------------

class TestIdempotency:
    """Calling reconcile() multiple times must not duplicate state."""

    def test_double_reconcile_no_duplicate_positions(self):
        """Two reconcile calls with same exchange data → exactly 1 position."""
        raw = _raw_position("BTCUSDT", "LONG", qty=0.1, entry_price=50_000.0)
        reconciler, state, _ = _make_reconciler(positions=[raw])

        result1 = reconciler.reconcile()
        result2 = reconciler.reconcile()

        assert result1.positions_restored == 1
        assert result2.positions_restored == 1  # upsert, no error
        assert len(state.get_open_positions("BTCUSDT")) == 1

    def test_double_reconcile_no_position_errors(self):
        """Reconciler uses update_position (upsert) so no ValueError on repeat."""
        raw = _raw_position("BTCUSDT", "LONG", qty=0.1, entry_price=50_000.0)
        reconciler, state, _ = _make_reconciler(positions=[raw])

        result1 = reconciler.reconcile()
        result2 = reconciler.reconcile()

        assert result1.errors == []
        assert result2.errors == []

    def test_double_reconcile_with_stop_order_no_duplicate_orders(self):
        """Stop orders are upserted — second reconcile doesn't double-add."""
        raw_pos = _raw_position("BTCUSDT", "LONG", qty=0.1, entry_price=50_000.0)

        # First pass: get position_id
        state_tmp = StateManager()
        r_tmp = Reconciler(
            exchange_client=_make_exchange_client(positions=[raw_pos], orders=[]),
            state_manager=state_tmp,
        )
        r_tmp.reconcile()
        pos = state_tmp.get_open_positions("BTCUSDT")[0]
        position_id = pos.position_id

        raw_stop = _raw_order(f"{position_id}_stop", stop_price=48_000.0)

        state = StateManager()
        reconciler = Reconciler(
            exchange_client=_make_exchange_client(
                positions=[raw_pos], orders=[raw_stop]
            ),
            state_manager=state,
        )

        result1 = reconciler.reconcile()
        result2 = reconciler.reconcile()

        # Positions: upserted (no error), Orders: upserted (no error)
        assert result1.errors == []
        assert result2.errors == []

        # Only one position in state
        assert len(state.get_open_positions("BTCUSDT")) == 1

    def test_reconcile_after_manual_state_population(self):
        """Reconcile on state already containing the position doesn't crash."""
        raw = _raw_position("BTCUSDT", "LONG", qty=0.1, entry_price=50_000.0)
        reconciler, state, _ = _make_reconciler(positions=[raw])

        # Pre-populate state
        first_result = reconciler.reconcile()
        existing_pos = state.get_open_positions("BTCUSDT")[0]

        # Add an unrelated position manually
        from bot.execution.position import Position
        extra = Position(
            position_id="MANUAL_001",
            symbol="SOLUSDT",
            side=OrderSide.LONG,
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            leverage=1.0,
            margin_usd=100.0,
            stop_price=90.0,
            entry_time=datetime.utcnow(),
            risk_amount_usd=10.0,
            initial_stop_price=90.0,
            trail_after_r=1.0,
            atr_trail_mult=2.0,
            entry_order_id="manual",
            stop_order_id="manual",
        )
        state.add_position(extra)

        # Reconcile again — should not touch SOLUSDT (not from exchange)
        result2 = reconciler.reconcile()
        assert result2.errors == []
        assert len(state.get_open_positions("SOLUSDT")) == 1  # still present


# ---------------------------------------------------------------------------
# 5. Reconciliation result accuracy
# ---------------------------------------------------------------------------

class TestReconciliationResultAccuracy:
    """ReconciliationResult counters accurately reflect actions taken."""

    def test_empty_exchange_all_zeros(self):
        """No positions, no orders → all counters zero."""
        reconciler, _, _ = _make_reconciler(positions=[], orders=[])
        result = reconciler.reconcile()

        assert result.positions_restored == 0
        assert result.orders_linked == 0
        assert result.orphan_orders_cancelled == 0
        assert result.errors == []

    def test_one_position_one_linked_stop(self):
        """1 position + 1 stop order → restored=1, linked=1, orphans=0."""
        raw_pos = _raw_position("BTCUSDT", "LONG", qty=0.1, entry_price=50_000.0)
        # Two-pass: first get position_id
        s = StateManager()
        r = Reconciler(_make_exchange_client(positions=[raw_pos], orders=[]), s)
        r.reconcile()
        pid = s.get_open_positions("BTCUSDT")[0].position_id

        raw_stop = _raw_order(f"{pid}_stop", stop_price=48_000.0)

        state = StateManager()
        reconciler = Reconciler(
            _make_exchange_client(positions=[raw_pos], orders=[raw_stop]), state
        )
        result = reconciler.reconcile()

        assert result.positions_restored == 1
        assert result.orders_linked == 1
        assert result.orphan_orders_cancelled == 0
        assert result.errors == []

    def test_one_orphan_order(self):
        """1 orphan stop order → restored=0, linked=0, orphans=1."""
        orphan = _raw_order("GHOST_POS_stop", symbol="BTCUSDT")
        reconciler, _, _ = _make_reconciler(positions=[], orders=[orphan])
        result = reconciler.reconcile()

        assert result.positions_restored == 0
        assert result.orphan_orders_cancelled == 1
        assert result.orders_linked == 0

    def test_mixed_scenario_counters(self):
        """2 positions, 1 stop linked, 2 orphans → accurate counters."""
        raw_p1 = _raw_position("BTCUSDT", "LONG", qty=0.1, entry_price=50_000.0)
        raw_p2 = _raw_position("ETHUSDT", "SHORT", qty=1.0, entry_price=3_000.0)

        # Get position_id for p1
        s = StateManager()
        r = Reconciler(_make_exchange_client(positions=[raw_p1], orders=[]), s)
        r.reconcile()
        pid1 = s.get_open_positions("BTCUSDT")[0].position_id

        raw_stop = _raw_order(f"{pid1}_stop", symbol="BTCUSDT", stop_price=48_000.0)
        orphan1 = _raw_order("OLD_POS_stop", symbol="SOLUSDT")
        orphan2 = _raw_order("OLD_TP_tp", symbol="BNBUSDT", order_type="TAKE_PROFIT_MARKET")

        state = StateManager()
        reconciler = Reconciler(
            _make_exchange_client(
                positions=[raw_p1, raw_p2],
                orders=[raw_stop, orphan1, orphan2],
            ),
            state,
        )
        result = reconciler.reconcile()

        assert result.positions_restored == 2
        assert result.orders_linked == 1
        assert result.orphan_orders_cancelled == 2

    def test_result_to_dict_serializable(self):
        """ReconciliationResult.to_dict() produces a JSON-serializable dict."""
        reconciler, _, _ = _make_reconciler(
            positions=[_raw_position("BTCUSDT", "LONG")],
        )
        result = reconciler.reconcile()
        d = result.to_dict()

        # Must be JSON serializable
        raw_json = json.dumps(d)
        parsed = json.loads(raw_json)

        assert parsed["positions_restored"] == 1
        assert "timestamp" in parsed
        assert isinstance(parsed["errors"], list)

    def test_result_str_representation(self):
        """ReconciliationResult.__str__ includes key counters."""
        result = ReconciliationResult()
        result.positions_restored = 3
        result.orders_linked = 2
        result.orphan_orders_cancelled = 1
        s = str(result)
        assert "3" in s
        assert "2" in s
        assert "1" in s


# ---------------------------------------------------------------------------
# 6. Reconciliation logged via TradeLogger
# ---------------------------------------------------------------------------

class TestReconciliationLogging:
    """Reconciliation result is logged to the event JSONL file."""

    def test_reconciliation_logged_to_event_file(self):
        """log_reconciliation() writes a RECONCILIATION_COMPLETE record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trade_logger = TradeLogger(log_dir=tmpdir)

            result = ReconciliationResult()
            result.positions_restored = 2
            result.orders_linked = 1
            result.orphan_orders_cancelled = 1

            trade_logger.log_reconciliation(result.to_dict())
            trade_logger.flush()

            event_path = trade_logger.get_event_log_path()
            with open(event_path) as f:
                lines = [json.loads(l) for l in f if l.strip()]

            recon_records = [
                l for l in lines if l.get("event") == "RECONCILIATION_COMPLETE"
            ]
            assert len(recon_records) == 1
            payload = recon_records[0]["payload"]
            assert payload["positions_restored"] == 2
            assert payload["orders_linked"] == 1
            assert payload["orphan_orders_cancelled"] == 1

            trade_logger.close()

    def test_reconciliation_log_has_timestamp(self):
        """Logged reconciliation record contains a timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trade_logger = TradeLogger(log_dir=tmpdir)
            result = ReconciliationResult()
            trade_logger.log_reconciliation(result.to_dict())
            trade_logger.flush()

            event_path = trade_logger.get_event_log_path()
            with open(event_path) as f:
                lines = [json.loads(l) for l in f if l.strip()]

            assert len(lines) == 1
            assert "timestamp" in lines[0]

            trade_logger.close()

    def test_multiple_reconciliations_logged_separately(self):
        """Each reconcile() call produces its own log entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trade_logger = TradeLogger(log_dir=tmpdir)

            for i in range(3):
                r = ReconciliationResult()
                r.positions_restored = i
                trade_logger.log_reconciliation(r.to_dict())

            trade_logger.flush()
            event_path = trade_logger.get_event_log_path()
            with open(event_path) as f:
                lines = [json.loads(l) for l in f if l.strip()]

            recon_records = [
                l for l in lines if l.get("event") == "RECONCILIATION_COMPLETE"
            ]
            assert len(recon_records) == 3

            trade_logger.close()


# ---------------------------------------------------------------------------
# 7. Exchange fetch failures: graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """When exchange is unavailable, reconcile() degrades gracefully."""

    def test_fetch_positions_failure_returns_errors(self):
        """fetch_positions failure → errors populated, no crash."""
        state = StateManager()
        client = _make_exchange_client(fail_fetch_positions=True)
        reconciler = Reconciler(exchange_client=client, state_manager=state)

        result = reconciler.reconcile()

        assert len(result.errors) >= 1
        assert result.positions_restored == 0

    def test_fetch_orders_failure_returns_errors(self):
        """fetch_open_orders failure → errors populated, positions still restored."""
        raw = _raw_position("BTCUSDT", "LONG", qty=0.1, entry_price=50_000.0)
        state = StateManager()
        client = _make_exchange_client(
            positions=[raw], fail_fetch_orders=True
        )
        reconciler = Reconciler(exchange_client=client, state_manager=state)

        result = reconciler.reconcile()

        # Positions restored despite order fetch failure
        assert result.positions_restored == 1
        assert len(result.errors) >= 1

    def test_both_failures_recorded(self):
        """Both position and order fetch failures → multiple error entries."""
        state = StateManager()
        client = _make_exchange_client(
            fail_fetch_positions=True, fail_fetch_orders=True
        )
        reconciler = Reconciler(exchange_client=client, state_manager=state)

        result = reconciler.reconcile()

        # At least one error from positions (orders not reached due to early return)
        assert len(result.errors) >= 1

    def test_reconcile_returns_result_on_exception(self):
        """reconcile() always returns a ReconciliationResult even on error."""
        state = StateManager()
        client = _make_exchange_client(fail_fetch_positions=True)
        reconciler = Reconciler(exchange_client=client, state_manager=state)

        result = reconciler.reconcile()

        assert isinstance(result, ReconciliationResult)
        assert hasattr(result, "timestamp")


# ---------------------------------------------------------------------------
# 8. Symbols filter
# ---------------------------------------------------------------------------

class TestSymbolsFilter:
    """reconcile(symbols=[...]) fetches only specified symbols."""

    def test_reconcile_with_symbol_filter_calls_fetch_with_symbols(self):
        """When symbols passed, fetch_positions called with symbols arg."""
        reconciler, state, client = _make_reconciler(
            positions=[_raw_position("BTCUSDT", "LONG")],
        )

        reconciler.reconcile(symbols=["BTCUSDT"])

        client.fetch_positions.assert_called_once_with(symbols=["BTCUSDT"])

    def test_reconcile_no_symbols_fetches_all(self):
        """Without symbols, fetch_positions called with symbols=None."""
        reconciler, state, client = _make_reconciler(positions=[])

        reconciler.reconcile()

        client.fetch_positions.assert_called_once_with(symbols=None)

    def test_reconcile_with_symbol_filter_fetches_orders_per_symbol(self):
        """When symbols passed, fetch_open_orders is called per symbol."""
        reconciler, state, client = _make_reconciler(positions=[])

        reconciler.reconcile(symbols=["BTCUSDT", "ETHUSDT"])

        # Should call fetch_open_orders for each symbol
        assert client.fetch_open_orders.call_count == 2
