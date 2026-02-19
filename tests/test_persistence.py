"""Persistence tests: TradeLogger + LogReader"""

import json
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest

from bot.execution.models import (
    Order,
    OrderPurpose,
    OrderSide,
    OrderStatus,
    OrderType,
)
from bot.execution.position import ExitReason, Position, PositionStatus
from bot.state.logger import (
    RECORD_TYPE_ERROR,
    RECORD_TYPE_EVENT,
    RECORD_TYPE_ORDER,
    RECORD_TYPE_TRADE,
    TradeLogger,
)
from bot.state.log_reader import LogReadError, LogReader, LogRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    """Temporary directory for log files."""
    return tmp_path / "logs"


@pytest.fixture
def trade_logger(tmp_log_dir: Path) -> Generator[TradeLogger, None, None]:
    """TradeLogger pointed at a temp directory."""
    logger = TradeLogger(log_dir=str(tmp_log_dir))
    yield logger
    logger.close()


@pytest.fixture
def log_reader(tmp_log_dir: Path) -> LogReader:
    """LogReader pointed at the same temp directory."""
    return LogReader(log_dir=str(tmp_log_dir))


def make_open_position(
    position_id: str = "BTCUSDT_LONG_50000_1000",
    symbol: str = "BTCUSDT",
    side: OrderSide = OrderSide.LONG,
    entry_price: float = 50_000.0,
    quantity: float = 0.1,
    stop_price: float = 49_500.0,
) -> Position:
    """Build a minimal open position for testing."""
    return Position(
        position_id=position_id,
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        quantity=quantity,
        notional_usd=entry_price * quantity,
        leverage=10.0,
        margin_usd=(entry_price * quantity) / 10.0,
        stop_price=stop_price,
        entry_time=datetime.utcnow(),
        risk_amount_usd=abs(entry_price - stop_price) * quantity,
        initial_stop_price=stop_price,
        trail_after_r=1.0,
        atr_trail_mult=2.0,
        entry_order_id="entry_001",
        stop_order_id="stop_001",
    )


def make_closed_position(
    position_id: str = "BTCUSDT_LONG_50000_1000",
    exit_price: float = 51_000.0,
    exit_reason: ExitReason = ExitReason.TP,
) -> Position:
    """Build a closed position for testing."""
    pos = make_open_position(position_id=position_id)
    pos.close_position(exit_price=exit_price, exit_reason=exit_reason)
    return pos


def make_order(
    client_order_id: str = "entry_001",
    symbol: str = "BTCUSDT",
    side: OrderSide = OrderSide.LONG,
    order_type: OrderType = OrderType.LIMIT,
    purpose: OrderPurpose = OrderPurpose.ENTRY,
    status: OrderStatus = OrderStatus.FILLED,
    quantity: float = 0.1,
    price: float = 50_000.0,
    filled_quantity: float = 0.1,
    avg_fill_price: float = 50_000.0,
    position_id: str = "BTCUSDT_LONG_50000_1000",
) -> Order:
    """Build a minimal order for testing."""
    order = Order(
        symbol=symbol,
        side=side,
        order_type=order_type,
        purpose=purpose,
        quantity=quantity,
        client_order_id=client_order_id,
        price=price,
        status=status,
        filled_quantity=filled_quantity,
        avg_fill_price=avg_fill_price,
        position_id=position_id,
        timestamp_submitted=datetime.utcnow(),
        timestamp_updated=datetime.utcnow(),
    )
    return order


# ---------------------------------------------------------------------------
# TradeLogger — directory setup
# ---------------------------------------------------------------------------


class TestTradeLoggerSetup:
    def test_creates_log_dir(self, tmp_path: Path) -> None:
        """TradeLogger creates log directory if it doesn't exist."""
        new_dir = tmp_path / "new" / "nested" / "logs"
        assert not new_dir.exists()
        logger = TradeLogger(log_dir=str(new_dir))
        assert new_dir.exists()
        logger.close()

    def test_get_trade_log_path(self, trade_logger: TradeLogger) -> None:
        """get_trade_log_path returns correct path for date."""
        d = date(2024, 1, 15)
        path = trade_logger.get_trade_log_path(d)
        assert path.name == "trades_2024-01-15.jsonl"

    def test_get_event_log_path(self, trade_logger: TradeLogger) -> None:
        """get_event_log_path returns correct path for date."""
        d = date(2024, 6, 30)
        path = trade_logger.get_event_log_path(d)
        assert path.name == "events_2024-06-30.jsonl"

    def test_list_empty_dir(self, trade_logger: TradeLogger) -> None:
        """list_trade_logs returns empty list when no files exist."""
        assert trade_logger.list_trade_logs() == []
        assert trade_logger.list_event_logs() == []


# ---------------------------------------------------------------------------
# TradeLogger — log_trade_closed
# ---------------------------------------------------------------------------


class TestLogTradeClosed:
    def test_creates_trade_file(self, trade_logger: TradeLogger, tmp_log_dir: Path) -> None:
        """log_trade_closed creates a trades JSONL file."""
        pos = make_closed_position()
        trade_logger.log_trade_closed(pos)

        files = list(tmp_log_dir.glob("trades_*.jsonl"))
        assert len(files) == 1

    def test_record_structure(self, trade_logger: TradeLogger, tmp_log_dir: Path) -> None:
        """log_trade_closed writes a valid JSONL record."""
        pos = make_closed_position()
        trade_logger.log_trade_closed(pos)

        trade_file = trade_logger.get_trade_log_path()
        with open(trade_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["record_type"] == RECORD_TYPE_TRADE
        assert record["event"] == "TRADE_CLOSED"
        assert "timestamp" in record
        assert record["payload"]["position_id"] == pos.position_id

    def test_pnl_in_payload(self, trade_logger: TradeLogger) -> None:
        """log_trade_closed includes PnL fields in payload."""
        pos = make_closed_position(exit_price=51_000.0, exit_reason=ExitReason.TP)
        trade_logger.log_trade_closed(pos)

        trade_file = trade_logger.get_trade_log_path()
        with open(trade_file, "r") as f:
            record = json.loads(f.readline())

        payload = record["payload"]
        assert "realized_pnl_usd" in payload
        assert "exit_reason" in payload
        assert payload["exit_reason"] == "TP"

    def test_open_position_logs_warning(
        self, trade_logger: TradeLogger, caplog
    ) -> None:
        """log_trade_closed on an open position logs a warning."""
        pos = make_open_position()
        import logging
        with caplog.at_level(logging.WARNING, logger="trading_bot.state.logger"):
            trade_logger.log_trade_closed(pos)
        assert any("open position" in r.message for r in caplog.records)

    def test_multiple_trades_appended(
        self, trade_logger: TradeLogger
    ) -> None:
        """Multiple log_trade_closed calls append to the same file."""
        for i in range(5):
            pos = make_closed_position(
                position_id=f"pos_{i}",
                exit_price=50_000.0 + i * 100,
            )
            trade_logger.log_trade_closed(pos)

        trade_file = trade_logger.get_trade_log_path()
        with open(trade_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# TradeLogger — log_trade_opened
# ---------------------------------------------------------------------------


class TestLogTradeOpened:
    def test_creates_trade_file(self, trade_logger: TradeLogger) -> None:
        """log_trade_opened creates a trade record."""
        pos = make_open_position()
        trade_logger.log_trade_opened(pos)

        trade_file = trade_logger.get_trade_log_path()
        with open(trade_file, "r") as f:
            record = json.loads(f.readline())

        assert record["event"] == "TRADE_OPENED"
        assert record["payload"]["position_id"] == pos.position_id
        assert record["payload"]["status"] == "OPEN"


# ---------------------------------------------------------------------------
# TradeLogger — log_order_filled
# ---------------------------------------------------------------------------


class TestLogOrderFilled:
    def test_order_fill_goes_to_events(
        self, trade_logger: TradeLogger
    ) -> None:
        """log_order_filled writes to the events log."""
        order = make_order()
        trade_logger.log_order_filled(order)

        event_file = trade_logger.get_event_log_path()
        assert event_file.exists()

        with open(event_file, "r") as f:
            record = json.loads(f.readline())

        assert record["record_type"] == RECORD_TYPE_ORDER
        assert record["event"] == "ORDER_FILLED"
        assert record["payload"]["client_order_id"] == order.client_order_id

    def test_order_cancel_event(self, trade_logger: TradeLogger) -> None:
        """log_order_cancelled writes a cancellation event."""
        order = make_order(status=OrderStatus.CANCELED)
        trade_logger.log_order_cancelled(order, reason="TTL expired")

        event_file = trade_logger.get_event_log_path()
        with open(event_file, "r") as f:
            record = json.loads(f.readline())

        assert record["event"] == "ORDER_CANCELLED"
        assert record["payload"]["cancel_reason"] == "TTL expired"


# ---------------------------------------------------------------------------
# TradeLogger — log_event / log_error
# ---------------------------------------------------------------------------


class TestLogEvent:
    def test_log_event_basic(self, trade_logger: TradeLogger) -> None:
        """log_event writes a generic event record."""
        trade_logger.log_event("RISK_CHECK_PASSED", {"notional": 10_000.0})

        event_file = trade_logger.get_event_log_path()
        with open(event_file, "r") as f:
            record = json.loads(f.readline())

        assert record["record_type"] == RECORD_TYPE_EVENT
        assert record["event"] == "RISK_CHECK_PASSED"
        assert record["payload"]["notional"] == 10_000.0

    def test_log_event_level_warning(self, trade_logger: TradeLogger) -> None:
        """log_event stores the level field."""
        trade_logger.log_event("HIGH_DRAWDOWN", {}, level="WARNING")

        event_file = trade_logger.get_event_log_path()
        with open(event_file, "r") as f:
            record = json.loads(f.readline())

        assert record["level"] == "WARNING"

    def test_log_error(self, trade_logger: TradeLogger) -> None:
        """log_error writes an ERROR record."""
        trade_logger.log_error("risk_manager", "Notional limit exceeded", {"notional": 60_000})

        event_file = trade_logger.get_event_log_path()
        with open(event_file, "r") as f:
            record = json.loads(f.readline())

        assert record["record_type"] == RECORD_TYPE_ERROR
        assert record["level"] == "ERROR"
        assert record["payload"]["component"] == "risk_manager"
        assert record["payload"]["message"] == "Notional limit exceeded"

    def test_log_kill_switch(self, trade_logger: TradeLogger) -> None:
        """log_kill_switch writes KILL_SWITCH_ACTIVATED event."""
        trade_logger.log_kill_switch("Daily loss limit reached")

        event_file = trade_logger.get_event_log_path()
        with open(event_file, "r") as f:
            record = json.loads(f.readline())

        assert record["event"] == "KILL_SWITCH_ACTIVATED"
        assert record["level"] == "WARNING"
        assert record["payload"]["reason"] == "Daily loss limit reached"

    def test_log_reconciliation(self, trade_logger: TradeLogger) -> None:
        """log_reconciliation writes RECONCILIATION_COMPLETE event."""
        result = {
            "timestamp": "2024-01-01T00:00:00",
            "positions_restored": 3,
            "orders_linked": 6,
            "orphan_orders_cancelled": 1,
            "unknown_orders_skipped": 0,
            "errors": [],
        }
        trade_logger.log_reconciliation(result)

        event_file = trade_logger.get_event_log_path()
        with open(event_file, "r") as f:
            record = json.loads(f.readline())

        assert record["event"] == "RECONCILIATION_COMPLETE"
        assert record["payload"]["positions_restored"] == 3


# ---------------------------------------------------------------------------
# TradeLogger — daily rotation
# ---------------------------------------------------------------------------


class TestDailyRotation:
    def test_rotation_changes_file(
        self, tmp_log_dir: Path
    ) -> None:
        """Log files rotate when date changes (simulated via direct file writes)."""
        # Write a record into a day-1 events file manually
        d1 = date(2024, 1, 15)
        d2 = date(2024, 1, 16)
        tmp_log_dir.mkdir(parents=True, exist_ok=True)

        for d, event_name in [(d1, "DAY1_EVENT"), (d2, "DAY2_EVENT")]:
            record = {
                "record_type": RECORD_TYPE_EVENT,
                "event": event_name,
                "timestamp": datetime.utcnow().isoformat(),
                "level": "INFO",
                "payload": {},
            }
            file_path = tmp_log_dir / f"events_{d.isoformat()}.jsonl"
            with open(file_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        # Both event files should exist
        event_files = list(tmp_log_dir.glob("events_*.jsonl"))
        assert len(event_files) == 2

        # LogReader can read both
        reader = LogReader(log_dir=str(tmp_log_dir))
        records = list(reader.read_events())
        assert len(records) == 2
        event_names = {r.event for r in records}
        assert event_names == {"DAY1_EVENT", "DAY2_EVENT"}

    def test_list_trade_logs_sorted(
        self, tmp_log_dir: Path
    ) -> None:
        """list_trade_logs returns files sorted by date."""
        # Create dummy log files manually (dir must exist first)
        tmp_log_dir.mkdir(parents=True, exist_ok=True)
        for d in ["2024-01-01", "2024-01-03", "2024-01-02"]:
            (tmp_log_dir / f"trades_{d}.jsonl").touch()

        logger = TradeLogger(log_dir=str(tmp_log_dir))
        files = logger.list_trade_logs()
        names = [f.name for f in files]
        assert names == sorted(names)
        logger.close()


# ---------------------------------------------------------------------------
# TradeLogger — flush / close
# ---------------------------------------------------------------------------


class TestFlushClose:
    def test_close_then_write_new_handle(
        self, tmp_log_dir: Path
    ) -> None:
        """After close(), a new write opens a fresh handle."""
        logger = TradeLogger(log_dir=str(tmp_log_dir))
        logger.log_event("BEFORE_CLOSE", {})
        logger.close()
        # After close, should work again
        logger.log_event("AFTER_CLOSE", {})
        logger.close()

        event_file = logger.get_event_log_path()
        with open(event_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2

    def test_flush_does_not_raise(self, trade_logger: TradeLogger) -> None:
        """flush() on logger with no open handles does not raise."""
        trade_logger.flush()  # No handles yet — should be a no-op


# ---------------------------------------------------------------------------
# LogReader — basic reading
# ---------------------------------------------------------------------------


class TestLogReaderBasic:
    def test_read_empty_dir(self, log_reader: LogReader) -> None:
        """read_trades on empty dir yields nothing."""
        records = list(log_reader.read_trades())
        assert records == []

    def test_read_single_trade_record(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """read_trades yields one LogRecord for one trade."""
        pos = make_closed_position()
        trade_logger.log_trade_closed(pos)
        trade_logger.flush()

        records = list(log_reader.read_trades())
        assert len(records) == 1
        record = records[0]
        assert isinstance(record, LogRecord)
        assert record.record_type == RECORD_TYPE_TRADE
        assert record.event == "TRADE_CLOSED"
        assert record.payload["position_id"] == pos.position_id

    def test_read_multiple_trade_records(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """read_trades yields all appended records."""
        positions = []
        for i in range(3):
            pos = make_closed_position(position_id=f"pos_{i}", exit_price=51_000.0 + i)
            trade_logger.log_trade_closed(pos)
            positions.append(pos)
        trade_logger.flush()

        records = list(log_reader.read_trades())
        assert len(records) == 3

    def test_read_event_records(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """read_events yields LogRecord from events log."""
        trade_logger.log_event("TEST_EVENT", {"key": "value"})
        trade_logger.flush()

        records = list(log_reader.read_events())
        assert len(records) == 1
        assert records[0].event == "TEST_EVENT"
        assert records[0].payload["key"] == "value"


# ---------------------------------------------------------------------------
# LogReader — LogRecord fields
# ---------------------------------------------------------------------------


class TestLogRecordFields:
    def test_timestamp_is_datetime(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """LogRecord.timestamp is a datetime object."""
        pos = make_closed_position()
        trade_logger.log_trade_closed(pos)
        trade_logger.flush()

        record = list(log_reader.read_trades())[0]
        assert isinstance(record.timestamp, datetime)

    def test_record_repr(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """LogRecord has a readable repr."""
        trade_logger.log_event("REPR_TEST", {})
        trade_logger.flush()

        record = list(log_reader.read_events())[0]
        r = repr(record)
        assert "REPR_TEST" in r
        assert "EVENT" in r

    def test_raw_field_is_original_line(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """LogRecord.raw contains the original JSON string."""
        trade_logger.log_event("RAW_TEST", {"x": 42})
        trade_logger.flush()

        record = list(log_reader.read_events())[0]
        assert "RAW_TEST" in record.raw
        parsed = json.loads(record.raw)
        assert parsed["event"] == "RAW_TEST"


# ---------------------------------------------------------------------------
# LogReader — get_closed_trades
# ---------------------------------------------------------------------------


class TestGetClosedTrades:
    def test_returns_only_closed_trades(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """get_closed_trades returns only TRADE_CLOSED records."""
        closed = make_closed_position(position_id="pos_closed")
        opened = make_open_position(position_id="pos_opened")

        trade_logger.log_trade_closed(closed)
        trade_logger.log_trade_opened(opened)
        trade_logger.flush()

        trades = log_reader.get_closed_trades()
        assert len(trades) == 1
        assert trades[0]["position_id"] == "pos_closed"

    def test_symbol_filter(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """get_closed_trades filters by symbol."""
        btc_pos = make_closed_position(position_id="btc_1")  # symbol defaults to BTCUSDT
        eth_pos = make_open_position(
            position_id="eth_1",
            symbol="ETHUSDT",
            entry_price=3000.0,
            stop_price=2950.0,
        )
        eth_pos.close_position(exit_price=3100.0, exit_reason=ExitReason.TP)

        trade_logger.log_trade_closed(btc_pos)
        trade_logger.log_trade_closed(eth_pos)
        trade_logger.flush()

        btc_trades = log_reader.get_closed_trades(symbol="BTCUSDT")
        assert len(btc_trades) == 1
        assert btc_trades[0]["symbol"] == "BTCUSDT"

        eth_trades = log_reader.get_closed_trades(symbol="ETHUSDT")
        assert len(eth_trades) == 1
        assert eth_trades[0]["symbol"] == "ETHUSDT"

    def test_sorted_by_entry_time(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """get_closed_trades returns records sorted by entry_time."""
        for i in range(3):
            pos = make_closed_position(position_id=f"pos_{i}")
            trade_logger.log_trade_closed(pos)
        trade_logger.flush()

        trades = log_reader.get_closed_trades()
        entry_times = [t["entry_time"] for t in trades]
        assert entry_times == sorted(entry_times)


# ---------------------------------------------------------------------------
# LogReader — get_opened_trades (crash recovery)
# ---------------------------------------------------------------------------


class TestGetOpenedTrades:
    def test_returns_open_trades_only(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """get_opened_trades returns positions not yet closed."""
        opened = make_open_position(position_id="still_open")
        opened_then_closed = make_open_position(position_id="then_closed")

        trade_logger.log_trade_opened(opened)
        trade_logger.log_trade_opened(opened_then_closed)

        # Close the second one
        opened_then_closed.close_position(51_000.0, ExitReason.TP)
        trade_logger.log_trade_closed(opened_then_closed)
        trade_logger.flush()

        open_trades = log_reader.get_opened_trades()
        assert len(open_trades) == 1
        assert open_trades[0]["position_id"] == "still_open"

    def test_no_open_trades(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """get_opened_trades returns empty list when all trades closed."""
        pos = make_open_position()
        trade_logger.log_trade_opened(pos)
        pos.close_position(51_000.0, ExitReason.SL)
        trade_logger.log_trade_closed(pos)
        trade_logger.flush()

        open_trades = log_reader.get_opened_trades()
        assert open_trades == []


# ---------------------------------------------------------------------------
# LogReader — get_events / get_errors
# ---------------------------------------------------------------------------


class TestGetEvents:
    def test_get_events_all(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """get_events returns all event records."""
        trade_logger.log_event("EVENT_A", {})
        trade_logger.log_event("EVENT_B", {})
        trade_logger.log_event("EVENT_C", {})
        trade_logger.flush()

        events = log_reader.get_events()
        assert len(events) == 3

    def test_get_events_filtered(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """get_events filters by event_name."""
        trade_logger.log_event("KILL_SWITCH_ACTIVATED", {"reason": "test"})
        trade_logger.log_event("RISK_CHECK_PASSED", {})
        trade_logger.log_event("KILL_SWITCH_ACTIVATED", {"reason": "again"})
        trade_logger.flush()

        ks_events = log_reader.get_events(event_name="KILL_SWITCH_ACTIVATED")
        assert len(ks_events) == 2

        risk_events = log_reader.get_events(event_name="RISK_CHECK_PASSED")
        assert len(risk_events) == 1

    def test_get_errors(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """get_errors returns only ERROR records."""
        trade_logger.log_error("component_a", "Something failed")
        trade_logger.log_event("NORMAL_EVENT", {})
        trade_logger.log_error("component_b", "Another failure")
        trade_logger.flush()

        errors = log_reader.get_errors()
        assert len(errors) == 2
        assert all(r.record_type == RECORD_TYPE_ERROR for r in errors)


# ---------------------------------------------------------------------------
# LogReader — compute_pnl_summary
# ---------------------------------------------------------------------------


class TestComputePnlSummary:
    def test_empty_summary(self, log_reader: LogReader) -> None:
        """compute_pnl_summary returns zeros when no trades."""
        summary = log_reader.compute_pnl_summary()
        assert summary["total_trades"] == 0
        assert summary["win_rate"] == 0.0
        assert summary["total_pnl_usd"] == 0.0

    def test_pnl_summary_with_trades(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """compute_pnl_summary correctly aggregates trade PnL."""
        # 2 winners, 1 loser
        winners = [
            make_closed_position(f"pos_w{i}", exit_price=51_000.0)
            for i in range(2)
        ]
        loser = make_closed_position("pos_l", exit_price=49_800.0, exit_reason=ExitReason.SL)

        for pos in winners:
            trade_logger.log_trade_closed(pos)
        trade_logger.log_trade_closed(loser)
        trade_logger.flush()

        summary = log_reader.compute_pnl_summary()
        assert summary["total_trades"] == 3
        assert summary["winning_trades"] == 2
        assert summary["losing_trades"] == 1
        assert abs(summary["win_rate"] - 2 / 3) < 0.001

    def test_exit_reason_counts(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """compute_pnl_summary includes exit_reason_counts."""
        reasons = [ExitReason.TP, ExitReason.SL, ExitReason.TP, ExitReason.TRAIL]
        exit_prices = [51_000.0, 49_800.0, 51_200.0, 50_600.0]

        for i, (reason, ep) in enumerate(zip(reasons, exit_prices)):
            pos = make_closed_position(f"pos_{i}", exit_price=ep, exit_reason=reason)
            trade_logger.log_trade_closed(pos)
        trade_logger.flush()

        summary = log_reader.compute_pnl_summary()
        counts = summary["exit_reason_counts"]
        assert counts.get("TP", 0) == 2
        assert counts.get("SL", 0) == 1
        assert counts.get("TRAIL", 0) == 1

    def test_symbol_filter_in_summary(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """compute_pnl_summary respects symbol filter."""
        btc = make_closed_position("btc_1", exit_price=51_000.0)
        eth_pos = make_open_position(
            position_id="eth_1",
            symbol="ETHUSDT",
            entry_price=3000.0,
            stop_price=2950.0,
        )
        eth_pos.close_position(3100.0, ExitReason.TP)

        trade_logger.log_trade_closed(btc)
        trade_logger.log_trade_closed(eth_pos)
        trade_logger.flush()

        btc_summary = log_reader.compute_pnl_summary(symbol="BTCUSDT")
        assert btc_summary["total_trades"] == 1

        eth_summary = log_reader.compute_pnl_summary(symbol="ETHUSDT")
        assert eth_summary["total_trades"] == 1


# ---------------------------------------------------------------------------
# LogReader — find_position_events
# ---------------------------------------------------------------------------


class TestFindPositionEvents:
    def test_finds_trade_and_event_records(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """find_position_events returns records from both logs for the position."""
        pos = make_open_position(position_id="target_pos")
        trade_logger.log_trade_opened(pos)
        trade_logger.log_event("TRAILING_ENABLED", {"position_id": "target_pos"})
        pos.close_position(51_000.0, ExitReason.TRAIL)
        trade_logger.log_trade_closed(pos)
        trade_logger.flush()

        events = log_reader.find_position_events("target_pos")
        # Should find TRADE_OPENED + TRAILING_ENABLED event + TRADE_CLOSED
        assert len(events) == 3

    def test_does_not_return_other_positions(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """find_position_events only returns records for the given position."""
        pos1 = make_open_position(position_id="pos_1")
        pos2 = make_open_position(position_id="pos_2")

        trade_logger.log_trade_opened(pos1)
        trade_logger.log_trade_opened(pos2)
        trade_logger.flush()

        events = log_reader.find_position_events("pos_1")
        assert all(
            r.payload.get("position_id") == "pos_1" for r in events
        )


# ---------------------------------------------------------------------------
# LogReader — error handling
# ---------------------------------------------------------------------------


class TestLogReaderErrorHandling:
    def test_skip_bad_lines_by_default(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """Bad JSONL lines are skipped when skip_errors=True."""
        # Write a valid record first
        trade_logger.log_trade_closed(make_closed_position())
        trade_logger.flush()

        # Corrupt the file by appending a bad line
        trade_file = trade_logger.get_trade_log_path()
        with open(trade_file, "a") as f:
            f.write("NOT_VALID_JSON\n")
            f.write("{\"incomplete\":\n")

        records = list(log_reader.read_trades(skip_errors=True))
        assert len(records) == 1  # Only the valid record

    def test_raise_on_bad_line_when_skip_errors_false(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """Bad JSONL lines raise LogReadError when skip_errors=False."""
        trade_logger.log_trade_closed(make_closed_position())
        trade_logger.flush()

        trade_file = trade_logger.get_trade_log_path()
        with open(trade_file, "a") as f:
            f.write("NOT_VALID_JSON\n")

        with pytest.raises(LogReadError):
            list(log_reader.read_trades(skip_errors=False))

    def test_nonexistent_file_returns_empty(
        self, log_reader: LogReader
    ) -> None:
        """read_trades on missing path returns empty iterator."""
        records = list(log_reader.read_trades())
        assert records == []

    def test_empty_lines_skipped(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """Empty lines in log file are skipped gracefully."""
        trade_logger.log_event("EVENT_1", {})
        trade_logger.flush()

        event_file = trade_logger.get_event_log_path()
        with open(event_file, "a") as f:
            f.write("\n\n\n")  # Blank lines

        records = list(log_reader.read_events())
        assert len(records) == 1

    def test_missing_record_type_raises_read_error(
        self, tmp_log_dir: Path, log_reader: LogReader
    ) -> None:
        """Record missing record_type field raises LogReadError."""
        today = datetime.utcnow().date().isoformat()
        trade_file = tmp_log_dir / f"trades_{today}.jsonl"
        trade_file.parent.mkdir(parents=True, exist_ok=True)
        with open(trade_file, "w") as f:
            # Missing "record_type"
            f.write(json.dumps({"event": "TRADE_CLOSED", "timestamp": "2024-01-01T00:00:00", "payload": {}}) + "\n")

        with pytest.raises(LogReadError):
            list(log_reader.read_trades(skip_errors=False))


# ---------------------------------------------------------------------------
# LogReader — date range filtering
# ---------------------------------------------------------------------------


class TestDateRangeFiltering:
    def _write_log_for_date(
        self,
        tmp_log_dir: Path,
        for_date: date,
        event_name: str,
    ) -> None:
        """Write a single record to a dated log file."""
        tmp_log_dir.mkdir(parents=True, exist_ok=True)
        file_path = tmp_log_dir / f"trades_{for_date.isoformat()}.jsonl"
        record = {
            "record_type": RECORD_TYPE_TRADE,
            "event": "TRADE_CLOSED",
            "timestamp": datetime.utcnow().isoformat(),
            "level": "INFO",
            "payload": {
                "position_id": event_name,
                "symbol": "BTCUSDT",
                "entry_time": "2024-01-01T00:00:00",
                "realized_pnl_usd": 100.0,
                "exit_reason": "TP",
                "pnl_r": 1.0,
                "fees_paid_usd": 1.0,
            },
        }
        with open(file_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def test_start_filter(self, tmp_log_dir: Path) -> None:
        """read_trades(start=...) skips files before start date."""
        reader = LogReader(log_dir=str(tmp_log_dir))
        self._write_log_for_date(tmp_log_dir, date(2024, 1, 1), "jan1")
        self._write_log_for_date(tmp_log_dir, date(2024, 1, 3), "jan3")
        self._write_log_for_date(tmp_log_dir, date(2024, 1, 5), "jan5")

        records = list(reader.read_trades(start=date(2024, 1, 3)))
        assert len(records) == 2
        ids = {r.payload["position_id"] for r in records}
        assert ids == {"jan3", "jan5"}

    def test_end_filter(self, tmp_log_dir: Path) -> None:
        """read_trades(end=...) skips files after end date."""
        reader = LogReader(log_dir=str(tmp_log_dir))
        self._write_log_for_date(tmp_log_dir, date(2024, 1, 1), "jan1")
        self._write_log_for_date(tmp_log_dir, date(2024, 1, 3), "jan3")
        self._write_log_for_date(tmp_log_dir, date(2024, 1, 5), "jan5")

        records = list(reader.read_trades(end=date(2024, 1, 3)))
        assert len(records) == 2
        ids = {r.payload["position_id"] for r in records}
        assert ids == {"jan1", "jan3"}

    def test_start_and_end_filter(self, tmp_log_dir: Path) -> None:
        """read_trades with both start and end filters correctly."""
        reader = LogReader(log_dir=str(tmp_log_dir))
        for d in [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]:
            self._write_log_for_date(tmp_log_dir, d, d.isoformat())

        records = list(reader.read_trades(start=date(2024, 1, 2), end=date(2024, 1, 2)))
        assert len(records) == 1
        assert records[0].payload["position_id"] == "2024-01-02"

    def test_no_filter_reads_all(self, tmp_log_dir: Path) -> None:
        """read_trades with no filter reads all files."""
        reader = LogReader(log_dir=str(tmp_log_dir))
        for d in [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]:
            self._write_log_for_date(tmp_log_dir, d, d.isoformat())

        records = list(reader.read_trades())
        assert len(records) == 3


# ---------------------------------------------------------------------------
# Integration: TradeLogger + LogReader roundtrip
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_write_and_read_trade(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """Full roundtrip: write a trade with TradeLogger, read with LogReader."""
        pos = make_open_position(
            position_id="BTC_RT_001",
            symbol="BTCUSDT",
            entry_price=45_000.0,
            quantity=0.2,
            stop_price=44_500.0,
        )
        trade_logger.log_trade_opened(pos)
        pos.close_position(exit_price=46_000.0, exit_reason=ExitReason.TP)
        trade_logger.log_trade_closed(pos)
        trade_logger.flush()

        trades = log_reader.get_closed_trades()
        assert len(trades) == 1
        t = trades[0]
        assert t["position_id"] == "BTC_RT_001"
        assert t["symbol"] == "BTCUSDT"
        assert t["exit_reason"] == "TP"
        assert t["exit_price"] == 46_000.0

    def test_crash_recovery_scenario(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """Crash recovery: open trades can be retrieved from log."""
        # Simulate: 2 positions opened, only 1 closed before "crash"
        pos1 = make_open_position(position_id="survivor", entry_price=50_000.0)
        pos2 = make_open_position(position_id="closed_one", entry_price=51_000.0)

        trade_logger.log_trade_opened(pos1)
        trade_logger.log_trade_opened(pos2)

        # pos2 closed before crash
        pos2.close_position(52_000.0, ExitReason.TP)
        trade_logger.log_trade_closed(pos2)
        trade_logger.flush()

        # "Restart": create new reader
        new_reader = LogReader(log_dir=str(trade_logger._log_dir))
        open_trades = new_reader.get_opened_trades()

        assert len(open_trades) == 1
        assert open_trades[0]["position_id"] == "survivor"

    def test_full_pnl_summary_roundtrip(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """PnL summary computed correctly from logged trades."""
        # 3 trades: 2 wins, 1 loss
        wins = [
            make_closed_position(f"win_{i}", exit_price=51_000.0)
            for i in range(2)
        ]
        loss = make_closed_position("loss_1", exit_price=49_800.0, exit_reason=ExitReason.SL)

        for pos in wins:
            trade_logger.log_trade_closed(pos)
        trade_logger.log_trade_closed(loss)
        trade_logger.flush()

        summary = log_reader.compute_pnl_summary()
        assert summary["total_trades"] == 3
        assert summary["winning_trades"] == 2
        assert summary["losing_trades"] == 1
        assert 0.666 < summary["win_rate"] < 0.667
        # 2 wins at +$10 each (qty=0.1, price diff=$1000 = $100 per win)
        # 1 loss at -$20 (qty=0.1, price diff=-$200 = -$20)
        # Total gross = $200 - $20 = $180
        assert summary["total_pnl_usd"] > 0

    def test_multiple_events_in_sequence(
        self, trade_logger: TradeLogger, log_reader: LogReader
    ) -> None:
        """A full trade lifecycle is logged and retrievable."""
        pos = make_open_position(position_id="lifecycle_pos")

        # Open
        trade_logger.log_trade_opened(pos)
        # Trail enabled
        trade_logger.log_event("TRAILING_ENABLED", {"position_id": pos.position_id, "price": 50_500.0})
        # Stop updated
        trade_logger.log_event("STOP_UPDATED", {"position_id": pos.position_id, "new_stop": 50_200.0})
        # Close
        pos.close_position(50_100.0, ExitReason.TRAIL)
        trade_logger.log_trade_closed(pos)
        trade_logger.flush()

        events = log_reader.find_position_events("lifecycle_pos")
        event_names = [e.event for e in events]
        assert "TRADE_OPENED" in event_names
        assert "TRAILING_ENABLED" in event_names
        assert "STOP_UPDATED" in event_names
        assert "TRADE_CLOSED" in event_names
