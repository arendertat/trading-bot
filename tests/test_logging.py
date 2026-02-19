"""
Tests for Task 15: Trade & Event Logging

Covers:
- LOG_SCHEMA.json field compliance (TradeRecord, sub-records)
- build_trade_record() factory correctness
- ReportingTradeLogger.log_full_trade() roundtrip
- Schema-validated vs raw logging distinction
- Edge cases: zero risk, open position guard, unknown strategy/regime
- Integration with bot.state.log_reader.LogReader
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Generator

import pytest

from bot.execution.models import OrderSide
from bot.execution.position import ExitReason, Position, PositionStatus
from bot.reporting.trade_logger import (
    CostsRecord,
    EntryOrderRecord,
    PortfolioRecord,
    ReportingTradeLogger,
    ResultRecord,
    RiskRecord,
    TradeRecord,
    build_trade_record,
)
from bot.state.logger import RECORD_TYPE_TRADE, TradeLogger
from bot.state.log_reader import LogReader


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    """Temporary log directory (created by TradeLogger on demand)."""
    return tmp_path / "logs"


@pytest.fixture
def reporting_logger(tmp_log_dir: Path) -> Generator[ReportingTradeLogger, None, None]:
    """ReportingTradeLogger backed by a temp directory."""
    rl = ReportingTradeLogger(log_dir=str(tmp_log_dir))
    yield rl
    rl.close()


@pytest.fixture
def log_reader(tmp_log_dir: Path) -> LogReader:
    """LogReader pointed at the same temp directory."""
    return LogReader(log_dir=str(tmp_log_dir))


def make_open_position(
    position_id: str = "BTCUSDT_LONG_50000_001",
    symbol: str = "BTCUSDT",
    side: OrderSide = OrderSide.LONG,
    entry_price: float = 50_000.0,
    quantity: float = 0.1,
    stop_price: float = 49_500.0,
    strategy: str = "TREND_PULLBACK",
    regime: str = "TREND",
    confidence: float = 0.75,
    leverage: float = 10.0,
) -> Position:
    """Build a minimal open position for testing."""
    notional = entry_price * quantity
    margin = notional / leverage
    risk = abs(entry_price - stop_price) * quantity
    return Position(
        position_id=position_id,
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        quantity=quantity,
        notional_usd=notional,
        leverage=leverage,
        margin_usd=margin,
        stop_price=stop_price,
        entry_time=datetime(2024, 2, 11, 10, 30, 0),
        risk_amount_usd=risk,
        initial_stop_price=stop_price,
        trail_after_r=1.0,
        atr_trail_mult=2.0,
        entry_order_id="entry_001",
        stop_order_id="stop_001",
        strategy=strategy,
        regime=regime,
        confidence=confidence,
    )


def make_closed_position(
    position_id: str = "BTCUSDT_LONG_50000_001",
    exit_price: float = 51_000.0,
    exit_reason: ExitReason = ExitReason.TP,
    **kwargs,
) -> Position:
    """Build a closed position for testing."""
    pos = make_open_position(position_id=position_id, **kwargs)
    pos.close_position(exit_price=exit_price, exit_reason=exit_reason)
    return pos


def make_portfolio(
    open_positions_count: int = 2,
    open_risk_pct: float = 0.02,
    correlation_bucket: str = "CRYPTO_L1",
    bucket_corr_max: float = 0.65,
) -> PortfolioRecord:
    """Build a minimal PortfolioRecord."""
    return PortfolioRecord(
        open_positions_count=open_positions_count,
        open_risk_pct=open_risk_pct,
        correlation_bucket=correlation_bucket,
        bucket_corr_max=bucket_corr_max,
    )


# ---------------------------------------------------------------------------
# EntryOrderRecord — LOG_SCHEMA.json compliance
# ---------------------------------------------------------------------------


class TestEntryOrderRecord:
    def test_required_fields_present(self) -> None:
        """EntryOrderRecord serializes all LOG_SCHEMA entry_order fields."""
        rec = EntryOrderRecord(
            client_order_id="entry_001",
            type="LIMIT",
            requested_price=50_000.0,
            filled_avg_price=50_010.0,
            filled_qty=0.1,
            status="FILLED",
        )
        d = rec.to_dict()

        # All LOG_SCHEMA.json entry_order fields must be present
        assert "client_order_id" in d
        assert "type" in d
        assert "requested_price" in d
        assert "filled_avg_price" in d
        assert "filled_qty" in d
        assert "status" in d

    def test_field_values(self) -> None:
        """EntryOrderRecord stores values correctly."""
        rec = EntryOrderRecord(
            client_order_id="oid_123",
            type="MARKET",
            requested_price=45_000.0,
            filled_avg_price=45_005.0,
            filled_qty=0.2,
            status="FILLED",
        )
        d = rec.to_dict()
        assert d["client_order_id"] == "oid_123"
        assert d["type"] == "MARKET"
        assert d["requested_price"] == 45_000.0
        assert d["filled_avg_price"] == 45_005.0
        assert d["filled_qty"] == 0.2
        assert d["status"] == "FILLED"

    def test_valid_order_types(self) -> None:
        """EntryOrderRecord accepts LIMIT and MARKET types."""
        for ot in ("LIMIT", "MARKET"):
            rec = EntryOrderRecord("id", ot, 1.0, 1.0, 1.0, "FILLED")
            assert rec.to_dict()["type"] == ot

    def test_valid_statuses(self) -> None:
        """EntryOrderRecord accepts all LOG_SCHEMA status values."""
        for status in ("NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED"):
            rec = EntryOrderRecord("id", "LIMIT", 1.0, 1.0, 1.0, status)
            assert rec.to_dict()["status"] == status


# ---------------------------------------------------------------------------
# RiskRecord — LOG_SCHEMA.json compliance
# ---------------------------------------------------------------------------


class TestRiskRecord:
    def test_required_fields_present(self) -> None:
        """RiskRecord serializes all LOG_SCHEMA risk fields."""
        rec = RiskRecord(
            equity_usd=10_000.0,
            risk_pct=0.01,
            risk_usd=100.0,
            stop_pct=0.01,
            stop_price=49_500.0,
            leverage=10.0,
            notional_usd=5_000.0,
            margin_used_usd=500.0,
            take_profit_price=52_000.0,
        )
        d = rec.to_dict()
        for key in (
            "equity_usd", "risk_pct", "risk_usd", "stop_pct", "stop_price",
            "take_profit_price", "leverage", "notional_usd", "margin_used_usd",
        ):
            assert key in d, f"Missing field: {key}"

    def test_take_profit_optional(self) -> None:
        """take_profit_price defaults to None when not provided."""
        rec = RiskRecord(
            equity_usd=10_000.0, risk_pct=0.01, risk_usd=100.0,
            stop_pct=0.01, stop_price=49_500.0, leverage=10.0,
            notional_usd=5_000.0, margin_used_usd=500.0,
        )
        assert rec.to_dict()["take_profit_price"] is None


# ---------------------------------------------------------------------------
# CostsRecord — LOG_SCHEMA.json compliance
# ---------------------------------------------------------------------------


class TestCostsRecord:
    def test_required_fields_present(self) -> None:
        """CostsRecord serializes all LOG_SCHEMA costs fields."""
        rec = CostsRecord(fees_usd=5.0, funding_usd=1.5, slippage_pct=0.0002)
        d = rec.to_dict()
        assert "fees_usd" in d
        assert "funding_usd" in d
        assert "slippage_pct" in d

    def test_zero_costs_valid(self) -> None:
        """CostsRecord with all zeros is valid."""
        rec = CostsRecord(fees_usd=0.0, funding_usd=0.0, slippage_pct=0.0)
        d = rec.to_dict()
        assert d["fees_usd"] == 0.0
        assert d["slippage_pct"] == 0.0


# ---------------------------------------------------------------------------
# ResultRecord — LOG_SCHEMA.json compliance
# ---------------------------------------------------------------------------


class TestResultRecord:
    def test_required_fields_present(self) -> None:
        """ResultRecord serializes all LOG_SCHEMA result fields."""
        rec = ResultRecord(
            exit_price=51_000.0,
            pnl_usd=100.0,
            pnl_r_multiple=1.0,
            reason="TP",
        )
        d = rec.to_dict()
        assert "exit_price" in d
        assert "pnl_usd" in d
        assert "pnl_r_multiple" in d
        assert "reason" in d

    def test_all_exit_reasons(self) -> None:
        """ResultRecord accepts all LOG_SCHEMA exit reason values."""
        for reason in ("TP", "SL", "TRAIL", "KILL_SWITCH", "MANUAL", "TIMEOUT"):
            rec = ResultRecord(exit_price=1.0, pnl_usd=0.0, pnl_r_multiple=0.0, reason=reason)
            assert rec.to_dict()["reason"] == reason

    def test_negative_pnl(self) -> None:
        """ResultRecord correctly stores negative PnL (loss)."""
        rec = ResultRecord(exit_price=49_500.0, pnl_usd=-50.0, pnl_r_multiple=-1.0, reason="SL")
        d = rec.to_dict()
        assert d["pnl_usd"] < 0
        assert d["pnl_r_multiple"] < 0


# ---------------------------------------------------------------------------
# PortfolioRecord — LOG_SCHEMA.json compliance
# ---------------------------------------------------------------------------


class TestPortfolioRecord:
    def test_required_fields_present(self) -> None:
        """PortfolioRecord serializes all LOG_SCHEMA portfolio fields."""
        rec = make_portfolio()
        d = rec.to_dict()
        assert "open_positions_count" in d
        assert "open_risk_pct" in d
        assert "correlation_bucket" in d
        assert "bucket_corr_max" in d

    def test_zero_positions(self) -> None:
        """PortfolioRecord with zero positions is valid."""
        rec = PortfolioRecord(
            open_positions_count=0,
            open_risk_pct=0.0,
            correlation_bucket="NONE",
            bucket_corr_max=0.0,
        )
        d = rec.to_dict()
        assert d["open_positions_count"] == 0
        assert d["open_risk_pct"] == 0.0


# ---------------------------------------------------------------------------
# TradeRecord — full LOG_SCHEMA.json compliance
# ---------------------------------------------------------------------------


class TestTradeRecord:
    def test_all_schema_fields_present(self) -> None:
        """TradeRecord.to_dict() includes every field from LOG_SCHEMA.json."""
        pos = make_closed_position()
        record = build_trade_record(
            position=pos,
            mode="PAPER_LIVE",
            equity_usd=10_000.0,
            risk_pct=0.01,
            portfolio=make_portfolio(),
        )
        d = record.to_dict()

        # Top-level LOG_SCHEMA.json fields
        for key in (
            "trade_id", "timestamp_open", "timestamp_close",
            "mode", "symbol", "strategy", "regime", "direction",
            "confidence_score", "entry_order", "risk", "costs", "result", "portfolio",
        ):
            assert key in d, f"Missing top-level field: {key}"

    def test_entry_order_nested_fields(self) -> None:
        """TradeRecord.entry_order contains all LOG_SCHEMA entry_order fields."""
        pos = make_closed_position()
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        d = record.to_dict()["entry_order"]
        for key in ("client_order_id", "type", "requested_price", "filled_avg_price", "filled_qty", "status"):
            assert key in d, f"Missing entry_order field: {key}"

    def test_risk_nested_fields(self) -> None:
        """TradeRecord.risk contains all LOG_SCHEMA risk fields."""
        pos = make_closed_position()
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        d = record.to_dict()["risk"]
        for key in ("equity_usd", "risk_pct", "risk_usd", "stop_pct", "stop_price",
                    "leverage", "notional_usd", "margin_used_usd"):
            assert key in d, f"Missing risk field: {key}"

    def test_costs_nested_fields(self) -> None:
        """TradeRecord.costs contains all LOG_SCHEMA costs fields."""
        pos = make_closed_position()
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        d = record.to_dict()["costs"]
        for key in ("fees_usd", "funding_usd", "slippage_pct"):
            assert key in d, f"Missing costs field: {key}"

    def test_result_nested_fields(self) -> None:
        """TradeRecord.result contains all LOG_SCHEMA result fields."""
        pos = make_closed_position()
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        d = record.to_dict()["result"]
        for key in ("exit_price", "pnl_usd", "pnl_r_multiple", "reason"):
            assert key in d, f"Missing result field: {key}"

    def test_portfolio_nested_fields(self) -> None:
        """TradeRecord.portfolio contains all LOG_SCHEMA portfolio fields."""
        pos = make_closed_position()
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        d = record.to_dict()["portfolio"]
        for key in ("open_positions_count", "open_risk_pct", "correlation_bucket", "bucket_corr_max"):
            assert key in d, f"Missing portfolio field: {key}"

    def test_mode_values(self) -> None:
        """TradeRecord accepts PAPER_LIVE and LIVE modes."""
        pos = make_closed_position()
        for mode in ("PAPER_LIVE", "LIVE"):
            record = build_trade_record(pos, mode, 10_000.0, 0.01, make_portfolio())
            assert record.to_dict()["mode"] == mode

    def test_direction_reflects_position_side(self) -> None:
        """TradeRecord direction matches the position side."""
        long_pos = make_closed_position(side=OrderSide.LONG)
        short_pos = make_open_position(
            position_id="short_001",
            side=OrderSide.SHORT,
            stop_price=50_500.0,
        )
        short_pos.close_position(49_000.0, ExitReason.TP)

        long_rec = build_trade_record(long_pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        short_rec = build_trade_record(short_pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())

        assert long_rec.direction == "LONG"
        assert short_rec.direction == "SHORT"

    def test_all_exit_reason_values(self) -> None:
        """TradeRecord.result.reason reflects all ExitReason values."""
        exit_map = {
            ExitReason.TP: "TP",
            ExitReason.SL: "SL",
            ExitReason.TRAIL: "TRAIL",
            ExitReason.KILL_SWITCH: "KILL_SWITCH",
            ExitReason.MANUAL: "MANUAL",
            ExitReason.TIMEOUT: "TIMEOUT",
        }
        for reason, expected in exit_map.items():
            pos = make_closed_position(exit_reason=reason)
            record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
            assert record.result.reason == expected

    def test_json_serializable(self) -> None:
        """TradeRecord.to_dict() produces JSON-serializable output."""
        pos = make_closed_position()
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        # Should not raise
        serialized = json.dumps(record.to_dict())
        parsed = json.loads(serialized)
        assert parsed["symbol"] == "BTCUSDT"


# ---------------------------------------------------------------------------
# build_trade_record() factory
# ---------------------------------------------------------------------------


class TestBuildTradeRecord:
    def test_raises_on_open_position(self) -> None:
        """build_trade_record raises ValueError for open positions."""
        pos = make_open_position()
        with pytest.raises(ValueError, match="open position"):
            build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())

    def test_trade_id_from_position_id(self) -> None:
        """trade_id is derived from position_id."""
        pos = make_closed_position(position_id="BTC_LONG_001")
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        assert record.trade_id == "BTC_LONG_001"

    def test_symbol_preserved(self) -> None:
        """Symbol is correctly copied from position."""
        pos = make_closed_position(symbol="ETHUSDT")
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        assert record.symbol == "ETHUSDT"

    def test_strategy_and_regime_preserved(self) -> None:
        """Strategy and regime are correctly copied from position."""
        pos = make_closed_position(strategy="RANGE_MEAN_REVERSION", regime="RANGE")
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        assert record.strategy == "RANGE_MEAN_REVERSION"
        assert record.regime == "RANGE"

    def test_confidence_score_preserved(self) -> None:
        """Confidence score is correctly copied from position."""
        pos = make_closed_position(confidence=0.82)
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        assert abs(record.confidence_score - 0.82) < 1e-9

    def test_equity_and_risk_pct_in_risk_block(self) -> None:
        """equity_usd and risk_pct are correctly stored in risk block."""
        pos = make_closed_position()
        record = build_trade_record(pos, "PAPER_LIVE", 15_000.0, 0.015, make_portfolio())
        assert record.risk.equity_usd == 15_000.0
        assert abs(record.risk.risk_pct - 0.015) < 1e-9

    def test_risk_usd_from_position(self) -> None:
        """risk_usd matches position.risk_amount_usd."""
        pos = make_closed_position(entry_price=50_000.0, stop_price=49_500.0, quantity=0.1)
        # risk = |50000 - 49500| * 0.1 = 50.0
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        assert abs(record.risk.risk_usd - 50.0) < 0.01

    def test_stop_pct_computed(self) -> None:
        """stop_pct is computed as |stop - entry| / entry."""
        pos = make_closed_position(entry_price=50_000.0, stop_price=49_500.0)
        # stop_pct = 500 / 50000 = 0.01
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        assert abs(record.risk.stop_pct - 0.01) < 1e-6

    def test_pnl_r_multiple_computed(self) -> None:
        """pnl_r_multiple = pnl_usd / risk_amount_usd."""
        pos = make_closed_position(
            entry_price=50_000.0, stop_price=49_500.0, quantity=0.1,
            exit_price=51_000.0,  # profit = $100, risk = $50 → R = 2.0
        )
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        # gross pnl = (51000 - 50000) * 0.1 = 100; risk = 50; R ≈ 2.0
        assert record.result.pnl_r_multiple > 0

    def test_slippage_default_zero(self) -> None:
        """Slippage defaults to 0.0 when not provided."""
        pos = make_closed_position()
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        assert record.costs.slippage_pct == 0.0

    def test_slippage_override(self) -> None:
        """Slippage can be overridden."""
        pos = make_closed_position()
        record = build_trade_record(
            pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio(),
            slippage_pct=0.0005,
        )
        assert abs(record.costs.slippage_pct - 0.0005) < 1e-9

    def test_fees_from_position(self) -> None:
        """fees_usd comes from position.fees_paid_usd."""
        pos = make_closed_position()
        pos.fees_paid_usd = 3.50
        pos.funding_paid_usd = 0.75
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        assert abs(record.costs.fees_usd - 3.50) < 0.001
        assert abs(record.costs.funding_usd - 0.75) < 0.001

    def test_unknown_strategy_fallback(self) -> None:
        """Positions without strategy get UNKNOWN in the record."""
        pos = make_closed_position(strategy=None)
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        assert record.strategy == "UNKNOWN"

    def test_unknown_regime_fallback(self) -> None:
        """Positions without regime get UNKNOWN in the record."""
        pos = make_closed_position(regime=None)
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        assert record.regime == "UNKNOWN"

    def test_zero_risk_r_multiple_safe(self) -> None:
        """R-multiple is 0.0 when risk_amount_usd is 0 (no ZeroDivisionError)."""
        pos = make_closed_position()
        pos.risk_amount_usd = 0.0
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        assert record.result.pnl_r_multiple == 0.0

    def test_timestamp_open_from_entry_time(self) -> None:
        """timestamp_open is derived from position.entry_time."""
        pos = make_closed_position()
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        # entry_time was set to 2024-02-11T10:30:00 in make_open_position
        assert "2024-02-11" in record.timestamp_open

    def test_timestamp_close_from_exit_time(self) -> None:
        """timestamp_close is derived from position.exit_time when set."""
        pos = make_closed_position()
        assert pos.exit_time is not None
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        # Should contain the exit_time date (not a fallback)
        assert record.timestamp_close != ""


# ---------------------------------------------------------------------------
# ReportingTradeLogger — log_full_trade
# ---------------------------------------------------------------------------


class TestReportingTradeLoggerLogFullTrade:
    def test_creates_trade_file(
        self, reporting_logger: ReportingTradeLogger, tmp_log_dir: Path
    ) -> None:
        """log_full_trade creates a trades JSONL file."""
        pos = make_closed_position()
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        reporting_logger.log_full_trade(record)
        reporting_logger.flush()

        files = list(tmp_log_dir.glob("trades_*.jsonl"))
        assert len(files) == 1

    def test_record_type_is_trade(
        self, reporting_logger: ReportingTradeLogger, tmp_log_dir: Path
    ) -> None:
        """log_full_trade writes RECORD_TYPE_TRADE record."""
        pos = make_closed_position()
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        reporting_logger.log_full_trade(record)
        reporting_logger.flush()

        trade_file = list(tmp_log_dir.glob("trades_*.jsonl"))[0]
        with open(trade_file) as f:
            data = json.loads(f.readline())

        assert data["record_type"] == RECORD_TYPE_TRADE
        assert data["event"] == "TRADE_RECORD"

    def test_payload_contains_schema_fields(
        self, reporting_logger: ReportingTradeLogger, tmp_log_dir: Path
    ) -> None:
        """log_full_trade payload contains all LOG_SCHEMA.json top-level fields."""
        pos = make_closed_position()
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        reporting_logger.log_full_trade(record)
        reporting_logger.flush()

        trade_file = list(tmp_log_dir.glob("trades_*.jsonl"))[0]
        with open(trade_file) as f:
            data = json.loads(f.readline())

        payload = data["payload"]
        for key in (
            "trade_id", "timestamp_open", "timestamp_close", "mode", "symbol",
            "strategy", "regime", "direction", "confidence_score",
            "entry_order", "risk", "costs", "result", "portfolio",
        ):
            assert key in payload, f"Missing payload field: {key}"

    def test_nested_entry_order_fields(
        self, reporting_logger: ReportingTradeLogger, tmp_log_dir: Path
    ) -> None:
        """log_full_trade payload.entry_order contains all sub-fields."""
        pos = make_closed_position()
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        reporting_logger.log_full_trade(record)
        reporting_logger.flush()

        trade_file = list(tmp_log_dir.glob("trades_*.jsonl"))[0]
        with open(trade_file) as f:
            data = json.loads(f.readline())

        entry_order = data["payload"]["entry_order"]
        for key in ("client_order_id", "type", "requested_price", "filled_avg_price", "filled_qty", "status"):
            assert key in entry_order, f"Missing entry_order field: {key}"

    def test_nested_risk_fields(
        self, reporting_logger: ReportingTradeLogger, tmp_log_dir: Path
    ) -> None:
        """log_full_trade payload.risk contains all sub-fields."""
        pos = make_closed_position()
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        reporting_logger.log_full_trade(record)
        reporting_logger.flush()

        trade_file = list(tmp_log_dir.glob("trades_*.jsonl"))[0]
        with open(trade_file) as f:
            data = json.loads(f.readline())

        risk = data["payload"]["risk"]
        for key in ("equity_usd", "risk_pct", "risk_usd", "stop_pct",
                    "stop_price", "leverage", "notional_usd", "margin_used_usd"):
            assert key in risk, f"Missing risk field: {key}"

    def test_pnl_values_correct(
        self, reporting_logger: ReportingTradeLogger, tmp_log_dir: Path
    ) -> None:
        """log_full_trade stores correct PnL values."""
        # Long 0.1 BTC, entry 50000, exit 51000 → gross $100
        pos = make_closed_position(entry_price=50_000.0, exit_price=51_000.0, quantity=0.1)
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        reporting_logger.log_full_trade(record)
        reporting_logger.flush()

        trade_file = list(tmp_log_dir.glob("trades_*.jsonl"))[0]
        with open(trade_file) as f:
            data = json.loads(f.readline())

        result = data["payload"]["result"]
        assert result["exit_price"] == 51_000.0
        assert result["pnl_usd"] > 0  # Profitable trade (fees not applied in this test)
        assert result["pnl_r_multiple"] > 0
        assert result["reason"] == "TP"

    def test_multiple_trades_appended(
        self, reporting_logger: ReportingTradeLogger, tmp_log_dir: Path
    ) -> None:
        """Multiple log_full_trade calls append records to the same file."""
        for i in range(4):
            pos = make_closed_position(
                position_id=f"BTC_{i:03d}",
                exit_price=50_000.0 + i * 100,
            )
            record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
            reporting_logger.log_full_trade(record)
        reporting_logger.flush()

        trade_file = list(tmp_log_dir.glob("trades_*.jsonl"))[0]
        with open(trade_file) as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == 4

    def test_distinct_from_raw_trade_closed(
        self, reporting_logger: ReportingTradeLogger, tmp_log_dir: Path
    ) -> None:
        """log_full_trade (TRADE_RECORD) is distinct from log_trade_closed (TRADE_CLOSED)."""
        pos = make_closed_position()
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())

        reporting_logger.log_full_trade(record)
        reporting_logger.log_trade_closed(pos)
        reporting_logger.flush()

        trade_file = list(tmp_log_dir.glob("trades_*.jsonl"))[0]
        with open(trade_file) as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == 2
        events = {json.loads(line)["event"] for line in lines}
        assert "TRADE_RECORD" in events
        assert "TRADE_CLOSED" in events


# ---------------------------------------------------------------------------
# ReportingTradeLogger — passthrough methods
# ---------------------------------------------------------------------------


class TestReportingTradeLoggerPassthrough:
    def test_log_trade_opened_passthrough(
        self, reporting_logger: ReportingTradeLogger, tmp_log_dir: Path
    ) -> None:
        """log_trade_opened delegates to underlying TradeLogger."""
        pos = make_open_position()
        reporting_logger.log_trade_opened(pos)
        reporting_logger.flush()

        trade_file = list(tmp_log_dir.glob("trades_*.jsonl"))[0]
        with open(trade_file) as f:
            data = json.loads(f.readline())
        assert data["event"] == "TRADE_OPENED"

    def test_log_event_passthrough(
        self, reporting_logger: ReportingTradeLogger, tmp_log_dir: Path
    ) -> None:
        """log_event delegates to underlying TradeLogger."""
        reporting_logger.log_event("REGIME_CHANGE", {"from": "TREND", "to": "RANGE"})
        reporting_logger.flush()

        event_file = list(tmp_log_dir.glob("events_*.jsonl"))[0]
        with open(event_file) as f:
            data = json.loads(f.readline())
        assert data["event"] == "REGIME_CHANGE"
        assert data["payload"]["from"] == "TREND"

    def test_log_error_passthrough(
        self, reporting_logger: ReportingTradeLogger, tmp_log_dir: Path
    ) -> None:
        """log_error delegates to underlying TradeLogger."""
        reporting_logger.log_error("risk_engine", "Open risk exceeded limit", {"risk": 0.08})
        reporting_logger.flush()

        event_file = list(tmp_log_dir.glob("events_*.jsonl"))[0]
        with open(event_file) as f:
            data = json.loads(f.readline())
        assert data["level"] == "ERROR"
        assert data["payload"]["component"] == "risk_engine"

    def test_log_kill_switch_passthrough(
        self, reporting_logger: ReportingTradeLogger, tmp_log_dir: Path
    ) -> None:
        """log_kill_switch delegates to underlying TradeLogger."""
        reporting_logger.log_kill_switch("Daily loss limit", {"open_positions": 2})
        reporting_logger.flush()

        event_file = list(tmp_log_dir.glob("events_*.jsonl"))[0]
        with open(event_file) as f:
            data = json.loads(f.readline())
        assert data["event"] == "KILL_SWITCH_ACTIVATED"
        assert data["level"] == "WARNING"

    def test_underlying_logger_property(
        self, reporting_logger: ReportingTradeLogger
    ) -> None:
        """underlying_logger property returns the TradeLogger instance."""
        underlying = reporting_logger.underlying_logger
        assert isinstance(underlying, TradeLogger)


# ---------------------------------------------------------------------------
# Integration: ReportingTradeLogger + LogReader roundtrip
# ---------------------------------------------------------------------------


class TestReportingLoggerLogReaderIntegration:
    def test_full_trade_readable_by_log_reader(
        self,
        reporting_logger: ReportingTradeLogger,
        log_reader: LogReader,
        tmp_log_dir: Path,
    ) -> None:
        """TradeRecord logged by ReportingTradeLogger is readable via LogReader."""
        pos = make_closed_position(
            position_id="BTC_RT_999",
            exit_price=51_500.0,
        )
        record = build_trade_record(
            pos, "PAPER_LIVE", 12_000.0, 0.012, make_portfolio()
        )
        reporting_logger.log_full_trade(record)
        reporting_logger.flush()

        all_records = list(log_reader.read_trades())
        assert len(all_records) == 1
        lr = all_records[0]
        assert lr.record_type == RECORD_TYPE_TRADE
        assert lr.event == "TRADE_RECORD"
        assert lr.payload["trade_id"] == "BTC_RT_999"
        assert lr.payload["symbol"] == "BTCUSDT"
        assert lr.payload["mode"] == "PAPER_LIVE"

    def test_full_trade_schema_survives_json_roundtrip(
        self,
        reporting_logger: ReportingTradeLogger,
        log_reader: LogReader,
    ) -> None:
        """All nested sub-records survive a JSON write/read roundtrip."""
        pos = make_closed_position()
        record = build_trade_record(pos, "LIVE", 20_000.0, 0.02, make_portfolio())
        reporting_logger.log_full_trade(record)
        reporting_logger.flush()

        lr_records = list(log_reader.read_trades())
        payload = lr_records[0].payload

        # Nested blocks survive
        assert isinstance(payload["entry_order"], dict)
        assert isinstance(payload["risk"], dict)
        assert isinstance(payload["costs"], dict)
        assert isinstance(payload["result"], dict)
        assert isinstance(payload["portfolio"], dict)

        # Spot-check values
        assert payload["risk"]["equity_usd"] == 20_000.0
        assert payload["costs"]["slippage_pct"] == 0.0
        assert payload["portfolio"]["open_positions_count"] == 2

    def test_multiple_strategies_distinguishable(
        self,
        reporting_logger: ReportingTradeLogger,
        log_reader: LogReader,
    ) -> None:
        """Trades from different strategies can be distinguished in log records."""
        strategies = ["TREND_PULLBACK", "RANGE_MEAN_REVERSION", "TREND_BREAKOUT"]
        for i, strat in enumerate(strategies):
            pos = make_closed_position(
                position_id=f"pos_{i}",
                strategy=strat,
                regime="TREND" if "TREND" in strat else "RANGE",
            )
            record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
            reporting_logger.log_full_trade(record)
        reporting_logger.flush()

        records = list(log_reader.read_trades())
        assert len(records) == 3
        logged_strategies = {r.payload["strategy"] for r in records}
        assert logged_strategies == set(strategies)

    def test_paper_and_live_modes_logged(
        self,
        reporting_logger: ReportingTradeLogger,
        log_reader: LogReader,
    ) -> None:
        """Both PAPER_LIVE and LIVE modes are logged and distinguishable."""
        for mode in ("PAPER_LIVE", "LIVE"):
            pos = make_closed_position(position_id=f"pos_{mode}")
            record = build_trade_record(pos, mode, 10_000.0, 0.01, make_portfolio())
            reporting_logger.log_full_trade(record)
        reporting_logger.flush()

        records = list(log_reader.read_trades())
        modes = {r.payload["mode"] for r in records}
        assert modes == {"PAPER_LIVE", "LIVE"}

    def test_short_position_direction_logged(
        self,
        reporting_logger: ReportingTradeLogger,
        log_reader: LogReader,
    ) -> None:
        """Short positions are logged with direction=SHORT."""
        pos = make_open_position(
            position_id="SHORT_001",
            side=OrderSide.SHORT,
            entry_price=50_000.0,
            stop_price=50_500.0,
        )
        pos.close_position(49_000.0, ExitReason.TP)
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        reporting_logger.log_full_trade(record)
        reporting_logger.flush()

        records = list(log_reader.read_trades())
        assert records[0].payload["direction"] == "SHORT"

    def test_sl_exit_reason_logged(
        self,
        reporting_logger: ReportingTradeLogger,
        log_reader: LogReader,
    ) -> None:
        """Stop loss exit reason is correctly logged."""
        pos = make_closed_position(exit_price=49_500.0, exit_reason=ExitReason.SL)
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        reporting_logger.log_full_trade(record)
        reporting_logger.flush()

        records = list(log_reader.read_trades())
        assert records[0].payload["result"]["reason"] == "SL"

    def test_kill_switch_exit_reason_logged(
        self,
        reporting_logger: ReportingTradeLogger,
        log_reader: LogReader,
    ) -> None:
        """Kill switch exit reason is correctly logged."""
        pos = make_closed_position(exit_price=50_000.0, exit_reason=ExitReason.KILL_SWITCH)
        record = build_trade_record(pos, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        reporting_logger.log_full_trade(record)
        reporting_logger.flush()

        records = list(log_reader.read_trades())
        assert records[0].payload["result"]["reason"] == "KILL_SWITCH"

    def test_close_and_reopen_logger(
        self, tmp_log_dir: Path, log_reader: LogReader
    ) -> None:
        """Closing and re-creating ReportingTradeLogger still appends to the same file."""
        pos1 = make_closed_position(position_id="first_pos")
        pos2 = make_closed_position(position_id="second_pos")

        rl1 = ReportingTradeLogger(log_dir=str(tmp_log_dir))
        rec1 = build_trade_record(pos1, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        rl1.log_full_trade(rec1)
        rl1.close()

        rl2 = ReportingTradeLogger(log_dir=str(tmp_log_dir))
        rec2 = build_trade_record(pos2, "PAPER_LIVE", 10_000.0, 0.01, make_portfolio())
        rl2.log_full_trade(rec2)
        rl2.close()

        records = list(log_reader.read_trades())
        assert len(records) == 2
        ids = {r.payload["trade_id"] for r in records}
        assert ids == {"first_pos", "second_pos"}
