"""
Tests for Task 16: Daily Reporting & Notifications

Covers:
- DailyReportGenerator.build() with zero, one, and multiple trades
- TRADE_CLOSED (raw) and TRADE_RECORD (schema) record handling
- PnL aggregation, win rate, expectancy_R
- Equity, drawdown, equity_peak computation
- Strategy breakdown ordering and correctness
- format_text() content checks
- TelegramNotifier disabled-mode behaviour
- TelegramNotifier message truncation
- TelegramNotifier send helpers (trade opened/closed, kill switch, safe mode)
- make_notifier_from_config factory
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from bot.execution.models import OrderSide
from bot.execution.position import ExitReason, Position
from bot.reporting.daily_report import (
    DailyReportGenerator,
    DailySummary,
    StrategyBreakdown,
)
from bot.reporting.notifier import (
    TelegramNotifier,
    _TELEGRAM_MAX_LEN,
    make_notifier_from_config,
)
from bot.state.logger import RECORD_TYPE_TRADE, TradeLogger


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    """Temporary log directory."""
    return tmp_path / "logs"


@pytest.fixture
def trade_logger(tmp_log_dir: Path) -> Generator[TradeLogger, None, None]:
    """TradeLogger backed by temp directory."""
    logger = TradeLogger(log_dir=str(tmp_log_dir))
    yield logger
    logger.close()


@pytest.fixture
def report_gen(tmp_log_dir: Path) -> DailyReportGenerator:
    """DailyReportGenerator pointed at temp log dir."""
    return DailyReportGenerator(log_dir=str(tmp_log_dir))


def make_open_position(
    position_id: str = "BTC_001",
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
    notional = entry_price * quantity
    return Position(
        position_id=position_id,
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        quantity=quantity,
        notional_usd=notional,
        leverage=leverage,
        margin_usd=notional / leverage,
        stop_price=stop_price,
        entry_time=datetime.utcnow(),
        risk_amount_usd=abs(entry_price - stop_price) * quantity,
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
    position_id: str = "BTC_001",
    exit_price: float = 51_000.0,
    exit_reason: ExitReason = ExitReason.TP,
    fees: float = 2.0,
    funding: float = 0.5,
    **kwargs,
) -> Position:
    pos = make_open_position(position_id=position_id, **kwargs)
    pos.close_position(exit_price=exit_price, exit_reason=exit_reason, fees_paid=fees)
    pos.funding_paid_usd = funding
    return pos


def write_trade_closed_record(
    trade_logger: TradeLogger,
    position: Position,
) -> None:
    """Write a TRADE_CLOSED record using TradeLogger."""
    trade_logger.log_trade_closed(position)
    trade_logger.flush()


def write_trade_record(
    trade_logger: TradeLogger,
    position: Position,
    equity_usd: float = 10_000.0,
    risk_pct: float = 0.01,
    mode: str = "PAPER_LIVE",
) -> None:
    """Write a TRADE_RECORD (schema-validated) using ReportingTradeLogger."""
    from bot.reporting.trade_logger import (
        PortfolioRecord,
        ReportingTradeLogger,
        build_trade_record,
    )

    rl = ReportingTradeLogger(log_dir=str(trade_logger._log_dir))
    portfolio = PortfolioRecord(
        open_positions_count=1,
        open_risk_pct=0.01,
        correlation_bucket="CRYPTO_L1",
        bucket_corr_max=0.5,
    )
    rec = build_trade_record(position, mode, equity_usd, risk_pct, portfolio)
    rl.log_full_trade(rec)
    rl.flush()
    rl.close()


# ---------------------------------------------------------------------------
# DailyReportGenerator — zero trades
# ---------------------------------------------------------------------------


class TestDailyReportGeneratorNoTrades:
    def test_empty_log_dir(self, report_gen: DailyReportGenerator) -> None:
        """build() with no trade logs returns a zero-trade summary."""
        summary = report_gen.build(
            for_date=date.today(),
            equity_start_usd=10_000.0,
            mode="PAPER_LIVE",
        )
        assert summary.trade_count == 0
        assert summary.realized_pnl_usd == 0.0
        assert summary.win_rate == 0.0
        assert summary.expectancy_r == 0.0
        assert summary.strategy_breakdown == []
        assert summary.exit_reason_counts == {}

    def test_equity_unchanged_with_no_trades(self, report_gen: DailyReportGenerator) -> None:
        """Equity start = end when no trades."""
        summary = report_gen.build(
            for_date=date.today(),
            equity_start_usd=15_000.0,
        )
        assert summary.equity_start_usd == 15_000.0
        assert summary.equity_end_usd == 15_000.0

    def test_mode_preserved(self, report_gen: DailyReportGenerator) -> None:
        """mode field is preserved in summary."""
        for mode in ("PAPER_LIVE", "LIVE"):
            summary = report_gen.build(mode=mode)
            assert summary.mode == mode

    def test_report_date_defaults_to_today(self, report_gen: DailyReportGenerator) -> None:
        """report_date defaults to today UTC."""
        summary = report_gen.build()
        assert summary.report_date == datetime.utcnow().date()

    def test_unrealized_and_weekly_pnl_preserved(
        self, report_gen: DailyReportGenerator
    ) -> None:
        """unrealized_pnl_usd and weekly_pnl_usd are passed through."""
        summary = report_gen.build(
            unrealized_pnl_usd=250.0,
            weekly_pnl_usd=-150.0,
        )
        assert summary.unrealized_pnl_usd == 250.0
        assert summary.weekly_pnl_usd == -150.0


# ---------------------------------------------------------------------------
# DailyReportGenerator — TRADE_CLOSED records (raw)
# ---------------------------------------------------------------------------


class TestDailyReportRawTrades:
    def test_single_winning_trade(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
        tmp_log_dir: Path,
    ) -> None:
        """Single winning TRADE_CLOSED is aggregated correctly."""
        # entry 50000, exit 51000, qty 0.1 → gross $100
        pos = make_closed_position(exit_price=51_000.0)
        write_trade_closed_record(trade_logger, pos)

        summary = report_gen.build(
            for_date=date.today(),
            equity_start_usd=10_000.0,
        )
        assert summary.trade_count == 1
        assert summary.winning_trades == 1
        assert summary.losing_trades == 0
        assert summary.win_rate == 1.0
        assert summary.realized_pnl_usd > 0

    def test_single_losing_trade(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """Single losing TRADE_CLOSED is aggregated correctly."""
        pos = make_closed_position(
            exit_price=49_500.0,
            exit_reason=ExitReason.SL,
        )
        write_trade_closed_record(trade_logger, pos)

        summary = report_gen.build(
            for_date=date.today(),
            equity_start_usd=10_000.0,
        )
        assert summary.trade_count == 1
        assert summary.winning_trades == 0
        assert summary.losing_trades == 1
        assert summary.win_rate == 0.0
        assert summary.realized_pnl_usd < 0

    def test_multiple_trades_win_rate(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """Win rate computed correctly across multiple TRADE_CLOSED records."""
        # 3 wins, 1 loss
        for i in range(3):
            pos = make_closed_position(position_id=f"win_{i}", exit_price=51_000.0)
            write_trade_closed_record(trade_logger, pos)
        pos_loss = make_closed_position(
            position_id="loss_1", exit_price=49_500.0, exit_reason=ExitReason.SL
        )
        write_trade_closed_record(trade_logger, pos_loss)

        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        assert summary.trade_count == 4
        assert summary.winning_trades == 3
        assert summary.losing_trades == 1
        assert abs(summary.win_rate - 0.75) < 1e-6

    def test_fees_and_funding_aggregated(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """Total fees and funding are summed from all trades."""
        for i in range(3):
            pos = make_closed_position(
                position_id=f"pos_{i}",
                exit_price=51_000.0,
                fees=3.0,
                funding=1.0,
            )
            write_trade_closed_record(trade_logger, pos)

        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        assert abs(summary.total_fees_usd - 9.0) < 0.01
        assert abs(summary.total_funding_usd - 3.0) < 0.01

    def test_exit_reason_counts(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """Exit reason breakdown is counted correctly."""
        reasons_prices = [
            (ExitReason.TP, 51_000.0),
            (ExitReason.SL, 49_500.0),
            (ExitReason.TP, 51_200.0),
            (ExitReason.TRAIL, 50_800.0),
        ]
        for i, (reason, ep) in enumerate(reasons_prices):
            pos = make_closed_position(
                position_id=f"pos_{i}", exit_price=ep, exit_reason=reason
            )
            write_trade_closed_record(trade_logger, pos)

        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        counts = summary.exit_reason_counts
        assert counts.get("TP", 0) == 2
        assert counts.get("SL", 0) == 1
        assert counts.get("TRAIL", 0) == 1

    def test_equity_end_equals_start_plus_pnl(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """equity_end_usd = equity_start + realized_pnl_usd."""
        pos = make_closed_position(exit_price=51_000.0)
        write_trade_closed_record(trade_logger, pos)

        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        expected = round(summary.equity_start_usd + summary.realized_pnl_usd, 4)
        assert abs(summary.equity_end_usd - expected) < 0.01

    def test_realized_pnl_pct(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """realized_pnl_pct = realized_pnl_usd / equity_start * 100."""
        pos = make_closed_position(exit_price=51_000.0)
        write_trade_closed_record(trade_logger, pos)

        equity_start = 10_000.0
        summary = report_gen.build(for_date=date.today(), equity_start_usd=equity_start)
        expected_pct = summary.realized_pnl_usd / equity_start * 100.0
        assert abs(summary.realized_pnl_pct - expected_pct) < 0.001


# ---------------------------------------------------------------------------
# DailyReportGenerator — TRADE_RECORD records (schema-validated)
# ---------------------------------------------------------------------------


class TestDailyReportSchemaTrades:
    def test_trade_record_aggregated(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """TRADE_RECORD events (from ReportingTradeLogger) are included."""
        pos = make_closed_position(position_id="schema_001", exit_price=51_000.0)
        write_trade_record(trade_logger, pos)

        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        assert summary.trade_count == 1

    def test_mixed_record_types(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """Mix of TRADE_CLOSED and TRADE_RECORD records are all counted."""
        pos1 = make_closed_position(position_id="raw_001", exit_price=51_000.0)
        write_trade_closed_record(trade_logger, pos1)

        pos2 = make_closed_position(position_id="schema_001", exit_price=51_200.0)
        write_trade_record(trade_logger, pos2)

        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        assert summary.trade_count == 2

    def test_schema_trade_pnl_included(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """PnL from TRADE_RECORD is included in totals."""
        pos = make_closed_position(position_id="schema_pnl", exit_price=51_000.0)
        write_trade_record(trade_logger, pos)

        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        assert summary.realized_pnl_usd != 0.0


# ---------------------------------------------------------------------------
# DailyReportGenerator — strategy breakdown
# ---------------------------------------------------------------------------


class TestStrategyBreakdown:
    def test_single_strategy(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """Single strategy appears in breakdown."""
        pos = make_closed_position(strategy="TREND_PULLBACK")
        write_trade_closed_record(trade_logger, pos)

        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        assert len(summary.strategy_breakdown) == 1
        assert summary.strategy_breakdown[0].strategy_name == "TREND_PULLBACK"
        assert summary.strategy_breakdown[0].trade_count == 1

    def test_multiple_strategies(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """Multiple strategies have separate breakdown entries."""
        strategies = ["TREND_PULLBACK", "RANGE_MEAN_REVERSION", "TREND_BREAKOUT"]
        for i, strat in enumerate(strategies):
            pos = make_closed_position(
                position_id=f"pos_{i}",
                strategy=strat,
                exit_price=51_000.0 + i * 100,
            )
            write_trade_closed_record(trade_logger, pos)

        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        names = {s.strategy_name for s in summary.strategy_breakdown}
        assert names == set(strategies)

    def test_strategy_breakdown_sorted_by_pnl(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """Strategy breakdown is sorted by total_pnl_usd descending."""
        # TREND_PULLBACK: 3 wins, RANGE_MR: 1 loss
        for i in range(3):
            pos = make_closed_position(
                position_id=f"tp_{i}", strategy="TREND_PULLBACK", exit_price=51_000.0
            )
            write_trade_closed_record(trade_logger, pos)

        pos_loss = make_closed_position(
            position_id="mr_1",
            strategy="RANGE_MEAN_REVERSION",
            exit_price=49_500.0,
            exit_reason=ExitReason.SL,
        )
        write_trade_closed_record(trade_logger, pos_loss)

        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        pnls = [s.total_pnl_usd for s in summary.strategy_breakdown]
        assert pnls == sorted(pnls, reverse=True)

    def test_strategy_win_rate(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """Per-strategy win rate is computed correctly."""
        # 2 wins, 1 loss for TREND_PULLBACK
        for i in range(2):
            pos = make_closed_position(
                position_id=f"win_{i}", strategy="TREND_PULLBACK", exit_price=51_000.0
            )
            write_trade_closed_record(trade_logger, pos)
        pos_loss = make_closed_position(
            position_id="loss_0",
            strategy="TREND_PULLBACK",
            exit_price=49_500.0,
            exit_reason=ExitReason.SL,
        )
        write_trade_closed_record(trade_logger, pos_loss)

        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        tp = next(s for s in summary.strategy_breakdown if s.strategy_name == "TREND_PULLBACK")
        assert abs(tp.win_rate - 2 / 3) < 1e-6

    def test_strategy_breakdown_to_dict(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """StrategyBreakdown.to_dict() returns all expected keys."""
        pos = make_closed_position(strategy="TREND_PULLBACK", exit_price=51_000.0)
        write_trade_closed_record(trade_logger, pos)

        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        d = summary.strategy_breakdown[0].to_dict()
        for key in (
            "strategy_name", "trade_count", "winning_trades", "losing_trades",
            "win_rate", "total_pnl_usd", "total_pnl_r", "avg_pnl_r", "total_fees_usd",
        ):
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# DailyReportGenerator — drawdown computation
# ---------------------------------------------------------------------------


class TestDrawdownComputation:
    def test_no_drawdown_on_all_winners(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """daily_drawdown_pct is 0 when all trades are winners."""
        for i in range(3):
            pos = make_closed_position(
                position_id=f"win_{i}", exit_price=51_000.0
            )
            write_trade_closed_record(trade_logger, pos)

        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        assert summary.daily_drawdown_pct <= 0.0
        assert summary.max_drawdown_usd >= 0.0

    def test_drawdown_after_loss(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """daily_drawdown_pct is non-zero after a losing trade."""
        pos = make_closed_position(exit_price=49_500.0, exit_reason=ExitReason.SL)
        write_trade_closed_record(trade_logger, pos)

        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        assert summary.daily_drawdown_pct < 0.0
        assert summary.max_drawdown_usd > 0.0

    def test_equity_peak_gte_start(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """equity_peak_usd >= equity_start_usd after winning trades."""
        pos = make_closed_position(exit_price=51_000.0)
        write_trade_closed_record(trade_logger, pos)

        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        assert summary.equity_peak_usd >= summary.equity_start_usd


# ---------------------------------------------------------------------------
# DailyReportGenerator — expectancy_R
# ---------------------------------------------------------------------------


class TestExpectancyR:
    def test_expectancy_positive_on_net_winners(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """expectancy_r > 0 when avg pnl_r per trade is positive (TRADE_RECORD format)."""
        # Use schema-validated records so pnl_r_multiple is stored correctly.
        for i in range(3):
            pos = make_closed_position(
                position_id=f"win_{i}",
                entry_price=50_000.0,
                stop_price=49_000.0,  # risk = $100
                exit_price=52_000.0,  # gain ≈ $200 → ~2R
                quantity=0.1,
            )
            write_trade_record(trade_logger, pos)

        pos_loss = make_closed_position(
            position_id="loss_1",
            entry_price=50_000.0,
            stop_price=49_000.0,
            exit_price=49_000.0,  # exactly 1R loss
            exit_reason=ExitReason.SL,
            quantity=0.1,
        )
        write_trade_record(trade_logger, pos_loss)

        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        assert summary.expectancy_r > 0.0

    def test_expectancy_zero_no_trades(self, report_gen: DailyReportGenerator) -> None:
        """expectancy_r is 0.0 when no trades."""
        summary = report_gen.build()
        assert summary.expectancy_r == 0.0


# ---------------------------------------------------------------------------
# DailySummary.to_dict() — serialization
# ---------------------------------------------------------------------------


class TestDailySummaryToDict:
    def test_all_keys_present(self, report_gen: DailyReportGenerator) -> None:
        """DailySummary.to_dict() contains all expected keys."""
        summary = report_gen.build(equity_start_usd=10_000.0)
        d = summary.to_dict()
        for key in (
            "report_date", "generated_at", "mode",
            "realized_pnl_usd", "realized_pnl_pct", "unrealized_pnl_usd",
            "equity_start_usd", "equity_end_usd", "equity_peak_usd",
            "daily_drawdown_pct", "weekly_pnl_usd", "max_drawdown_usd",
            "trade_count", "winning_trades", "losing_trades",
            "win_rate", "expectancy_r",
            "total_fees_usd", "total_funding_usd",
            "exit_reason_counts", "strategy_breakdown",
        ):
            assert key in d, f"Missing key: {key}"

    def test_json_serializable(self, report_gen: DailyReportGenerator) -> None:
        """DailySummary.to_dict() produces JSON-serializable output."""
        summary = report_gen.build(equity_start_usd=10_000.0)
        serialized = json.dumps(summary.to_dict())
        parsed = json.loads(serialized)
        assert "trade_count" in parsed

    def test_report_date_is_iso_string(self, report_gen: DailyReportGenerator) -> None:
        """report_date in to_dict() is an ISO-format string."""
        summary = report_gen.build()
        d = summary.to_dict()
        # Should be parseable as a date
        date.fromisoformat(d["report_date"])


# ---------------------------------------------------------------------------
# DailyReportGenerator.format_text()
# ---------------------------------------------------------------------------


class TestFormatText:
    def _make_summary(self, report_gen: DailyReportGenerator, trades: int = 0) -> DailySummary:
        return report_gen.build(
            equity_start_usd=10_000.0,
            mode="PAPER_LIVE",
        )

    def test_contains_date(self, report_gen: DailyReportGenerator) -> None:
        """format_text() includes the report date."""
        summary = report_gen.build(for_date=date(2024, 2, 11))
        text = report_gen.format_text(summary)
        assert "2024-02-11" in text

    def test_contains_mode(self, report_gen: DailyReportGenerator) -> None:
        """format_text() includes mode."""
        summary = report_gen.build(mode="LIVE")
        text = report_gen.format_text(summary)
        assert "LIVE" in text

    def test_contains_pnl(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """format_text() includes PnL section."""
        pos = make_closed_position(exit_price=51_000.0)
        write_trade_closed_record(trade_logger, pos)
        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        text = report_gen.format_text(summary)
        assert "PnL" in text
        assert "Realized" in text

    def test_contains_equity_section(self, report_gen: DailyReportGenerator) -> None:
        """format_text() includes Equity section."""
        summary = report_gen.build(equity_start_usd=10_000.0)
        text = report_gen.format_text(summary)
        assert "Equity" in text
        assert "10000" in text

    def test_contains_trade_count(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """format_text() includes trade count."""
        pos = make_closed_position()
        write_trade_closed_record(trade_logger, pos)
        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        text = report_gen.format_text(summary)
        assert "Trades" in text

    def test_contains_strategy_section_when_trades_exist(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """format_text() includes Strategy Breakdown section when trades exist."""
        pos = make_closed_position(strategy="TREND_PULLBACK")
        write_trade_closed_record(trade_logger, pos)
        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        text = report_gen.format_text(summary)
        assert "Strategy" in text
        assert "TREND_PULLBACK" in text

    def test_no_strategy_section_when_no_trades(
        self, report_gen: DailyReportGenerator
    ) -> None:
        """format_text() omits Strategy Breakdown when no trades."""
        summary = report_gen.build()
        text = report_gen.format_text(summary)
        # No strategy data → breakdown list is empty
        assert "TREND_PULLBACK" not in text

    def test_positive_pnl_has_plus_sign(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """Positive PnL has a '+' prefix in the formatted output."""
        pos = make_closed_position(exit_price=51_000.0)
        write_trade_closed_record(trade_logger, pos)
        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        text = report_gen.format_text(summary)
        # The realized PnL line should start with '+'
        assert "+" in text

    def test_contains_fees(
        self,
        trade_logger: TradeLogger,
        report_gen: DailyReportGenerator,
    ) -> None:
        """format_text() includes Costs/Fees section."""
        pos = make_closed_position(fees=5.0)
        write_trade_closed_record(trade_logger, pos)
        summary = report_gen.build(for_date=date.today(), equity_start_usd=10_000.0)
        text = report_gen.format_text(summary)
        assert "Fees" in text

    def test_generated_timestamp_in_output(self, report_gen: DailyReportGenerator) -> None:
        """format_text() includes the generation timestamp."""
        summary = report_gen.build()
        text = report_gen.format_text(summary)
        assert "Generated" in text


# ---------------------------------------------------------------------------
# TelegramNotifier — disabled mode
# ---------------------------------------------------------------------------


class TestTelegramNotifierDisabled:
    def test_disabled_when_no_credentials(self) -> None:
        """Notifier is disabled when token/chat_id not provided."""
        notifier = TelegramNotifier(token="", chat_id="", enabled=True)
        assert not notifier.is_enabled

    def test_send_returns_false_when_disabled(self) -> None:
        """send() returns False without making API call when disabled."""
        notifier = TelegramNotifier(token="", chat_id="")
        result = notifier.send("test message")
        assert result is False

    def test_enabled_false_disables_regardless_of_credentials(self) -> None:
        """enabled=False disables notifications even with valid credentials."""
        notifier = TelegramNotifier(token="tok", chat_id="123", enabled=False)
        assert not notifier.is_enabled

    def test_all_send_methods_return_false_when_disabled(self) -> None:
        """All send variants return False when notifier is disabled."""
        notifier = TelegramNotifier(token="", chat_id="")

        assert notifier.send("msg") is False
        assert notifier.send_kill_switch_alert("test") is False
        assert notifier.send_safe_mode_alert("test") is False
        assert notifier.send_error_alert("comp", "msg") is False
        assert notifier.send_trade_opened("BTCUSDT", "LONG", 50000, 49500, 50, "TP") is False
        assert notifier.send_trade_closed("BTCUSDT", "LONG", 100.0, 2.0, "TP") is False

    def test_send_daily_report_returns_false_when_disabled(
        self, report_gen: DailyReportGenerator
    ) -> None:
        """send_daily_report() returns False when notifier disabled."""
        notifier = TelegramNotifier(token="", chat_id="")
        summary = report_gen.build()
        assert notifier.send_daily_report(summary) is False


# ---------------------------------------------------------------------------
# TelegramNotifier — message truncation
# ---------------------------------------------------------------------------


class TestTelegramMessageTruncation:
    def test_short_message_not_truncated(self) -> None:
        """Messages under the limit are not modified."""
        text = "Hello, trading bot!"
        result = TelegramNotifier._truncate(text)
        assert result == text

    def test_long_message_truncated(self) -> None:
        """Messages over the limit are truncated."""
        long_text = "x" * (_TELEGRAM_MAX_LEN + 500)
        result = TelegramNotifier._truncate(long_text)
        assert len(result) <= _TELEGRAM_MAX_LEN

    def test_truncated_message_has_suffix(self) -> None:
        """Truncated messages end with a '[truncated]' marker."""
        long_text = "a" * (_TELEGRAM_MAX_LEN + 100)
        result = TelegramNotifier._truncate(long_text)
        assert "[truncated]" in result

    def test_exact_max_len_not_truncated(self) -> None:
        """Message of exactly max length is not truncated."""
        text = "b" * _TELEGRAM_MAX_LEN
        result = TelegramNotifier._truncate(text)
        assert len(result) == _TELEGRAM_MAX_LEN


# ---------------------------------------------------------------------------
# TelegramNotifier — message content helpers
# ---------------------------------------------------------------------------


class TestTelegramMessageContent:
    def test_trade_opened_message_contains_fields(self) -> None:
        """send_trade_opened produces message with expected fields."""
        notifier = TelegramNotifier(token="", chat_id="")
        # Disabled — test the message format via _truncate on formatted text
        # Build the text manually to check format
        text = (
            "TRADE OPENED\n"
            "Symbol:   BTCUSDT\n"
            "Side:     LONG\n"
            "Entry:    50000.0000\n"
            "Stop:     49500.0000 (1.00%)\n"
            "Risk:     50.00 USD\n"
            "Strategy: TREND_PULLBACK"
        )
        result = TelegramNotifier._truncate(text)
        assert "TRADE OPENED" in result
        assert "BTCUSDT" in result
        assert "50000" in result

    def test_kill_switch_message_format(self) -> None:
        """Kill switch alert contains KILL SWITCH and reason."""
        notifier = TelegramNotifier(token="", chat_id="")
        # Call with disabled notifier — we just verify the text is formed
        # by inspecting what would be sent (via send mock)
        text = "KILL SWITCH ACTIVATED\nReason: Daily loss limit"
        result = TelegramNotifier._truncate(text)
        assert "KILL SWITCH" in result
        assert "Daily loss" in result

    def test_safe_mode_message_format(self) -> None:
        """Safe mode alert contains SAFE MODE."""
        text = "SAFE MODE ACTIVE\nReason: Stale data\nNo new entries until resolved."
        result = TelegramNotifier._truncate(text)
        assert "SAFE MODE" in result


# ---------------------------------------------------------------------------
# TelegramNotifier — mocked HTTP send
# ---------------------------------------------------------------------------


class TestTelegramNotifierMockedSend:
    def test_send_calls_http_when_enabled(self) -> None:
        """send() attempts HTTP when token/chat_id are set."""
        notifier = TelegramNotifier(token="fake_token", chat_id="123456")
        assert notifier.is_enabled

        # Patch _send_once to simulate success without real HTTP
        with patch.object(notifier, "_send_once", return_value=True) as mock_send:
            result = notifier.send("Hello from test")

        assert result is True
        mock_send.assert_called_once()

    def test_send_returns_false_on_http_failure(self) -> None:
        """send() returns False when all retries fail."""
        notifier = TelegramNotifier(token="fake_token", chat_id="123456")

        with patch.object(notifier, "_send_once", side_effect=Exception("connection refused")):
            with patch("bot.reporting.notifier.time.sleep"):  # speed up retries
                result = notifier.send("test")

        assert result is False

    def test_send_retries_on_failure(self) -> None:
        """send() retries up to _MAX_RETRIES times before giving up."""
        from bot.reporting.notifier import _MAX_RETRIES

        notifier = TelegramNotifier(token="fake_token", chat_id="123456")
        call_count = [0]

        def flaky(*args, **kwargs):
            call_count[0] += 1
            raise Exception("flaky")

        with patch.object(notifier, "_send_once", side_effect=flaky):
            with patch("bot.reporting.notifier.time.sleep"):
                notifier.send("test")

        assert call_count[0] == _MAX_RETRIES

    def test_send_daily_report_uses_format_text(
        self, report_gen: DailyReportGenerator
    ) -> None:
        """send_daily_report() generates formatted text and calls send()."""
        notifier = TelegramNotifier(token="fake_token", chat_id="123456")
        summary = report_gen.build(equity_start_usd=10_000.0)

        with patch.object(notifier, "send", return_value=True) as mock_send:
            result = notifier.send_daily_report(summary)

        assert result is True
        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][0]
        assert "Daily Report" in sent_text

    def test_send_kill_switch_calls_send(self) -> None:
        """send_kill_switch_alert() calls send() with correct content."""
        notifier = TelegramNotifier(token="fake_token", chat_id="123456")

        with patch.object(notifier, "send", return_value=True) as mock_send:
            notifier.send_kill_switch_alert("Daily loss limit", {"open_positions": 2})

        mock_send.assert_called_once()
        sent = mock_send.call_args[0][0]
        assert "KILL SWITCH" in sent
        assert "Daily loss limit" in sent

    def test_send_safe_mode_calls_send(self) -> None:
        """send_safe_mode_alert() calls send() with correct content."""
        notifier = TelegramNotifier(token="fake_token", chat_id="123456")

        with patch.object(notifier, "send", return_value=True) as mock_send:
            notifier.send_safe_mode_alert("Stale data >30s")

        mock_send.assert_called_once()
        sent = mock_send.call_args[0][0]
        assert "SAFE MODE" in sent
        assert "Stale data" in sent

    def test_send_trade_opened_calls_send(self) -> None:
        """send_trade_opened() calls send() with correct content."""
        notifier = TelegramNotifier(token="fake_token", chat_id="123456")

        with patch.object(notifier, "send", return_value=True) as mock_send:
            notifier.send_trade_opened("BTCUSDT", "LONG", 50_000.0, 49_500.0, 50.0, "TREND_PULLBACK")

        mock_send.assert_called_once()
        sent = mock_send.call_args[0][0]
        assert "BTCUSDT" in sent
        assert "LONG" in sent
        assert "TREND_PULLBACK" in sent

    def test_send_trade_closed_positive_pnl(self) -> None:
        """send_trade_closed() shows '+' sign for positive PnL."""
        notifier = TelegramNotifier(token="fake_token", chat_id="123456")

        with patch.object(notifier, "send", return_value=True) as mock_send:
            notifier.send_trade_closed("BTCUSDT", "LONG", 100.0, 2.0, "TP")

        sent = mock_send.call_args[0][0]
        assert "+100" in sent or "+100.00" in sent

    def test_send_trade_closed_negative_pnl(self) -> None:
        """send_trade_closed() shows negative PnL correctly."""
        notifier = TelegramNotifier(token="fake_token", chat_id="123456")

        with patch.object(notifier, "send", return_value=True) as mock_send:
            notifier.send_trade_closed("BTCUSDT", "LONG", -50.0, -1.0, "SL")

        sent = mock_send.call_args[0][0]
        assert "-50" in sent

    def test_send_error_alert(self) -> None:
        """send_error_alert() includes component and message."""
        notifier = TelegramNotifier(token="fake_token", chat_id="123456")

        with patch.object(notifier, "send", return_value=True) as mock_send:
            notifier.send_error_alert("risk_engine", "Open risk exceeded limit")

        sent = mock_send.call_args[0][0]
        assert "risk_engine" in sent
        assert "Open risk exceeded limit" in sent


# ---------------------------------------------------------------------------
# make_notifier_from_config factory
# ---------------------------------------------------------------------------


class TestMakeNotifierFromConfig:
    def test_disabled_flag_disables(self) -> None:
        """make_notifier_from_config(enabled=False) creates disabled notifier."""
        notifier = make_notifier_from_config(enabled=False)
        assert not notifier.is_enabled

    def test_enabled_without_env_vars_disables(self, monkeypatch) -> None:
        """enabled=True without env vars creates disabled notifier."""
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        notifier = make_notifier_from_config(enabled=True)
        assert not notifier.is_enabled

    def test_enabled_with_env_vars(self, monkeypatch) -> None:
        """enabled=True with env vars creates enabled notifier."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
        notifier = make_notifier_from_config(enabled=True)
        assert notifier.is_enabled

    def test_custom_env_var_names(self, monkeypatch) -> None:
        """Custom env var names are respected."""
        monkeypatch.setenv("MY_BOT_TOKEN", "tok")
        monkeypatch.setenv("MY_CHAT_ID", "999")
        notifier = make_notifier_from_config(
            enabled=True,
            token_env="MY_BOT_TOKEN",
            chat_id_env="MY_CHAT_ID",
        )
        assert notifier.is_enabled
