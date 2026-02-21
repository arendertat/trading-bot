"""Tests for report JSON extensions."""

from datetime import datetime, timezone

from bot.backtest.account import BacktestTrade
from bot.backtest.engine import BacktestResult
from bot.backtest.reporter import build_report


def test_report_extensions_present():
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trade = BacktestTrade(
        trade_id="T1",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        exit_price=101.0,
        quantity=1.0,
        notional_usd=100.0,
        entry_time=now,
        exit_time=now,
        exit_reason="TP",
        pnl_usd=1.0,
        pnl_r=1.0,
        risk_usd=1.0,
        fees_usd=0.1,
        slippage_usd=0.0,
        funding_usd=0.0,
        gross_pnl_usd=1.1,
        net_pnl_usd=1.0,
        strategy="TrendPullback",
        regime="TREND",
        stop_price=99.0,
        tp_price=102.0,
    )

    result = BacktestResult(
        trades=[trade],
        equity_curve=[(0, 10000.0), (1, 10001.0)],
        initial_equity=10000.0,
        final_equity=10001.0,
        symbols=["BTCUSDT"],
        start=now,
        end=now,
        total_bars=10,
        total_signals=1,
        total_entries=1,
        rejected_risk=0,
        rejected_spread=0,
        reject_reason_counts={"COST_GATE": 1},
        data_quality={},
        regime_bar_counts={"TREND": 10},
        time_filter_min_samples=20,
        time_filter_avg_r_threshold=-0.2,
    )

    report = build_report(result)
    assert "bars" in report.regime_distribution
    assert "trades" in report.regime_distribution
    assert "COST_GATE" in report.reject_reason_distribution
    assert "total_gross_pnl_usd" in report.cost_breakdown
