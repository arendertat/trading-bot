"""
Backtest performance reporter.

Computes summary statistics from BacktestResult:
- Net return, CAGR
- Win rate, profit factor, avg R
- Max drawdown
- Sharpe ratio (annualised, risk-free = 0)
- Per-strategy breakdown
- Per-regime breakdown
- Exit analysis (SL/TP/TRAIL/EOD with avg R per exit type)
- MAE/MFE analysis (entry quality)
- Time analysis (day-of-week, hour-of-day)
- Strategy × Regime matrix
- Consecutive loss streaks
- Kill switch block count
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from bot.backtest.account import BacktestTrade
from bot.backtest.engine import BacktestResult


@dataclass
class StrategyStats:
    """Performance stats for a single strategy."""
    name: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_usd: float = 0.0
    total_pnl_r: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0


@dataclass
class ExitStats:
    """Stats for a specific exit type."""
    count: int = 0
    avg_r: float = 0.0
    avg_pnl_usd: float = 0.0
    pct_of_trades: float = 0.0
    avg_mae_usd: float = 0.0
    avg_mfe_usd: float = 0.0
    avg_mae_r: float = 0.0
    avg_mfe_r: float = 0.0
    avg_mfe_capture_pct: float = 0.0


@dataclass
class MAEMFEStats:
    """Max Adverse / Favourable Excursion analysis."""
    avg_mae_usd: float = 0.0       # avg worst unrealised loss during trade
    avg_mfe_usd: float = 0.0       # avg best unrealised profit during trade
    avg_mae_r: float = 0.0         # MAE expressed as R-multiple
    avg_mfe_r: float = 0.0         # MFE expressed as R-multiple
    avg_mae_mfe_ratio: float = 0.0 # |MAE| / MFE (entry quality)
    # For winning trades: how much of MFE did they actually capture?
    avg_mfe_capture_pct: float = 0.0   # pnl_usd / mfe_usd (only for MFE > 0)
    # For losing trades: how deep did they go before stopping out?
    avg_sl_efficiency: float = 0.0     # |mae_usd| / risk_usd (should be ~1.0 for tight stops)


@dataclass
class TimeStats:
    """PnL breakdown by time periods."""
    by_weekday: Dict[str, float] = field(default_factory=dict)    # "Mon" → avg_r
    by_hour: Dict[int, float] = field(default_factory=dict)       # 0-23 → avg_r
    by_weekday_trades: Dict[str, int] = field(default_factory=dict)
    by_hour_trades: Dict[int, int] = field(default_factory=dict)


@dataclass
class BacktestReport:
    """Full backtest performance report."""
    # Overview
    initial_equity: float
    final_equity: float
    net_return_pct: float
    cagr_pct: float

    # Trade summary
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # PnL metrics
    total_pnl_usd: float
    total_fees_usd: float
    avg_pnl_r: float
    avg_win_r: float
    avg_loss_r: float
    best_trade_r: float
    worst_trade_r: float
    profit_factor: float
    total_gross_pnl_usd: float
    total_slippage_usd: float
    total_funding_usd: float
    total_net_pnl_usd: float
    avg_fee_per_trade: float
    avg_slippage_per_trade: float

    # Risk metrics
    max_drawdown_pct: float
    max_drawdown_usd: float
    sharpe_ratio: float
    calmar_ratio: float

    # Exit analysis (counts)
    sl_exits: int
    tp_exits: int
    trail_exits: int
    eod_exits: int

    # Detailed exit stats
    exit_stats: Dict[str, ExitStats]

    # MAE/MFE analysis
    mae_mfe: MAEMFEStats

    # Entry quality
    early_reversal_pct: float
    early_reversal_count: int
    early_reversal_threshold_r: float
    early_window_bars: int

    # Time analysis
    time_stats: TimeStats
    time_filter_suggestions: Dict[str, List]

    # Strategy × Regime matrix  regime → strategy → {trades, win_rate, avg_r, pnl_usd}
    strategy_regime_matrix: Dict[str, Dict[str, Dict[str, float]]]

    # Consecutive loss analysis
    max_consecutive_losses: int
    max_consecutive_wins: int
    avg_consecutive_losses: float
    worst_loss_streak_pnl_usd: float   # total loss during worst streak

    # Kill switch
    kill_switch_blocks: int   # estimated from trade gaps > 7 days

    # Per-strategy breakdown
    by_strategy: Dict[str, StrategyStats]

    # Per-regime breakdown
    by_regime: Dict[str, Dict[str, float]]
    regime_distribution: Dict[str, Dict[str, dict]]
    reject_reason_distribution: Dict[str, int]
    cost_breakdown: Dict[str, float]
    monthly_summary: List[Dict[str, float]]
    drawdown_episodes: List[Dict[str, float]]

    # Params
    symbols: List[str]
    start: str
    end: str
    total_bars: int
    total_signals: int
    total_entries: int
    rejected_risk: int
    spread_gate_rejects: int

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        sep = "─" * 64
        print(f"\n{sep}")
        print(f"  BACKTEST REPORT")
        print(f"  {self.start}  →  {self.end}")
        print(f"  Symbols: {', '.join(self.symbols)}")
        print(sep)
        print(f"  Equity:        {self.initial_equity:>12,.2f} → {self.final_equity:>12,.2f} USD")
        print(f"  Net return:    {self.net_return_pct:>+11.1f}%")
        print(f"  CAGR:          {self.cagr_pct:>+11.1f}%")
        print(f"  Max drawdown:  {self.max_drawdown_pct:>+11.1f}%  ({self.max_drawdown_usd:+.2f} USD)")
        print(f"  Sharpe ratio:  {self.sharpe_ratio:>11.2f}")
        print(f"  Calmar ratio:  {self.calmar_ratio:>11.2f}")
        print(sep)
        print(f"  Total trades:  {self.total_trades:>11}")
        print(f"  Win rate:      {self.win_rate:>11.1%}")
        print(f"  Profit factor: {self.profit_factor:>11.2f}")
        print(f"  Avg PnL (R):   {self.avg_pnl_r:>+11.3f}R")
        print(f"  Avg win (R):   {self.avg_win_r:>+11.3f}R")
        print(f"  Avg loss (R):  {self.avg_loss_r:>+11.3f}R")
        print(f"  Best trade:    {self.best_trade_r:>+11.3f}R")
        print(f"  Worst trade:   {self.worst_trade_r:>+11.3f}R")
        print(f"  Total fees:    {self.total_fees_usd:>11.2f} USD")
        print(f"  Total gross:   {self.total_gross_pnl_usd:>11.2f} USD")
        print(f"  Total slippage:{self.total_slippage_usd:>11.2f} USD")
        print(f"  Total funding: {self.total_funding_usd:>11.2f} USD")
        print(f"  Total net:     {self.total_net_pnl_usd:>11.2f} USD")
        print(sep)

        # ── Exit analysis ──────────────────────────────────────────────
        print(f"  EXIT ANALYSIS:")
        for reason in ["SL", "TP", "TRAIL", "EOD"]:
            es = self.exit_stats.get(reason)
            if es and es.count > 0:
                print(
                    f"    {reason:<6}  count={es.count:>4}  ({es.pct_of_trades:>5.1%})  "
                    f"avg_R={es.avg_r:>+6.3f}R  avg_PnL={es.avg_pnl_usd:>+8.2f} USD  "
                    f"MAE={es.avg_mae_r:>+6.2f}R  MFE={es.avg_mfe_r:>+6.2f}R"
                )
        print(sep)

        # ── MAE/MFE ───────────────────────────────────────────────────
        m = self.mae_mfe
        print(f"  ENTRY QUALITY (MAE/MFE):")
        print(f"    Avg MAE:          {m.avg_mae_usd:>+10.2f} USD  ({m.avg_mae_r:>+6.3f}R)")
        print(f"    Avg MFE:          {m.avg_mfe_usd:>+10.2f} USD  ({m.avg_mfe_r:>+6.3f}R)")
        if m.avg_mae_mfe_ratio > 0:
            print(f"    MAE/MFE ratio:    {m.avg_mae_mfe_ratio:>10.2f}  ← lower is better")
        print(f"    MFE capture (win):{m.avg_mfe_capture_pct:>10.1%}  ← how much profit was taken")
        print(f"    SL efficiency:    {m.avg_sl_efficiency:>10.2f}x  ← 1.0 = stop hit at exactly risk")
        print(sep)

        # ── Consecutive streaks ────────────────────────────────────────
        print(f"  STREAKS:")
        print(f"    Max consecutive losses:  {self.max_consecutive_losses:>4}")
        print(f"    Max consecutive wins:    {self.max_consecutive_wins:>4}")
        print(f"    Avg loss streak len:     {self.avg_consecutive_losses:>7.1f}")
        print(f"    Worst streak PnL:        {self.worst_loss_streak_pnl_usd:>+10.2f} USD")
        if self.kill_switch_blocks > 0:
            print(f"    Kill switch blocks:      {self.kill_switch_blocks:>4}  (estimated)")
        print(sep)

        # ── Time analysis ──────────────────────────────────────────────
        if self.time_stats.by_weekday:
            print(f"  BY DAY OF WEEK (avg R):")
            days_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            for day in days_order:
                avg_r = self.time_stats.by_weekday.get(day, 0.0)
                n = self.time_stats.by_weekday_trades.get(day, 0)
                if n > 0:
                    bar = "█" * min(int(abs(avg_r) * 20), 20)
                    sign = "+" if avg_r >= 0 else "-"
                    print(f"    {day}  trades={n:>4}  avg_R={avg_r:>+6.3f}  {sign}{bar}")
            print(sep)

        # ── Strategy × Regime matrix ───────────────────────────────────
        if self.strategy_regime_matrix:
            print(f"  STRATEGY × REGIME MATRIX:")
            all_regimes = sorted({r for strat in self.strategy_regime_matrix.values() for r in strat})
            # Header
            header = f"    {'Strategy':<28}"
            for r in all_regimes:
                header += f"  {r[:10]:<10}"
            print(header)
            for strat, regime_data in self.strategy_regime_matrix.items():
                row = f"    {strat:<28}"
                for r in all_regimes:
                    d = regime_data.get(r)
                    if d and d.get("trades", 0) > 0:
                        row += f"  {d['avg_r']:>+5.2f}R/{int(d['trades']):>3}t"
                    else:
                        row += f"  {'—':>10}"
                print(row)
            print(sep)

        # ── Per-strategy breakdown ─────────────────────────────────────
        if self.by_strategy:
            print(f"  BY STRATEGY:")
            for name, s in self.by_strategy.items():
                print(
                    f"    {name:<30}  trades={s.trades:>4}  "
                    f"win={s.win_rate:.0%}  PF={s.profit_factor:.2f}  "
                    f"avgWin={s.avg_win_r:>+.2f}R  avgLoss={s.avg_loss_r:>+.2f}R  "
                    f"PnL={s.total_pnl_usd:>+.2f}$"
                )
            print(sep)

        # ── Per-regime breakdown ───────────────────────────────────────
        if self.by_regime:
            print(f"  BY REGIME:")
            for regime, stats in self.by_regime.items():
                print(
                    f"    {regime:<15}  trades={int(stats.get('trades', 0)):>4}  "
                    f"win={stats.get('win_rate', 0):.0%}  "
                    f"avg_R={stats.get('avg_r', 0):+.2f}  "
                    f"PnL={stats.get('total_pnl_usd', 0):>+.2f}$"
                )
            print(sep)

        print(f"  Signals: {self.total_signals}  |  Entries: {self.total_entries}  |  "
              f"Rejected: {self.rejected_risk}  |  Spread gate: {self.spread_gate_rejects}  "
              f"|  Bars: {self.total_bars:,}")
        if self.early_reversal_count > 0:
            print(
                f"  Early reversals: {self.early_reversal_count} "
                f"({self.early_reversal_pct:.1%})  "
                f"threshold={self.early_reversal_threshold_r:+.2f}R "
                f"window={self.early_window_bars} bars"
            )
        print(sep)
        print()


def build_report(result: BacktestResult) -> BacktestReport:
    """
    Compute all performance metrics from a BacktestResult.

    Args:
        result: Completed backtest result.

    Returns:
        BacktestReport with all metrics populated.
    """
    trades = result.trades
    equity_curve = result.equity_curve

    # ── Basic returns ─────────────────────────────────────────────────
    initial = result.initial_equity
    final = result.final_equity
    net_return_pct = (final / initial - 1.0) * 100.0 if initial > 0 else 0.0

    days = (result.end - result.start).total_seconds() / 86400
    years = max(days / 365.25, 1e-6)
    cagr_pct = ((final / initial) ** (1.0 / years) - 1.0) * 100.0 if initial > 0 else 0.0

    # ── Trade stats ───────────────────────────────────────────────────
    total_trades = len(trades)
    winning = [t for t in trades if t.pnl_usd > 0]
    losing = [t for t in trades if t.pnl_usd <= 0]
    win_rate = len(winning) / total_trades if total_trades > 0 else 0.0

    total_pnl = sum(t.pnl_usd for t in trades)
    total_fees = sum(t.fees_usd for t in trades)
    total_gross = sum(getattr(t, "gross_pnl_usd", 0.0) for t in trades)
    total_slippage = sum(getattr(t, "slippage_usd", 0.0) for t in trades)
    total_funding = sum(getattr(t, "funding_usd", 0.0) for t in trades)
    total_net = sum(getattr(t, "net_pnl_usd", t.pnl_usd) for t in trades)
    total_pnl_r = sum(t.pnl_r for t in trades)
    avg_pnl_r = total_pnl_r / total_trades if total_trades > 0 else 0.0

    avg_win_r = sum(t.pnl_r for t in winning) / len(winning) if winning else 0.0
    avg_loss_r = sum(t.pnl_r for t in losing) / len(losing) if losing else 0.0

    gross_wins = sum(t.pnl_usd for t in winning) if winning else 0.0
    gross_losses = abs(sum(t.pnl_usd for t in losing)) if losing else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    best_r = max((t.pnl_r for t in trades), default=0.0)
    worst_r = min((t.pnl_r for t in trades), default=0.0)

    # ── Drawdown ──────────────────────────────────────────────────────
    max_dd_pct, max_dd_usd = _compute_max_drawdown(equity_curve)

    # ── Sharpe & Calmar ───────────────────────────────────────────────
    sharpe = _compute_sharpe(equity_curve)
    calmar = (cagr_pct / abs(max_dd_pct)) if max_dd_pct < 0 else 0.0

    # ── Exit reasons (basic counts) ───────────────────────────────────
    sl_exits = sum(1 for t in trades if t.exit_reason == "SL")
    tp_exits = sum(1 for t in trades if t.exit_reason == "TP")
    trail_exits = sum(1 for t in trades if t.exit_reason == "TRAIL")
    eod_exits = sum(1 for t in trades if t.exit_reason == "EOD")

    # ── Detailed exit stats ────────────────────────────────────────────
    exit_stats: Dict[str, ExitStats] = {}
    for reason in ["SL", "TP", "TRAIL", "EOD"]:
        group = [t for t in trades if t.exit_reason == reason]
        if group:
            mfe_vals = [t.max_favourable_excursion for t in group]
            mae_vals = [t.max_adverse_excursion for t in group]
            mfe_r_vals = [
                (t.max_favourable_excursion / t.risk_usd) if t.risk_usd > 0 else 0.0
                for t in group
            ]
            mae_r_vals = [
                (t.max_adverse_excursion / t.risk_usd) if t.risk_usd > 0 else 0.0
                for t in group
            ]
            mfe_capture_vals = [
                (t.pnl_usd / t.max_favourable_excursion) if t.max_favourable_excursion > 0 else 0.0
                for t in group
            ]
            exit_stats[reason] = ExitStats(
                count=len(group),
                avg_r=sum(t.pnl_r for t in group) / len(group),
                avg_pnl_usd=sum(t.pnl_usd for t in group) / len(group),
                pct_of_trades=len(group) / total_trades if total_trades > 0 else 0.0,
                avg_mae_usd=sum(mae_vals) / len(mae_vals) if mae_vals else 0.0,
                avg_mfe_usd=sum(mfe_vals) / len(mfe_vals) if mfe_vals else 0.0,
                avg_mae_r=sum(mae_r_vals) / len(mae_r_vals) if mae_r_vals else 0.0,
                avg_mfe_r=sum(mfe_r_vals) / len(mfe_r_vals) if mfe_r_vals else 0.0,
                avg_mfe_capture_pct=sum(mfe_capture_vals) / len(mfe_capture_vals) if mfe_capture_vals else 0.0,
            )

    # ── MAE/MFE analysis ──────────────────────────────────────────────
    mae_mfe = _compute_mae_mfe(trades)

    # ── Time analysis ─────────────────────────────────────────────────
    time_stats = _compute_time_stats(trades)
    time_filter_suggestions = _compute_time_filter_suggestions(
        time_stats,
        result.time_filter_min_samples,
        result.time_filter_avg_r_threshold,
    )

    # ── Strategy × Regime matrix ──────────────────────────────────────
    strategy_regime_matrix = _compute_strategy_regime_matrix(trades)

    # ── Consecutive streaks ────────────────────────────────────────────
    (
        max_consec_losses, max_consec_wins,
        avg_consec_losses, worst_streak_pnl
    ) = _compute_streaks(trades)

    # ── Entry quality: early reversal rate ────────────────────────────
    early_window_bars = trades[0].early_window_bars if trades else 3
    early_threshold_r = -0.5
    early_reversal_count = sum(
        1 for t in trades if t.early_mae_r <= early_threshold_r
    )
    early_reversal_pct = early_reversal_count / total_trades if total_trades > 0 else 0.0

    # ── Kill switch estimation ─────────────────────────────────────────
    # Count gaps > 5 days between consecutive trades (indicates pause period)
    kill_switch_blocks = _estimate_kill_switch_blocks(trades, gap_days=5)

    # ── Per-strategy stats ────────────────────────────────────────────
    by_strategy: Dict[str, StrategyStats] = {}
    strat_groups: Dict[str, List[BacktestTrade]] = defaultdict(list)
    for t in trades:
        strat_groups[t.strategy].append(t)
    for name, group in strat_groups.items():
        wins_g = [t for t in group if t.pnl_usd > 0]
        loss_g = [t for t in group if t.pnl_usd <= 0]
        gw = sum(t.pnl_usd for t in wins_g) if wins_g else 0.0
        gl = abs(sum(t.pnl_usd for t in loss_g)) if loss_g else 0.0
        pf = gw / gl if gl > 0 else float("inf")
        by_strategy[name] = StrategyStats(
            name=name,
            trades=len(group),
            wins=len(wins_g),
            losses=len(loss_g),
            total_pnl_usd=sum(t.pnl_usd for t in group),
            total_pnl_r=sum(t.pnl_r for t in group),
            avg_win_r=sum(t.pnl_r for t in wins_g) / len(wins_g) if wins_g else 0.0,
            avg_loss_r=sum(t.pnl_r for t in loss_g) / len(loss_g) if loss_g else 0.0,
            profit_factor=pf,
            win_rate=len(wins_g) / len(group) if group else 0.0,
        )

    # ── Per-regime stats ──────────────────────────────────────────────
    by_regime: Dict[str, Dict[str, float]] = {}
    regime_groups: Dict[str, List[BacktestTrade]] = defaultdict(list)
    for t in trades:
        regime_groups[t.regime].append(t)
    for regime, group in regime_groups.items():
        wins_g = [t for t in group if t.pnl_usd > 0]
        avg_r = sum(t.pnl_r for t in group) / len(group) if group else 0.0
        by_regime[regime] = {
            "trades": float(len(group)),
            "win_rate": len(wins_g) / len(group) if group else 0.0,
            "avg_r": avg_r,
            "total_pnl_usd": sum(t.pnl_usd for t in group),
        }

    monthly_summary = _compute_monthly_summary(trades, equity_curve)
    regime_distribution = _compute_regime_distribution(
        result.regime_bar_counts, trades
    )
    drawdown_episodes = _compute_drawdown_episodes(equity_curve, trades)
    reject_reason_distribution = _compute_reject_reason_distribution(
        result.reject_reason_counts
    )
    cost_breakdown = _compute_cost_breakdown(
        total_gross, total_fees, total_slippage, total_funding, total_net, total_trades
    )

    return BacktestReport(
        initial_equity=initial,
        final_equity=final,
        net_return_pct=net_return_pct,
        cagr_pct=cagr_pct,
        total_trades=total_trades,
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=win_rate,
        total_pnl_usd=total_pnl,
        total_fees_usd=total_fees,
        total_gross_pnl_usd=total_gross,
        total_slippage_usd=total_slippage,
        total_funding_usd=total_funding,
        total_net_pnl_usd=total_net,
        avg_fee_per_trade=(total_fees / total_trades) if total_trades > 0 else 0.0,
        avg_slippage_per_trade=(total_slippage / total_trades) if total_trades > 0 else 0.0,
        avg_pnl_r=avg_pnl_r,
        avg_win_r=avg_win_r,
        avg_loss_r=avg_loss_r,
        best_trade_r=best_r,
        worst_trade_r=worst_r,
        profit_factor=profit_factor,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_usd=max_dd_usd,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        sl_exits=sl_exits,
        tp_exits=tp_exits,
        trail_exits=trail_exits,
        eod_exits=eod_exits,
        exit_stats=exit_stats,
        mae_mfe=mae_mfe,
        time_stats=time_stats,
        time_filter_suggestions=time_filter_suggestions,
        strategy_regime_matrix=strategy_regime_matrix,
        max_consecutive_losses=max_consec_losses,
        max_consecutive_wins=max_consec_wins,
        avg_consecutive_losses=avg_consec_losses,
        worst_loss_streak_pnl_usd=worst_streak_pnl,
        kill_switch_blocks=kill_switch_blocks,
        early_reversal_pct=early_reversal_pct,
        early_reversal_count=early_reversal_count,
        early_reversal_threshold_r=early_threshold_r,
        early_window_bars=early_window_bars,
        by_strategy=by_strategy,
        by_regime=by_regime,
        regime_distribution=regime_distribution,
        reject_reason_distribution=reject_reason_distribution,
        cost_breakdown=cost_breakdown,
        monthly_summary=monthly_summary,
        drawdown_episodes=drawdown_episodes,
        symbols=result.symbols,
        start=result.start.strftime("%Y-%m-%d"),
        end=result.end.strftime("%Y-%m-%d"),
        total_bars=result.total_bars,
        total_signals=result.total_signals,
        total_entries=result.total_entries,
        rejected_risk=result.rejected_risk,
        spread_gate_rejects=result.rejected_spread,
    )


# ── Metric helpers ───────────────────────────────────────────────────────


def _compute_max_drawdown(
    equity_curve: List[Tuple[int, float]],
) -> Tuple[float, float]:
    """Return (max_drawdown_pct, max_drawdown_usd). Both are <= 0."""
    if len(equity_curve) < 2:
        return 0.0, 0.0
    peak = equity_curve[0][1]
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    for _, eq in equity_curve:
        if eq > peak:
            peak = eq
        dd_usd = eq - peak
        dd_pct = (dd_usd / peak * 100.0) if peak > 0 else 0.0
        if dd_usd < max_dd_usd:
            max_dd_usd = dd_usd
            max_dd_pct = dd_pct
    return max_dd_pct, max_dd_usd


def _compute_sharpe(
    equity_curve: List[Tuple[int, float]],
    periods_per_year: int = 365,
) -> float:
    """Annualised Sharpe ratio from equity curve. Risk-free rate = 0."""
    if len(equity_curve) < 10:
        return 0.0

    daily: Dict[int, float] = {}
    for ts, eq in equity_curve:
        day_key = ts // 86_400_000
        daily[day_key] = eq

    sorted_days = sorted(daily.keys())
    if len(sorted_days) < 5:
        return 0.0

    daily_returns: List[float] = []
    for i in range(1, len(sorted_days)):
        prev = daily[sorted_days[i - 1]]
        curr = daily[sorted_days[i]]
        if prev > 0:
            daily_returns.append(curr / prev - 1.0)

    if len(daily_returns) < 5:
        return 0.0

    n = len(daily_returns)
    mean_r = sum(daily_returns) / n
    variance = sum((r - mean_r) ** 2 for r in daily_returns) / n
    std_r = math.sqrt(variance)

    if std_r < 1e-10:
        return 0.0

    return (mean_r / std_r) * math.sqrt(periods_per_year)


def _compute_monthly_summary(
    trades: List[BacktestTrade],
    equity_curve: List[Tuple[int, float]],
) -> List[Dict[str, float]]:
    """Compute monthly summary stats."""
    by_month: Dict[str, List[BacktestTrade]] = defaultdict(list)
    for t in trades:
        month = t.exit_time.strftime("%Y-%m")
        by_month[month].append(t)

    equity_by_month: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for ts, eq in equity_curve:
        dt = datetime.utcfromtimestamp(ts / 1000)
        month = dt.strftime("%Y-%m")
        equity_by_month[month].append((ts, eq))

    summary = []
    for month in sorted(by_month.keys()):
        group = by_month[month]
        total_trades = len(group)
        total_pnl = sum(t.pnl_usd for t in group)
        total_fees = sum(t.fees_usd for t in group)
        win_rate = sum(1 for t in group if t.pnl_usd > 0) / total_trades if total_trades else 0.0
        expectancy = sum(t.pnl_r for t in group) / total_trades if total_trades else 0.0
        eq_curve = equity_by_month.get(month, [])
        dd_pct, dd_usd = _compute_max_drawdown(eq_curve) if len(eq_curve) > 1 else (0.0, 0.0)
        summary.append({
            "month": month,
            "trades": total_trades,
            "net_pnl_usd": total_pnl,
            "fees_usd": total_fees,
            "win_rate": win_rate,
            "expectancy_r": expectancy,
            "max_drawdown_pct": dd_pct,
            "max_drawdown_usd": dd_usd,
        })
    return summary


def _compute_regime_distribution(
    regime_bar_counts: Dict[str, int],
    trades: List[BacktestTrade],
) -> Dict[str, Dict[str, dict]]:
    """Compute regime distribution for bars and trades."""
    bars = {k: int(v) for k, v in regime_bar_counts.items()}
    trade_groups: Dict[str, List[BacktestTrade]] = defaultdict(list)
    for t in trades:
        trade_groups[t.regime].append(t)

    trades_summary: Dict[str, dict] = {}
    all_regimes = set(bars.keys()) | set(trade_groups.keys()) | {"CHOP_NO_TRADE"}
    for regime in sorted(all_regimes):
        group = trade_groups.get(regime, [])
        if regime == "CHOP_NO_TRADE":
            trades_summary[regime] = {"count": len(group)}
            continue
        wins = [t for t in group if t.pnl_usd > 0]
        gross_wins = sum(t.pnl_usd for t in wins)
        gross_losses = abs(sum(t.pnl_usd for t in group if t.pnl_usd <= 0))
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else 0.0
        avg_r = sum(t.pnl_r for t in group) / len(group) if group else 0.0
        trades_summary[regime] = {
            "count": len(group),
            "win_rate": (len(wins) / len(group)) if group else 0.0,
            "profit_factor": profit_factor,
            "avg_r": avg_r,
            "net_pnl_usd": sum(t.pnl_usd for t in group),
        }

    return {
        "bars": bars,
        "trades": trades_summary,
    }


def _compute_drawdown_episodes(
    equity_curve: List[Tuple[int, float]],
    trades: List[BacktestTrade],
) -> List[Dict[str, float]]:
    """Identify drawdown episodes from equity curve."""
    episodes: List[Dict[str, float]] = []
    if len(equity_curve) < 2:
        return episodes

    peak_eq = equity_curve[0][1]
    peak_ts = equity_curve[0][0]
    in_dd = False
    trough_eq = peak_eq

    for ts, eq in equity_curve[1:]:
        if eq >= peak_eq:
            if in_dd:
                episodes.append(_build_dd_episode(peak_ts, ts, peak_eq, trough_eq, trades))
                in_dd = False
            peak_eq = eq
            peak_ts = ts
            trough_eq = eq
            continue

        in_dd = True
        if eq < trough_eq:
            trough_eq = eq

    if in_dd:
        episodes.append(_build_dd_episode(peak_ts, equity_curve[-1][0], peak_eq, trough_eq, trades))

    return episodes


def _build_dd_episode(
    start_ts: int,
    end_ts: int,
    peak_eq: float,
    trough_eq: float,
    trades: List[BacktestTrade],
) -> Dict[str, float]:
    dd_usd = trough_eq - peak_eq
    dd_pct = (dd_usd / peak_eq * 100.0) if peak_eq > 0 else 0.0
    duration_days = (end_ts - start_ts) / 1000 / 86400
    start_dt = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc)
    trade_count = sum(
        1 for t in trades
        if t.exit_time is not None and start_dt <= t.exit_time <= end_dt
    )
    return {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "depth_pct": dd_pct,
        "depth_usd": dd_usd,
        "duration_days": duration_days,
        "trades": trade_count,
    }


def _compute_reject_reason_distribution(
    counts: Dict[str, int],
) -> Dict[str, int]:
    keys = [
        "COST_GATE",
        "CHOP_GATE",
        "SPREAD_GATE",
        "RISK_BLOCK",
        "COOLDOWN",
        "INSUFFICIENT_CONFIDENCE",
        "INSUFFICIENT_MARGIN",
    ]
    return {k: int(counts.get(k, 0)) for k in keys}


def _compute_cost_breakdown(
    total_gross: float,
    total_fees: float,
    total_slippage: float,
    total_funding: float,
    total_net: float,
    total_trades: int,
) -> Dict[str, float]:
    net_over_gross = (total_net / total_gross) if total_gross != 0 else 0.0
    return {
        "total_gross_pnl_usd": total_gross,
        "total_fees_usd": total_fees,
        "total_slippage_usd": total_slippage,
        "total_funding_usd": total_funding,
        "total_net_pnl_usd": total_net,
        "avg_fee_per_trade_usd": (total_fees / total_trades) if total_trades > 0 else 0.0,
        "avg_slippage_per_trade_usd": (total_slippage / total_trades) if total_trades > 0 else 0.0,
        "net_over_gross_ratio": net_over_gross,
    }


def _compute_mae_mfe(trades: List[BacktestTrade]) -> MAEMFEStats:
    """Compute MAE/MFE statistics across all trades."""
    if not trades:
        return MAEMFEStats()

    maes = [t.max_adverse_excursion for t in trades]   # negative values (losses)
    mfes = [t.max_favourable_excursion for t in trades]  # positive values (gains)
    risks = [t.risk_usd for t in trades]

    avg_mae_usd = sum(maes) / len(maes)
    avg_mfe_usd = sum(mfes) / len(mfes)
    avg_mae_mfe_ratio = abs(avg_mae_usd) / avg_mfe_usd if avg_mfe_usd > 0 else 0.0

    # R-multiples: MAE/risk and MFE/risk
    mae_r_vals = [
        mae / r if r > 0 else 0.0
        for mae, r in zip(maes, risks)
    ]
    mfe_r_vals = [
        mfe / r if r > 0 else 0.0
        for mfe, r in zip(mfes, risks)
    ]
    avg_mae_r = sum(mae_r_vals) / len(mae_r_vals)
    avg_mfe_r = sum(mfe_r_vals) / len(mfe_r_vals)

    # MFE capture: for winning trades, how much of max profit was taken
    win_captures = []
    for t in trades:
        if t.max_favourable_excursion > 0 and t.pnl_usd > 0:
            win_captures.append(t.pnl_usd / t.max_favourable_excursion)
    avg_mfe_capture = sum(win_captures) / len(win_captures) if win_captures else 0.0

    # SL efficiency: for losing trades, how close to stop did actual loss get?
    # ratio = |actual_loss| / risk_usd  (1.0 = perfect, >1 = slippage beyond stop)
    sl_effs = []
    for t in trades:
        if t.pnl_usd < 0 and t.risk_usd > 0:
            sl_effs.append(abs(t.pnl_usd) / t.risk_usd)
    avg_sl_eff = sum(sl_effs) / len(sl_effs) if sl_effs else 0.0

    return MAEMFEStats(
        avg_mae_usd=avg_mae_usd,
        avg_mfe_usd=avg_mfe_usd,
        avg_mae_r=avg_mae_r,
        avg_mfe_r=avg_mfe_r,
        avg_mae_mfe_ratio=avg_mae_mfe_ratio,
        avg_mfe_capture_pct=avg_mfe_capture,
        avg_sl_efficiency=avg_sl_eff,
    )


def _compute_time_stats(trades: List[BacktestTrade]) -> TimeStats:
    """Break down avg R by day-of-week and hour-of-day."""
    _DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    weekday_r: Dict[str, List[float]] = defaultdict(list)
    hour_r: Dict[int, List[float]] = defaultdict(list)

    for t in trades:
        if t.entry_time is None:
            continue
        day_name = _DAYS[t.entry_time.weekday()]
        weekday_r[day_name].append(t.pnl_r)
        hour_r[t.entry_time.hour].append(t.pnl_r)

    by_weekday = {d: sum(v) / len(v) for d, v in weekday_r.items()}
    by_hour = {h: sum(v) / len(v) for h, v in hour_r.items()}
    by_weekday_trades = {d: len(v) for d, v in weekday_r.items()}
    by_hour_trades = {h: len(v) for h, v in hour_r.items()}

    return TimeStats(
        by_weekday=by_weekday,
        by_hour=by_hour,
        by_weekday_trades=by_weekday_trades,
        by_hour_trades=by_hour_trades,
    )


def _compute_time_filter_suggestions(
    time_stats: TimeStats,
    min_samples: int,
    avg_r_threshold: float,
) -> Dict[str, List]:
    bad_hours = [
        hour for hour, avg_r in time_stats.by_hour.items()
        if time_stats.by_hour_trades.get(hour, 0) >= min_samples and avg_r < avg_r_threshold
    ]
    bad_weekdays = [
        day for day, avg_r in time_stats.by_weekday.items()
        if time_stats.by_weekday_trades.get(day, 0) >= min_samples and avg_r < avg_r_threshold
    ]
    return {
        "bad_hours": sorted(bad_hours),
        "bad_weekdays": bad_weekdays,
        "min_samples": min_samples,
        "avg_r_threshold": avg_r_threshold,
    }


def _compute_strategy_regime_matrix(
    trades: List[BacktestTrade],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Build a strategy × regime matrix.
    Returns: strategy → regime → {trades, win_rate, avg_r, pnl_usd}
    """
    # strategy → regime → list of trades
    matrix: Dict[str, Dict[str, List[BacktestTrade]]] = defaultdict(lambda: defaultdict(list))
    for t in trades:
        matrix[t.strategy][t.regime].append(t)

    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    for strat, regime_map in matrix.items():
        result[strat] = {}
        for regime, group in regime_map.items():
            wins = [t for t in group if t.pnl_usd > 0]
            result[strat][regime] = {
                "trades": float(len(group)),
                "win_rate": len(wins) / len(group) if group else 0.0,
                "avg_r": sum(t.pnl_r for t in group) / len(group) if group else 0.0,
                "pnl_usd": sum(t.pnl_usd for t in group),
            }
    return result


def _compute_streaks(
    trades: List[BacktestTrade],
) -> Tuple[int, int, float, float]:
    """
    Compute consecutive win/loss streaks.

    Returns:
        (max_consecutive_losses, max_consecutive_wins,
         avg_consecutive_losses, worst_loss_streak_pnl_usd)
    """
    if not trades:
        return 0, 0, 0.0, 0.0

    max_losses = 0
    max_wins = 0
    cur_losses = 0
    cur_wins = 0

    loss_streaks: List[int] = []
    cur_streak_pnl = 0.0
    worst_streak_pnl = 0.0
    streak_pnl_map: List[float] = []  # pnl of each loss streak

    for t in trades:
        if t.pnl_usd <= 0:
            cur_losses += 1
            cur_wins = 0
            cur_streak_pnl += t.pnl_usd
            if cur_losses > max_losses:
                max_losses = cur_losses
        else:
            if cur_losses > 0:
                loss_streaks.append(cur_losses)
                streak_pnl_map.append(cur_streak_pnl)
                if cur_streak_pnl < worst_streak_pnl:
                    worst_streak_pnl = cur_streak_pnl
            cur_losses = 0
            cur_streak_pnl = 0.0
            cur_wins += 1
            if cur_wins > max_wins:
                max_wins = cur_wins

    # Flush trailing loss streak
    if cur_losses > 0:
        loss_streaks.append(cur_losses)
        streak_pnl_map.append(cur_streak_pnl)
        if cur_streak_pnl < worst_streak_pnl:
            worst_streak_pnl = cur_streak_pnl

    avg_loss_streak = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0.0

    return max_losses, max_wins, avg_loss_streak, worst_streak_pnl


def _estimate_kill_switch_blocks(
    trades: List[BacktestTrade],
    gap_days: float = 5.0,
) -> int:
    """
    Estimate how many times the kill switch fired by counting
    gaps between consecutive trades that are > gap_days.
    """
    if len(trades) < 2:
        return 0

    sorted_trades = sorted(trades, key=lambda t: t.entry_time)
    gap_seconds = gap_days * 86400
    blocks = 0
    for i in range(1, len(sorted_trades)):
        delta = (
            sorted_trades[i].entry_time - sorted_trades[i - 1].exit_time
        ).total_seconds()
        if delta > gap_seconds:
            blocks += 1
    return blocks
