"""
Daily Summary Report Generator

Reads closed trade logs via LogReader and produces a structured daily
summary report with PnL, drawdown, win rate, fees, and per-strategy
performance breakdown.

The report is available as:
- DailySummary dataclass (machine-readable)
- Formatted text string suitable for Telegram or console output

Usage::

    from bot.reporting.daily_report import DailyReportGenerator

    gen = DailyReportGenerator(log_dir="logs/")
    summary = gen.build(for_date=date.today())
    print(gen.format_text(summary))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from bot.state.log_reader import LogReader


module_logger = logging.getLogger("trading_bot.reporting.daily_report")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class StrategyBreakdown:
    """
    Per-strategy performance breakdown for a single day.

    Attributes:
        strategy_name: Strategy identifier
        trade_count: Number of closed trades
        winning_trades: Number of profitable trades
        losing_trades: Number of losing trades
        win_rate: Winning trades / total trades (0.0–1.0)
        total_pnl_usd: Sum of realized PnL in USD
        total_pnl_r: Sum of PnL in R-multiples
        avg_pnl_r: Average R per trade
        total_fees_usd: Sum of fees paid
    """

    strategy_name: str
    trade_count: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl_usd: float
    total_pnl_r: float
    avg_pnl_r: float
    total_fees_usd: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dict."""
        return {
            "strategy_name": self.strategy_name,
            "trade_count": self.trade_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "total_pnl_usd": round(self.total_pnl_usd, 4),
            "total_pnl_r": round(self.total_pnl_r, 4),
            "avg_pnl_r": round(self.avg_pnl_r, 4),
            "total_fees_usd": round(self.total_fees_usd, 4),
        }


@dataclass
class DailySummary:
    """
    Daily trading summary report.

    Attributes:
        report_date: The date this summary covers (UTC)
        generated_at: Timestamp when the report was generated
        mode: Bot execution mode (PAPER_LIVE | LIVE | BACKTEST)

        realized_pnl_usd: Sum of closed trade PnL in USD
        realized_pnl_pct: realized_pnl_usd / equity_start * 100
        unrealized_pnl_usd: Sum of unrealized PnL on open positions (if provided)

        equity_start_usd: Account equity at start of day
        equity_end_usd: Account equity at end of day (start + realized)
        equity_peak_usd: Highest equity seen during the day

        daily_drawdown_pct: Max intra-day drawdown from equity_peak (negative)
        weekly_pnl_usd: Rolling 7-day realized PnL in USD (if provided)
        max_drawdown_usd: Max drawdown in USD over the period

        trade_count: Number of closed trades
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        win_rate: winning_trades / trade_count (0.0–1.0)
        expectancy_r: Expected R per trade (positive = profitable edge)

        total_fees_usd: Total trading fees paid
        total_funding_usd: Total funding fees paid

        exit_reason_counts: Breakdown of exit reasons
        strategy_breakdown: Per-strategy performance metrics
    """

    report_date: date
    generated_at: str  # ISO-8601

    # Mode
    mode: str

    # PnL
    realized_pnl_usd: float
    realized_pnl_pct: float
    unrealized_pnl_usd: float

    # Equity
    equity_start_usd: float
    equity_end_usd: float
    equity_peak_usd: float

    # Drawdown
    daily_drawdown_pct: float
    weekly_pnl_usd: float
    max_drawdown_usd: float

    # Trade stats
    trade_count: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    expectancy_r: float

    # Costs
    total_fees_usd: float
    total_funding_usd: float

    # Breakdowns
    exit_reason_counts: Dict[str, int]
    strategy_breakdown: List[StrategyBreakdown]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dict (JSON-safe)."""
        return {
            "report_date": self.report_date.isoformat(),
            "generated_at": self.generated_at,
            "mode": self.mode,
            "realized_pnl_usd": round(self.realized_pnl_usd, 4),
            "realized_pnl_pct": round(self.realized_pnl_pct, 4),
            "unrealized_pnl_usd": round(self.unrealized_pnl_usd, 4),
            "equity_start_usd": round(self.equity_start_usd, 4),
            "equity_end_usd": round(self.equity_end_usd, 4),
            "equity_peak_usd": round(self.equity_peak_usd, 4),
            "daily_drawdown_pct": round(self.daily_drawdown_pct, 4),
            "weekly_pnl_usd": round(self.weekly_pnl_usd, 4),
            "max_drawdown_usd": round(self.max_drawdown_usd, 4),
            "trade_count": self.trade_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "expectancy_r": round(self.expectancy_r, 4),
            "total_fees_usd": round(self.total_fees_usd, 4),
            "total_funding_usd": round(self.total_funding_usd, 4),
            "exit_reason_counts": self.exit_reason_counts,
            "strategy_breakdown": [s.to_dict() for s in self.strategy_breakdown],
        }


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------


class DailyReportGenerator:
    """
    Generates DailySummary reports from JSONL trade logs.

    Reads TRADE_CLOSED / TRADE_RECORD events from the trade log for the
    given date, aggregates metrics, and returns a DailySummary.

    Usage::

        gen = DailyReportGenerator(log_dir="logs/")
        summary = gen.build(
            for_date=date(2024, 2, 11),
            equity_start_usd=10_000.0,
            mode="PAPER_LIVE",
        )
        text = gen.format_text(summary)
    """

    def __init__(self, log_dir: str = "logs") -> None:
        """
        Initialise DailyReportGenerator.

        Args:
            log_dir: Directory containing JSONL log files
        """
        self._reader = LogReader(log_dir=log_dir)
        module_logger.info(f"DailyReportGenerator initialised (log_dir={log_dir!r})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        for_date: Optional[date] = None,
        equity_start_usd: float = 0.0,
        unrealized_pnl_usd: float = 0.0,
        weekly_pnl_usd: float = 0.0,
        mode: str = "PAPER_LIVE",
    ) -> DailySummary:
        """
        Build a DailySummary for the given date.

        Reads all TRADE_CLOSED / TRADE_RECORD entries from the trade log
        for the specified date and aggregates statistics.

        Args:
            for_date: Date to generate report for (defaults to today UTC)
            equity_start_usd: Account equity at start of day
            unrealized_pnl_usd: Current unrealized PnL on open positions
            weekly_pnl_usd: Rolling 7-day realized PnL in USD
            mode: Bot execution mode string

        Returns:
            DailySummary dataclass
        """
        target_date = for_date or datetime.utcnow().date()
        module_logger.info(f"Building daily report for {target_date.isoformat()}")

        # Collect all closed trades for the day from both TRADE_CLOSED and
        # TRADE_RECORD events (Task 15 schema-validated records).
        trades = self._collect_trades(target_date)

        # Aggregate metrics
        summary = self._aggregate(
            trades=trades,
            report_date=target_date,
            equity_start_usd=equity_start_usd,
            unrealized_pnl_usd=unrealized_pnl_usd,
            weekly_pnl_usd=weekly_pnl_usd,
            mode=mode,
        )

        module_logger.info(
            f"Daily report built: {summary.trade_count} trades, "
            f"PnL={summary.realized_pnl_usd:.2f} USD, "
            f"win_rate={summary.win_rate:.1%}"
        )
        return summary

    def format_text(self, summary: DailySummary) -> str:
        """
        Format a DailySummary as a human-readable text report.

        Suitable for Telegram messages (plain text, no HTML/Markdown).

        Args:
            summary: DailySummary to format

        Returns:
            Formatted multi-line string
        """
        lines: List[str] = []
        sign = "+" if summary.realized_pnl_usd >= 0 else ""

        lines.append(f"=== Daily Report {summary.report_date.isoformat()} ===")
        lines.append(f"Mode: {summary.mode}")
        lines.append("")

        # PnL
        lines.append("--- PnL ---")
        lines.append(f"Realized PnL:    {sign}{summary.realized_pnl_usd:>10.2f} USD  ({sign}{summary.realized_pnl_pct:.2f}%)")
        lines.append(f"Unrealized PnL:  {summary.unrealized_pnl_usd:>10.2f} USD")
        lines.append(f"Weekly PnL:      {summary.weekly_pnl_usd:>10.2f} USD")
        lines.append("")

        # Equity & Drawdown
        lines.append("--- Equity ---")
        lines.append(f"Start:           {summary.equity_start_usd:>10.2f} USD")
        lines.append(f"End:             {summary.equity_end_usd:>10.2f} USD")
        lines.append(f"Peak:            {summary.equity_peak_usd:>10.2f} USD")
        lines.append(f"Daily DD:        {summary.daily_drawdown_pct:>10.2f}%")
        lines.append(f"Max DD:          {summary.max_drawdown_usd:>10.2f} USD")
        lines.append("")

        # Trades
        lines.append("--- Trades ---")
        lines.append(f"Total:           {summary.trade_count:>10d}")
        lines.append(f"Win / Loss:      {summary.winning_trades:>4d} / {summary.losing_trades:<4d}")
        if summary.trade_count > 0:
            lines.append(f"Win Rate:        {summary.win_rate:>10.1%}")
            lines.append(f"Expectancy R:    {summary.expectancy_r:>+10.3f} R")
        lines.append("")

        # Costs
        lines.append("--- Costs ---")
        lines.append(f"Fees:            {summary.total_fees_usd:>10.2f} USD")
        lines.append(f"Funding:         {summary.total_funding_usd:>10.2f} USD")
        lines.append("")

        # Exit reasons
        if summary.exit_reason_counts:
            lines.append("--- Exit Reasons ---")
            for reason, count in sorted(summary.exit_reason_counts.items()):
                lines.append(f"  {reason:<16s}: {count}")
            lines.append("")

        # Strategy breakdown
        if summary.strategy_breakdown:
            lines.append("--- Strategy Breakdown ---")
            for sb in sorted(summary.strategy_breakdown, key=lambda s: s.total_pnl_usd, reverse=True):
                sign_s = "+" if sb.total_pnl_usd >= 0 else ""
                lines.append(
                    f"  {sb.strategy_name:<26s}"
                    f" {sb.trade_count:>2d}T"
                    f"  WR={sb.win_rate:.0%}"
                    f"  PnL={sign_s}{sb.total_pnl_usd:.2f} USD"
                    f"  E[R]={sb.avg_pnl_r:+.2f}"
                )

        lines.append("")
        lines.append(f"Generated: {summary.generated_at}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_trades(self, target_date: date) -> List[Dict[str, Any]]:
        """
        Collect all closed trade payloads for the given date.

        Accepts both legacy TRADE_CLOSED records (raw position dict) and
        schema-validated TRADE_RECORD events from Task 15.

        Args:
            target_date: Date to collect trades for

        Returns:
            List of trade payload dicts
        """
        trades: List[Dict[str, Any]] = []

        for record in self._reader.read_trades(start=target_date, end=target_date):
            if record.event == "TRADE_CLOSED":
                # Raw position dict — normalize to common field names
                payload = self._normalize_raw_trade(record.payload)
                trades.append(payload)
            elif record.event == "TRADE_RECORD":
                # Schema-validated record — already in correct shape
                payload = self._normalize_schema_trade(record.payload)
                trades.append(payload)

        module_logger.debug(f"Collected {len(trades)} trades for {target_date.isoformat()}")
        return trades

    def _normalize_raw_trade(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a TRADE_CLOSED (raw position dict) payload to common schema.

        Args:
            payload: Raw position payload from TradeLogger.log_trade_closed()

        Returns:
            Normalized dict with standard field names
        """
        return {
            "trade_id": payload.get("position_id", ""),
            "symbol": payload.get("symbol", ""),
            "strategy": payload.get("strategy", "UNKNOWN"),
            "regime": payload.get("regime", "UNKNOWN"),
            "direction": payload.get("side", ""),
            "realized_pnl_usd": payload.get("realized_pnl_usd", 0.0),
            "pnl_r": payload.get("pnl_r", 0.0),
            "fees_usd": payload.get("fees_paid_usd", 0.0),
            "funding_usd": payload.get("funding_paid_usd", 0.0),
            "exit_reason": payload.get("exit_reason", "UNKNOWN"),
            "entry_time": payload.get("entry_time", ""),
            "exit_time": payload.get("exit_time", ""),
        }

    def _normalize_schema_trade(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a TRADE_RECORD (schema-validated) payload to common schema.

        Args:
            payload: Schema-validated trade payload

        Returns:
            Normalized dict with standard field names
        """
        result = payload.get("result", {})
        costs = payload.get("costs", {})
        return {
            "trade_id": payload.get("trade_id", ""),
            "symbol": payload.get("symbol", ""),
            "strategy": payload.get("strategy", "UNKNOWN"),
            "regime": payload.get("regime", "UNKNOWN"),
            "direction": payload.get("direction", ""),
            "realized_pnl_usd": result.get("pnl_usd", 0.0),
            "pnl_r": result.get("pnl_r_multiple", 0.0),
            "fees_usd": costs.get("fees_usd", 0.0),
            "funding_usd": costs.get("funding_usd", 0.0),
            "exit_reason": result.get("reason", "UNKNOWN"),
            "entry_time": payload.get("timestamp_open", ""),
            "exit_time": payload.get("timestamp_close", ""),
        }

    def _aggregate(
        self,
        trades: List[Dict[str, Any]],
        report_date: date,
        equity_start_usd: float,
        unrealized_pnl_usd: float,
        weekly_pnl_usd: float,
        mode: str,
    ) -> DailySummary:
        """
        Aggregate trade list into a DailySummary.

        Args:
            trades: Normalized trade dicts
            report_date: Report date
            equity_start_usd: Starting equity
            unrealized_pnl_usd: Current unrealized PnL
            weekly_pnl_usd: Rolling weekly PnL
            mode: Bot mode string

        Returns:
            DailySummary dataclass
        """
        # PnL totals
        pnl_values = [t["realized_pnl_usd"] for t in trades]
        r_values = [t["pnl_r"] for t in trades]
        realized_pnl_usd = sum(pnl_values)
        total_pnl_r = sum(r_values)

        # Trade counts
        total = len(trades)
        winning = [p for p in pnl_values if p > 0]
        losing = [p for p in pnl_values if p <= 0]
        win_rate = len(winning) / total if total > 0 else 0.0
        expectancy_r = total_pnl_r / total if total > 0 else 0.0

        # Costs
        total_fees_usd = sum(t["fees_usd"] for t in trades)
        total_funding_usd = sum(t["funding_usd"] for t in trades)

        # Equity
        realized_pnl_pct = (realized_pnl_usd / equity_start_usd * 100.0) if equity_start_usd > 0 else 0.0
        equity_end_usd = equity_start_usd + realized_pnl_usd

        # Intra-day drawdown: simulate cumulative equity curve
        equity_peak_usd, max_drawdown_usd, daily_drawdown_pct = self._compute_drawdown(
            equity_start=equity_start_usd,
            pnl_sequence=pnl_values,
        )

        # Exit reasons
        exit_reason_counts: Dict[str, int] = {}
        for t in trades:
            reason = t.get("exit_reason") or "UNKNOWN"
            exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1

        # Strategy breakdown
        strategy_breakdown = self._compute_strategy_breakdown(trades)

        return DailySummary(
            report_date=report_date,
            generated_at=datetime.utcnow().isoformat(),
            mode=mode,
            realized_pnl_usd=round(realized_pnl_usd, 4),
            realized_pnl_pct=round(realized_pnl_pct, 4),
            unrealized_pnl_usd=round(unrealized_pnl_usd, 4),
            equity_start_usd=round(equity_start_usd, 4),
            equity_end_usd=round(equity_end_usd, 4),
            equity_peak_usd=round(equity_peak_usd, 4),
            daily_drawdown_pct=round(daily_drawdown_pct, 4),
            weekly_pnl_usd=round(weekly_pnl_usd, 4),
            max_drawdown_usd=round(max_drawdown_usd, 4),
            trade_count=total,
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=round(win_rate, 6),
            expectancy_r=round(expectancy_r, 6),
            total_fees_usd=round(total_fees_usd, 4),
            total_funding_usd=round(total_funding_usd, 4),
            exit_reason_counts=exit_reason_counts,
            strategy_breakdown=strategy_breakdown,
        )

    def _compute_drawdown(
        self,
        equity_start: float,
        pnl_sequence: List[float],
    ) -> tuple[float, float, float]:
        """
        Compute equity peak, max drawdown USD, and daily drawdown pct.

        Simulates a sequential equity curve from the PnL sequence.

        Args:
            equity_start: Starting equity
            pnl_sequence: Ordered list of realized PnL values

        Returns:
            (equity_peak_usd, max_drawdown_usd, daily_drawdown_pct)
        """
        if not pnl_sequence or equity_start <= 0:
            return equity_start, 0.0, 0.0

        equity = equity_start
        peak = equity_start
        max_dd_usd = 0.0

        for pnl in pnl_sequence:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd_usd:
                max_dd_usd = dd

        daily_dd_pct = (-max_dd_usd / peak * 100.0) if peak > 0 else 0.0
        return peak, max_dd_usd, daily_dd_pct

    def _compute_strategy_breakdown(
        self, trades: List[Dict[str, Any]]
    ) -> List[StrategyBreakdown]:
        """
        Compute per-strategy performance metrics.

        Args:
            trades: Normalized trade dicts

        Returns:
            List of StrategyBreakdown sorted by total_pnl_usd descending
        """
        # Group by strategy
        by_strategy: Dict[str, List[Dict[str, Any]]] = {}
        for t in trades:
            strat = t.get("strategy") or "UNKNOWN"
            by_strategy.setdefault(strat, []).append(t)

        breakdowns: List[StrategyBreakdown] = []
        for strat_name, strat_trades in by_strategy.items():
            pnl_vals = [t["realized_pnl_usd"] for t in strat_trades]
            r_vals = [t["pnl_r"] for t in strat_trades]
            count = len(strat_trades)
            wins = [p for p in pnl_vals if p > 0]

            breakdowns.append(
                StrategyBreakdown(
                    strategy_name=strat_name,
                    trade_count=count,
                    winning_trades=len(wins),
                    losing_trades=count - len(wins),
                    win_rate=len(wins) / count if count > 0 else 0.0,
                    total_pnl_usd=sum(pnl_vals),
                    total_pnl_r=sum(r_vals),
                    avg_pnl_r=sum(r_vals) / count if count > 0 else 0.0,
                    total_fees_usd=sum(t["fees_usd"] for t in strat_trades),
                )
            )

        breakdowns.sort(key=lambda s: s.total_pnl_usd, reverse=True)
        return breakdowns
