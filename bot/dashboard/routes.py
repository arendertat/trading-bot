"""
Dashboard API routes.

All endpoints return JSON. The frontend polls these every few seconds.

Endpoints
---------
GET /api/status          — Bot health: kill switch, safe mode, mode
GET /api/positions        — Open positions with unrealized PnL
GET /api/trades           — Recent closed trades (last N days)
GET /api/performance      — Per-strategy metrics (win rate, expectancy, etc.)
GET /api/equity           — Equity curve data points from trade log
GET /api/summary          — Aggregated portfolio summary (PnL today/week)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger("trading_bot.dashboard")


def build_router(
    state_manager=None,
    perf_tracker=None,
    log_reader=None,
    kill_switch=None,
    safe_mode=None,
    risk_limits=None,
) -> APIRouter:
    """Build and return an APIRouter with all dashboard endpoints."""

    router = APIRouter()

    # ------------------------------------------------------------------
    # GET /api/status
    # ------------------------------------------------------------------

    @router.get("/status", summary="Bot health and mode")
    async def get_status() -> Dict[str, Any]:
        """
        Returns current bot health state.

        Fields:
        - kill_switch_active: bool
        - kill_switch_reason: str | null
        - safe_mode_active: bool
        - open_positions: int
        - timestamp: ISO UTC
        """
        ks_active = False
        ks_reason = None
        sm_active = False
        open_pos = 0

        if kill_switch:
            try:
                ks_active = kill_switch.is_active()
                ks_reason = kill_switch.get_active_reason() if ks_active else None
            except Exception:
                pass

        if safe_mode:
            try:
                sm_active = safe_mode.is_active
            except Exception:
                pass

        if state_manager:
            try:
                open_pos = state_manager.open_position_count()
            except Exception:
                pass

        return {
            "kill_switch_active": ks_active,
            "kill_switch_reason": ks_reason,
            "safe_mode_active": sm_active,
            "open_positions": open_pos,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # GET /api/positions
    # ------------------------------------------------------------------

    @router.get("/positions", summary="Open positions")
    async def get_positions() -> List[Dict[str, Any]]:
        """
        Returns all currently open positions with live PnL.
        """
        if not state_manager:
            return []

        try:
            positions = state_manager.get_open_positions()
            result = []
            for p in positions:
                holding_secs = p.holding_time_seconds
                result.append({
                    "position_id": p.position_id,
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "entry_price": round(p.entry_price, 4),
                    "stop_price": round(p.stop_price, 4),
                    "tp_price": round(p.tp_price, 4) if p.tp_price else None,
                    "quantity": round(p.quantity, 6),
                    "notional_usd": round(p.notional_usd, 2),
                    "leverage": p.leverage,
                    "unrealized_pnl_usd": round(p.unrealized_pnl_usd, 2),
                    "fees_paid_usd": round(p.fees_paid_usd, 4),
                    "funding_paid_usd": round(p.funding_paid_usd, 4),
                    "trailing_enabled": p.trailing_enabled,
                    "strategy": p.strategy,
                    "regime": p.regime,
                    "entry_time": p.entry_time.isoformat(),
                    "holding_minutes": round(holding_secs / 60, 1),
                    "risk_amount_usd": round(p.risk_amount_usd, 2),
                })
            return result
        except Exception as e:
            logger.error(f"Dashboard /positions error: {e}")
            return []

    # ------------------------------------------------------------------
    # GET /api/trades
    # ------------------------------------------------------------------

    @router.get("/trades", summary="Recent closed trades")
    async def get_trades(
        days: int = Query(default=7, ge=1, le=90, description="Lookback days"),
        limit: int = Query(default=50, ge=1, le=500, description="Max trades"),
    ) -> List[Dict[str, Any]]:
        """
        Returns recently closed trades from JSONL log.
        """
        if not log_reader:
            return []

        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)
            trades = log_reader.get_closed_trades(start=start, end=end)
            # Most recent first
            trades = sorted(trades, key=lambda t: t.get("exit_time", ""), reverse=True)
            return trades[:limit]
        except Exception as e:
            logger.error(f"Dashboard /trades error: {e}")
            return []

    # ------------------------------------------------------------------
    # GET /api/performance
    # ------------------------------------------------------------------

    @router.get("/performance", summary="Per-strategy performance metrics")
    async def get_performance() -> Dict[str, Any]:
        """
        Returns rolling metrics for each strategy.
        """
        if not perf_tracker:
            return {"strategies": {}}

        try:
            strategies = {}
            for name in perf_tracker.get_all_strategy_names():
                metrics = perf_tracker.get_metrics(name)
                count = perf_tracker.get_trade_count(name)
                if metrics:
                    strategies[name] = {
                        "total_trades": metrics.total_trades,
                        "win_rate": round(metrics.win_rate, 4),
                        "avg_r": round(metrics.avg_r, 4),
                        "expectancy_r": round(metrics.expectancy_r, 4),
                        "max_drawdown_pct": round(metrics.max_drawdown_pct, 4),
                        "confidence": round(metrics.confidence, 4),
                        "fees_total": round(metrics.fees_total, 4),
                        "funding_total": round(metrics.funding_total, 4),
                    }
                else:
                    strategies[name] = {
                        "total_trades": count,
                        "win_rate": None,
                        "avg_r": None,
                        "expectancy_r": None,
                        "max_drawdown_pct": None,
                        "confidence": None,
                        "fees_total": None,
                        "funding_total": None,
                        "note": f"Need ≥10 trades (have {count})",
                    }
            return {"strategies": strategies}
        except Exception as e:
            logger.error(f"Dashboard /performance error: {e}")
            return {"strategies": {}}

    # ------------------------------------------------------------------
    # GET /api/equity
    # ------------------------------------------------------------------

    @router.get("/equity", summary="Equity curve data points")
    async def get_equity(
        days: int = Query(default=30, ge=1, le=365, description="Lookback days"),
    ) -> Dict[str, Any]:
        """
        Returns cumulative PnL over time for equity curve chart.
        Each data point: {time, cumulative_pnl_usd, trade_count}
        """
        if not log_reader:
            return {"points": [], "total_pnl_usd": 0.0}

        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)
            trades = log_reader.get_closed_trades(start=start, end=end)
            trades_sorted = sorted(trades, key=lambda t: t.get("exit_time", ""))

            points = []
            cumulative = 0.0
            for i, t in enumerate(trades_sorted):
                pnl = t.get("realized_pnl_usd", 0.0) or 0.0
                cumulative += pnl
                points.append({
                    "time": t.get("exit_time"),
                    "cumulative_pnl_usd": round(cumulative, 2),
                    "trade_pnl_usd": round(pnl, 2),
                    "trade_count": i + 1,
                    "symbol": t.get("symbol"),
                    "exit_reason": t.get("exit_reason"),
                })

            return {
                "points": points,
                "total_pnl_usd": round(cumulative, 2),
                "trade_count": len(points),
            }
        except Exception as e:
            logger.error(f"Dashboard /equity error: {e}")
            return {"points": [], "total_pnl_usd": 0.0}

    # ------------------------------------------------------------------
    # GET /api/summary
    # ------------------------------------------------------------------

    @router.get("/summary", summary="Portfolio summary (PnL today/week, stats)")
    async def get_summary(
        days: int = Query(default=30, ge=1, le=365),
    ) -> Dict[str, Any]:
        """
        Returns aggregated portfolio stats: PnL, win rate, fees.
        """
        pnl_summary: Dict[str, Any] = {}
        state_snap: Dict[str, Any] = {}

        if log_reader:
            try:
                end = datetime.now(timezone.utc)
                start = end - timedelta(days=days)
                pnl_summary = log_reader.compute_pnl_summary(start=start, end=end)
            except Exception as e:
                logger.error(f"Dashboard /summary log_reader error: {e}")

        if state_manager:
            try:
                state_snap = state_manager.snapshot()
            except Exception as e:
                logger.error(f"Dashboard /summary state_manager error: {e}")

        # Today's and this week's PnL from recent trades
        today_pnl = 0.0
        week_pnl = 0.0
        if log_reader:
            try:
                now = datetime.now(timezone.utc)
                today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                week_start = today_start - timedelta(days=now.weekday())
                today_summary = log_reader.compute_pnl_summary(start=today_start, end=now)
                week_summary = log_reader.compute_pnl_summary(start=week_start, end=now)
                today_pnl = today_summary.get("total_pnl_usd", 0.0)
                week_pnl = week_summary.get("total_pnl_usd", 0.0)
            except Exception:
                pass

        return {
            "pnl_today_usd": round(today_pnl, 2),
            "pnl_week_usd": round(week_pnl, 2),
            "pnl_summary": pnl_summary,
            "state": state_snap,
            "lookback_days": days,
        }

    return router
