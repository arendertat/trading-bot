"""
Dashboard FastAPI application factory.

Creates and configures the FastAPI app with all routes.
Serves the single-page HTML dashboard and JSON API endpoints.

Usage (standalone)::

    uvicorn bot.dashboard.app:create_app --factory --host 0.0.0.0 --port 8080

Usage (from BotRunner)::

    from bot.dashboard.app import create_app
    app = create_app(state_manager, perf_tracker, log_reader, kill_switch, safe_mode)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from bot.dashboard.routes import build_router

_STATIC_DIR = Path(__file__).parent / "static"
_TEMPLATE_DIR = Path(__file__).parent / "templates"


def create_app(
    state_manager=None,
    perf_tracker=None,
    log_reader=None,
    kill_switch=None,
    safe_mode=None,
    risk_limits=None,
) -> FastAPI:
    """
    Create and configure the FastAPI dashboard app.

    Parameters are optional — when None, endpoints return empty/default data.
    This allows the dashboard to start even before BotRunner is fully wired.

    Parameters
    ----------
    state_manager : StateManager
        Live position + order state.
    perf_tracker : PerformanceTracker
        Per-strategy rolling metrics.
    log_reader : LogReader
        Historical JSONL trade reader.
    kill_switch : KillSwitch
        Daily/weekly PnL stop state.
    safe_mode : SafeMode
        Exchange health gate state.
    risk_limits : RiskLimits
        Portfolio exposure limits (for config display).
    """
    app = FastAPI(
        title="Trading Bot Dashboard",
        description="Binance USDT-M Futures paper trading monitor",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
    )

    # Mount static files (CSS, JS)
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Build and attach API router
    router = build_router(
        state_manager=state_manager,
        perf_tracker=perf_tracker,
        log_reader=log_reader,
        kill_switch=kill_switch,
        safe_mode=safe_mode,
        risk_limits=risk_limits,
    )
    app.include_router(router, prefix="/api")

    # Serve single-page dashboard at root
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index():
        html_path = _TEMPLATE_DIR / "index.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Dashboard loading…</h1>", status_code=200)

    return app
