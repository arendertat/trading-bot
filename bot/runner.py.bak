"""
BotRunner — Production/Paper-Live entry point.

Wires together all subsystems and drives the 5-minute candle-close loop:

    BinanceFuturesClient
        └── Reconciler          (startup: rebuild state from exchange)
    Scheduler
        └── on_candle_close()   (every 5m: regime → strategy → risk → log)
        └── on_daily_open()     (00:00 UTC: refresh universe filter)
        └── on_daily_report()   (00:05 UTC: send daily summary)
    SafeMode                    (blocks new entries on unhealthy conditions)
    KillSwitch                  (daily/weekly PnL stop)
    StateManager                (tracks open positions in memory)
    TradeLogger / ReportingTradeLogger

PAPER_LIVE mode: connects to real exchange for data feed, but never
places real orders.  Simulates fills using paper slippage/fee assumptions.

Usage::

    from bot.runner import BotRunner
    runner = BotRunner(config)
    runner.start()          # blocks until Ctrl-C or kill switch
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from datetime import datetime
from typing import Optional

from bot.config.models import BotConfig
from bot.core.constants import BotMode
from bot.exchange.binance_client import BinanceFuturesClient
from bot.exchange.exceptions import ExchangeError, AuthError
from bot.health.safe_mode import SafeMode, SafeModeReason
from bot.core.scheduler import Scheduler
from bot.risk.kill_switch import KillSwitch
from bot.state.state_manager import StateManager
from bot.state.reconciler import Reconciler
from bot.reporting.trade_logger import TradeLogger
from bot.utils.logger import setup_logging, get_logger


logger = logging.getLogger("trading_bot.runner")


class BotRunner:
    """
    Top-level orchestrator: initialises all subsystems and runs the main loop.

    Parameters
    ----------
    config : BotConfig
        Fully-validated bot configuration.
    """

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self._running = False

        # ── Logging ────────────────────────────────────────────────────
        setup_logging(config.logging)
        logger.info("=" * 60)
        logger.info("Binance USDT-M Futures Bot — starting up")
        logger.info(f"  Mode    : {config.mode}")
        logger.info(f"  Exchange: {config.exchange.name} "
                    f"(testnet={config.exchange.testnet})")
        logger.info(f"  Max pos : {config.risk.max_open_positions}")
        logger.info(f"  Risk/trade: {config.risk.risk_per_trade_pct * 100:.2f}%")
        logger.info("=" * 60)

        # ── Exchange client ────────────────────────────────────────────
        self._client = self._build_exchange_client()

        # ── Core subsystems ────────────────────────────────────────────
        self._state = StateManager()
        self._kill_switch = KillSwitch(config.risk)
        self._safe_mode = SafeMode(recovery_seconds=60)

        # ── Reporting ──────────────────────────────────────────────────
        log_dir = config.logging.log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._trade_logger = TradeLogger(log_dir=log_dir)

        # ── Scheduler ─────────────────────────────────────────────────
        self._scheduler = Scheduler(
            on_candle_close=self._on_candle_close,
            on_daily_open=self._on_daily_open,
            on_daily_report=self._on_daily_report,
        )

        # ── Signal handling (Ctrl-C / SIGTERM) ────────────────────────
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the bot:
        1. Reconcile state from exchange
        2. Run the scheduler loop (blocks until stop())
        """
        logger.info("BotRunner: startup reconciliation …")
        self._reconcile()

        logger.info("BotRunner: entering main event loop")
        self._running = True
        try:
            self._scheduler.run_forever()
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Gracefully stop the bot (called by signal handler)."""
        logger.info("BotRunner: stop requested")
        self._scheduler.stop()
        self._running = False

    # ------------------------------------------------------------------
    # Startup reconciliation
    # ------------------------------------------------------------------

    def _reconcile(self) -> None:
        """Rebuild internal state from exchange on startup/restart."""
        if self.config.mode == BotMode.PAPER_LIVE:
            logger.info("PAPER_LIVE mode: skipping live reconciliation "
                        "(no real positions to restore)")
            return

        try:
            reconciler = Reconciler(self._client, self._state)
            result = reconciler.reconcile()
            logger.info(
                f"Reconciliation done: {result.positions_restored} positions "
                f"restored, {result.orphan_orders_cancelled} orphans cancelled, "
                f"{result.orders_linked} orders linked"
            )
            if result.errors:
                for err in result.errors:
                    logger.warning(f"Reconciliation warning: {err}")
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            self._safe_mode.trigger(
                SafeModeReason.UNEXPECTED_EXCEPTION,
                f"Startup reconciliation error: {e}",
            )

    # ------------------------------------------------------------------
    # Scheduled callbacks
    # ------------------------------------------------------------------

    def _on_candle_close(self, now: datetime) -> None:
        """
        Main 5-minute decision pipeline.

        In PAPER_LIVE mode this:
        1. Fetches latest kline data
        2. Runs regime detection
        3. Evaluates strategy signals
        4. Checks risk constraints
        5. Simulates order fills (paper)
        6. Updates state and logs

        Full trading logic will be wired in as each module is completed.
        Currently: connectivity check + kill-switch update + state log.
        """
        logger.debug(f"on_candle_close: {now.isoformat()}")

        # ── Health check ────────────────────────────────────────────────
        if self._safe_mode.is_active:
            logger.warning("SafeMode active — skipping candle pipeline")
            self._safe_mode.record_healthy_check()
            return

        # ── Kill-switch guard ────────────────────────────────────────────
        if self._kill_switch.is_active:
            logger.warning(
                f"Kill switch active ({self._kill_switch.get_active_reason()}) "
                f"— no new entries"
            )

        # ── Fetch balance (health probe) ─────────────────────────────────
        try:
            balance = self._client.fetch_balance_usdt()
            equity_usd = balance["total"]
            free_margin = balance["free"]
            logger.info(
                f"[{now.strftime('%H:%M')} UTC] "
                f"Equity: ${equity_usd:,.2f}  Free: ${free_margin:,.2f}  "
                f"Positions: {self._state.open_position_count()}"
            )
            self._safe_mode.record_healthy_check()

        except AuthError as e:
            logger.error(f"Auth error fetching balance: {e}")
            self._safe_mode.trigger(SafeModeReason.BALANCE_FETCH_FAILED, str(e))
            return
        except ExchangeError as e:
            logger.warning(f"Exchange error fetching balance: {e}")
            self._safe_mode.trigger(SafeModeReason.BALANCE_FETCH_FAILED, str(e))
            return
        except Exception as e:
            logger.error(f"Unexpected error in candle pipeline: {e}")
            self._safe_mode.trigger(SafeModeReason.UNEXPECTED_EXCEPTION, str(e))
            return

        # ── Update kill switch with latest PnL ──────────────────────────
        # NOTE: Full PnL tracking wired in later milestones.
        # For now we pass zero realized PnL so kill switch stays inactive.
        self._kill_switch.update_pnl(
            realized_pnl_today=0.0,
            realized_pnl_week=0.0,
            equity_usd=equity_usd,
            now_utc=now,
        )

        # ── [PAPER_LIVE] Trading pipeline placeholder ────────────────────
        # Strategy signal generation, regime detection, risk checks, and
        # paper order simulation are wired here as those modules are
        # completed. The scaffold above handles all safety gates.

    def _on_daily_open(self, now: datetime) -> None:
        """00:00 UTC — refresh universe filter."""
        logger.info(f"Daily open event: {now.date()} UTC")
        # Universe refresh will be wired once universe selector is integrated.

    def _on_daily_report(self, now: datetime) -> None:
        """00:05 UTC — generate daily summary."""
        logger.info(f"Daily report event: {now.date()} UTC")
        open_pos = self._state.open_position_count()
        logger.info(
            f"Daily summary | Open positions: {open_pos} | "
            f"Kill switch: {self._kill_switch.is_active}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_exchange_client(self) -> BinanceFuturesClient:
        """Build exchange client from config + environment."""
        api_key = os.getenv(self.config.exchange.api_key_env, "")
        api_secret = os.getenv(self.config.exchange.api_secret_env, "")

        if not api_key or not api_secret:
            raise ValueError(
                f"API credentials missing. Set {self.config.exchange.api_key_env} "
                f"and {self.config.exchange.api_secret_env} in .env"
            )

        client = BinanceFuturesClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=self.config.exchange.testnet,
            recv_window_ms=self.config.exchange.recv_window_ms,
        )

        # Connectivity check
        if client.ping():
            logger.info(
                f"Exchange connected "
                f"({'TESTNET' if self.config.exchange.testnet else 'LIVE'})"
            )
        else:
            logger.warning("Exchange ping failed — connectivity may be limited")

        return client

    def _shutdown(self) -> None:
        """Clean up resources on exit."""
        logger.info("BotRunner: shutting down …")
        try:
            self._trade_logger.close()
        except Exception:
            pass
        logger.info("BotRunner: shutdown complete")

    def _handle_signal(self, signum: int, frame) -> None:
        """Handle SIGINT / SIGTERM gracefully."""
        sig_name = signal.Signals(signum).name
        logger.info(f"BotRunner: received {sig_name} — stopping")
        self.stop()
