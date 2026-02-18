"""
BotRunner — Production/Paper-Live entry point.

Wires together all subsystems and drives the 5-minute candle-close loop:

    BinanceFuturesClient
        └── Reconciler          (startup: rebuild state from exchange)
    Scheduler
        └── on_candle_close()   (every 5m: universe → klines → features →
                                  regime → strategy → risk → paper-fill)
        └── on_daily_open()     (00:00 UTC: refresh universe filter)
        └── on_daily_report()   (00:05 UTC: send daily summary)
    SafeMode                    (blocks new entries on unhealthy conditions)
    KillSwitch                  (daily/weekly PnL stop)
    StateManager                (tracks open positions in memory)
    TradeLogger

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
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from bot.config.models import BotConfig
from bot.core.constants import BotMode, RegimeType
from bot.core.constants import OrderSide as CoreOrderSide
from bot.core.constants import PositionStatus as CorePositionStatus
from bot.core.performance_tracker import PerformanceTracker
from bot.core.strategy_selector import StrategySelector
from bot.core.types import Position as CorePosition
from bot.data.candle_store import CandleStore
from bot.data.feature_engine import FeatureEngine
from bot.data.klines_ingestor import KlinesIngestor
from bot.exchange.binance_client import BinanceFuturesClient
from bot.exchange.exceptions import ExchangeError, AuthError
from bot.execution.models import OrderSide as ExecOrderSide
from bot.execution.position import Position as ExecPosition, PositionStatus, ExitReason
from bot.health.safe_mode import SafeMode, SafeModeReason
from bot.core.scheduler import Scheduler
from bot.regime.detector import RegimeDetector
from bot.risk.correlation_filter import CorrelationFilter
from bot.risk.kill_switch import KillSwitch
from bot.risk.position_sizing import PositionSizingCalculator
from bot.risk.risk_engine import RiskEngine
from bot.risk.risk_limits import RiskLimits
from bot.execution.trailing_stop import TrailingStopManager
from bot.state.log_reader import LogReader
from bot.state.reconciler import Reconciler
from bot.state.state_manager import StateManager
from bot.state.logger import TradeLogger
from bot.strategies.base import FeatureSet, Strategy
from bot.strategies.trend_pullback import TrendPullbackStrategy
from bot.strategies.trend_breakout import TrendBreakoutStrategy
from bot.strategies.range_mean_reversion import RangeMeanReversionStrategy
from bot.universe.selector import UniverseSelector
from bot.utils.logger import setup_logging


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

        # ── Data pipeline ─────────────────────────────────────────────
        self._candle_store = CandleStore()
        self._klines_ingestor = KlinesIngestor(self._client, self._candle_store)
        self._feature_engine = FeatureEngine(self._candle_store, config.timeframes)

        # ── Universe selection ─────────────────────────────────────────
        self._universe_selector = UniverseSelector(
            self._client, self._candle_store, config.universe
        )
        self._active_symbols: List[str] = []

        # ── Regime detection ──────────────────────────────────────────
        self._regime_detector = RegimeDetector(config.regime)

        # ── Strategies ────────────────────────────────────────────────
        self._strategies: Dict[str, Strategy] = {
            "TrendPullback": TrendPullbackStrategy(
                config.strategies.trend_pullback.model_dump()
            ),
            "TrendBreakout": TrendBreakoutStrategy(
                config.strategies.trend_breakout.model_dump()
            ),
            "RangeMeanReversion": RangeMeanReversionStrategy(
                config.strategies.range_mean_reversion.model_dump()
            ),
        }

        # ── Performance tracker + strategy selector ───────────────────
        self._perf_tracker = PerformanceTracker(config.performance.window_trades)
        self._strategy_selector = StrategySelector(
            performance_tracker=self._perf_tracker,
            strategies=self._strategies,
            stability_hours=config.performance.max_strategy_switches_per_day * 24,
        )

        # ── Trailing stop manager ─────────────────────────────────────
        self._trailing_stop_manager = TrailingStopManager()

        # ── Risk engine ───────────────────────────────────────────────
        self._position_sizing = PositionSizingCalculator(config)
        self._risk_limits = RiskLimits(config.risk)
        self._correlation_filter = CorrelationFilter(config.risk)
        self._risk_engine = RiskEngine(
            config=config,
            kill_switch=self._kill_switch,
            position_sizing=self._position_sizing,
            risk_limits=self._risk_limits,
            correlation_filter=self._correlation_filter,
        )

        # ── Reporting ──────────────────────────────────────────────────
        log_dir = config.logging.log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._trade_logger = TradeLogger(log_dir=log_dir)

        # ── PnL tracking (paper: equity-delta based) ───────────────────
        self._realized_pnl_today: float = 0.0
        self._realized_pnl_week: float = 0.0

        # ── Scheduler ─────────────────────────────────────────────────
        self._scheduler = Scheduler(
            on_candle_close=self._on_candle_close,
            on_daily_open=self._on_daily_open,
            on_daily_report=self._on_daily_report,
            on_weekly_reset=self._on_weekly_reset,
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
        2. Build initial universe
        3. Warm up candle history
        4. Run the scheduler loop (blocks until stop())
        """
        logger.info("BotRunner: startup reconciliation …")
        self._reconcile()

        logger.info("BotRunner: building initial universe …")
        self._refresh_universe(datetime.now(timezone.utc))

        if self._active_symbols:
            logger.info(
                f"BotRunner: warming up candle history for {self._active_symbols} …"
            )
            try:
                self._klines_ingestor.warmup(self._active_symbols)
            except Exception as e:
                logger.warning(f"Candle warmup failed (non-fatal): {e}")

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
            # Bulgu 5: Paper mode recovery — restore open positions from JSONL trade log
            self._recover_paper_positions()
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

    def _recover_paper_positions(self) -> None:
        """
        Bulgu 5: Restore open paper positions from JSONL trade log after crash.

        Reads today's trade log, finds TRADE_OPENED records that have no
        corresponding TRADE_CLOSED, and reconstructs ExecPosition objects.
        """
        log_dir = self.config.logging.log_dir
        try:
            reader = LogReader(log_dir=log_dir)
            open_trade_payloads = reader.get_opened_trades()
        except Exception as e:
            logger.warning(f"PAPER_LIVE: Could not read trade log for recovery: {e}")
            return

        if not open_trade_payloads:
            logger.info("PAPER_LIVE: No open positions found in trade log — clean start")
            return

        restored = 0
        for payload in open_trade_payloads:
            try:
                side = (
                    ExecOrderSide.LONG
                    if payload.get("side") == "LONG"
                    else ExecOrderSide.SHORT
                )
                position = ExecPosition(
                    position_id=payload["position_id"],
                    symbol=payload["symbol"],
                    side=side,
                    entry_price=float(payload["entry_price"]),
                    quantity=float(payload["quantity"]),
                    notional_usd=float(payload.get("notional_usd", 0)),
                    leverage=float(payload.get("leverage", 1)),
                    margin_usd=float(payload.get("margin_usd", 0)),
                    stop_price=float(payload["stop_price"]),
                    tp_price=float(payload["tp_price"]) if payload.get("tp_price") else None,
                    entry_time=datetime.fromisoformat(payload["entry_time"]) if payload.get("entry_time") else datetime.now(timezone.utc),
                    risk_amount_usd=float(payload.get("risk_amount_usd", 0)),
                    initial_stop_price=float(payload.get("initial_stop_price", payload["stop_price"])),
                    trail_after_r=float(payload.get("trail_after_r", 1.0)),
                    atr_trail_mult=float(payload.get("atr_trail_mult", 2.0)),
                    entry_order_id=payload.get("entry_order_id", ""),
                    stop_order_id=payload.get("stop_order_id", ""),
                    tp_order_id=payload.get("tp_order_id"),
                    fees_paid_usd=float(payload.get("fees_paid_usd", 0)),
                    strategy=payload.get("strategy", ""),
                    regime=payload.get("regime", ""),
                )
                self._state.add_position(position)
                restored += 1
                logger.info(
                    f"PAPER_LIVE recovery: restored {position.position_id} "
                    f"({position.symbol} {side.value} @ {position.entry_price})"
                )
            except Exception as e:
                logger.warning(f"PAPER_LIVE: Could not restore position {payload.get('position_id')}: {e}")

        logger.info(f"PAPER_LIVE recovery complete: {restored} position(s) restored")

    # ------------------------------------------------------------------
    # Scheduled callbacks
    # ------------------------------------------------------------------

    def _on_candle_close(self, now: datetime) -> None:
        """
        Main 5-minute decision pipeline.

        Flow:
          1. Safety gates (safe mode)
          2. Fetch balance (health probe + equity snapshot)
          3. Update kill switch with latest realized PnL
          4. Update candles for all active symbols
          5. Check open positions for SL/TP exits (paper simulation)
          6. If kill switch not active: run entry pipeline per symbol
        """
        logger.debug(f"on_candle_close: {now.isoformat()}")

        # ── 1. Safe mode gate ────────────────────────────────────────────
        if self._safe_mode.is_active:
            logger.warning("SafeMode active — skipping candle pipeline")
            self._safe_mode.record_healthy_check()
            return

        # ── 2. Fetch balance ─────────────────────────────────────────────
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

        # ── 3. Kill switch update ────────────────────────────────────────
        self._kill_switch.update_pnl(
            realized_pnl_today=self._realized_pnl_today,
            realized_pnl_week=self._realized_pnl_week,
            equity_usd=equity_usd,
            now_utc=now,
        )

        kill_active = self._kill_switch.is_active()
        if kill_active:
            logger.warning(
                f"Kill switch active ({self._kill_switch.get_active_reason()}) "
                f"— no new entries"
            )

        # ── 4. Update candle data ────────────────────────────────────────
        if self._active_symbols:
            try:
                self._klines_ingestor.update(self._active_symbols)
            except Exception as e:
                logger.warning(f"Candle update failed: {e}")

        # ── 5. Check open position exits ─────────────────────────────────
        self._check_position_exits(equity_usd)

        # ── 6. Entry pipeline ────────────────────────────────────────────
        if not kill_active:
            self._run_entry_pipeline(now, equity_usd, free_margin)

    def _on_daily_open(self, now: datetime) -> None:
        """00:00 UTC — refresh universe filter and reset daily PnL."""
        logger.info(f"Daily open event: {now.date()} UTC")
        self._refresh_universe(now)
        self._realized_pnl_today = 0.0
        logger.info("Daily PnL counter reset")
        # Bulgu 2: Update correlation matrix with fresh 1h data
        self._update_correlation_matrix()

    def _on_weekly_reset(self, now: datetime) -> None:
        """Monday 00:00 UTC — reset weekly PnL window."""
        logger.info(
            f"Weekly reset event: {now.date()} UTC "
            f"(ISO week {now.isocalendar()[1]})"
        )
        self._realized_pnl_week = 0.0
        logger.info(
            f"Weekly PnL counter reset. "
            f"Active positions: {self._state.open_position_count()}"
        )

    def _on_daily_report(self, now: datetime) -> None:
        """00:05 UTC — generate daily summary."""
        logger.info(f"Daily report event: {now.date()} UTC")
        open_pos = self._state.open_position_count()
        logger.info(
            f"Daily summary | Date: {now.date()} | "
            f"Open positions: {open_pos} | "
            f"Kill switch: {self._kill_switch.is_active()} | "
            f"PnL today: ${self._realized_pnl_today:+,.2f} | "
            f"PnL week: ${self._realized_pnl_week:+,.2f} | "
            f"Active symbols: {self._active_symbols}"
        )

    # ------------------------------------------------------------------
    # Universe management
    # ------------------------------------------------------------------

    def _update_correlation_matrix(self) -> None:
        """
        Bulgu 2: Build 1h price arrays for active symbols and update correlation matrix.

        Requires at least 72 1h candles (~3 days) per symbol.
        Called daily at 00:00 UTC after universe refresh.
        """
        min_bars = self.config.timeframes.corr_lookback_hours
        price_data: Dict[str, "np.ndarray"] = {}

        for symbol in self._active_symbols:
            candles_1h = self._candle_store.get_candles(symbol, "1h")
            if candles_1h and len(candles_1h) >= min_bars:
                price_data[symbol] = np.array([c.close for c in candles_1h[-min_bars:]])

        if not price_data:
            logger.debug("Correlation matrix update skipped — insufficient 1h data")
            return

        try:
            self._correlation_filter.update_correlation_matrix(price_data)
            logger.info(
                f"Correlation matrix updated for {len(price_data)} symbols"
            )
        except Exception as e:
            logger.warning(f"Correlation matrix update failed: {e}")

    def _refresh_universe(self, now: datetime) -> None:
        """Rebuild daily universe from exchange data."""
        try:
            symbols = self._universe_selector.build_daily_universe(now)
            self._active_symbols = symbols
            logger.info(f"Universe refreshed: {len(symbols)} symbols — {symbols}")
        except Exception as e:
            logger.error(f"Universe refresh failed: {e}")
            if not self._active_symbols:
                self._active_symbols = list(self.config.universe.whitelist)
                if self._active_symbols:
                    logger.info(f"Using whitelist as fallback: {self._active_symbols}")

    # ------------------------------------------------------------------
    # Entry pipeline
    # ------------------------------------------------------------------

    def _run_entry_pipeline(
        self,
        now: datetime,
        equity_usd: float,
        free_margin: float,
    ) -> None:
        """
        Iterate over active symbols and evaluate entry conditions.

        Skips symbols that already have an open position.
        """
        for symbol in self._active_symbols:
            if self._state.has_open_position(symbol):
                logger.debug(f"{symbol}: open position exists — skipping entry")
                continue
            try:
                self._evaluate_symbol_entry(symbol, now, equity_usd, free_margin)
            except Exception as e:
                logger.error(f"{symbol}: entry pipeline error: {e}")

    def _evaluate_symbol_entry(
        self,
        symbol: str,
        now: datetime,
        equity_usd: float,
        free_margin: float,
    ) -> None:
        """Full entry evaluation for one symbol: features → regime → strategy → risk → fill."""

        # ── Compute features ─────────────────────────────────────────
        features_dict = self._feature_engine.compute_features(symbol)
        if features_dict is None:
            logger.debug(f"{symbol}: insufficient data — skipping")
            return

        try:
            feature_set = FeatureSet(
                rsi_5m=features_dict["rsi14"],
                ema20_5m=features_dict["ema20_5m"],
                ema50_5m=features_dict["ema50_5m"],
                ema20_1h=features_dict["ema20_1h"],
                ema50_1h=features_dict["ema50_1h"],
                atr_5m=features_dict["atr14"],
                bb_upper_5m=features_dict.get("bb_upper") or 0.0,
                bb_lower_5m=features_dict.get("bb_lower") or 0.0,
                bb_middle_5m=features_dict.get("bb_middle") or 0.0,
                high_20_bars=features_dict.get("high_20") or 0.0,
                low_20_bars=features_dict.get("low_20") or 0.0,
                volume_z_5m=features_dict.get("vol_z") or 0.0,
                adx_5m=features_dict.get("adx14"),
                atr_z_5m=features_dict.get("atr_z"),
                bb_width_5m=features_dict.get("bb_width"),
            )
        except KeyError as e:
            logger.warning(f"{symbol}: missing feature {e} — skipping")
            return

        # ── Detect regime ─────────────────────────────────────────────
        regime_result = self._regime_detector.detect_regime(
            symbol=symbol,
            adx=features_dict.get("adx14") or 0.0,
            atr_z=features_dict.get("atr_z") or 0.0,
            bb_width=features_dict.get("bb_width") or 0.02,
            ema20_5m=features_dict["ema20_5m"],
            ema50_5m=features_dict["ema50_5m"],
            ema20_1h=features_dict["ema20_1h"],
            ema50_1h=features_dict["ema50_1h"],
        )

        if regime_result.regime == RegimeType.CHOP_NO_TRADE:
            logger.debug(
                f"{symbol}: CHOP_NO_TRADE (conf={regime_result.confidence:.2f}) — skip"
            )
            return

        logger.debug(
            f"{symbol}: regime={regime_result.regime.value} "
            f"conf={regime_result.confidence:.2f}"
        )

        # ── Select strategy ───────────────────────────────────────────
        strategy = self._strategy_selector.select_strategy(
            regime=regime_result.regime,
            symbol=symbol,
        )
        if strategy is None:
            logger.debug(f"{symbol}: no strategy for {regime_result.regime.value}")
            return

        # ── Current price from last candle ────────────────────────────
        candles_5m = self._candle_store.get_candles(symbol, "5m")
        if not candles_5m:
            return
        current_price = candles_5m[-1].close

        # ── Generate signal ───────────────────────────────────────────
        signal = strategy.generate_signal(
            features=feature_set,
            regime_result=regime_result,
            symbol=symbol,
            current_price=current_price,
            timestamp=int(now.timestamp()),
        )
        if signal is None:
            return

        logger.info(
            f"{symbol}: signal {signal.side.value} @ {current_price:.4f} "
            f"stop={signal.stop_price:.4f} tp={signal.tp_price:.4f} "
            f"R={signal.target_r:.2f} [{strategy.name}]"
        )

        # ── Risk validation ───────────────────────────────────────────
        open_positions = self._state.get_open_positions()
        core_positions = self._to_core_positions(open_positions)
        exec_side = (
            ExecOrderSide.LONG if signal.side == CoreOrderSide.LONG
            else ExecOrderSide.SHORT
        )

        risk_result = self._risk_engine.validate_entry(
            symbol=symbol,
            side=exec_side,
            regime=regime_result.regime,
            stop_pct=signal.stop_pct,
            current_price=current_price,
            equity_usd=equity_usd,
            free_margin_usd=free_margin,
            open_positions=core_positions,
            risk_per_trade_pct=self._effective_risk_pct(),
        )

        if not risk_result.approved:
            logger.info(f"{symbol}: risk rejected — {risk_result.rejection_reason}")
            return

        # ── Paper fill ────────────────────────────────────────────────
        self._paper_fill_entry(
            symbol=symbol,
            signal=signal,
            risk_result=risk_result,
            regime_result=regime_result,
            strategy_name=strategy.name,
            current_price=current_price,
            now=now,
        )

    # ------------------------------------------------------------------
    # Paper fill helpers
    # ------------------------------------------------------------------

    def _paper_fill_entry(
        self,
        symbol: str,
        signal,
        risk_result,
        regime_result,
        strategy_name: str,
        current_price: float,
        now: datetime,
    ) -> None:
        """Simulate LIMIT order fill with slippage + maker fee."""
        slippage = self.config.execution.paper_slippage_limit_pct
        fee_pct = self.config.execution.maker_fee_pct
        ps = risk_result.position_size

        # LONG fills above, SHORT fills below
        if signal.side == CoreOrderSide.LONG:
            fill_price = current_price * (1.0 + slippage)
            exec_side = ExecOrderSide.LONG
        else:
            fill_price = current_price * (1.0 - slippage)
            exec_side = ExecOrderSide.SHORT

        entry_fee_usd = ps.notional_usd * fee_pct
        position_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"

        position = ExecPosition(
            position_id=position_id,
            symbol=symbol,
            side=exec_side,
            entry_price=fill_price,
            quantity=ps.quantity,
            notional_usd=ps.notional_usd,
            leverage=ps.leverage,
            margin_usd=ps.margin_required_usd,
            stop_price=signal.stop_price,
            tp_price=signal.tp_price,
            entry_time=now,
            risk_amount_usd=ps.risk_usd,
            initial_stop_price=signal.stop_price,
            trail_after_r=signal.trail_after_r,
            atr_trail_mult=signal.atr_trail_mult,
            entry_order_id=f"{position_id}-ENTRY",
            stop_order_id=f"{position_id}-STOP",
            tp_order_id=f"{position_id}-TP" if signal.tp_price else None,
            fees_paid_usd=entry_fee_usd,
            strategy=strategy_name,
            regime=regime_result.regime.value,
        )

        self._state.add_position(position)

        logger.info(
            f"[PAPER FILL ENTRY] {symbol} {exec_side.value} @ {fill_price:.4f} "
            f"qty={ps.quantity:.6f} notional=${ps.notional_usd:.2f} "
            f"stop={signal.stop_price:.4f} tp={signal.tp_price:.4f} "
            f"id={position_id} strategy={strategy_name}"
        )

    def _paper_fill_exit(
        self,
        position: ExecPosition,
        exit_price: float,
        exit_reason: ExitReason,
    ) -> None:
        """Simulate position exit with taker fee."""
        fee_pct = self.config.execution.taker_fee_pct
        exit_fee_usd = position.notional_usd * fee_pct

        position.close_position(
            exit_price=exit_price,
            exit_reason=exit_reason,
            fees_paid=exit_fee_usd,
        )
        self._state.update_position(position)

        # Update PnL accumulators
        self._realized_pnl_today += position.realized_pnl_usd
        self._realized_pnl_week += position.realized_pnl_usd

        r_multiple = (
            position.realized_pnl_usd / position.risk_amount_usd
            if position.risk_amount_usd > 0
            else 0.0
        )

        logger.info(
            f"[PAPER FILL EXIT] {position.symbol} {position.side.value} "
            f"@ {exit_price:.4f} reason={exit_reason.value} "
            f"PnL=${position.realized_pnl_usd:+.2f} R={r_multiple:+.2f} "
            f"(today: ${self._realized_pnl_today:+.2f})"
        )

        # Bulgu 4: Log closed trade to JSONL
        try:
            self._trade_logger.log_trade_closed(position)
        except Exception as e:
            logger.warning(f"Trade log write failed: {e}")

        # Bulgu 4: Feed trade into performance tracker for strategy selection
        if position.strategy:
            try:
                self._perf_tracker.add_trade(
                    strategy=position.strategy,
                    pnl_r=r_multiple,
                    pnl_usd=position.realized_pnl_usd,
                    fees=position.fees_paid_usd,
                    funding=position.funding_paid_usd,
                    timestamp=int(position.exit_time.timestamp()) if position.exit_time else 0,
                )
            except Exception as e:
                logger.warning(f"PerformanceTracker update failed: {e}")

    # ------------------------------------------------------------------
    # Exit monitoring
    # ------------------------------------------------------------------

    def _check_position_exits(self, equity_usd: float) -> None:
        """
        Paper-mode SL/TP simulation with trailing stop.

        For each open position:
          1. Update trailing stop (Bulgu 1) using current ATR from feature engine
          2. Check SL hit (candle low for LONG / candle high for SHORT)
          3. Check TP hit
        """
        open_positions = self._state.get_open_positions()
        for position in open_positions:
            candles = self._candle_store.get_candles(position.symbol, "5m")
            if not candles:
                continue

            c = candles[-1]
            current_price = c.close

            # Bulgu 1: Update trailing stop before checking exits
            try:
                features = self._feature_engine.compute_features(position.symbol)
                if features and features.get("atr14"):
                    self._trailing_stop_manager.update_trailing_stop(
                        position=position,
                        current_price=current_price,
                        atr=float(features["atr14"]),
                    )
            except Exception as e:
                logger.debug(f"{position.symbol}: trailing stop update failed: {e}")

            # Determine exit reason (TRAIL if trailing stop was triggered)
            if position.side == ExecOrderSide.LONG:
                if c.low <= position.stop_price:
                    reason = ExitReason.TRAIL if position.trailing_enabled else ExitReason.SL
                    self._paper_fill_exit(position, position.stop_price, reason)
                elif position.tp_price and c.high >= position.tp_price:
                    self._paper_fill_exit(position, position.tp_price, ExitReason.TP)
            else:  # SHORT
                if c.high >= position.stop_price:
                    reason = ExitReason.TRAIL if position.trailing_enabled else ExitReason.SL
                    self._paper_fill_exit(position, position.stop_price, reason)
                elif position.tp_price and c.low <= position.tp_price:
                    self._paper_fill_exit(position, position.tp_price, ExitReason.TP)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_risk_pct(self) -> Optional[float]:
        """Return reduced risk pct during post-weekly-stop recovery, else None."""
        if self._kill_switch.is_reduced_risk_active():
            return self.config.risk.reduced_risk_after_pause_pct
        return None

    def _to_core_positions(
        self, positions: List[ExecPosition]
    ) -> List[CorePosition]:
        """
        Convert execution.position.Position list to core.types.Position list.

        RiskEngine (via RiskLimits/CorrelationFilter) only accesses
        .symbol, .side (CoreOrderSide), .notional, .risk_amount_usd proxy fields.
        """
        from datetime import datetime as _dt

        core = []
        for p in positions:
            core_side = (
                CoreOrderSide.LONG
                if p.side == ExecOrderSide.LONG
                else CoreOrderSide.SHORT
            )
            core.append(
                CorePosition(
                    symbol=p.symbol,
                    side=core_side,
                    entry_price=p.entry_price,
                    quantity=p.quantity,
                    notional=p.notional_usd,
                    leverage=p.leverage,
                    margin=p.margin_usd,
                    stop_price=p.stop_price,
                    tp_price=p.tp_price,
                    unrealized_pnl=p.unrealized_pnl_usd,
                    realized_pnl=p.realized_pnl_usd,
                    entry_time=p.entry_time,
                    trade_id=p.position_id,
                    status=CorePositionStatus.OPEN,
                )
            )
        return core

    def _build_exchange_client(self) -> BinanceFuturesClient:
        """Build exchange client from config + environment."""
        api_key = os.getenv(self.config.exchange.api_key_env, "")
        api_secret = os.getenv(self.config.exchange.api_secret_env, "")

        if not api_key or not api_secret:
            raise ValueError(
                f"API credentials missing. Set {self.config.exchange.api_key_env} "
                f"and {self.config.exchange.api_secret_env} in .env"
            )

        # Bulgu 1.2: Basic format validation
        from bot.exchange.binance_client import _validate_api_credentials
        _validate_api_credentials(api_key, api_secret)

        # Bulgu 9.2: Live trading requires explicit env var confirmation
        if not self.config.exchange.testnet:
            live_confirmed = os.getenv("LIVE_TRADING_CONFIRMED", "").strip().lower()
            if live_confirmed != "true":
                raise ValueError(
                    "LIVE TRADING IS NOT ENABLED.\n"
                    "To trade with real money, set the environment variable:\n"
                    "  LIVE_TRADING_CONFIRMED=true\n"
                    "This is a safety measure to prevent accidental live trading."
                )

        client = BinanceFuturesClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=self.config.exchange.testnet,
            recv_window_ms=self.config.exchange.recv_window_ms,
        )

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
