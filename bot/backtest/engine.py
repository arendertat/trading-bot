"""
Backtest engine — bar-by-bar simulation orchestrator.

Loads historical OHLCV data, replays candles in chronological order,
and runs the full production pipeline (features → regime → strategy →
risk → execution) against a simulated BacktestAccount.

Usage:
    engine = BacktestEngine(config, exchange)
    result = engine.run(
        symbols=["BTCUSDT", "ETHUSDT"],
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from bot.backtest.account import BacktestAccount, BacktestTrade, OpenBacktestPosition
from bot.backtest.data_provider import HistoricalDataProvider
from bot.config.models import BotConfig
from bot.core.constants import OrderSide, RegimeType
from bot.core.types import Candle
from bot.data.candle_store import CandleStore
from bot.data.feature_engine import FeatureEngine
from bot.regime.detector import RegimeDetector
from bot.risk.correlation_filter import CorrelationFilter
from bot.risk.kill_switch import KillSwitch
from bot.risk.position_sizing import PositionSizingCalculator
from bot.risk.risk_engine import RiskEngine
from bot.risk.risk_limits import RiskLimits
from bot.strategies.base import FeatureSet, Strategy
from bot.strategies.range_mean_reversion import RangeMeanReversionStrategy
from bot.strategies.trend_breakout import TrendBreakoutStrategy
from bot.strategies.trend_pullback import TrendPullbackStrategy
from bot.utils.jsonl_logger import JsonlLogger
from bot.utils.edge import compute_setup_quality, estimate_cost_gate, passes_cost_gate

logger = logging.getLogger("trading_bot.backtest.engine")

# Minimum bars in the rolling window before the engine starts trading.
# Enough for all indicators (ADX=14, EMA50, BB20, ATR14, z-score100).
_MIN_WARMUP_BARS_5M = 120   # 10 hours
_MIN_WARMUP_BARS_1H = 60    # 60 hours
_MIN_WARMUP_BARS_4H = 30    # 120 hours

# CandleStore rolling window sizes for backtest (larger than live to hold history)
_BT_STORE_LIMITS = {
    "5m": 400,
    "1h": 200,
    "4h": 120,
}

# Timeframe milliseconds
_TF_MS = {
    "5m":   300_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
}


@dataclass
class BacktestResult:
    """Summary output of a completed backtest run."""
    trades: List[BacktestTrade]
    equity_curve: List[Tuple[int, float]]
    initial_equity: float
    final_equity: float
    symbols: List[str]
    start: datetime
    end: datetime
    total_bars: int
    total_signals: int
    total_entries: int
    rejected_risk: int
    rejected_spread: int
    reject_reason_counts: Dict[str, int] = field(default_factory=dict)
    features_at_entry: Dict[str, dict] = field(default_factory=dict)
    data_quality: Dict[str, dict] = field(default_factory=dict)
    regime_bar_counts: Dict[str, int] = field(default_factory=dict)
    time_filter_min_samples: int = 0
    time_filter_avg_r_threshold: float = 0.0


class BacktestEngine:
    """
    Bar-by-bar backtest engine.

    Reuses all production components without modification:
    - FeatureEngine  (technical indicators)
    - RegimeDetector (market regime)
    - Strategy classes (signal generation)
    - RiskEngine     (position sizing + kill switch)
    - BacktestAccount (simulated fills, PnL, equity)
    """

    def __init__(
        self,
        config: BotConfig,
        exchange,  # ccxt exchange instance for historical data
        initial_equity: float = 10_000.0,
    ) -> None:
        self._config = config
        self._exchange = exchange
        self._initial_equity = initial_equity

        # Build production components
        self._store = CandleStore(custom_limits=_BT_STORE_LIMITS)
        self._feature_engine = FeatureEngine(
            self._store,
            config.timeframes,
            config.regime,
            config.strategies,
        )
        self._regime_detector = RegimeDetector(config.regime)

        # Risk stack
        self._kill_switch = KillSwitch(config.risk)
        self._position_sizing = PositionSizingCalculator(config)
        self._risk_limits = RiskLimits(config.risk)
        self._corr_filter = CorrelationFilter(config.risk)
        self._risk_engine = RiskEngine(
            config=config,
            kill_switch=self._kill_switch,
            position_sizing=self._position_sizing,
            risk_limits=self._risk_limits,
            correlation_filter=self._corr_filter,
        )

        # Strategies
        strat_cfg = config.strategies
        def _with_risk_defaults(cfg: dict) -> dict:
            cfg = dict(cfg)
            cfg.setdefault("min_stop_pct", config.risk.min_stop_pct)
            cfg.setdefault("min_stop_usd", config.risk.min_stop_usd)
            return cfg

        self._strategies: List[Strategy] = []
        if strat_cfg.trend_pullback.enabled:
            self._strategies.append(
                TrendPullbackStrategy(_with_risk_defaults(strat_cfg.trend_pullback.model_dump()))
            )
        if strat_cfg.trend_breakout.enabled:
            self._strategies.append(
                TrendBreakoutStrategy(_with_risk_defaults(strat_cfg.trend_breakout.model_dump()))
            )
        if strat_cfg.range_mean_reversion.enabled:
            self._strategies.append(
                RangeMeanReversionStrategy(_with_risk_defaults(strat_cfg.range_mean_reversion.model_dump()))
            )

        # Simulated account
        self._account = BacktestAccount(
            initial_equity=initial_equity,
            maker_fee_pct=config.execution.maker_fee_pct,
            taker_fee_pct=config.execution.taker_fee_pct,
            slippage_pct=config.execution.paper_slippage_limit_pct,
        )

        logger.info(
            f"BacktestEngine initialised: "
            f"equity={initial_equity:,.0f} USD, "
            f"strategies={[type(s).__name__ for s in self._strategies]}"
        )
        log_dir = config.logging.log_dir
        self._regime_log = JsonlLogger(f"{log_dir.rstrip('/')}/regime_decisions.jsonl")
        self._strategy_log = JsonlLogger(f"{log_dir.rstrip('/')}/strategy_selection.jsonl")

    # ── Public API ─────────────────────────────────────────────────────

    def run(
        self,
        symbols: List[str],
        start: datetime,
        end: Optional[datetime] = None,
    ) -> BacktestResult:
        """
        Run a full backtest over the given symbols and date range.

        1. Fetches historical data from Binance (5m + 1h + 4h).
        2. Builds a sorted timeline of 5m bars across all symbols.
        3. For each bar: feeds candles → computes features → generates signals
           → validates risk → opens/closes simulated positions.

        Returns a BacktestResult with all closed trades and the equity curve.
        """
        if end is None:
            end = datetime.now(timezone.utc)

        logger.info(
            f"Backtest start: symbols={symbols}, "
            f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}"
        )

        # ── Step 1: Fetch historical data into an unlimited store ─────
        # CandleStore uses a deque with a max size, so we use a very large
        # limit for the fetch store to hold all historical bars.
        days = max((end - start).days, 1)
        fetch_store = CandleStore(custom_limits={
            "5m":  days * 288 + 500,   # 288 5m bars per day + buffer
            "1h":  days * 24  + 100,
            "4h":  days * 6   + 50,
        })
        provider = HistoricalDataProvider(self._exchange, fetch_store)
        timeframes = ["5m", "1h", "4h"]
        for symbol in symbols:
            for tf in timeframes:
                count = provider.fetch(symbol, tf, start, end)
                logger.info(f"  {symbol} {tf}: {count} candles")

        data_quality = _compute_data_quality(fetch_store, symbols, timeframes, _TF_MS)

        # ── Step 2: Build 5m timeline ──────────────────────────────────
        # Collect all unique 5m timestamps (sorted)
        all_5m_candles: List[Tuple[int, str, Candle]] = []
        for symbol in symbols:
            candles = fetch_store.get_candles(symbol, "5m")
            for c in candles:
                all_5m_candles.append((c.timestamp, symbol, c))

        all_5m_candles.sort(key=lambda x: x[0])

        total_bars = len(all_5m_candles)
        logger.info(f"Timeline: {total_bars} total 5m bars across {len(symbols)} symbols")

        # ── Step 3: Build replay store (rolling window for indicators) ──
        # The replay store is a fresh rolling-window store fed bar-by-bar.
        # 1h and 4h candles are pre-sorted for fast chronological feeding.
        replay_store = CandleStore(custom_limits=_BT_STORE_LIMITS)
        feature_engine = FeatureEngine(
            replay_store,
            self._config.timeframes,
            self._config.regime,
            self._config.strategies,
        )

        # Pre-load 1h and 4h candles from the full fetch_store
        tf1h_candles: Dict[str, List[Candle]] = {}
        tf4h_candles: Dict[str, List[Candle]] = {}
        for symbol in symbols:
            tf1h_candles[symbol] = sorted(
                fetch_store.get_candles(symbol, "1h"), key=lambda c: c.timestamp
            )
            tf4h_candles[symbol] = sorted(
                fetch_store.get_candles(symbol, "4h"), key=lambda c: c.timestamp
            )

        # ── Step 4: Bar-by-bar simulation ──────────────────────────────
        total_signals = 0
        total_entries = 0
        rejected_risk = 0
        rejected_spread = 0
        reject_reason_counts = {
            "COST_GATE": 0,
            "CHOP_GATE": 0,
            "SPREAD_GATE": 0,
            "RISK_BLOCK": 0,
            "COOLDOWN": 0,
            "INSUFFICIENT_CONFIDENCE": 0,
            "INSUFFICIENT_MARGIN": 0,
        }
        features_at_entry: Dict[str, dict] = {}
        regime_bar_counts: Dict[str, int] = {}

        # Track which 1h/4h candles have been loaded up to the current 5m bar
        tf1h_idx: Dict[str, int] = {s: 0 for s in symbols}
        tf4h_idx: Dict[str, int] = {s: 0 for s in symbols}

        prev_ts_ms = 0

        last_progress = -1
        for i, (bar_ts, symbol, candle_5m) in enumerate(all_5m_candles, start=1):
            if total_bars > 0:
                progress = int((i / total_bars) * 100)
                if progress != last_progress and progress % 5 == 0:
                    print(f"[BACKTEST] {progress}% ({i}/{total_bars})")
                    last_progress = progress
            now = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc)

            # Day/week rollover for account PnL tracking
            self._account.tick_day(now)

            # Feed any 1h candles whose timestamp <= bar_ts
            idx = tf1h_idx[symbol]
            candles_1h = tf1h_candles[symbol]
            while idx < len(candles_1h) and candles_1h[idx].timestamp <= bar_ts:
                replay_store.add_candle(symbol, "1h", candles_1h[idx])
                idx += 1
            tf1h_idx[symbol] = idx

            # Feed any 4h candles whose timestamp <= bar_ts
            idx = tf4h_idx[symbol]
            candles_4h = tf4h_candles[symbol]
            while idx < len(candles_4h) and candles_4h[idx].timestamp <= bar_ts:
                replay_store.add_candle(symbol, "4h", candles_4h[idx])
                idx += 1
            tf4h_idx[symbol] = idx

            # Feed the 5m candle
            replay_store.add_candle(symbol, "5m", candle_5m)

            # Snapshot equity once per unique timestamp
            if bar_ts != prev_ts_ms:
                self._account.snapshot_equity(bar_ts)
                self._risk_engine.tick_cooldowns()
                prev_ts_ms = bar_ts

            current_price = candle_5m.close

            # ── Check existing position exits ────────────────────────
            self._check_exits(symbol, candle_5m, now)

            # ── Feature computation ──────────────────────────────────
            if not replay_store.has_enough_data(symbol, "5m", _MIN_WARMUP_BARS_5M):
                continue

            raw_features = feature_engine.compute_features(symbol)
            if raw_features is None:
                continue

            feature_set = _build_feature_set(raw_features, current_price)

            # ── Regime detection ─────────────────────────────────────
            regime_result = self._regime_detector.detect_regime(
                symbol=symbol,
                adx=raw_features.get("adx14", 0.0),
                atr_z=raw_features.get("atr_z", 0.0),
                bb_width=raw_features.get("bb_width", 0.03),
                ema20_5m=raw_features.get("ema20_5m", current_price),
                ema50_5m=raw_features.get("ema50_5m", current_price),
                ema20_1h=raw_features.get("ema20_1h", current_price),
                ema50_1h=raw_features.get("ema50_1h", current_price),
                kaufman_er=raw_features.get("kaufman_er"),
                flip_rate=raw_features.get("flip_rate"),
                choppiness=raw_features.get("choppiness"),
                adx_momentum=raw_features.get("adx_momentum"),
                ema20_1h_slope=raw_features.get("ema20_1h_slope"),
                ema1h_spread_pct=raw_features.get("ema1h_spread_pct"),
                bb_width_pct_rank=raw_features.get("bb_width_pct_rank"),
                rsi_5m=raw_features.get("rsi14"),
                bb_upper=raw_features.get("bb_upper"),
                bb_lower=raw_features.get("bb_lower"),
                last_close_5m=current_price,
                rsi_extreme_low=self._config.strategies.range_mean_reversion.rsi_long_extreme,
                rsi_extreme_high=self._config.strategies.range_mean_reversion.rsi_short_extreme,
            )

            # ── Regime decision log (per bar) ────────────────────────
            spread_pct = 0.0  # Backtest uses no order book; spread gate is a no-op here
            spread_ok = True
            regime_bar_counts[regime_result.regime.value] = (
                regime_bar_counts.get(regime_result.regime.value, 0) + 1
            )
            self._regime_log.log(
                {
                    "ts": bar_ts,
                    "symbol": symbol,
                    "regime": regime_result.regime.value,
                    "confidence": regime_result.confidence,
                    "inputs": {
                        "adx_5m": raw_features.get("adx14"),
                        "atr_z_5m": raw_features.get("atr_z"),
                        "bb_width_5m": raw_features.get("bb_width"),
                        "rsi_5m": raw_features.get("rsi14"),
                        "ema20_1h": raw_features.get("ema20_1h"),
                        "ema50_1h": raw_features.get("ema50_1h"),
                        "trend_direction": regime_result.trend_direction or "flat",
                    },
                    "microstructure": {
                        "spread_pct": spread_pct,
                        "funding_rate": 0.0,
                        "spread_ok": spread_ok,
                        "funding_ok": True,
                    },
                    "scores": {
                        "trend_score": regime_result.trend_score,
                        "range_score": regime_result.range_score,
                        "high_vol_score": regime_result.high_vol_score,
                        "chop_score": regime_result.chop_score,
                    },
                    "reasons": regime_result.reasons,
                },
                ts=now,
            )

            if regime_result.regime == RegimeType.CHOP_NO_TRADE:
                if regime_result.gate_reason in ("CHOP_GATE", "TREND_SCORE"):
                    reject_reason_counts["CHOP_GATE"] += 1
                elif any("Low confidence" in r for r in regime_result.reasons):
                    reject_reason_counts["INSUFFICIENT_CONFIDENCE"] += 1
                continue

            # ── Time filters (entry only) ───────────────────────────
            bad_hours = set(self._config.time_filters.bad_hours)
            bad_weekdays = set(self._config.time_filters.bad_weekdays)
            if now.hour in bad_hours or now.strftime("%a") in bad_weekdays:
                reject_reason_counts["COOLDOWN"] += 1
                continue

            # ── Signal generation ────────────────────────────────────
            # Skip if symbol already has an open position
            open_ids = [p.trade_id for p in self._account.open_positions if p.symbol == symbol]
            if open_ids:
                # Still update trailing stops for open positions
                self._update_trailing_stops(symbol, candle_5m, raw_features)
                continue

            # ── Entry-time spread gate ──────────────────────────────
            if not spread_ok:
                rejected_spread += 1
                reject_reason_counts["SPREAD_GATE"] += 1
                self._strategy_log.log(
                    {
                        "ts": bar_ts,
                        "symbol": symbol,
                        "regime": regime_result.regime.value,
                        "candidates": [type(s).__name__ for s in self._strategies],
                        "rejected": [{"name": "ALL", "reason": "spread_gate"}],
                        "selected": None,
                        "reason": "SPREAD_GATE",
                    },
                    ts=now,
                )
                continue

            strategy_evaluations = []
            for strategy in self._strategies:
                # Check regime compatibility
                if regime_result.regime not in strategy.compatible_regimes:
                    strategy_evaluations.append({
                        "name": type(strategy).__name__,
                        "reason": "incompatible_regime",
                    })
                    continue

                signal = strategy.generate_signal(
                    features=feature_set,
                    regime_result=regime_result,
                    symbol=symbol,
                    current_price=current_price,
                    timestamp=bar_ts,
                )

                if signal is None or not signal.entry:
                    strategy_evaluations.append({
                        "name": type(strategy).__name__,
                        "reason": "no_signal",
                    })
                    continue

                total_signals += 1

                # ── Risk validation ──────────────────────────────────
                core_positions = self._account.get_core_positions(
                    {symbol: current_price}
                )

                # Update kill switch with current PnL
                self._kill_switch.update_pnl(
                    realized_pnl_today=self._account.daily_pnl,
                    realized_pnl_week=self._account.weekly_pnl,
                    equity_usd=self._account.equity,
                    now_utc=now,
                )

                risk_result = self._risk_engine.validate_entry(
                    symbol=symbol,
                    side=signal.side,
                    regime=regime_result.regime,
                    stop_pct=signal.stop_pct,
                    current_price=current_price,
                    equity_usd=self._account.equity,
                    free_margin_usd=self._account.free_margin,
                    open_positions=core_positions,
                )

                if not risk_result.approved:
                    rejected_risk += 1
                    if "Cooldown active" in risk_result.rejection_reason:
                        reject_reason_counts["COOLDOWN"] += 1
                    elif "Insufficient margin" in risk_result.rejection_reason or "INSUFFICIENT_MARGIN" in risk_result.rejection_reason:
                        reject_reason_counts["INSUFFICIENT_MARGIN"] += 1
                    else:
                        reject_reason_counts["RISK_BLOCK"] += 1
                    logger.debug(
                        f"{symbol}: entry rejected — {risk_result.rejection_reason}"
                    )
                    strategy_evaluations.append({
                        "name": type(strategy).__name__,
                        "reason": f"risk_reject:{risk_result.rejection_reason}",
                    })
                    continue

                ps = risk_result.position_size
                if ps is None or not ps.approved:
                    rejected_risk += 1
                    reject_reason_counts["RISK_BLOCK"] += 1
                    strategy_evaluations.append({
                        "name": type(strategy).__name__,
                        "reason": "position_sizing_reject",
                    })
                    continue

                # ── Cost-aware gate ────────────────────────────────
                setup_quality = compute_setup_quality(feature_set, signal.side, self._config.cost_gate)
                cost_gate = estimate_cost_gate(
                    risk_usd=ps.risk_usd,
                    notional_usd=ps.notional_usd,
                    config=self._config.cost_gate,
                    setup_quality_score=setup_quality,
                )
                if not passes_cost_gate(cost_gate, self._config.gates):
                    reject_reason_counts["COST_GATE"] += 1
                    strategy_evaluations.append({
                        "name": type(strategy).__name__,
                        "reason": "cost_gate",
                    })
                    continue

                # ── Open position ────────────────────────────────────
                pos = self._account.open_position(
                    symbol=symbol,
                    side=signal.side,
                    entry_price_raw=current_price,
                    quantity=ps.quantity,
                    notional_usd=ps.notional_usd,
                    margin_usd=ps.margin_required_usd,
                    risk_usd=ps.risk_usd,
                    leverage_used=ps.leverage,
                    stop_price=signal.stop_price,
                    tp_price=signal.tp_price,
                    entry_time=now,
                    strategy=type(strategy).__name__,
                    regime=regime_result.regime.value,
                    trail_after_r=signal.trail_after_r,
                    atr_trail_mult=signal.atr_trail_mult,
                    regime_confidence=regime_result.confidence,
                    estimated_cost_usd=cost_gate.estimated_cost_usd,
                    estimated_cost_r=cost_gate.estimated_cost_r,
                    expected_edge_r=cost_gate.expected_edge_r,
                    setup_quality_score=cost_gate.setup_quality_score,
                )

                if pos is not None:
                    pos.trail_enabled = signal.trail_enabled
                    total_entries += 1
                    self._strategy_log.log(
                        {
                            "ts": bar_ts,
                            "symbol": symbol,
                            "regime": regime_result.regime.value,
                            "candidates": [type(s).__name__ for s in self._strategies],
                            "rejected": strategy_evaluations,
                            "selected": type(strategy).__name__,
                            "reason": "selected",
                        },
                        ts=now,
                    )
                    features_at_entry[pos.trade_id] = {
                        "symbol": symbol,
                        "timestamp": bar_ts,
                        "strategy": type(strategy).__name__,
                        "regime": regime_result.regime.value,
                        "regime_confidence": regime_result.confidence,
                        "side": signal.side.value if signal.side else None,
                        "price": current_price,
                        "cost_gate": {
                            "estimated_cost_usd": cost_gate.estimated_cost_usd,
                            "estimated_cost_r": cost_gate.estimated_cost_r,
                            "expected_edge_r": cost_gate.expected_edge_r,
                            "setup_quality_score": cost_gate.setup_quality_score,
                        },
                        "signal": {
                            "reason": signal.reason,
                            "stop_pct": signal.stop_pct,
                            "target_r": signal.target_r,
                            "stop_price": signal.stop_price,
                            "tp_price": signal.tp_price,
                            "trail_enabled": signal.trail_enabled,
                            "trail_after_r": signal.trail_after_r,
                            "atr_trail_mult": signal.atr_trail_mult,
                        },
                        "features": asdict(feature_set),
                    }
                    logger.debug(
                        f"{symbol}: OPEN {signal.side.value} @ {current_price:.4f} "
                        f"SL={signal.stop_price:.4f} TP={signal.tp_price:.4f} "
                        f"risk={ps.risk_usd:.2f} USD "
                        f"[{type(strategy).__name__}]"
                    )
                    # Only take first qualifying signal per bar
                    break
            else:
                if strategy_evaluations:
                    self._strategy_log.log(
                        {
                            "ts": bar_ts,
                            "symbol": symbol,
                            "regime": regime_result.regime.value,
                            "candidates": [type(s).__name__ for s in self._strategies],
                            "rejected": strategy_evaluations,
                            "selected": None,
                            "reason": "no_strategy_selected",
                        },
                        ts=now,
                    )

        # ── Step 5: Close all open positions at end of data ───────────
        last_ts = all_5m_candles[-1][0] if all_5m_candles else 0
        last_time = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        self._close_all_eod(last_time)

        final_equity = self._account.equity
        logger.info(
            f"Backtest complete: {len(self._account.closed_trades)} trades, "
            f"equity {self._initial_equity:,.2f} → {final_equity:,.2f} USD "
            f"({(final_equity/self._initial_equity - 1)*100:+.1f}%)"
        )

        return BacktestResult(
            trades=self._account.closed_trades,
            equity_curve=self._account.equity_curve,
            initial_equity=self._initial_equity,
            final_equity=final_equity,
            symbols=symbols,
            start=start,
            end=end,
            total_bars=total_bars,
            total_signals=total_signals,
            total_entries=total_entries,
            rejected_risk=rejected_risk,
            rejected_spread=rejected_spread,
            reject_reason_counts=reject_reason_counts,
            features_at_entry=features_at_entry,
            data_quality=data_quality,
            regime_bar_counts=regime_bar_counts,
            time_filter_min_samples=self._config.time_filters.min_samples,
            time_filter_avg_r_threshold=self._config.time_filters.avg_r_threshold,
        )

    # ── Internal helpers ────────────────────────────────────────────────

    def _check_exits(
        self,
        symbol: str,
        candle: Candle,
        now: datetime,
    ) -> None:
        """
        Check SL / TP / trailing-stop hits for all open positions on this symbol.

        Uses the candle's high/low to simulate intra-bar hits:
        - LONG SL: low <= stop_price
        - LONG TP: high >= tp_price
        - SHORT SL: high >= stop_price
        - SHORT TP: low <= tp_price
        """
        to_close: List[Tuple[str, float, str]] = []  # (trade_id, exit_price, reason)

        for pos in list(self._account.open_positions):
            if pos.symbol != symbol:
                continue

            # Update MFE/MAE
            self._account.update_mfe_mae(pos.trade_id, candle.close, candle.timestamp)

            # Effective stop: use trailing stop if active, else initial stop
            effective_stop = pos.trailing_stop if pos.trailing_stop is not None else pos.stop_price
            stop_reason = "TRAIL" if pos.trailing_stop is not None else "SL"

            if pos.side == OrderSide.LONG:
                # SL hit
                if candle.low <= effective_stop:
                    to_close.append((pos.trade_id, effective_stop, stop_reason))
                # TP hit
                elif pos.tp_price is not None and candle.high >= pos.tp_price:
                    to_close.append((pos.trade_id, pos.tp_price, "TP"))
            else:  # SHORT
                # SL hit
                if candle.high >= effective_stop:
                    to_close.append((pos.trade_id, effective_stop, stop_reason))
                # TP hit
                elif pos.tp_price is not None and candle.low <= pos.tp_price:
                    to_close.append((pos.trade_id, pos.tp_price, "TP"))

        for trade_id, exit_price, reason in to_close:
            trade = self._account.close_position(
                trade_id=trade_id,
                exit_price_raw=exit_price,
                exit_time=now,
                exit_reason=reason,
                is_market=(reason in ("SL", "TRAIL")),  # SL/TRAIL → market fill; TP → limit
            )
            if trade:
                if reason in ("SL", "TRAIL"):
                    self._risk_engine.record_sl_exit(trade.symbol)
                pnl_str = f"{trade.pnl_usd:+.2f} USD ({trade.pnl_r:+.2f}R)"
                logger.debug(
                    f"{symbol}: CLOSE {reason} {trade.side} @ {exit_price:.4f} {pnl_str}"
                )

    def _update_trailing_stops(
        self,
        symbol: str,
        candle: Candle,
        raw_features: dict,
    ) -> None:
        """Update trailing stop prices for open positions on this symbol."""
        atr = raw_features.get("atr14", 0.0)
        for pos in self._account.open_positions:
            if pos.symbol != symbol or not pos.trail_enabled:
                continue
            r = pos.r_multiple(candle.close)
            if r < pos.trail_after_r:
                continue  # Not yet at activation threshold
            # Compute new trailing stop distance = atr * multiplier
            trail_dist = atr * pos.atr_trail_mult
            if pos.side == OrderSide.LONG:
                new_stop = candle.close - trail_dist
                if pos.trailing_stop is None or new_stop > pos.trailing_stop:
                    pos.trailing_stop = new_stop
            else:
                new_stop = candle.close + trail_dist
                if pos.trailing_stop is None or new_stop < pos.trailing_stop:
                    pos.trailing_stop = new_stop

    def _close_all_eod(self, now: datetime) -> None:
        """Close all remaining open positions at the last available price (EOD)."""
        for pos in list(self._account.open_positions):
            self._account.close_position(
                trade_id=pos.trade_id,
                exit_price_raw=pos.entry_price,  # use entry price as last known
                exit_time=now,
                exit_reason="EOD",
                is_market=True,
            )


# ── Helpers ─────────────────────────────────────────────────────────────

def _build_feature_set(raw: dict, current_price: float) -> FeatureSet:
    """Convert raw feature dict to FeatureSet dataclass."""
    return FeatureSet(
        rsi_5m=raw.get("rsi14", 50.0),
        ema20_5m=raw.get("ema20_5m", current_price),
        ema50_5m=raw.get("ema50_5m", current_price),
        ema20_1h=raw.get("ema20_1h", current_price),
        ema50_1h=raw.get("ema50_1h", current_price),
        atr_5m=raw.get("atr14", 0.0),
        atr_by_lookback=raw.get("atr_by_lookback"),
        bb_upper_5m=raw.get("bb_upper", current_price * 1.02),
        bb_lower_5m=raw.get("bb_lower", current_price * 0.98),
        bb_middle_5m=raw.get("bb_middle", current_price),
        high_20_bars=raw.get("high_20", current_price),
        low_20_bars=raw.get("low_20", current_price),
        volume_z_5m=raw.get("vol_z", 0.0),
        adx_5m=raw.get("adx14"),
        atr_z_5m=raw.get("atr_z"),
        bb_width_5m=raw.get("bb_width"),
        ema20_4h=raw.get("ema20_4h"),
        ema50_4h=raw.get("ema50_4h"),
        book_imbalance_ratio=None,  # Not available in backtest
    )


def _compute_data_quality(
    store: CandleStore,
    symbols: List[str],
    timeframes: List[str],
    tf_ms_map: Dict[str, int],
) -> Dict[str, dict]:
    """
    Compute simple data-quality diagnostics from fetched candles.

    Returns:
        Dict with bars per symbol/timeframe, gap counts, and missing bar estimates.
    """
    bars_total = 0
    total_gaps = 0
    total_missing = 0
    bars_per_symbol: Dict[str, Dict[str, int]] = {}
    gaps_detected: Dict[str, Dict[str, int]] = {}
    missing_bars: Dict[str, Dict[str, int]] = {}

    for symbol in symbols:
        bars_per_symbol[symbol] = {}
        gaps_detected[symbol] = {}
        missing_bars[symbol] = {}
        for tf in timeframes:
            candles = store.get_candles(symbol, tf)
            bars = len(candles)
            bars_total += bars
            bars_per_symbol[symbol][tf] = bars

            gaps = 0
            missing = 0
            step = tf_ms_map.get(tf, 0)
            if step > 0 and bars > 1:
                prev_ts = candles[0].timestamp
                for c in candles[1:]:
                    diff = c.timestamp - prev_ts
                    if diff > step:
                        gap_bars = diff // step - 1
                        if gap_bars > 0:
                            gaps += 1
                            missing += int(gap_bars)
                    prev_ts = c.timestamp

            gaps_detected[symbol][tf] = gaps
            missing_bars[symbol][tf] = missing
            total_gaps += gaps
            total_missing += missing

    return {
        "bars_total": bars_total,
        "bars_per_symbol": bars_per_symbol,
        "gaps_detected": gaps_detected,
        "missing_bars": missing_bars,
        "total_gaps": total_gaps,
        "total_missing_bars": total_missing,
        "timeframes": timeframes,
    }
