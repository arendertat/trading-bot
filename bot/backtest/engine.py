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
from dataclasses import dataclass, field
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
        self._feature_engine = FeatureEngine(self._store, config.timeframes)
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
        self._strategies: List[Strategy] = []
        if strat_cfg.trend_pullback.enabled:
            self._strategies.append(
                TrendPullbackStrategy(strat_cfg.trend_pullback.model_dump())
            )
        if strat_cfg.trend_breakout.enabled:
            self._strategies.append(
                TrendBreakoutStrategy(strat_cfg.trend_breakout.model_dump())
            )
        if strat_cfg.range_mean_reversion.enabled:
            self._strategies.append(
                RangeMeanReversionStrategy(strat_cfg.range_mean_reversion.model_dump())
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
        for symbol in symbols:
            for tf in ["5m", "1h", "4h"]:
                count = provider.fetch(symbol, tf, start, end)
                logger.info(f"  {symbol} {tf}: {count} candles")

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
        feature_engine = FeatureEngine(replay_store, self._config.timeframes)

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

        # Track which 1h/4h candles have been loaded up to the current 5m bar
        tf1h_idx: Dict[str, int] = {s: 0 for s in symbols}
        tf4h_idx: Dict[str, int] = {s: 0 for s in symbols}

        prev_ts_ms = 0

        for bar_ts, symbol, candle_5m in all_5m_candles:
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
            )

            if regime_result.regime == RegimeType.CHOP_NO_TRADE:
                continue

            # ── Signal generation ────────────────────────────────────
            # Skip if symbol already has an open position
            open_ids = [p.trade_id for p in self._account.open_positions if p.symbol == symbol]
            if open_ids:
                # Still update trailing stops for open positions
                self._update_trailing_stops(symbol, candle_5m, raw_features)
                continue

            for strategy in self._strategies:
                # Check regime compatibility
                if regime_result.regime not in strategy.compatible_regimes:
                    continue

                signal = strategy.generate_signal(
                    features=feature_set,
                    regime_result=regime_result,
                    symbol=symbol,
                    current_price=current_price,
                    timestamp=bar_ts,
                )

                if signal is None or not signal.entry:
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
                    logger.debug(
                        f"{symbol}: entry rejected — {risk_result.rejection_reason}"
                    )
                    continue

                ps = risk_result.position_size
                if ps is None or not ps.approved:
                    rejected_risk += 1
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
                    stop_price=signal.stop_price,
                    tp_price=signal.tp_price,
                    entry_time=now,
                    strategy=type(strategy).__name__,
                    regime=regime_result.regime.value,
                    trail_after_r=signal.trail_after_r,
                    atr_trail_mult=signal.atr_trail_mult,
                )

                if pos is not None:
                    pos.trail_enabled = signal.trail_enabled
                    total_entries += 1
                    logger.debug(
                        f"{symbol}: OPEN {signal.side.value} @ {current_price:.4f} "
                        f"SL={signal.stop_price:.4f} TP={signal.tp_price:.4f} "
                        f"risk={ps.risk_usd:.2f} USD "
                        f"[{type(strategy).__name__}]"
                    )
                    # Only take first qualifying signal per bar
                    break

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
            self._account.update_mfe_mae(pos.trade_id, candle.close)

            # Effective stop: use trailing stop if active, else initial stop
            effective_stop = pos.trailing_stop if pos.trailing_stop is not None else pos.stop_price

            if pos.side == OrderSide.LONG:
                # SL hit
                if candle.low <= effective_stop:
                    to_close.append((pos.trade_id, effective_stop, "SL"))
                # TP hit
                elif pos.tp_price is not None and candle.high >= pos.tp_price:
                    to_close.append((pos.trade_id, pos.tp_price, "TP"))
            else:  # SHORT
                # SL hit
                if candle.high >= effective_stop:
                    to_close.append((pos.trade_id, effective_stop, "SL"))
                # TP hit
                elif pos.tp_price is not None and candle.low <= pos.tp_price:
                    to_close.append((pos.trade_id, pos.tp_price, "TP"))

        for trade_id, exit_price, reason in to_close:
            trade = self._account.close_position(
                trade_id=trade_id,
                exit_price_raw=exit_price,
                exit_time=now,
                exit_reason=reason,
                is_market=(reason == "SL"),  # SL → market fill; TP → limit
            )
            if trade:
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
