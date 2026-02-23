"""Feature engine for computing technical indicator snapshots"""

import logging
from typing import Optional, Dict
import pandas as pd

from bot.data.candle_store import CandleStore
from bot.data import features
from bot.config.models import TimeframesConfig, RegimeConfig, StrategiesConfig


logger = logging.getLogger("trading_bot.data")


class FeatureEngine:
    """
    Computes technical indicator snapshots from CandleStore.

    Returns latest feature values per symbol based on stored candle history.
    """

    def __init__(
        self,
        candle_store: CandleStore,
        timeframes_config: TimeframesConfig,
        regime_config: Optional[RegimeConfig] = None,
        strategies_config: Optional[StrategiesConfig] = None,
    ):
        """
        Initialize FeatureEngine.

        Args:
            candle_store: CandleStore instance
            timeframes_config: Timeframes configuration
        """
        self.candle_store = candle_store
        self.config = timeframes_config
        self.regime_config = regime_config
        self.strategies_config = strategies_config

        logger.info("FeatureEngine initialized")

    def compute_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Compute latest feature snapshot for a symbol.

        Computes:
        - RSI(14) on 5m
        - ADX(14) on 5m
        - ATR(14) on 5m
        - EMA(20), EMA(50) on 5m
        - EMA(20), EMA(50) on 1h
        - Bollinger Band width on 5m
        - Volume z-score on 5m
        - ATR z-score on 5m
        - Kaufman ER on 5m (configurable)
        - Flip rate on 5m (configurable)
        - EMA20/EMA50 spread pct on 1h
        - BB width percentile rank (optional, configurable)

        Args:
            symbol: Trading pair

        Returns:
            Feature dict or None if insufficient data
        """
        # Minimum bars needed for all indicators
        # ADX needs 2*period, z-score needs window, so check for max requirement
        min_bars_5m = max(
            14 * 2,  # ADX needs 2x period
            50,  # EMA50
            self.config.zscore_lookback
        )
        atr_lookbacks = {14}
        if self.strategies_config is not None:
            try:
                atr_lookbacks.update([
                    self.strategies_config.trend_pullback.stop_atr_lookback,
                    self.strategies_config.trend_breakout.stop_atr_lookback,
                    self.strategies_config.range_mean_reversion.stop_atr_lookback,
                ])
            except Exception:
                pass
        if atr_lookbacks:
            min_bars_5m = max(min_bars_5m, max(atr_lookbacks) + 1)
        trend_cfg = None
        if self.regime_config is not None:
            chop_cfg = self.regime_config.chop
            trend_cfg = self.regime_config.trend_score
            min_bars_5m = max(
                min_bars_5m,
                chop_cfg.er_lookback + 1,
                chop_cfg.flip_lookback + 1,
                chop_cfg.bb_width_percentile_lookback,
                chop_cfg.choppiness_lookback + 1,
                trend_cfg.adx_momentum_lookback + 1,
            )
        min_bars_1h = 50  # For EMA50 on 1h
        if self.regime_config is not None:
            trend_cfg = self.regime_config.trend_score
            min_bars_1h = max(min_bars_1h, trend_cfg.ema_slope_lookback_1h + 1)

        # Check if we have enough data
        if not self.candle_store.has_enough_data(symbol, "5m", min_bars_5m):
            logger.debug(f"Insufficient 5m data for {symbol}")
            return None

        if not self.candle_store.has_enough_data(symbol, "1h", min_bars_1h):
            logger.debug(f"Insufficient 1h data for {symbol}")
            return None

        # Get candles
        candles_5m = self.candle_store.get_candles(symbol, "5m")
        candles_1h = self.candle_store.get_candles(symbol, "1h")

        # 4h candles — optional for multi-TF confirmation (Özellik 6)
        # Requires at least 50 bars; gracefully absent during warmup
        candles_4h = (
            self.candle_store.get_candles(symbol, "4h")
            if self.candle_store.has_enough_data(symbol, "4h", 50)
            else None
        )

        # Convert to pandas DataFrames
        df_5m = self._candles_to_df(candles_5m)
        df_1h = self._candles_to_df(candles_1h)

        # Compute indicators (5m)
        rsi_series = features.rsi(df_5m['close'], period=14)
        adx_series = features.adx(df_5m['high'], df_5m['low'], df_5m['close'], period=14)
        atr_series_by_lb = {
            lb: features.atr(df_5m['high'], df_5m['low'], df_5m['close'], period=lb)
            for lb in sorted(atr_lookbacks)
        }
        atr_series = atr_series_by_lb.get(14)
        ema20_5m_series = features.ema(df_5m['close'], period=20)
        ema50_5m_series = features.ema(df_5m['close'], period=50)

        bb_result = features.bollinger_bands(df_5m['close'], period=20, std=2.0)

        vol_z_series = features.zscore(df_5m['volume'], window=self.config.zscore_lookback)
        atr_z_series = features.zscore(atr_series, window=self.config.zscore_lookback) if atr_series is not None else None

        # Compute indicators (1h)
        ema20_1h_series = features.ema(df_1h['close'], period=20)
        ema50_1h_series = features.ema(df_1h['close'], period=50)

        # Compute indicators (4h) — None if insufficient data
        ema20_4h_val = None
        ema50_4h_val = None
        if candles_4h:
            df_4h = self._candles_to_df(candles_4h)
            ema20_4h_series = features.ema(df_4h['close'], period=20)
            ema50_4h_series = features.ema(df_4h['close'], period=50)
            try:
                ema20_4h_val = float(ema20_4h_series.iloc[-1]) if ema20_4h_series is not None else None
                ema50_4h_val = float(ema50_4h_series.iloc[-1]) if ema50_4h_series is not None else None
            except (IndexError, TypeError):
                pass

        # CHOP / trend features (5m/1h)
        kaufman_er_val = None
        flip_rate_val = None
        bb_width_pct_rank_val = None
        choppiness_val = None
        adx_momentum_val = None
        ema20_1h_slope_val = None
        if self.regime_config is not None:
            chop_cfg = self.regime_config.chop
            trend_cfg = self.regime_config.trend_score
            if len(df_5m['close']) >= chop_cfg.er_lookback + 1:
                kaufman_er_val = features.kaufman_er(
                    df_5m['close'], lookback=chop_cfg.er_lookback
                )
            if len(df_5m['close']) >= chop_cfg.flip_lookback + 1:
                flip_rate_val = features.flip_rate(
                    df_5m['close'], lookback=chop_cfg.flip_lookback
                )
            if len(df_5m['close']) >= chop_cfg.choppiness_lookback + 1:
                choppiness_val = features.choppiness_index(
                    df_5m['high'], df_5m['low'], df_5m['close'],
                    lookback=chop_cfg.choppiness_lookback
                )
            if chop_cfg.bb_width_percentile_lookback and bb_result is not None:
                bb_width_series = bb_result[3]
                if len(bb_width_series) >= chop_cfg.bb_width_percentile_lookback:
                    bb_width_pct_rank_val = features.percentile_rank_last(
                        bb_width_series,
                        lookback=chop_cfg.bb_width_percentile_lookback
                    )
            if trend_cfg and adx_series is not None and len(adx_series) >= trend_cfg.adx_momentum_lookback + 1:
                adx_momentum_val = features.adx_momentum(
                    adx_series, lookback=trend_cfg.adx_momentum_lookback
                )

        # Rolling 20-bar high/low (for breakout strategy)
        high_20_series = df_5m['high'].rolling(window=20).max()
        low_20_series = df_5m['low'].rolling(window=20).min()

        # Extract latest values
        try:
            atr_by_lookback = {}
            for lb, series in atr_series_by_lb.items():
                try:
                    atr_by_lookback[int(lb)] = float(series.iloc[-1]) if series is not None else None
                except (IndexError, TypeError, ValueError):
                    atr_by_lookback[int(lb)] = None

            last_close_1h = float(df_1h['close'].iloc[-1])
            ema20_1h_val = float(ema20_1h_series.iloc[-1]) if ema20_1h_series is not None else None
            ema50_1h_val = float(ema50_1h_series.iloc[-1]) if ema50_1h_series is not None else None
            ema1h_spread_pct = None
            if ema20_1h_val is not None and ema50_1h_val is not None:
                denom = max(abs(last_close_1h), 1e-9)
                ema1h_spread_pct = abs(ema20_1h_val - ema50_1h_val) / denom

            if trend_cfg and ema20_1h_series is not None and len(ema20_1h_series) >= trend_cfg.ema_slope_lookback_1h + 1:
                ema20_1h_slope_val = features.ema_slope(
                    ema20_1h_series, lookback=trend_cfg.ema_slope_lookback_1h
                )

            snapshot = {
                "rsi14": float(rsi_series.iloc[-1]) if rsi_series is not None else None,
                "adx14": float(adx_series.iloc[-1]) if adx_series is not None else None,
                "atr14": float(atr_series.iloc[-1]) if atr_series is not None else None,
                "atr_by_lookback": atr_by_lookback,
                "ema20_5m": float(ema20_5m_series.iloc[-1]) if ema20_5m_series is not None else None,
                "ema50_5m": float(ema50_5m_series.iloc[-1]) if ema50_5m_series is not None else None,
                "ema20_1h": ema20_1h_val,
                "ema50_1h": ema50_1h_val,
                "ema20_4h": ema20_4h_val,
                "ema50_4h": ema50_4h_val,
                "bb_width": float(bb_result[3].iloc[-1]) if bb_result is not None else None,
                "bb_middle": float(bb_result[0].iloc[-1]) if bb_result is not None else None,
                "bb_upper": float(bb_result[1].iloc[-1]) if bb_result is not None else None,
                "bb_lower": float(bb_result[2].iloc[-1]) if bb_result is not None else None,
                "high_20": float(high_20_series.iloc[-1]),
                "low_20": float(low_20_series.iloc[-1]),
                "vol_z": float(vol_z_series.iloc[-1]) if vol_z_series is not None else None,
                "atr_z": float(atr_z_series.iloc[-1]) if atr_z_series is not None else None,
                "kaufman_er": kaufman_er_val,
                "flip_rate": flip_rate_val,
                "choppiness": choppiness_val,
                "adx_momentum": adx_momentum_val,
                "ema20_1h_slope": ema20_1h_slope_val,
                "ema1h_spread_pct": ema1h_spread_pct,
                "bb_width_pct_rank": bb_width_pct_rank_val,
            }

            # book_imbalance_ratio is fetched externally (runner) and injected
            snapshot["book_imbalance_ratio"] = None

            # Validate core features are present
            # ema20_4h/ema50_4h are optional (may be None during 4h warmup)
            optional_keys = {
                "bb_middle", "bb_upper", "bb_lower", "high_20", "low_20",
                "ema20_4h", "ema50_4h", "book_imbalance_ratio",
                "kaufman_er", "flip_rate", "choppiness", "adx_momentum",
                "ema20_1h_slope", "ema1h_spread_pct", "bb_width_pct_rank",
                "atr_by_lookback",
            }
            core_snapshot = {k: v for k, v in snapshot.items() if k not in optional_keys}
            if any(v is None for v in core_snapshot.values()):
                logger.warning(f"Some core features are None for {symbol}: {core_snapshot}")
                return None

            logger.debug(f"Computed features for {symbol}: RSI={snapshot['rsi14']:.2f}, ADX={snapshot['adx14']:.2f}")
            return snapshot

        except (IndexError, KeyError, TypeError) as e:
            logger.error(f"Error extracting feature values for {symbol}: {e}")
            return None

    @staticmethod
    def _candles_to_df(candles: list) -> pd.DataFrame:
        """
        Convert list of Candle objects to pandas DataFrame.

        Args:
            candles: List of Candle objects

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        data = {
            'timestamp': [c.timestamp for c in candles],
            'open': [c.open for c in candles],
            'high': [c.high for c in candles],
            'low': [c.low for c in candles],
            'close': [c.close for c in candles],
            'volume': [c.volume for c in candles],
        }

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        return df
