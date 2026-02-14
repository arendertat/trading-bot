"""Feature engine for computing technical indicator snapshots"""

import logging
from typing import Optional, Dict
import pandas as pd

from bot.data.candle_store import CandleStore
from bot.data import features
from bot.config.models import TimeframesConfig


logger = logging.getLogger("trading_bot.data")


class FeatureEngine:
    """
    Computes technical indicator snapshots from CandleStore.

    Returns latest feature values per symbol based on stored candle history.
    """

    def __init__(self, candle_store: CandleStore, timeframes_config: TimeframesConfig):
        """
        Initialize FeatureEngine.

        Args:
            candle_store: CandleStore instance
            timeframes_config: Timeframes configuration
        """
        self.candle_store = candle_store
        self.config = timeframes_config

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
        min_bars_1h = 50  # For EMA50 on 1h

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

        # Convert to pandas DataFrames
        df_5m = self._candles_to_df(candles_5m)
        df_1h = self._candles_to_df(candles_1h)

        # Compute indicators (5m)
        rsi_series = features.rsi(df_5m['close'], period=14)
        adx_series = features.adx(df_5m['high'], df_5m['low'], df_5m['close'], period=14)
        atr_series = features.atr(df_5m['high'], df_5m['low'], df_5m['close'], period=14)
        ema20_5m_series = features.ema(df_5m['close'], period=20)
        ema50_5m_series = features.ema(df_5m['close'], period=50)

        bb_result = features.bollinger_bands(df_5m['close'], period=20, std=2.0)

        vol_z_series = features.zscore(df_5m['volume'], window=self.config.zscore_lookback)
        atr_z_series = features.zscore(atr_series, window=self.config.zscore_lookback) if atr_series is not None else None

        # Compute indicators (1h)
        ema20_1h_series = features.ema(df_1h['close'], period=20)
        ema50_1h_series = features.ema(df_1h['close'], period=50)

        # Rolling 20-bar high/low (for breakout strategy)
        high_20_series = df_5m['high'].rolling(window=20).max()
        low_20_series = df_5m['low'].rolling(window=20).min()

        # Extract latest values
        try:
            snapshot = {
                "rsi14": float(rsi_series.iloc[-1]) if rsi_series is not None else None,
                "adx14": float(adx_series.iloc[-1]) if adx_series is not None else None,
                "atr14": float(atr_series.iloc[-1]) if atr_series is not None else None,
                "ema20_5m": float(ema20_5m_series.iloc[-1]) if ema20_5m_series is not None else None,
                "ema50_5m": float(ema50_5m_series.iloc[-1]) if ema50_5m_series is not None else None,
                "ema20_1h": float(ema20_1h_series.iloc[-1]) if ema20_1h_series is not None else None,
                "ema50_1h": float(ema50_1h_series.iloc[-1]) if ema50_1h_series is not None else None,
                "bb_width": float(bb_result[3].iloc[-1]) if bb_result is not None else None,
                "bb_middle": float(bb_result[0].iloc[-1]) if bb_result is not None else None,
                "bb_upper": float(bb_result[1].iloc[-1]) if bb_result is not None else None,
                "bb_lower": float(bb_result[2].iloc[-1]) if bb_result is not None else None,
                "high_20": float(high_20_series.iloc[-1]),
                "low_20": float(low_20_series.iloc[-1]),
                "vol_z": float(vol_z_series.iloc[-1]) if vol_z_series is not None else None,
                "atr_z": float(atr_z_series.iloc[-1]) if atr_z_series is not None else None,
            }

            # Validate core features are present (bb_middle/upper/lower/high_20/low_20 optional)
            optional_keys = {"bb_middle", "bb_upper", "bb_lower", "high_20", "low_20"}
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
