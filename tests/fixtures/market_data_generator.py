"""Market data generator for integration testing"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass

from bot.core.types import Candle
from bot.strategies.base import FeatureSet


@dataclass
class MarketScenario:
    """
    Market scenario specification for data generation.

    Defines the characteristics of a market period.
    """
    name: str
    duration_candles: int  # Number of 5m candles
    trend_strength: float  # 0.0 = no trend, 1.0 = strong trend
    volatility: float  # 0.0 = low vol, 1.0 = high vol
    range_bound: bool  # True = sideways market
    starting_price: float


class MarketDataGenerator:
    """
    Generate realistic OHLCV candle data for testing.

    Supports different market regimes:
    - TREND: Strong directional movement with pullbacks
    - RANGE: Sideways movement within boundaries
    - HIGH_VOL: Large price swings
    - CHOP: Low volatility, no clear direction
    """

    def __init__(self, seed: int = 42):
        """
        Initialize market data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)

    def generate_trend_candles(
        self,
        starting_price: float,
        num_candles: int,
        direction: str = "bullish",
        volatility: float = 0.5
    ) -> List[Candle]:
        """
        Generate trending market candles.

        Args:
            starting_price: Initial price
            num_candles: Number of candles to generate
            direction: "bullish" or "bearish"
            volatility: 0.0 to 1.0 (higher = more volatility)

        Returns:
            List of Candle objects
        """
        candles = []
        current_price = starting_price
        base_timestamp = int(datetime(2024, 1, 1).timestamp() * 1000)

        # Trend parameters
        trend_strength = 0.002 if direction == "bullish" else -0.002  # 0.2% per candle
        base_volatility = 0.003 * (1 + volatility)  # Base price movement

        for i in range(num_candles):
            # Add trend component
            trend_move = current_price * trend_strength

            # Add random walk component
            random_move = np.random.randn() * current_price * base_volatility

            # Calculate OHLC with trend bias
            if direction == "bullish":
                open_price = current_price
                close_price = current_price + trend_move + random_move
                high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.002))
                low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.001))
            else:  # bearish
                open_price = current_price
                close_price = current_price + trend_move + random_move
                high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.001))
                low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.002))

            # Volume (higher during trend moves)
            base_volume = 1000000
            volume = base_volume * (1 + abs(random_move / current_price) * 10)

            candle = Candle(
                timestamp=base_timestamp + (i * 5 * 60 * 1000),  # 5 minute intervals
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            candles.append(candle)

            current_price = close_price

        return candles

    def generate_range_candles(
        self,
        center_price: float,
        num_candles: int,
        range_pct: float = 0.02
    ) -> List[Candle]:
        """
        Generate range-bound market candles.

        Args:
            center_price: Center price of the range
            num_candles: Number of candles to generate
            range_pct: Range width as percentage (0.02 = 2%)

        Returns:
            List of Candle objects
        """
        candles = []
        current_price = center_price
        base_timestamp = int(datetime(2024, 1, 1).timestamp() * 1000)

        upper_bound = center_price * (1 + range_pct)
        lower_bound = center_price * (1 - range_pct)
        base_volatility = 0.001

        for i in range(num_candles):
            # Mean reversion force
            distance_from_center = (current_price - center_price) / center_price
            reversion_force = -distance_from_center * 0.5

            # Random walk
            random_move = np.random.randn() * current_price * base_volatility

            # Calculate close with mean reversion
            close_price = current_price + (current_price * reversion_force) + random_move

            # Clamp to range boundaries
            close_price = max(lower_bound, min(upper_bound, close_price))

            open_price = current_price
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.001))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.001))

            # Clamp OHLC to bounds
            high_price = min(high_price, upper_bound)
            low_price = max(low_price, lower_bound)

            # Lower volume in range
            volume = 800000 * (1 + np.random.uniform(-0.2, 0.2))

            candle = Candle(
                timestamp=base_timestamp + (i * 5 * 60 * 1000),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            candles.append(candle)

            current_price = close_price

        return candles

    def generate_high_vol_candles(
        self,
        starting_price: float,
        num_candles: int
    ) -> List[Candle]:
        """
        Generate high volatility market candles.

        Args:
            starting_price: Initial price
            num_candles: Number of candles to generate

        Returns:
            List of Candle objects
        """
        candles = []
        current_price = starting_price
        base_timestamp = int(datetime(2024, 1, 1).timestamp() * 1000)

        high_volatility = 0.01  # 1% per candle

        for i in range(num_candles):
            # Large random moves
            random_move = np.random.randn() * current_price * high_volatility

            open_price = current_price
            close_price = current_price + random_move

            # Wide OHLC ranges
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0.002, 0.005))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0.002, 0.005))

            # Very high volume during vol spikes
            volume = 2000000 * (1 + abs(random_move / current_price) * 20)

            candle = Candle(
                timestamp=base_timestamp + (i * 5 * 60 * 1000),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            candles.append(candle)

            current_price = close_price

        return candles

    def generate_30_day_scenario(
        self,
        starting_price: float = 50000.0
    ) -> Tuple[List[Candle], Dict[str, List[Tuple[int, int]]]]:
        """
        Generate 30-day market scenario with mixed regimes.

        Creates realistic market data covering all regime types.

        Args:
            starting_price: Initial price

        Returns:
            (candles, regime_periods)
            - candles: List of all candles (8640 candles for 30 days)
            - regime_periods: Dict mapping regime name to list of (start_idx, end_idx)
        """
        all_candles = []
        regime_periods = {
            "TREND_BULLISH": [],
            "TREND_BEARISH": [],
            "RANGE": [],
            "HIGH_VOL": []
        }

        current_price = starting_price
        candle_idx = 0

        # Day 1-5: Bullish trend (1440 candles)
        start_idx = candle_idx
        trend_candles = self.generate_trend_candles(
            current_price, 1440, direction="bullish", volatility=0.3
        )
        all_candles.extend(trend_candles)
        candle_idx += len(trend_candles)
        regime_periods["TREND_BULLISH"].append((start_idx, candle_idx - 1))
        current_price = trend_candles[-1].close

        # Day 6-10: Range (1440 candles)
        start_idx = candle_idx
        range_candles = self.generate_range_candles(
            current_price, 1440, range_pct=0.015
        )
        all_candles.extend(range_candles)
        candle_idx += len(range_candles)
        regime_periods["RANGE"].append((start_idx, candle_idx - 1))
        current_price = range_candles[-1].close

        # Day 11-13: High volatility (864 candles)
        start_idx = candle_idx
        high_vol_candles = self.generate_high_vol_candles(current_price, 864)
        all_candles.extend(high_vol_candles)
        candle_idx += len(high_vol_candles)
        regime_periods["HIGH_VOL"].append((start_idx, candle_idx - 1))
        current_price = high_vol_candles[-1].close

        # Day 14-18: Bearish trend (1440 candles)
        start_idx = candle_idx
        trend_candles = self.generate_trend_candles(
            current_price, 1440, direction="bearish", volatility=0.4
        )
        all_candles.extend(trend_candles)
        candle_idx += len(trend_candles)
        regime_periods["TREND_BEARISH"].append((start_idx, candle_idx - 1))
        current_price = trend_candles[-1].close

        # Day 19-23: Range (1440 candles)
        start_idx = candle_idx
        range_candles = self.generate_range_candles(
            current_price, 1440, range_pct=0.02
        )
        all_candles.extend(range_candles)
        candle_idx += len(range_candles)
        regime_periods["RANGE"].append((start_idx, candle_idx - 1))
        current_price = range_candles[-1].close

        # Day 24-26: Bullish trend (864 candles)
        start_idx = candle_idx
        trend_candles = self.generate_trend_candles(
            current_price, 864, direction="bullish", volatility=0.5
        )
        all_candles.extend(trend_candles)
        candle_idx += len(trend_candles)
        regime_periods["TREND_BULLISH"].append((start_idx, candle_idx - 1))
        current_price = trend_candles[-1].close

        # Day 27-30: High volatility (1152 candles)
        start_idx = candle_idx
        high_vol_candles = self.generate_high_vol_candles(current_price, 1152)
        all_candles.extend(high_vol_candles)
        candle_idx += len(high_vol_candles)
        regime_periods["HIGH_VOL"].append((start_idx, candle_idx - 1))

        return all_candles, regime_periods

    def generate_features_from_candles(
        self,
        candles: List[Candle],
        lookback: int = 100
    ) -> FeatureSet:
        """
        Generate mock features from candle data.

        Simplified feature calculation for testing purposes.

        Args:
            candles: List of recent candles (needs at least lookback candles)
            lookback: Number of candles to use for calculations

        Returns:
            FeatureSet with calculated indicators
        """
        if len(candles) < lookback:
            raise ValueError(f"Need at least {lookback} candles, got {len(candles)}")

        # Use last N candles
        recent = candles[-lookback:]
        closes = np.array([c.close for c in recent])
        highs = np.array([c.high for c in recent])
        lows = np.array([c.low for c in recent])
        volumes = np.array([c.volume for c in recent])

        # RSI calculation (simplified)
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi_5m = 100 - (100 / (1 + rs))

        # EMAs
        ema20_5m = np.mean(closes[-20:])
        ema50_5m = np.mean(closes[-50:])
        ema20_1h = np.mean(closes[-20:])  # Simplified (should use 1h candles)
        ema50_1h = np.mean(closes[-50:])

        # ATR
        tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))
        atr_5m = np.mean(tr[-14:])

        # Bollinger Bands
        bb_middle = np.mean(closes[-20:])
        bb_std = np.std(closes[-20:])
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)

        # High/Low
        high_20 = np.max(highs[-20:])
        low_20 = np.min(lows[-20:])

        # Volume z-score
        vol_mean = np.mean(volumes)
        vol_std = np.std(volumes)
        volume_z = (volumes[-1] - vol_mean) / vol_std if vol_std > 0 else 0

        return FeatureSet(
            rsi_5m=rsi_5m,
            ema20_5m=ema20_5m,
            ema50_5m=ema50_5m,
            ema20_1h=ema20_1h,
            ema50_1h=ema50_1h,
            atr_5m=atr_5m,
            bb_upper_5m=bb_upper,
            bb_lower_5m=bb_lower,
            bb_middle_5m=bb_middle,
            high_20_bars=high_20,
            low_20_bars=low_20,
            volume_z_5m=volume_z
        )
