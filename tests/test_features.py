"""Unit tests for technical indicator functions"""

import pytest
import pandas as pd
import numpy as np

from bot.data import features


class TestEMA:
    """Test Exponential Moving Average"""

    def test_ema_basic(self):
        """Test EMA on simple series"""
        series = pd.Series([10, 11, 12, 13, 14, 15])
        result = features.ema(series, period=3)

        assert result is not None
        assert len(result) == len(series)
        # EMA should be smooth and increasing
        assert result.iloc[-1] > result.iloc[0]

    def test_ema_insufficient_data(self):
        """Test EMA returns None when insufficient data"""
        series = pd.Series([10, 11])
        result = features.ema(series, period=5)

        assert result is None

    def test_ema_exact_period(self):
        """Test EMA with exactly period bars"""
        series = pd.Series([10, 11, 12, 13, 14])
        result = features.ema(series, period=5)

        assert result is not None
        assert len(result) == 5


class TestRSI:
    """Test Relative Strength Index"""

    def test_rsi_rising_series(self):
        """Test RSI on rising price series"""
        series = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
        result = features.rsi(series, period=14)

        assert result is not None
        # Rising series should have RSI > 50
        assert result.iloc[-1] > 50

    def test_rsi_falling_series(self):
        """Test RSI on falling price series"""
        series = pd.Series([24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10])
        result = features.rsi(series, period=14)

        assert result is not None
        # Falling series should have RSI < 50
        assert result.iloc[-1] < 50

    def test_rsi_flat_series(self):
        """Test RSI on flat series (no movement)"""
        series = pd.Series([50.0] * 20)
        result = features.rsi(series, period=14)

        assert result is not None
        # Flat series should have RSI around 50 or NaN (no gains/losses)
        # Due to division by zero handling, might be NaN
        assert pd.isna(result.iloc[-1]) or abs(result.iloc[-1] - 50) < 10

    def test_rsi_insufficient_data(self):
        """Test RSI returns None when insufficient data"""
        series = pd.Series([10, 11, 12])
        result = features.rsi(series, period=14)

        assert result is None

    def test_rsi_range(self):
        """Test RSI stays within 0-100 range"""
        series = pd.Series(range(50, 100))  # Strong uptrend
        result = features.rsi(series, period=14)

        assert result is not None
        # RSI should be between 0 and 100
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()


class TestATR:
    """Test Average True Range"""

    def test_atr_basic(self):
        """Test ATR on sample data"""
        high = pd.Series([12, 13, 14, 13, 15, 16, 15, 17, 18, 17, 19, 20, 19, 21, 22])
        low = pd.Series([10, 11, 12, 11, 13, 14, 13, 15, 16, 15, 17, 18, 17, 19, 20])
        close = pd.Series([11, 12, 13, 12, 14, 15, 14, 16, 17, 16, 18, 19, 18, 20, 21])

        result = features.atr(high, low, close, period=14)

        assert result is not None
        assert len(result) == len(high)
        # ATR should be positive
        assert result.iloc[-1] > 0

    def test_atr_insufficient_data(self):
        """Test ATR returns None when insufficient data"""
        high = pd.Series([12, 13])
        low = pd.Series([10, 11])
        close = pd.Series([11, 12])

        result = features.atr(high, low, close, period=14)

        assert result is None

    def test_atr_volatile_market(self):
        """Test ATR increases with volatility"""
        # Low volatility
        high1 = pd.Series([101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])
        low1 = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114])
        close1 = pd.Series([100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5])

        # High volatility
        high2 = pd.Series([110, 120, 115, 130, 125, 140, 135, 150, 145, 160, 155, 170, 165, 180, 175])
        low2 = pd.Series([100, 105, 100, 110, 105, 120, 115, 130, 125, 140, 135, 150, 145, 160, 155])
        close2 = pd.Series([105, 115, 110, 125, 120, 135, 130, 145, 140, 155, 150, 165, 160, 175, 170])

        atr1 = features.atr(high1, low1, close1, period=14)
        atr2 = features.atr(high2, low2, close2, period=14)

        assert atr1 is not None
        assert atr2 is not None
        # Higher volatility should have higher ATR
        assert atr2.iloc[-1] > atr1.iloc[-1]


class TestADX:
    """Test Average Directional Index"""

    def test_adx_basic(self):
        """Test ADX on sample data"""
        high = pd.Series(range(100, 130))
        low = pd.Series(range(98, 128))
        close = pd.Series(range(99, 129))

        result = features.adx(high, low, close, period=14)

        assert result is not None
        # ADX should be between 0 and 100
        assert result.iloc[-1] >= 0
        assert result.iloc[-1] <= 100

    def test_adx_insufficient_data(self):
        """Test ADX returns None when insufficient data"""
        high = pd.Series([12, 13, 14])
        low = pd.Series([10, 11, 12])
        close = pd.Series([11, 12, 13])

        result = features.adx(high, low, close, period=14)

        assert result is None

    def test_adx_trending_market(self):
        """Test ADX on strong trend (should be high)"""
        # Strong uptrend
        high = pd.Series(range(100, 150))
        low = pd.Series(range(98, 148))
        close = pd.Series(range(99, 149))

        result = features.adx(high, low, close, period=14)

        assert result is not None
        # Strong trend should have higher ADX (typically > 20-25)
        assert result.iloc[-1] > 15


class TestBollingerBands:
    """Test Bollinger Bands"""

    def test_bollinger_basic(self):
        """Test Bollinger Bands on sample data"""
        close = pd.Series(range(100, 125))

        result = features.bollinger_bands(close, period=20, std=2.0)

        assert result is not None
        middle, upper, lower, width = result

        # Upper should be above middle, middle above lower
        assert upper.iloc[-1] > middle.iloc[-1]
        assert middle.iloc[-1] > lower.iloc[-1]

        # Width should be positive
        assert width.iloc[-1] > 0

    def test_bollinger_insufficient_data(self):
        """Test Bollinger Bands returns None when insufficient data"""
        close = pd.Series([100, 101, 102])

        result = features.bollinger_bands(close, period=20, std=2.0)

        assert result is None

    def test_bollinger_returns_structure(self):
        """Test Bollinger Bands returns correct structure"""
        close = pd.Series(range(100, 130))

        result = features.bollinger_bands(close, period=20, std=2.0)

        assert result is not None
        assert len(result) == 4  # (middle, upper, lower, width)


class TestZScore:
    """Test Z-Score calculation"""

    def test_zscore_basic(self):
        """Test z-score on sample data"""
        values = pd.Series([10, 12, 11, 13, 12, 14, 13, 15] + [14] * 100)

        result = features.zscore(values, window=100)

        assert result is not None
        assert len(result) == len(values)

    def test_zscore_constant_series(self):
        """Test z-score on constant series (should be 0 or NaN)"""
        values = pd.Series([50.0] * 150)

        result = features.zscore(values, window=100)

        assert result is not None
        # Constant series has std=0, so z-score should be NaN or 0
        last_val = result.iloc[-1]
        assert pd.isna(last_val) or abs(last_val) < 0.01

    def test_zscore_insufficient_data(self):
        """Test z-score returns None when insufficient data"""
        values = pd.Series([10, 11, 12])

        result = features.zscore(values, window=100)

        assert result is None

    def test_zscore_outlier_detection(self):
        """Test z-score identifies outliers"""
        # Normal values around 100, then a spike
        values = pd.Series([100] * 100 + [150])

        result = features.zscore(values, window=100)

        assert result is not None
        # Last value (spike) should have high z-score
        assert abs(result.iloc[-1]) > 2.0


class TestLogReturns:
    """Test Log Returns calculation"""

    def test_log_returns_basic(self):
        """Test log returns on simple series"""
        close = pd.Series([100, 102, 101, 103, 105])

        result = features.log_returns(close)

        assert result is not None
        assert len(result) == len(close)
        # First value should be NaN (no previous value)
        assert pd.isna(result.iloc[0])

    def test_log_returns_insufficient_data(self):
        """Test log returns returns None when insufficient data"""
        close = pd.Series([100])

        result = features.log_returns(close)

        assert result is None

    def test_log_returns_positive_change(self):
        """Test log returns for positive price change"""
        close = pd.Series([100, 110])

        result = features.log_returns(close)

        assert result is not None
        # Price increased, so log return should be positive
        assert result.iloc[-1] > 0

    def test_log_returns_negative_change(self):
        """Test log returns for negative price change"""
        close = pd.Series([100, 90])

        result = features.log_returns(close)

        assert result is not None
        # Price decreased, so log return should be negative
        assert result.iloc[-1] < 0


class TestChopFeatures:
    """Test CHOP-related features"""

    def test_kaufman_er_trend_high(self):
        """Monotonic trend should yield high ER"""
        close = pd.Series(range(100, 140))
        er = features.kaufman_er(close, lookback=20)
        assert er is not None
        assert er > 0.7

    def test_kaufman_er_chop_low(self):
        """Alternating series should yield low ER"""
        close = pd.Series([100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100,
                           101, 100, 101, 100, 101, 100, 101, 100, 101, 100, 101])
        er = features.kaufman_er(close, lookback=20)
        assert er is not None
        assert er < 0.3

    def test_flip_rate_steady_low(self):
        """Steady uptrend should have low flip rate"""
        close = pd.Series(range(100, 125))
        fr = features.flip_rate(close, lookback=20)
        assert fr is not None
        assert fr <= 0.1

    def test_flip_rate_alternating_high(self):
        """Alternating returns should have high flip rate"""
        close = pd.Series([100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100,
                           101, 100, 101, 100, 101, 100, 101, 100, 101, 100, 101])
        fr = features.flip_rate(close, lookback=20)
        assert fr is not None
        assert fr >= 0.8

    def test_choppiness_index_range(self):
        """Choppiness index should be within 0-100"""
        high = pd.Series([10 + i for i in range(30)])
        low = pd.Series([9 + i for i in range(30)])
        close = pd.Series([9.5 + i for i in range(30)])
        chop = features.choppiness_index(high, low, close, lookback=20)
        assert chop is not None
        assert 0.0 <= chop <= 100.0

    def test_adx_momentum_basic(self):
        """ADX momentum should be positive for rising series"""
        adx_series = pd.Series([10 + i for i in range(30)])
        mom = features.adx_momentum(adx_series, lookback=10)
        assert mom is not None
        assert mom > 0

    def test_ema_slope_basic(self):
        """EMA slope should be positive for rising series"""
        ema_series = pd.Series([100 + i for i in range(30)])
        slope = features.ema_slope(ema_series, lookback=10)
        assert slope is not None
        assert slope > 0
