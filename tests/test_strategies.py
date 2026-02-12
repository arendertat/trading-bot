"""Unit tests for trading strategies"""

import pytest
from bot.strategies.trend_pullback import TrendPullbackStrategy
from bot.strategies.trend_breakout import TrendBreakoutStrategy
from bot.strategies.range_mean_reversion import RangeMeanReversionStrategy
from bot.strategies.base import FeatureSet, StrategySignal
from bot.regime.models import RegimeResult
from bot.core.constants import OrderSide, RegimeType


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def trend_pullback_config():
    """Trend Pullback strategy config"""
    return {
        "enabled": True,
        "stop_pct": 0.01,
        "target_r_multiple": 1.5,
        "pullback_rsi_long_min": 40,
        "pullback_rsi_long_max": 50,
        "pullback_rsi_short_min": 50,
        "pullback_rsi_short_max": 60,
        "ema20_band_pct": 0.002,
        "trail_after_r": 1.0,
        "atr_trail_mult": 2.0
    }


@pytest.fixture
def trend_breakout_config():
    """Trend Breakout strategy config"""
    return {
        "enabled": True,
        "stop_pct": 0.01,
        "breakout_volume_z_min": 1.0,
        "atr_trail_mult": 2.5,
        "trail_after_r": 0.0  # Immediate trailing
    }


@pytest.fixture
def range_mean_reversion_config():
    """Range Mean Reversion strategy config"""
    return {
        "enabled": True,
        "stop_pct": 0.008,
        "target_r_multiple": 1.2,
        "rsi_long_extreme": 25,
        "rsi_short_extreme": 75,
        "trail_after_r": 0  # No trailing
    }


# ============================================================================
# Trend Pullback Strategy Tests
# ============================================================================

class TestTrendPullbackStrategy:
    """Unit tests for Trend Pullback strategy"""

    @pytest.fixture
    def strategy(self, trend_pullback_config):
        """Strategy instance"""
        return TrendPullbackStrategy(trend_pullback_config)

    def test_long_entry_conditions_met(self, strategy):
        """Test LONG entry when all conditions met"""
        features = FeatureSet(
            rsi_5m=45.0,  # In pullback range (40-50)
            ema20_5m=50000.0,
            ema50_5m=49500.0,
            ema20_1h=50000.0,
            ema50_1h=49000.0,
            atr_5m=500.0,
            bb_upper_5m=51000.0,
            bb_lower_5m=49000.0,
            bb_middle_5m=50000.0,
            high_20_bars=50500.0,
            low_20_bars=49500.0,
            volume_z_5m=0.5
        )

        regime = RegimeResult(
            symbol="BTCUSDT",
            regime=RegimeType.TREND,
            confidence=0.75,
            adx=30.0,
            atr_z=1.0,
            bb_width=0.03,
            ema20_1h=50000.0,
            ema50_1h=49000.0,
            reasons=["Bullish trend"],
            trend_direction="bullish"
        )

        current_price = 50100.0  # Near EMA20 (within 0.2%)

        entry, side, reason = strategy.entry_conditions(features, regime, "BTCUSDT", current_price)

        assert entry is True
        assert side == OrderSide.LONG
        assert "pullback LONG" in reason

    def test_long_entry_rsi_too_high(self, strategy):
        """Test LONG entry rejected when RSI too high"""
        features = FeatureSet(
            rsi_5m=55.0,  # Above pullback range
            ema20_5m=50000.0,
            ema50_5m=49500.0,
            ema20_1h=50000.0,
            ema50_1h=49000.0,
            atr_5m=500.0,
            bb_upper_5m=51000.0,
            bb_lower_5m=49000.0,
            bb_middle_5m=50000.0,
            high_20_bars=50500.0,
            low_20_bars=49500.0,
            volume_z_5m=0.5
        )

        regime = RegimeResult(
            symbol="BTCUSDT",
            regime=RegimeType.TREND,
            confidence=0.75,
            adx=30.0,
            atr_z=1.0,
            bb_width=0.03,
            ema20_1h=50000.0,
            ema50_1h=49000.0,
            reasons=["Bullish trend"],
            trend_direction="bullish"
        )

        current_price = 50100.0

        entry, side, reason = strategy.entry_conditions(features, regime, "BTCUSDT", current_price)

        assert entry is False
        assert side is None

    def test_short_entry_conditions_met(self, strategy):
        """Test SHORT entry when all conditions met"""
        features = FeatureSet(
            rsi_5m=55.0,  # In pullback range (50-60)
            ema20_5m=50000.0,
            ema50_5m=50500.0,
            ema20_1h=49000.0,
            ema50_1h=50000.0,  # Bearish: EMA20 < EMA50
            atr_5m=500.0,
            bb_upper_5m=51000.0,
            bb_lower_5m=49000.0,
            bb_middle_5m=50000.0,
            high_20_bars=50500.0,
            low_20_bars=49500.0,
            volume_z_5m=0.5
        )

        regime = RegimeResult(
            symbol="BTCUSDT",
            regime=RegimeType.TREND,
            confidence=0.75,
            adx=30.0,
            atr_z=1.0,
            bb_width=0.03,
            ema20_1h=49000.0,
            ema50_1h=50000.0,
            reasons=["Bearish trend"],
            trend_direction="bearish"
        )

        current_price = 49900.0  # Near EMA20

        entry, side, reason = strategy.entry_conditions(features, regime, "BTCUSDT", current_price)

        assert entry is True
        assert side == OrderSide.SHORT
        assert "pullback SHORT" in reason

    def test_rejected_in_range_regime(self, strategy):
        """Test entry rejected when not in TREND regime"""
        features = FeatureSet(
            rsi_5m=45.0,
            ema20_5m=50000.0,
            ema50_5m=49500.0,
            ema20_1h=50000.0,
            ema50_1h=49000.0,
            atr_5m=500.0,
            bb_upper_5m=51000.0,
            bb_lower_5m=49000.0,
            bb_middle_5m=50000.0,
            high_20_bars=50500.0,
            low_20_bars=49500.0,
            volume_z_5m=0.5
        )

        regime = RegimeResult(
            symbol="BTCUSDT",
            regime=RegimeType.RANGE,  # Wrong regime
            confidence=0.75,
            adx=15.0,
            atr_z=0.5,
            bb_width=0.02,
            ema20_1h=50000.0,
            ema50_1h=50000.0,
            reasons=["Range market"],
            trend_direction=None
        )

        current_price = 50100.0

        entry, side, reason = strategy.entry_conditions(features, regime, "BTCUSDT", current_price)

        assert entry is False
        assert "Not in TREND regime" in reason

    def test_stop_loss_calculation_long(self, strategy):
        """Test stop loss calculated correctly for LONG"""
        entry_price = 50000.0
        side = OrderSide.LONG
        atr = 500.0

        stop_price = strategy.calculate_stop_loss(entry_price, side, atr)

        expected_stop = 50000.0 * 0.99  # 1% below entry
        assert stop_price == pytest.approx(expected_stop, rel=1e-6)

    def test_stop_loss_calculation_short(self, strategy):
        """Test stop loss calculated correctly for SHORT"""
        entry_price = 50000.0
        side = OrderSide.SHORT
        atr = 500.0

        stop_price = strategy.calculate_stop_loss(entry_price, side, atr)

        expected_stop = 50000.0 * 1.01  # 1% above entry
        assert stop_price == pytest.approx(expected_stop, rel=1e-6)

    def test_take_profit_calculation_long(self, strategy):
        """Test take profit calculated correctly for LONG"""
        entry_price = 50000.0
        stop_price = 49500.0  # 1% stop
        side = OrderSide.LONG

        tp_price = strategy.calculate_take_profit(entry_price, stop_price, side)

        risk_distance = 500.0
        target_distance = 500.0 * 1.5  # 1.5R
        expected_tp = 50000.0 + 750.0

        assert tp_price == pytest.approx(expected_tp, rel=1e-6)

    def test_take_profit_calculation_short(self, strategy):
        """Test take profit calculated correctly for SHORT"""
        entry_price = 50000.0
        stop_price = 50500.0  # 1% stop
        side = OrderSide.SHORT

        tp_price = strategy.calculate_take_profit(entry_price, stop_price, side)

        risk_distance = 500.0
        target_distance = 500.0 * 1.5  # 1.5R
        expected_tp = 50000.0 - 750.0

        assert tp_price == pytest.approx(expected_tp, rel=1e-6)

    def test_generate_signal_complete(self, strategy):
        """Test full signal generation"""
        features = FeatureSet(
            rsi_5m=45.0,
            ema20_5m=50000.0,
            ema50_5m=49500.0,
            ema20_1h=50000.0,
            ema50_1h=49000.0,
            atr_5m=500.0,
            bb_upper_5m=51000.0,
            bb_lower_5m=49000.0,
            bb_middle_5m=50000.0,
            high_20_bars=50500.0,
            low_20_bars=49500.0,
            volume_z_5m=0.5
        )

        regime = RegimeResult(
            symbol="BTCUSDT",
            regime=RegimeType.TREND,
            confidence=0.75,
            adx=30.0,
            atr_z=1.0,
            bb_width=0.03,
            ema20_1h=50000.0,
            ema50_1h=49000.0,
            reasons=["Bullish trend"],
            trend_direction="bullish"
        )

        current_price = 50100.0

        signal = strategy.generate_signal(features, regime, "BTCUSDT", current_price)

        assert signal is not None
        assert signal.entry is True
        assert signal.side == OrderSide.LONG
        assert signal.stop_pct == pytest.approx(0.01, rel=1e-3)
        assert signal.target_r == pytest.approx(1.5, rel=0.1)
        assert signal.trail_enabled is True
        assert signal.symbol == "BTCUSDT"


# ============================================================================
# Trend Breakout Strategy Tests
# ============================================================================

class TestTrendBreakoutStrategy:
    """Unit tests for Trend Breakout strategy"""

    @pytest.fixture
    def strategy(self, trend_breakout_config):
        """Strategy instance"""
        return TrendBreakoutStrategy(trend_breakout_config)

    def test_long_breakout_with_volume(self, strategy):
        """Test LONG breakout with volume confirmation"""
        features = FeatureSet(
            rsi_5m=60.0,
            ema20_5m=50000.0,
            ema50_5m=49500.0,
            ema20_1h=50000.0,
            ema50_1h=49000.0,
            atr_5m=500.0,
            bb_upper_5m=51000.0,
            bb_lower_5m=49000.0,
            bb_middle_5m=50000.0,
            high_20_bars=50000.0,  # 20-bar high
            low_20_bars=49000.0,
            volume_z_5m=1.5  # Above threshold
        )

        regime = RegimeResult(
            symbol="BTCUSDT",
            regime=RegimeType.TREND,
            confidence=0.75,
            adx=30.0,
            atr_z=1.0,
            bb_width=0.03,
            ema20_1h=50000.0,
            ema50_1h=49000.0,
            reasons=["Bullish trend"],
            trend_direction="bullish"
        )

        current_price = 50100.0  # Above 20-bar high

        entry, side, reason = strategy.entry_conditions(features, regime, "BTCUSDT", current_price)

        assert entry is True
        assert side == OrderSide.LONG
        assert "breakout LONG" in reason

    def test_long_breakout_without_volume(self, strategy):
        """Test LONG breakout rejected without volume confirmation"""
        features = FeatureSet(
            rsi_5m=60.0,
            ema20_5m=50000.0,
            ema50_5m=49500.0,
            ema20_1h=50000.0,
            ema50_1h=49000.0,
            atr_5m=500.0,
            bb_upper_5m=51000.0,
            bb_lower_5m=49000.0,
            bb_middle_5m=50000.0,
            high_20_bars=50000.0,
            low_20_bars=49000.0,
            volume_z_5m=0.5  # Below threshold
        )

        regime = RegimeResult(
            symbol="BTCUSDT",
            regime=RegimeType.TREND,
            confidence=0.75,
            adx=30.0,
            atr_z=1.0,
            bb_width=0.03,
            ema20_1h=50000.0,
            ema50_1h=49000.0,
            reasons=["Bullish trend"],
            trend_direction="bullish"
        )

        current_price = 50100.0

        entry, side, reason = strategy.entry_conditions(features, regime, "BTCUSDT", current_price)

        assert entry is False

    def test_short_breakout_with_volume(self, strategy):
        """Test SHORT breakout with volume confirmation"""
        features = FeatureSet(
            rsi_5m=40.0,
            ema20_5m=50000.0,
            ema50_5m=50500.0,
            ema20_1h=49000.0,
            ema50_1h=50000.0,
            atr_5m=500.0,
            bb_upper_5m=51000.0,
            bb_lower_5m=49000.0,
            bb_middle_5m=50000.0,
            high_20_bars=51000.0,
            low_20_bars=50000.0,  # 20-bar low
            volume_z_5m=1.5
        )

        regime = RegimeResult(
            symbol="BTCUSDT",
            regime=RegimeType.TREND,
            confidence=0.75,
            adx=30.0,
            atr_z=1.0,
            bb_width=0.03,
            ema20_1h=49000.0,
            ema50_1h=50000.0,
            reasons=["Bearish trend"],
            trend_direction="bearish"
        )

        current_price = 49900.0  # Below 20-bar low

        entry, side, reason = strategy.entry_conditions(features, regime, "BTCUSDT", current_price)

        assert entry is True
        assert side == OrderSide.SHORT
        assert "breakout SHORT" in reason

    def test_tp_very_far_for_trailing(self, strategy):
        """Test TP is set very far (100R) for trailing-based exit"""
        entry_price = 50000.0
        stop_price = 49500.0
        side = OrderSide.LONG

        tp_price = strategy.calculate_take_profit(entry_price, stop_price, side)

        risk_distance = 500.0
        expected_tp = 50000.0 + (risk_distance * 100)

        assert tp_price == pytest.approx(expected_tp, rel=1e-6)


# ============================================================================
# Range Mean Reversion Strategy Tests
# ============================================================================

class TestRangeMeanReversionStrategy:
    """Unit tests for Range Mean Reversion strategy"""

    @pytest.fixture
    def strategy(self, range_mean_reversion_config):
        """Strategy instance"""
        return RangeMeanReversionStrategy(range_mean_reversion_config)

    def test_long_oversold_bb_touch(self, strategy):
        """Test LONG entry on oversold RSI + BB lower touch"""
        features = FeatureSet(
            rsi_5m=20.0,  # Oversold < 25
            ema20_5m=50000.0,
            ema50_5m=50000.0,
            ema20_1h=50000.0,
            ema50_1h=50000.0,
            atr_5m=500.0,
            bb_upper_5m=51000.0,
            bb_lower_5m=49000.0,  # BB lower band
            bb_middle_5m=50000.0,
            high_20_bars=50500.0,
            low_20_bars=49500.0,
            volume_z_5m=0.5
        )

        regime = RegimeResult(
            symbol="BTCUSDT",
            regime=RegimeType.RANGE,
            confidence=0.75,
            adx=15.0,
            atr_z=0.5,
            bb_width=0.02,
            ema20_1h=50000.0,
            ema50_1h=50000.0,
            reasons=["Range market"],
            trend_direction=None
        )

        current_price = 49005.0  # At BB lower (within 0.1%)

        entry, side, reason = strategy.entry_conditions(features, regime, "BTCUSDT", current_price)

        assert entry is True
        assert side == OrderSide.LONG
        assert "mean reversion LONG" in reason

    def test_short_overbought_bb_touch(self, strategy):
        """Test SHORT entry on overbought RSI + BB upper touch"""
        features = FeatureSet(
            rsi_5m=80.0,  # Overbought > 75
            ema20_5m=50000.0,
            ema50_5m=50000.0,
            ema20_1h=50000.0,
            ema50_1h=50000.0,
            atr_5m=500.0,
            bb_upper_5m=51000.0,  # BB upper band
            bb_lower_5m=49000.0,
            bb_middle_5m=50000.0,
            high_20_bars=50500.0,
            low_20_bars=49500.0,
            volume_z_5m=0.5
        )

        regime = RegimeResult(
            symbol="BTCUSDT",
            regime=RegimeType.RANGE,
            confidence=0.75,
            adx=15.0,
            atr_z=0.5,
            bb_width=0.02,
            ema20_1h=50000.0,
            ema50_1h=50000.0,
            reasons=["Range market"],
            trend_direction=None
        )

        current_price = 50995.0  # At BB upper (within 0.1%)

        entry, side, reason = strategy.entry_conditions(features, regime, "BTCUSDT", current_price)

        assert entry is True
        assert side == OrderSide.SHORT
        assert "mean reversion SHORT" in reason

    def test_rejected_not_at_bb(self, strategy):
        """Test entry rejected when RSI extreme but price not at BB"""
        features = FeatureSet(
            rsi_5m=20.0,  # Oversold
            ema20_5m=50000.0,
            ema50_5m=50000.0,
            ema20_1h=50000.0,
            ema50_1h=50000.0,
            atr_5m=500.0,
            bb_upper_5m=51000.0,
            bb_lower_5m=49000.0,
            bb_middle_5m=50000.0,
            high_20_bars=50500.0,
            low_20_bars=49500.0,
            volume_z_5m=0.5
        )

        regime = RegimeResult(
            symbol="BTCUSDT",
            regime=RegimeType.RANGE,
            confidence=0.75,
            adx=15.0,
            atr_z=0.5,
            bb_width=0.02,
            ema20_1h=50000.0,
            ema50_1h=50000.0,
            reasons=["Range market"],
            trend_direction=None
        )

        current_price = 49500.0  # Not at BB lower

        entry, side, reason = strategy.entry_conditions(features, regime, "BTCUSDT", current_price)

        assert entry is False

    def test_tighter_stop_for_range(self, strategy):
        """Test stop loss is tighter (0.8%) for range strategy"""
        entry_price = 50000.0
        side = OrderSide.LONG
        atr = 500.0

        stop_price = strategy.calculate_stop_loss(entry_price, side, atr)

        expected_stop = 50000.0 * 0.992  # 0.8% below entry
        assert stop_price == pytest.approx(expected_stop, rel=1e-6)

    def test_tp_calculation_1_2r(self, strategy):
        """Test take profit at 1.2R"""
        entry_price = 50000.0
        stop_price = 49600.0  # 0.8% stop (400 risk)
        side = OrderSide.LONG

        tp_price = strategy.calculate_take_profit(entry_price, stop_price, side)

        risk_distance = 400.0
        target_distance = 400.0 * 1.2  # 1.2R
        expected_tp = 50000.0 + 480.0

        assert tp_price == pytest.approx(expected_tp, rel=1e-6)
