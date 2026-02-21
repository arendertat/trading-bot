"""Unit tests for regime detection module"""

import pytest
from bot.regime.detector import RegimeDetector
from bot.regime.models import RegimeResult
from bot.config.models import RegimeConfig
from bot.core.constants import RegimeType


@pytest.fixture
def regime_config():
    """Default regime configuration"""
    return RegimeConfig(
        trend_adx_min=25.0,
        range_adx_max=20.0,
        high_vol_atr_z=1.5,
        confidence_threshold=0.55,
        bb_width_range_min=0.01,
        bb_width_range_max=0.05,
    )


@pytest.fixture
def detector(regime_config):
    """RegimeDetector instance"""
    return RegimeDetector(regime_config)


class TestRegimeDetection:
    """Test regime detection logic"""

    def test_trend_regime_bullish(self, detector):
        """Test TREND regime detection (bullish)"""
        result = detector.detect_regime(
            symbol="BTC/USDT",
            adx=30.0,  # > trend_adx_min (25)
            atr_z=1.0,
            bb_width=0.03,
            ema20_5m=50100.0,
            ema50_5m=50000.0,
            ema20_1h=50200.0,  # Bullish: EMA20 > EMA50
            ema50_1h=50000.0,
        )

        assert result.regime == RegimeType.TREND
        assert result.confidence >= 0.55
        assert "Trend (bullish)" in str(result.reasons)

    def test_range_regime(self, detector):
        """Test RANGE regime detection"""
        result = detector.detect_regime(
            symbol="BTC/USDT",
            adx=15.0,  # < range_adx_max (20)
            atr_z=0.5,
            bb_width=0.02,  # Within range [0.01, 0.05]
            ema20_5m=50000.0,
            ema50_5m=50000.0,
            ema20_1h=50000.0,
            ema50_1h=50000.0,
            rsi_5m=20.0,
            rsi_extreme_low=25.0,
            rsi_extreme_high=75.0,
        )

        assert result.regime == RegimeType.RANGE
        assert result.confidence >= 0.55

    def test_high_vol_regime(self, detector):
        """Test HIGH_VOL regime detection"""
        result = detector.detect_regime(
            symbol="BTC/USDT",
            adx=20.0,
            atr_z=2.5,  # > high_vol_atr_z (1.5)
            bb_width=0.08,
            ema20_5m=50000.0,
            ema50_5m=50000.0,
            ema20_1h=50000.0,
            ema50_1h=50000.0,
        )

        assert result.regime == RegimeType.HIGH_VOL
        assert result.confidence >= 0.55

    def test_chop_no_trade_low_confidence(self, detector):
        """Test CHOP_NO_TRADE when confidence < threshold"""
        result = detector.detect_regime(
            symbol="BTC/USDT",
            adx=22.0,  # Between range_max and trend_min
            atr_z=1.0,
            bb_width=0.015,
            ema20_5m=50000.0,
            ema50_5m=50000.0,
            ema20_1h=50000.0,
            ema50_1h=50000.0,
        )

        assert result.regime == RegimeType.CHOP_NO_TRADE
        assert result.confidence < 0.55

    def test_confidence_clamped(self, detector):
        """Test confidence is clamped to [0, 1]"""
        result = detector.detect_regime(
            symbol="BTC/USDT",
            adx=100.0,  # Very high ADX
            atr_z=10.0,  # Very high volatility
            bb_width=0.03,
            ema20_5m=50000.0,
            ema50_5m=50000.0,
            ema20_1h=50100.0,
            ema50_1h=50000.0,
        )

        assert 0.0 <= result.confidence <= 1.0

    def test_regime_result_post_init(self):
        """Test RegimeResult confidence clamping in post_init"""
        result = RegimeResult(
            symbol="BTC/USDT",
            regime=RegimeType.TREND,
            confidence=1.5,  # > 1.0
            adx=30.0,
            atr_z=1.0,
            bb_width=0.03,
            ema20_1h=50100.0,
            ema50_1h=50000.0,
            reasons=[],
        )

        assert result.confidence == 1.0

    def test_spread_filter_fail_forces_chop(self, detector):
        """Test CHOP_NO_TRADE when spread filter fails"""
        result = detector.detect_regime(
            symbol="BTC/USDT",
            adx=30.0,
            atr_z=1.0,
            bb_width=0.03,
            ema20_5m=50100.0,
            ema50_5m=50000.0,
            ema20_1h=50200.0,
            ema50_1h=50000.0,
            spread_ok=False,  # Bad spread
        )

        assert result.regime == RegimeType.CHOP_NO_TRADE
        assert result.confidence == 1.0
        assert "Spread filter failed" in result.reasons

    def test_chop_score_forces_chop(self, detector):
        """Test chop_score gate forces CHOP_NO_TRADE"""
        result = detector.detect_regime(
            symbol="BTC/USDT",
            adx=10.0,
            atr_z=0.8,
            bb_width=0.015,
            ema20_5m=50000.0,
            ema50_5m=50000.0,
            ema20_1h=50010.0,
            ema50_1h=50000.0,
            kaufman_er=0.1,
            flip_rate=0.9,
            ema1h_spread_pct=0.0005,
            bb_width_pct_rank=0.1,
        )

        assert result.regime == RegimeType.CHOP_NO_TRADE

    def test_high_vol_precedence_over_chop(self, detector):
        """Test HIGH_VOL has priority over chop_score"""
        result = detector.detect_regime(
            symbol="BTC/USDT",
            adx=10.0,
            atr_z=3.0,  # HIGH_VOL true
            bb_width=0.015,
            ema20_5m=50000.0,
            ema50_5m=50000.0,
            ema20_1h=50010.0,
            ema50_1h=50000.0,
            kaufman_er=0.1,
            flip_rate=0.9,
            ema1h_spread_pct=0.0005,
            bb_width_pct_rank=0.1,
        )

        assert result.regime == RegimeType.HIGH_VOL

    def test_debounce_requires_persistence(self):
        """Test regime requires persistence before switching"""
        cfg = RegimeConfig(
            trend_adx_min=25.0,
            range_adx_max=20.0,
            high_vol_atr_z=1.5,
            confidence_threshold=0.55,
            bb_width_range_min=0.01,
            bb_width_range_max=0.05,
        )
        cfg.chop.regime_persistence_bars = 2
        cfg.chop.regime_switch_cooldown_bars = 0
        cfg.chop.range_requires_extremes = False
        detector = RegimeDetector(cfg)

        first = detector.detect_regime(
            symbol="BTC/USDT",
            adx=30.0,
            atr_z=1.0,
            bb_width=0.03,
            ema20_5m=50100.0,
            ema50_5m=50000.0,
            ema20_1h=50200.0,
            ema50_1h=50000.0,
        )
        assert first.regime == RegimeType.TREND

        # First RANGE proposal should be held
        second = detector.detect_regime(
            symbol="BTC/USDT",
            adx=15.0,
            atr_z=1.0,
            bb_width=0.02,
            ema20_5m=50000.0,
            ema50_5m=50000.0,
            ema20_1h=50000.0,
            ema50_1h=50000.0,
        )
        assert second.regime == RegimeType.TREND

        # Second consecutive RANGE switches
        third = detector.detect_regime(
            symbol="BTC/USDT",
            adx=15.0,
            atr_z=1.0,
            bb_width=0.02,
            ema20_5m=50000.0,
            ema50_5m=50000.0,
            ema20_1h=50000.0,
            ema50_1h=50000.0,
        )
        assert third.regime == RegimeType.RANGE

    def test_trend_direction_bullish(self, detector):
        """Test trend direction is correctly identified as bullish"""
        result = detector.detect_regime(
            symbol="BTC/USDT",
            adx=30.0,
            atr_z=1.0,
            bb_width=0.03,
            ema20_5m=50100.0,
            ema50_5m=50000.0,
            ema20_1h=50200.0,  # Bullish: EMA20 > EMA50
            ema50_1h=50000.0,
        )

        assert result.regime == RegimeType.TREND
        assert result.trend_direction == "bullish"

    def test_trend_direction_bearish(self, detector):
        """Test trend direction is correctly identified as bearish"""
        result = detector.detect_regime(
            symbol="BTC/USDT",
            adx=30.0,
            atr_z=1.0,
            bb_width=0.03,
            ema20_5m=50100.0,
            ema50_5m=50000.0,
            ema20_1h=49800.0,  # Bearish: EMA20 < EMA50
            ema50_1h=50000.0,
        )

        assert result.regime == RegimeType.TREND
        assert result.trend_direction == "bearish"
