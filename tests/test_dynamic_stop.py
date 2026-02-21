"""Tests for ATR-based dynamic stop per regime."""

from bot.strategies.trend_pullback import TrendPullbackStrategy
from bot.core.constants import OrderSide, RegimeType
from bot.regime.models import RegimeResult


def test_dynamic_stop_regime_multiplier():
    config = {
        "enabled": True,
        "dynamic_stop_enabled": True,
        "stop_atr_multiplier": 1.5,
        "stop_atr_multiplier_by_regime": {
            "trend": 2.0,
            "range": 1.0,
            "high_vol": 3.0,
        },
    }
    strategy = TrendPullbackStrategy(config)
    regime = RegimeResult(
        symbol="BTCUSDT",
        regime=RegimeType.TREND,
        confidence=0.8,
        adx=30.0,
        atr_z=1.0,
        bb_width=0.03,
        ema20_1h=100.0,
        ema50_1h=90.0,
        reasons=[],
        trend_direction="bullish",
    )

    entry = 100.0
    atr = 2.0
    stop = strategy.calculate_stop_loss(entry, OrderSide.LONG, atr, regime)
    assert abs(stop - (entry - 4.0)) < 1e-6
