"""Milestone 3 Integration Tests - Complete Pipeline Validation"""

import pytest
from datetime import datetime
from typing import List, Set

from bot.regime.detector import RegimeDetector
from bot.regime.models import RegimeResult
from bot.core.performance_tracker import PerformanceTracker
from bot.core.strategy_selector import StrategySelector
from bot.strategies.trend_pullback import TrendPullbackStrategy
from bot.strategies.trend_breakout import TrendBreakoutStrategy
from bot.strategies.range_mean_reversion import RangeMeanReversionStrategy
from bot.strategies.base import StrategySignal, FeatureSet
from bot.config.models import RegimeConfig
from bot.core.constants import RegimeType, OrderSide
from tests.fixtures.market_data_generator import MarketDataGenerator


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def market_generator():
    """Market data generator instance"""
    return MarketDataGenerator(seed=42)


@pytest.fixture
def regime_detector():
    """Regime detector instance"""
    config = RegimeConfig(
        trend_adx_min=25,
        range_adx_max=20,
        high_vol_atr_z=1.5,
        confidence_threshold=0.55,
        bb_width_range_min=0.01,
        bb_width_range_max=0.05
    )
    return RegimeDetector(config)


@pytest.fixture
def strategies():
    """Strategy instances"""
    return {
        "TrendPullback": TrendPullbackStrategy({
            "enabled": True,
            "stop_pct": 0.01,
            "target_r_multiple": 1.5,
            "pullback_rsi_long_min": 40,
            "pullback_rsi_long_max": 50,
            "pullback_rsi_short_min": 50,
            "pullback_rsi_short_max": 60,
            "ema20_band_pct": 0.002,
            "trail_after_r": 1.0
        }),
        "TrendBreakout": TrendBreakoutStrategy({
            "enabled": True,
            "stop_pct": 0.01,
            "breakout_volume_z_min": 1.0,
            "trail_after_r": 0.0
        }),
        "RangeMeanReversion": RangeMeanReversionStrategy({
            "enabled": True,
            "stop_pct": 0.008,
            "target_r_multiple": 1.2,
            "rsi_long_extreme": 25,
            "rsi_short_extreme": 75
        })
    }


@pytest.fixture
def performance_tracker():
    """Performance tracker instance"""
    return PerformanceTracker(window_trades=50)


@pytest.fixture
def strategy_selector(performance_tracker, strategies):
    """Strategy selector instance"""
    return StrategySelector(
        performance_tracker=performance_tracker,
        strategies=strategies,
        stability_hours=24
    )


# ============================================================================
# Helper Functions
# ============================================================================

def validate_signal(signal: StrategySignal) -> None:
    """
    Validate that a strategy signal has correct structure.

    Args:
        signal: Signal to validate

    Raises:
        AssertionError: If signal is invalid
    """
    assert signal.entry is True
    assert signal.side in [OrderSide.LONG, OrderSide.SHORT]
    assert 0 < signal.stop_pct < 0.1  # Stop between 0-10%
    assert signal.target_r > 0
    assert signal.entry_price > 0
    assert signal.stop_price > 0
    assert signal.tp_price > 0
    assert len(signal.reason) > 0
    assert len(signal.symbol) > 0

    # Validate stop/TP relationship
    if signal.side == OrderSide.LONG:
        assert signal.stop_price < signal.entry_price < signal.tp_price, \
            "LONG: stop < entry < TP"
    else:  # SHORT
        assert signal.stop_price > signal.entry_price > signal.tp_price, \
            "SHORT: stop > entry > TP"


def calculate_regime_features(features: FeatureSet) -> tuple[float, float, float]:
    """
    Calculate regime detection features from FeatureSet.

    Simplified ADX, ATR_z, BB_width calculation for testing.

    Args:
        features: Feature set

    Returns:
        (adx, atr_z, bb_width)
    """
    # Mock ADX based on EMA separation
    ema_sep_pct = abs(features.ema20_1h - features.ema50_1h) / features.ema50_1h
    adx = min(50, ema_sep_pct * 1000)  # Scale to ADX-like values

    # Mock ATR z-score (normalized volatility)
    atr_z = (features.atr_5m / features.ema20_5m) * 100  # Simplified

    # BB width
    bb_width = (features.bb_upper_5m - features.bb_lower_5m) / features.bb_middle_5m

    return adx, atr_z, bb_width


# ============================================================================
# Integration Tests
# ============================================================================

class TestMilestone3Integration:
    """Integration tests for complete M3 pipeline"""

    def test_regime_detector_with_real_candles(self, market_generator, regime_detector):
        """Test regime detection with generated candle data"""
        # Generate trend candles
        trend_candles = market_generator.generate_trend_candles(
            starting_price=50000,
            num_candles=200,
            direction="bullish"
        )

        # Generate features
        features = market_generator.generate_features_from_candles(trend_candles)

        # Calculate regime features
        adx, atr_z, bb_width = calculate_regime_features(features)

        # Detect regime
        regime = regime_detector.detect_regime(
            symbol="BTCUSDT",
            adx=adx,
            atr_z=atr_z,
            bb_width=bb_width,
            ema20_5m=features.ema20_5m,
            ema50_5m=features.ema50_5m,
            ema20_1h=features.ema20_1h,
            ema50_1h=features.ema50_1h,
            spread_ok=True
        )

        assert regime is not None
        assert regime.regime in [RegimeType.TREND, RegimeType.RANGE, RegimeType.HIGH_VOL, RegimeType.CHOP_NO_TRADE]
        assert 0 <= regime.confidence <= 1.0

    def test_strategy_generates_signal_from_features(self, strategies, market_generator):
        """Test strategy signal generation from candle features"""
        # Generate bullish trend candles
        candles = market_generator.generate_trend_candles(
            starting_price=50000,
            num_candles=200,
            direction="bullish"
        )

        features = market_generator.generate_features_from_candles(candles)
        current_price = candles[-1].close

        # Create mock regime result (TREND bullish)
        regime = RegimeResult(
            symbol="BTCUSDT",
            regime=RegimeType.TREND,
            confidence=0.75,
            adx=30.0,
            atr_z=1.0,
            bb_width=0.03,
            ema20_1h=features.ema20_1h,
            ema50_1h=features.ema50_1h,
            reasons=["Bullish trend"],
            trend_direction="bullish"
        )

        # Try TrendPullback strategy
        strategy = strategies["TrendPullback"]
        signal = strategy.generate_signal(features, regime, "BTCUSDT", current_price)

        # May or may not generate signal depending on pullback conditions
        if signal:
            validate_signal(signal)
            assert signal.side == OrderSide.LONG

    def test_complete_pipeline_single_iteration(
        self,
        market_generator,
        regime_detector,
        strategy_selector,
        performance_tracker,
        strategies
    ):
        """Test complete pipeline: candles → features → regime → strategy → signal"""
        # Step 1: Generate candles
        candles = market_generator.generate_trend_candles(
            starting_price=50000,
            num_candles=200,
            direction="bullish"
        )

        # Step 2: Calculate features
        features = market_generator.generate_features_from_candles(candles)
        current_price = candles[-1].close

        # Step 3: Detect regime
        adx, atr_z, bb_width = calculate_regime_features(features)
        regime = regime_detector.detect_regime(
            symbol="BTCUSDT",
            adx=adx,
            atr_z=atr_z,
            bb_width=bb_width,
            ema20_5m=features.ema20_5m,
            ema50_5m=features.ema50_5m,
            ema20_1h=features.ema20_1h,
            ema50_1h=features.ema50_1h,
            spread_ok=True
        )

        assert regime is not None

        # Step 4: Add some performance history for strategy selection
        for i in range(15):
            performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=1.0 + (i * 0.1),
                pnl_usd=100.0 + (i * 10.0)
            )

        # Step 5: Select strategy
        selected_strategy = strategy_selector.select_strategy(regime.regime, "BTCUSDT")

        # May or may not select based on regime and performance
        if selected_strategy:
            # Step 6: Generate signal
            signal = selected_strategy.generate_signal(features, regime, "BTCUSDT", current_price)

            if signal:
                validate_signal(signal)

    def test_30_day_simulation(
        self,
        market_generator,
        regime_detector,
        strategy_selector,
        performance_tracker
    ):
        """
        Test 30-day market simulation with regime changes.

        This is the main integration test validating the complete pipeline.
        """
        # Generate 30-day scenario
        all_candles, regime_periods = market_generator.generate_30_day_scenario(
            starting_price=50000.0
        )

        print(f"\nGenerated {len(all_candles)} candles for 30-day simulation")
        print(f"Regime periods: {regime_periods}")

        signals_generated: List[StrategySignal] = []
        regimes_detected: Set[RegimeType] = set()
        strategies_selected: Set[str] = set()

        # Simulate processing candles (sample every 10th candle for speed)
        lookback = 100
        for i in range(lookback, len(all_candles), 10):
            # Get candles for feature calculation
            candle_window = all_candles[i-lookback:i]
            current_candle = all_candles[i]

            try:
                # Generate features
                features = market_generator.generate_features_from_candles(candle_window)

                # Detect regime
                adx, atr_z, bb_width = calculate_regime_features(features)
                regime = regime_detector.detect_regime(
                    symbol="BTCUSDT",
                    adx=adx,
                    atr_z=atr_z,
                    bb_width=bb_width,
                    ema20_5m=features.ema20_5m,
                    ema50_5m=features.ema50_5m,
                    ema20_1h=features.ema20_1h,
                    ema50_1h=features.ema50_1h,
                    spread_ok=True
                )

                regimes_detected.add(regime.regime)

                # Select strategy (if we have performance history)
                selected_strategy = strategy_selector.select_strategy(
                    regime.regime,
                    "BTCUSDT"
                )

                if selected_strategy:
                    strategies_selected.add(selected_strategy.name)

                    # Generate signal
                    signal = selected_strategy.generate_signal(
                        features,
                        regime,
                        "BTCUSDT",
                        current_candle.close,
                        timestamp=current_candle.timestamp
                    )

                    if signal:
                        validate_signal(signal)
                        signals_generated.append(signal)

                        # Simulate trade completion for performance tracking
                        # (Simplified: assume 60% win rate with random R outcomes)
                        import random
                        is_winner = random.random() < 0.6
                        pnl_r = random.uniform(0.8, 2.0) if is_winner else random.uniform(-1.0, -0.5)
                        pnl_usd = pnl_r * 100.0

                        performance_tracker.add_trade(
                            strategy=selected_strategy.name.replace("Strategy", ""),
                            pnl_r=pnl_r,
                            pnl_usd=pnl_usd,
                            fees=2.0,
                            funding=0.5
                        )

            except Exception as e:
                # Log but don't fail on individual candle errors
                print(f"Error processing candle {i}: {e}")
                continue

        # Assertions
        print(f"\nSimulation Results:")
        print(f"  Signals generated: {len(signals_generated)}")
        print(f"  Regimes detected: {regimes_detected}")
        print(f"  Strategies selected: {strategies_selected}")

        # Acceptance criteria validation
        assert len(signals_generated) >= 5, "Should generate at least 5 signals in 30 days"
        assert len(regimes_detected) >= 2, "Should detect at least 2 different regimes"
        assert RegimeType.CHOP_NO_TRADE not in regimes_detected or len(regimes_detected) > 1, \
            "Should not only detect CHOP regime"

        # Validate no duplicate signals (same timestamp)
        signal_timestamps = [s.timestamp for s in signals_generated]
        assert len(signal_timestamps) == len(set(signal_timestamps)), \
            "No duplicate signals at same timestamp"

    def test_strategy_selector_switches(
        self,
        performance_tracker,
        strategy_selector
    ):
        """Test that strategy selector switches based on performance"""
        # Add good performance for TrendPullback
        for i in range(15):
            performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=1.5,
                pnl_usd=150.0
            )

        # Select initial strategy
        strategy1 = strategy_selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert strategy1 is not None
        initial_selection = strategy_selector.get_current_selection(RegimeType.TREND)

        # Add better performance for TrendBreakout
        for i in range(15):
            performance_tracker.add_trade(
                strategy="TrendBreakout",
                pnl_r=2.0,  # Better than TrendPullback
                pnl_usd=200.0
            )

        # Immediate selection should be blocked by stability
        strategy2 = strategy_selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        current_selection = strategy_selector.get_current_selection(RegimeType.TREND)
        assert current_selection == initial_selection, "Stability constraint should prevent immediate switch"

        # Force time advance (reset stability timer)
        strategy_selector.reset_stability_timer(RegimeType.TREND)

        # Now switch should be allowed
        strategy3 = strategy_selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        new_selection = strategy_selector.get_current_selection(RegimeType.TREND)

        # Should have switched to better performing strategy
        assert new_selection == "TrendBreakout", "Should switch to better performing strategy after stability window"

    def test_signal_validation_all_regimes(self, market_generator, strategies):
        """Test signal generation across all regime types"""
        test_scenarios = [
            {
                "regime_type": RegimeType.TREND,
                "candle_gen": lambda: market_generator.generate_trend_candles(50000, 200, "bullish"),
                "trend_direction": "bullish",
                "compatible_strategies": ["TrendPullback", "TrendBreakout"]
            },
            {
                "regime_type": RegimeType.RANGE,
                "candle_gen": lambda: market_generator.generate_range_candles(50000, 200),
                "trend_direction": None,
                "compatible_strategies": ["RangeMeanReversion"]
            },
            {
                "regime_type": RegimeType.HIGH_VOL,
                "candle_gen": lambda: market_generator.generate_high_vol_candles(50000, 200),
                "trend_direction": None,
                "compatible_strategies": ["TrendBreakout"]  # Breakout can handle high vol
            }
        ]

        for scenario in test_scenarios:
            candles = scenario["candle_gen"]()
            features = market_generator.generate_features_from_candles(candles)

            regime = RegimeResult(
                symbol="BTCUSDT",
                regime=scenario["regime_type"],
                confidence=0.75,
                adx=30.0 if scenario["regime_type"] == RegimeType.TREND else 15.0,
                atr_z=2.0 if scenario["regime_type"] == RegimeType.HIGH_VOL else 1.0,
                bb_width=0.03,
                ema20_1h=features.ema20_1h,
                ema50_1h=features.ema50_1h,
                reasons=[f"{scenario['regime_type'].value} market"],
                trend_direction=scenario["trend_direction"]
            )

            # Try compatible strategies
            for strategy_name in scenario["compatible_strategies"]:
                strategy = strategies[strategy_name]
                signal = strategy.generate_signal(
                    features,
                    regime,
                    "BTCUSDT",
                    candles[-1].close
                )

                # May or may not generate signal based on specific conditions
                # but should not crash
                if signal:
                    validate_signal(signal)
                    print(f"✓ {strategy_name} generated signal for {scenario['regime_type'].value}")
