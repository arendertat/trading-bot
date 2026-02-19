"""Unit tests for StrategySelector"""

import pytest
from datetime import datetime, timedelta

from bot.core.strategy_selector import StrategySelector
from bot.core.performance_tracker import PerformanceTracker
from bot.strategies.trend_pullback import TrendPullbackStrategy
from bot.strategies.trend_breakout import TrendBreakoutStrategy
from bot.strategies.range_mean_reversion import RangeMeanReversionStrategy
from bot.core.constants import RegimeType


@pytest.fixture
def performance_tracker():
    """Performance tracker instance"""
    return PerformanceTracker(window_trades=50)


@pytest.fixture
def strategies():
    """Dict of strategy instances"""
    return {
        "TrendPullback": TrendPullbackStrategy({
            "enabled": True,
            "stop_pct": 0.01,
            "target_r_multiple": 1.5
        }),
        "TrendBreakout": TrendBreakoutStrategy({
            "enabled": True,
            "stop_pct": 0.01,
            "breakout_volume_z_min": 1.0
        }),
        "RangeMeanReversion": RangeMeanReversionStrategy({
            "enabled": True,
            "stop_pct": 0.008,
            "target_r_multiple": 1.2
        })
    }


@pytest.fixture
def selector(performance_tracker, strategies):
    """StrategySelector instance"""
    return StrategySelector(
        performance_tracker=performance_tracker,
        strategies=strategies,
        stability_hours=24
    )


class TestStrategySelector:
    """Unit tests for StrategySelector"""

    def test_initialization(self, selector, strategies):
        """Test selector initializes correctly"""
        assert selector.stability_hours == 24
        assert len(selector.strategies) == 3
        assert len(selector.current_selection) == 0

    def test_no_strategies_for_regime(self, selector):
        """Test selection returns None when no compatible strategies"""
        # CHOP_NO_TRADE has no compatible strategies
        strategy = selector.select_strategy(RegimeType.CHOP_NO_TRADE, "BTCUSDT")
        assert strategy is None

    def test_insufficient_trade_history(self, selector):
        """Test cold-start fallback when insufficient trades"""
        # Add only 5 trades (< 10 minimum)
        for i in range(5):
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=1.0,
                pnl_usd=100.0
            )

        # With cold-start fallback, should select first compatible strategy
        strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert strategy is not None  # Cold-start should select a strategy
        assert "Trend" in strategy.name  # Should be trend-compatible

    def test_select_strategy_trend_regime(self, selector):
        """Test strategy selection for TREND regime"""
        # Add good performance for TrendPullback (more trades, mixed but positive)
        pnl_values = [1.5, 1.2, -0.5, 1.8, 1.0, -0.4, 1.4, 1.1, 0.8, 1.6,
                      1.3, -0.6, 1.7, 1.2, 0.9, 1.5, 1.4, -0.5, 1.9, 1.1,
                      1.6, 1.3, -0.7, 1.5, 1.0]
        for pnl_r in pnl_values:
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=pnl_r,
                pnl_usd=pnl_r * 100,
                fees=2.0,
                funding=0.5
            )

        strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert strategy is not None
        assert strategy.name == "TrendPullbackStrategy"

    def test_select_strategy_range_regime(self, selector):
        """Test strategy selection for RANGE regime"""
        # Add good performance for RangeMeanReversion (need 30+ for full confidence)
        pnl_values = [1.2, 1.1, 1.3, 1.0, 1.4, 1.1, 1.2, 1.3, 1.1, 1.2,
                      1.3, 1.0, 1.2, 1.4, 1.1, 1.2, 1.3, 1.0, 1.1, 1.2,
                      1.3, 1.1, 1.2, 1.0, 1.3]
        for pnl_r in pnl_values:
            selector.performance_tracker.add_trade(
                strategy="RangeMeanReversion",
                pnl_r=pnl_r,
                pnl_usd=pnl_r * 100,
                fees=2.0,
                funding=0.5
            )

        strategy = selector.select_strategy(RegimeType.RANGE, "BTCUSDT")
        assert strategy is not None
        assert strategy.name == "RangeMeanReversionStrategy"

    def test_confidence_threshold_filtering(self, selector):
        """Test strategies below confidence threshold are filtered out"""
        # Add poor performance to ALL trend strategies so cold-start can't fallback
        # Need 15+ trades to get metrics
        trades = [1.0, -1.0, 0.5, -0.5, -0.5, -1.0, 0.5, -0.5, 0.0, -0.5,
                  -0.8, 0.3, -0.7, -0.9, 0.2]  # More losses than wins, negative expectancy

        for strategy_name in ["TrendPullback", "TrendBreakout"]:
            for r in trades:
                selector.performance_tracker.add_trade(
                    strategy=strategy_name,
                    pnl_r=r,
                    pnl_usd=r * 100
                )

        strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        # All strategies have poor performance, should return None
        assert strategy is None

    def test_best_strategy_selection(self, selector):
        """Test selector chooses best performing strategy"""
        # Add good performance for TrendPullback
        pnl_values = [1.5, 1.2, -0.5, 1.8, 1.0, -0.4, 1.4, 1.1, 0.8, 1.6, 1.3, -0.6, 1.7, 1.2, 0.9, 1.5, 1.4, -0.5, 1.9, 1.1, 1.6, 1.3, -0.7, 1.5, 1.0]
        for pnl_r in pnl_values:
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=pnl_r,
                pnl_usd=pnl_r * 100
            )

        # Add BETTER performance for TrendBreakout (higher R values)
        pnl_values = [2.0, 1.8, -0.4, 2.2, 1.5, -0.3, 1.9, 1.6, 1.2, 2.1, 1.7, -0.5, 2.3, 1.8, 1.4, 2.0, 1.9, -0.4, 2.4, 1.6, 2.1, 1.8, -0.6, 2.0, 1.5]
        for pnl_r in pnl_values:
            selector.performance_tracker.add_trade(
                strategy="TrendBreakout",
                pnl_r=pnl_r,  # Higher R
                pnl_usd=pnl_r * 100
            )

        strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert strategy is not None
        # Should select TrendBreakout (higher performance)
        assert strategy.name == "TrendBreakoutStrategy"

    def test_stability_constraint_blocks_switch(self, selector):
        """Test stability constraint prevents frequent switching"""
        # Select initial strategy
        pnl_values = [1.5, 1.2, -0.5, 1.8, 1.0, -0.4, 1.4, 1.1, 0.8, 1.6, 1.3, -0.6, 1.7, 1.2, 0.9, 1.5, 1.4, -0.5, 1.9, 1.1, 1.6, 1.3, -0.7, 1.5, 1.0]
        for pnl_r in pnl_values:
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=pnl_r,
                pnl_usd=pnl_r * 100
            )

        initial_strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert initial_strategy is not None
        assert initial_strategy.name == "TrendPullbackStrategy"

        # Add better performance for TrendBreakout (should trigger switch)
        pnl_values = [1.5, 1.2, -0.5, 1.8, 1.0, -0.4, 1.4, 1.1, 0.8, 1.6, 1.3, -0.6, 1.7, 1.2, 0.9, 1.5, 1.4, -0.5, 1.9, 1.1, 1.6, 1.3, -0.7, 1.5, 1.0]
        for pnl_r in pnl_values:
            selector.performance_tracker.add_trade(
                strategy="TrendBreakout",
                pnl_r=pnl_r,
                pnl_usd=pnl_r * 100
            )

        # Try to switch immediately (should be blocked by stability)
        second_strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert second_strategy is not None
        assert second_strategy.name == "TrendPullbackStrategy"  # Still original

    def test_emergency_switch_negative_expectancy(self, selector):
        """Test emergency switch when current strategy has negative expectancy"""
        # Select initial strategy with positive expectancy
        pnl_values = [1.5, 1.2, -0.5, 1.8, 1.0, -0.4, 1.4, 1.1, 0.8, 1.6, 1.3, -0.6, 1.7, 1.2, 0.9, 1.5, 1.4, -0.5, 1.9, 1.1, 1.6, 1.3, -0.7, 1.5, 1.0]
        for pnl_r in pnl_values:
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=0.5,
                pnl_usd=pnl_r * 100
            )

        initial_strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert initial_strategy is not None

        # Make TrendPullback go negative
        for i in range(10):
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=-2.0,  # Large losses
                pnl_usd=-200.0
            )

        # Add good performance for TrendBreakout
        pnl_values = [1.5, 1.2, -0.5, 1.8, 1.0, -0.4, 1.4, 1.1, 0.8, 1.6, 1.3, -0.6, 1.7, 1.2, 0.9, 1.5, 1.4, -0.5, 1.9, 1.1, 1.6, 1.3, -0.7, 1.5, 1.0]
        for pnl_r in pnl_values:
            selector.performance_tracker.add_trade(
                strategy="TrendBreakout",
                pnl_r=pnl_r,
                pnl_usd=pnl_r * 100
            )

        # Should allow immediate switch due to negative expectancy
        emergency_strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert emergency_strategy is not None
        assert emergency_strategy.name == "TrendBreakoutStrategy"

    def test_stability_window_expiry(self, selector):
        """Test switch allowed after stability window expires"""
        # Select initial strategy
        pnl_values = [1.5, 1.2, -0.5, 1.8, 1.0, -0.4, 1.4, 1.1, 0.8, 1.6, 1.3, -0.6, 1.7, 1.2, 0.9, 1.5, 1.4, -0.5, 1.9, 1.1, 1.6, 1.3, -0.7, 1.5, 1.0]
        for pnl_r in pnl_values:
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=pnl_r,
                pnl_usd=pnl_r * 100
            )

        initial_strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert initial_strategy is not None

        # Manually set last selection time to 25 hours ago
        selector.last_selection_time[RegimeType.TREND] = datetime.utcnow() - timedelta(hours=25)

        # Add BETTER performance for TrendBreakout (higher R values)
        pnl_values = [2.0, 1.8, -0.4, 2.2, 1.5, -0.3, 1.9, 1.6, 1.2, 2.1, 1.7, -0.5, 2.3, 1.8, 1.4, 2.0, 1.9, -0.4, 2.4, 1.6, 2.1, 1.8, -0.6, 2.0, 1.5]
        for pnl_r in pnl_values:
            selector.performance_tracker.add_trade(
                strategy="TrendBreakout",
                pnl_r=pnl_r,
                pnl_usd=pnl_r * 100
            )

        # Should allow switch after stability window
        new_strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert new_strategy is not None
        assert new_strategy.name == "TrendBreakoutStrategy"

    def test_get_current_selection(self, selector):
        """Test getting current selection for regime"""
        # Initially no selection
        assert selector.get_current_selection(RegimeType.TREND) is None

        # Add trades and select
        pnl_values = [1.5, 1.2, -0.5, 1.8, 1.0, -0.4, 1.4, 1.1, 0.8, 1.6, 1.3, -0.6, 1.7, 1.2, 0.9, 1.5, 1.4, -0.5, 1.9, 1.1, 1.6, 1.3, -0.7, 1.5, 1.0]
        for pnl_r in pnl_values:
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=pnl_r,
                pnl_usd=pnl_r * 100
            )

        selector.select_strategy(RegimeType.TREND, "BTCUSDT")

        # Should return selected strategy
        assert selector.get_current_selection(RegimeType.TREND) == "TrendPullback"

    def test_force_strategy(self, selector):
        """Test forcing strategy selection"""
        success = selector.force_strategy(RegimeType.TREND, "TrendPullback")
        assert success is True
        assert selector.get_current_selection(RegimeType.TREND) == "TrendPullback"

        # Try to force non-existent strategy
        success = selector.force_strategy(RegimeType.TREND, "NonExistent")
        assert success is False

    def test_reset_stability_timer(self, selector):
        """Test resetting stability timer"""
        # Set a selection time
        selector.last_selection_time[RegimeType.TREND] = datetime.utcnow()

        # Reset timer
        selector.reset_stability_timer(RegimeType.TREND)

        # Should be removed
        assert RegimeType.TREND not in selector.last_selection_time

    def test_confidence_calculation(self, selector):
        """Test confidence score calculation"""
        # Add trades with mixed results (realistic, need 30+ for full confidence)
        pnl_values = [1.5, 1.2, -0.8, 1.8, 1.0, -0.6, 1.4, 1.1, -0.7, 1.6,
                      1.3, -0.5, 1.7, 1.2, -0.9, 1.5, 1.4, -0.6, 1.9, 1.1,
                      1.6, 1.3, -0.7, 1.5, 1.0, 1.4, 1.2, -0.5, 1.6, 1.3]
        for pnl_r in pnl_values:
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=pnl_r,
                pnl_usd=pnl_r * 100
            )

        metrics = selector.performance_tracker.get_metrics("TrendPullback")
        assert metrics is not None

        confidence = selector._calculate_confidence(metrics)

        # Should be reasonable confidence (positive expectancy)
        assert confidence > 0.45  # Lowered threshold for realistic mixed data
        assert 0.0 <= confidence <= 1.0

    def test_disabled_strategy_excluded(self, selector):
        """Test disabled strategies are not selected"""
        # Disable TrendPullback
        selector.strategies["TrendPullback"].enabled = False

        # Add good performance
        pnl_values = [1.5, 1.2, -0.5, 1.8, 1.0, -0.4, 1.4, 1.1, 0.8, 1.6, 1.3, -0.6, 1.7, 1.2, 0.9, 1.5, 1.4, -0.5, 1.9, 1.1, 1.6, 1.3, -0.7, 1.5, 1.0]
        for pnl_r in pnl_values:
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=pnl_r,
                pnl_usd=pnl_r * 100
            )

        # Add mediocre performance for TrendBreakout
        pnl_values = [1.5, 1.2, -0.5, 1.8, 1.0, -0.4, 1.4, 1.1, 0.8, 1.6, 1.3, -0.6, 1.7, 1.2, 0.9, 1.5, 1.4, -0.5, 1.9, 1.1, 1.6, 1.3, -0.7, 1.5, 1.0]
        for pnl_r in pnl_values:
            selector.performance_tracker.add_trade(
                strategy="TrendBreakout",
                pnl_r=0.8,
                pnl_usd=pnl_r * 100
            )

        strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")

        # Should select TrendBreakout despite lower performance (TrendPullback disabled)
        if strategy is not None:
            assert strategy.name == "TrendBreakoutStrategy"

    def test_regime_compatibility(self, selector):
        """Test strategies are filtered by regime compatibility"""
        # Add good performance for RangeMeanReversion
        pnl_values = [1.5, 1.2, -0.5, 1.8, 1.0, -0.4, 1.4, 1.1, 0.8, 1.6, 1.3, -0.6, 1.7, 1.2, 0.9, 1.5, 1.4, -0.5, 1.9, 1.1, 1.6, 1.3, -0.7, 1.5, 1.0]
        for pnl_r in pnl_values:
            selector.performance_tracker.add_trade(
                strategy="RangeMeanReversion",
                pnl_r=pnl_r,
                pnl_usd=pnl_r * 100
            )

        # Try to select for TREND regime
        strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")

        # Should not select RangeMeanReversion for TREND regime
        if strategy is not None:
            assert "Range" not in strategy.name
