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
        """Test selection returns None when insufficient trades"""
        # Add only 5 trades (< 10 minimum)
        for i in range(5):
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=1.0,
                pnl_usd=100.0
            )

        strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert strategy is None

    def test_select_strategy_trend_regime(self, selector):
        """Test strategy selection for TREND regime"""
        # Add good performance for TrendPullback
        for i in range(20):
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=1.5,
                pnl_usd=150.0,
                fees=2.0,
                funding=0.5
            )

        strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert strategy is not None
        assert strategy.name == "TrendPullbackStrategy"

    def test_select_strategy_range_regime(self, selector):
        """Test strategy selection for RANGE regime"""
        # Add good performance for RangeMeanReversion
        for i in range(20):
            selector.performance_tracker.add_trade(
                strategy="RangeMeanReversion",
                pnl_r=1.2,
                pnl_usd=120.0,
                fees=2.0,
                funding=0.5
            )

        strategy = selector.select_strategy(RegimeType.RANGE, "BTCUSDT")
        assert strategy is not None
        assert strategy.name == "RangeMeanReversionStrategy"

    def test_confidence_threshold_filtering(self, selector):
        """Test strategies below confidence threshold are filtered out"""
        # Add mediocre performance (low expectancy, high drawdown)
        trades = [1.0, -1.0, 0.5, -0.5, -0.5, -1.0, 0.5, -0.5, 0.0, -0.5]
        for r in trades:
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=r,
                pnl_usd=r * 100
            )

        strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        # Should return None due to low confidence
        assert strategy is None

    def test_best_strategy_selection(self, selector):
        """Test selector chooses best performing strategy"""
        # Add good performance for TrendPullback
        for i in range(15):
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=1.5,
                pnl_usd=150.0
            )

        # Add better performance for TrendBreakout
        for i in range(15):
            selector.performance_tracker.add_trade(
                strategy="TrendBreakout",
                pnl_r=2.0,  # Higher R
                pnl_usd=200.0
            )

        strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert strategy is not None
        # Should select TrendBreakout (higher performance)
        assert strategy.name == "TrendBreakoutStrategy"

    def test_stability_constraint_blocks_switch(self, selector):
        """Test stability constraint prevents frequent switching"""
        # Select initial strategy
        for i in range(15):
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=1.0,
                pnl_usd=100.0
            )

        initial_strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert initial_strategy is not None
        assert initial_strategy.name == "TrendPullbackStrategy"

        # Add better performance for TrendBreakout (should trigger switch)
        for i in range(15):
            selector.performance_tracker.add_trade(
                strategy="TrendBreakout",
                pnl_r=2.0,
                pnl_usd=200.0
            )

        # Try to switch immediately (should be blocked by stability)
        second_strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert second_strategy is not None
        assert second_strategy.name == "TrendPullbackStrategy"  # Still original

    def test_emergency_switch_negative_expectancy(self, selector):
        """Test emergency switch when current strategy has negative expectancy"""
        # Select initial strategy with positive expectancy
        for i in range(15):
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=0.5,
                pnl_usd=50.0
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
        for i in range(15):
            selector.performance_tracker.add_trade(
                strategy="TrendBreakout",
                pnl_r=1.5,
                pnl_usd=150.0
            )

        # Should allow immediate switch due to negative expectancy
        emergency_strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert emergency_strategy is not None
        assert emergency_strategy.name == "TrendBreakoutStrategy"

    def test_stability_window_expiry(self, selector):
        """Test switch allowed after stability window expires"""
        # Select initial strategy
        for i in range(15):
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=1.0,
                pnl_usd=100.0
            )

        initial_strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")
        assert initial_strategy is not None

        # Manually set last selection time to 25 hours ago
        selector.last_selection_time[RegimeType.TREND] = datetime.utcnow() - timedelta(hours=25)

        # Add better performance for TrendBreakout
        for i in range(15):
            selector.performance_tracker.add_trade(
                strategy="TrendBreakout",
                pnl_r=2.0,
                pnl_usd=200.0
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
        for i in range(15):
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=1.0,
                pnl_usd=100.0
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
        # Add trades with known characteristics
        for i in range(20):
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=1.5,  # Good expectancy
                pnl_usd=150.0
            )

        metrics = selector.performance_tracker.get_metrics("TrendPullback")
        assert metrics is not None

        confidence = selector._calculate_confidence(metrics)

        # Should be high confidence (good expectancy, low drawdown)
        assert confidence > 0.6
        assert 0.0 <= confidence <= 1.0

    def test_disabled_strategy_excluded(self, selector):
        """Test disabled strategies are not selected"""
        # Disable TrendPullback
        selector.strategies["TrendPullback"].enabled = False

        # Add good performance
        for i in range(15):
            selector.performance_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=2.0,
                pnl_usd=200.0
            )

        # Add mediocre performance for TrendBreakout
        for i in range(15):
            selector.performance_tracker.add_trade(
                strategy="TrendBreakout",
                pnl_r=0.8,
                pnl_usd=80.0
            )

        strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")

        # Should select TrendBreakout despite lower performance (TrendPullback disabled)
        if strategy is not None:
            assert strategy.name == "TrendBreakoutStrategy"

    def test_regime_compatibility(self, selector):
        """Test strategies are filtered by regime compatibility"""
        # Add good performance for RangeMeanReversion
        for i in range(15):
            selector.performance_tracker.add_trade(
                strategy="RangeMeanReversion",
                pnl_r=1.5,
                pnl_usd=150.0
            )

        # Try to select for TREND regime
        strategy = selector.select_strategy(RegimeType.TREND, "BTCUSDT")

        # Should not select RangeMeanReversion for TREND regime
        if strategy is not None:
            assert "Range" not in strategy.name
