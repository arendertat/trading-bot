"""Unit tests for PerformanceTracker"""

import pytest
import numpy as np
from datetime import datetime

from bot.core.performance_tracker import PerformanceTracker, StrategyMetrics, TradeRecord
from bot.core.types import Trade
from bot.core.constants import OrderSide, RegimeType


@pytest.fixture
def tracker():
    """Performance tracker with 50 trade window"""
    return PerformanceTracker(window_trades=50)


@pytest.fixture
def small_tracker():
    """Performance tracker with 10 trade window for testing FIFO"""
    return PerformanceTracker(window_trades=10)


class TestPerformanceTracker:
    """Unit tests for PerformanceTracker"""

    def test_initialization(self, tracker):
        """Test tracker initializes with empty state"""
        assert tracker.window == 50
        assert len(tracker.trades) == 0

    def test_add_single_trade(self, tracker):
        """Test adding a single trade"""
        tracker.add_trade(
            strategy="TrendPullback",
            pnl_r=1.5,
            pnl_usd=150.0,
            fees=2.0,
            funding=0.5
        )

        assert tracker.get_trade_count("TrendPullback") == 1
        assert "TrendPullback" in tracker.get_all_strategy_names()

    def test_add_multiple_trades(self, tracker):
        """Test adding multiple trades to same strategy"""
        for i in range(10):
            tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=1.0 + (i * 0.1),
                pnl_usd=100.0 + (i * 10.0),
                fees=2.0,
                funding=0.5
            )

        assert tracker.get_trade_count("TrendPullback") == 10

    def test_fifo_eviction(self, small_tracker):
        """Test FIFO eviction when window is full"""
        # Add 15 trades to a 10-trade window
        for i in range(15):
            small_tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=float(i),
                pnl_usd=float(i * 100),
                fees=2.0,
                funding=0.0
            )

        # Should only have last 10 trades
        assert small_tracker.get_trade_count("TrendPullback") == 10

        # Verify oldest trades were evicted (first 5 should be gone)
        metrics = small_tracker.get_metrics("TrendPullback")
        assert metrics is not None
        # Average of trades 5-14 is (5+6+7+8+9+10+11+12+13+14) / 10 = 9.5
        assert metrics.avg_r == pytest.approx(9.5, rel=0.01)

    def test_metrics_insufficient_trades(self, tracker):
        """Test metrics returns None when < 10 trades"""
        for i in range(5):
            tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=1.0,
                pnl_usd=100.0,
                fees=2.0,
                funding=0.0
            )

        metrics = tracker.get_metrics("TrendPullback")
        assert metrics is None

    def test_metrics_win_rate_calculation(self, tracker):
        """Test win rate calculated correctly"""
        # Add 20 trades: 12 wins, 8 losses
        for i in range(12):
            tracker.add_trade(strategy="TrendPullback", pnl_r=1.0, pnl_usd=100.0)
        for i in range(8):
            tracker.add_trade(strategy="TrendPullback", pnl_r=-1.0, pnl_usd=-100.0)

        metrics = tracker.get_metrics("TrendPullback")
        assert metrics is not None
        assert metrics.win_rate == pytest.approx(0.6, rel=0.01)  # 12/20 = 0.6
        assert metrics.total_trades == 20

    def test_metrics_avg_r_calculation(self, tracker):
        """Test average R calculated correctly"""
        # Add trades with known R values
        r_values = [1.5, -1.0, 2.0, -0.5, 1.0, 0.5, -0.8, 1.2, 0.8, -1.2]
        for r in r_values:
            tracker.add_trade(strategy="TrendPullback", pnl_r=r, pnl_usd=r * 100)

        metrics = tracker.get_metrics("TrendPullback")
        assert metrics is not None

        expected_avg = np.mean(r_values)
        assert metrics.avg_r == pytest.approx(expected_avg, rel=0.01)

    def test_metrics_expectancy_r_calculation(self, tracker):
        """Test expectancy R calculation with wins and losses"""
        # Add 15 wins with avg 1.5R and 10 losses with avg -1.0R
        for i in range(15):
            tracker.add_trade(strategy="TrendPullback", pnl_r=1.5, pnl_usd=150.0)
        for i in range(10):
            tracker.add_trade(strategy="TrendPullback", pnl_r=-1.0, pnl_usd=-100.0)

        metrics = tracker.get_metrics("TrendPullback")
        assert metrics is not None

        # Expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
        # = (15/25 * 1.5) + (10/25 * -1.0)
        # = (0.6 * 1.5) + (0.4 * -1.0) = 0.9 - 0.4 = 0.5
        assert metrics.expectancy_r == pytest.approx(0.5, rel=0.01)

    def test_metrics_drawdown_calculation(self, tracker):
        """Test max drawdown calculation"""
        # Simulated P&L sequence with drawdown
        pnl_sequence = [100, 50, -50, -100, 50, 100, 150, -50, 100, 200]

        for pnl in pnl_sequence:
            tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=pnl / 100.0,
                pnl_usd=pnl,
                fees=0,
                funding=0
            )

        metrics = tracker.get_metrics("TrendPullback")
        assert metrics is not None

        # Cumulative: [100, 150, 100, 0, 50, 150, 300, 250, 350, 550]
        # Running max: [100, 150, 150, 150, 150, 150, 300, 300, 350, 550]
        # Drawdown: [0, 0, -50, -150, -100, 0, 0, -50, 0, 0]
        # Max DD: -150
        # Total positive: 100+50+50+100+150+100+200 = 750
        # Max DD%: 150/750 = 0.2 = 20%

        assert metrics.max_drawdown_pct == pytest.approx(0.2, rel=0.05)

    def test_metrics_fees_and_funding(self, tracker):
        """Test fees and funding totals"""
        for i in range(10):
            tracker.add_trade(
                strategy="TrendPullback",
                pnl_r=1.0,
                pnl_usd=100.0,
                fees=2.5,
                funding=0.5
            )

        metrics = tracker.get_metrics("TrendPullback")
        assert metrics is not None
        assert metrics.fees_total == pytest.approx(25.0, rel=0.01)  # 10 * 2.5
        assert metrics.funding_total == pytest.approx(5.0, rel=0.01)  # 10 * 0.5

    def test_multiple_strategies(self, tracker):
        """Test tracking multiple strategies independently"""
        # Add trades to Strategy A
        for i in range(15):
            tracker.add_trade(strategy="StrategyA", pnl_r=1.0, pnl_usd=100.0)

        # Add trades to Strategy B
        for i in range(20):
            tracker.add_trade(strategy="StrategyB", pnl_r=0.5, pnl_usd=50.0)

        assert tracker.get_trade_count("StrategyA") == 15
        assert tracker.get_trade_count("StrategyB") == 20

        metrics_a = tracker.get_metrics("StrategyA")
        metrics_b = tracker.get_metrics("StrategyB")

        assert metrics_a is not None
        assert metrics_b is not None
        assert metrics_a.avg_r == pytest.approx(1.0, rel=0.01)
        assert metrics_b.avg_r == pytest.approx(0.5, rel=0.01)

    def test_add_trade_from_record(self, tracker):
        """Test adding trade from Trade dataclass"""
        trade = Trade(
            trade_id="test_001",
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            strategy="TrendPullback",
            regime=RegimeType.TREND,
            entry_time=datetime.utcnow(),
            exit_time=datetime.utcnow(),
            entry_price=50000.0,
            exit_price=51000.0,
            quantity=0.1,
            realized_pnl=100.0,
            fees=2.0,
            funding=0.5,
            r_multiple=1.5
        )

        tracker.add_trade_from_record(trade)

        assert tracker.get_trade_count("TrendPullback") == 1

    def test_clear_strategy(self, tracker):
        """Test clearing trades for a strategy"""
        for i in range(10):
            tracker.add_trade(strategy="TrendPullback", pnl_r=1.0, pnl_usd=100.0)

        assert tracker.get_trade_count("TrendPullback") == 10

        tracker.clear_strategy("TrendPullback")

        assert tracker.get_trade_count("TrendPullback") == 0

    def test_rebuild_from_trades(self, tracker):
        """Test rebuilding tracker from historical trades"""
        # Create historical trades
        trades = []
        for i in range(20):
            trade = Trade(
                trade_id=f"trade_{i}",
                symbol="BTCUSDT",
                side=OrderSide.LONG,
                strategy="TrendPullback",
                regime=RegimeType.TREND,
                entry_time=datetime.utcnow(),
                exit_time=datetime.utcnow(),
                entry_price=50000.0,
                exit_price=50000.0 + (i * 100),
                quantity=0.1,
                realized_pnl=i * 10.0,
                fees=2.0,
                funding=0.5,
                r_multiple=i * 0.1
            )
            trades.append(trade)

        tracker.rebuild_from_trades(trades)

        assert tracker.get_trade_count("TrendPullback") == 20

        metrics = tracker.get_metrics("TrendPullback")
        assert metrics is not None
        assert metrics.total_trades == 20

    def test_confidence_clamping(self, tracker):
        """Test that confidence is clamped to [0, 1]"""
        # Add enough trades to get metrics
        for i in range(15):
            tracker.add_trade(strategy="TrendPullback", pnl_r=2.0, pnl_usd=200.0)

        metrics = tracker.get_metrics("TrendPullback")
        assert metrics is not None
        assert 0.0 <= metrics.confidence <= 1.0

    def test_zero_trades(self, tracker):
        """Test behavior with zero trades"""
        # Check initial state (no strategies)
        assert len(tracker.get_all_strategy_names()) == 0

        # Check metrics for non-existent strategy
        metrics = tracker.get_metrics("NonExistentStrategy")
        assert metrics is None

        # get_trade_count creates empty deque due to defaultdict
        # So we check count is 0 but strategy now exists
        assert tracker.get_trade_count("NonExistentStrategy") == 0
        assert len(tracker.get_all_strategy_names()) == 1  # defaultdict created entry
