"""Tests for correlation filter"""

from datetime import datetime

import numpy as np
import pytest

from bot.config.models import RiskConfig
from bot.core.constants import OrderSide, PositionStatus
from bot.core.types import Position
from bot.risk.correlation_filter import CorrelationFilter


@pytest.fixture
def default_config():
    """Create default risk configuration"""
    return RiskConfig(
        correlation_threshold=0.85,
        hedge_corr_max=0.6,
    )


@pytest.fixture
def correlation_filter(default_config):
    """Create correlation filter instance"""
    return CorrelationFilter(default_config)


@pytest.fixture
def sample_price_data():
    """Create sample price data for correlation calculation"""
    np.random.seed(42)

    # Generate 72 hourly prices for multiple symbols
    # Create a common trend component for high correlation
    common_trend = np.cumsum(np.random.normal(0.001, 0.02, 72))

    # BTC and ETH - highly correlated (share common trend)
    btc_prices = 50000 * (1 + common_trend + np.random.normal(0, 0.005, 72))
    eth_prices = 3000 * (1 + common_trend + np.random.normal(0, 0.005, 72))

    # SOL - moderate correlation with BTC
    sol_prices = 100 * (1 + 0.5 * common_trend + np.random.normal(0, 0.01, 72))

    # MATIC - low correlation (mostly random)
    matic_prices = 1.0 * (1 + np.cumsum(np.random.normal(0, 0.02, 72)))

    return {
        "BTCUSDT": btc_prices,
        "ETHUSDT": eth_prices,
        "SOLUSDT": sol_prices,
        "MATICUSDT": matic_prices,
    }


@pytest.fixture
def btc_long_position():
    """Create BTC long position"""
    return Position(
        symbol="BTCUSDT",
        side=OrderSide.LONG,
        entry_price=50000.0,
        quantity=0.1,
        notional=5000.0,
        leverage=2.0,
        margin=2500.0,
        stop_price=49000.0,
        tp_price=52000.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        entry_time=datetime.utcnow(),
        trade_id="trade_1",
        status=PositionStatus.OPEN,
    )


@pytest.fixture
def eth_short_position():
    """Create ETH short position"""
    return Position(
        symbol="ETHUSDT",
        side=OrderSide.SHORT,
        entry_price=3000.0,
        quantity=1.0,
        notional=3000.0,
        leverage=1.5,
        margin=2000.0,
        stop_price=3150.0,
        tp_price=2800.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        entry_time=datetime.utcnow(),
        trade_id="trade_2",
        status=PositionStatus.OPEN,
    )


class TestCorrelationCalculation:
    """Test correlation calculation functionality"""

    def test_perfect_positive_correlation(self, correlation_filter):
        """Test with perfectly correlated series"""
        prices_a = np.array([100, 101, 102, 103, 104, 105])
        prices_b = np.array([200, 202, 204, 206, 208, 210])  # 2x of A

        corr = correlation_filter._calculate_correlation(prices_a, prices_b)

        assert corr == pytest.approx(1.0, abs=0.01)

    def test_perfect_negative_correlation(self, correlation_filter):
        """Test with perfectly negatively correlated series"""
        # When A goes up, B goes down by same magnitude
        prices_a = np.array([100.0, 102.0, 101.0, 103.0, 102.0, 104.0])
        prices_b = np.array([200.0, 198.0, 199.0, 197.0, 198.0, 196.0])  # Opposite moves

        corr = correlation_filter._calculate_correlation(prices_a, prices_b)

        # Should be close to -1
        assert corr < -0.95

    def test_zero_correlation(self, correlation_filter):
        """Test with uncorrelated series"""
        np.random.seed(42)
        prices_a = np.cumsum(np.random.normal(0, 1, 100)) + 100
        prices_b = np.cumsum(np.random.normal(0, 1, 100)) + 200

        corr = correlation_filter._calculate_correlation(prices_a, prices_b)

        # Random walk should have low correlation
        assert abs(corr) < 0.3

    def test_insufficient_data(self, correlation_filter):
        """Test with insufficient data points"""
        prices_a = np.array([100])
        prices_b = np.array([200])

        corr = correlation_filter._calculate_correlation(prices_a, prices_b)

        assert corr == 0.0

    def test_mismatched_lengths(self, correlation_filter):
        """Test with mismatched array lengths"""
        prices_a = np.array([100, 101, 102, 103])
        prices_b = np.array([200, 202, 204, 206, 208, 210])

        # Should handle gracefully by using minimum length
        corr = correlation_filter._calculate_correlation(prices_a, prices_b)

        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0

    def test_nan_handling(self, correlation_filter):
        """Test with NaN values in data"""
        prices_a = np.array([100, 101, np.nan, 103, 104])
        prices_b = np.array([200, 202, 204, 206, 208])

        corr = correlation_filter._calculate_correlation(prices_a, prices_b)

        # Should handle NaN gracefully
        assert isinstance(corr, float)


class TestCorrelationMatrixUpdate:
    """Test correlation matrix update"""

    def test_update_with_sample_data(self, correlation_filter, sample_price_data):
        """Test updating correlation matrix with sample data"""
        correlation_filter.update_correlation_matrix(sample_price_data)

        # Check that correlations were calculated
        assert len(correlation_filter.correlation_cache) > 0

        # BTC-ETH should be highly correlated (both trending up)
        btc_eth_corr = abs(correlation_filter.get_correlation("BTCUSDT", "ETHUSDT"))
        assert btc_eth_corr > 0.7

    def test_update_with_single_symbol(self, correlation_filter):
        """Test with only one symbol (no pairs)"""
        price_data = {"BTCUSDT": np.random.normal(50000, 100, 72)}

        correlation_filter.update_correlation_matrix(price_data)

        # Should not crash, no correlations calculated
        assert len(correlation_filter.correlation_cache) == 0

    def test_bidirectional_storage(self, correlation_filter, sample_price_data):
        """Test that correlations are stored bidirectionally"""
        correlation_filter.update_correlation_matrix(sample_price_data)

        # Check both directions
        corr_ab = correlation_filter.get_correlation("BTCUSDT", "ETHUSDT")
        corr_ba = correlation_filter.get_correlation("ETHUSDT", "BTCUSDT")

        assert corr_ab == corr_ba

    def test_get_uncached_correlation(self, correlation_filter):
        """Test getting correlation that hasn't been calculated"""
        corr = correlation_filter.get_correlation("BTCUSDT", "UNKNOWN")

        assert corr == 0.0


class TestCorrelationFilter:
    """Test correlation filtering logic"""

    def test_no_existing_positions(self, correlation_filter):
        """Test with no existing positions (should always pass)"""
        approved, reason = correlation_filter.check_correlation_filter(
            new_symbol="BTCUSDT",
            new_side=OrderSide.LONG,
            open_positions=[],
        )

        assert approved is True
        assert reason == ""

    def test_same_direction_high_correlation_rejected(
        self, correlation_filter, sample_price_data, btc_long_position
    ):
        """Test that same direction + high correlation is rejected"""
        # Update correlations
        correlation_filter.update_correlation_matrix(sample_price_data)

        # Try to add ETH LONG (highly correlated with BTC LONG)
        approved, reason = correlation_filter.check_correlation_filter(
            new_symbol="ETHUSDT",
            new_side=OrderSide.LONG,
            open_positions=[btc_long_position],
        )

        # Should be rejected due to high correlation + same direction
        assert approved is False
        assert "correlation" in reason.lower()

    def test_opposite_direction_allowed_as_hedge(
        self, correlation_filter, sample_price_data, btc_long_position
    ):
        """Test that opposite direction is allowed as hedge"""
        correlation_filter.update_correlation_matrix(sample_price_data)

        # Try to add SOL SHORT (opposite to BTC LONG)
        # SOL has moderate correlation with BTC (< hedge_corr_max)
        approved, reason = correlation_filter.check_correlation_filter(
            new_symbol="SOLUSDT",
            new_side=OrderSide.SHORT,
            open_positions=[btc_long_position],
        )

        # Should be allowed as hedge (moderate correlation, opposite direction)
        assert approved is True

    def test_low_correlation_allowed(
        self, correlation_filter, sample_price_data, btc_long_position
    ):
        """Test that low correlation symbols are allowed"""
        correlation_filter.update_correlation_matrix(sample_price_data)

        # Try to add MATIC LONG (low correlation with BTC)
        approved, reason = correlation_filter.check_correlation_filter(
            new_symbol="MATICUSDT",
            new_side=OrderSide.LONG,
            open_positions=[btc_long_position],
        )

        # Should be allowed
        assert approved is True

    def test_no_correlation_data_allowed(self, correlation_filter, btc_long_position):
        """Test that position is allowed if no correlation data exists"""
        # Don't update correlation matrix

        approved, reason = correlation_filter.check_correlation_filter(
            new_symbol="ETHUSDT",
            new_side=OrderSide.LONG,
            open_positions=[btc_long_position],
        )

        # Should be allowed (conservative approach)
        assert approved is True

    def test_hedge_correlation_too_high_rejected(self, correlation_filter):
        """Test that hedge is rejected if correlation is too high"""
        # Create artificial very high correlation (0.95)
        correlation_filter.correlation_cache[("BTCUSDT", "ETHUSDT")] = 0.95
        correlation_filter.correlation_cache[("ETHUSDT", "BTCUSDT")] = 0.95

        btc_position = Position(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            notional=5000.0,
            leverage=2.0,
            margin=2500.0,
            stop_price=49000.0,
            tp_price=None,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            entry_time=datetime.utcnow(),
            trade_id="trade_1",
        )

        # Try to add ETH SHORT as hedge (but correlation too high)
        approved, reason = correlation_filter.check_correlation_filter(
            new_symbol="ETHUSDT",
            new_side=OrderSide.SHORT,
            open_positions=[btc_position],
        )

        # Should be rejected (correlation 0.95 > hedge_corr_max 0.6)
        assert approved is False
        assert "hedge" in reason.lower()


class TestCorrelatedPositions:
    """Test getting correlated positions"""

    def test_get_correlated_positions_empty(self, correlation_filter):
        """Test with no open positions"""
        correlated = correlation_filter.get_correlated_positions(
            symbol="BTCUSDT", open_positions=[]
        )

        assert len(correlated) == 0

    def test_get_correlated_positions(
        self, correlation_filter, sample_price_data, btc_long_position, eth_short_position
    ):
        """Test getting correlated positions"""
        correlation_filter.update_correlation_matrix(sample_price_data)

        # Get positions correlated with SOL
        correlated = correlation_filter.get_correlated_positions(
            symbol="SOLUSDT",
            open_positions=[btc_long_position, eth_short_position],
        )

        # Should find correlations (sorted by strength)
        assert isinstance(correlated, list)

        # Each entry should be (Position, correlation)
        for pos, corr in correlated:
            assert isinstance(pos, Position)
            assert isinstance(corr, float)
            assert 0.0 <= corr <= 1.0

    def test_get_correlated_positions_custom_threshold(
        self, correlation_filter, sample_price_data, btc_long_position
    ):
        """Test with custom correlation threshold"""
        correlation_filter.update_correlation_matrix(sample_price_data)

        # Get with very high threshold (0.95) - should find fewer
        high_threshold = correlation_filter.get_correlated_positions(
            symbol="ETHUSDT",
            open_positions=[btc_long_position],
            threshold=0.95,
        )

        # Get with low threshold (0.5) - should find more
        low_threshold = correlation_filter.get_correlated_positions(
            symbol="ETHUSDT",
            open_positions=[btc_long_position],
            threshold=0.5,
        )

        assert len(low_threshold) >= len(high_threshold)


class TestCorrelationSummary:
    """Test correlation summary"""

    def test_summary_with_no_positions(self, correlation_filter):
        """Test summary with no open positions"""
        summary = correlation_filter.get_correlation_summary([])

        assert summary["avg_correlation"] == 0.0
        assert summary["max_correlation"] == 0.0
        assert summary["total_pairs"] == 0

    def test_summary_with_one_position(self, correlation_filter, btc_long_position):
        """Test summary with single position (no pairs)"""
        summary = correlation_filter.get_correlation_summary([btc_long_position])

        assert summary["avg_correlation"] == 0.0
        assert summary["total_pairs"] == 0

    def test_summary_with_multiple_positions(
        self, correlation_filter, sample_price_data, btc_long_position, eth_short_position
    ):
        """Test summary with multiple positions"""
        correlation_filter.update_correlation_matrix(sample_price_data)

        summary = correlation_filter.get_correlation_summary(
            [btc_long_position, eth_short_position]
        )

        # Should have 1 pair (BTC-ETH)
        assert summary["total_pairs"] == 1
        assert summary["avg_correlation"] > 0.0
        assert summary["max_correlation"] >= summary["avg_correlation"]
        assert summary["min_correlation"] <= summary["avg_correlation"]

    def test_summary_correlated_pairs_count(
        self, correlation_filter, sample_price_data
    ):
        """Test counting of highly correlated pairs"""
        correlation_filter.update_correlation_matrix(sample_price_data)

        # Create 3 positions
        positions = [
            Position(
                symbol="BTCUSDT",
                side=OrderSide.LONG,
                entry_price=50000.0,
                quantity=0.1,
                notional=5000.0,
                leverage=2.0,
                margin=2500.0,
                stop_price=49000.0,
                tp_price=None,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_time=datetime.utcnow(),
                trade_id="trade_1",
            ),
            Position(
                symbol="ETHUSDT",
                side=OrderSide.SHORT,
                entry_price=3000.0,
                quantity=1.0,
                notional=3000.0,
                leverage=1.5,
                margin=2000.0,
                stop_price=3150.0,
                tp_price=None,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_time=datetime.utcnow(),
                trade_id="trade_2",
            ),
            Position(
                symbol="MATICUSDT",
                side=OrderSide.LONG,
                entry_price=1.0,
                quantity=1000.0,
                notional=1000.0,
                leverage=1.0,
                margin=1000.0,
                stop_price=0.95,
                tp_price=None,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_time=datetime.utcnow(),
                trade_id="trade_3",
            ),
        ]

        summary = correlation_filter.get_correlation_summary(positions)

        # Should have 3 pairs: BTC-ETH, BTC-MATIC, ETH-MATIC
        assert summary["total_pairs"] == 3
        assert summary["correlated_pairs"] >= 0


class TestCacheManagement:
    """Test correlation cache management"""

    def test_clear_cache(self, correlation_filter, sample_price_data):
        """Test clearing correlation cache"""
        correlation_filter.update_correlation_matrix(sample_price_data)

        # Verify cache has data
        assert len(correlation_filter.correlation_cache) > 0

        # Clear cache
        correlation_filter.clear_cache()

        # Verify cache is empty
        assert len(correlation_filter.correlation_cache) == 0

    def test_cache_persistence(self, correlation_filter, sample_price_data):
        """Test that cache persists across multiple checks"""
        correlation_filter.update_correlation_matrix(sample_price_data)

        initial_size = len(correlation_filter.correlation_cache)

        # Multiple lookups should not change cache size
        correlation_filter.get_correlation("BTCUSDT", "ETHUSDT")
        correlation_filter.get_correlation("BTCUSDT", "SOLUSDT")

        assert len(correlation_filter.correlation_cache) == initial_size


class TestEdgeCases:
    """Test edge cases"""

    def test_same_symbol_position(self, correlation_filter):
        """Test adding position in same symbol (should have perfect correlation)"""
        # Set up perfect correlation for same symbol
        correlation_filter.correlation_cache[("BTCUSDT", "BTCUSDT")] = 1.0

        btc_position = Position(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            notional=5000.0,
            leverage=2.0,
            margin=2500.0,
            stop_price=49000.0,
            tp_price=None,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            entry_time=datetime.utcnow(),
            trade_id="trade_1",
        )

        # Try to add another LONG in same symbol
        approved, reason = correlation_filter.check_correlation_filter(
            new_symbol="BTCUSDT",
            new_side=OrderSide.LONG,
            open_positions=[btc_position],
        )

        # Should be rejected
        assert approved is False

    def test_negative_correlation_handling(self, correlation_filter):
        """Test that negative correlations are handled with abs()"""
        # Set negative correlation (stored as calculated, abs applied in check)
        correlation_filter.correlation_cache[("BTCUSDT", "INVERSE")] = -0.9
        correlation_filter.correlation_cache[("INVERSE", "BTCUSDT")] = -0.9

        btc_position = Position(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            notional=5000.0,
            leverage=2.0,
            margin=2500.0,
            stop_price=49000.0,
            tp_price=None,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            entry_time=datetime.utcnow(),
            trade_id="trade_1",
        )

        # Try to add LONG in negatively correlated symbol
        # abs(-0.9) = 0.9 > threshold (0.85), should be rejected
        approved, reason = correlation_filter.check_correlation_filter(
            new_symbol="INVERSE",
            new_side=OrderSide.LONG,
            open_positions=[btc_position],
        )

        # Should be rejected because abs(-0.9) = 0.9 > 0.85
        assert approved is False
