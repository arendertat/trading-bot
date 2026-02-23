"""Comprehensive tests for position sizing calculator"""

import pytest

from bot.config.models import (
    BotConfig,
    ExchangeConfig,
    ExecutionConfig,
    LeverageConfig,
    LoggingConfig,
    NotificationConfig,
    PerformanceConfig,
    RegimeConfig,
    RiskConfig,
    StrategiesConfig,
    StrategyRangeMeanReversionConfig,
    StrategyTrendBreakoutConfig,
    StrategyTrendPullbackConfig,
    TimeframesConfig,
    UniverseConfig,
)
from bot.core.constants import BotMode, RegimeType
from bot.risk.position_sizing import PositionSizingCalculator


@pytest.fixture
def default_config():
    """Create default bot configuration for testing"""
    return BotConfig(
        mode=BotMode.PAPER_LIVE,
        timezone="UTC",
        exchange=ExchangeConfig(),
        universe=UniverseConfig(),
        timeframes=TimeframesConfig(),
        risk=RiskConfig(
            risk_per_trade_pct=0.01,  # 1%
            max_total_open_risk_pct=0.025,  # 2.5%
            max_open_positions=2,
            max_same_direction_positions=2,
            available_margin_ratio=1.0,
            max_margin_utilization=1.0,
            min_stop_pct=0.0,
            min_stop_usd=0.0,
            insufficient_margin_log_every_n=1000,
        ),
        regime=RegimeConfig(),
        strategies=StrategiesConfig(
            trend_pullback=StrategyTrendPullbackConfig(),
            trend_breakout=StrategyTrendBreakoutConfig(),
            range_mean_reversion=StrategyRangeMeanReversionConfig(),
        ),
        leverage=LeverageConfig(trend=2.0, range=1.5, high_vol=1.0),
        execution=ExecutionConfig(),
        performance=PerformanceConfig(),
        notifications=NotificationConfig(),
        logging=LoggingConfig(),
    )


@pytest.fixture
def calculator(default_config):
    """Create position sizing calculator instance"""
    return PositionSizingCalculator(default_config)


class TestPositionSizingBasics:
    """Test basic position sizing calculations"""

    def test_trend_regime_calculation(self, calculator):
        """Test position sizing in TREND regime"""
        result = calculator.calculate(
            equity_usd=10000.0,
            stop_pct=0.01,  # 1% stop
            regime=RegimeType.TREND,
            current_price=50000.0,
            free_margin_usd=10000.0,
            min_notional_usd=5.0,
        )

        # Expected: risk = 10000 * 0.01 = 100
        # notional = 100 / 0.01 = 10000
        # leverage = 2.0 (TREND)
        # margin = 10000 / 2.0 = 5000
        # quantity = 10000 / 50000 = 0.2

        assert result.approved is True
        assert result.risk_usd == pytest.approx(100.0)
        assert result.notional_usd == pytest.approx(10000.0)
        assert result.leverage == pytest.approx(2.0)
        assert result.margin_required_usd == pytest.approx(5000.0)
        assert result.quantity == pytest.approx(0.2)

    def test_range_regime_calculation(self, calculator):
        """Test position sizing in RANGE regime"""
        result = calculator.calculate(
            equity_usd=10000.0,
            stop_pct=0.008,  # 0.8% stop
            regime=RegimeType.RANGE,
            current_price=30000.0,
            free_margin_usd=10000.0,
        )

        # Expected: risk = 100
        # notional = 100 / 0.008 = 12500
        # leverage = 1.5 (RANGE)
        # margin = 12500 / 1.5 = 8333.33

        assert result.approved is True
        assert result.risk_usd == pytest.approx(100.0)
        assert result.notional_usd == pytest.approx(12500.0)
        assert result.leverage == pytest.approx(1.5)
        assert result.margin_required_usd == pytest.approx(8333.33, rel=1e-2)

    def test_high_vol_regime_calculation(self, calculator):
        """Test position sizing in HIGH_VOL regime"""
        result = calculator.calculate(
            equity_usd=10000.0,
            stop_pct=0.02,  # 2% stop
            regime=RegimeType.HIGH_VOL,
            current_price=25000.0,
            free_margin_usd=10000.0,
        )

        # Expected: leverage = 1.0 (HIGH_VOL)
        # notional = 100 / 0.02 = 5000
        # margin = 5000 / 1.0 = 5000

        assert result.approved is True
        assert result.leverage == pytest.approx(1.0)
        assert result.notional_usd == pytest.approx(5000.0)
        assert result.margin_required_usd == pytest.approx(5000.0)


class TestValidations:
    """Test position sizing validations"""

    def test_insufficient_margin_rejection(self, calculator):
        """Test rejection when insufficient margin"""
        result = calculator.calculate(
            equity_usd=10000.0,
            stop_pct=0.005,  # 0.5% stop -> large notional
            regime=RegimeType.TREND,
            current_price=50000.0,
            free_margin_usd=5000.0,  # Limited margin
        )

        # Expected: notional = 100 / 0.005 = 20000
        # margin = 20000 / 2.0 = 10000
        # But free_margin = 5000 < 10000

        assert result.approved is False
        assert "Insufficient margin" in result.rejection_reason
        assert result.margin_required_usd == pytest.approx(10000.0)

    def test_below_minimum_notional_rejection(self, calculator):
        """Test rejection when below minimum notional"""
        result = calculator.calculate(
            equity_usd=100.0,  # Small equity
            stop_pct=0.02,  # 2% stop
            regime=RegimeType.TREND,
            current_price=50000.0,
            free_margin_usd=100.0,
            min_notional_usd=10.0,
        )

        # Expected: risk = 100 * 0.01 = 1
        # notional = 1 / 0.02 = 50
        # But if we use smaller equity:
        # Let's recalculate with equity=50
        result = calculator.calculate(
            equity_usd=50.0,
            stop_pct=0.1,  # 10% stop
            regime=RegimeType.TREND,
            current_price=50000.0,
            free_margin_usd=100.0,
            min_notional_usd=10.0,
        )

        # risk = 50 * 0.01 = 0.5
        # notional = 0.5 / 0.1 = 5.0 < 10.0

        assert result.approved is False
        assert "below minimum" in result.rejection_reason
        assert result.notional_usd < 10.0

    def test_invalid_stop_pct(self, calculator):
        """Test rejection with invalid stop percentage"""
        result = calculator.calculate(
            equity_usd=10000.0,
            stop_pct=0.0,  # Invalid
            regime=RegimeType.TREND,
            current_price=50000.0,
            free_margin_usd=10000.0,
        )

        assert result.approved is False
        assert "Invalid stop_pct" in result.rejection_reason

        result = calculator.calculate(
            equity_usd=10000.0,
            stop_pct=-0.01,  # Negative
            regime=RegimeType.TREND,
            current_price=50000.0,
            free_margin_usd=10000.0,
        )
        assert result.approved is False
        assert "Invalid stop_pct" in result.rejection_reason


class TestMarginClamp:
    """Tests for margin-aware clamp behavior."""

    def test_margin_clamp_reduces_notional(self, default_config):
        default_config.risk.available_margin_ratio = 1.0
        default_config.risk.max_margin_utilization = 0.5
        calculator = PositionSizingCalculator(default_config)

        result = calculator.calculate(
            equity_usd=10000.0,
            stop_pct=0.005,  # 0.5% stop -> big notional
            regime=RegimeType.TREND,
            current_price=50000.0,
            free_margin_usd=10000.0,
        )

        # Clamp notional to available_margin * max_util * leverage
        assert result.approved is True
        assert result.notional_usd == pytest.approx(10000.0)
        assert result.margin_required_usd == pytest.approx(5000.0)

    def test_margin_clamp_below_min_notional_rejects(self, default_config):
        default_config.risk.available_margin_ratio = 0.1
        default_config.risk.max_margin_utilization = 0.5
        calculator = PositionSizingCalculator(default_config)

        result = calculator.calculate(
            equity_usd=10000.0,
            stop_pct=0.01,
            regime=RegimeType.RANGE,
            current_price=20000.0,
            free_margin_usd=10000.0,
            min_notional_usd=1000.0,
        )

        assert result.approved is False
        assert "INSUFFICIENT_MARGIN" in result.rejection_reason

        assert result.approved is False
        assert "Invalid stop_pct" in result.rejection_reason

    def test_invalid_price(self, calculator):
        """Test rejection with invalid price"""
        result = calculator.calculate(
            equity_usd=10000.0,
            stop_pct=0.01,
            regime=RegimeType.TREND,
            current_price=0.0,  # Invalid
            free_margin_usd=10000.0,
        )

        assert result.approved is False
        assert "Invalid current_price" in result.rejection_reason


class TestParametrizedScenarios:
    """Parametrized tests for various scenarios"""

    @pytest.mark.parametrize(
        "equity,stop_pct,regime,expected_notional,expected_leverage",
        [
            (10000, 0.01, RegimeType.TREND, 10000, 2.0),
            (10000, 0.01, RegimeType.RANGE, 10000, 1.5),
            (10000, 0.01, RegimeType.HIGH_VOL, 10000, 1.0),
            (5000, 0.005, RegimeType.TREND, 10000, 2.0),
            (20000, 0.02, RegimeType.RANGE, 10000, 1.5),
        ],
    )
    def test_various_regimes_and_stops(
        self, calculator, equity, stop_pct, regime, expected_notional, expected_leverage
    ):
        """Test various combinations of equity, stop, and regime"""
        result = calculator.calculate(
            equity_usd=equity,
            stop_pct=stop_pct,
            regime=regime,
            current_price=50000.0,
            free_margin_usd=equity,
        )

        assert result.approved is True
        assert result.notional_usd == pytest.approx(expected_notional, rel=1e-2)
        assert result.leverage == pytest.approx(expected_leverage)

    @pytest.mark.parametrize(
        "equity,stop_pct,price,expected_quantity",
        [
            (10000, 0.01, 50000, 0.2),  # notional=10000, qty=10000/50000
            (10000, 0.01, 25000, 0.4),  # notional=10000, qty=10000/25000
            (10000, 0.02, 50000, 0.1),  # notional=5000, qty=5000/50000
            (5000, 0.01, 10000, 0.5),  # notional=5000, qty=5000/10000
        ],
    )
    def test_quantity_calculations(self, calculator, equity, stop_pct, price, expected_quantity):
        """Test quantity calculations for various scenarios"""
        result = calculator.calculate(
            equity_usd=equity,
            stop_pct=stop_pct,
            regime=RegimeType.TREND,
            current_price=price,
            free_margin_usd=equity,
        )

        assert result.approved is True
        assert result.quantity == pytest.approx(expected_quantity, rel=1e-2)


class TestRiskOverrides:
    """Test risk percentage overrides"""

    def test_custom_risk_percentage(self, calculator):
        """Test using custom risk percentage"""
        # Default is 1%, override to 2%
        result = calculator.calculate(
            equity_usd=10000.0,
            stop_pct=0.01,
            regime=RegimeType.TREND,
            current_price=50000.0,
            free_margin_usd=10000.0,
            risk_per_trade_pct=0.02,  # 2% override
        )

        # Expected: risk = 10000 * 0.02 = 200
        # notional = 200 / 0.01 = 20000

        assert result.approved is True
        assert result.risk_usd == pytest.approx(200.0)
        assert result.notional_usd == pytest.approx(20000.0)

    def test_reduced_risk_after_pause(self, calculator):
        """Test reduced risk scenario (after weekly pause)"""
        result = calculator.calculate(
            equity_usd=10000.0,
            stop_pct=0.01,
            regime=RegimeType.TREND,
            current_price=50000.0,
            free_margin_usd=10000.0,
            risk_per_trade_pct=0.005,  # 0.5% reduced risk
        )

        # Expected: risk = 10000 * 0.005 = 50
        # notional = 50 / 0.01 = 5000

        assert result.approved is True
        assert result.risk_usd == pytest.approx(50.0)
        assert result.notional_usd == pytest.approx(5000.0)


class TestLeverageMapping:
    """Test leverage mapping functionality"""

    def test_get_leverage_for_regime(self, calculator):
        """Test getting leverage for each regime"""
        assert calculator.get_leverage_for_regime(RegimeType.TREND) == 2.0
        assert calculator.get_leverage_for_regime(RegimeType.RANGE) == 1.5
        assert calculator.get_leverage_for_regime(RegimeType.HIGH_VOL) == 1.0
        assert calculator.get_leverage_for_regime(RegimeType.CHOP_NO_TRADE) == 1.0

    def test_custom_leverage_config(self, default_config):
        """Test custom leverage configuration"""
        custom_config = default_config.model_copy(deep=True)
        custom_config.leverage.trend = 1.8
        custom_config.leverage.range = 1.3
        custom_config.leverage.high_vol = 1.0

        calc = PositionSizingCalculator(custom_config)

        assert calc.get_leverage_for_regime(RegimeType.TREND) == 1.8
        assert calc.get_leverage_for_regime(RegimeType.RANGE) == 1.3


class TestMaxPositionEstimation:
    """Test maximum position count estimation"""

    def test_max_position_count_default(self, calculator):
        """Test max position estimation with default parameters"""
        max_positions = calculator.calculate_max_position_count(
            equity_usd=10000.0,
            avg_stop_pct=0.01,
            avg_regime_leverage=1.5,
        )

        # risk_per_trade = 10000 * 0.01 = 100
        # notional = 100 / 0.01 = 10000
        # margin = 10000 / 1.5 = 6666.67
        # max_by_margin = 10000 / 6666.67 = 1.5 -> 1
        #
        # max_total_risk = 10000 * 0.025 = 250
        # max_by_risk = 250 / 100 = 2.5 -> 2
        #
        # config max = 2
        # min(1, 2, 2) = 1

        assert max_positions == 1

    def test_max_position_count_smaller_stops(self, calculator):
        """Test max position with smaller stops (larger positions)"""
        max_positions = calculator.calculate_max_position_count(
            equity_usd=20000.0,
            avg_stop_pct=0.02,  # Larger stops = smaller positions
            avg_regime_leverage=2.0,
        )

        # risk_per_trade = 20000 * 0.01 = 200
        # notional = 200 / 0.02 = 10000
        # margin = 10000 / 2.0 = 5000
        # max_by_margin = 20000 / 5000 = 4
        #
        # max_total_risk = 20000 * 0.025 = 500
        # max_by_risk = 500 / 200 = 2.5 -> 2
        #
        # config max = 2
        # min(4, 2, 2) = 2

        assert max_positions == 2

    def test_max_position_respects_config_limit(self, default_config):
        """Test that max positions respects config limit"""
        config = default_config.model_copy(deep=True)
        config.risk.max_open_positions = 1  # Strict limit

        calc = PositionSizingCalculator(config)

        max_positions = calc.calculate_max_position_count(
            equity_usd=100000.0,  # Large equity
            avg_stop_pct=0.05,  # Large stops
            avg_regime_leverage=2.0,
        )

        # Even with large equity, respect config limit
        assert max_positions == 1


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_small_equity(self, calculator):
        """Test with very small equity"""
        result = calculator.calculate(
            equity_usd=10.0,
            stop_pct=0.01,
            regime=RegimeType.TREND,
            current_price=50000.0,
            free_margin_usd=10.0,
            min_notional_usd=5.0,
        )

        # risk = 10 * 0.01 = 0.1
        # notional = 0.1 / 0.01 = 10
        # Should pass minimum notional of 5

        assert result.approved is True
        assert result.notional_usd == pytest.approx(10.0)

    def test_very_large_equity(self, calculator):
        """Test with very large equity"""
        result = calculator.calculate(
            equity_usd=1000000.0,
            stop_pct=0.01,
            regime=RegimeType.TREND,
            current_price=50000.0,
            free_margin_usd=1000000.0,
        )

        # risk = 1000000 * 0.01 = 10000
        # notional = 10000 / 0.01 = 1000000

        assert result.approved is True
        assert result.notional_usd == pytest.approx(1000000.0)

    def test_very_tight_stop(self, calculator):
        """Test with very tight stop (large position)"""
        result = calculator.calculate(
            equity_usd=10000.0,
            stop_pct=0.001,  # 0.1% stop
            regime=RegimeType.HIGH_VOL,
            current_price=50000.0,
            free_margin_usd=10000.0,
        )

        # risk = 100
        # notional = 100 / 0.001 = 100000
        # margin = 100000 / 1.0 = 100000
        # Should fail on insufficient margin

        assert result.approved is False
        assert "Insufficient margin" in result.rejection_reason

    def test_very_wide_stop(self, calculator):
        """Test with very wide stop (small position)"""
        result = calculator.calculate(
            equity_usd=10000.0,
            stop_pct=0.5,  # 50% stop (very wide)
            regime=RegimeType.TREND,
            current_price=50000.0,
            free_margin_usd=10000.0,
            min_notional_usd=5.0,
        )

        # risk = 100
        # notional = 100 / 0.5 = 200

        assert result.approved is True
        assert result.notional_usd == pytest.approx(200.0)


class TestRealWorldScenarios:
    """Test realistic trading scenarios"""

    def test_btc_long_trend_regime(self, calculator):
        """Test BTC long in trend regime"""
        result = calculator.calculate(
            equity_usd=10000.0,
            stop_pct=0.015,  # 1.5% stop
            regime=RegimeType.TREND,
            current_price=65000.0,  # BTC price
            free_margin_usd=10000.0,
        )

        assert result.approved is True
        assert result.leverage == 2.0
        # notional = 100 / 0.015 = 6666.67
        # quantity = 6666.67 / 65000 ≈ 0.1026 BTC

    def test_eth_short_range_regime(self, calculator):
        """Test ETH short in range regime"""
        result = calculator.calculate(
            equity_usd=10000.0,
            stop_pct=0.008,  # 0.8% stop
            regime=RegimeType.RANGE,
            current_price=3500.0,  # ETH price
            free_margin_usd=10000.0,
        )

        assert result.approved is True
        assert result.leverage == 1.5
        # notional = 100 / 0.008 = 12500
        # quantity = 12500 / 3500 ≈ 3.57 ETH

    def test_multiple_positions_margin_check(self, calculator):
        """Test margin availability for multiple positions"""
        equity = 10000.0
        free_margin = 10000.0

        # First position
        result1 = calculator.calculate(
            equity_usd=equity,
            stop_pct=0.01,
            regime=RegimeType.TREND,
            current_price=50000.0,
            free_margin_usd=free_margin,
        )

        assert result1.approved is True
        margin_used_1 = result1.margin_required_usd

        # Second position with remaining margin
        remaining_margin = free_margin - margin_used_1
        result2 = calculator.calculate(
            equity_usd=equity,
            stop_pct=0.01,
            regime=RegimeType.RANGE,
            current_price=30000.0,
            free_margin_usd=remaining_margin,
        )

        # Second position requires more margin than first (lower leverage)
        # margin1 = 10000 / 2.0 = 5000
        # margin2 = 10000 / 1.5 = 6666.67
        # remaining = 5000, need 6666.67 -> should fail

        assert result2.approved is False
        assert "Insufficient margin" in result2.rejection_reason
