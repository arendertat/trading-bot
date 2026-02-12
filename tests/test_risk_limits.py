"""Tests for portfolio risk limits"""

from datetime import datetime

import pytest

from bot.config.models import RiskConfig
from bot.core.constants import OrderSide, PositionStatus
from bot.core.types import Position
from bot.risk.risk_limits import RiskLimits


@pytest.fixture
def default_config():
    """Create default risk configuration"""
    return RiskConfig(
        risk_per_trade_pct=0.01,
        max_total_open_risk_pct=0.025,  # 2.5%
        max_open_positions=2,
        max_same_direction_positions=2,
    )


@pytest.fixture
def risk_limits(default_config):
    """Create risk limits instance"""
    return RiskLimits(default_config)


@pytest.fixture
def sample_long_position():
    """Create sample long position"""
    return Position(
        symbol="BTCUSDT",
        side=OrderSide.LONG,
        entry_price=50000.0,
        quantity=0.1,
        notional=5000.0,
        leverage=2.0,
        margin=2500.0,
        stop_price=49000.0,  # 1000 USD stop distance
        tp_price=52000.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        entry_time=datetime.utcnow(),
        trade_id="trade_1",
        status=PositionStatus.OPEN,
    )


@pytest.fixture
def sample_short_position():
    """Create sample short position"""
    return Position(
        symbol="ETHUSDT",
        side=OrderSide.SHORT,
        entry_price=3000.0,
        quantity=1.0,
        notional=3000.0,
        leverage=1.5,
        margin=2000.0,
        stop_price=3150.0,  # 150 USD stop distance
        tp_price=2800.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        entry_time=datetime.utcnow(),
        trade_id="trade_2",
        status=PositionStatus.OPEN,
    )


class TestOpenRiskLimit:
    """Test total open risk limit validation"""

    def test_empty_portfolio(self, risk_limits):
        """Test with no open positions"""
        approved, reason = risk_limits.check_open_risk_limit(
            new_position_risk_usd=100.0,
            open_positions=[],
            equity_usd=10000.0,
        )

        assert approved is True
        assert reason == ""

    def test_within_limit_single_position(self, risk_limits):
        """Test adding position within limit"""
        equity = 10000.0
        # Max risk = 10000 * 0.025 = 250 USD
        new_risk = 200.0  # Within limit

        approved, reason = risk_limits.check_open_risk_limit(
            new_position_risk_usd=new_risk,
            open_positions=[],
            equity_usd=equity,
        )

        assert approved is True

    def test_at_limit(self, risk_limits):
        """Test at exact risk limit"""
        equity = 10000.0
        # Max risk = 10000 * 0.025 = 250 USD
        new_risk = 250.0  # Exactly at limit

        approved, reason = risk_limits.check_open_risk_limit(
            new_position_risk_usd=new_risk,
            open_positions=[],
            equity_usd=equity,
        )

        # Should be rejected (>= to reject at exact limit)
        assert approved is False
        assert "exceeds limit" in reason.lower() or "limit" in reason.lower()

    def test_exceeds_limit(self, risk_limits):
        """Test exceeding risk limit"""
        equity = 10000.0
        # Max risk = 250 USD
        new_risk = 300.0  # Exceeds limit

        approved, reason = risk_limits.check_open_risk_limit(
            new_position_risk_usd=new_risk,
            open_positions=[],
            equity_usd=equity,
        )

        assert approved is False
        assert "exceeds limit" in reason

    def test_with_existing_positions(self, risk_limits, sample_long_position):
        """Test with existing positions"""
        equity = 10000.0
        # Max risk = 250 USD

        # Existing position has 100 USD risk (50000 - 49000) * 0.1
        existing_risk = (50000.0 - 49000.0) * 0.1  # 100 USD
        assert existing_risk == 100.0

        # Try to add 100 USD more (total = 200 USD, within 250 limit)
        approved, reason = risk_limits.check_open_risk_limit(
            new_position_risk_usd=100.0,
            open_positions=[sample_long_position],
            equity_usd=equity,
        )

        assert approved is True

    def test_existing_positions_exceed_limit(self, risk_limits, sample_long_position):
        """Test when existing + new would exceed limit"""
        equity = 10000.0
        # Max risk = 250 USD
        # Existing = 100 USD
        # New = 200 USD
        # Total = 300 USD > 250 USD

        approved, reason = risk_limits.check_open_risk_limit(
            new_position_risk_usd=200.0,
            open_positions=[sample_long_position],
            equity_usd=equity,
        )

        assert approved is False

    def test_multiple_existing_positions(
        self, risk_limits, sample_long_position, sample_short_position
    ):
        """Test with multiple existing positions"""
        equity = 10000.0
        # Max risk = 250 USD

        # Long position: 100 USD risk
        # Short position: 150 USD risk (3150 - 3000) * 1.0
        # Total existing = 250 USD
        # Can't add any more

        approved, reason = risk_limits.check_open_risk_limit(
            new_position_risk_usd=50.0,
            open_positions=[sample_long_position, sample_short_position],
            equity_usd=equity,
        )

        assert approved is False


class TestMaxPositions:
    """Test maximum positions limit"""

    def test_no_positions(self, risk_limits):
        """Test with no open positions"""
        approved, reason = risk_limits.check_max_positions([])

        assert approved is True

    def test_below_max(self, risk_limits, sample_long_position):
        """Test with positions below max"""
        # Config max = 2, current = 1
        approved, reason = risk_limits.check_max_positions([sample_long_position])

        assert approved is True

    def test_at_max(self, risk_limits, sample_long_position, sample_short_position):
        """Test at maximum positions"""
        # Config max = 2, current = 2
        approved, reason = risk_limits.check_max_positions(
            [sample_long_position, sample_short_position]
        )

        # Should be rejected (>= not >)
        assert approved is False
        assert "Max positions" in reason

    def test_custom_max_positions(self, sample_long_position):
        """Test with custom max positions config"""
        config = RiskConfig(max_open_positions=5)
        limits = RiskLimits(config)

        # 3 positions < 5
        positions = [sample_long_position] * 3
        approved, reason = limits.check_max_positions(positions)

        assert approved is True


class TestSameDirectionLimit:
    """Test same-direction positions limit"""

    def test_no_positions(self, risk_limits):
        """Test with no open positions"""
        approved, reason = risk_limits.check_same_direction_limit("LONG", [])

        assert approved is True

    def test_below_limit_same_direction(self, risk_limits, sample_long_position):
        """Test below limit for same direction"""
        # Config max same direction = 2
        # Current LONG = 1, trying to add another LONG
        approved, reason = risk_limits.check_same_direction_limit(
            "LONG", [sample_long_position]
        )

        assert approved is True

    def test_at_limit_same_direction(self, risk_limits):
        """Test at limit for same direction"""
        # Create 2 LONG positions (at max)
        long1 = Position(
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
        long2 = Position(
            symbol="ETHUSDT",
            side=OrderSide.LONG,
            entry_price=3000.0,
            quantity=1.0,
            notional=3000.0,
            leverage=1.5,
            margin=2000.0,
            stop_price=2900.0,
            tp_price=None,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            entry_time=datetime.utcnow(),
            trade_id="trade_2",
        )

        # Try to add third LONG
        approved, reason = risk_limits.check_same_direction_limit("LONG", [long1, long2])

        assert approved is False
        assert "LONG" in reason

    def test_opposite_direction_allowed(self, risk_limits, sample_long_position):
        """Test that opposite direction is not affected"""
        # Have 1 LONG, try to add SHORT
        approved, reason = risk_limits.check_same_direction_limit(
            "SHORT", [sample_long_position]
        )

        assert approved is True

    def test_mixed_directions(self, risk_limits, sample_long_position, sample_short_position):
        """Test with mixed direction positions"""
        # Have 1 LONG and 1 SHORT
        # Try to add another LONG (below limit of 2)
        approved, reason = risk_limits.check_same_direction_limit(
            "LONG", [sample_long_position, sample_short_position]
        )

        assert approved is True


class TestRiskCalculations:
    """Test risk calculation functions"""

    def test_calculate_total_open_risk_empty(self, risk_limits):
        """Test with no positions"""
        total = risk_limits._calculate_total_open_risk([])
        assert total == 0.0

    def test_calculate_total_open_risk_single(self, risk_limits, sample_long_position):
        """Test with single position"""
        # BTC: (50000 - 49000) * 0.1 = 100 USD
        total = risk_limits._calculate_total_open_risk([sample_long_position])
        assert total == pytest.approx(100.0)

    def test_calculate_total_open_risk_multiple(
        self, risk_limits, sample_long_position, sample_short_position
    ):
        """Test with multiple positions"""
        # BTC: 100 USD
        # ETH: (3150 - 3000) * 1.0 = 150 USD
        # Total: 250 USD
        total = risk_limits._calculate_total_open_risk(
            [sample_long_position, sample_short_position]
        )
        assert total == pytest.approx(250.0)

    def test_get_available_risk_budget_empty(self, risk_limits):
        """Test available risk budget with no positions"""
        equity = 10000.0
        # Max = 10000 * 0.025 = 250
        available = risk_limits.get_available_risk_budget([], equity)
        assert available == pytest.approx(250.0)

    def test_get_available_risk_budget_with_positions(
        self, risk_limits, sample_long_position
    ):
        """Test available risk budget with positions"""
        equity = 10000.0
        # Max = 250, Used = 100, Available = 150
        available = risk_limits.get_available_risk_budget([sample_long_position], equity)
        assert available == pytest.approx(150.0)

    def test_get_available_risk_budget_at_limit(
        self, risk_limits, sample_long_position, sample_short_position
    ):
        """Test when at risk limit"""
        equity = 10000.0
        # Max = 250, Used = 250, Available = 0
        available = risk_limits.get_available_risk_budget(
            [sample_long_position, sample_short_position], equity
        )
        assert available == pytest.approx(0.0)


class TestPortfolioRiskSummary:
    """Test portfolio risk summary"""

    def test_empty_portfolio(self, risk_limits):
        """Test summary with empty portfolio"""
        equity = 10000.0
        summary = risk_limits.get_portfolio_risk_summary([], equity)

        assert summary["total_positions"] == 0
        assert summary["long_positions"] == 0
        assert summary["short_positions"] == 0
        assert summary["current_open_risk_usd"] == 0.0
        assert summary["current_open_risk_pct"] == 0.0
        assert summary["max_open_risk_pct"] == 0.025
        assert summary["available_risk_usd"] == pytest.approx(250.0)
        assert summary["max_positions"] == 2
        assert summary["available_position_slots"] == 2

    def test_single_position(self, risk_limits, sample_long_position):
        """Test summary with single position"""
        equity = 10000.0
        summary = risk_limits.get_portfolio_risk_summary([sample_long_position], equity)

        assert summary["total_positions"] == 1
        assert summary["long_positions"] == 1
        assert summary["short_positions"] == 0
        assert summary["current_open_risk_usd"] == pytest.approx(100.0)
        assert summary["current_open_risk_pct"] == pytest.approx(0.01)
        assert summary["available_risk_usd"] == pytest.approx(150.0)
        assert summary["available_position_slots"] == 1
        assert summary["available_long_slots"] == 1
        assert summary["available_short_slots"] == 2

    def test_multiple_positions(
        self, risk_limits, sample_long_position, sample_short_position
    ):
        """Test summary with multiple positions"""
        equity = 10000.0
        summary = risk_limits.get_portfolio_risk_summary(
            [sample_long_position, sample_short_position], equity
        )

        assert summary["total_positions"] == 2
        assert summary["long_positions"] == 1
        assert summary["short_positions"] == 1
        assert summary["current_open_risk_usd"] == pytest.approx(250.0)
        assert summary["current_open_risk_pct"] == pytest.approx(0.025)
        assert summary["available_risk_usd"] == pytest.approx(0.0)
        assert summary["available_position_slots"] == 0
        assert summary["available_long_slots"] == 1
        assert summary["available_short_slots"] == 1


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_zero_equity(self, risk_limits):
        """Test with zero equity"""
        # Should not crash
        approved, reason = risk_limits.check_open_risk_limit(
            new_position_risk_usd=100.0,
            open_positions=[],
            equity_usd=0.0,
        )

        # Should reject (can't calculate percentage with zero equity)
        assert approved is False
        assert "equity" in reason.lower()

    def test_very_small_risk(self, risk_limits):
        """Test with very small risk amount"""
        equity = 10000.0
        approved, reason = risk_limits.check_open_risk_limit(
            new_position_risk_usd=0.01,
            open_positions=[],
            equity_usd=equity,
        )

        assert approved is True

    def test_negative_available_risk(self, risk_limits):
        """Test when available risk would be negative"""
        equity = 10000.0

        # Create position with very high risk
        high_risk_position = Position(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            entry_price=50000.0,
            quantity=1.0,
            notional=50000.0,
            leverage=1.0,
            margin=50000.0,
            stop_price=45000.0,  # 5000 USD risk
            tp_price=None,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            entry_time=datetime.utcnow(),
            trade_id="trade_1",
        )

        available = risk_limits.get_available_risk_budget([high_risk_position], equity)

        # Should be 0, not negative
        assert available == 0.0


class TestCustomConfiguration:
    """Test with custom configurations"""

    def test_higher_risk_limit(self):
        """Test with higher risk limit"""
        config = RiskConfig(max_total_open_risk_pct=0.05)  # 5%
        limits = RiskLimits(config)

        equity = 10000.0
        # Max = 500 USD

        approved, reason = limits.check_open_risk_limit(
            new_position_risk_usd=400.0,
            open_positions=[],
            equity_usd=equity,
        )

        assert approved is True

    def test_more_positions_allowed(self):
        """Test with higher position limits"""
        config = RiskConfig(
            max_open_positions=5,
            max_same_direction_positions=3,
        )
        limits = RiskLimits(config)

        # Create 3 positions (should be allowed)
        positions = [
            Position(
                symbol=f"SYM{i}USDT",
                side=OrderSide.LONG,
                entry_price=1000.0,
                quantity=1.0,
                notional=1000.0,
                leverage=1.0,
                margin=1000.0,
                stop_price=990.0,
                tp_price=None,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_time=datetime.utcnow(),
                trade_id=f"trade_{i}",
            )
            for i in range(3)
        ]

        approved, _ = limits.check_max_positions(positions)
        assert approved is True

        approved, _ = limits.check_same_direction_limit("LONG", positions)
        assert approved is False  # At max same direction
