"""Integration tests for risk engine"""

from datetime import datetime

import numpy as np
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
from bot.core.constants import BotMode, OrderSide, PositionStatus, RegimeType
from bot.core.types import Position
from bot.risk.correlation_filter import CorrelationFilter
from bot.risk.kill_switch import KillSwitch
from bot.risk.position_sizing import PositionSizingCalculator
from bot.risk.risk_engine import RiskEngine
from bot.risk.risk_limits import RiskLimits


@pytest.fixture
def default_config():
    """Create default bot configuration"""
    return BotConfig(
        mode=BotMode.PAPER_LIVE,
        timezone="UTC",
        exchange=ExchangeConfig(),
        universe=UniverseConfig(),
        timeframes=TimeframesConfig(),
        risk=RiskConfig(
            risk_per_trade_pct=0.01,
            max_total_open_risk_pct=0.025,
            max_open_positions=2,
            max_same_direction_positions=2,
            correlation_threshold=0.85,
            hedge_corr_max=0.6,
            daily_stop_pct=-0.04,
            weekly_stop_pct=-0.1,
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
def risk_engine(default_config):
    """Create fully configured risk engine"""
    kill_switch = KillSwitch(default_config.risk)
    position_sizing = PositionSizingCalculator(default_config)
    risk_limits = RiskLimits(default_config.risk)
    correlation_filter = CorrelationFilter(default_config.risk)

    return RiskEngine(
        config=default_config,
        kill_switch=kill_switch,
        position_sizing=position_sizing,
        risk_limits=risk_limits,
        correlation_filter=correlation_filter,
    )


class TestBasicValidation:
    """Test basic validation scenarios"""

    def test_validate_first_position_approved(self, risk_engine):
        """Test that first position is approved with clean state"""
        result = risk_engine.validate_entry(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            regime=RegimeType.TREND,
            stop_pct=0.01,
            current_price=50000.0,
            equity_usd=10000.0,
            free_margin_usd=10000.0,
            open_positions=[],
        )

        assert result.approved is True
        assert result.position_size is not None
        assert result.position_size.approved is True

    def test_validate_second_position_approved(self, risk_engine):
        """Test that second position is approved within limits"""
        # Create first position
        btc_position = Position(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            notional=5000.0,
            leverage=2.0,
            margin=2500.0,
            stop_price=49500.0,  # 50 USD risk (0.5% * 0.1)
            tp_price=51000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            entry_time=datetime.utcnow(),
            trade_id="trade_1",
            status=PositionStatus.OPEN,
        )

        # Try to add uncorrelated position
        result = risk_engine.validate_entry(
            symbol="SOLUSDT",
            side=OrderSide.SHORT,
            regime=RegimeType.RANGE,
            stop_pct=0.01,
            current_price=100.0,
            equity_usd=10000.0,
            free_margin_usd=7500.0,  # After first position margin
            open_positions=[btc_position],
        )

        assert result.approved is True


class TestKillSwitchIntegration:
    """Test kill switch integration"""

    def test_daily_stop_blocks_entry(self, risk_engine):
        """Test that daily stop blocks new entries"""
        # Trigger daily stop
        risk_engine.kill_switch.update_pnl(
            realized_pnl_today=-500.0,
            realized_pnl_week=-500.0,
            equity_usd=10000.0,
            now_utc=datetime.utcnow(),
        )

        assert risk_engine.kill_switch.is_active() is True

        # Try to enter position
        result = risk_engine.validate_entry(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            regime=RegimeType.TREND,
            stop_pct=0.01,
            current_price=50000.0,
            equity_usd=10000.0,
            free_margin_usd=10000.0,
            open_positions=[],
        )

        assert result.approved is False
        assert "kill switch" in result.rejection_reason.lower()

    def test_weekly_pause_blocks_entry(self, risk_engine):
        """Test that weekly pause blocks new entries"""
        # Trigger weekly stop
        risk_engine.kill_switch.update_pnl(
            realized_pnl_today=-1000.0,
            realized_pnl_week=-1000.0,
            equity_usd=10000.0,
            now_utc=datetime.utcnow(),
        )

        assert risk_engine.kill_switch.is_active() is True

        result = risk_engine.validate_entry(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            regime=RegimeType.TREND,
            stop_pct=0.01,
            current_price=50000.0,
            equity_usd=10000.0,
            free_margin_usd=10000.0,
            open_positions=[],
        )

        assert result.approved is False
        assert "kill switch" in result.rejection_reason.lower()


class TestPositionLimits:
    """Test position limit validation"""

    def test_max_positions_rejected(self, risk_engine):
        """Test rejection when max positions reached"""
        # Create 2 positions (at max)
        positions = [
            Position(
                symbol="BTCUSDT",
                side=OrderSide.LONG,
                entry_price=50000.0,
                quantity=0.1,
                notional=5000.0,
                leverage=2.0,
                margin=2500.0,
                stop_price=49500.0,
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
                stop_price=3030.0,
                tp_price=None,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_time=datetime.utcnow(),
                trade_id="trade_2",
            ),
        ]

        # Try to add third position
        result = risk_engine.validate_entry(
            symbol="SOLUSDT",
            side=OrderSide.LONG,
            regime=RegimeType.TREND,
            stop_pct=0.01,
            current_price=100.0,
            equity_usd=10000.0,
            free_margin_usd=5000.0,
            open_positions=positions,
        )

        assert result.approved is False
        assert "max positions" in result.rejection_reason.lower()

    def test_max_same_direction_rejected(self, default_config):
        """Test rejection when max same direction reached"""
        # Create custom config with higher max_positions but same max_same_direction
        custom_config = default_config.model_copy(deep=True)
        custom_config.risk.max_open_positions = 5
        custom_config.risk.max_same_direction_positions = 2

        # Create risk engine with custom config
        kill_switch = KillSwitch(custom_config.risk)
        position_sizing = PositionSizingCalculator(custom_config)
        risk_limits = RiskLimits(custom_config.risk)
        correlation_filter = CorrelationFilter(custom_config.risk)

        engine = RiskEngine(
            config=custom_config,
            kill_switch=kill_switch,
            position_sizing=position_sizing,
            risk_limits=risk_limits,
            correlation_filter=correlation_filter,
        )

        # Create 2 LONG positions (at max same direction)
        positions = [
            Position(
                symbol="BTCUSDT",
                side=OrderSide.LONG,
                entry_price=50000.0,
                quantity=0.1,
                notional=5000.0,
                leverage=2.0,
                margin=2500.0,
                stop_price=49500.0,
                tp_price=None,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_time=datetime.utcnow(),
                trade_id="trade_1",
            ),
            Position(
                symbol="ETHUSDT",
                side=OrderSide.LONG,
                entry_price=3000.0,
                quantity=1.0,
                notional=3000.0,
                leverage=1.5,
                margin=2000.0,
                stop_price=2970.0,
                tp_price=None,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_time=datetime.utcnow(),
                trade_id="trade_2",
            ),
        ]

        # Try to add third LONG (max_positions allows 5, but max_same_direction is 2)
        result = engine.validate_entry(
            symbol="SOLUSDT",
            side=OrderSide.LONG,
            regime=RegimeType.TREND,
            stop_pct=0.01,
            current_price=100.0,
            equity_usd=10000.0,
            free_margin_usd=5000.0,
            open_positions=positions,
        )

        assert result.approved is False
        assert "long" in result.rejection_reason.lower()


class TestOpenRiskLimit:
    """Test open risk limit validation"""

    def test_open_risk_limit_rejected(self, risk_engine):
        """Test rejection when open risk limit would be exceeded"""
        # Create position with 200 USD risk (close to 2% limit)
        position = Position(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            entry_price=50000.0,
            quantity=0.4,
            notional=20000.0,
            leverage=2.0,
            margin=10000.0,  # Within available margin
            stop_price=49500.0,  # 200 USD risk (500 * 0.4)
            tp_price=None,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            entry_time=datetime.utcnow(),
            trade_id="trade_1",
        )

        # Try to add another position with 100 USD risk
        # Total would be 300 USD = 3% > 2.5% limit
        result = risk_engine.validate_entry(
            symbol="ETHUSDT",
            side=OrderSide.SHORT,
            regime=RegimeType.HIGH_VOL,  # Use 1x leverage
            stop_pct=0.01,
            current_price=10000.0,  # Price = 10000, notional = 100/0.01 = 10000
            equity_usd=10000.0,
            free_margin_usd=10000.0,  # Plenty of margin available
            open_positions=[position],
        )

        # Should fail on open risk limit (not margin)
        assert result.approved is False
        assert "risk" in result.rejection_reason.lower()


class TestCorrelationFilter:
    """Test correlation filter integration"""

    def test_high_correlation_same_direction_rejected(self, risk_engine):
        """Test that high correlation + same direction is rejected"""
        # Set up high correlation between BTC and ETH
        risk_engine.correlation_filter.correlation_cache[("BTCUSDT", "ETHUSDT")] = 0.9
        risk_engine.correlation_filter.correlation_cache[("ETHUSDT", "BTCUSDT")] = 0.9

        btc_position = Position(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            notional=5000.0,
            leverage=2.0,
            margin=2500.0,
            stop_price=49500.0,
            tp_price=None,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            entry_time=datetime.utcnow(),
            trade_id="trade_1",
        )

        # Try to add ETH LONG (same direction, high correlation)
        result = risk_engine.validate_entry(
            symbol="ETHUSDT",
            side=OrderSide.LONG,
            regime=RegimeType.TREND,
            stop_pct=0.01,
            current_price=3000.0,
            equity_usd=10000.0,
            free_margin_usd=7500.0,
            open_positions=[btc_position],
        )

        assert result.approved is False
        assert "correlation" in result.rejection_reason.lower()

    def test_low_correlation_allowed(self, risk_engine):
        """Test that low correlation positions are allowed"""
        # Set up low correlation
        risk_engine.correlation_filter.correlation_cache[("BTCUSDT", "MATICUSDT")] = 0.3
        risk_engine.correlation_filter.correlation_cache[("MATICUSDT", "BTCUSDT")] = 0.3

        btc_position = Position(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            notional=5000.0,
            leverage=2.0,
            margin=2500.0,
            stop_price=49500.0,
            tp_price=None,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            entry_time=datetime.utcnow(),
            trade_id="trade_1",
        )

        # Try to add MATIC LONG (low correlation)
        result = risk_engine.validate_entry(
            symbol="MATICUSDT",
            side=OrderSide.LONG,
            regime=RegimeType.TREND,
            stop_pct=0.01,
            current_price=1.0,
            equity_usd=10000.0,
            free_margin_usd=7500.0,
            open_positions=[btc_position],
        )

        assert result.approved is True


class TestPortfolioStatus:
    """Test portfolio status reporting"""

    def test_portfolio_status_empty(self, risk_engine):
        """Test portfolio status with no positions"""
        status = risk_engine.get_portfolio_status([], 10000.0)

        assert status["total_positions"] == 0
        assert status["long_positions"] == 0
        assert status["short_positions"] == 0
        assert status["current_open_risk_usd"] == 0.0
        assert status["daily_stop_active"] is False
        assert status["weekly_pause_active"] is False

    def test_portfolio_status_with_positions(self, risk_engine):
        """Test portfolio status with open positions"""
        positions = [
            Position(
                symbol="BTCUSDT",
                side=OrderSide.LONG,
                entry_price=50000.0,
                quantity=0.1,
                notional=5000.0,
                leverage=2.0,
                margin=2500.0,
                stop_price=49500.0,
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
                stop_price=3030.0,
                tp_price=None,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_time=datetime.utcnow(),
                trade_id="trade_2",
            ),
        ]

        status = risk_engine.get_portfolio_status(positions, 10000.0)

        assert status["total_positions"] == 2
        assert status["long_positions"] == 1
        assert status["short_positions"] == 1
        assert status["current_open_risk_usd"] > 0
        assert "available_risk_usd" in status


class TestValidationOrder:
    """Test that validation checks happen in correct order"""

    def test_kill_switch_checked_first(self, risk_engine):
        """Test that kill switch is checked before other validations"""
        # Trigger kill switch
        risk_engine.kill_switch.state.daily_stop_active = True

        # Even with no positions (would normally pass)
        result = risk_engine.validate_entry(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            regime=RegimeType.TREND,
            stop_pct=0.01,
            current_price=50000.0,
            equity_usd=10000.0,
            free_margin_usd=10000.0,
            open_positions=[],
        )

        # Should fail on kill switch (first check)
        assert result.approved is False
        assert "kill switch" in result.rejection_reason.lower()

    def test_max_positions_checked_before_sizing(self, risk_engine):
        """Test that max positions is checked before position sizing"""
        # Create 2 positions (at max)
        positions = [
            Position(
                symbol=f"SYM{i}USDT",
                side=OrderSide.LONG,
                entry_price=100.0,
                quantity=1.0,
                notional=100.0,
                leverage=1.0,
                margin=100.0,
                stop_price=99.0,
                tp_price=None,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_time=datetime.utcnow(),
                trade_id=f"trade_{i}",
            )
            for i in range(2)
        ]

        result = risk_engine.validate_entry(
            symbol="NEWUSDT",
            side=OrderSide.LONG,
            regime=RegimeType.TREND,
            stop_pct=0.01,
            current_price=100.0,
            equity_usd=10000.0,
            free_margin_usd=9000.0,
            open_positions=positions,
        )

        # Should fail on max positions (before sizing)
        assert result.approved is False
        assert "max positions" in result.rejection_reason.lower()
        # Position size should not be calculated
        assert result.position_size is None


class TestRiskPerTradeOverride:
    """Test risk percentage override (for reduced risk periods)"""

    def test_custom_risk_percentage(self, risk_engine):
        """Test using custom risk percentage"""
        # Normal: 1% = 100 USD risk
        # Custom: 0.5% = 50 USD risk
        result = risk_engine.validate_entry(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            regime=RegimeType.TREND,
            stop_pct=0.01,
            current_price=50000.0,
            equity_usd=10000.0,
            free_margin_usd=10000.0,
            open_positions=[],
            risk_per_trade_pct=0.005,  # Override to 0.5%
        )

        assert result.approved is True
        assert result.position_size.risk_usd == pytest.approx(50.0)
