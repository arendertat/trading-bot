"""Tests for SL cooldown gating."""

from bot.config.loader import load_config
from bot.core.constants import OrderSide, RegimeType
from bot.risk.correlation_filter import CorrelationFilter
from bot.risk.kill_switch import KillSwitch
from bot.risk.position_sizing import PositionSizingCalculator
from bot.risk.risk_engine import RiskEngine
from bot.risk.risk_limits import RiskLimits


def _make_risk_engine(cooldown_bars: int) -> RiskEngine:
    config = load_config("config/config.example.json", load_env=False)
    config.risk.cooldown_after_sl_bars = cooldown_bars
    kill_switch = KillSwitch(config.risk)
    position_sizing = PositionSizingCalculator(config)
    risk_limits = RiskLimits(config.risk)
    corr_filter = CorrelationFilter(config.risk)
    return RiskEngine(
        config=config,
        kill_switch=kill_switch,
        position_sizing=position_sizing,
        risk_limits=risk_limits,
        correlation_filter=corr_filter,
    )


def test_cooldown_blocks_then_expires():
    engine = _make_risk_engine(2)
    engine.record_sl_exit("BTCUSDT")

    result_1 = engine.validate_entry(
        symbol="BTCUSDT",
        side=OrderSide.LONG,
        regime=RegimeType.TREND,
        stop_pct=0.01,
        current_price=50000.0,
        equity_usd=10000.0,
        free_margin_usd=10000.0,
        open_positions=[],
    )
    assert result_1.approved is False
    assert "Cooldown active" in result_1.rejection_reason

    engine.tick_cooldowns()
    result_2 = engine.validate_entry(
        symbol="BTCUSDT",
        side=OrderSide.LONG,
        regime=RegimeType.TREND,
        stop_pct=0.01,
        current_price=50000.0,
        equity_usd=10000.0,
        free_margin_usd=10000.0,
        open_positions=[],
    )
    assert result_2.approved is False

    engine.tick_cooldowns()
    result_3 = engine.validate_entry(
        symbol="BTCUSDT",
        side=OrderSide.LONG,
        regime=RegimeType.TREND,
        stop_pct=0.01,
        current_price=50000.0,
        equity_usd=10000.0,
        free_margin_usd=10000.0,
        open_positions=[],
    )
    assert result_3.approved is True
