"""Configuration loading and validation tests"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from bot.config.loader import load_config, substitute_env_vars
from bot.config.models import BotConfig
from bot.config.validator import validate_config_constraints


def create_temp_config(config_data: dict) -> str:
    """Create a temporary config file"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    json.dump(config_data, temp_file)
    temp_file.close()
    return temp_file.name


@pytest.fixture
def valid_config_data():
    """Valid configuration data"""
    return {
        "mode": "PAPER_LIVE",
        "timezone": "UTC",
        "exchange": {
            "name": "binance",
            "api_key_env": "BINANCE_API_KEY",
            "api_secret_env": "BINANCE_API_SECRET",
            "usdtm_perp": True,
            "margin_mode": "ISOLATED",
            "recv_window_ms": 5000
        },
        "universe": {
            "min_24h_volume_usdt": 100000000,
            "max_spread_pct": 0.0005,
            "max_abs_funding_rate": 0.0015,
            "min_atr_ratio": 0.005,
            "max_monitored_symbols": 6,
            "whitelist": [],
            "blacklist": [],
            "hedge_max_combined_funding": 0.0015
        },
        "timeframes": {
            "signal_tf": "5m",
            "trend_tf": "1h",
            "zscore_lookback": 100,
            "corr_tf": "1h",
            "corr_lookback_hours": 72
        },
        "risk": {
            "risk_per_trade_pct": 0.01,
            "max_total_open_risk_pct": 0.025,
            "max_open_positions": 2,
            "max_same_direction_positions": 2,
            "correlation_threshold": 0.85,
            "hedge_corr_max": 0.6,
            "daily_stop_pct": -0.04,
            "weekly_stop_pct": -0.1,
            "pause_days_after_weekly_stop": 7,
            "reduced_risk_after_pause_pct": 0.005,
            "reduced_risk_days": 3
        },
        "regime": {
            "trend_adx_min": 25,
            "range_adx_max": 20,
            "high_vol_atr_z": 1.5,
            "confidence_threshold": 0.55,
            "bb_width_range_min": 0.01,
            "bb_width_range_max": 0.05
        },
        "strategies": {
            "trend_pullback": {
                "enabled": True,
                "stop_pct": 0.01,
                "target_r_multiple": 1.5,
                "pullback_rsi_long_min": 40,
                "pullback_rsi_long_max": 50,
                "pullback_rsi_short_min": 50,
                "pullback_rsi_short_max": 60,
                "ema20_band_pct": 0.002,
                "trail_after_r": 1.0,
                "atr_trail_mult": 2.0
            },
            "trend_breakout": {
                "enabled": True,
                "stop_pct": 0.01,
                "breakout_lookback_bars": 20,
                "breakout_volume_z_min": 1.0,
                "atr_trail_mult": 2.5
            },
            "range_mean_reversion": {
                "enabled": True,
                "stop_pct": 0.008,
                "target_r_multiple": 1.2,
                "rsi_long_extreme": 25,
                "rsi_short_extreme": 75
            }
        },
        "leverage": {
            "trend": 2.0,
            "range": 1.5,
            "high_vol": 1.0
        },
        "execution": {
            "entry_order_type": "LIMIT",
            "limit_ttl_seconds": 30,
            "limit_retry_count": 1,
            "stop_order_type": "STOP_MARKET",
            "kill_switch_order_type": "MARKET",
            "paper_slippage_limit_pct": 0.0002,
            "paper_slippage_market_pct": 0.0008,
            "paper_slippage_stop_pct": 0.001,
            "maker_fee_pct": 0.0002,
            "taker_fee_pct": 0.0004,
            "enable_funding_in_paper": False
        },
        "performance": {
            "window_trades": 50,
            "min_trades_before_confidence": 20,
            "dd_penalty_weight": 0.5,
            "max_strategy_switches_per_day": 1
        },
        "notifications": {
            "telegram_enabled": False,
            "telegram_token_env": "TELEGRAM_BOT_TOKEN",
            "telegram_chat_id_env": "TELEGRAM_CHAT_ID",
            "daily_report_time_utc": "00:05"
        },
        "logging": {
            "log_dir": "./logs",
            "trade_log_file": "trades.jsonl",
            "event_log_file": "events.jsonl",
            "log_level": "INFO"
        }
    }


class TestConfigLoading:
    """Test configuration loading"""

    def test_load_valid_config(self, valid_config_data):
        """Test loading a valid configuration"""
        config_path = create_temp_config(valid_config_data)
        try:
            config = load_config(config_path, load_env=False)
            assert config.mode == "PAPER_LIVE"
            assert config.risk.risk_per_trade_pct == 0.01
            assert config.risk.max_open_positions == 2
        finally:
            os.unlink(config_path)

    def test_missing_config_file(self):
        """Test error when config file doesn't exist"""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.json", load_env=False)

    def test_invalid_json(self):
        """Test error with invalid JSON"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_file.write("{ invalid json }")
        temp_file.close()
        try:
            with pytest.raises(json.JSONDecodeError):
                load_config(temp_file.name, load_env=False)
        finally:
            os.unlink(temp_file.name)


class TestConfigValidation:
    """Test configuration validation"""

    def test_invalid_risk_pct_too_high(self, valid_config_data):
        """Test validation fails when risk_per_trade_pct > 0.1 (10%)"""
        valid_config_data["risk"]["risk_per_trade_pct"] = 0.15
        config_path = create_temp_config(valid_config_data)
        try:
            with pytest.raises(ValueError, match="validation failed"):
                load_config(config_path, load_env=False)
        finally:
            os.unlink(config_path)

    def test_invalid_risk_pct_negative(self, valid_config_data):
        """Test validation fails when risk_per_trade_pct is negative"""
        valid_config_data["risk"]["risk_per_trade_pct"] = -0.01
        config_path = create_temp_config(valid_config_data)
        try:
            with pytest.raises(ValueError, match="validation failed"):
                load_config(config_path, load_env=False)
        finally:
            os.unlink(config_path)

    def test_same_direction_exceeds_max_positions(self, valid_config_data):
        """Test validation fails when max_same_direction > max_open_positions"""
        valid_config_data["risk"]["max_open_positions"] = 2
        valid_config_data["risk"]["max_same_direction_positions"] = 3
        config_path = create_temp_config(valid_config_data)
        try:
            with pytest.raises(ValueError, match="max_same_direction_positions cannot exceed"):
                load_config(config_path, load_env=False)
        finally:
            os.unlink(config_path)

    def test_total_risk_less_than_per_trade(self, valid_config_data):
        """Test validation fails when max_total_open_risk < risk_per_trade"""
        valid_config_data["risk"]["risk_per_trade_pct"] = 0.03
        valid_config_data["risk"]["max_total_open_risk_pct"] = 0.02
        config_path = create_temp_config(valid_config_data)
        try:
            with pytest.raises(ValueError, match="max_total_open_risk_pct must be >="):
                load_config(config_path, load_env=False)
        finally:
            os.unlink(config_path)

    def test_reduced_risk_exceeds_normal(self, valid_config_data):
        """Test validation fails when reduced_risk >= normal risk"""
        valid_config_data["risk"]["risk_per_trade_pct"] = 0.01
        valid_config_data["risk"]["reduced_risk_after_pause_pct"] = 0.015
        config_path = create_temp_config(valid_config_data)
        try:
            with pytest.raises(ValueError, match="reduced_risk_after_pause_pct must be <"):
                load_config(config_path, load_env=False)
        finally:
            os.unlink(config_path)

    def test_daily_stop_worse_than_weekly(self, valid_config_data):
        """Test validation fails when daily stop is worse than weekly stop"""
        valid_config_data["risk"]["daily_stop_pct"] = -0.15
        valid_config_data["risk"]["weekly_stop_pct"] = -0.1
        config_path = create_temp_config(valid_config_data)
        try:
            with pytest.raises(ValueError, match="daily_stop_pct must be less severe"):
                load_config(config_path, load_env=False)
        finally:
            os.unlink(config_path)

    def test_hedge_corr_exceeds_threshold(self, valid_config_data):
        """Test validation fails when hedge_corr_max >= correlation_threshold"""
        valid_config_data["risk"]["correlation_threshold"] = 0.85
        valid_config_data["risk"]["hedge_corr_max"] = 0.9
        config_path = create_temp_config(valid_config_data)
        try:
            with pytest.raises(ValueError, match="hedge_corr_max must be <"):
                load_config(config_path, load_env=False)
        finally:
            os.unlink(config_path)

    def test_adx_range_overlap(self, valid_config_data):
        """Test validation fails when ADX ranges overlap"""
        valid_config_data["regime"]["range_adx_max"] = 30
        valid_config_data["regime"]["trend_adx_min"] = 25
        config_path = create_temp_config(valid_config_data)
        try:
            with pytest.raises(ValueError, match="range_adx_max must be <"):
                load_config(config_path, load_env=False)
        finally:
            os.unlink(config_path)

    def test_invalid_rsi_range(self, valid_config_data):
        """Test validation fails with invalid RSI range"""
        valid_config_data["strategies"]["trend_pullback"]["pullback_rsi_long_min"] = 50
        valid_config_data["strategies"]["trend_pullback"]["pullback_rsi_long_max"] = 40
        config_path = create_temp_config(valid_config_data)
        try:
            with pytest.raises(ValueError, match="pullback_rsi_long_min must be <"):
                load_config(config_path, load_env=False)
        finally:
            os.unlink(config_path)

    def test_invalid_leverage_too_high(self, valid_config_data):
        """Test validation fails when leverage > 2.0"""
        valid_config_data["leverage"]["trend"] = 3.0
        config_path = create_temp_config(valid_config_data)
        try:
            with pytest.raises(ValueError, match="validation failed"):
                load_config(config_path, load_env=False)
        finally:
            os.unlink(config_path)


class TestEnvVarSubstitution:
    """Test environment variable substitution"""

    def test_env_var_substitution(self):
        """Test that environment variables are substituted correctly"""
        os.environ["TEST_API_KEY"] = "test_key_123"
        os.environ["TEST_API_SECRET"] = "test_secret_456"

        data = {
            "api_key": "${TEST_API_KEY}",
            "api_secret": "${TEST_API_SECRET}",
            "nested": {
                "value": "${TEST_API_KEY}"
            }
        }

        result = substitute_env_vars(data)

        assert result["api_key"] == "test_key_123"
        assert result["api_secret"] == "test_secret_456"
        assert result["nested"]["value"] == "test_key_123"

        # Cleanup
        del os.environ["TEST_API_KEY"]
        del os.environ["TEST_API_SECRET"]

    def test_missing_env_var_raises_error(self):
        """Test that missing environment variable raises error"""
        data = {"api_key": "${NONEXISTENT_VAR}"}

        with pytest.raises(ValueError, match="Environment variable NONEXISTENT_VAR not found"):
            substitute_env_vars(data)


class TestAdditionalValidation:
    """Test additional validation constraints"""

    def test_all_strategies_disabled(self, valid_config_data):
        """Test that at least one strategy must be enabled"""
        valid_config_data["strategies"]["trend_pullback"]["enabled"] = False
        valid_config_data["strategies"]["trend_breakout"]["enabled"] = False
        valid_config_data["strategies"]["range_mean_reversion"]["enabled"] = False

        config_path = create_temp_config(valid_config_data)
        try:
            with pytest.raises(ValueError, match="At least one strategy must be enabled"):
                load_config(config_path, load_env=False)
        finally:
            os.unlink(config_path)

    def test_invalid_margin_mode(self, valid_config_data):
        """Test that only ISOLATED margin mode is allowed"""
        valid_config_data["exchange"]["margin_mode"] = "CROSS"

        config_path = create_temp_config(valid_config_data)
        try:
            with pytest.raises(ValueError, match="Only ISOLATED margin mode is supported"):
                load_config(config_path, load_env=False)
        finally:
            os.unlink(config_path)

    def test_leverage_exceeds_spec_limit(self, valid_config_data):
        """Test that leverage cannot exceed 2.0x per spec"""
        valid_config_data["leverage"]["trend"] = 2.5

        config_path = create_temp_config(valid_config_data)
        try:
            # This should fail at Pydantic level (le=2.0)
            with pytest.raises(ValueError, match="validation failed"):
                load_config(config_path, load_env=False)
        finally:
            os.unlink(config_path)

    def test_whitelist_blacklist_overlap(self, valid_config_data):
        """Test that whitelist and blacklist cannot overlap"""
        valid_config_data["universe"]["whitelist"] = ["BTCUSDT", "ETHUSDT"]
        valid_config_data["universe"]["blacklist"] = ["ETHUSDT", "BNBUSDT"]

        config_path = create_temp_config(valid_config_data)
        try:
            with pytest.raises(ValueError, match="appear in both whitelist and blacklist"):
                load_config(config_path, load_env=False)
        finally:
            os.unlink(config_path)


class TestMissingFields:
    """Test handling of missing required fields"""

    def test_missing_mode(self, valid_config_data):
        """Test that missing mode field raises error"""
        del valid_config_data["mode"]
        config_path = create_temp_config(valid_config_data)
        try:
            with pytest.raises(ValueError, match="validation failed"):
                load_config(config_path, load_env=False)
        finally:
            os.unlink(config_path)

    def test_missing_risk_section(self, valid_config_data):
        """Test that missing risk section raises error"""
        del valid_config_data["risk"]
        config_path = create_temp_config(valid_config_data)
        try:
            with pytest.raises(ValueError, match="validation failed"):
                load_config(config_path, load_env=False)
        finally:
            os.unlink(config_path)
