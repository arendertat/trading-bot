"""Pydantic models for configuration validation"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from bot.core.constants import BotMode


class ExchangeConfig(BaseModel):
    """Exchange connection configuration"""
    name: str = "binance"
    api_key_env: str = "BINANCE_API_KEY"
    api_secret_env: str = "BINANCE_API_SECRET"
    usdtm_perp: bool = True
    margin_mode: str = "ISOLATED"
    recv_window_ms: int = Field(default=5000, gt=0, le=60000)
    testnet: bool = False  # Use Binance Futures Testnet (https://testnet.binancefuture.com)

    @field_validator("margin_mode")
    @classmethod
    def validate_margin_mode(cls, v: str) -> str:
        if v.upper() not in ["ISOLATED", "CROSS"]:
            raise ValueError("margin_mode must be ISOLATED or CROSS")
        return v.upper()


class UniverseConfig(BaseModel):
    """Universe selection configuration"""
    min_24h_volume_usdt: float = Field(default=100_000_000, gt=0)
    max_spread_pct: float = Field(default=0.0005, gt=0, le=0.01)
    max_abs_funding_rate: float = Field(default=0.0015, gt=0, le=0.01)
    min_atr_ratio: float = Field(default=0.005, gt=0, le=0.1)
    max_monitored_symbols: int = Field(default=6, ge=1, le=20)
    whitelist: List[str] = Field(default_factory=list)
    blacklist: List[str] = Field(default_factory=list)
    hedge_max_combined_funding: float = Field(default=0.0015, gt=0, le=0.01)


class TimeframesConfig(BaseModel):
    """Timeframe configuration"""
    signal_tf: str = "5m"
    trend_tf: str = "1h"
    zscore_lookback: int = Field(default=100, ge=20, le=500)
    corr_tf: str = "1h"
    corr_lookback_hours: int = Field(default=72, ge=24, le=720)

    @field_validator("signal_tf", "trend_tf", "corr_tf")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        valid_tfs = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]
        if v not in valid_tfs:
            raise ValueError(f"Invalid timeframe: {v}. Must be one of {valid_tfs}")
        return v


class RiskConfig(BaseModel):
    """Risk management configuration"""
    risk_per_trade_pct: float = Field(default=0.01, gt=0, le=0.1)  # 0.01 = 1%
    max_total_open_risk_pct: float = Field(default=0.025, gt=0, le=0.1)  # 0.025 = 2.5%
    max_open_positions: int = Field(default=2, ge=1, le=10)
    max_same_direction_positions: int = Field(default=2, ge=1, le=10)
    correlation_threshold: float = Field(default=0.85, ge=0, le=1.0)
    hedge_corr_max: float = Field(default=0.6, ge=0, le=1.0)
    daily_stop_pct: float = Field(default=-0.04, ge=-0.5, lt=0)  # -0.04 = -4%
    weekly_stop_pct: float = Field(default=-0.1, ge=-0.5, lt=0)  # -0.1 = -10%
    pause_days_after_weekly_stop: int = Field(default=7, ge=1, le=30)
    reduced_risk_after_pause_pct: float = Field(default=0.005, gt=0, le=0.1)  # 0.005 = 0.5%
    reduced_risk_days: int = Field(default=3, ge=1, le=30)


class RegimeConfig(BaseModel):
    """Regime detection configuration"""
    trend_adx_min: float = Field(default=25, ge=15, le=50)
    range_adx_max: float = Field(default=20, ge=10, le=30)
    high_vol_atr_z: float = Field(default=1.5, ge=1.0, le=5.0)
    confidence_threshold: float = Field(default=0.55, ge=0.3, le=0.9)
    bb_width_range_min: Optional[float] = Field(default=0.01, ge=0, le=0.1)
    bb_width_range_max: Optional[float] = Field(default=0.05, ge=0, le=0.2)


class StrategyTrendPullbackConfig(BaseModel):
    """Trend Pullback strategy configuration"""
    enabled: bool = True
    stop_pct: float = Field(default=0.01, gt=0, le=0.05)  # 0.01 = 1%
    target_r_multiple: float = Field(default=1.5, ge=0.5, le=5.0)
    pullback_rsi_long_min: float = Field(default=40, ge=0, le=100)
    pullback_rsi_long_max: float = Field(default=50, ge=0, le=100)
    pullback_rsi_short_min: float = Field(default=50, ge=0, le=100)
    pullback_rsi_short_max: float = Field(default=60, ge=0, le=100)
    ema20_band_pct: float = Field(default=0.002, ge=0, le=0.05)
    trail_after_r: float = Field(default=1.0, ge=0, le=5.0)
    atr_trail_mult: float = Field(default=2.0, ge=0.5, le=10.0)


class StrategyTrendBreakoutConfig(BaseModel):
    """Trend Breakout strategy configuration"""
    enabled: bool = True
    stop_pct: float = Field(default=0.01, gt=0, le=0.05)
    breakout_lookback_bars: int = Field(default=20, ge=5, le=100)
    breakout_volume_z_min: float = Field(default=1.0, ge=0, le=5.0)
    atr_trail_mult: float = Field(default=2.5, ge=0.5, le=10.0)


class StrategyRangeMeanReversionConfig(BaseModel):
    """Range Mean Reversion strategy configuration"""
    enabled: bool = True
    stop_pct: float = Field(default=0.008, gt=0, le=0.05)
    target_r_multiple: float = Field(default=1.2, ge=0.5, le=5.0)
    rsi_long_extreme: float = Field(default=25, ge=0, le=50)
    rsi_short_extreme: float = Field(default=75, ge=50, le=100)


class StrategiesConfig(BaseModel):
    """All strategies configuration"""
    trend_pullback: StrategyTrendPullbackConfig
    trend_breakout: StrategyTrendBreakoutConfig
    range_mean_reversion: StrategyRangeMeanReversionConfig


class LeverageConfig(BaseModel):
    """Leverage mapping per regime"""
    trend: float = Field(default=2.0, ge=1.0, le=2.0)
    range: float = Field(default=1.5, ge=1.0, le=2.0)
    high_vol: float = Field(default=1.0, ge=1.0, le=2.0)


class ExecutionConfig(BaseModel):
    """Execution configuration"""
    entry_order_type: str = "LIMIT"
    limit_ttl_seconds: int = Field(default=30, ge=5, le=300)
    limit_retry_count: int = Field(default=1, ge=0, le=5)
    stop_order_type: str = "STOP_MARKET"
    kill_switch_order_type: str = "MARKET"
    paper_slippage_limit_pct: float = Field(default=0.0002, ge=0, le=0.01)
    paper_slippage_market_pct: float = Field(default=0.0008, ge=0, le=0.02)
    paper_slippage_stop_pct: float = Field(default=0.001, ge=0, le=0.02)
    maker_fee_pct: float = Field(default=0.0002, ge=0, le=0.01)
    taker_fee_pct: float = Field(default=0.0004, ge=0, le=0.01)
    enable_funding_in_paper: bool = False


class PerformanceConfig(BaseModel):
    """Performance tracking configuration"""
    window_trades: int = Field(default=50, ge=10, le=500)
    min_trades_before_confidence: int = Field(default=20, ge=5, le=100)
    dd_penalty_weight: float = Field(default=0.5, ge=0, le=2.0)
    max_strategy_switches_per_day: int = Field(default=1, ge=0, le=10)


class NotificationConfig(BaseModel):
    """Notification configuration"""
    telegram_enabled: bool = False
    telegram_token_env: str = "TELEGRAM_BOT_TOKEN"
    telegram_chat_id_env: str = "TELEGRAM_CHAT_ID"
    daily_report_time_utc: str = "00:05"

    @field_validator("daily_report_time_utc")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        import re
        if not re.match(r"^\d{2}:\d{2}$", v):
            raise ValueError("daily_report_time_utc must be in HH:MM format")
        hours, minutes = map(int, v.split(":"))
        if hours >= 24 or minutes >= 60:
            raise ValueError("Invalid time format")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration"""
    log_dir: str = "./logs"
    trade_log_file: str = "trades.jsonl"
    event_log_file: str = "events.jsonl"
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()


class BotConfig(BaseModel):
    """Root configuration model"""
    mode: BotMode
    timezone: str = "UTC"
    exchange: ExchangeConfig
    universe: UniverseConfig
    timeframes: TimeframesConfig
    risk: RiskConfig
    regime: RegimeConfig
    strategies: StrategiesConfig
    leverage: LeverageConfig
    execution: ExecutionConfig
    performance: PerformanceConfig
    notifications: NotificationConfig
    logging: LoggingConfig

    @model_validator(mode="after")
    def validate_config(self):
        """Cross-field validation"""
        # Validate risk constraints
        if self.risk.max_same_direction_positions > self.risk.max_open_positions:
            raise ValueError(
                "max_same_direction_positions cannot exceed max_open_positions"
            )

        # Validate total open risk > per trade risk
        if self.risk.max_total_open_risk_pct < self.risk.risk_per_trade_pct:
            raise ValueError(
                "max_total_open_risk_pct must be >= risk_per_trade_pct"
            )

        # Validate reduced risk < normal risk
        if self.risk.reduced_risk_after_pause_pct >= self.risk.risk_per_trade_pct:
            raise ValueError(
                "reduced_risk_after_pause_pct must be < risk_per_trade_pct"
            )

        # Validate daily stop is less severe than weekly stop
        if self.risk.daily_stop_pct < self.risk.weekly_stop_pct:
            raise ValueError(
                "daily_stop_pct must be less severe than weekly_stop_pct "
                "(closer to zero)"
            )

        # Validate hedge correlation is less than bucket correlation
        if self.risk.hedge_corr_max >= self.risk.correlation_threshold:
            raise ValueError(
                "hedge_corr_max must be < correlation_threshold"
            )

        # Validate ADX ranges don't overlap
        if self.regime.range_adx_max >= self.regime.trend_adx_min:
            raise ValueError(
                "range_adx_max must be < trend_adx_min (no overlap)"
            )

        # Validate BB width range
        if self.regime.bb_width_range_min and self.regime.bb_width_range_max:
            if self.regime.bb_width_range_min >= self.regime.bb_width_range_max:
                raise ValueError(
                    "bb_width_range_min must be < bb_width_range_max"
                )

        # Validate RSI ranges for trend pullback
        tp = self.strategies.trend_pullback
        if tp.pullback_rsi_long_min >= tp.pullback_rsi_long_max:
            raise ValueError(
                "pullback_rsi_long_min must be < pullback_rsi_long_max"
            )
        if tp.pullback_rsi_short_min >= tp.pullback_rsi_short_max:
            raise ValueError(
                "pullback_rsi_short_min must be < pullback_rsi_short_max"
            )

        # Validate mean reversion RSI extremes
        mr = self.strategies.range_mean_reversion
        if mr.rsi_long_extreme >= mr.rsi_short_extreme:
            raise ValueError(
                "rsi_long_extreme must be < rsi_short_extreme"
            )

        return self
