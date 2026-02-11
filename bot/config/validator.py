"""Additional configuration validation logic"""

from bot.config.models import BotConfig


def validate_config_constraints(config: BotConfig) -> None:
    """
    Perform additional cross-field validation beyond Pydantic model validators.

    Args:
        config: BotConfig instance to validate

    Raises:
        ValueError: If validation fails
    """
    # Validate that at least one strategy is enabled
    strategies = [
        config.strategies.trend_pullback.enabled,
        config.strategies.trend_breakout.enabled,
        config.strategies.range_mean_reversion.enabled,
    ]
    if not any(strategies):
        raise ValueError("At least one strategy must be enabled")

    # Validate timezone
    if config.timezone != "UTC":
        raise ValueError("Only UTC timezone is currently supported")

    # Validate exchange
    if config.exchange.name.lower() != "binance":
        raise ValueError("Only Binance exchange is currently supported")

    if config.exchange.margin_mode != "ISOLATED":
        raise ValueError("Only ISOLATED margin mode is supported (per spec)")

    # Validate leverage constraints (max 2x per spec)
    if config.leverage.trend > 2.0:
        raise ValueError("trend leverage cannot exceed 2.0x (per spec)")
    if config.leverage.range > 2.0:
        raise ValueError("range leverage cannot exceed 2.0x (per spec)")
    if config.leverage.high_vol > 2.0:
        raise ValueError("high_vol leverage cannot exceed 2.0x (per spec)")

    # Validate that performance window is larger than min trades
    if config.performance.window_trades < config.performance.min_trades_before_confidence:
        raise ValueError(
            "window_trades must be >= min_trades_before_confidence"
        )

    # Validate universe constraints
    if config.universe.whitelist and config.universe.blacklist:
        # Check for overlap
        overlap = set(config.universe.whitelist) & set(config.universe.blacklist)
        if overlap:
            raise ValueError(
                f"Symbol(s) appear in both whitelist and blacklist: {overlap}"
            )
