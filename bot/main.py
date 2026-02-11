"""Main entry point for the trading bot"""

import sys
from pathlib import Path

from bot.config.loader import load_config
from bot.config.validator import validate_config_constraints
from bot.utils.logger import setup_logging


def main():
    """Main entry point"""
    try:
        # Load configuration
        config_path = "config/config.json"
        print(f"Loading config from {config_path}...")
        config = load_config(config_path)

        # Additional validation
        validate_config_constraints(config)

        # Setup logging
        logger = setup_logging(config.logging)
        logger.info("=" * 60)
        logger.info("Binance USDT-M Futures Trading Bot - Milestone 1")
        logger.info("=" * 60)
        logger.info(f"Mode: {config.mode}")
        logger.info(f"Exchange: {config.exchange.name}")
        logger.info(f"Margin Mode: {config.exchange.margin_mode}")
        logger.info(f"Max Open Positions: {config.risk.max_open_positions}")
        logger.info(f"Risk Per Trade: {config.risk.risk_per_trade_pct * 100:.2f}%")
        logger.info(f"Max Total Open Risk: {config.risk.max_total_open_risk_pct * 100:.2f}%")
        logger.info(f"Leverage - Trend: {config.leverage.trend}x, Range: {config.leverage.range}x, High Vol: {config.leverage.high_vol}x")
        logger.info("=" * 60)
        logger.info("[OK] Config validation successful")
        logger.info("[OK] Bot skeleton initialized")
        logger.info("=" * 60)
        logger.info("Note: Trading logic not yet implemented (Milestone 2+)")
        logger.info("Exiting gracefully.")

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create config/config.json from config/config.example.json")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
