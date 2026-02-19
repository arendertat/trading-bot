"""Main entry point for the trading bot"""

import os
import sys

from bot.config.loader import load_config


def main() -> None:
    """Main entry point — load config and start BotRunner."""
    import argparse
    parser = argparse.ArgumentParser(description="Binance USDT-M Futures Trading Bot")
    parser.add_argument(
        "config_path",
        nargs="?",
        default=None,
        help="Path to config JSON file (overrides CONFIG_PATH env var)"
    )
    parser.add_argument(
        "--close-all",
        action="store_true",
        default=False,
        help="Close all open positions when bot shuts down (SIGINT/SIGTERM)"
    )
    args = parser.parse_args()

    config_path = (
        os.getenv("CONFIG_PATH")
        or args.config_path
        or "config/config.json"
    )

    print(f"Loading config from: {config_path}")
    if args.close_all:
        print("[WARNING] --close-all active: positions will be closed on shutdown")

    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"[ERROR] Config file not found: {e}")
        print("  → Ensure config/config.json exists")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] Config validation failed: {e}")
        sys.exit(1)

    # Import runner here to avoid issues at module import time
    from bot.runner import BotRunner

    try:
        runner = BotRunner(config, close_on_shutdown=args.close_all)
        runner.start()
    except ValueError as e:
        # API credentials missing or config error
        print(f"\n[ERROR] {e}")
        print("\nSetup checklist:")
        print("  1. Copy .env.example → .env")
        print("  2. Fill in BINANCE_API_KEY and BINANCE_API_SECRET")
        print("     (Testnet keys from https://testnet.binancefuture.com)")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
        sys.exit(0)
    except Exception as e:
        # Bulgu 9.1: Print error type only; full traceback goes to stderr (not stdout logs)
        # to avoid leaking exchange response details containing credential fragments.
        print(f"\n[FATAL] {type(e).__name__}: {e}")
        import traceback
        import sys as _sys
        traceback.print_exc(file=_sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
