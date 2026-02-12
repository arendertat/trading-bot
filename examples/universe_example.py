#!/usr/bin/env python3
"""
Example script demonstrating universe selection module usage.

This is a standalone example showing how to:
1. Initialize the universe selector
2. Build a daily universe
3. Inspect eligibility results

NOTE: This is for demonstration only. In production, this would be
integrated into the main bot loop with daily scheduling.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.config.loader import load_config
from bot.exchange.binance_client import BinanceFuturesClient
from bot.data.candle_store import CandleStore
from bot.universe.selector import UniverseSelector
from bot.utils.logger import setup_logging


def main():
    """Run universe selection example"""

    print("=" * 60)
    print("Universe Selection Example")
    print("=" * 60)
    print()

    # Load configuration
    print("[1/5] Loading configuration...")
    try:
        config = load_config("config/config.json")
        print(f"      Mode: {config.mode}")
        print(f"      Max monitored symbols: {config.universe.max_monitored_symbols}")
        print(f"      Min 24h volume: ${config.universe.min_24h_volume_usdt:,.0f}")
        print(f"      Max spread: {config.universe.max_spread_pct:.4%}")
        print(f"      Max funding rate: {config.universe.max_abs_funding_rate:.4%}")
        print(f"      Min ATR ratio: {config.universe.min_atr_ratio:.4%}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        print("Please ensure config/config.json exists.")
        return 1

    # Setup logging
    logger = setup_logging(config.logging)

    # Initialize exchange client
    print("[2/5] Initializing exchange client...")
    try:
        exchange_client = BinanceFuturesClient.from_config(config.exchange)

        # Test connectivity
        if not exchange_client.ping():
            print("ERROR: Cannot reach exchange")
            return 1

        print("      Exchange client initialized successfully")
        print()
    except Exception as e:
        print(f"ERROR: Failed to initialize exchange client: {e}")
        print("Please check API credentials in .env file.")
        return 1

    # Initialize candle store
    print("[3/5] Initializing candle store...")
    candle_store = CandleStore()
    print("      Candle store initialized")
    print()

    # Create universe selector
    print("[4/5] Creating universe selector...")
    selector = UniverseSelector(
        exchange_client=exchange_client,
        candle_store=candle_store,
        config=config.universe,
    )
    print("      Universe selector ready")
    print()

    # Build daily universe
    print("[5/5] Building daily universe...")
    print("      This may take 1-2 minutes (fetching market data)...")
    print()

    try:
        now_utc = datetime.utcnow()
        selected_symbols = selector.build_daily_universe(now_utc)

        print("-" * 60)
        print("RESULTS")
        print("-" * 60)
        print()
        print(f"Timestamp: {now_utc.isoformat()} UTC")
        print(f"Selected Symbols: {len(selected_symbols)}")
        print()

        if selected_symbols:
            print("Symbols (ranked by liquidity/quality score):")
            for i, symbol in enumerate(selected_symbols, 1):
                print(f"  {i}. {symbol}")
        else:
            print("WARNING: No symbols selected (all failed filters)")

        print()
        print("=" * 60)
        print("Example complete!")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"ERROR: Failed to build universe: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
