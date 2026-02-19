#!/usr/bin/env python3
"""
Verification script for universe selection module.

Checks:
1. All module files compile without errors
2. All imports work correctly
3. Classes can be instantiated (with mocks)
4. Key methods exist and are callable
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def verify_imports():
    """Verify all imports work"""
    print("[1/4] Verifying imports...")

    try:
        # Core universe module
        from bot.universe.models import SymbolEligibility
        from bot.universe.selector import UniverseSelector

        # Dependencies
        from bot.config.models import UniverseConfig
        from bot.exchange.binance_client import BinanceFuturesClient
        from bot.data.candle_store import CandleStore

        print("      ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"      ✗ Import failed: {e}")
        return False


def verify_dataclass():
    """Verify SymbolEligibility dataclass"""
    print("[2/4] Verifying SymbolEligibility dataclass...")

    try:
        from bot.universe.models import SymbolEligibility

        # Create instance
        eligibility = SymbolEligibility(
            symbol="BTC/USDT",
            pass_volume=True,
            pass_spread=True,
            pass_funding=True,
            pass_atr_ratio=True,
            score=100.0,
            reasons=[],
        )

        # Check property
        assert eligibility.is_eligible is True

        # Create failing instance
        eligibility_fail = SymbolEligibility(
            symbol="ETH/USDT",
            pass_volume=False,
            pass_spread=True,
            pass_funding=True,
            pass_atr_ratio=True,
            score=0.0,
            reasons=["Volume too low"],
        )

        assert eligibility_fail.is_eligible is False

        print("      ✓ SymbolEligibility works correctly")
        return True
    except Exception as e:
        print(f"      ✗ SymbolEligibility failed: {e}")
        return False


def verify_universe_selector():
    """Verify UniverseSelector class"""
    print("[3/4] Verifying UniverseSelector class...")

    try:
        from bot.universe.selector import UniverseSelector
        from bot.config.models import UniverseConfig
        from bot.data.candle_store import CandleStore
        from unittest.mock import Mock

        # Create mock client
        mock_client = Mock()

        # Create config
        config = UniverseConfig(
            min_24h_volume_usdt=100_000_000,
            max_spread_pct=0.0005,
            max_abs_funding_rate=0.0015,
            min_atr_ratio=0.005,
            max_monitored_symbols=6,
            whitelist=[],
            blacklist=[],
            hedge_max_combined_funding=0.0015,
        )

        # Create candle store
        store = CandleStore()

        # Instantiate selector
        selector = UniverseSelector(
            exchange_client=mock_client,
            candle_store=store,
            config=config,
        )

        # Verify key methods exist
        assert hasattr(selector, 'build_daily_universe')
        assert callable(selector.build_daily_universe)

        assert hasattr(selector, '_apply_whitelist_blacklist')
        assert callable(selector._apply_whitelist_blacklist)

        assert hasattr(selector, '_evaluate_symbol')
        assert callable(selector._evaluate_symbol)

        # Test whitelist/blacklist method
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        result = selector._apply_whitelist_blacklist(symbols)
        assert result == symbols

        print("      ✓ UniverseSelector instantiation and methods OK")
        return True
    except Exception as e:
        print(f"      ✗ UniverseSelector failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_exchange_extensions():
    """Verify exchange client extensions"""
    print("[4/4] Verifying exchange client extensions...")

    try:
        from bot.exchange.binance_client import BinanceFuturesClient

        # Check new methods exist
        methods = [
            'list_usdtm_perp_symbols',
            'fetch_24h_tickers',
            'fetch_funding_rates',
        ]

        for method in methods:
            assert hasattr(BinanceFuturesClient, method), f"Missing method: {method}"
            assert callable(getattr(BinanceFuturesClient, method))

        print("      ✓ Exchange client extensions OK")
        return True
    except Exception as e:
        print(f"      ✗ Exchange client extensions failed: {e}")
        return False


def main():
    """Run all verification checks"""
    print("=" * 60)
    print("Universe Selection Module Verification")
    print("=" * 60)
    print()

    # Check dependencies first
    print("Checking dependencies...")
    try:
        import pandas
        import ccxt
        import pydantic
        print("      ✓ Required dependencies installed")
        print()
    except ImportError as e:
        print(f"      ✗ Missing dependency: {e}")
        print()
        print("=" * 60)
        print("DEPENDENCIES NOT INSTALLED")
        print("=" * 60)
        print()
        print("Please install dependencies:")
        print("  pip install -r requirements.txt")
        print()
        print("Then re-run this script.")
        return 1

    results = []

    results.append(verify_imports())
    results.append(verify_dataclass())
    results.append(verify_universe_selector())
    results.append(verify_exchange_extensions())

    print()
    print("=" * 60)

    if all(results):
        print("✓ ALL CHECKS PASSED")
        print("=" * 60)
        print()
        print("Universe selection module is ready to use!")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("=" * 60)
        print()
        print("Please fix the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
