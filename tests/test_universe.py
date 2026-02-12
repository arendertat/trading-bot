"""Unit tests for universe selection module"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from bot.universe.selector import UniverseSelector
from bot.universe.models import SymbolEligibility
from bot.config.models import UniverseConfig
from bot.exchange.binance_client import BinanceFuturesClient
from bot.data.candle_store import CandleStore
from bot.core.types import Candle


@pytest.fixture
def universe_config():
    """Default universe configuration"""
    return UniverseConfig(
        min_24h_volume_usdt=100_000_000,
        max_spread_pct=0.0005,
        max_abs_funding_rate=0.0015,
        min_atr_ratio=0.005,
        max_monitored_symbols=6,
        whitelist=[],
        blacklist=[],
        hedge_max_combined_funding=0.0015,
    )


@pytest.fixture
def mock_exchange_client():
    """Mock Binance futures client"""
    client = Mock(spec=BinanceFuturesClient)
    return client


@pytest.fixture
def candle_store():
    """Real candle store for testing"""
    return CandleStore()


@pytest.fixture
def universe_selector(mock_exchange_client, candle_store, universe_config):
    """UniverseSelector instance with mocks"""
    return UniverseSelector(
        exchange_client=mock_exchange_client,
        candle_store=candle_store,
        config=universe_config,
    )


class TestSymbolEligibility:
    """Test SymbolEligibility dataclass"""

    def test_is_eligible_all_pass(self):
        """Test that symbol is eligible when all checks pass"""
        eligibility = SymbolEligibility(
            symbol="BTC/USDT",
            pass_volume=True,
            pass_spread=True,
            pass_funding=True,
            pass_atr_ratio=True,
            score=100.0,
            reasons=[],
        )

        assert eligibility.is_eligible

    def test_is_eligible_one_fails(self):
        """Test that symbol is not eligible if any check fails"""
        # Volume fails
        eligibility = SymbolEligibility(
            symbol="BTC/USDT",
            pass_volume=False,
            pass_spread=True,
            pass_funding=True,
            pass_atr_ratio=True,
            score=0.0,
            reasons=["Volume too low"],
        )

        assert not eligibility.is_eligible

    def test_is_eligible_all_fail(self):
        """Test that symbol is not eligible when all checks fail"""
        eligibility = SymbolEligibility(
            symbol="BTC/USDT",
            pass_volume=False,
            pass_spread=False,
            pass_funding=False,
            pass_atr_ratio=False,
            score=0.0,
            reasons=["All checks failed"],
        )

        assert not eligibility.is_eligible


class TestWhitelistBlacklist:
    """Test whitelist and blacklist filtering"""

    def test_no_whitelist_no_blacklist(self, universe_selector):
        """Test that all symbols pass through when no whitelist/blacklist"""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

        result = universe_selector._apply_whitelist_blacklist(symbols)

        assert set(result) == set(symbols)

    def test_whitelist_only(self, mock_exchange_client, candle_store):
        """Test whitelist filtering"""
        config = UniverseConfig(
            min_24h_volume_usdt=100_000_000,
            max_spread_pct=0.0005,
            max_abs_funding_rate=0.0015,
            min_atr_ratio=0.005,
            max_monitored_symbols=6,
            whitelist=["BTC/USDT", "ETH/USDT"],
            blacklist=[],
        )
        selector = UniverseSelector(mock_exchange_client, candle_store, config)

        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "MATIC/USDT"]

        result = selector._apply_whitelist_blacklist(symbols)

        assert set(result) == {"BTC/USDT", "ETH/USDT"}

    def test_blacklist_only(self, mock_exchange_client, candle_store):
        """Test blacklist filtering"""
        config = UniverseConfig(
            min_24h_volume_usdt=100_000_000,
            max_spread_pct=0.0005,
            max_abs_funding_rate=0.0015,
            min_atr_ratio=0.005,
            max_monitored_symbols=6,
            whitelist=[],
            blacklist=["SOL/USDT"],
        )
        selector = UniverseSelector(mock_exchange_client, candle_store, config)

        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

        result = selector._apply_whitelist_blacklist(symbols)

        assert set(result) == {"BTC/USDT", "ETH/USDT"}

    def test_whitelist_and_blacklist(self, mock_exchange_client, candle_store):
        """Test that blacklist takes precedence over whitelist"""
        config = UniverseConfig(
            min_24h_volume_usdt=100_000_000,
            max_spread_pct=0.0005,
            max_abs_funding_rate=0.0015,
            min_atr_ratio=0.005,
            max_monitored_symbols=6,
            whitelist=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            blacklist=["SOL/USDT"],
        )
        selector = UniverseSelector(mock_exchange_client, candle_store, config)

        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "MATIC/USDT"]

        result = selector._apply_whitelist_blacklist(symbols)

        # SOL/USDT should be excluded by blacklist
        assert set(result) == {"BTC/USDT", "ETH/USDT"}

    def test_empty_symbols_list(self, universe_selector):
        """Test that empty input returns empty output"""
        symbols = []

        result = universe_selector._apply_whitelist_blacklist(symbols)

        assert result == []


class TestVolumeFilter:
    """Test 24h volume filtering"""

    def test_volume_pass(self, universe_selector):
        """Test symbol passes volume filter"""
        ticker = {
            'quote_volume_usdt': 150_000_000,
            'bid': 50000.0,
            'ask': 50010.0,
        }

        eligibility = universe_selector._evaluate_symbol(
            symbol="BTC/USDT",
            ticker=ticker,
            funding_rate=0.0001,
        )

        assert eligibility.pass_volume

    def test_volume_fail(self, universe_selector):
        """Test symbol fails volume filter"""
        ticker = {
            'quote_volume_usdt': 50_000_000,  # Below min of 100M
            'bid': 50000.0,
            'ask': 50010.0,
        }

        eligibility = universe_selector._evaluate_symbol(
            symbol="BTC/USDT",
            ticker=ticker,
            funding_rate=0.0001,
        )

        assert not eligibility.pass_volume
        assert any("Volume too low" in r for r in eligibility.reasons)

    def test_volume_exact_threshold(self, universe_selector):
        """Test symbol passes when volume equals threshold"""
        ticker = {
            'quote_volume_usdt': 100_000_000,  # Exact min
            'bid': 50000.0,
            'ask': 50010.0,
        }

        eligibility = universe_selector._evaluate_symbol(
            symbol="BTC/USDT",
            ticker=ticker,
            funding_rate=0.0001,
        )

        assert eligibility.pass_volume


class TestSpreadFilter:
    """Test spread filtering"""

    def test_spread_pass(self, universe_selector):
        """Test symbol passes spread filter"""
        ticker = {
            'quote_volume_usdt': 150_000_000,
            'bid': 50000.0,
            'ask': 50020.0,  # 0.0004 = 0.04% spread
        }

        eligibility = universe_selector._evaluate_symbol(
            symbol="BTC/USDT",
            ticker=ticker,
            funding_rate=0.0001,
        )

        assert eligibility.pass_spread

    def test_spread_fail(self, universe_selector):
        """Test symbol fails spread filter"""
        ticker = {
            'quote_volume_usdt': 150_000_000,
            'bid': 50000.0,
            'ask': 50040.0,  # 0.0008 = 0.08% spread (> 0.05%)
        }

        eligibility = universe_selector._evaluate_symbol(
            symbol="BTC/USDT",
            ticker=ticker,
            funding_rate=0.0001,
        )

        assert not eligibility.pass_spread
        assert any("Spread too wide" in r for r in eligibility.reasons)

    def test_missing_bid_ask(self, universe_selector):
        """Test symbol fails when bid/ask missing"""
        ticker = {
            'quote_volume_usdt': 150_000_000,
            'bid': None,
            'ask': None,
        }

        eligibility = universe_selector._evaluate_symbol(
            symbol="BTC/USDT",
            ticker=ticker,
            funding_rate=0.0001,
        )

        assert not eligibility.pass_spread
        assert any("Missing bid/ask" in r for r in eligibility.reasons)


class TestFundingRateFilter:
    """Test funding rate filtering"""

    def test_funding_pass_positive(self, universe_selector):
        """Test symbol passes funding filter (positive rate)"""
        ticker = {
            'quote_volume_usdt': 150_000_000,
            'bid': 50000.0,
            'ask': 50010.0,
        }

        eligibility = universe_selector._evaluate_symbol(
            symbol="BTC/USDT",
            ticker=ticker,
            funding_rate=0.0010,  # Within 0.0015 limit
        )

        assert eligibility.pass_funding

    def test_funding_pass_negative(self, universe_selector):
        """Test symbol passes funding filter (negative rate)"""
        ticker = {
            'quote_volume_usdt': 150_000_000,
            'bid': 50000.0,
            'ask': 50010.0,
        }

        eligibility = universe_selector._evaluate_symbol(
            symbol="BTC/USDT",
            ticker=ticker,
            funding_rate=-0.0010,  # Within 0.0015 limit (abs)
        )

        assert eligibility.pass_funding

    def test_funding_fail_positive(self, universe_selector):
        """Test symbol fails funding filter (too high positive)"""
        ticker = {
            'quote_volume_usdt': 150_000_000,
            'bid': 50000.0,
            'ask': 50010.0,
        }

        eligibility = universe_selector._evaluate_symbol(
            symbol="BTC/USDT",
            ticker=ticker,
            funding_rate=0.0020,  # > 0.0015 limit
        )

        assert not eligibility.pass_funding
        assert any("Funding rate too high" in r for r in eligibility.reasons)

    def test_funding_fail_negative(self, universe_selector):
        """Test symbol fails funding filter (too high negative)"""
        ticker = {
            'quote_volume_usdt': 150_000_000,
            'bid': 50000.0,
            'ask': 50010.0,
        }

        eligibility = universe_selector._evaluate_symbol(
            symbol="BTC/USDT",
            ticker=ticker,
            funding_rate=-0.0020,  # abs > 0.0015 limit
        )

        assert not eligibility.pass_funding


class TestATRRatioFilter:
    """Test ATR ratio filtering"""

    def test_atr_ratio_pass(self, universe_selector, candle_store):
        """Test symbol passes ATR ratio filter"""
        # Create mock candles with sufficient volatility
        symbol = "BTC/USDT"
        base_price = 50000.0

        # Generate candles with ~1% ATR
        candles = []
        for i in range(20):
            candle = Candle(
                timestamp=i * 300000,  # 5m intervals
                open=base_price,
                high=base_price + 500,  # ~1% range
                low=base_price - 500,
                close=base_price + 100,
                volume=1000,
            )
            candles.append(candle)
            candle_store.add_candle(symbol, "5m", candle)

        ticker = {
            'quote_volume_usdt': 150_000_000,
            'bid': base_price,
            'ask': base_price + 10,
        }

        eligibility = universe_selector._evaluate_symbol(
            symbol=symbol,
            ticker=ticker,
            funding_rate=0.0001,
        )

        # ATR ratio should be > 0.005 (0.5%)
        assert eligibility.pass_atr_ratio

    def test_atr_ratio_fail_insufficient_data(self, universe_selector):
        """Test symbol fails ATR ratio when insufficient candles"""
        ticker = {
            'quote_volume_usdt': 150_000_000,
            'bid': 50000.0,
            'ask': 50010.0,
        }

        # No candles in store
        eligibility = universe_selector._evaluate_symbol(
            symbol="BTC/USDT",
            ticker=ticker,
            funding_rate=0.0001,
        )

        assert not eligibility.pass_atr_ratio
        assert any("ATR calculation failed" in r for r in eligibility.reasons)


class TestLiquidityScoring:
    """Test liquidity/quality scoring"""

    def test_score_deterministic(self, universe_selector, candle_store):
        """Test that score is deterministic for same inputs"""
        # Setup candles for ATR
        symbol = "BTC/USDT"
        for i in range(20):
            candle = Candle(
                timestamp=i * 300000,
                open=50000.0,
                high=50500.0,
                low=49500.0,
                close=50000.0,
                volume=1000,
            )
            candle_store.add_candle(symbol, "5m", candle)

        ticker = {
            'quote_volume_usdt': 200_000_000,
            'bid': 50000.0,
            'ask': 50020.0,
        }

        # Compute score multiple times
        scores = []
        for _ in range(3):
            eligibility = universe_selector._evaluate_symbol(
                symbol=symbol,
                ticker=ticker,
                funding_rate=0.0001,
            )
            scores.append(eligibility.score)

        # All scores should be identical
        assert len(set(scores)) == 1
        assert scores[0] > 0

    def test_higher_volume_higher_score(self, universe_selector, candle_store):
        """Test that higher volume gives higher score"""
        # Setup candles
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            for i in range(20):
                candle = Candle(
                    timestamp=i * 300000,
                    open=50000.0,
                    high=50500.0,
                    low=49500.0,
                    close=50000.0,
                    volume=1000,
                )
                candle_store.add_candle(symbol, "5m", candle)

        # BTC has higher volume
        ticker_btc = {
            'quote_volume_usdt': 300_000_000,
            'bid': 50000.0,
            'ask': 50020.0,
        }

        # ETH has lower volume
        ticker_eth = {
            'quote_volume_usdt': 150_000_000,
            'bid': 3000.0,
            'ask': 3001.0,  # Similar spread ratio
        }

        eligibility_btc = universe_selector._evaluate_symbol(
            symbol="BTC/USDT",
            ticker=ticker_btc,
            funding_rate=0.0001,
        )

        eligibility_eth = universe_selector._evaluate_symbol(
            symbol="ETH/USDT",
            ticker=ticker_eth,
            funding_rate=0.0001,
        )

        # BTC should have higher score due to higher volume
        assert eligibility_btc.score > eligibility_eth.score

    def test_lower_spread_higher_score(self, universe_selector, candle_store):
        """Test that lower spread gives higher score"""
        # Setup candles
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            for i in range(20):
                candle = Candle(
                    timestamp=i * 300000,
                    open=50000.0,
                    high=50500.0,
                    low=49500.0,
                    close=50000.0,
                    volume=1000,
                )
                candle_store.add_candle(symbol, "5m", candle)

        # BTC has tighter spread
        ticker_btc = {
            'quote_volume_usdt': 200_000_000,
            'bid': 50000.0,
            'ask': 50010.0,  # 0.02% spread
        }

        # ETH has wider spread
        ticker_eth = {
            'quote_volume_usdt': 200_000_000,
            'bid': 50000.0,
            'ask': 50025.0,  # 0.05% spread
        }

        eligibility_btc = universe_selector._evaluate_symbol(
            symbol="BTC/USDT",
            ticker=ticker_btc,
            funding_rate=0.0001,
        )

        eligibility_eth = universe_selector._evaluate_symbol(
            symbol="ETH/USDT",
            ticker=ticker_eth,
            funding_rate=0.0001,
        )

        # BTC should have higher score due to tighter spread
        assert eligibility_btc.score > eligibility_eth.score


class TestRanking:
    """Test ranking and top N selection"""

    def test_returns_top_n_symbols(self, mock_exchange_client, candle_store, universe_config):
        """Test that build_daily_universe returns top N symbols"""
        # Mock exchange responses
        all_symbols = [f"SYM{i}/USDT" for i in range(10)]
        mock_exchange_client.list_usdtm_perp_symbols.return_value = all_symbols

        # Mock tickers with varying volumes
        tickers = {
            symbol: {
                'quote_volume_usdt': 100_000_000 + i * 10_000_000,  # Increasing volume
                'bid': 100.0,
                'ask': 100.02,
            }
            for i, symbol in enumerate(all_symbols)
        }
        mock_exchange_client.fetch_24h_tickers.return_value = tickers

        # Mock funding rates (all good)
        funding_rates = {symbol: 0.0001 for symbol in all_symbols}
        mock_exchange_client.fetch_funding_rates.return_value = funding_rates

        # Mock klines for ATR
        mock_klines = [
            {
                'timestamp': i * 300000,
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.0,
                'volume': 1000.0,
            }
            for i in range(20)
        ]
        mock_exchange_client.fetch_klines.return_value = mock_klines

        # Set max_monitored_symbols to 6
        universe_config.max_monitored_symbols = 6

        selector = UniverseSelector(mock_exchange_client, candle_store, universe_config)

        result = selector.build_daily_universe(datetime.utcnow())

        # Should return exactly 6 symbols
        assert len(result) == 6

        # Should be the top 6 by volume (SYM9, SYM8, ..., SYM4)
        # (highest volume symbols should be selected)
        expected_top = [f"SYM{i}/USDT" for i in range(9, 3, -1)]
        assert set(result) == set(expected_top)

    def test_ranking_is_deterministic(self, mock_exchange_client, candle_store, universe_config):
        """Test that ranking is deterministic"""
        all_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        mock_exchange_client.list_usdtm_perp_symbols.return_value = all_symbols

        tickers = {
            "BTC/USDT": {'quote_volume_usdt': 300_000_000, 'bid': 50000.0, 'ask': 50010.0},
            "ETH/USDT": {'quote_volume_usdt': 200_000_000, 'bid': 3000.0, 'ask': 3001.0},
            "SOL/USDT": {'quote_volume_usdt': 150_000_000, 'bid': 100.0, 'ask': 100.05},
        }
        mock_exchange_client.fetch_24h_tickers.return_value = tickers

        funding_rates = {symbol: 0.0001 for symbol in all_symbols}
        mock_exchange_client.fetch_funding_rates.return_value = funding_rates

        mock_klines = [
            {
                'timestamp': i * 300000,
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.0,
                'volume': 1000.0,
            }
            for i in range(20)
        ]
        mock_exchange_client.fetch_klines.return_value = mock_klines

        selector = UniverseSelector(mock_exchange_client, candle_store, universe_config)

        # Build universe multiple times
        results = [
            selector.build_daily_universe(datetime.utcnow())
            for _ in range(3)
        ]

        # All results should be identical
        assert results[0] == results[1] == results[2]

    def test_no_eligible_symbols_returns_empty(self, mock_exchange_client, candle_store, universe_config):
        """Test that empty list is returned when no symbols pass filters"""
        all_symbols = ["BTC/USDT", "ETH/USDT"]
        mock_exchange_client.list_usdtm_perp_symbols.return_value = all_symbols

        # All tickers have too low volume
        tickers = {
            "BTC/USDT": {'quote_volume_usdt': 50_000_000, 'bid': 50000.0, 'ask': 50010.0},
            "ETH/USDT": {'quote_volume_usdt': 30_000_000, 'bid': 3000.0, 'ask': 3001.0},
        }
        mock_exchange_client.fetch_24h_tickers.return_value = tickers

        funding_rates = {symbol: 0.0001 for symbol in all_symbols}
        mock_exchange_client.fetch_funding_rates.return_value = funding_rates

        mock_klines = []
        mock_exchange_client.fetch_klines.return_value = mock_klines

        selector = UniverseSelector(mock_exchange_client, candle_store, universe_config)

        result = selector.build_daily_universe(datetime.utcnow())

        assert result == []


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_no_ticker_data(self, universe_selector):
        """Test evaluation when ticker is None"""
        eligibility = universe_selector._evaluate_symbol(
            symbol="BTC/USDT",
            ticker=None,
            funding_rate=0.0001,
        )

        assert not eligibility.is_eligible
        assert "No ticker data" in eligibility.reasons
        assert eligibility.score == 0.0

    def test_zero_bid_price(self, universe_selector):
        """Test evaluation when bid is zero"""
        ticker = {
            'quote_volume_usdt': 150_000_000,
            'bid': 0.0,
            'ask': 50010.0,
        }

        eligibility = universe_selector._evaluate_symbol(
            symbol="BTC/USDT",
            ticker=ticker,
            funding_rate=0.0001,
        )

        # Should fail spread check
        assert not eligibility.pass_spread

    def test_empty_candidate_list_after_filters(self, mock_exchange_client, candle_store, universe_config):
        """Test that empty list is returned when whitelist excludes all symbols"""
        universe_config.whitelist = ["NONEXISTENT/USDT"]

        all_symbols = ["BTC/USDT", "ETH/USDT"]
        mock_exchange_client.list_usdtm_perp_symbols.return_value = all_symbols

        selector = UniverseSelector(mock_exchange_client, candle_store, universe_config)

        result = selector.build_daily_universe(datetime.utcnow())

        assert result == []
