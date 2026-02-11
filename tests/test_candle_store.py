"""Unit tests for CandleStore"""

import pytest
import threading
import time

from bot.data.candle_store import CandleStore
from bot.core.types import Candle


class TestCandleStore:
    """Test CandleStore functionality"""

    def test_add_and_get_candles(self):
        """Test adding and retrieving candles"""
        store = CandleStore()

        # Add some candles
        candles = [
            Candle(timestamp=1000, open=100, high=102, low=99, close=101, volume=1000),
            Candle(timestamp=2000, open=101, high=103, low=100, close=102, volume=1100),
            Candle(timestamp=3000, open=102, high=104, low=101, close=103, volume=1200),
        ]

        for candle in candles:
            store.add_candle("BTC/USDT", "5m", candle)

        # Retrieve candles
        retrieved = store.get_candles("BTC/USDT", "5m")

        assert len(retrieved) == 3
        assert retrieved[0].timestamp == 1000
        assert retrieved[-1].timestamp == 3000

    def test_rolling_window_truncation(self):
        """Test that rolling window truncates old candles"""
        # Set small window for testing
        store = CandleStore(custom_limits={"5m": 3})

        # Add more candles than limit
        for i in range(5):
            candle = Candle(
                timestamp=i * 1000,
                open=100 + i,
                high=102 + i,
                low=99 + i,
                close=101 + i,
                volume=1000 + i
            )
            store.add_candle("BTC/USDT", "5m", candle)

        # Should only have last 3 candles
        retrieved = store.get_candles("BTC/USDT", "5m")

        assert len(retrieved) == 3
        assert retrieved[0].timestamp == 2000  # Oldest kept
        assert retrieved[-1].timestamp == 4000  # Newest

    def test_get_latest_candle(self):
        """Test getting most recent candle"""
        store = CandleStore()

        # Add candles
        candles = [
            Candle(timestamp=1000, open=100, high=102, low=99, close=101, volume=1000),
            Candle(timestamp=2000, open=101, high=103, low=100, close=102, volume=1100),
            Candle(timestamp=3000, open=102, high=104, low=101, close=103, volume=1200),
        ]

        for candle in candles:
            store.add_candle("BTC/USDT", "5m", candle)

        latest = store.get_latest_candle("BTC/USDT", "5m")

        assert latest is not None
        assert latest.timestamp == 3000
        assert latest.close == 103

    def test_get_latest_candle_empty(self):
        """Test getting latest candle when no data"""
        store = CandleStore()

        latest = store.get_latest_candle("BTC/USDT", "5m")

        assert latest is None

    def test_has_enough_data(self):
        """Test checking for sufficient data"""
        store = CandleStore()

        # Add 10 candles
        for i in range(10):
            candle = Candle(
                timestamp=i * 1000,
                open=100,
                high=102,
                low=99,
                close=101,
                volume=1000
            )
            store.add_candle("BTC/USDT", "5m", candle)

        # Should have enough for 10 bars
        assert store.has_enough_data("BTC/USDT", "5m", 10)

        # Should not have enough for 11 bars
        assert not store.has_enough_data("BTC/USDT", "5m", 11)

    def test_get_candles_with_limit(self):
        """Test getting limited number of candles"""
        store = CandleStore()

        # Add 10 candles
        for i in range(10):
            candle = Candle(
                timestamp=i * 1000,
                open=100 + i,
                high=102 + i,
                low=99 + i,
                close=101 + i,
                volume=1000 + i
            )
            store.add_candle("BTC/USDT", "5m", candle)

        # Get only last 5
        retrieved = store.get_candles("BTC/USDT", "5m", limit=5)

        assert len(retrieved) == 5
        assert retrieved[0].timestamp == 5000  # 6th candle (0-indexed: 5)
        assert retrieved[-1].timestamp == 9000  # 10th candle

    def test_multiple_symbols(self):
        """Test storing candles for multiple symbols"""
        store = CandleStore()

        # Add candles for BTC
        btc_candle = Candle(timestamp=1000, open=50000, high=51000, low=49000, close=50500, volume=100)
        store.add_candle("BTC/USDT", "5m", btc_candle)

        # Add candles for ETH
        eth_candle = Candle(timestamp=1000, open=3000, high=3100, low=2900, close=3050, volume=500)
        store.add_candle("ETH/USDT", "5m", eth_candle)

        # Retrieve separately
        btc_candles = store.get_candles("BTC/USDT", "5m")
        eth_candles = store.get_candles("ETH/USDT", "5m")

        assert len(btc_candles) == 1
        assert len(eth_candles) == 1
        assert btc_candles[0].close == 50500
        assert eth_candles[0].close == 3050

    def test_multiple_timeframes(self):
        """Test storing candles for multiple timeframes"""
        store = CandleStore()

        # Add 5m candle
        candle_5m = Candle(timestamp=1000, open=100, high=102, low=99, close=101, volume=1000)
        store.add_candle("BTC/USDT", "5m", candle_5m)

        # Add 1h candle
        candle_1h = Candle(timestamp=1000, open=100, high=105, low=95, close=103, volume=10000)
        store.add_candle("BTC/USDT", "1h", candle_1h)

        # Retrieve separately
        candles_5m = store.get_candles("BTC/USDT", "5m")
        candles_1h = store.get_candles("BTC/USDT", "1h")

        assert len(candles_5m) == 1
        assert len(candles_1h) == 1
        assert candles_5m[0].high == 102
        assert candles_1h[0].high == 105

    def test_get_count(self):
        """Test getting candle count"""
        store = CandleStore()

        # Add 5 candles
        for i in range(5):
            candle = Candle(
                timestamp=i * 1000,
                open=100,
                high=102,
                low=99,
                close=101,
                volume=1000
            )
            store.add_candle("BTC/USDT", "5m", candle)

        count = store.get_count("BTC/USDT", "5m")

        assert count == 5

    def test_clear_specific_symbol(self):
        """Test clearing candles for specific symbol"""
        store = CandleStore()

        # Add candles for two symbols
        btc_candle = Candle(timestamp=1000, open=50000, high=51000, low=49000, close=50500, volume=100)
        eth_candle = Candle(timestamp=1000, open=3000, high=3100, low=2900, close=3050, volume=500)

        store.add_candle("BTC/USDT", "5m", btc_candle)
        store.add_candle("ETH/USDT", "5m", eth_candle)

        # Clear BTC only
        store.clear(symbol="BTC/USDT")

        # BTC should be empty, ETH should remain
        assert store.get_count("BTC/USDT", "5m") == 0
        assert store.get_count("ETH/USDT", "5m") == 1

    def test_clear_all(self):
        """Test clearing all candles"""
        store = CandleStore()

        # Add candles for two symbols
        btc_candle = Candle(timestamp=1000, open=50000, high=51000, low=49000, close=50500, volume=100)
        eth_candle = Candle(timestamp=1000, open=3000, high=3100, low=2900, close=3050, volume=500)

        store.add_candle("BTC/USDT", "5m", btc_candle)
        store.add_candle("ETH/USDT", "5m", eth_candle)

        # Clear all
        store.clear()

        # Both should be empty
        assert store.get_count("BTC/USDT", "5m") == 0
        assert store.get_count("ETH/USDT", "5m") == 0

    def test_thread_safety_concurrent_writes(self):
        """Test thread safety with concurrent writes"""
        store = CandleStore()

        def add_candles(symbol: str, start_ts: int, count: int):
            """Helper to add candles in a thread"""
            for i in range(count):
                candle = Candle(
                    timestamp=start_ts + i * 1000,
                    open=100,
                    high=102,
                    low=99,
                    close=101,
                    volume=1000
                )
                store.add_candle(symbol, "5m", candle)
                time.sleep(0.001)  # Small delay to simulate concurrent access

        # Create threads
        thread1 = threading.Thread(target=add_candles, args=("BTC/USDT", 1000, 50))
        thread2 = threading.Thread(target=add_candles, args=("ETH/USDT", 2000, 50))

        # Start threads
        thread1.start()
        thread2.start()

        # Wait for completion
        thread1.join()
        thread2.join()

        # Verify counts
        assert store.get_count("BTC/USDT", "5m") == 50
        assert store.get_count("ETH/USDT", "5m") == 50

    def test_thread_safety_read_while_write(self):
        """Test thread safety when reading while writing"""
        store = CandleStore()

        # Pre-populate with some candles
        for i in range(10):
            candle = Candle(
                timestamp=i * 1000,
                open=100,
                high=102,
                low=99,
                close=101,
                volume=1000
            )
            store.add_candle("BTC/USDT", "5m", candle)

        results = []

        def writer():
            """Add more candles"""
            for i in range(10, 20):
                candle = Candle(
                    timestamp=i * 1000,
                    open=100,
                    high=102,
                    low=99,
                    close=101,
                    volume=1000
                )
                store.add_candle("BTC/USDT", "5m", candle)
                time.sleep(0.001)

        def reader():
            """Read candles repeatedly"""
            for _ in range(10):
                candles = store.get_candles("BTC/USDT", "5m")
                results.append(len(candles))
                time.sleep(0.002)

        # Create and start threads
        write_thread = threading.Thread(target=writer)
        read_thread = threading.Thread(target=reader)

        write_thread.start()
        read_thread.start()

        # Wait for completion
        write_thread.join()
        read_thread.join()

        # Verify final count
        assert store.get_count("BTC/USDT", "5m") == 20

        # All reads should have returned valid lengths (between 10 and 20)
        assert all(10 <= count <= 20 for count in results)
