"""Thread-safe candle storage with rolling windows"""

import logging
from collections import deque, defaultdict
from threading import Lock
from typing import Optional, Dict, Deque

from bot.core.types import Candle


logger = logging.getLogger("trading_bot.data")


class CandleStore:
    """
    Thread-safe candle storage with per-symbol, per-timeframe rolling windows.

    Designed to be WebSocket-ready: concurrent writes from WebSocket can safely
    call add_candle() while other threads read.

    Rolling window limits:
    - 5m: 300 candles (25 hours)
    - 1h: 200 candles (~8 days)
    """

    # Default window sizes per timeframe
    DEFAULT_LIMITS = {
        "5m": 300,
        "1h": 200,
        "4h": 100,  # ~16 days
    }

    def __init__(self, custom_limits: Optional[Dict[str, int]] = None):
        """
        Initialize CandleStore.

        Args:
            custom_limits: Optional dict mapping timeframe -> max candles
        """
        self.limits = {**self.DEFAULT_LIMITS, **(custom_limits or {})}

        # Storage: {symbol: {timeframe: deque[Candle]}}
        # Bulgu 7: Use plain dict for inner layer to avoid wrong maxlen on first access.
        # add_candle() always creates the deque with the correct timeframe limit.
        self._store: Dict[str, Dict[str, Deque[Candle]]] = defaultdict(dict)

        # Thread safety
        self._lock = Lock()

        logger.info(f"CandleStore initialized with limits: {self.limits}")

    def add_candle(self, symbol: str, timeframe: str, candle: Candle) -> None:
        """
        Add a candle to the store (thread-safe).

        If the rolling window is full, oldest candle is automatically dropped.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Timeframe (e.g., "5m", "1h")
            candle: Candle object
        """
        with self._lock:
            # Ensure timeframe deque has correct maxlen
            if timeframe not in self._store[symbol]:
                maxlen = self.limits.get(timeframe, 300)
                self._store[symbol][timeframe] = deque(maxlen=maxlen)

            self._store[symbol][timeframe].append(candle)

            logger.debug(
                f"Added candle to {symbol} {timeframe}: "
                f"{len(self._store[symbol][timeframe])} total"
            )

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: Optional[int] = None
    ) -> list[Candle]:
        """
        Get candles for a symbol/timeframe (thread-safe).

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            limit: Max number of candles to return (most recent). None = all.

        Returns:
            List of candles (oldest to newest)
        """
        with self._lock:
            # Bulgu 7: Guard against missing timeframe — return empty list rather than
            # letting defaultdict auto-create a deque with the wrong maxlen.
            if timeframe not in self._store[symbol]:
                return []
            candles = list(self._store[symbol][timeframe])

            if limit is not None and limit < len(candles):
                return candles[-limit:]

            return candles

    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[Candle]:
        """
        Get most recent candle (thread-safe).

        Args:
            symbol: Trading pair
            timeframe: Timeframe

        Returns:
            Latest candle or None if no data
        """
        with self._lock:
            # Bulgu 7: Guard against missing timeframe
            if timeframe not in self._store[symbol]:
                return None
            candles = self._store[symbol][timeframe]
            return candles[-1] if candles else None

    def has_enough_data(
        self,
        symbol: str,
        timeframe: str,
        min_bars: int
    ) -> bool:
        """
        Check if we have sufficient candles for calculations.

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            min_bars: Minimum required candles

        Returns:
            True if we have at least min_bars candles
        """
        with self._lock:
            return len(self._store[symbol][timeframe]) >= min_bars

    def get_count(self, symbol: str, timeframe: str) -> int:
        """
        Get number of candles stored.

        Args:
            symbol: Trading pair
            timeframe: Timeframe

        Returns:
            Number of candles
        """
        with self._lock:
            if timeframe not in self._store[symbol]:
                return 0
            return len(self._store[symbol][timeframe])

    def clear(self, symbol: Optional[str] = None) -> None:
        """
        Clear stored candles (thread-safe).

        Args:
            symbol: If provided, clear only this symbol. Otherwise clear all.
        """
        with self._lock:
            if symbol:
                self._store[symbol].clear()
                logger.info(f"Cleared candles for {symbol}")
            else:
                self._store.clear()
                logger.info("Cleared all candles")
