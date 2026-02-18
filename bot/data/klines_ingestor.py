"""Klines ingestor for fetching and storing candle data (REST-based)"""

import logging
from typing import List

from bot.exchange.binance_client import BinanceFuturesClient
from bot.data.candle_store import CandleStore
from bot.core.types import Candle


logger = logging.getLogger("trading_bot.data")


class KlinesIngestor:
    """
    Fetches klines from exchange and stores them in CandleStore.

    REST-based implementation that's WebSocket-ready:
    - Logic is separated so WebSocket can directly call candle_store.add_candle()
    - Normalization from exchange format to Candle dataclass is isolated
    """

    def __init__(
        self,
        exchange_client: BinanceFuturesClient,
        candle_store: CandleStore
    ):
        """
        Initialize KlinesIngestor.

        Args:
            exchange_client: Binance client
            candle_store: CandleStore instance
        """
        self.client = exchange_client
        self.candle_store = candle_store

        logger.info("KlinesIngestor initialized")

    def warmup(
        self,
        symbols: List[str],
        timeframe_5m_limit: int = 300,
        timeframe_1h_limit: int = 200,
        timeframe_4h_limit: int = 100,
    ) -> None:
        """
        Fetch historical klines to warm up the candle store.

        Fetches enough history for technical indicators:
        - 5m: default 300 candles (25 hours)
        - 1h: default 200 candles (~8 days)
        - 4h: default 100 candles (~16 days) — for multi-TF confirmation

        Args:
            symbols: List of symbols to warm up
            timeframe_5m_limit: Number of 5m candles to fetch
            timeframe_1h_limit: Number of 1h candles to fetch
            timeframe_4h_limit: Number of 4h candles to fetch
        """
        logger.info(f"Starting warmup for {len(symbols)} symbols...")

        for symbol in symbols:
            try:
                # Fetch 5m candles
                logger.debug(f"Fetching {timeframe_5m_limit} 5m candles for {symbol}")
                klines_5m = self.client.fetch_klines(
                    symbol=symbol,
                    timeframe="5m",
                    limit=timeframe_5m_limit
                )

                # Normalize and store
                for kline_dict in klines_5m:
                    candle = self._normalize_kline(kline_dict)
                    self.candle_store.add_candle(symbol, "5m", candle)

                logger.info(f"Loaded {len(klines_5m)} 5m candles for {symbol}")

                # Fetch 1h candles
                logger.debug(f"Fetching {timeframe_1h_limit} 1h candles for {symbol}")
                klines_1h = self.client.fetch_klines(
                    symbol=symbol,
                    timeframe="1h",
                    limit=timeframe_1h_limit
                )

                # Normalize and store
                for kline_dict in klines_1h:
                    candle = self._normalize_kline(kline_dict)
                    self.candle_store.add_candle(symbol, "1h", candle)

                logger.info(f"Loaded {len(klines_1h)} 1h candles for {symbol}")

                # Fetch 4h candles for multi-TF confirmation
                logger.debug(f"Fetching {timeframe_4h_limit} 4h candles for {symbol}")
                klines_4h = self.client.fetch_klines(
                    symbol=symbol,
                    timeframe="4h",
                    limit=timeframe_4h_limit
                )

                for kline_dict in klines_4h:
                    candle = self._normalize_kline(kline_dict)
                    self.candle_store.add_candle(symbol, "4h", candle)

                logger.info(f"Loaded {len(klines_4h)} 4h candles for {symbol}")

            except Exception as e:
                logger.error(f"Failed to warm up {symbol}: {e}")
                raise

        logger.info(f"Warmup complete for {len(symbols)} symbols")

    def update(self, symbols: List[str]) -> None:
        """
        Fetch latest candles for each symbol/timeframe.

        This is meant to be called periodically (e.g., every 5 minutes)
        to keep the candle store up to date.

        Args:
            symbols: List of symbols to update
        """
        for symbol in symbols:
            try:
                # Fetch latest 5m candles (limit=2 to ensure we get the latest closed candle)
                klines_5m = self.client.fetch_klines(
                    symbol=symbol,
                    timeframe="5m",
                    limit=2
                )

                if klines_5m:
                    # Add latest candle
                    latest_5m = self._normalize_kline(klines_5m[-1])
                    self.candle_store.add_candle(symbol, "5m", latest_5m)
                    logger.debug(f"Updated 5m candle for {symbol}")

                # Fetch latest 1h candles
                klines_1h = self.client.fetch_klines(
                    symbol=symbol,
                    timeframe="1h",
                    limit=2
                )

                if klines_1h:
                    # Add latest candle
                    latest_1h = self._normalize_kline(klines_1h[-1])
                    self.candle_store.add_candle(symbol, "1h", latest_1h)
                    logger.debug(f"Updated 1h candle for {symbol}")

                # Fetch latest 4h candle — update every cycle so candle is fresh
                klines_4h = self.client.fetch_klines(
                    symbol=symbol,
                    timeframe="4h",
                    limit=2
                )

                if klines_4h:
                    latest_4h = self._normalize_kline(klines_4h[-1])
                    self.candle_store.add_candle(symbol, "4h", latest_4h)
                    logger.debug(f"Updated 4h candle for {symbol}")

            except Exception as e:
                logger.error(f"Failed to update {symbol}: {e}")
                # Don't raise - continue updating other symbols

    @staticmethod
    def _normalize_kline(kline_dict: dict) -> Candle:
        """
        Normalize exchange kline dict to Candle dataclass.

        Args:
            kline_dict: Kline dict from BinanceFuturesClient.fetch_klines()
                Expected keys: timestamp, open, high, low, close, volume

        Returns:
            Candle object
        """
        return Candle(
            timestamp=int(kline_dict['timestamp']),
            open=float(kline_dict['open']),
            high=float(kline_dict['high']),
            low=float(kline_dict['low']),
            close=float(kline_dict['close']),
            volume=float(kline_dict['volume']),
        )
