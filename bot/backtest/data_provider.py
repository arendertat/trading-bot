"""
Historical data provider for backtesting.

Fetches OHLCV data from Binance API with pagination support,
then loads it into a CandleStore for the backtest engine.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from bot.core.types import Candle
from bot.data.candle_store import CandleStore

logger = logging.getLogger("trading_bot.backtest.data_provider")

# Binance max candles per request
_BINANCE_LIMIT = 1500

# Timeframe to milliseconds
_TF_MS = {
    "1m":   60_000,
    "5m":   300_000,
    "15m":  900_000,
    "1h":  3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class HistoricalDataProvider:
    """
    Fetches historical OHLCV data from Binance and populates a CandleStore.

    Uses pagination to fetch arbitrarily long histories, respecting
    Binance's 1500-candle-per-request limit.

    Args:
        exchange: ccxt exchange instance (from BinanceFuturesClient.exchange)
        candle_store: CandleStore to populate
        rate_limit_sleep: Seconds to sleep between paginated requests (avoid 429)
    """

    def __init__(
        self,
        exchange,
        candle_store: CandleStore,
        rate_limit_sleep: float = 0.3,
    ) -> None:
        self._exchange = exchange
        self._store = candle_store
        self._rate_limit_sleep = rate_limit_sleep

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> int:
        """
        Fetch historical candles for a symbol/timeframe into the CandleStore.

        Args:
            symbol: e.g. "BTCUSDT" or "BTC/USDT"
            timeframe: e.g. "5m", "1h", "4h"
            start: Start datetime (UTC)
            end: End datetime (UTC). Defaults to now.

        Returns:
            Number of candles loaded
        """
        if end is None:
            end = datetime.now(timezone.utc)

        # Normalise symbol for ccxt (needs slash)
        ccxt_symbol = _normalise_symbol(symbol)

        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        tf_ms = _TF_MS.get(timeframe)
        if tf_ms is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Use one of {list(_TF_MS)}")

        total = 0
        cursor = since_ms

        logger.info(
            f"Fetching {ccxt_symbol} {timeframe} from "
            f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} …"
        )

        while cursor < end_ms:
            try:
                raw = self._exchange.fetch_ohlcv(
                    ccxt_symbol,
                    timeframe=timeframe,
                    since=cursor,
                    limit=_BINANCE_LIMIT,
                )
            except Exception as e:
                logger.error(f"fetch_ohlcv failed for {ccxt_symbol} {timeframe}: {e}")
                break

            if not raw:
                break

            batch_loaded = 0
            for row in raw:
                ts, o, h, l, c, v = row
                # Skip candles beyond end
                if ts >= end_ms:
                    break
                candle = Candle(
                    timestamp=ts,
                    open=float(o),
                    high=float(h),
                    low=float(l),
                    close=float(c),
                    volume=float(v),
                )
                self._store.add_candle(symbol, timeframe, candle)
                batch_loaded += 1

            total += batch_loaded
            logger.debug(
                f"{ccxt_symbol} {timeframe}: fetched {batch_loaded} candles "
                f"(total={total}, cursor={_ts_to_str(raw[-1][0])})"
            )

            # Move cursor to next batch
            last_ts = raw[-1][0]
            cursor = last_ts + tf_ms

            # Stop if we got fewer than requested (end of data)
            if len(raw) < _BINANCE_LIMIT:
                break

            # Rate limit
            if self._rate_limit_sleep > 0:
                time.sleep(self._rate_limit_sleep)

        logger.info(f"{ccxt_symbol} {timeframe}: {total} candles loaded into store")
        return total

    def fetch_multi(
        self,
        symbols: List[str],
        timeframes: List[str],
        start: datetime,
        end: Optional[datetime] = None,
    ) -> Dict[Tuple[str, str], int]:
        """
        Fetch multiple symbol/timeframe combinations.

        Returns:
            Dict mapping (symbol, timeframe) -> candle count
        """
        results: Dict[Tuple[str, str], int] = {}
        for symbol in symbols:
            for tf in timeframes:
                count = self.fetch(symbol, tf, start, end)
                results[(symbol, tf)] = count
        return results


def _normalise_symbol(symbol: str) -> str:
    """
    Convert BTCUSDT → BTCUSDT:USDT for ccxt binanceusdm (perpetual futures).

    ccxt binanceusdm uses the format "BASE/QUOTE:SETTLE", e.g.:
      BTCUSDT  →  BTC/USDT:USDT
      ETHUSDT  →  ETH/USDT:USDT
    """
    if ":" in symbol:
        return symbol  # already in futures format
    if "/" in symbol and ":" not in symbol:
        # e.g. BTC/USDT → BTC/USDT:USDT
        return symbol + ":USDT"
    # Raw e.g. BTCUSDT → BTC/USDT:USDT
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}/USDT:USDT"
    return symbol


def _ts_to_str(ts_ms: int) -> str:
    """Convert Unix ms timestamp to readable string."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
