"""
WebSocket connection manager for Binance USDT-M Futures kline streams.

Manages persistent WebSocket connections with automatic reconnection,
heartbeat monitoring, and REST fallback on failure.

Architecture:
    WSManager
        └── runs in a background daemon thread (asyncio event loop)
        └── on candle close event → calls on_candle_close_cb(symbol, timeframe, candle)
        └── on disconnect → reconnects after backoff delay
        └── on repeated failures → triggers REST fallback mode

Usage::

    from bot.data.ws_manager import WSManager
    from bot.data.candle_store import CandleStore

    store = CandleStore()
    ws = WSManager(candle_store=store, symbols=["BTC/USDT", "ETH/USDT"])
    ws.start()       # non-blocking, spawns daemon thread
    # ...
    ws.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Set

from bot.core.types import Candle
from bot.data.candle_store import CandleStore

logger = logging.getLogger("trading_bot.data.ws")

# Binance USDT-M Futures WebSocket base URL
_WS_BASE = "wss://fstream.binance.com/stream"
_WS_TESTNET_BASE = "wss://stream.binancefuture.com/stream"

# Reconnect settings
_RECONNECT_DELAY_SECONDS = 5
_MAX_RECONNECT_DELAY_SECONDS = 60
_HEARTBEAT_TIMEOUT_SECONDS = 30

# Timeframes to subscribe (must match CandleStore DEFAULT_LIMITS keys)
_SUBSCRIBED_TIMEFRAMES = ["5m", "1h", "4h"]

# Binance kline interval strings
_TF_TO_INTERVAL: Dict[str, str] = {
    "5m": "5m",
    "1h": "1h",
    "4h": "4h",
}


def _symbol_to_stream(symbol: str, timeframe: str) -> str:
    """Convert symbol + timeframe to Binance stream name.

    e.g. "BTC/USDT" + "5m" → "btcusdt@kline_5m"
    """
    base = symbol.replace("/", "").lower()
    interval = _TF_TO_INTERVAL[timeframe]
    return f"{base}@kline_{interval}"


def _parse_kline_message(msg: dict) -> Optional[tuple[str, str, Candle]]:
    """
    Parse a Binance combined stream kline message.

    Returns (symbol, timeframe, Candle) if the candle is closed, else None.

    Binance combined stream message format::
        {
          "stream": "btcusdt@kline_5m",
          "data": {
            "e": "kline",
            "k": {
              "t": 1234567890000,  # open time ms
              "o": "67000.0",
              "h": "67500.0",
              "l": "66800.0",
              "c": "67200.0",
              "v": "100.5",
              "x": true            # is candle closed?
            }
          }
        }
    """
    try:
        stream = msg.get("stream", "")
        data = msg.get("data", {})
        k = data.get("k", {})

        if not k or not k.get("x"):  # x=true means candle is closed
            return None

        # Parse timeframe from stream name e.g. "btcusdt@kline_5m" → "5m"
        if "@kline_" not in stream:
            return None
        interval = stream.split("@kline_")[1]

        # Map interval back to our timeframe key
        interval_to_tf = {v: k for k, v in _TF_TO_INTERVAL.items()}
        timeframe = interval_to_tf.get(interval)
        if timeframe is None:
            return None

        # Reconstruct symbol from stream prefix e.g. "btcusdt" → lookup needed
        # We store symbol mapping in WSManager._stream_to_symbol
        # Return stream name so caller can map it
        candle = Candle(
            timestamp=int(k["t"]),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
        )
        return stream, timeframe, candle

    except (KeyError, ValueError, TypeError) as e:
        logger.debug(f"Failed to parse kline message: {e}")
        return None


class WSManager:
    """
    Manages Binance USDT-M Futures kline WebSocket connections.

    Runs a background asyncio event loop in a daemon thread.
    On candle close: adds candle to CandleStore and calls optional callback.
    On disconnect: reconnects with exponential backoff.
    After max_failures consecutive reconnect failures: triggers REST fallback.

    Parameters
    ----------
    candle_store : CandleStore
        Shared candle store to populate with received candles.
    symbols : list[str]
        Symbols to subscribe (e.g. ["BTC/USDT", "ETH/USDT"]).
    testnet : bool
        Use Binance testnet WebSocket URL.
    on_candle_close : callable, optional
        Called with (symbol, timeframe, Candle) on each closed candle.
    on_fallback : callable, optional
        Called with no arguments when WebSocket falls back to REST mode.
    max_failures : int
        Consecutive reconnect failures before triggering REST fallback.
    """

    def __init__(
        self,
        candle_store: CandleStore,
        symbols: List[str],
        testnet: bool = False,
        on_candle_close: Optional[Callable[[str, str, Candle], None]] = None,
        on_fallback: Optional[Callable[[], None]] = None,
        max_failures: int = 5,
    ) -> None:
        self._candle_store = candle_store
        self._symbols = symbols
        self._testnet = testnet
        self._on_candle_close_cb = on_candle_close
        self._on_fallback_cb = on_fallback
        self._max_failures = max_failures

        self._running = False
        self._fallback_active = False
        self._consecutive_failures = 0
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None

        # Build stream name → symbol mapping for message parsing
        self._stream_to_symbol: Dict[str, str] = {}
        self._all_streams: List[str] = []
        for symbol in symbols:
            for tf in _SUBSCRIBED_TIMEFRAMES:
                stream = _symbol_to_stream(symbol, tf)
                self._stream_to_symbol[stream] = symbol
                self._all_streams.append(stream)

        base = _WS_TESTNET_BASE if testnet else _WS_BASE
        streams_param = "/".join(self._all_streams)
        self._ws_url = f"{base}?streams={streams_param}"

        logger.info(
            f"WSManager: {len(symbols)} symbols × {len(_SUBSCRIBED_TIMEFRAMES)} TFs "
            f"= {len(self._all_streams)} streams | testnet={testnet}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start WebSocket in a background daemon thread (non-blocking)."""
        if self._running:
            logger.warning("WSManager already running")
            return
        self._running = True
        self._fallback_active = False
        self._consecutive_failures = 0
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="ws-manager",
            daemon=True,
        )
        self._thread.start()
        logger.info("WSManager: background thread started")

    def stop(self) -> None:
        """Stop WebSocket and background thread gracefully."""
        logger.info("WSManager: stopping …")
        self._running = False
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("WSManager: stopped")

    @property
    def is_running(self) -> bool:
        """True if WebSocket stream is active (not in fallback mode)."""
        return self._running and not self._fallback_active

    @property
    def is_fallback_active(self) -> bool:
        """True if WebSocket failed and REST fallback is in use."""
        return self._fallback_active

    def update_symbols(self, symbols: List[str]) -> None:
        """
        Update the symbol list and reconnect.

        Called when universe changes (daily refresh).
        """
        logger.info(f"WSManager: updating symbols → {symbols}")
        self._symbols = symbols
        # Rebuild stream maps
        self._stream_to_symbol = {}
        self._all_streams = []
        for symbol in symbols:
            for tf in _SUBSCRIBED_TIMEFRAMES:
                stream = _symbol_to_stream(symbol, tf)
                self._stream_to_symbol[stream] = symbol
                self._all_streams.append(stream)
        base = _WS_TESTNET_BASE if self._testnet else _WS_BASE
        streams_param = "/".join(self._all_streams)
        self._ws_url = f"{base}?streams={streams_param}"

        # Trigger reconnect by stopping current connection
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)

    # ------------------------------------------------------------------
    # Internal: event loop thread
    # ------------------------------------------------------------------

    def _run_event_loop(self) -> None:
        """Entry point for background thread — runs asyncio event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._connect_loop())
        except Exception as e:
            logger.error(f"WSManager: event loop crashed: {e}")
        finally:
            loop.close()
            self._loop = None

    async def _connect_loop(self) -> None:
        """Outer reconnection loop — retries on disconnect with backoff."""
        delay = _RECONNECT_DELAY_SECONDS
        while self._running:
            self._stop_event = asyncio.Event()
            try:
                await self._stream_loop()
                # Clean stop requested
                if not self._running:
                    break
                # Unexpected disconnect — reset delay on success
                self._consecutive_failures = 0
                delay = _RECONNECT_DELAY_SECONDS
                logger.info(f"WSManager: reconnecting in {delay}s …")
                await asyncio.sleep(delay)

            except Exception as e:
                self._consecutive_failures += 1
                logger.warning(
                    f"WSManager: connection error #{self._consecutive_failures}: {e}"
                )
                if self._consecutive_failures >= self._max_failures:
                    logger.error(
                        f"WSManager: {self._consecutive_failures} consecutive failures — "
                        f"activating REST fallback"
                    )
                    self._fallback_active = True
                    if self._on_fallback_cb:
                        try:
                            self._on_fallback_cb()
                        except Exception as cb_err:
                            logger.error(f"WSManager: fallback callback error: {cb_err}")
                    break

                delay = min(delay * 2, _MAX_RECONNECT_DELAY_SECONDS)
                logger.info(f"WSManager: retrying in {delay}s …")
                await asyncio.sleep(delay)

    async def _stream_loop(self) -> None:
        """
        Connect and process messages until stop_event or disconnect.

        Uses websockets library for the actual WS connection.
        """
        try:
            import websockets  # local import — optional dependency
        except ImportError:
            raise RuntimeError(
                "websockets package not installed. "
                "Run: pip install 'websockets>=12.0'"
            )

        logger.info(f"WSManager: connecting to {self._ws_url[:80]}…")
        async with websockets.connect(
            self._ws_url,
            ping_interval=20,
            ping_timeout=_HEARTBEAT_TIMEOUT_SECONDS,
            close_timeout=5,
        ) as ws:
            logger.info("WSManager: connected ✓")
            self._consecutive_failures = 0

            recv_task = asyncio.ensure_future(ws.recv())
            stop_task = asyncio.ensure_future(self._stop_event.wait())

            while self._running:
                done, _ = await asyncio.wait(
                    [recv_task, stop_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if stop_task in done:
                    logger.info("WSManager: stop requested — closing connection")
                    await ws.close()
                    break

                if recv_task in done:
                    try:
                        raw = recv_task.result()
                        self._handle_message(raw)
                    except Exception as e:
                        logger.debug(f"WSManager: message handling error: {e}")
                    recv_task = asyncio.ensure_future(ws.recv())

            # Cancel pending tasks
            for task in [recv_task, stop_task]:
                if not task.done():
                    task.cancel()

    def _handle_message(self, raw: str) -> None:
        """Parse and process a raw WebSocket message string."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("WSManager: invalid JSON message")
            return

        result = _parse_kline_message(msg)
        if result is None:
            return  # Not a closed candle — ignore in-progress candles

        stream_name, timeframe, candle = result
        symbol = self._stream_to_symbol.get(stream_name)
        if symbol is None:
            logger.debug(f"WSManager: unknown stream {stream_name}")
            return

        # Add to candle store
        self._candle_store.add_candle(symbol, timeframe, candle)

        logger.debug(
            f"WS candle: {symbol} {timeframe} close={candle.close} "
            f"ts={candle.timestamp}"
        )

        # Fire callback (e.g. scheduler trigger)
        if self._on_candle_close_cb:
            try:
                self._on_candle_close_cb(symbol, timeframe, candle)
            except Exception as e:
                logger.error(f"WSManager: on_candle_close callback error: {e}")
