"""
KlineStream — WebSocket-aware kline feed with REST fallback.

Wraps WSManager + KlinesIngestor to provide a unified interface:
- Normal mode: WebSocket delivers closed candles in real-time
- Fallback mode: REST polling every 5 minutes (original behaviour)

The Scheduler's on_candle_close callback is triggered either by:
- WSManager receiving a closed 5m candle (WebSocket mode)
- The Scheduler's own 5-minute timer (REST fallback mode)

Usage (in BotRunner)::

    stream = KlineStream(
        client=self._client,
        candle_store=self._candle_store,
        symbols=self._active_symbols,
        testnet=self.config.exchange.testnet,
        on_ws_candle_close=self._on_ws_candle_close,  # optional override
    )
    stream.start()
    # ...
    stream.stop()
    is_ws_active = stream.is_ws_active
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, List, Optional

from bot.core.types import Candle
from bot.data.candle_store import CandleStore
from bot.data.klines_ingestor import KlinesIngestor
from bot.data.ws_manager import WSManager
from bot.exchange.binance_client import BinanceFuturesClient

logger = logging.getLogger("trading_bot.data.kline_stream")


class KlineStream:
    """
    Unified kline feed: WebSocket-primary, REST-fallback.

    In WebSocket mode the 5m closed candle arrives within ~100ms of close.
    In REST fallback mode the Scheduler timer polls every 5 minutes as before.

    Parameters
    ----------
    client : BinanceFuturesClient
        Exchange client for REST fallback.
    candle_store : CandleStore
        Shared candle store.
    symbols : list[str]
        Active symbols to subscribe.
    testnet : bool
        Use Binance testnet streams.
    on_ws_candle_close : callable, optional
        Called with (symbol, timeframe, Candle) when a 5m candle closes via WS.
        Allows BotRunner to trigger pipeline earlier than the 5m REST poll.
    """

    def __init__(
        self,
        client: BinanceFuturesClient,
        candle_store: CandleStore,
        symbols: List[str],
        testnet: bool = False,
        on_ws_candle_close: Optional[Callable[[str, str, Candle], None]] = None,
    ) -> None:
        self._client = client
        self._candle_store = candle_store
        self._symbols = symbols
        self._testnet = testnet
        self._on_ws_candle_close_cb = on_ws_candle_close

        # REST ingestor (always available as fallback)
        self._ingestor = KlinesIngestor(client, candle_store)

        # WebSocket manager (optional, starts lazily)
        self._ws: Optional[WSManager] = None
        self._ws_enabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start WebSocket stream. Falls back to REST if WS fails."""
        logger.info(
            f"KlineStream: starting for {len(self._symbols)} symbols "
            f"(testnet={self._testnet})"
        )
        self._ws_enabled = True
        self._ws = WSManager(
            candle_store=self._candle_store,
            symbols=self._symbols,
            testnet=self._testnet,
            on_candle_close=self._on_ws_candle,
            on_fallback=self._on_ws_fallback,
        )
        self._ws.start()

    def stop(self) -> None:
        """Stop WebSocket stream."""
        self._ws_enabled = False
        if self._ws:
            self._ws.stop()
            self._ws = None
        logger.info("KlineStream: stopped")

    def warmup(self, symbols: Optional[List[str]] = None) -> None:
        """Warm up candle history via REST (always REST, WS has no history)."""
        target = symbols or self._symbols
        self._ingestor.warmup(target)

    def update(self, symbols: Optional[List[str]] = None) -> None:
        """
        Fetch latest candles via REST.

        Called by Scheduler in REST fallback mode, or as a safety net
        in WebSocket mode to fill any gaps (e.g. missed candles during reconnect).
        """
        target = symbols or self._symbols
        self._ingestor.update(target)

    def update_symbols(self, symbols: List[str]) -> None:
        """Update active symbols and reconnect WebSocket if active."""
        self._symbols = symbols
        if self._ws and self._ws_enabled:
            self._ws.update_symbols(symbols)

    @property
    def is_ws_active(self) -> bool:
        """True if WebSocket is connected and delivering candles."""
        return bool(self._ws and self._ws.is_running)

    @property
    def is_fallback_active(self) -> bool:
        """True if WebSocket has failed and REST fallback is in use."""
        return bool(self._ws and self._ws.is_fallback_active)

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _on_ws_candle(self, symbol: str, timeframe: str, candle: Candle) -> None:
        """Called by WSManager on every closed candle."""
        # Only fire the external callback for 5m candles
        # (1h and 4h candle closes don't trigger the decision pipeline)
        if timeframe == "5m" and self._on_ws_candle_close_cb:
            try:
                self._on_ws_candle_close_cb(symbol, timeframe, candle)
            except Exception as e:
                logger.error(f"KlineStream: ws candle callback error: {e}")

    def _on_ws_fallback(self) -> None:
        """Called by WSManager when it exhausts reconnect attempts."""
        logger.warning(
            "KlineStream: WebSocket failed — REST polling mode active. "
            "Scheduler 5m timer will drive the pipeline."
        )
        # Scheduler's REST polling continues unaffected;
        # no action needed here beyond logging.
