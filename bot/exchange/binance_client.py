"""Binance USDT-M Perpetual Futures REST client wrapper"""

import time
from typing import Optional, Dict, List, Any
import logging

import ccxt

from bot.config.models import ExchangeConfig
from bot.exchange.health_monitor import HealthMonitor
from bot.exchange.exceptions import (
    ExchangeError,
    RateLimitError,
    TimestampError,
    AuthError,
    OrderNotFoundError,
    InsufficientBalanceError,
)


logger = logging.getLogger("trading_bot.exchange")


class BinanceFuturesClient:
    """
    Production-grade Binance USDT-M Perpetual Futures REST client.

    Features:
    - Time synchronization with server
    - Exponential backoff retry for transient errors
    - Idempotent order placement (via clientOrderId checking)
    - Custom exception hierarchy
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        recv_window_ms: int = 5000,
        max_retries: int = 3,
        health_error_threshold: int = 5,
    ):
        """
        Initialize Binance Futures client.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet if True
            recv_window_ms: recvWindow parameter for signed requests
            max_retries: Maximum number of retries for transient errors
            health_error_threshold: Number of consecutive errors before marking unhealthy
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.recv_window_ms = recv_window_ms
        self.max_retries = max_retries

        # Initialize health monitor (Fix #4)
        self.health_monitor = HealthMonitor(error_threshold=health_error_threshold)

        # Initialize ccxt client
        TESTNET_FAPI = 'https://demo-fapi.binance.com'
        exchange_config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'recvWindow': recv_window_ms},
        }
        if testnet:
            # ccxt sandbox mode deprecated for Binance Futures;
            # override URLs directly to testnet.binancefuture.com
            exchange_config['urls'] = {
                'api': {
                    'fapiPublic': f'{TESTNET_FAPI}/fapi/v1',
                    'fapiPublicV2': f'{TESTNET_FAPI}/fapi/v2',
                    'fapiPublicV3': f'{TESTNET_FAPI}/fapi/v3',
                    'fapiPrivate': f'{TESTNET_FAPI}/fapi/v1',
                    'fapiPrivateV2': f'{TESTNET_FAPI}/fapi/v2',
                    'fapiPrivateV3': f'{TESTNET_FAPI}/fapi/v3',
                }
            }
        self.exchange = ccxt.binanceusdm(exchange_config)

        # Initialize time synchronization
        try:
            self.exchange.load_time_difference()
            logger.debug("Initial time synchronization completed")
        except Exception as e:
            logger.warning(f"Initial time sync failed (will retry on first error): {e}")

        logger.info(f"Binance Futures client initialized (testnet={testnet})")

    @classmethod
    def from_config(cls, config: ExchangeConfig) -> "BinanceFuturesClient":
        """Create client from config"""
        import os
        api_key = os.getenv(config.api_key_env)
        api_secret = os.getenv(config.api_secret_env)

        if not api_key or not api_secret:
            raise ValueError(
                f"API credentials not found in environment: "
                f"{config.api_key_env}, {config.api_secret_env}"
            )

        testnet = getattr(config, 'testnet', False)
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            recv_window_ms=config.recv_window_ms,
        )

    def get_server_time_ms(self) -> int:
        """Get Binance server time in milliseconds"""
        def _fetch():
            return self.exchange.fetch_time()

        response = self._retry_with_backoff(_fetch)
        return int(response)

    def sync_time(self) -> None:
        """
        Synchronize local time with Binance server using ccxt built-in mechanism.

        Uses ccxt's load_time_difference() which automatically adjusts timestamps
        for subsequent requests.
        """
        try:
            self.exchange.load_time_difference()
            logger.info("Time synchronized using ccxt load_time_difference()")
        except Exception as e:
            logger.error(f"Time sync failed: {e}")
            raise TimestampError(f"Time synchronization failed: {e}")

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """
        Execute function with exponential backoff retry.

        Handles transient errors:
        - Network errors
        - Rate limit errors (with delay)
        - Timestamp errors (auto-sync time and retry immediately)
          Detected via InvalidNonce exception or error message containing 'timestamp'/'recvwindow'

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result

        Raises:
            ExchangeError: On non-retryable errors or max retries exceeded
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                # Fix #4: Record success in health monitor
                self.health_monitor.record_success()
                return result

            except ccxt.DDoSProtection as e:
                # Rate limit - wait and retry
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(wait_time)
                last_exception = RateLimitError(f"Rate limit exceeded: {e}")
                # Fix #4: Record failure in health monitor
                self.health_monitor.record_failure(last_exception)

            except ccxt.RequestTimeout as e:
                # Network timeout - retry with backoff
                wait_time = 2 ** attempt
                logger.warning(f"Request timeout, retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(wait_time)
                last_exception = ExchangeError(f"Request timeout: {e}")
                # Fix #4: Record failure in health monitor
                self.health_monitor.record_failure(last_exception)

            except ccxt.ExchangeNotAvailable as e:
                # Exchange down - retry with backoff
                wait_time = 2 ** attempt
                logger.warning(f"Exchange unavailable, retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(wait_time)
                last_exception = ExchangeError(f"Exchange not available: {e}")
                # Fix #4: Record failure in health monitor
                self.health_monitor.record_failure(last_exception)

            except ccxt.InvalidNonce as e:
                # Timestamp error - sync time using ccxt and retry immediately
                logger.warning(f"Timestamp error, syncing time (attempt {attempt + 1}/{self.max_retries})")
                try:
                    self.exchange.load_time_difference()
                    logger.debug("Time difference loaded successfully")
                except Exception as sync_error:
                    logger.error(f"Time sync failed during retry: {sync_error}")
                last_exception = TimestampError(f"Timestamp error: {e}")
                # Fix #4: Record failure in health monitor
                self.health_monitor.record_failure(last_exception)

            except ccxt.AuthenticationError as e:
                # Auth error - not retryable
                auth_error = AuthError(f"Authentication failed: {e}")
                # Fix #4: Record failure in health monitor
                self.health_monitor.record_failure(auth_error)
                raise auth_error

            except ccxt.InsufficientFunds as e:
                # Insufficient balance - not retryable
                balance_error = InsufficientBalanceError(f"Insufficient balance: {e}")
                # Fix #4: Record failure in health monitor
                self.health_monitor.record_failure(balance_error)
                raise balance_error

            except ccxt.OrderNotFound as e:
                # Order not found - not retryable
                not_found_error = OrderNotFoundError(f"Order not found: {e}")
                # Fix #4: Record failure in health monitor
                self.health_monitor.record_failure(not_found_error)
                raise not_found_error

            except Exception as e:
                # Check if error message contains timestamp/recvWindow keywords
                error_msg = str(e).lower()
                if 'timestamp' in error_msg or 'recvwindow' in error_msg:
                    # Timestamp-related error - sync time and retry immediately
                    logger.warning(f"Timestamp-related error detected, syncing time (attempt {attempt + 1}/{self.max_retries}): {e}")
                    try:
                        self.exchange.load_time_difference()
                        logger.debug("Time difference loaded successfully")
                    except Exception as sync_error:
                        logger.error(f"Time sync failed during retry: {sync_error}")
                    last_exception = TimestampError(f"Timestamp error: {e}")
                else:
                    # Unknown error - log and retry with backoff
                    logger.error(f"Unexpected error: {e} (attempt {attempt + 1}/{self.max_retries})")
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    last_exception = ExchangeError(f"Unexpected error: {e}")
                # Fix #4: Record failure in health monitor
                self.health_monitor.record_failure(last_exception)

        # Max retries exceeded
        final_error = ExchangeError(f"Max retries ({self.max_retries}) exceeded. Last error: {last_exception}")
        # Fix #4: Record final failure in health monitor
        self.health_monitor.record_failure(final_error)
        raise final_error

    def ping(self) -> bool:
        """
        Ping exchange to check connectivity.

        Returns:
            True if exchange is reachable
        """
        try:
            self.get_server_time_ms()
            return True
        except Exception:
            return False

    def fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        since: Optional[int] = None,
    ) -> List[Dict]:
        """
        Fetch OHLCV klines/candlesticks.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Timeframe (e.g., "5m", "1h")
            limit: Number of candles to fetch (max 1500)
            since: Timestamp in ms to fetch from (optional)

        Returns:
            List of OHLCV dicts with keys: timestamp, open, high, low, close, volume
        """
        def _fetch():
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            # Convert to list of dicts
            return [
                {
                    'timestamp': candle[0],
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5],
                }
                for candle in ohlcv
            ]

        return self._retry_with_backoff(_fetch)

    def fetch_funding_rate(self, symbol: str) -> float:
        """
        Fetch current funding rate for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")

        Returns:
            Current funding rate (e.g., 0.0001 = 0.01%)
        """
        def _fetch():
            funding = self.exchange.fetch_funding_rate(symbol)
            return float(funding['fundingRate'])

        return self._retry_with_backoff(_fetch)

    def fetch_balance_usdt(self) -> Dict[str, float]:
        """
        Fetch USDT balance for futures account.

        Returns:
            Dict with keys: total, free, used
        """
        def _fetch():
            if self.testnet:
                # Demo/testnet: use fapiPrivateV2 balance endpoint directly
                # to avoid ccxt routing to spot sapi endpoint first
                response = self.exchange.fapiPrivateV2GetBalance()
                usdt = next((a for a in response if a.get('asset') == 'USDT'), {})
                total = float(usdt.get('balance', 0))
                free = float(usdt.get('availableBalance', 0))
                return {'total': total, 'free': free, 'used': total - free}
            else:
                balance = self.exchange.fetch_balance({'type': 'future'})
                usdt_balance = balance.get('USDT', {})
                return {
                    'total': float(usdt_balance.get('total', 0)),
                    'free': float(usdt_balance.get('free', 0)),
                    'used': float(usdt_balance.get('used', 0)),
                }

        return self._retry_with_backoff(_fetch)

    def fetch_positions(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """
        Fetch open positions.

        Args:
            symbols: List of symbols to fetch (None = all)

        Returns:
            List of position dicts
        """
        def _fetch():
            positions = self.exchange.fetch_positions(symbols)
            # Filter out positions with zero size
            return [
                pos for pos in positions
                if float(pos.get('contracts', 0)) != 0
            ]

        return self._retry_with_backoff(_fetch)

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Fetch open orders.

        Args:
            symbol: Trading pair (None = all symbols)

        Returns:
            List of open order dicts
        """
        def _fetch():
            return self.exchange.fetch_open_orders(symbol)

        return self._retry_with_backoff(_fetch)

    def _check_existing_order(self, client_order_id: str, symbol: str) -> Optional[Dict]:
        """
        Check if an order with the given clientOrderId already exists.

        Checks in order:
        1. Open orders
        2. Recently closed orders (last 200)

        An order is considered "existing" if:
        - clientOrderId matches AND
        - symbol matches AND
        - (filled amount > 0 OR status == 'open')

        Orders are ignored (can retry) if:
        - status in ['canceled','cancelled','rejected','expired'] AND
        - filled amount == 0

        Args:
            client_order_id: Client order ID to check
            symbol: Trading pair

        Returns:
            Existing order dict if found, None otherwise
        """
        ignored_statuses = {'canceled', 'cancelled', 'rejected', 'expired'}

        try:
            # 1. Check open orders first (fastest)
            open_orders = self.fetch_open_orders(symbol)
            for order in open_orders:
                if (order.get('clientOrderId') == client_order_id and
                    order.get('symbol') == symbol):
                    logger.info(f"Found existing open order with clientOrderId={client_order_id}, symbol={symbol}")
                    return order

            # 2. Check recently closed orders (last 200)
            try:
                def _fetch_closed():
                    return self.exchange.fetch_closed_orders(symbol, limit=200)

                closed_orders = self._retry_with_backoff(_fetch_closed)
                for order in closed_orders:
                    if (order.get('clientOrderId') == client_order_id and
                        order.get('symbol') == symbol):
                        status = order.get('status', '').lower()
                        filled = float(order.get('filled', 0))

                        # Order exists if it has fills OR is open
                        if filled > 0 or status == 'open':
                            logger.info(f"Found existing order with clientOrderId={client_order_id}, symbol={symbol}, status={status}, filled={filled}")
                            return order

                        # Ignore canceled/rejected/expired orders with no fills
                        if status in ignored_statuses and filled == 0:
                            logger.debug(f"Found {status} order with clientOrderId={client_order_id}, symbol={symbol} with no fills, ignoring")
                        else:
                            # Unknown case - log for debugging
                            logger.debug(f"Found order with clientOrderId={client_order_id}, symbol={symbol}, status={status}, filled={filled}, not treating as existing")

            except Exception as e:
                logger.debug(f"Failed to check closed orders: {e}")

        except Exception as e:
            logger.warning(f"Failed to check existing orders: {e}")

        return None

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        Place an order (with idempotency check).

        If an open order with the same clientOrderId exists, returns it instead of placing new order.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            side: "buy" or "sell"
            order_type: "limit" or "market" (case-insensitive)
            quantity: Order quantity (contracts)
            price: Order price (required for limit orders)
            reduce_only: Reduce-only flag
            client_order_id: Custom client order ID for idempotency
            params: Additional parameters

        Returns:
            Order dict

        Raises:
            NotImplementedError: If order_type is not LIMIT or MARKET
        """
        # Fix #3: Restrict to LIMIT and MARKET only until Milestone 7
        order_type_upper = order_type.upper()
        if order_type_upper not in {"LIMIT", "MARKET"}:
            raise NotImplementedError(
                f"Order type {order_type} not supported until Milestone 7 execution engine."
            )

        # Idempotency check: if clientOrderId provided, check for existing order
        if client_order_id:
            existing = self._check_existing_order(client_order_id, symbol)
            if existing:
                logger.info(f"Returning existing order instead of placing duplicate: {client_order_id}")
                return existing

        def _place():
            order_params = params or {}

            if reduce_only:
                order_params['reduceOnly'] = True

            if client_order_id:
                order_params['clientOrderId'] = client_order_id

            # Use lowercase for ccxt
            ccxt_type = order_type.lower()

            order = self.exchange.create_order(
                symbol=symbol,
                type=ccxt_type,
                side=side.lower(),
                amount=quantity,
                price=price,
                params=order_params,
            )

            logger.info(f"Order placed: {order.get('id')} | {side} {quantity} {symbol} @ {price}")
            return order

        return self._retry_with_backoff(_place)

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict:
        """
        Cancel an order.

        Args:
            symbol: Trading pair
            order_id: Exchange order ID (optional if client_order_id provided)
            client_order_id: Client order ID (optional if order_id provided)

        Returns:
            Cancelled order dict
        """
        if not order_id and not client_order_id:
            raise ValueError("Either order_id or client_order_id must be provided")

        def _cancel():
            params = {}
            if client_order_id:
                params['clientOrderId'] = client_order_id

            # ccxt requires order_id, but will use clientOrderId from params if available
            order_id_to_use = order_id or client_order_id

            result = self.exchange.cancel_order(order_id_to_use, symbol, params)
            logger.info(f"Order cancelled: {order_id_to_use}")
            return result

        return self._retry_with_backoff(_cancel)

    def cancel_all_orders(self, symbol: str) -> List[Dict]:
        """
        Cancel all open orders for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            List of cancelled order dicts
        """
        def _cancel_all():
            result = self.exchange.cancel_all_orders(symbol)
            logger.info(f"All orders cancelled for {symbol}")
            return result if isinstance(result, list) else [result]

        return self._retry_with_backoff(_cancel_all)

    def list_usdtm_perp_symbols(self) -> List[str]:
        """
        List all available USDT-M perpetual symbols.

        Returns:
            List of symbol names (e.g., ["BTC/USDT", "ETH/USDT"])
        """
        def _list():
            markets = self.exchange.load_markets()
            # Filter for USDT-M perpetuals
            symbols = [
                symbol for symbol, market in markets.items()
                if market.get('type') == 'future'
                and market.get('linear') is True
                and market.get('settle') == 'USDT'
                and market.get('active') is True
            ]
            logger.info(f"Found {len(symbols)} USDT-M perpetual symbols")
            return symbols

        return self._retry_with_backoff(_list)

    def fetch_24h_tickers(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Fetch 24h ticker data for symbols.

        Args:
            symbols: List of symbols (None = all)

        Returns:
            Dict mapping symbol -> {quote_volume_usdt, bid, ask, ...}
        """
        def _fetch():
            if symbols:
                # Fetch tickers for specific symbols
                tickers = {}
                for symbol in symbols:
                    ticker = self.exchange.fetch_ticker(symbol)
                    tickers[symbol] = {
                        'quote_volume_usdt': float(ticker.get('quoteVolume', 0)),
                        'bid': float(ticker.get('bid', 0)) if ticker.get('bid') else None,
                        'ask': float(ticker.get('ask', 0)) if ticker.get('ask') else None,
                        'last': float(ticker.get('last', 0)),
                    }
                return tickers
            else:
                # Fetch all tickers at once
                all_tickers = self.exchange.fetch_tickers()
                return {
                    symbol: {
                        'quote_volume_usdt': float(ticker.get('quoteVolume', 0)),
                        'bid': float(ticker.get('bid', 0)) if ticker.get('bid') else None,
                        'ask': float(ticker.get('ask', 0)) if ticker.get('ask') else None,
                        'last': float(ticker.get('last', 0)),
                    }
                    for symbol, ticker in all_tickers.items()
                }

        return self._retry_with_backoff(_fetch)

    def fetch_funding_rates(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch current funding rates for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dict mapping symbol -> funding_rate
        """
        def _fetch():
            rates = {}
            for symbol in symbols:
                try:
                    funding = self.exchange.fetch_funding_rate(symbol)
                    rates[symbol] = float(funding['fundingRate'])
                except Exception as e:
                    logger.warning(f"Failed to fetch funding rate for {symbol}: {e}")
                    rates[symbol] = 0.0
            return rates

        return self._retry_with_backoff(_fetch)
