"""Order management with deterministic clientOrderId generation"""

import logging
from typing import Dict, Optional

from bot.exchange.binance_client import BinanceFuturesClient


logger = logging.getLogger("trading_bot.exchange")


class OrderManager:
    """
    Manages order placement with deterministic clientOrderId generation.

    Ensures:
    - Unique, deterministic clientOrderIds
    - Idempotent order placement
    - Optional in-memory cache of recent orders
    """

    def __init__(self, client: BinanceFuturesClient, cache_size: int = 1000):
        """
        Initialize OrderManager.

        Args:
            client: BinanceFuturesClient instance
            cache_size: Max size of clientOrderId cache
        """
        self.client = client
        self.cache_size = cache_size

        # Cache mapping clientOrderId -> last seen order
        # Used for quick lookups without API calls
        self._order_cache: Dict[str, Dict] = {}

    @staticmethod
    def build_client_order_id(trade_id: str, role: str) -> str:
        """
        Build deterministic clientOrderId.

        Format: {trade_id}_{role}

        Args:
            trade_id: Unique trade identifier (e.g., "T20240211_001")
            role: Order role (e.g., "entry", "stop", "tp", "trail")

        Returns:
            Deterministic clientOrderId string

        Examples:
            >>> build_client_order_id("T20240211_001", "entry")
            'T20240211_001_entry'
            >>> build_client_order_id("T20240211_001", "stop")
            'T20240211_001_stop'
        """
        if not trade_id or not role:
            raise ValueError("trade_id and role cannot be empty")

        # Ensure valid format (no spaces, special chars that might break API)
        client_order_id = f"{trade_id}_{role}"

        # Binance clientOrderId max length is 36 characters
        if len(client_order_id) > 36:
            raise ValueError(f"clientOrderId too long: {len(client_order_id)} > 36")

        return client_order_id

    def _update_cache(self, client_order_id: str, order: Dict) -> None:
        """Update order cache with LRU-like behavior"""
        if len(self._order_cache) >= self.cache_size:
            # Remove oldest entry (first key)
            oldest_key = next(iter(self._order_cache))
            del self._order_cache[oldest_key]

        self._order_cache[client_order_id] = order

    def _check_cache(self, client_order_id: str) -> Optional[Dict]:
        """Check if order exists in cache"""
        return self._order_cache.get(client_order_id)

    def ensure_unique_and_place(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float],
        trade_id: str,
        role: str,
        reduce_only: bool = False,
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        Ensure unique order placement using deterministic clientOrderId.

        Flow:
        1. Build deterministic clientOrderId
        2. Check cache for existing order
        3. If not in cache, check exchange for existing order
        4. If no existing order, place new order
        5. Update cache

        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            order_type: "limit", "market", etc.
            quantity: Order quantity
            price: Order price (optional for market orders)
            trade_id: Unique trade identifier
            role: Order role (entry, stop, tp, trail)
            reduce_only: Reduce-only flag
            params: Additional parameters

        Returns:
            Order dict (either existing or newly placed)
        """
        # Build deterministic clientOrderId
        client_order_id = self.build_client_order_id(trade_id, role)

        # Check cache first (fast path)
        cached_order = self._check_cache(client_order_id)
        if cached_order:
            logger.debug(f"Order found in cache: {client_order_id}")
            return cached_order

        # Place order (client will check exchange for existing orders)
        order = self.client.place_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            reduce_only=reduce_only,
            client_order_id=client_order_id,
            params=params,
        )

        # Update cache
        self._update_cache(client_order_id, order)

        return order

    def clear_cache(self) -> None:
        """Clear order cache"""
        self._order_cache.clear()
        logger.debug("Order cache cleared")
