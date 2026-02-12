"""Order management with TTL, retry, and idempotency"""

import logging
import time
from datetime import datetime
from typing import Optional, List, Dict
from threading import Lock

from bot.config.models import ExecutionConfig
from bot.exchange.binance_client import BinanceFuturesClient
from bot.execution.models import (
    Order,
    OrderStatus,
    OrderSide,
    OrderType,
    OrderPurpose,
    FillEvent
)
from bot.execution.order_lifecycle import OrderLifecycle


logger = logging.getLogger("trading_bot.execution.order_manager")


class OrderManager:
    """
    Order manager with TTL, retry logic, and idempotency.

    Features:
    - Idempotent order placement (checks for existing orders)
    - TTL-based order expiry and cancellation
    - Automatic retry on failed LIMIT orders
    - Fill tracking and position creation
    - Thread-safe order tracking

    Idempotency Strategy:
        Each order has a deterministic client_order_id.
        Before placing, check if order with same client_order_id exists.
        If exists, return existing order (don't duplicate).
    """

    def __init__(
        self,
        exchange_client: BinanceFuturesClient,
        config: ExecutionConfig,
        lifecycle: Optional[OrderLifecycle] = None
    ):
        """
        Initialize OrderManager.

        Args:
            exchange_client: Binance futures client
            config: Execution configuration
            lifecycle: Order lifecycle manager (creates new if None)
        """
        self.client = exchange_client
        self.config = config
        self.lifecycle = lifecycle or OrderLifecycle()

        # Order tracking
        self._orders: Dict[str, Order] = {}  # client_order_id → Order
        self._orders_lock = Lock()

        logger.info(
            f"OrderManager initialized: TTL={config.limit_ttl_seconds}s, "
            f"retries={config.limit_retry_count}"
        )

    def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        purpose: OrderPurpose,
        client_order_id: str,
        position_id: Optional[str] = None,
        reduce_only: bool = False
    ) -> Order:
        """
        Place LIMIT order with TTL and idempotency.

        Args:
            symbol: Trading pair
            side: Order side (LONG/SHORT)
            quantity: Order quantity
            price: Limit price
            purpose: Order purpose
            client_order_id: Deterministic client order ID
            position_id: Associated position ID
            reduce_only: Whether order is reduce-only

        Returns:
            Order (existing or newly placed)

        Raises:
            Exception: If order placement fails after retries
        """
        # Check for existing order (idempotency)
        existing = self._check_existing_order(symbol, client_order_id)
        if existing:
            logger.info(
                f"Order {client_order_id} already exists (status: {existing.status.value}). "
                "Returning existing order (idempotent)."
            )
            return existing

        # Create new order
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            purpose=purpose,
            quantity=quantity,
            price=price,
            client_order_id=client_order_id,
            position_id=position_id,
            ttl_seconds=self.config.limit_ttl_seconds
        )

        # Track order
        self._add_order(order)

        # Place order with retry logic
        order = self._place_with_retry(order, reduce_only=reduce_only)

        return order

    def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        purpose: OrderPurpose,
        client_order_id: str,
        position_id: Optional[str] = None,
        reduce_only: bool = False
    ) -> Order:
        """
        Place MARKET order (immediate execution).

        Args:
            symbol: Trading pair
            side: Order side
            quantity: Order quantity
            purpose: Order purpose
            client_order_id: Client order ID
            position_id: Associated position ID
            reduce_only: Whether order is reduce-only

        Returns:
            Order (filled or rejected)

        Raises:
            Exception: If order placement fails
        """
        # Check for existing order (idempotency)
        existing = self._check_existing_order(symbol, client_order_id)
        if existing:
            logger.info(f"Order {client_order_id} already exists. Returning existing.")
            return existing

        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            purpose=purpose,
            quantity=quantity,
            client_order_id=client_order_id,
            position_id=position_id
        )

        # Track order
        self._add_order(order)

        # Place order
        try:
            exchange_side = "buy" if side == OrderSide.LONG else "sell"

            response = self.client.place_order(
                symbol=symbol,
                side=exchange_side,
                order_type="market",
                quantity=quantity,
                client_order_id=client_order_id,
                reduce_only=reduce_only
            )

            # Update order from response
            order = self._update_order_from_exchange(order, response)

            logger.info(
                f"MARKET order placed: {client_order_id} ({symbol}, {side.value}, "
                f"{quantity:.6f})"
            )

        except Exception as e:
            logger.error(f"Failed to place MARKET order {client_order_id}: {e}")
            self.lifecycle.mark_rejected(order, reason=str(e))
            raise

        return order

    def place_stop_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        client_order_id: str,
        position_id: str,
        reduce_only: bool = True
    ) -> Order:
        """
        Place STOP_MARKET order.

        Args:
            symbol: Trading pair
            side: Order side
            quantity: Order quantity
            stop_price: Stop trigger price
            client_order_id: Client order ID
            position_id: Associated position ID
            reduce_only: Whether order is reduce-only (default: True)

        Returns:
            Order

        Raises:
            Exception: If placement fails
        """
        # Check for existing order
        existing = self._check_existing_order(symbol, client_order_id)
        if existing:
            logger.info(f"Stop order {client_order_id} already exists.")
            return existing

        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_MARKET,
            purpose=OrderPurpose.STOP,
            quantity=quantity,
            stop_price=stop_price,
            client_order_id=client_order_id,
            position_id=position_id
        )

        # Track order
        self._add_order(order)

        # Place order
        try:
            exchange_side = "buy" if side == OrderSide.LONG else "sell"

            response = self.client.place_order(
                symbol=symbol,
                side=exchange_side,
                order_type="stop_market",
                quantity=quantity,
                stop_price=stop_price,
                client_order_id=client_order_id,
                reduce_only=reduce_only
            )

            order = self._update_order_from_exchange(order, response)

            logger.info(
                f"STOP order placed: {client_order_id} (trigger: {stop_price:.2f})"
            )

        except Exception as e:
            logger.error(f"Failed to place STOP order {client_order_id}: {e}")
            self.lifecycle.mark_rejected(order, reason=str(e))
            raise

        return order

    def place_take_profit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        tp_price: float,
        client_order_id: str,
        position_id: str,
        reduce_only: bool = True
    ) -> Order:
        """
        Place TAKE_PROFIT_MARKET order.

        Args:
            symbol: Trading pair
            side: Order side
            quantity: Order quantity
            tp_price: Take-profit trigger price
            client_order_id: Client order ID
            position_id: Associated position ID
            reduce_only: Whether order is reduce-only (default: True)

        Returns:
            Order

        Raises:
            Exception: If placement fails
        """
        # Check for existing order
        existing = self._check_existing_order(symbol, client_order_id)
        if existing:
            logger.info(f"TP order {client_order_id} already exists.")
            return existing

        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            purpose=OrderPurpose.TAKE_PROFIT,
            quantity=quantity,
            stop_price=tp_price,  # Note: TP uses stop_price field
            client_order_id=client_order_id,
            position_id=position_id
        )

        # Track order
        self._add_order(order)

        # Place order
        try:
            exchange_side = "buy" if side == OrderSide.LONG else "sell"

            response = self.client.place_order(
                symbol=symbol,
                side=exchange_side,
                order_type="take_profit_market",
                quantity=quantity,
                stop_price=tp_price,
                client_order_id=client_order_id,
                reduce_only=reduce_only
            )

            order = self._update_order_from_exchange(order, response)

            logger.info(
                f"TP order placed: {client_order_id} (target: {tp_price:.2f})"
            )

        except Exception as e:
            logger.error(f"Failed to place TP order {client_order_id}: {e}")
            self.lifecycle.mark_rejected(order, reason=str(e))
            raise

        return order

    def cancel_order(self, order: Order, reason: str = "Manual") -> Order:
        """
        Cancel order.

        Args:
            order: Order to cancel
            reason: Cancellation reason

        Returns:
            Updated order

        Raises:
            Exception: If cancellation fails
        """
        if not self.lifecycle.can_cancel(order):
            logger.warning(
                f"Order {order.client_order_id} cannot be canceled "
                f"(status: {order.status.value})"
            )
            return order

        try:
            # Cancel on exchange
            self.client.cancel_order(
                symbol=order.symbol,
                order_id=order.exchange_order_id
            )

            # Update status
            order = self.lifecycle.mark_canceled(order, reason=reason)

            logger.info(f"Order {order.client_order_id} canceled: {reason}")

        except Exception as e:
            logger.error(f"Failed to cancel order {order.client_order_id}: {e}")
            raise

        return order

    def check_order_fills(self, order: Order) -> Order:
        """
        Check order fill status from exchange.

        Args:
            order: Order to check

        Returns:
            Updated order
        """
        if order.is_closed:
            return order

        try:
            # Fetch order status from exchange
            response = self.client.fetch_order(
                symbol=order.symbol,
                order_id=order.exchange_order_id
            )

            # Update order
            order = self._update_order_from_exchange(order, response)

        except Exception as e:
            logger.error(f"Failed to check order {order.client_order_id}: {e}")

        return order

    def process_ttl_expiry(self, order: Order) -> Optional[Order]:
        """
        Process TTL expiry for LIMIT order.

        Checks if order TTL expired. If expired:
        1. Cancel order
        2. If retries remaining, place new order
        3. Otherwise, mark as expired

        Args:
            order: Order to process

        Returns:
            Updated order or new retry order (None if no action needed)
        """
        # Only for LIMIT orders with TTL
        if order.order_type != OrderType.LIMIT or order.ttl_seconds is None:
            return None

        # Check expiry
        if not self.lifecycle.check_ttl_expiry(order):
            return None  # Not expired

        # TTL expired - cancel order
        try:
            self.cancel_order(order, reason="TTL expired")
        except Exception as e:
            logger.error(f"Failed to cancel expired order {order.client_order_id}: {e}")

        # Check if retry available
        if order.retry_count < self.config.limit_retry_count:
            logger.info(
                f"Order {order.client_order_id} expired. Retrying "
                f"({order.retry_count + 1}/{self.config.limit_retry_count})..."
            )

            # Create retry order
            retry_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                purpose=order.purpose,
                quantity=order.quantity,
                price=order.price,
                client_order_id=f"{order.client_order_id}_retry{order.retry_count + 1}",
                position_id=order.position_id,
                ttl_seconds=self.config.limit_ttl_seconds,
                retry_count=order.retry_count + 1
            )

            # Track and place retry
            self._add_order(retry_order)
            retry_order = self._place_single_limit_order(retry_order)

            return retry_order

        else:
            # No retries left
            logger.warning(
                f"Order {order.client_order_id} expired with no retries remaining"
            )
            order = self.lifecycle.mark_expired(order)
            return order

    def get_order(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID"""
        with self._orders_lock:
            return self._orders.get(client_order_id)

    def get_all_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all tracked orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of orders
        """
        with self._orders_lock:
            orders = list(self._orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders"""
        all_orders = self.get_all_orders(symbol)
        return [o for o in all_orders if o.is_open]

    def _check_existing_order(
        self,
        symbol: str,
        client_order_id: str
    ) -> Optional[Order]:
        """
        Check for existing order with same client_order_id.

        Args:
            symbol: Trading pair
            client_order_id: Client order ID

        Returns:
            Existing order or None
        """
        # Check local tracking
        local_order = self.get_order(client_order_id)
        if local_order:
            return local_order

        # Check exchange (in case of restart/recovery)
        try:
            open_orders = self.client.fetch_open_orders(symbol)

            for ex_order in open_orders:
                if ex_order.get("clientOrderId") == client_order_id:
                    logger.info(
                        f"Found existing order on exchange: {client_order_id}"
                    )

                    # Reconstruct order from exchange data
                    order = self._order_from_exchange_response(ex_order)
                    self._add_order(order)

                    return order

        except Exception as e:
            logger.error(f"Failed to check existing orders for {symbol}: {e}")

        return None

    def _place_with_retry(self, order: Order, reduce_only: bool = False) -> Order:
        """
        Place LIMIT order with retry logic.

        Args:
            order: Order to place
            reduce_only: Reduce-only flag

        Returns:
            Updated order

        Raises:
            Exception: If all retries fail
        """
        max_retries = self.config.limit_retry_count + 1  # Initial + retries
        last_error = None

        for attempt in range(max_retries):
            try:
                order = self._place_single_limit_order(order, reduce_only=reduce_only)

                # Wait for TTL
                time.sleep(self.config.limit_ttl_seconds)

                # Check fill status
                order = self.check_order_fills(order)

                # If filled, done
                if order.is_filled or order.is_partially_filled:
                    return order

                # If still open, cancel and retry
                if order.is_open:
                    logger.info(
                        f"Order {order.client_order_id} not filled after TTL. "
                        f"Attempt {attempt + 1}/{max_retries}"
                    )
                    self.cancel_order(order, reason="TTL expired - not filled")

                    # If more retries available, continue
                    if attempt < max_retries - 1:
                        order.retry_count += 1
                        continue
                    else:
                        # No more retries
                        order = self.lifecycle.mark_expired(order)
                        return order

                # Other terminal state
                return order

            except Exception as e:
                last_error = e
                logger.error(
                    f"Attempt {attempt + 1}/{max_retries} failed for {order.client_order_id}: {e}"
                )

                if attempt < max_retries - 1:
                    time.sleep(1)  # Brief pause before retry
                    continue
                else:
                    break

        # All retries exhausted
        self.lifecycle.mark_rejected(order, reason=f"All retries failed: {last_error}")
        raise Exception(f"Failed to place order after {max_retries} attempts: {last_error}")

    def _place_single_limit_order(
        self,
        order: Order,
        reduce_only: bool = False
    ) -> Order:
        """Place single LIMIT order without retry logic"""
        exchange_side = "buy" if order.side == OrderSide.LONG else "sell"

        response = self.client.place_order(
            symbol=order.symbol,
            side=exchange_side,
            order_type="limit",
            quantity=order.quantity,
            price=order.price,
            client_order_id=order.client_order_id,
            reduce_only=reduce_only,
            time_in_force="GTC"  # Good-til-canceled
        )

        # Update order from response
        order = self._update_order_from_exchange(order, response)

        logger.info(
            f"LIMIT order placed: {order.client_order_id} ({order.symbol}, "
            f"{order.side.value}, {order.quantity:.6f} @ {order.price:.2f})"
        )

        return order

    def _update_order_from_exchange(self, order: Order, response: dict) -> Order:
        """Update order from exchange API response"""
        order.exchange_order_id = response.get("orderId") or response.get("id")

        # Update status
        status_str = response.get("status", "").upper()
        if status_str == "NEW":
            order = self.lifecycle.submit_order(order, order.exchange_order_id)
            order = self.lifecycle.mark_open(order)
        elif status_str == "FILLED":
            # Apply fill
            filled_qty = float(response.get("executedQty", 0))
            avg_price = float(response.get("avgPrice") or response.get("price", 0))

            fill_event = FillEvent(
                order_id=order.client_order_id,
                exchange_order_id=order.exchange_order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=filled_qty,
                price=avg_price,
                fee=0.0,  # Will be updated from trade data
                timestamp=datetime.utcnow(),
                is_partial=False,
                cumulative_filled=filled_qty
            )

            order = self.lifecycle.apply_fill(order, fill_event)

        elif status_str == "PARTIALLY_FILLED":
            filled_qty = float(response.get("executedQty", 0))
            avg_price = float(response.get("avgPrice") or response.get("price", 0))

            fill_event = FillEvent(
                order_id=order.client_order_id,
                exchange_order_id=order.exchange_order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=filled_qty - order.filled_quantity,
                price=avg_price,
                fee=0.0,
                timestamp=datetime.utcnow(),
                is_partial=True,
                cumulative_filled=filled_qty
            )

            order = self.lifecycle.apply_fill(order, fill_event)

        elif status_str in ["CANCELED", "EXPIRED"]:
            order = self.lifecycle.mark_canceled(order, reason=status_str)

        elif status_str == "REJECTED":
            order = self.lifecycle.mark_rejected(order, reason="Exchange rejected")

        return order

    def _order_from_exchange_response(self, response: dict) -> Order:
        """Create Order object from exchange API response"""
        # Parse order type
        order_type_str = response.get("type", "").upper()
        if "LIMIT" in order_type_str:
            order_type = OrderType.LIMIT
        elif "MARKET" in order_type_str:
            order_type = OrderType.MARKET
        elif "STOP" in order_type_str:
            order_type = OrderType.STOP_MARKET
        elif "TAKE_PROFIT" in order_type_str:
            order_type = OrderType.TAKE_PROFIT_MARKET
        else:
            order_type = OrderType.LIMIT

        # Parse side
        side_str = response.get("side", "").upper()
        side = OrderSide.LONG if side_str == "BUY" else OrderSide.SHORT

        # Create order
        order = Order(
            symbol=response.get("symbol"),
            side=side,
            order_type=order_type,
            purpose=OrderPurpose.ENTRY,  # Will be updated if stop/TP
            quantity=float(response.get("origQty", 0)),
            price=float(response.get("price", 0)) if response.get("price") else None,
            stop_price=float(response.get("stopPrice", 0)) if response.get("stopPrice") else None,
            client_order_id=response.get("clientOrderId"),
            exchange_order_id=response.get("orderId") or response.get("id"),
            filled_quantity=float(response.get("executedQty", 0)),
            avg_fill_price=float(response.get("avgPrice", 0)) if response.get("avgPrice") else None,
            timestamp_created=datetime.utcnow(),
            timestamp_submitted=datetime.utcnow()
        )

        # Update from response
        return self._update_order_from_exchange(order, response)

    def _add_order(self, order: Order) -> None:
        """Add order to tracking"""
        with self._orders_lock:
            self._orders[order.client_order_id] = order

        logger.debug(f"Order {order.client_order_id} added to tracking")
