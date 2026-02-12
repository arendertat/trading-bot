"""Execution engine data models"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any


class OrderStatus(str, Enum):
    """Order status states"""
    NEW = "NEW"  # Created but not submitted
    SUBMITTED = "SUBMITTED"  # Submitted to exchange
    OPEN = "OPEN"  # Active on exchange
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # Partial fill
    FILLED = "FILLED"  # Fully filled
    CANCELED = "CANCELED"  # Canceled by bot or exchange
    REJECTED = "REJECTED"  # Rejected by exchange
    EXPIRED = "EXPIRED"  # TTL expired


class OrderSide(str, Enum):
    """Order side (direction)"""
    LONG = "LONG"  # Buy/long
    SHORT = "SHORT"  # Sell/short


class OrderType(str, Enum):
    """Order type"""
    LIMIT = "LIMIT"  # Limit order
    MARKET = "MARKET"  # Market order
    STOP_MARKET = "STOP_MARKET"  # Stop-loss market order
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"  # Take-profit market order


class OrderPurpose(str, Enum):
    """Order purpose/role in position"""
    ENTRY = "ENTRY"  # Entry order
    STOP = "STOP"  # Stop-loss order
    TAKE_PROFIT = "TAKE_PROFIT"  # Take-profit order
    EXIT = "EXIT"  # Manual exit order


@dataclass
class Order:
    """
    Order representation with lifecycle tracking.

    Attributes:
        symbol: Trading pair (e.g., "BTCUSDT")
        side: Order side (LONG/SHORT)
        order_type: Order type (LIMIT/MARKET/STOP_MARKET/TAKE_PROFIT_MARKET)
        purpose: Order purpose (ENTRY/STOP/TAKE_PROFIT/EXIT)
        quantity: Order quantity in base currency
        price: Limit price (None for market orders)
        stop_price: Stop/trigger price for stop orders
        client_order_id: Deterministic client order ID
        exchange_order_id: Exchange-assigned order ID
        status: Current order status
        filled_quantity: Filled quantity
        avg_fill_price: Average fill price
        timestamp_created: Order creation timestamp (local)
        timestamp_submitted: Exchange submission timestamp
        timestamp_updated: Last update timestamp
        ttl_seconds: Time-to-live in seconds (for LIMIT orders)
        retry_count: Number of retries attempted
        fees_paid: Trading fees paid
        position_id: Associated position ID
        metadata: Additional order metadata
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    purpose: OrderPurpose
    quantity: float
    client_order_id: str

    # Optional fields
    price: Optional[float] = None
    stop_price: Optional[float] = None
    exchange_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    timestamp_created: Optional[datetime] = None
    timestamp_submitted: Optional[datetime] = None
    timestamp_updated: Optional[datetime] = None
    ttl_seconds: Optional[int] = None
    retry_count: int = 0
    fees_paid: float = 0.0
    position_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize timestamps if not set"""
        if self.timestamp_created is None:
            self.timestamp_created = datetime.utcnow()

    @property
    def is_entry_order(self) -> bool:
        """Check if this is an entry order"""
        return self.purpose == OrderPurpose.ENTRY

    @property
    def is_exit_order(self) -> bool:
        """Check if this is an exit order (stop/TP/manual)"""
        return self.purpose in [OrderPurpose.STOP, OrderPurpose.TAKE_PROFIT, OrderPurpose.EXIT]

    @property
    def is_open(self) -> bool:
        """Check if order is open on exchange"""
        return self.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled"""
        return self.status == OrderStatus.FILLED

    @property
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled"""
        return self.status == OrderStatus.PARTIALLY_FILLED

    @property
    def is_closed(self) -> bool:
        """Check if order is in terminal state"""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]

    @property
    def remaining_quantity(self) -> float:
        """Get unfilled quantity"""
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def fill_percentage(self) -> float:
        """Get fill percentage (0.0 to 1.0)"""
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for logging"""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "purpose": self.purpose.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "timestamp_created": self.timestamp_created.isoformat() if self.timestamp_created else None,
            "timestamp_submitted": self.timestamp_submitted.isoformat() if self.timestamp_submitted else None,
            "timestamp_updated": self.timestamp_updated.isoformat() if self.timestamp_updated else None,
            "ttl_seconds": self.ttl_seconds,
            "retry_count": self.retry_count,
            "fees_paid": self.fees_paid,
            "position_id": self.position_id,
            "metadata": self.metadata
        }


@dataclass
class FillEvent:
    """
    Order fill event.

    Attributes:
        order_id: Client order ID
        exchange_order_id: Exchange order ID
        symbol: Trading pair
        side: Order side
        quantity: Filled quantity
        price: Fill price
        fee: Trading fee
        timestamp: Fill timestamp
        is_partial: Whether this is a partial fill
        cumulative_filled: Total filled quantity so far
    """
    order_id: str
    exchange_order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    timestamp: datetime
    is_partial: bool = False
    cumulative_filled: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert fill event to dictionary"""
        return {
            "order_id": self.order_id,
            "exchange_order_id": self.exchange_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "fee": self.fee,
            "timestamp": self.timestamp.isoformat(),
            "is_partial": self.is_partial,
            "cumulative_filled": self.cumulative_filled
        }
