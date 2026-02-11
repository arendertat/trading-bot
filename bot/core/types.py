"""Core data types for the trading bot"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from bot.core.constants import OrderSide, OrderStatus, OrderType, PositionStatus, RegimeType


@dataclass
class Candle:
    """OHLCV candle data"""
    timestamp: int  # Unix timestamp in milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Position:
    """Trading position"""
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    notional: float
    leverage: float
    margin: float
    stop_price: float
    tp_price: Optional[float]
    unrealized_pnl: float
    realized_pnl: float
    entry_time: datetime
    trade_id: str
    status: PositionStatus = PositionStatus.OPEN


@dataclass
class Order:
    """Order data"""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: Optional[float]
    quantity: float
    status: OrderStatus
    filled_quantity: float = 0.0
    fill_price: Optional[float] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class Trade:
    """Completed trade record"""
    trade_id: str
    symbol: str
    side: OrderSide
    strategy: str
    regime: RegimeType
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    realized_pnl: float
    fees: float
    funding: float
    r_multiple: float


@dataclass
class KillSwitchState:
    """Kill switch state tracking"""
    daily_stop_active: bool = False
    weekly_pause_active: bool = False
    pause_end_date: Optional[datetime] = None
    reduced_risk_end_date: Optional[datetime] = None
    current_daily_pnl: float = 0.0
    current_weekly_pnl: float = 0.0
    last_daily_reset: Optional[datetime] = None
    last_weekly_reset: Optional[datetime] = None
