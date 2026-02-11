"""Core constants and enums for the trading bot"""

from enum import Enum


class BotMode(str, Enum):
    """Bot operating mode"""
    PAPER_LIVE = "PAPER_LIVE"
    LIVE = "LIVE"


class RegimeType(str, Enum):
    """Market regime types"""
    TREND = "TREND"
    RANGE = "RANGE"
    HIGH_VOL = "HIGH_VOL"
    CHOP_NO_TRADE = "CHOP_NO_TRADE"


class OrderSide(str, Enum):
    """Order side (direction)"""
    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(str, Enum):
    """Order types"""
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderStatus(str, Enum):
    """Order execution status"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


class PositionStatus(str, Enum):
    """Position status"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
