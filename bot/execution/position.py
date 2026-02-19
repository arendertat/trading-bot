"""Position tracking and management"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any

from bot.execution.models import OrderSide


class PositionStatus(str, Enum):
    """Position status"""
    OPEN = "OPEN"  # Position is open
    CLOSED = "CLOSED"  # Position is closed


class ExitReason(str, Enum):
    """Exit reason tracking"""
    TP = "TP"  # Take profit hit
    SL = "SL"  # Stop loss hit
    TRAIL = "TRAIL"  # Trailing stop hit
    KILL_SWITCH = "KILL_SWITCH"  # Emergency exit
    MANUAL = "MANUAL"  # User-initiated close
    TIMEOUT = "TIMEOUT"  # Max holding period exceeded


@dataclass
class Position:
    """
    Trading position with trailing stop support.

    Attributes:
        position_id: Unique position identifier
        symbol: Trading pair
        side: Position side (LONG/SHORT)
        entry_price: Average entry price
        quantity: Position quantity
        notional_usd: Position notional value in USD
        leverage: Position leverage
        margin_usd: Margin used
        stop_price: Current stop loss price
        tp_price: Take profit price (optional)
        entry_time: Position entry timestamp
        exit_time: Position exit timestamp
        status: Position status (OPEN/CLOSED)
        unrealized_pnl_usd: Unrealized PnL in USD
        realized_pnl_usd: Realized PnL in USD
        fees_paid_usd: Total fees paid
        funding_paid_usd: Total funding paid

        # Risk tracking
        risk_amount_usd: Risk amount at entry (1R)
        initial_stop_price: Original stop price (never changes)

        # Trailing stop
        trail_after_r: Enable trailing after this R multiple (e.g., 1.0)
        atr_trail_mult: ATR multiplier for trail distance (e.g., 2.0)
        trailing_enabled: Whether trailing is currently active
        highest_price_seen: Highest price seen (LONG) / lowest (SHORT)

        # Exit tracking
        exit_reason: Exit reason (TP/SL/TRAIL/etc.)
        exit_price: Actual exit price

        # Order tracking
        entry_order_id: Entry order client ID
        stop_order_id: Current stop order client ID
        tp_order_id: TP order client ID (optional)

        # Metadata
        strategy: Strategy that created this position
        regime: Market regime at entry
        confidence: Strategy confidence score
        metadata: Additional metadata
    """
    position_id: str
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    notional_usd: float
    leverage: float
    margin_usd: float
    stop_price: float
    entry_time: datetime

    # Risk
    risk_amount_usd: float
    initial_stop_price: float

    # Trailing
    trail_after_r: float
    atr_trail_mult: float

    # Order IDs
    entry_order_id: str
    stop_order_id: str

    # Optional fields
    tp_price: Optional[float] = None
    tp_order_id: Optional[str] = None
    exit_time: Optional[datetime] = None
    status: PositionStatus = PositionStatus.OPEN
    unrealized_pnl_usd: float = 0.0
    realized_pnl_usd: float = 0.0
    fees_paid_usd: float = 0.0
    funding_paid_usd: float = 0.0
    trailing_enabled: bool = False
    highest_price_seen: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    exit_price: Optional[float] = None
    strategy: Optional[str] = None
    regime: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize calculated fields"""
        if self.highest_price_seen is None:
            self.highest_price_seen = self.entry_price

    @property
    def is_open(self) -> bool:
        """Check if position is open"""
        return self.status == PositionStatus.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if position is closed"""
        return self.status == PositionStatus.CLOSED

    @property
    def is_long(self) -> bool:
        """Check if position is LONG"""
        return self.side == OrderSide.LONG

    @property
    def is_short(self) -> bool:
        """Check if position is SHORT"""
        return self.side == OrderSide.SHORT

    @property
    def pnl_r(self) -> float:
        """
        Get PnL in R multiples.

        R = risk_amount_usd (1R = initial risk)
        PnL in R = unrealized_pnl_usd / risk_amount_usd
        """
        if self.risk_amount_usd == 0:
            return 0.0
        return self.unrealized_pnl_usd / self.risk_amount_usd

    @property
    def distance_to_stop_pct(self) -> float:
        """Get distance to stop loss as percentage of entry price"""
        return abs(self.stop_price - self.entry_price) / self.entry_price

    @property
    def distance_to_tp_pct(self) -> Optional[float]:
        """Get distance to TP as percentage of entry price"""
        if self.tp_price is None:
            return None
        return abs(self.tp_price - self.entry_price) / self.entry_price

    @property
    def holding_time_seconds(self) -> float:
        """Get holding time in seconds"""
        if self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds()
        else:
            return (datetime.utcnow() - self.entry_time).total_seconds()

    def update_unrealized_pnl(self, current_price: float) -> float:
        """
        Update unrealized PnL based on current price.

        Args:
            current_price: Current market price

        Returns:
            Updated unrealized PnL in USD
        """
        price_diff = current_price - self.entry_price

        if self.is_short:
            price_diff = -price_diff  # Inverse for shorts

        # PnL = price_diff * quantity
        self.unrealized_pnl_usd = price_diff * self.quantity

        return self.unrealized_pnl_usd

    def update_highest_price_seen(self, current_price: float) -> bool:
        """
        Update highest/lowest price seen (for trailing).

        Args:
            current_price: Current market price

        Returns:
            True if highest/lowest was updated
        """
        updated = False

        if self.is_long:
            # Track highest price for LONG
            if current_price > self.highest_price_seen:
                self.highest_price_seen = current_price
                updated = True
        else:
            # Track lowest price for SHORT
            if current_price < self.highest_price_seen:
                self.highest_price_seen = current_price
                updated = True

        return updated

    def should_enable_trailing(self) -> bool:
        """
        Check if trailing should be enabled.

        Trailing enabled when: pnl_r >= trail_after_r

        Returns:
            True if trailing should be enabled
        """
        if self.trailing_enabled:
            return True  # Already enabled

        return self.pnl_r >= self.trail_after_r

    def close_position(
        self,
        exit_price: float,
        exit_reason: ExitReason,
        fees_paid: float = 0.0
    ) -> None:
        """
        Close position and finalize PnL.

        Args:
            exit_price: Exit price
            exit_reason: Reason for exit
            fees_paid: Exit fees paid
        """
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.exit_time = datetime.utcnow()
        self.status = PositionStatus.CLOSED

        # Calculate realized PnL
        price_diff = exit_price - self.entry_price

        if self.is_short:
            price_diff = -price_diff

        gross_pnl = price_diff * self.quantity
        self.fees_paid_usd += fees_paid

        # Realized PnL = gross PnL - all fees
        self.realized_pnl_usd = gross_pnl - self.fees_paid_usd - self.funding_paid_usd
        self.unrealized_pnl_usd = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for logging"""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "notional_usd": self.notional_usd,
            "leverage": self.leverage,
            "margin_usd": self.margin_usd,
            "stop_price": self.stop_price,
            "tp_price": self.tp_price,
            "initial_stop_price": self.initial_stop_price,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "status": self.status.value,
            "unrealized_pnl_usd": self.unrealized_pnl_usd,
            "realized_pnl_usd": self.realized_pnl_usd,
            "pnl_r": self.pnl_r,
            "fees_paid_usd": self.fees_paid_usd,
            "funding_paid_usd": self.funding_paid_usd,
            "trailing_enabled": self.trailing_enabled,
            "trail_after_r": self.trail_after_r,
            "atr_trail_mult": self.atr_trail_mult,
            "highest_price_seen": self.highest_price_seen,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "exit_price": self.exit_price,
            "entry_order_id": self.entry_order_id,
            "stop_order_id": self.stop_order_id,
            "tp_order_id": self.tp_order_id,
            "strategy": self.strategy,
            "regime": self.regime,
            "confidence": self.confidence,
            "holding_time_seconds": self.holding_time_seconds,
            "metadata": self.metadata
        }
