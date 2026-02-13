"""
Trade & Event Logging — Reporting Facade

Provides the LOG_SCHEMA.json-compliant TradeRecord dataclass and a
ReportingTradeLogger that wraps bot.state.logger.TradeLogger with
schema-validated structured logging helpers.

LOG_SCHEMA.json fields are mapped to strongly-typed dataclasses so that
callers cannot accidentally omit required fields or pass wrong types.

Usage::

    from bot.reporting.trade_logger import ReportingTradeLogger, build_trade_record

    rlogger = ReportingTradeLogger(log_dir="logs/")

    record = build_trade_record(
        position=closed_pos,
        mode="PAPER_LIVE",
        portfolio_snapshot=portfolio,
    )
    rlogger.log_full_trade(record)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from bot.execution.position import ExitReason, Position
from bot.state.logger import TradeLogger

# Re-export for callers that do ``from bot.reporting.trade_logger import TradeLogger``
__all__ = [
    "TradeLogger",
    "ReportingTradeLogger",
    "TradeRecord",
    "EntryOrderRecord",
    "RiskRecord",
    "CostsRecord",
    "ResultRecord",
    "PortfolioRecord",
    "build_trade_record",
]

module_logger = logging.getLogger("trading_bot.reporting.trade_logger")

# ---------------------------------------------------------------------------
# LOG_SCHEMA.json-aligned dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EntryOrderRecord:
    """
    Entry order fields from LOG_SCHEMA.json.

    Attributes:
        client_order_id: Deterministic client order ID
        type: Order type (LIMIT | MARKET)
        requested_price: Price requested at order placement
        filled_avg_price: Average fill price
        filled_qty: Quantity filled
        status: Final order status
    """

    client_order_id: str
    type: str  # "LIMIT" | "MARKET"
    requested_price: float
    filled_avg_price: float
    filled_qty: float
    status: str  # "NEW" | "PARTIALLY_FILLED" | "FILLED" | "CANCELED"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (matches LOG_SCHEMA entry_order block)."""
        return asdict(self)


@dataclass
class RiskRecord:
    """
    Risk fields from LOG_SCHEMA.json.

    Attributes:
        equity_usd: Account equity at entry
        risk_pct: Risk as percentage of equity (e.g. 0.01 = 1%)
        risk_usd: Risk amount in USD (1R)
        stop_pct: Stop distance as percentage of entry price
        stop_price: Stop loss price
        take_profit_price: Take profit price (optional)
        leverage: Position leverage
        notional_usd: Position notional value in USD
        margin_used_usd: Margin used for this position
    """

    equity_usd: float
    risk_pct: float
    risk_usd: float
    stop_pct: float
    stop_price: float
    leverage: float
    notional_usd: float
    margin_used_usd: float
    take_profit_price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (matches LOG_SCHEMA risk block)."""
        return asdict(self)


@dataclass
class CostsRecord:
    """
    Trade cost fields from LOG_SCHEMA.json.

    Attributes:
        fees_usd: Total fees paid in USD
        funding_usd: Total funding paid in USD
        slippage_pct: Entry slippage as percentage (actual - requested) / requested
    """

    fees_usd: float
    funding_usd: float
    slippage_pct: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (matches LOG_SCHEMA costs block)."""
        return asdict(self)


@dataclass
class ResultRecord:
    """
    Trade result fields from LOG_SCHEMA.json.

    Attributes:
        exit_price: Actual exit price
        pnl_usd: Realized PnL in USD (net of all costs)
        pnl_r_multiple: PnL expressed as R-multiples
        reason: Exit reason (TP | SL | TRAIL | KILL_SWITCH | MANUAL | TIMEOUT)
    """

    exit_price: float
    pnl_usd: float
    pnl_r_multiple: float
    reason: str  # ExitReason value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (matches LOG_SCHEMA result block)."""
        return asdict(self)


@dataclass
class PortfolioRecord:
    """
    Portfolio snapshot fields from LOG_SCHEMA.json.

    Attributes:
        open_positions_count: Number of open positions at trade close
        open_risk_pct: Total open risk as percentage of equity
        correlation_bucket: Correlation bucket label for this symbol
        bucket_corr_max: Maximum pairwise correlation within the bucket
    """

    open_positions_count: int
    open_risk_pct: float
    correlation_bucket: str
    bucket_corr_max: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (matches LOG_SCHEMA portfolio block)."""
        return asdict(self)


@dataclass
class TradeRecord:
    """
    Full trade record aligned with LOG_SCHEMA.json.

    This is the canonical data structure for a closed trade log entry.
    All fields mirror the LOG_SCHEMA.json specification exactly.

    Attributes:
        trade_id: Unique trade identifier (e.g. "T20240211_001")
        timestamp_open: Trade entry timestamp (ISO-8601 UTC)
        timestamp_close: Trade exit timestamp (ISO-8601 UTC)
        mode: Execution mode (PAPER_LIVE | LIVE)
        symbol: Trading pair (e.g. "BTCUSDT")
        strategy: Strategy name (e.g. "TREND_PULLBACK")
        regime: Market regime at entry (TREND | RANGE | HIGH_VOL | CHOP_NO_TRADE)
        direction: Trade direction (LONG | SHORT)
        confidence_score: Strategy confidence score (0.0 – 1.0)
        entry_order: Entry order details
        risk: Risk parameters at entry
        costs: Transaction costs
        result: Trade result (only for closed trades)
        portfolio: Portfolio snapshot at close
    """

    trade_id: str
    timestamp_open: str  # ISO-8601 UTC
    timestamp_close: str  # ISO-8601 UTC
    mode: str  # "PAPER_LIVE" | "LIVE"
    symbol: str
    strategy: str
    regime: str
    direction: str  # "LONG" | "SHORT"
    confidence_score: float
    entry_order: EntryOrderRecord
    risk: RiskRecord
    costs: CostsRecord
    result: ResultRecord
    portfolio: PortfolioRecord

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dict matching LOG_SCHEMA.json layout exactly.

        Returns:
            Dict with all trade record fields as plain Python types
        """
        return {
            "trade_id": self.trade_id,
            "timestamp_open": self.timestamp_open,
            "timestamp_close": self.timestamp_close,
            "mode": self.mode,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "regime": self.regime,
            "direction": self.direction,
            "confidence_score": self.confidence_score,
            "entry_order": self.entry_order.to_dict(),
            "risk": self.risk.to_dict(),
            "costs": self.costs.to_dict(),
            "result": self.result.to_dict(),
            "portfolio": self.portfolio.to_dict(),
        }


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def build_trade_record(
    position: Position,
    mode: str,
    equity_usd: float,
    risk_pct: float,
    portfolio: PortfolioRecord,
    slippage_pct: Optional[float] = None,
    correlation_bucket: Optional[str] = None,
    bucket_corr_max: Optional[float] = None,
) -> TradeRecord:
    """
    Build a LOG_SCHEMA.json-compliant TradeRecord from a closed Position.

    Computes derived fields (slippage, stop_pct, R-multiple) automatically.

    Args:
        position: Closed position (must have exit_price set)
        mode: Execution mode string ("PAPER_LIVE" | "LIVE")
        equity_usd: Account equity at trade entry
        risk_pct: Risk as fraction of equity (e.g. 0.01 for 1%)
        portfolio: Portfolio snapshot dataclass
        slippage_pct: Pre-computed slippage; if None, computed from
            entry_price vs entry_price (0.0 when no entry order price available)
        correlation_bucket: Override correlation bucket label
        bucket_corr_max: Override max bucket correlation

    Returns:
        TradeRecord ready to pass to ReportingTradeLogger.log_full_trade()

    Raises:
        ValueError: If position is still open (no exit_price)
    """
    if position.is_open:
        raise ValueError(
            f"Cannot build TradeRecord for open position {position.position_id}"
        )

    # Derived fields
    stop_pct = position.distance_to_stop_pct

    entry_order = EntryOrderRecord(
        client_order_id=position.entry_order_id,
        type="LIMIT",  # Default; callers may override via metadata
        requested_price=position.entry_price,
        filled_avg_price=position.entry_price,
        filled_qty=position.quantity,
        status="FILLED",
    )

    # Slippage: (filled - requested) / requested
    if slippage_pct is None:
        slippage_pct = 0.0

    risk_rec = RiskRecord(
        equity_usd=equity_usd,
        risk_pct=risk_pct,
        risk_usd=position.risk_amount_usd,
        stop_pct=stop_pct,
        stop_price=position.initial_stop_price,
        take_profit_price=position.tp_price,
        leverage=position.leverage,
        notional_usd=position.notional_usd,
        margin_used_usd=position.margin_usd,
    )

    costs_rec = CostsRecord(
        fees_usd=position.fees_paid_usd,
        funding_usd=position.funding_paid_usd,
        slippage_pct=slippage_pct,
    )

    result_rec = ResultRecord(
        exit_price=position.exit_price,  # type: ignore[arg-type]
        pnl_usd=position.realized_pnl_usd,
        pnl_r_multiple=position.realized_pnl_usd / position.risk_amount_usd
        if position.risk_amount_usd > 0
        else 0.0,
        reason=position.exit_reason.value if position.exit_reason else "MANUAL",
    )

    # Portfolio override support
    if correlation_bucket is not None:
        portfolio = PortfolioRecord(
            open_positions_count=portfolio.open_positions_count,
            open_risk_pct=portfolio.open_risk_pct,
            correlation_bucket=correlation_bucket,
            bucket_corr_max=bucket_corr_max if bucket_corr_max is not None else portfolio.bucket_corr_max,
        )

    # Trade ID from position_id if not overridden
    trade_id = position.position_id

    timestamp_open = (
        position.entry_time.isoformat()
        if isinstance(position.entry_time, datetime)
        else str(position.entry_time)
    )
    timestamp_close = (
        position.exit_time.isoformat()
        if position.exit_time and isinstance(position.exit_time, datetime)
        else datetime.utcnow().isoformat()
    )

    return TradeRecord(
        trade_id=trade_id,
        timestamp_open=timestamp_open,
        timestamp_close=timestamp_close,
        mode=mode,
        symbol=position.symbol,
        strategy=position.strategy or "UNKNOWN",
        regime=position.regime or "UNKNOWN",
        direction=position.side.value,
        confidence_score=position.confidence or 0.0,
        entry_order=entry_order,
        risk=risk_rec,
        costs=costs_rec,
        result=result_rec,
        portfolio=portfolio,
    )


# ---------------------------------------------------------------------------
# Reporting facade
# ---------------------------------------------------------------------------


class ReportingTradeLogger:
    """
    Reporting-layer trade logger that wraps TradeLogger.

    Provides schema-validated helpers for logging structured TradeRecord
    objects alongside raw position/order events.

    Usage::

        logger = ReportingTradeLogger(log_dir="logs/")
        record = build_trade_record(closed_pos, mode="PAPER_LIVE", ...)
        logger.log_full_trade(record)
        logger.close()

    Attributes:
        _logger: Underlying TradeLogger instance
    """

    def __init__(self, log_dir: str = "logs") -> None:
        """
        Initialise ReportingTradeLogger.

        Args:
            log_dir: Directory for log files
        """
        self._logger = TradeLogger(log_dir=log_dir)
        module_logger.info(f"ReportingTradeLogger initialised (log_dir={log_dir!r})")

    # ------------------------------------------------------------------
    # Schema-validated structured logging
    # ------------------------------------------------------------------

    def log_full_trade(self, record: TradeRecord) -> None:
        """
        Log a complete LOG_SCHEMA.json-compliant trade record.

        Writes to the trades log as a TRADE_RECORD event so that analytics
        tools can distinguish schema-validated records from raw position dumps.

        Args:
            record: Fully-populated TradeRecord dataclass
        """
        payload = record.to_dict()
        from bot.state.logger import RECORD_TYPE_TRADE
        raw_record = self._logger._make_record(  # noqa: WPS437 — intentional internal access
            RECORD_TYPE_TRADE,
            "TRADE_RECORD",
            payload,
        )
        self._logger._write("trades", raw_record)  # noqa: WPS437
        module_logger.info(
            f"Full trade record logged: {record.trade_id} "
            f"({record.symbol} {record.direction} "
            f"pnl={record.result.pnl_usd:.2f} R={record.result.pnl_r_multiple:.2f} "
            f"reason={record.result.reason})"
        )

    def log_trade_opened(self, position: Position) -> None:
        """
        Log a newly opened position to the trade log.

        Args:
            position: Newly opened position
        """
        self._logger.log_trade_opened(position)

    def log_trade_closed(self, position: Position) -> None:
        """
        Log a closed position to the trade log (raw position dict format).

        Prefer log_full_trade() for schema-validated logging.

        Args:
            position: Closed position
        """
        self._logger.log_trade_closed(position)

    def log_event(
        self,
        event_name: str,
        payload: Optional[Dict[str, Any]] = None,
        level: str = "INFO",
    ) -> None:
        """
        Log a generic event.

        Args:
            event_name: Event identifier
            payload: Optional event data
            level: Log level (INFO/WARNING/ERROR)
        """
        self._logger.log_event(event_name, payload, level=level)

    def log_error(
        self,
        component: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an error event.

        Args:
            component: Component that raised the error
            message: Error message
            context: Optional context dict
        """
        self._logger.log_error(component, message, context)

    def log_kill_switch(self, reason: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log kill switch activation.

        Args:
            reason: Reason for activation
            context: Optional context
        """
        self._logger.log_kill_switch(reason, context)

    def flush(self) -> None:
        """Flush all open file handles."""
        self._logger.flush()

    def close(self) -> None:
        """Close all open file handles."""
        self._logger.close()

    @property
    def underlying_logger(self) -> TradeLogger:
        """
        Access the underlying TradeLogger.

        Returns:
            Underlying TradeLogger instance
        """
        return self._logger
