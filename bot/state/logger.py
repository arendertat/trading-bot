"""JSONL-based trade and event logger with daily rotation"""

import json
import logging
import os
from datetime import datetime, date, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

from bot.execution.position import Position
from bot.execution.models import Order


module_logger = logging.getLogger("trading_bot.state.logger")

# Log record type constants
RECORD_TYPE_TRADE = "TRADE"
RECORD_TYPE_EVENT = "EVENT"
RECORD_TYPE_ORDER = "ORDER"
RECORD_TYPE_ERROR = "ERROR"


class TradeLogger:
    """
    Thread-safe JSONL logger for trades, orders, and events.

    Writes one JSON object per line to daily rotating files:
        <log_dir>/trades_YYYY-MM-DD.jsonl  — closed trade records
        <log_dir>/events_YYYY-MM-DD.jsonl  — significant events

    File rotation happens automatically at midnight UTC.

    Each record contains:
        - record_type: TRADE | EVENT | ORDER | ERROR
        - timestamp: ISO-8601 UTC timestamp
        - payload: event-specific data

    Usage:
        logger = TradeLogger(log_dir="logs/")
        logger.log_trade_closed(position)
        logger.log_event("ORDER_PLACED", {"order_id": "...", "symbol": "BTCUSDT"})
        logger.log_order_filled(order)
        logger.log_error("risk_check", "Notional limit exceeded", {"notional": 50000})
    """

    def __init__(self, log_dir: str = "logs") -> None:
        """
        Initialise TradeLogger.

        Args:
            log_dir: Directory for log files (created if missing)
        """
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._lock = RLock()

        # Current open file handles — keyed by "trades" / "events"
        self._handles: Dict[str, Any] = {}
        # Date for which handles are valid
        self._current_date: Optional[date] = None

        module_logger.info(f"TradeLogger initialised (log_dir={self._log_dir.resolve()})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_trade_closed(self, position: Position) -> None:
        """
        Log a closed trade to the trade log.

        Args:
            position: Closed position to log
        """
        if position.is_open:
            module_logger.warning(
                f"log_trade_closed called with open position {position.position_id}"
            )

        payload = position.to_dict()
        record = self._make_record(RECORD_TYPE_TRADE, "TRADE_CLOSED", payload)
        self._write("trades", record)
        module_logger.info(
            f"Trade logged: {position.position_id} "
            f"({position.symbol} {position.side.value} "
            f"pnl={position.realized_pnl_usd:.2f} "
            f"reason={position.exit_reason.value if position.exit_reason else 'N/A'})"
        )

    def log_trade_opened(self, position: Position) -> None:
        """
        Log a newly opened trade to the trade log.

        Args:
            position: Newly opened position to log
        """
        payload = position.to_dict()
        record = self._make_record(RECORD_TYPE_TRADE, "TRADE_OPENED", payload)
        self._write("trades", record)
        module_logger.info(
            f"Trade opened logged: {position.position_id} "
            f"({position.symbol} {position.side.value} @ {position.entry_price})"
        )

    def log_order_filled(self, order: Order) -> None:
        """
        Log an order fill event.

        Args:
            order: Filled order to log
        """
        payload = {
            "client_order_id": order.client_order_id,
            "exchange_order_id": order.exchange_order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "purpose": order.purpose.value,
            "quantity": order.quantity,
            "filled_quantity": order.filled_quantity,
            "price": order.price,
            "stop_price": order.stop_price,
            "avg_fill_price": order.avg_fill_price,
            "status": order.status.value,
            "position_id": order.position_id,
            "timestamp_created": order.timestamp_created.isoformat() if order.timestamp_created else None,
            "timestamp_submitted": order.timestamp_submitted.isoformat() if order.timestamp_submitted else None,
            "timestamp_updated": order.timestamp_updated.isoformat() if order.timestamp_updated else None,
        }
        record = self._make_record(RECORD_TYPE_ORDER, "ORDER_FILLED", payload)
        self._write("events", record)
        module_logger.debug(f"Order fill logged: {order.client_order_id}")

    def log_order_cancelled(self, order: Order, reason: str = "") -> None:
        """
        Log an order cancellation.

        Args:
            order: Cancelled order
            reason: Cancellation reason
        """
        payload = {
            "client_order_id": order.client_order_id,
            "exchange_order_id": order.exchange_order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "purpose": order.purpose.value,
            "quantity": order.quantity,
            "filled_quantity": order.filled_quantity,
            "status": order.status.value,
            "position_id": order.position_id,
            "cancel_reason": reason,
        }
        record = self._make_record(RECORD_TYPE_ORDER, "ORDER_CANCELLED", payload)
        self._write("events", record)
        module_logger.debug(f"Order cancellation logged: {order.client_order_id}")

    def log_event(
        self,
        event_name: str,
        payload: Optional[Dict[str, Any]] = None,
        level: str = "INFO",
    ) -> None:
        """
        Log a generic event to the event log.

        Args:
            event_name: Event identifier (e.g., "KILL_SWITCH_ACTIVATED")
            payload: Optional event data dict
            level: Log level string (INFO/WARNING/ERROR)
        """
        record = self._make_record(
            RECORD_TYPE_EVENT,
            event_name,
            payload or {},
            level=level,
        )
        self._write("events", record)
        module_logger.debug(f"Event logged: {event_name}")

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
        payload = {
            "component": component,
            "message": message,
            "context": context or {},
        }
        record = self._make_record(RECORD_TYPE_ERROR, "ERROR", payload, level="ERROR")
        self._write("events", record)
        module_logger.debug(f"Error logged from {component}: {message}")

    def log_reconciliation(self, result_dict: Dict[str, Any]) -> None:
        """
        Log a reconciliation summary.

        Args:
            result_dict: ReconciliationResult.to_dict() output
        """
        record = self._make_record(RECORD_TYPE_EVENT, "RECONCILIATION_COMPLETE", result_dict)
        self._write("events", record)
        module_logger.info("Reconciliation result logged")

    def log_kill_switch(self, reason: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log kill switch activation.

        Args:
            reason: Reason for kill switch
            context: Optional context (e.g., open positions count)
        """
        payload = {"reason": reason, "context": context or {}}
        record = self._make_record(RECORD_TYPE_EVENT, "KILL_SWITCH_ACTIVATED", payload, level="WARNING")
        self._write("events", record)
        module_logger.warning(f"Kill switch activation logged: {reason}")

    def flush(self) -> None:
        """Flush all open file handles."""
        with self._lock:
            for handle in self._handles.values():
                try:
                    handle.flush()
                except Exception as e:
                    module_logger.error(f"Error flushing log handle: {e}")

    def close(self) -> None:
        """Close all open file handles."""
        with self._lock:
            for name, handle in list(self._handles.items()):
                try:
                    handle.flush()
                    handle.close()
                    module_logger.debug(f"Closed log handle: {name}")
                except Exception as e:
                    module_logger.error(f"Error closing log handle {name}: {e}")
            self._handles.clear()
            self._current_date = None

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_trade_log_path(self, for_date: Optional[date] = None) -> Path:
        """
        Get trade log file path for a given date.

        Args:
            for_date: Date (UTC) — defaults to today

        Returns:
            Path to trade log file
        """
        d = for_date or datetime.now(timezone.utc).date()
        return self._log_dir / f"trades_{d.isoformat()}.jsonl"

    def get_event_log_path(self, for_date: Optional[date] = None) -> Path:
        """
        Get event log file path for a given date.

        Args:
            for_date: Date (UTC) — defaults to today

        Returns:
            Path to event log file
        """
        d = for_date or datetime.now(timezone.utc).date()
        return self._log_dir / f"events_{d.isoformat()}.jsonl"

    def list_trade_logs(self) -> list:
        """Return sorted list of all trade log paths."""
        return sorted(self._log_dir.glob("trades_*.jsonl"))

    def list_event_logs(self) -> list:
        """Return sorted list of all event log paths."""
        return sorted(self._log_dir.glob("events_*.jsonl"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_str(value: Any) -> Any:
        """
        Bulgu 9.4: Strip newline/carriage-return characters from string values
        to prevent JSONL format corruption via log injection.
        """
        if isinstance(value, str):
            return value.replace("\n", "\\n").replace("\r", "\\r")
        return value

    @classmethod
    def _sanitize_payload(cls, payload: Any) -> Any:
        """Recursively sanitize string values in a payload dict/list."""
        if isinstance(payload, dict):
            return {k: cls._sanitize_payload(v) for k, v in payload.items()}
        if isinstance(payload, list):
            return [cls._sanitize_payload(item) for item in payload]
        return cls._sanitize_str(payload)

    def _make_record(
        self,
        record_type: str,
        event_name: str,
        payload: Dict[str, Any],
        level: str = "INFO",
    ) -> str:
        """
        Build a JSONL record string.

        Args:
            record_type: TRADE | EVENT | ORDER | ERROR
            event_name: Specific event name
            payload: Event payload dict
            level: Log level string

        Returns:
            JSON-encoded string (no trailing newline)
        """
        record = {
            "record_type": record_type,
            "event": event_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "payload": self._sanitize_payload(payload),
        }
        return json.dumps(record, default=str)

    def _get_handle(self, stream: str) -> Any:
        """
        Get (or create) file handle for stream, rotating if date changed.

        Args:
            stream: "trades" or "events"

        Returns:
            Open file handle (text mode, append)
        """
        today = datetime.now(timezone.utc).date()

        # Rotate if date changed
        if self._current_date != today:
            self._rotate(today)

        if stream not in self._handles:
            path = (
                self.get_trade_log_path(today)
                if stream == "trades"
                else self.get_event_log_path(today)
            )
            self._handles[stream] = open(path, "a", encoding="utf-8")  # noqa: WPS515
            module_logger.debug(f"Opened log file: {path}")

        return self._handles[stream]

    def _rotate(self, new_date: date) -> None:
        """
        Close current handles and reset for new date.

        Args:
            new_date: New date to rotate to
        """
        for name, handle in list(self._handles.items()):
            try:
                handle.flush()
                handle.close()
            except Exception as e:
                module_logger.error(f"Error during log rotation for {name}: {e}")
        self._handles.clear()
        self._current_date = new_date
        module_logger.info(f"Log files rotated to date: {new_date.isoformat()}")

    def _write(self, stream: str, record_line: str) -> None:
        """
        Write a record line to a stream, thread-safely.

        Args:
            stream: "trades" or "events"
            record_line: JSON string to write
        """
        with self._lock:
            try:
                handle = self._get_handle(stream)
                handle.write(record_line + "\n")
                handle.flush()
            except Exception as e:
                module_logger.error(
                    f"Failed to write to {stream} log: {e}",
                    exc_info=True,
                )
