"""JSONL log reader for crash recovery and analytics"""

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

from bot.state.logger import RECORD_TYPE_TRADE, RECORD_TYPE_EVENT, RECORD_TYPE_ORDER, RECORD_TYPE_ERROR


module_logger = logging.getLogger("trading_bot.state.log_reader")


class LogRecord:
    """
    Parsed JSONL log record.

    Attributes:
        record_type: TRADE | EVENT | ORDER | ERROR
        event: Specific event name (e.g., "TRADE_CLOSED")
        timestamp: Parsed UTC datetime
        level: Log level string
        payload: Event-specific data dict
        raw: Original JSON string
    """

    __slots__ = ("record_type", "event", "timestamp", "level", "payload", "raw")

    def __init__(
        self,
        record_type: str,
        event: str,
        timestamp: datetime,
        level: str,
        payload: Dict[str, Any],
        raw: str,
    ) -> None:
        self.record_type = record_type
        self.event = event
        self.timestamp = timestamp
        self.level = level
        self.payload = payload
        self.raw = raw

    def __repr__(self) -> str:
        return (
            f"LogRecord(type={self.record_type}, event={self.event}, "
            f"ts={self.timestamp.isoformat()})"
        )


class LogReadError(Exception):
    """Raised when a log line cannot be parsed."""
    pass


class LogReader:
    """
    Reads and parses JSONL log files produced by TradeLogger.

    Supports:
    - Iterating over all records in a date range
    - Filtering by record type or event name
    - Recovering closed trade records for analytics
    - Recovering open trades for crash recovery

    Usage:
        reader = LogReader(log_dir="logs/")

        # Iterate all trade records for today
        for record in reader.read_trades():
            print(record.payload["position_id"], record.payload["realized_pnl_usd"])

        # Get closed trades in date range
        trades = reader.get_closed_trades(start=date(2024, 1, 1), end=date(2024, 1, 31))

        # Get events filtered by name
        events = reader.get_events(event_name="KILL_SWITCH_ACTIVATED")
    """

    def __init__(self, log_dir: str = "logs") -> None:
        """
        Initialise LogReader.

        Args:
            log_dir: Directory containing log files
        """
        self._log_dir = Path(log_dir)
        module_logger.info(f"LogReader initialised (log_dir={self._log_dir.resolve()})")

    # ------------------------------------------------------------------
    # Public API — iterators
    # ------------------------------------------------------------------

    def read_trades(
        self,
        start: Optional[date] = None,
        end: Optional[date] = None,
        skip_errors: bool = True,
    ) -> Iterator[LogRecord]:
        """
        Iterate over trade log records in date range.

        Args:
            start: Start date (inclusive), defaults to all
            end: End date (inclusive), defaults to today
            skip_errors: If True, skip unparseable lines; else raise LogReadError

        Yields:
            LogRecord objects from trade logs
        """
        paths = self._get_trade_paths(start, end)
        yield from self._iter_paths(paths, skip_errors=skip_errors)

    def read_events(
        self,
        start: Optional[date] = None,
        end: Optional[date] = None,
        skip_errors: bool = True,
    ) -> Iterator[LogRecord]:
        """
        Iterate over event log records in date range.

        Args:
            start: Start date (inclusive), defaults to all
            end: End date (inclusive), defaults to today
            skip_errors: If True, skip unparseable lines; else raise LogReadError

        Yields:
            LogRecord objects from event logs
        """
        paths = self._get_event_paths(start, end)
        yield from self._iter_paths(paths, skip_errors=skip_errors)

    # ------------------------------------------------------------------
    # Public API — filtered helpers
    # ------------------------------------------------------------------

    def get_closed_trades(
        self,
        start: Optional[date] = None,
        end: Optional[date] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get closed trade payloads.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)
            symbol: Optional symbol filter

        Returns:
            List of trade payload dicts, sorted by entry_time ascending
        """
        trades = []
        for record in self.read_trades(start=start, end=end):
            if record.event != "TRADE_CLOSED":
                continue
            payload = record.payload
            if symbol and payload.get("symbol") != symbol:
                continue
            trades.append(payload)

        # Sort by entry_time for deterministic order
        trades.sort(key=lambda t: t.get("entry_time", ""))
        return trades

    def get_opened_trades(
        self,
        start: Optional[date] = None,
        end: Optional[date] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get opened (not yet closed) trade payloads.

        Useful for crash recovery: find positions that were opened
        but not yet closed in the trade log.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)
            symbol: Optional symbol filter

        Returns:
            List of trade payload dicts for trades still marked OPEN
        """
        open_trades: Dict[str, Dict[str, Any]] = {}

        for record in self.read_trades(start=start, end=end):
            payload = record.payload
            position_id = payload.get("position_id", "")

            if symbol and payload.get("symbol") != symbol:
                continue

            if record.event == "TRADE_OPENED":
                open_trades[position_id] = payload
            elif record.event == "TRADE_CLOSED":
                # Remove from open set — trade is closed
                open_trades.pop(position_id, None)

        return list(open_trades.values())

    def get_events(
        self,
        event_name: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> List[LogRecord]:
        """
        Get event log records, optionally filtered by event name.

        Args:
            event_name: Event name filter (e.g., "KILL_SWITCH_ACTIVATED")
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            List of matching LogRecord objects
        """
        results = []
        for record in self.read_events(start=start, end=end):
            if event_name and record.event != event_name:
                continue
            results.append(record)
        return results

    def get_errors(
        self,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> List[LogRecord]:
        """
        Get all ERROR records from event logs.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            List of error LogRecord objects
        """
        results = []
        for record in self.read_events(start=start, end=end):
            if record.record_type == RECORD_TYPE_ERROR:
                results.append(record)
        return results

    # ------------------------------------------------------------------
    # Analytics helpers
    # ------------------------------------------------------------------

    def compute_pnl_summary(
        self,
        start: Optional[date] = None,
        end: Optional[date] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute PnL summary from closed trade logs.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)
            symbol: Optional symbol filter

        Returns:
            Summary dict with total_trades, total_pnl_usd, win_rate, avg_pnl_r, etc.
        """
        trades = self.get_closed_trades(start=start, end=end, symbol=symbol)

        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl_usd": 0.0,
                "total_pnl_r": 0.0,
                "avg_pnl_usd": 0.0,
                "avg_pnl_r": 0.0,
                "max_winner_usd": 0.0,
                "max_loser_usd": 0.0,
                "total_fees_usd": 0.0,
                "exit_reason_counts": {},
            }

        total_pnl_usd = sum(t.get("realized_pnl_usd", 0.0) for t in trades)
        total_pnl_r = sum(t.get("pnl_r", 0.0) for t in trades)
        total_fees_usd = sum(t.get("fees_paid_usd", 0.0) for t in trades)
        pnl_values = [t.get("realized_pnl_usd", 0.0) for t in trades]

        winning = [p for p in pnl_values if p > 0]
        losing = [p for p in pnl_values if p <= 0]

        # Exit reason breakdown
        exit_reason_counts: Dict[str, int] = {}
        for t in trades:
            reason = t.get("exit_reason") or "UNKNOWN"
            exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1

        total = len(trades)
        return {
            "total_trades": total,
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / total if total > 0 else 0.0,
            "total_pnl_usd": round(total_pnl_usd, 4),
            "total_pnl_r": round(total_pnl_r, 4),
            "avg_pnl_usd": round(total_pnl_usd / total, 4) if total > 0 else 0.0,
            "avg_pnl_r": round(total_pnl_r / total, 4) if total > 0 else 0.0,
            "max_winner_usd": round(max(pnl_values), 4) if pnl_values else 0.0,
            "max_loser_usd": round(min(pnl_values), 4) if pnl_values else 0.0,
            "total_fees_usd": round(total_fees_usd, 4),
            "exit_reason_counts": exit_reason_counts,
        }

    def find_position_events(self, position_id: str) -> List[LogRecord]:
        """
        Find all events related to a specific position.

        Searches both trade and event logs across all dates.

        Args:
            position_id: Position ID to search for

        Returns:
            List of related LogRecord objects, sorted by timestamp
        """
        results = []

        for record in self.read_trades():
            payload = record.payload
            if payload.get("position_id") == position_id:
                results.append(record)

        for record in self.read_events():
            payload = record.payload
            # Check direct position_id field and nested context
            if (
                payload.get("position_id") == position_id
                or payload.get("context", {}).get("position_id") == position_id
            ):
                results.append(record)

        results.sort(key=lambda r: r.timestamp)
        return results

    # ------------------------------------------------------------------
    # File path helpers
    # ------------------------------------------------------------------

    def list_trade_logs(self) -> List[Path]:
        """Return sorted list of all trade log paths."""
        return sorted(self._log_dir.glob("trades_*.jsonl"))

    def list_event_logs(self) -> List[Path]:
        """Return sorted list of all event log paths."""
        return sorted(self._log_dir.glob("events_*.jsonl"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_trade_paths(
        self, start: Optional[date], end: Optional[date]
    ) -> List[Path]:
        """Get sorted trade log paths filtered by date range."""
        return self._filter_by_date(self.list_trade_logs(), start, end)

    def _get_event_paths(
        self, start: Optional[date], end: Optional[date]
    ) -> List[Path]:
        """Get sorted event log paths filtered by date range."""
        return self._filter_by_date(self.list_event_logs(), start, end)

    def _filter_by_date(
        self,
        paths: List[Path],
        start: Optional[date],
        end: Optional[date],
    ) -> List[Path]:
        """
        Filter log paths by date range.

        Log files are named: <type>_YYYY-MM-DD.jsonl
        """
        result = []
        for path in paths:
            file_date = self._extract_date_from_path(path)
            if file_date is None:
                continue
            if start and file_date < start:
                continue
            if end and file_date > end:
                continue
            result.append(path)
        return result

    def _extract_date_from_path(self, path: Path) -> Optional[date]:
        """
        Extract date from log file name.

        Expected format: trades_YYYY-MM-DD.jsonl or events_YYYY-MM-DD.jsonl

        Args:
            path: Log file path

        Returns:
            Parsed date or None if not parseable
        """
        stem = path.stem  # e.g., "trades_2024-01-15"
        parts = stem.split("_", 1)
        if len(parts) != 2:
            return None
        try:
            return date.fromisoformat(parts[1])
        except ValueError:
            return None

    def _iter_paths(
        self,
        paths: List[Path],
        skip_errors: bool = True,
    ) -> Generator[LogRecord, None, None]:
        """
        Iterate over log records in a list of paths.

        Args:
            paths: Log file paths to iterate
            skip_errors: If True, log and skip bad lines; else raise

        Yields:
            Parsed LogRecord objects
        """
        for path in paths:
            yield from self._iter_file(path, skip_errors=skip_errors)

    def _iter_file(
        self,
        path: Path,
        skip_errors: bool = True,
    ) -> Generator[LogRecord, None, None]:
        """
        Iterate over records in a single log file.

        Args:
            path: Log file path
            skip_errors: If True, log and skip bad lines; else raise

        Yields:
            Parsed LogRecord objects

        Raises:
            LogReadError: If skip_errors=False and a line cannot be parsed
        """
        if not path.exists():
            module_logger.warning(f"Log file not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = self._parse_line(line)
                    yield record
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    msg = f"Parse error in {path}:{line_num}: {e}"
                    if skip_errors:
                        module_logger.warning(msg)
                    else:
                        raise LogReadError(msg) from e

    def _parse_line(self, line: str) -> LogRecord:
        """
        Parse a single JSONL record line.

        Args:
            line: JSON string

        Returns:
            LogRecord

        Raises:
            json.JSONDecodeError: If not valid JSON
            KeyError: If required fields missing
            ValueError: If timestamp cannot be parsed
        """
        data = json.loads(line)

        record_type = data["record_type"]
        event = data["event"]
        timestamp_str = data["timestamp"]
        level = data.get("level", "INFO")
        payload = data.get("payload", {})

        # Parse ISO-8601 timestamp
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except ValueError:
            raise ValueError(f"Cannot parse timestamp: {timestamp_str!r}")

        return LogRecord(
            record_type=record_type,
            event=event,
            timestamp=timestamp,
            level=level,
            payload=payload,
            raw=line,
        )
