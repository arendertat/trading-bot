"""
Scheduler & Event Loop (Task 18)

Drives the 5-minute candle-close decision pipeline and daily/weekly
maintenance tasks without any external scheduler dependency (pure stdlib).

Event schedule
--------------
- Every 5 m  : on_candle_close  → decision pipeline
- 00:00 UTC  : on_daily_open    → refresh universe
- 00:05 UTC  : on_daily_report  → send daily report
- Monday 00:00 UTC : on_weekly_reset → reset weekly PnL window

Design notes
------------
* Time source is injectable (``clock`` callable) to make unit tests
  deterministic without mocking ``datetime.utcnow``.
* ``run_once()`` processes all due events for *one* tick and returns
  immediately — the outer loop calls it and sleeps; this makes the
  scheduler easy to test without threads.
* ``run_forever()`` is the production entry-point: it runs until
  ``stop()`` is called or a fatal exception escapes.
* All callbacks are plain callables; exceptions inside them are caught
  and logged so one bad callback cannot kill the loop.
"""

import logging
import time as _time_module
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Event as _StopEvent
from typing import Callable, Optional

logger = logging.getLogger("trading_bot.core.scheduler")


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class ScheduledEvent(str, Enum):
    CANDLE_CLOSE = "candle_close"      # Every 5 m
    DAILY_OPEN = "daily_open"          # 00:00 UTC
    DAILY_REPORT = "daily_report"      # 00:05 UTC
    WEEKLY_RESET = "weekly_reset"      # Monday 00:00 UTC


# ---------------------------------------------------------------------------
# State dataclass (serialisable for restarts)
# ---------------------------------------------------------------------------

@dataclass
class SchedulerState:
    """Persisted scheduler timestamps"""
    last_candle_close: Optional[datetime] = None
    last_daily_open: Optional[datetime] = None
    last_daily_report: Optional[datetime] = None
    last_weekly_reset: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """
    5-minute candle-close driven event loop with daily/weekly tasks.

    Parameters
    ----------
    on_candle_close : callable
        Called every 5-minute candle close: ``fn(now: datetime) -> None``
    on_daily_open : callable, optional
        Called at 00:00 UTC: ``fn(now: datetime) -> None``
    on_daily_report : callable, optional
        Called at 00:05 UTC: ``fn(now: datetime) -> None``
    on_weekly_reset : callable, optional
        Called Monday 00:00 UTC: ``fn(now: datetime) -> None``
    candle_interval_seconds : int
        How often to fire on_candle_close (default 300 = 5 minutes).
    clock : callable, optional
        Returns current UTC datetime; defaults to ``datetime.utcnow``.
        Override in tests for deterministic behaviour.
    sleep_fn : callable, optional
        Replaces ``time.sleep``; injectable for tests.

    Usage
    -----
    ::

        def pipeline(now):
            ...  # regime detect, strategy select, risk check, order place

        scheduler = Scheduler(on_candle_close=pipeline)
        scheduler.run_forever()
    """

    DAILY_OPEN_MINUTE = 0      # 00:00 UTC
    DAILY_REPORT_MINUTE = 5    # 00:05 UTC

    def __init__(
        self,
        on_candle_close: Callable[[datetime], None],
        on_daily_open: Optional[Callable[[datetime], None]] = None,
        on_daily_report: Optional[Callable[[datetime], None]] = None,
        on_weekly_reset: Optional[Callable[[datetime], None]] = None,
        candle_interval_seconds: int = 300,
        clock: Optional[Callable[[], datetime]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> None:
        if candle_interval_seconds <= 0:
            raise ValueError("candle_interval_seconds must be positive")

        self._on_candle_close = on_candle_close
        self._on_daily_open = on_daily_open
        self._on_daily_report = on_daily_report
        self._on_weekly_reset = on_weekly_reset
        self._candle_interval = candle_interval_seconds
        self._clock = clock or datetime.utcnow
        self._sleep = sleep_fn or _time_module.sleep
        self._stop_event = _StopEvent()
        self._state = SchedulerState()

        logger.info(
            f"Scheduler initialised: candle_interval={candle_interval_seconds}s"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        """
        Start the event loop and block until ``stop()`` is called.

        The loop:
        1. Calls ``run_once(now)``
        2. Sleeps until the next 5-minute boundary
        3. Repeats

        Any exception from callbacks is caught and logged.
        """
        logger.info("Scheduler: starting event loop")
        self._stop_event.clear()

        while not self._stop_event.is_set():
            now = self._clock()

            try:
                self.run_once(now)
            except Exception as e:
                logger.error(f"Scheduler: unhandled exception in run_once: {e}")

            if self._stop_event.is_set():
                break

            sleep_secs = self._seconds_until_next_candle(now)
            logger.debug(f"Scheduler: sleeping {sleep_secs:.1f}s until next candle")
            self._sleep(sleep_secs)

        logger.info("Scheduler: event loop stopped")

    def run_once(self, now: Optional[datetime] = None) -> list:
        """
        Process all events due at ``now``.

        Designed for unit testing: call with a specific timestamp and inspect
        which events fired.

        Parameters
        ----------
        now : datetime, optional
            Current time (defaults to clock()).

        Returns
        -------
        list[ScheduledEvent]
            Events that fired this tick (in firing order).
        """
        if now is None:
            now = self._clock()

        fired: list = []

        # ── Daily events (fire once per calendar day) ──────────────────

        # 00:00 UTC — universe refresh
        if self._should_fire_daily_open(now):
            fired.append(ScheduledEvent.DAILY_OPEN)
            self._state.last_daily_open = now
            self._safe_call(self._on_daily_open, now, ScheduledEvent.DAILY_OPEN)

        # 00:05 UTC — daily report
        if self._should_fire_daily_report(now):
            fired.append(ScheduledEvent.DAILY_REPORT)
            self._state.last_daily_report = now
            self._safe_call(self._on_daily_report, now, ScheduledEvent.DAILY_REPORT)

        # ── Weekly reset (Monday 00:00 UTC) ────────────────────────────
        if self._should_fire_weekly_reset(now):
            fired.append(ScheduledEvent.WEEKLY_RESET)
            self._state.last_weekly_reset = now
            self._safe_call(self._on_weekly_reset, now, ScheduledEvent.WEEKLY_RESET)

        # ── 5-minute candle close ──────────────────────────────────────
        if self._should_fire_candle_close(now):
            fired.append(ScheduledEvent.CANDLE_CLOSE)
            self._state.last_candle_close = now
            self._safe_call(self._on_candle_close, now, ScheduledEvent.CANDLE_CLOSE)

        return fired

    def stop(self) -> None:
        """Signal the event loop to stop after the current tick."""
        logger.info("Scheduler: stop requested")
        self._stop_event.set()

    def load_state(self, state: SchedulerState) -> None:
        """
        Restore scheduler timestamps after a bot restart.

        Prevents re-firing daily/weekly events that already ran today.
        """
        self._state = state
        logger.info("Scheduler: state restored")

    @property
    def state(self) -> SchedulerState:
        """Return a snapshot of the current scheduler state."""
        s = self._state
        return SchedulerState(
            last_candle_close=s.last_candle_close,
            last_daily_open=s.last_daily_open,
            last_daily_report=s.last_daily_report,
            last_weekly_reset=s.last_weekly_reset,
        )

    # ------------------------------------------------------------------
    # Due-date predicates
    # ------------------------------------------------------------------

    def _should_fire_candle_close(self, now: datetime) -> bool:
        """Fire if at least candle_interval_seconds have elapsed since last fire."""
        if self._state.last_candle_close is None:
            return True
        elapsed = (now - self._state.last_candle_close).total_seconds()
        return elapsed >= self._candle_interval

    def _should_fire_daily_open(self, now: datetime) -> bool:
        """
        Fire once at 00:00 UTC (hour=0, minute in [0, DAILY_REPORT_MINUTE)).

        Uses a minute-wide window so we fire even if the tick lands a
        minute late.
        """
        if now.hour != 0 or now.minute != self.DAILY_OPEN_MINUTE:
            return False
        return self._not_fired_today(self._state.last_daily_open, now)

    def _should_fire_daily_report(self, now: datetime) -> bool:
        """Fire once at 00:05 UTC."""
        if now.hour != 0 or now.minute != self.DAILY_REPORT_MINUTE:
            return False
        return self._not_fired_today(self._state.last_daily_report, now)

    def _should_fire_weekly_reset(self, now: datetime) -> bool:
        """Fire on Monday 00:00 UTC, at most once per ISO week."""
        if now.isoweekday() != 1:   # Monday = 1
            return False
        if now.hour != 0 or now.minute != 0:
            return False
        if self._state.last_weekly_reset is None:
            return True
        last = self._state.last_weekly_reset
        # Different ISO (year, week) = new week
        return (now.isocalendar()[:2]) != (last.isocalendar()[:2])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _not_fired_today(last: Optional[datetime], now: datetime) -> bool:
        """Return True if last is None or on a different UTC calendar date."""
        if last is None:
            return True
        return last.date() < now.date()

    def _seconds_until_next_candle(self, now: datetime) -> float:
        """
        Calculate seconds until the next 5-minute candle boundary.

        E.g. at 12:03:20 → next boundary is 12:05:00 → 1 min 40 s = 100 s.
        Returns at least 1 second to avoid busy-looping.
        """
        interval = self._candle_interval
        ts = now.timestamp()
        next_boundary = (ts // interval + 1) * interval
        sleep_secs = next_boundary - ts
        return max(sleep_secs, 1.0)

    def _safe_call(
        self,
        fn: Optional[Callable],
        now: datetime,
        event: ScheduledEvent,
    ) -> None:
        """Call *fn* if not None; catch and log any exception."""
        if fn is None:
            return
        try:
            logger.info(f"Scheduler: firing {event.value} at {now.isoformat()}")
            fn(now)
        except Exception as e:
            logger.error(
                f"Scheduler: callback {event.value} raised exception: {e}"
            )
