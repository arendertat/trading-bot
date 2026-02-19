"""Tests for Scheduler & Event Loop (Task 18)"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, call, patch

from bot.core.scheduler import (
    Scheduler,
    ScheduledEvent,
    SchedulerState,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def ts(year=2026, month=1, day=5, hour=0, minute=0, second=0):
    """Create a UTC datetime (Monday = Jan 5 2026)."""
    return datetime(year, month, day, hour, minute, second)


def make_scheduler(
    candle_cb=None,
    daily_open_cb=None,
    daily_report_cb=None,
    weekly_reset_cb=None,
    candle_interval=300,
    clock_time: datetime = None,
    sleep_fn=None,
):
    """Factory that wires a fixed-clock scheduler."""
    if candle_cb is None:
        candle_cb = MagicMock()
    _clock_time = [clock_time or ts()]

    def _clock():
        return _clock_time[0]

    _sleep = sleep_fn or MagicMock()

    sched = Scheduler(
        on_candle_close=candle_cb,
        on_daily_open=daily_open_cb,
        on_daily_report=daily_report_cb,
        on_weekly_reset=weekly_reset_cb,
        candle_interval_seconds=candle_interval,
        clock=_clock,
        sleep_fn=_sleep,
    )
    return sched, _clock_time, candle_cb


# ===========================================================================
# TestSchedulerInit
# ===========================================================================

class TestSchedulerInit:
    def test_creates_with_minimal_args(self):
        sched = Scheduler(on_candle_close=MagicMock())
        assert sched is not None

    def test_invalid_candle_interval_zero(self):
        with pytest.raises(ValueError):
            Scheduler(on_candle_close=MagicMock(), candle_interval_seconds=0)

    def test_invalid_candle_interval_negative(self):
        with pytest.raises(ValueError):
            Scheduler(on_candle_close=MagicMock(), candle_interval_seconds=-1)

    def test_state_starts_empty(self):
        sched = Scheduler(on_candle_close=MagicMock())
        s = sched.state
        assert s.last_candle_close is None
        assert s.last_daily_open is None
        assert s.last_daily_report is None
        assert s.last_weekly_reset is None

    def test_state_snapshot_is_copy(self):
        sched = Scheduler(on_candle_close=MagicMock())
        s1 = sched.state
        s2 = sched.state
        assert s1 is not s2


# ===========================================================================
# TestCandleClose
# ===========================================================================

class TestCandleClose:
    def test_fires_on_first_call(self):
        sched, clk, cb = make_scheduler()
        clk[0] = ts(hour=10, minute=0)
        events = sched.run_once(clk[0])
        assert ScheduledEvent.CANDLE_CLOSE in events
        cb.assert_called_once()

    def test_fires_with_now_argument(self):
        sched, clk, cb = make_scheduler(candle_interval=300)
        now = ts(hour=10, minute=0)
        sched.run_once(now)
        cb.assert_called_once_with(now)

    def test_does_not_fire_before_interval(self):
        sched, clk, cb = make_scheduler(candle_interval=300)
        t0 = ts(hour=10, minute=0)
        sched.run_once(t0)
        cb.reset_mock()
        # 4 minutes later — not due yet
        sched.run_once(t0 + timedelta(seconds=240))
        cb.assert_not_called()

    def test_fires_after_interval(self):
        sched, clk, cb = make_scheduler(candle_interval=300)
        t0 = ts(hour=10, minute=0)
        sched.run_once(t0)
        cb.reset_mock()
        sched.run_once(t0 + timedelta(seconds=300))
        cb.assert_called_once()

    def test_fires_after_over_interval(self):
        sched, clk, cb = make_scheduler(candle_interval=300)
        t0 = ts(hour=10, minute=0)
        sched.run_once(t0)
        cb.reset_mock()
        sched.run_once(t0 + timedelta(seconds=350))
        cb.assert_called_once()

    def test_state_updated_after_fire(self):
        sched, clk, cb = make_scheduler(candle_interval=300)
        t0 = ts(hour=10, minute=0)
        sched.run_once(t0)
        assert sched.state.last_candle_close == t0

    def test_fires_multiple_times(self):
        sched, clk, cb = make_scheduler(candle_interval=300)
        for i in range(3):
            sched.run_once(ts(hour=10, minute=0) + timedelta(seconds=i * 300))
        assert cb.call_count == 3

    def test_callback_receives_now(self):
        calls = []
        sched, clk, _ = make_scheduler(
            candle_cb=lambda now: calls.append(now),
            candle_interval=60,
        )
        t = ts(hour=12, minute=30)
        sched.run_once(t)
        assert calls == [t]


# ===========================================================================
# TestDailyOpen
# ===========================================================================

class TestDailyOpen:
    def test_fires_at_midnight(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(daily_open_cb=cb)
        sched.run_once(ts(hour=0, minute=0))
        cb.assert_called_once()

    def test_does_not_fire_outside_midnight(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(daily_open_cb=cb)
        sched.run_once(ts(hour=12, minute=0))
        cb.assert_not_called()

    def test_does_not_fire_at_wrong_minute(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(daily_open_cb=cb)
        sched.run_once(ts(hour=0, minute=1))
        cb.assert_not_called()

    def test_fires_once_per_day(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(daily_open_cb=cb)
        day1 = ts(year=2026, month=1, day=6, hour=0, minute=0)
        sched.run_once(day1)
        # Same day again (e.g. scheduler tick re-runs)
        sched.run_once(day1)
        assert cb.call_count == 1

    def test_fires_again_next_day(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(daily_open_cb=cb)
        day1 = ts(year=2026, month=1, day=6, hour=0, minute=0)
        day2 = ts(year=2026, month=1, day=7, hour=0, minute=0)
        sched.run_once(day1)
        sched.run_once(day2)
        assert cb.call_count == 2

    def test_fires_with_correct_timestamp(self):
        calls = []
        sched, clk, _ = make_scheduler(daily_open_cb=lambda now: calls.append(now))
        midnight = ts(hour=0, minute=0)
        sched.run_once(midnight)
        assert calls == [midnight]

    def test_no_callback_does_not_raise(self):
        sched, clk, _ = make_scheduler(daily_open_cb=None)
        sched.run_once(ts(hour=0, minute=0))  # must not raise

    def test_state_updated_after_fire(self):
        sched, clk, _ = make_scheduler(daily_open_cb=MagicMock())
        t = ts(hour=0, minute=0)
        sched.run_once(t)
        assert sched.state.last_daily_open == t


# ===========================================================================
# TestDailyReport
# ===========================================================================

class TestDailyReport:
    def test_fires_at_00_05(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(daily_report_cb=cb)
        sched.run_once(ts(hour=0, minute=5))
        cb.assert_called_once()

    def test_does_not_fire_at_midnight(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(daily_report_cb=cb)
        sched.run_once(ts(hour=0, minute=0))
        cb.assert_not_called()

    def test_does_not_fire_at_wrong_minute(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(daily_report_cb=cb)
        sched.run_once(ts(hour=0, minute=6))
        cb.assert_not_called()

    def test_fires_once_per_day(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(daily_report_cb=cb)
        t = ts(hour=0, minute=5)
        sched.run_once(t)
        sched.run_once(t)
        assert cb.call_count == 1

    def test_fires_again_next_day(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(daily_report_cb=cb)
        sched.run_once(ts(year=2026, month=1, day=6, hour=0, minute=5))
        sched.run_once(ts(year=2026, month=1, day=7, hour=0, minute=5))
        assert cb.call_count == 2

    def test_state_updated_after_fire(self):
        sched, clk, _ = make_scheduler(daily_report_cb=MagicMock())
        t = ts(hour=0, minute=5)
        sched.run_once(t)
        assert sched.state.last_daily_report == t

    def test_no_callback_does_not_raise(self):
        sched, clk, _ = make_scheduler(daily_report_cb=None)
        sched.run_once(ts(hour=0, minute=5))


# ===========================================================================
# TestWeeklyReset
# ===========================================================================

class TestWeeklyReset:
    def _monday(self, week_offset=0):
        # Jan 5 2026 is Monday
        base = datetime(2026, 1, 5, 0, 0, 0)
        return base + timedelta(weeks=week_offset)

    def test_fires_on_monday_midnight(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(weekly_reset_cb=cb)
        sched.run_once(self._monday())
        cb.assert_called_once()

    def test_does_not_fire_on_tuesday(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(weekly_reset_cb=cb)
        tuesday = datetime(2026, 1, 7, 0, 0, 0)
        sched.run_once(tuesday)
        cb.assert_not_called()

    def test_does_not_fire_monday_non_midnight(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(weekly_reset_cb=cb)
        sched.run_once(datetime(2026, 1, 6, 12, 0, 0))
        cb.assert_not_called()

    def test_fires_once_per_week(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(weekly_reset_cb=cb)
        monday = self._monday()
        sched.run_once(monday)
        sched.run_once(monday)  # same tick again
        assert cb.call_count == 1

    def test_fires_again_next_monday(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(weekly_reset_cb=cb)
        sched.run_once(self._monday(0))
        sched.run_once(self._monday(1))
        assert cb.call_count == 2

    def test_state_updated_after_fire(self):
        sched, clk, _ = make_scheduler(weekly_reset_cb=MagicMock())
        monday = self._monday()
        sched.run_once(monday)
        assert sched.state.last_weekly_reset == monday

    def test_no_callback_does_not_raise(self):
        sched, clk, _ = make_scheduler(weekly_reset_cb=None)
        sched.run_once(self._monday())


# ===========================================================================
# TestRunOnceReturnValue
# ===========================================================================

class TestRunOnceReturnValue:
    def test_empty_list_when_nothing_due(self):
        sched, clk, cb = make_scheduler(candle_interval=300)
        t0 = ts(hour=10, minute=0)
        sched.run_once(t0)  # first call fires candle
        # 1 minute later — nothing due
        events = sched.run_once(t0 + timedelta(seconds=60))
        assert ScheduledEvent.CANDLE_CLOSE not in events

    def test_candle_close_in_list(self):
        sched, clk, cb = make_scheduler()
        events = sched.run_once(ts(hour=10, minute=0))
        assert ScheduledEvent.CANDLE_CLOSE in events

    def test_multiple_events_at_monday_midnight(self):
        daily_open_cb = MagicMock()
        weekly_cb = MagicMock()
        sched, clk, cb = make_scheduler(
            daily_open_cb=daily_open_cb,
            weekly_reset_cb=weekly_cb,
        )
        monday_midnight = datetime(2026, 1, 5, 0, 0, 0)
        events = sched.run_once(monday_midnight)
        assert ScheduledEvent.DAILY_OPEN in events
        assert ScheduledEvent.WEEKLY_RESET in events
        assert ScheduledEvent.CANDLE_CLOSE in events

    def test_daily_open_and_candle_at_midnight(self):
        daily_open_cb = MagicMock()
        sched, clk, cb = make_scheduler(daily_open_cb=daily_open_cb)
        events = sched.run_once(ts(hour=0, minute=0))
        assert ScheduledEvent.DAILY_OPEN in events
        assert ScheduledEvent.CANDLE_CLOSE in events

    def test_daily_report_at_00_05(self):
        report_cb = MagicMock()
        sched, clk, cb = make_scheduler(daily_report_cb=report_cb)
        events = sched.run_once(ts(hour=0, minute=5))
        assert ScheduledEvent.DAILY_REPORT in events

    def test_returns_list_type(self):
        sched, clk, cb = make_scheduler()
        result = sched.run_once(ts())
        assert isinstance(result, list)


# ===========================================================================
# TestCallbackExceptionHandling
# ===========================================================================

class TestCallbackExceptionHandling:
    def test_exception_in_candle_callback_does_not_propagate(self):
        def bad_cb(now):
            raise RuntimeError("pipeline exploded")

        sched = Scheduler(on_candle_close=bad_cb)
        # Should not raise
        sched.run_once(ts(hour=10))

    def test_exception_in_daily_open_does_not_propagate(self):
        def bad_daily(now):
            raise ValueError("universe refresh failed")

        sched = Scheduler(
            on_candle_close=MagicMock(),
            on_daily_open=bad_daily,
        )
        sched.run_once(ts(hour=0, minute=0))

    def test_exception_in_weekly_reset_does_not_propagate(self):
        def bad_weekly(now):
            raise Exception("pnl reset exploded")

        sched = Scheduler(
            on_candle_close=MagicMock(),
            on_weekly_reset=bad_weekly,
        )
        monday = datetime(2026, 1, 6, 0, 0, 0)
        sched.run_once(monday)

    def test_other_callbacks_still_fire_after_exception(self):
        candle_calls = []

        def bad_daily(now):
            raise RuntimeError("boom")

        sched = Scheduler(
            on_candle_close=lambda now: candle_calls.append(now),
            on_daily_open=bad_daily,
        )
        sched.run_once(ts(hour=0, minute=0))
        assert len(candle_calls) == 1


# ===========================================================================
# TestSecondsUntilNextCandle
# ===========================================================================

class TestSecondsUntilNextCandle:
    def test_at_boundary_returns_interval(self):
        sched = Scheduler(on_candle_close=MagicMock(), candle_interval_seconds=300)
        # Exactly at 12:00:00 → next boundary is 12:05:00 → 300s
        now = datetime(2026, 1, 1, 12, 0, 0)
        sleep_s = sched._seconds_until_next_candle(now)
        assert abs(sleep_s - 300.0) < 1.0

    def test_mid_interval_returns_remaining(self):
        sched = Scheduler(on_candle_close=MagicMock(), candle_interval_seconds=300)
        # 12:03:20 → next is 12:05:00 → 100s
        now = datetime(2026, 1, 1, 12, 3, 20)
        sleep_s = sched._seconds_until_next_candle(now)
        assert abs(sleep_s - 100.0) < 1.0

    def test_never_returns_zero(self):
        sched = Scheduler(on_candle_close=MagicMock(), candle_interval_seconds=300)
        # Force a boundary-exact time
        now = datetime(2026, 1, 1, 12, 5, 0)
        sleep_s = sched._seconds_until_next_candle(now)
        assert sleep_s >= 1.0

    def test_custom_interval(self):
        sched = Scheduler(on_candle_close=MagicMock(), candle_interval_seconds=60)
        now = datetime(2026, 1, 1, 12, 0, 30)  # 30s into 1-min interval
        sleep_s = sched._seconds_until_next_candle(now)
        assert abs(sleep_s - 30.0) < 1.0


# ===========================================================================
# TestLoadState
# ===========================================================================

class TestLoadState:
    def test_load_state_prevents_duplicate_daily_open(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(daily_open_cb=cb)
        today = ts(year=2026, month=1, day=6, hour=0, minute=0)
        # Pre-load state as if daily_open already fired today
        sched.load_state(SchedulerState(last_daily_open=today))
        sched.run_once(today)
        cb.assert_not_called()

    def test_load_state_allows_next_day_daily_open(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(daily_open_cb=cb)
        yesterday = ts(year=2026, month=1, day=5, hour=0, minute=0)
        today = ts(year=2026, month=1, day=6, hour=0, minute=0)
        sched.load_state(SchedulerState(last_daily_open=yesterday))
        sched.run_once(today)
        cb.assert_called_once()

    def test_load_state_prevents_weekly_reset_replay(self):
        cb = MagicMock()
        sched, clk, _ = make_scheduler(weekly_reset_cb=cb)
        monday = datetime(2026, 1, 5, 0, 0, 0)
        sched.load_state(SchedulerState(last_weekly_reset=monday))
        sched.run_once(monday)
        cb.assert_not_called()

    def test_load_state_prevents_candle_replay(self):
        sched, clk, cb = make_scheduler(candle_interval=300)
        t = ts(hour=10, minute=0)
        sched.load_state(SchedulerState(last_candle_close=t))
        # 1 minute after loaded state — interval not elapsed
        sched.run_once(t + timedelta(seconds=60))
        cb.assert_not_called()

    def test_state_snapshot_reflects_loaded(self):
        sched, clk, _ = make_scheduler()
        t = ts(hour=0, minute=0)
        state = SchedulerState(last_daily_open=t)
        sched.load_state(state)
        assert sched.state.last_daily_open == t


# ===========================================================================
# TestRunForever
# ===========================================================================

class TestRunForever:
    def test_stop_terminates_loop(self):
        tick_count = [0]
        sleep_calls = []

        def candle_cb(now):
            tick_count[0] += 1
            if tick_count[0] >= 2:
                sched.stop()

        times = [
            ts(hour=10, minute=0),
            ts(hour=10, minute=5),
            ts(hour=10, minute=10),
        ]
        idx = [0]

        def clock():
            t = times[min(idx[0], len(times) - 1)]
            idx[0] += 1
            return t

        sched = Scheduler(
            on_candle_close=candle_cb,
            candle_interval_seconds=300,
            clock=clock,
            sleep_fn=lambda s: sleep_calls.append(s),
        )
        sched.run_forever()
        assert tick_count[0] == 2

    def test_run_forever_calls_sleep(self):
        call_count = [0]
        sleep_calls = []

        # Stop only on 2nd tick — so 1st tick completes and sleep is called
        def candle_cb(now):
            call_count[0] += 1
            if call_count[0] >= 2:
                sched.stop()

        times = [
            ts(hour=10, minute=0),
            ts(hour=10, minute=0),
            ts(hour=10, minute=5),
            ts(hour=10, minute=5),
        ]
        idx = [0]

        def clock():
            t = times[min(idx[0], len(times) - 1)]
            idx[0] += 1
            return t

        sched = Scheduler(
            on_candle_close=candle_cb,
            candle_interval_seconds=300,
            clock=clock,
            sleep_fn=lambda s: sleep_calls.append(s),
        )
        sched.run_forever()
        # Sleep should have been called at least once (after first tick)
        assert len(sleep_calls) >= 1

    def test_exception_in_run_once_does_not_kill_loop(self):
        call_count = [0]

        def bad_candle(now):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first tick blew up")
            if call_count[0] >= 2:
                sched.stop()

        times = [
            ts(hour=10, minute=0),
            ts(hour=10, minute=5),
            ts(hour=10, minute=5),
        ]
        idx = [0]

        def clock():
            t = times[min(idx[0], len(times) - 1)]
            idx[0] += 1
            return t

        sched = Scheduler(
            on_candle_close=bad_candle,
            candle_interval_seconds=300,
            clock=clock,
            sleep_fn=MagicMock(),
        )
        sched.run_forever()
        assert call_count[0] == 2


# ===========================================================================
# TestScheduledEventEnum
# ===========================================================================

class TestScheduledEventEnum:
    def test_all_events_have_string_values(self):
        for evt in ScheduledEvent:
            assert isinstance(evt.value, str)

    def test_event_values(self):
        assert ScheduledEvent.CANDLE_CLOSE.value == "candle_close"
        assert ScheduledEvent.DAILY_OPEN.value == "daily_open"
        assert ScheduledEvent.DAILY_REPORT.value == "daily_report"
        assert ScheduledEvent.WEEKLY_RESET.value == "weekly_reset"


# ===========================================================================
# TestSchedulerStateDataclass
# ===========================================================================

class TestSchedulerStateDataclass:
    def test_all_fields_default_none(self):
        s = SchedulerState()
        assert s.last_candle_close is None
        assert s.last_daily_open is None
        assert s.last_daily_report is None
        assert s.last_weekly_reset is None

    def test_fields_settable(self):
        t = ts()
        s = SchedulerState(last_candle_close=t, last_daily_open=t)
        assert s.last_candle_close == t
        assert s.last_daily_open == t
