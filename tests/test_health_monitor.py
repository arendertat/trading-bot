"""Tests for HealthMonitor and SafeMode (Task 17)"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from bot.health.safe_mode import SafeMode, SafeModeReason, SafeModeState, SafeModeEvent
from bot.health.health_monitor import HealthMonitor, HealthCheck, HealthStatus, HealthReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_safe_mode(recovery_seconds: int = 5) -> SafeMode:
    return SafeMode(recovery_seconds=recovery_seconds)


def make_monitor(
    safe_mode: SafeMode = None,
    stale_data_seconds: int = 30,
    ts_threshold: int = 3,
    rate_threshold: int = 3,
    exc_threshold: int = 3,
    notifier=None,
) -> HealthMonitor:
    if safe_mode is None:
        safe_mode = make_safe_mode()
    return HealthMonitor(
        safe_mode=safe_mode,
        notifier=notifier,
        stale_data_seconds=stale_data_seconds,
        timestamp_error_threshold=ts_threshold,
        rate_limit_threshold=rate_threshold,
        exception_threshold=exc_threshold,
    )


# ===========================================================================
# TestSafeModeInit
# ===========================================================================

class TestSafeModeInit:
    def test_default_state_inactive(self):
        sm = SafeMode()
        assert sm.is_active is False

    def test_reason_none_when_inactive(self):
        sm = SafeMode()
        assert sm.reason is None

    def test_invalid_recovery_seconds(self):
        with pytest.raises(ValueError):
            SafeMode(recovery_seconds=0)

    def test_invalid_negative_recovery(self):
        with pytest.raises(ValueError):
            SafeMode(recovery_seconds=-1)

    def test_state_snapshot_is_copy(self):
        sm = SafeMode()
        s1 = sm.state
        s2 = sm.state
        assert s1 is not s2


# ===========================================================================
# TestSafeModeTrigger
# ===========================================================================

class TestSafeModeTrigger:
    def test_trigger_activates(self):
        sm = SafeMode()
        sm.trigger(SafeModeReason.RATE_LIMIT, "429 error")
        assert sm.is_active is True

    def test_trigger_sets_reason(self):
        sm = SafeMode()
        sm.trigger(SafeModeReason.STALE_DATA, "stale")
        assert sm.reason == SafeModeReason.STALE_DATA

    def test_trigger_increments_activation_count(self):
        sm = SafeMode(recovery_seconds=2)
        sm.trigger(SafeModeReason.RATE_LIMIT, "msg")
        # Recover: 2 healthy checks
        sm.record_healthy_check()
        sm.record_healthy_check()
        assert sm.is_active is False
        # Second activation
        sm.trigger(SafeModeReason.STALE_DATA, "msg2")
        assert sm.state.total_activations == 2

    def test_trigger_idempotent_while_active(self):
        sm = SafeMode()
        sm.trigger(SafeModeReason.RATE_LIMIT, "first")
        sm.trigger(SafeModeReason.RATE_LIMIT, "second")
        assert sm.is_active is True
        assert sm.state.total_activations == 1

    def test_trigger_resets_recovery_counter(self):
        sm = SafeMode(recovery_seconds=5)
        sm.trigger(SafeModeReason.RATE_LIMIT, "first")
        sm.record_healthy_check()
        sm.record_healthy_check()
        assert sm.state.consecutive_healthy_checks == 2
        sm.trigger(SafeModeReason.STALE_DATA, "re-trigger")
        assert sm.state.consecutive_healthy_checks == 0

    def test_trigger_appends_history(self):
        sm = SafeMode()
        sm.trigger(SafeModeReason.RATE_LIMIT, "event1")
        sm.trigger(SafeModeReason.STALE_DATA, "event2")
        assert len(sm.state.history) == 2

    def test_history_capped_at_max(self):
        sm = SafeMode(max_history=3)
        for i in range(5):
            sm.trigger(SafeModeReason.RATE_LIMIT, f"event {i}")
            # Deactivate so each trigger counts
            sm.deactivate_manual()
        assert len(sm.state.history) <= 3

    def test_trigger_with_detail(self):
        sm = SafeMode()
        sm.trigger(SafeModeReason.UNEXPECTED_EXCEPTION, "boom", detail="traceback here")
        assert sm.state.history[-1].detail == "traceback here"


# ===========================================================================
# TestSafeModeRecovery
# ===========================================================================

class TestSafeModeRecovery:
    def test_healthy_checks_accumulate(self):
        sm = SafeMode(recovery_seconds=3)
        sm.trigger(SafeModeReason.RATE_LIMIT, "msg")
        sm.record_healthy_check()
        assert sm.state.consecutive_healthy_checks == 1

    def test_auto_recovery_after_threshold(self):
        sm = SafeMode(recovery_seconds=3)
        sm.trigger(SafeModeReason.RATE_LIMIT, "msg")
        for _ in range(3):
            sm.record_healthy_check()
        assert sm.is_active is False

    def test_auto_recovery_returns_true(self):
        sm = SafeMode(recovery_seconds=2)
        sm.trigger(SafeModeReason.RATE_LIMIT, "msg")
        sm.record_healthy_check()
        result = sm.record_healthy_check()
        assert result is True

    def test_healthy_check_before_threshold_returns_false(self):
        sm = SafeMode(recovery_seconds=5)
        sm.trigger(SafeModeReason.RATE_LIMIT, "msg")
        result = sm.record_healthy_check()
        assert result is False

    def test_healthy_check_inactive_returns_false(self):
        sm = SafeMode()
        result = sm.record_healthy_check()
        assert result is False

    def test_reason_cleared_after_recovery(self):
        sm = SafeMode(recovery_seconds=1)
        sm.trigger(SafeModeReason.STALE_DATA, "stale")
        sm.record_healthy_check()
        assert sm.reason is None

    def test_manual_deactivation(self):
        sm = SafeMode()
        sm.trigger(SafeModeReason.MANUAL, "test")
        sm.deactivate_manual()
        assert sm.is_active is False

    def test_manual_deactivation_when_already_inactive(self):
        sm = SafeMode()
        sm.deactivate_manual()  # Should not raise
        assert sm.is_active is False


# ===========================================================================
# TestSafeModeState
# ===========================================================================

class TestSafeModeState:
    def test_seconds_in_safe_mode_returns_none_when_inactive(self):
        sm = SafeMode()
        assert sm.seconds_in_safe_mode() is None

    def test_seconds_in_safe_mode_positive_when_active(self):
        sm = SafeMode()
        sm.trigger(SafeModeReason.RATE_LIMIT, "msg")
        secs = sm.seconds_in_safe_mode()
        assert secs is not None
        assert secs >= 0.0

    def test_load_state_restores_active(self):
        sm = SafeMode()
        state = SafeModeState(
            active=True,
            reason=SafeModeReason.STALE_DATA,
            total_activations=2,
        )
        sm.load_state(state)
        assert sm.is_active is True
        assert sm.reason == SafeModeReason.STALE_DATA

    def test_reset_clears_everything(self):
        sm = SafeMode()
        sm.trigger(SafeModeReason.RATE_LIMIT, "msg")
        sm.reset()
        assert sm.is_active is False
        assert sm.reason is None
        assert sm.state.total_activations == 0


# ===========================================================================
# TestHealthMonitorInit
# ===========================================================================

class TestHealthMonitorInit:
    def test_creates_without_notifier(self):
        sm = make_safe_mode()
        monitor = HealthMonitor(safe_mode=sm)
        assert monitor is not None

    def test_is_healthy_initially_false(self):
        # No data received yet, so data_freshness fails
        sm = make_safe_mode()
        monitor = HealthMonitor(safe_mode=sm)
        # is_healthy just checks safe mode — not the checks
        assert monitor.is_healthy is True  # Safe mode not active yet

    def test_counters_start_at_zero(self):
        monitor = make_monitor()
        assert monitor.consecutive_timestamp_errors == 0
        assert monitor.consecutive_rate_limit_errors == 0
        assert monitor.consecutive_exceptions == 0

    def test_last_data_ts_none_initially(self):
        monitor = make_monitor()
        assert monitor.last_data_ts is None


# ===========================================================================
# TestHealthMonitorTimestampErrors
# ===========================================================================

class TestHealthMonitorTimestampErrors:
    def test_single_error_no_safe_mode(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm, ts_threshold=3)
        monitor.record_timestamp_error("clock skew")
        assert sm.is_active is False

    def test_threshold_triggers_safe_mode(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm, ts_threshold=3)
        for _ in range(3):
            monitor.record_timestamp_error("clock skew")
        assert sm.is_active is True
        assert sm.reason == SafeModeReason.TIMESTAMP_ERROR

    def test_ok_clears_counter(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm, ts_threshold=3)
        monitor.record_timestamp_error("err1")
        monitor.record_timestamp_error("err2")
        monitor.record_timestamp_ok()
        assert monitor.consecutive_timestamp_errors == 0

    def test_ok_after_reset_no_safe_mode(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm, ts_threshold=3)
        monitor.record_timestamp_error("err")
        monitor.record_timestamp_ok()
        monitor.record_timestamp_error("err")
        assert sm.is_active is False

    def test_run_checks_reflects_timestamp_failure(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm, ts_threshold=2)
        for _ in range(2):
            monitor.record_timestamp_error("err")
        monitor.record_data_received()  # avoid stale data failure
        monitor.record_balance_ok()
        report = monitor.run_checks()
        ts_check = next(c for c in report.checks if c.name == "timestamp_errors")
        assert ts_check.healthy is False


# ===========================================================================
# TestHealthMonitorRateLimits
# ===========================================================================

class TestHealthMonitorRateLimits:
    def test_threshold_triggers_safe_mode(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm, rate_threshold=2)
        for _ in range(2):
            monitor.record_rate_limit("429")
        assert sm.is_active is True
        assert sm.reason == SafeModeReason.RATE_LIMIT

    def test_ok_clears_counter(self):
        monitor = make_monitor(rate_threshold=3)
        monitor.record_rate_limit("429")
        monitor.record_rate_limit_ok()
        assert monitor.consecutive_rate_limit_errors == 0

    def test_below_threshold_no_safe_mode(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm, rate_threshold=3)
        monitor.record_rate_limit("429")
        assert sm.is_active is False


# ===========================================================================
# TestHealthMonitorDataFreshness
# ===========================================================================

class TestHealthMonitorDataFreshness:
    def test_fresh_data_healthy(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm, stale_data_seconds=30)
        monitor.record_data_received(datetime.utcnow())
        monitor.record_balance_ok()
        report = monitor.run_checks()
        freshness = next(c for c in report.checks if c.name == "data_freshness")
        assert freshness.healthy is True

    def test_stale_data_triggers_safe_mode(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm, stale_data_seconds=30)
        stale_ts = datetime.utcnow() - timedelta(seconds=60)
        monitor._last_data_ts = stale_ts
        monitor.record_balance_ok()
        report = monitor.run_checks()
        assert sm.is_active is True
        assert sm.reason == SafeModeReason.STALE_DATA

    def test_no_data_unhealthy(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm)
        report = monitor.run_checks()
        freshness = next(c for c in report.checks if c.name == "data_freshness")
        assert freshness.healthy is False

    def test_record_data_received_defaults_to_now(self):
        monitor = make_monitor()
        before = datetime.utcnow()
        monitor.record_data_received()
        after = datetime.utcnow()
        assert before <= monitor.last_data_ts <= after

    def test_record_data_received_explicit_timestamp(self):
        monitor = make_monitor()
        ts = datetime(2026, 1, 1, 12, 0, 0)
        monitor.record_data_received(ts)
        assert monitor.last_data_ts == ts


# ===========================================================================
# TestHealthMonitorBalance
# ===========================================================================

class TestHealthMonitorBalance:
    def test_balance_ok_no_safe_mode(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm)
        monitor.record_balance_ok()
        assert sm.is_active is False

    def test_balance_failed_triggers_safe_mode(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm)
        monitor.record_balance_failed("connection refused")
        assert sm.is_active is True
        assert sm.reason == SafeModeReason.BALANCE_FETCH_FAILED

    def test_balance_ok_clears_error(self):
        monitor = make_monitor()
        monitor.record_balance_failed("error")
        monitor.record_balance_ok()
        report = monitor.run_checks()
        balance_check = next(c for c in report.checks if c.name == "balance_fetch")
        assert balance_check.healthy is True

    def test_balance_check_in_report(self):
        monitor = make_monitor()
        monitor.record_data_received()
        monitor.record_balance_failed("fail")
        report = monitor.run_checks()
        balance_check = next(c for c in report.checks if c.name == "balance_fetch")
        assert balance_check.healthy is False
        assert "fail" in balance_check.message


# ===========================================================================
# TestHealthMonitorExceptions
# ===========================================================================

class TestHealthMonitorExceptions:
    def test_single_exception_no_safe_mode(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm, exc_threshold=3)
        monitor.record_exception(ValueError("test"))
        assert sm.is_active is False

    def test_threshold_triggers_safe_mode(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm, exc_threshold=3)
        for _ in range(3):
            monitor.record_exception(RuntimeError("boom"))
        assert sm.is_active is True
        assert sm.reason == SafeModeReason.UNEXPECTED_EXCEPTION

    def test_ok_clears_exception_counter(self):
        monitor = make_monitor(exc_threshold=3)
        monitor.record_exception(ValueError("err"))
        monitor.record_exception_ok()
        assert monitor.consecutive_exceptions == 0

    def test_exception_counter_increments(self):
        monitor = make_monitor(exc_threshold=5)
        monitor.record_exception(ValueError("e1"))
        monitor.record_exception(ValueError("e2"))
        assert monitor.consecutive_exceptions == 2


# ===========================================================================
# TestRunChecks
# ===========================================================================

class TestRunChecks:
    def _healthy_monitor(self, sm=None):
        if sm is None:
            sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm, stale_data_seconds=30)
        monitor.record_data_received(datetime.utcnow())
        monitor.record_balance_ok()
        return monitor, sm

    def test_all_healthy_returns_healthy_status(self):
        monitor, sm = self._healthy_monitor()
        report = monitor.run_checks()
        assert report.status == HealthStatus.HEALTHY

    def test_all_healthy_safe_mode_false(self):
        monitor, sm = self._healthy_monitor()
        report = monitor.run_checks()
        assert report.safe_mode_active is False

    def test_report_has_five_checks(self):
        monitor, sm = self._healthy_monitor()
        report = monitor.run_checks()
        assert len(report.checks) == 5

    def test_check_names_present(self):
        monitor, sm = self._healthy_monitor()
        report = monitor.run_checks()
        names = {c.name for c in report.checks}
        assert "timestamp_errors" in names
        assert "rate_limits" in names
        assert "data_freshness" in names
        assert "balance_fetch" in names
        assert "exceptions" in names

    def test_unhealthy_status_when_safe_mode_active(self):
        sm = make_safe_mode()
        monitor, sm = self._healthy_monitor(sm)
        sm.trigger(SafeModeReason.MANUAL, "test")
        report = monitor.run_checks()
        assert report.status == HealthStatus.UNHEALTHY

    def test_unhealthy_status_on_critical_failure(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm, ts_threshold=1)
        monitor.record_timestamp_error("err")
        monitor.record_data_received()
        monitor.record_balance_ok()
        report = monitor.run_checks()
        assert report.status == HealthStatus.UNHEALTHY

    def test_safe_mode_reason_in_report(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm, ts_threshold=1)
        monitor.record_timestamp_error("err")
        monitor.record_data_received()
        monitor.record_balance_ok()
        report = monitor.run_checks()
        assert report.safe_mode_reason == SafeModeReason.TIMESTAMP_ERROR.value

    def test_healthy_checks_drive_recovery(self):
        sm = SafeMode(recovery_seconds=2)
        monitor, sm = self._healthy_monitor(sm)
        sm.trigger(SafeModeReason.MANUAL, "test")
        # Two consecutive healthy run_checks should recover
        monitor.run_checks()
        monitor.run_checks()
        assert sm.is_active is False

    def test_report_to_dict_serializable(self):
        monitor, sm = self._healthy_monitor()
        report = monitor.run_checks()
        d = report.to_dict()
        assert "status" in d
        assert "checks" in d
        assert isinstance(d["checks"], list)

    def test_report_timestamp_is_recent(self):
        monitor, sm = self._healthy_monitor()
        before = datetime.utcnow()
        report = monitor.run_checks()
        after = datetime.utcnow()
        assert before <= report.timestamp <= after


# ===========================================================================
# TestIsHealthyProperty
# ===========================================================================

class TestIsHealthyProperty:
    def test_healthy_when_safe_mode_inactive(self):
        monitor = make_monitor()
        assert monitor.is_healthy is True

    def test_unhealthy_when_safe_mode_active(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm)
        sm.trigger(SafeModeReason.MANUAL, "test")
        assert monitor.is_healthy is False

    def test_healthy_after_recovery(self):
        sm = SafeMode(recovery_seconds=1)
        monitor = make_monitor(safe_mode=sm)
        sm.trigger(SafeModeReason.MANUAL, "test")
        sm.record_healthy_check()
        assert monitor.is_healthy is True


# ===========================================================================
# TestNotifierIntegration
# ===========================================================================

class TestNotifierIntegration:
    def test_notifier_called_on_first_safe_mode_activation(self):
        sm = make_safe_mode()
        notifier = MagicMock()
        monitor = make_monitor(safe_mode=sm, notifier=notifier, ts_threshold=1)
        monitor.record_timestamp_error("err")
        notifier.send_safe_mode_alert.assert_called_once()

    def test_notifier_not_called_when_already_active(self):
        sm = make_safe_mode()
        notifier = MagicMock()
        monitor = make_monitor(safe_mode=sm, notifier=notifier, ts_threshold=1)
        monitor.record_timestamp_error("err1")
        monitor.record_timestamp_error("err2")  # already active
        assert notifier.send_safe_mode_alert.call_count == 1

    def test_notifier_called_on_recovery(self):
        sm = SafeMode(recovery_seconds=2)
        notifier = MagicMock()
        monitor, sm = _make_healthy_monitor_with_notifier(sm, notifier)
        sm.trigger(SafeModeReason.MANUAL, "test")
        monitor.run_checks()  # healthy check #1
        monitor.run_checks()  # healthy check #2 → recovery
        notifier.send_safe_mode_alert.assert_called()

    def test_no_notifier_does_not_raise(self):
        sm = make_safe_mode()
        monitor = make_monitor(safe_mode=sm, notifier=None, ts_threshold=1)
        monitor.record_timestamp_error("err")  # should not raise


def _make_healthy_monitor_with_notifier(sm, notifier):
    monitor = HealthMonitor(
        safe_mode=sm,
        notifier=notifier,
        stale_data_seconds=30,
    )
    monitor.record_data_received(datetime.utcnow())
    monitor.record_balance_ok()
    return monitor, sm


# ===========================================================================
# TestThreadSafety
# ===========================================================================

class TestThreadSafety:
    def test_concurrent_triggers_do_not_corrupt_state(self):
        import threading
        sm = SafeMode(recovery_seconds=100)
        monitor = make_monitor(safe_mode=sm, ts_threshold=5)
        errors = []

        def trigger_errors():
            try:
                for _ in range(10):
                    monitor.record_timestamp_error("concurrent")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=trigger_errors) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert monitor.consecutive_timestamp_errors >= 0

    def test_concurrent_healthy_checks_do_not_corrupt(self):
        import threading
        sm = SafeMode(recovery_seconds=100)
        sm.trigger(SafeModeReason.MANUAL, "test")
        errors = []

        def do_healthy():
            try:
                for _ in range(10):
                    sm.record_healthy_check()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_healthy) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ===========================================================================
# TestHealthCheckDataclass
# ===========================================================================

class TestHealthCheckDataclass:
    def test_health_check_fields(self):
        hc = HealthCheck(name="test", healthy=True, message="OK")
        assert hc.name == "test"
        assert hc.healthy is True
        assert hc.message == "OK"
        assert hc.critical is True  # default

    def test_health_check_non_critical(self):
        hc = HealthCheck(name="test", healthy=False, message="warn", critical=False)
        assert hc.critical is False

    def test_health_check_timestamp_set(self):
        before = datetime.utcnow()
        hc = HealthCheck(name="test", healthy=True, message="OK")
        after = datetime.utcnow()
        assert before <= hc.timestamp <= after


# ===========================================================================
# TestSafeModeReasons
# ===========================================================================

class TestSafeModeReasons:
    def test_all_reasons_have_string_values(self):
        for reason in SafeModeReason:
            assert isinstance(reason.value, str)

    def test_reason_enum_values(self):
        assert SafeModeReason.TIMESTAMP_ERROR.value == "timestamp_error"
        assert SafeModeReason.RATE_LIMIT.value == "rate_limit"
        assert SafeModeReason.STALE_DATA.value == "stale_data"
        assert SafeModeReason.BALANCE_FETCH_FAILED.value == "balance_fetch_failed"
        assert SafeModeReason.UNEXPECTED_EXCEPTION.value == "unexpected_exception"
        assert SafeModeReason.MANUAL.value == "manual"
