"""Health monitor — detects unhealthy conditions and drives safe mode"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import RLock
from typing import Callable, Dict, List, Optional

from bot.health.safe_mode import SafeMode, SafeModeReason
from bot.reporting.notifier import TelegramNotifier

logger = logging.getLogger("trading_bot.health.monitor")


class HealthStatus(str, Enum):
    """Overall health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"     # One non-critical check failing
    UNHEALTHY = "unhealthy"   # Critical check failing or safe mode active


@dataclass
class HealthCheck:
    """Result of a single named health check"""
    name: str
    healthy: bool
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    critical: bool = True     # Critical checks trigger safe mode


@dataclass
class HealthReport:
    """Aggregated health report"""
    status: HealthStatus
    checks: List[HealthCheck]
    safe_mode_active: bool
    safe_mode_reason: Optional[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "safe_mode_active": self.safe_mode_active,
            "safe_mode_reason": self.safe_mode_reason,
            "timestamp": self.timestamp.isoformat(),
            "checks": [
                {
                    "name": c.name,
                    "healthy": c.healthy,
                    "message": c.message,
                    "critical": c.critical,
                    "timestamp": c.timestamp.isoformat(),
                }
                for c in self.checks
            ],
        }


class HealthMonitor:
    """
    Health monitor for the trading bot.

    Tracks five health dimensions:

    1. **Exchange timestamp** — Binance clock-skew / recv_window errors.
       After *timestamp_error_threshold* consecutive errors, triggers safe mode.

    2. **Rate limits** — Exchange 429 / rate-limit responses.
       After *rate_limit_threshold* consecutive errors, triggers safe mode.

    3. **WebSocket data freshness** — Last data timestamp must be within
       *stale_data_seconds* of now.

    4. **Balance fetch** — Whether the last balance fetch succeeded.

    5. **Unexpected exceptions** — Unhandled exceptions in the main loop.
       After *exception_threshold* consecutive exceptions, triggers safe mode.

    The monitor exposes:
    - ``run_checks()`` → HealthReport   (call once per main-loop cycle)
    - ``record_*()`` methods for each signal
    - ``is_healthy`` property

    Thread-safe via internal RLock.

    Usage::

        monitor = HealthMonitor(safe_mode=sm, notifier=notifier)

        # On every main loop cycle:
        monitor.record_data_received(last_ws_timestamp)
        report = monitor.run_checks()

        # On exchange error:
        monitor.record_timestamp_error("recv_window exceeded")
        monitor.record_rate_limit("429 response")

        # On successful balance fetch:
        monitor.record_balance_ok()

        # On unhandled exception:
        monitor.record_exception(exc)
    """

    def __init__(
        self,
        safe_mode: SafeMode,
        notifier: Optional[TelegramNotifier] = None,
        stale_data_seconds: int = 30,
        timestamp_error_threshold: int = 3,
        rate_limit_threshold: int = 3,
        exception_threshold: int = 3,
    ) -> None:
        """
        Initialise HealthMonitor.

        Args:
            safe_mode: SafeMode instance to drive.
            notifier: Optional Telegram notifier for alerts.
            stale_data_seconds: Max seconds since last WS data before stale.
            timestamp_error_threshold: Consecutive timestamp errors before safe mode.
            rate_limit_threshold: Consecutive rate-limit errors before safe mode.
            exception_threshold: Consecutive unhandled exceptions before safe mode.
        """
        self._safe_mode = safe_mode
        self._notifier = notifier
        self._stale_data_seconds = stale_data_seconds
        self._timestamp_error_threshold = timestamp_error_threshold
        self._rate_limit_threshold = rate_limit_threshold
        self._exception_threshold = exception_threshold

        self._lock = RLock()

        # Counters / timestamps
        self._consecutive_timestamp_errors: int = 0
        self._consecutive_rate_limit_errors: int = 0
        self._consecutive_exceptions: int = 0
        self._last_data_ts: Optional[datetime] = None
        self._balance_ok: bool = True
        self._last_balance_error: Optional[str] = None

        logger.info(
            f"HealthMonitor initialised: "
            f"stale_data={stale_data_seconds}s, "
            f"ts_err_threshold={timestamp_error_threshold}, "
            f"rate_limit_threshold={rate_limit_threshold}, "
            f"exception_threshold={exception_threshold}"
        )

    # ------------------------------------------------------------------
    # Signal recording methods (called by other components)
    # ------------------------------------------------------------------

    def record_timestamp_error(self, message: str = "") -> None:
        """
        Record a Binance timestamp / recv_window error.

        Args:
            message: Error description for logging.
        """
        with self._lock:
            self._consecutive_timestamp_errors += 1
            logger.warning(
                f"Timestamp error #{self._consecutive_timestamp_errors}: {message}"
            )
            if self._consecutive_timestamp_errors >= self._timestamp_error_threshold:
                self._trigger_safe_mode(
                    SafeModeReason.TIMESTAMP_ERROR,
                    f"Persistent timestamp errors ({self._consecutive_timestamp_errors}): {message}",
                )

    def record_timestamp_ok(self) -> None:
        """Reset timestamp error counter on a successful exchange call."""
        with self._lock:
            if self._consecutive_timestamp_errors > 0:
                logger.debug("Timestamp errors cleared")
            self._consecutive_timestamp_errors = 0

    def record_rate_limit(self, message: str = "") -> None:
        """
        Record an exchange rate-limit (429) response.

        Args:
            message: Error description.
        """
        with self._lock:
            self._consecutive_rate_limit_errors += 1
            logger.warning(
                f"Rate limit #{self._consecutive_rate_limit_errors}: {message}"
            )
            if self._consecutive_rate_limit_errors >= self._rate_limit_threshold:
                self._trigger_safe_mode(
                    SafeModeReason.RATE_LIMIT,
                    f"Repeated rate-limit errors ({self._consecutive_rate_limit_errors}): {message}",
                )

    def record_rate_limit_ok(self) -> None:
        """Reset rate-limit counter after a successful request."""
        with self._lock:
            if self._consecutive_rate_limit_errors > 0:
                logger.debug("Rate-limit errors cleared")
            self._consecutive_rate_limit_errors = 0

    def record_data_received(self, ts: Optional[datetime] = None) -> None:
        """
        Record that fresh market data was received.

        Args:
            ts: Timestamp of the data (defaults to utcnow).
        """
        with self._lock:
            self._last_data_ts = ts or datetime.utcnow()

    def record_balance_ok(self) -> None:
        """Record a successful balance fetch."""
        with self._lock:
            self._balance_ok = True
            self._last_balance_error = None

    def record_balance_failed(self, message: str = "") -> None:
        """
        Record a failed balance fetch; triggers safe mode immediately.

        Args:
            message: Error description.
        """
        with self._lock:
            self._balance_ok = False
            self._last_balance_error = message
            self._trigger_safe_mode(
                SafeModeReason.BALANCE_FETCH_FAILED,
                f"Balance fetch failed: {message}",
            )

    def record_exception(self, exc: Exception) -> None:
        """
        Record an unhandled exception in the main loop.

        Args:
            exc: The exception that was caught.
        """
        with self._lock:
            self._consecutive_exceptions += 1
            logger.error(
                f"Unexpected exception #{self._consecutive_exceptions}: {exc}"
            )
            if self._consecutive_exceptions >= self._exception_threshold:
                self._trigger_safe_mode(
                    SafeModeReason.UNEXPECTED_EXCEPTION,
                    f"Repeated exceptions ({self._consecutive_exceptions}): {exc}",
                    detail=str(exc),
                )

    def record_exception_ok(self) -> None:
        """Reset exception counter after a clean main-loop cycle."""
        with self._lock:
            if self._consecutive_exceptions > 0:
                logger.debug("Exception counter cleared")
            self._consecutive_exceptions = 0

    # ------------------------------------------------------------------
    # Core check runner
    # ------------------------------------------------------------------

    def run_checks(self) -> HealthReport:
        """
        Run all health checks and update safe mode accordingly.

        Should be called once per main-loop cycle (e.g. every 5 minutes).

        Returns:
            HealthReport with per-check results and overall status.
        """
        with self._lock:
            checks: List[HealthCheck] = []

            # 1. Timestamp error check
            checks.append(self._check_timestamp_errors())

            # 2. Rate-limit check
            checks.append(self._check_rate_limits())

            # 3. Data freshness check
            checks.append(self._check_data_freshness())

            # 4. Balance check
            checks.append(self._check_balance())

            # 5. Exception check
            checks.append(self._check_exceptions())

            # Determine whether to record healthy or trigger safe mode
            critical_failures = [c for c in checks if not c.healthy and c.critical]
            non_critical_failures = [c for c in checks if not c.healthy and not c.critical]

            if not critical_failures:
                # All critical checks passed — record healthy check
                deactivated = self._safe_mode.record_healthy_check()
                if deactivated:
                    logger.info("HealthMonitor: safe mode auto-recovered")
                    if self._notifier:
                        self._notifier.send_safe_mode_alert(
                            reason="Auto-recovered: all health checks passing",
                        )

            # Compute overall status
            if self._safe_mode.is_active or critical_failures:
                status = HealthStatus.UNHEALTHY
            elif non_critical_failures:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            report = HealthReport(
                status=status,
                checks=checks,
                safe_mode_active=self._safe_mode.is_active,
                safe_mode_reason=(
                    self._safe_mode.reason.value
                    if self._safe_mode.reason
                    else None
                ),
            )

            if status != HealthStatus.HEALTHY:
                logger.warning(
                    f"Health status: {status.value} — "
                    f"critical_failures={[c.name for c in critical_failures]}"
                )

            return report

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        """True when no critical checks are failing and safe mode is off."""
        return not self._safe_mode.is_active

    @property
    def consecutive_timestamp_errors(self) -> int:
        with self._lock:
            return self._consecutive_timestamp_errors

    @property
    def consecutive_rate_limit_errors(self) -> int:
        with self._lock:
            return self._consecutive_rate_limit_errors

    @property
    def consecutive_exceptions(self) -> int:
        with self._lock:
            return self._consecutive_exceptions

    @property
    def last_data_ts(self) -> Optional[datetime]:
        with self._lock:
            return self._last_data_ts

    # ------------------------------------------------------------------
    # Private check implementations
    # ------------------------------------------------------------------

    def _check_timestamp_errors(self) -> HealthCheck:
        count = self._consecutive_timestamp_errors
        healthy = count < self._timestamp_error_threshold
        return HealthCheck(
            name="timestamp_errors",
            healthy=healthy,
            message=(
                "OK"
                if healthy
                else f"{count} consecutive timestamp errors (threshold={self._timestamp_error_threshold})"
            ),
            critical=True,
        )

    def _check_rate_limits(self) -> HealthCheck:
        count = self._consecutive_rate_limit_errors
        healthy = count < self._rate_limit_threshold
        return HealthCheck(
            name="rate_limits",
            healthy=healthy,
            message=(
                "OK"
                if healthy
                else f"{count} consecutive rate-limit errors (threshold={self._rate_limit_threshold})"
            ),
            critical=True,
        )

    def _check_data_freshness(self) -> HealthCheck:
        if self._last_data_ts is None:
            # No data received yet — treat as unhealthy only if we've been
            # running long enough to expect data.
            return HealthCheck(
                name="data_freshness",
                healthy=False,
                message="No market data received yet",
                critical=True,
            )

        age_seconds = (datetime.utcnow() - self._last_data_ts).total_seconds()
        healthy = age_seconds <= self._stale_data_seconds

        if not healthy:
            self._trigger_safe_mode(
                SafeModeReason.STALE_DATA,
                f"WebSocket data stale: {age_seconds:.0f}s old (threshold={self._stale_data_seconds}s)",
            )

        return HealthCheck(
            name="data_freshness",
            healthy=healthy,
            message=(
                f"OK (data age={age_seconds:.0f}s)"
                if healthy
                else f"STALE: data age={age_seconds:.0f}s (threshold={self._stale_data_seconds}s)"
            ),
            critical=True,
        )

    def _check_balance(self) -> HealthCheck:
        return HealthCheck(
            name="balance_fetch",
            healthy=self._balance_ok,
            message=(
                "OK"
                if self._balance_ok
                else f"Balance fetch failed: {self._last_balance_error}"
            ),
            critical=True,
        )

    def _check_exceptions(self) -> HealthCheck:
        count = self._consecutive_exceptions
        healthy = count < self._exception_threshold
        return HealthCheck(
            name="exceptions",
            healthy=healthy,
            message=(
                "OK"
                if healthy
                else f"{count} consecutive exceptions (threshold={self._exception_threshold})"
            ),
            critical=True,
        )

    def _trigger_safe_mode(
        self,
        reason: SafeModeReason,
        message: str,
        detail: Optional[str] = None,
    ) -> None:
        """Trigger safe mode and optionally send Telegram alert."""
        was_active = self._safe_mode.is_active
        self._safe_mode.trigger(reason, message, detail=detail)

        if not was_active and self._notifier:
            # Only alert on first activation to avoid spam
            self._notifier.send_safe_mode_alert(reason=message)
