"""Safe mode state machine — blocks new entries on unhealthy conditions"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import RLock
from typing import List, Optional

logger = logging.getLogger("trading_bot.health.safe_mode")


class SafeModeReason(str, Enum):
    """Reasons that can trigger safe mode"""
    TIMESTAMP_ERROR = "timestamp_error"          # Binance recv_window / clock skew
    RATE_LIMIT = "rate_limit"                    # Exchange rate limit hit
    STALE_DATA = "stale_data"                    # WebSocket data older than threshold
    BALANCE_FETCH_FAILED = "balance_fetch_failed"  # Cannot fetch account balance
    UNEXPECTED_EXCEPTION = "unexpected_exception"  # Unhandled exception in main loop
    MANUAL = "manual"                            # Operator-triggered


@dataclass
class SafeModeEvent:
    """A single safe-mode trigger event"""
    reason: SafeModeReason
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    detail: Optional[str] = None


@dataclass
class SafeModeState:
    """Current safe-mode state — serialisable for recovery"""
    active: bool = False
    reason: Optional[SafeModeReason] = None
    activated_at: Optional[datetime] = None
    last_healthy_at: Optional[datetime] = None
    consecutive_healthy_checks: int = 0
    total_activations: int = 0
    history: List[SafeModeEvent] = field(default_factory=list)


class SafeMode:
    """
    Safe-mode manager.

    Safe mode blocks new position entries while allowing existing positions
    to continue running.  It activates on any of the SafeModeReason triggers
    and auto-recovers after *recovery_seconds* of consecutive healthy checks.

    Thread-safe: all public methods are protected by an internal RLock.

    Usage::

        sm = SafeMode(recovery_seconds=60, max_history=50)

        # Trigger from exchange error handler
        sm.trigger(SafeModeReason.RATE_LIMIT, "429 Too Many Requests")

        # Call every health-check cycle
        sm.record_healthy_check()

        # Gate new entries
        if sm.is_active:
            return  # skip new entries
    """

    def __init__(
        self,
        recovery_seconds: int = 60,
        max_history: int = 100,
    ) -> None:
        """
        Initialise SafeMode.

        Args:
            recovery_seconds: Consecutive seconds of healthy checks required
                              before auto-recovery.  Each healthy check counts
                              as one second for simplicity; callers should invoke
                              record_healthy_check() on each health-check cycle.
            max_history: Maximum number of trigger events to keep in history.
        """
        if recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be positive")

        self._recovery_seconds = recovery_seconds
        self._max_history = max_history
        self._lock = RLock()
        self._state = SafeModeState()

        logger.info(
            f"SafeMode initialised: recovery_seconds={recovery_seconds}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True when safe mode is currently active (new entries blocked)."""
        with self._lock:
            return self._state.active

    @property
    def reason(self) -> Optional[SafeModeReason]:
        """The most recent trigger reason, or None if not active."""
        with self._lock:
            return self._state.reason

    @property
    def state(self) -> SafeModeState:
        """Return a snapshot of the current state (not a live reference)."""
        with self._lock:
            # Return a shallow copy so callers cannot mutate internal state
            s = self._state
            return SafeModeState(
                active=s.active,
                reason=s.reason,
                activated_at=s.activated_at,
                last_healthy_at=s.last_healthy_at,
                consecutive_healthy_checks=s.consecutive_healthy_checks,
                total_activations=s.total_activations,
                history=list(s.history),
            )

    def trigger(
        self,
        reason: SafeModeReason,
        message: str,
        detail: Optional[str] = None,
    ) -> None:
        """
        Activate safe mode.

        Idempotent: calling while already active only appends to history and
        resets the recovery counter (extends safe mode duration).

        Args:
            reason: Why safe mode is being triggered.
            message: Human-readable description.
            detail: Optional extra context (e.g. exception traceback).
        """
        with self._lock:
            event = SafeModeEvent(
                reason=reason,
                message=message,
                detail=detail,
            )
            self._append_history(event)

            if not self._state.active:
                self._state.active = True
                self._state.reason = reason
                self._state.activated_at = datetime.utcnow()
                self._state.total_activations += 1
                logger.warning(
                    f"SAFE MODE ACTIVATED — reason={reason.value}: {message}"
                )
            else:
                # Already active — reset recovery counter
                self._state.consecutive_healthy_checks = 0
                logger.warning(
                    f"Safe mode re-triggered — reason={reason.value}: {message}"
                )

    def record_healthy_check(self) -> bool:
        """
        Record a successful health check.

        If safe mode is active and enough consecutive healthy checks have
        accumulated, auto-recovery is triggered.

        Returns:
            True if safe mode was deactivated by this call, False otherwise.
        """
        with self._lock:
            self._state.last_healthy_at = datetime.utcnow()

            if not self._state.active:
                self._state.consecutive_healthy_checks += 1
                return False

            self._state.consecutive_healthy_checks += 1

            if self._state.consecutive_healthy_checks >= self._recovery_seconds:
                self._deactivate(reason="auto-recovery after healthy checks")
                return True

            remaining = self._recovery_seconds - self._state.consecutive_healthy_checks
            logger.debug(
                f"Safe mode: healthy check #{self._state.consecutive_healthy_checks}"
                f" ({remaining} more needed for recovery)"
            )
            return False

    def deactivate_manual(self, reason: str = "manual operator reset") -> None:
        """
        Manually deactivate safe mode (operator command).

        Args:
            reason: Why it's being manually cleared.
        """
        with self._lock:
            if self._state.active:
                self._deactivate(reason=reason)
            else:
                logger.info("Safe mode deactivate_manual called but already inactive")

    def seconds_in_safe_mode(self) -> Optional[float]:
        """
        Return how long safe mode has been active, or None if not active.
        """
        with self._lock:
            if not self._state.active or self._state.activated_at is None:
                return None
            return (datetime.utcnow() - self._state.activated_at).total_seconds()

    def load_state(self, state: SafeModeState) -> None:
        """
        Restore state after bot restart.

        Args:
            state: Previously persisted SafeModeState.
        """
        with self._lock:
            self._state = state
            logger.info(
                f"SafeMode state loaded: active={state.active}, "
                f"reason={state.reason}"
            )

    def reset(self) -> None:
        """Hard reset — clears all state. Use with caution."""
        with self._lock:
            logger.warning("SafeMode: hard reset")
            self._state = SafeModeState()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _deactivate(self, reason: str) -> None:
        """Internal deactivation — must hold lock."""
        logger.warning(
            f"SAFE MODE DEACTIVATED — {reason} "
            f"(was active for "
            f"{self._state.activated_at and (datetime.utcnow() - self._state.activated_at).total_seconds():.0f}s)"
        )
        self._state.active = False
        self._state.reason = None
        self._state.activated_at = None
        self._state.consecutive_healthy_checks = 0

    def _append_history(self, event: SafeModeEvent) -> None:
        """Append to history, evicting oldest if over max_history."""
        self._state.history.append(event)
        if len(self._state.history) > self._max_history:
            self._state.history = self._state.history[-self._max_history:]
