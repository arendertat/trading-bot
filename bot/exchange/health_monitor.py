"""Exchange health monitoring"""

import time
import logging
from typing import Optional
from dataclasses import dataclass


logger = logging.getLogger("trading_bot.exchange")


@dataclass
class HealthStatus:
    """Health status snapshot"""
    is_healthy: bool
    last_success_ts: Optional[float]
    consecutive_errors: int
    last_error: Optional[str]
    last_error_ts: Optional[float]


class HealthMonitor:
    """
    Monitor exchange connectivity health.

    Tracks:
    - Last successful request timestamp
    - Consecutive error count
    - Error types and timestamps

    Simple rules for Milestone 2 (SAFE_MODE comes later).
    """

    def __init__(self, error_threshold: int = 5):
        """
        Initialize HealthMonitor.

        Args:
            error_threshold: Number of consecutive errors before marking unhealthy
        """
        self.error_threshold = error_threshold

        self.last_success_ts: Optional[float] = None
        self.consecutive_errors: int = 0
        self.last_error: Optional[str] = None
        self.last_error_ts: Optional[float] = None

        self._total_requests: int = 0
        self._total_errors: int = 0

    def record_success(self) -> None:
        """Record a successful request"""
        self.last_success_ts = time.time()
        self.consecutive_errors = 0
        self._total_requests += 1

        logger.debug("Exchange request successful")

    def record_failure(self, exception: Exception) -> None:
        """
        Record a failed request.

        Args:
            exception: Exception that occurred
        """
        self.consecutive_errors += 1
        self.last_error = str(exception)
        self.last_error_ts = time.time()
        self._total_requests += 1
        self._total_errors += 1

        logger.warning(
            f"Exchange request failed (consecutive: {self.consecutive_errors}): {exception}"
        )

    def is_healthy(self) -> bool:
        """
        Check if exchange connection is healthy.

        Returns:
            True if healthy (consecutive errors below threshold)
        """
        return self.consecutive_errors < self.error_threshold

    def get_status(self) -> HealthStatus:
        """
        Get current health status snapshot.

        Returns:
            HealthStatus instance
        """
        return HealthStatus(
            is_healthy=self.is_healthy(),
            last_success_ts=self.last_success_ts,
            consecutive_errors=self.consecutive_errors,
            last_error=self.last_error,
            last_error_ts=self.last_error_ts,
        )

    def get_error_rate(self) -> float:
        """
        Calculate error rate.

        Returns:
            Error rate (0.0 to 1.0)
        """
        if self._total_requests == 0:
            return 0.0

        return self._total_errors / self._total_requests

    def reset(self) -> None:
        """Reset health monitor state"""
        self.last_success_ts = None
        self.consecutive_errors = 0
        self.last_error = None
        self.last_error_ts = None
        self._total_requests = 0
        self._total_errors = 0

        logger.info("Health monitor reset")
