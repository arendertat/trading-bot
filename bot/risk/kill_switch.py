"""Kill switch implementation for daily and weekly stop losses"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from bot.config.models import RiskConfig
from bot.core.types import KillSwitchState

logger = logging.getLogger("trading_bot.risk.kill_switch")


class KillSwitch:
    """
    Manages daily and weekly stop loss kill switches.

    Daily stop: Blocks new entries until next UTC day if daily loss exceeds threshold.
    Weekly stop: Pauses trading for N days if weekly loss exceeds threshold,
                 then reduces risk for subsequent days.
    """

    def __init__(self, config: RiskConfig):
        """
        Initialize kill switch manager.

        Args:
            config: Risk configuration
        """
        self.config = config
        self.state = KillSwitchState()

        logger.info(
            f"KillSwitch initialized: "
            f"daily_stop={self.config.daily_stop_pct:.2%}, "
            f"weekly_stop={self.config.weekly_stop_pct:.2%}, "
            f"pause_days={self.config.pause_days_after_weekly_stop}, "
            f"reduced_risk={self.config.reduced_risk_after_pause_pct:.2%}"
        )

    def update_pnl(
        self,
        realized_pnl_today: float,
        realized_pnl_week: float,
        equity_usd: float,
        now_utc: datetime,
    ) -> None:
        """
        Update PnL tracking and check kill switch conditions.

        Args:
            realized_pnl_today: Total realized PnL for current UTC day
            realized_pnl_week: Total realized PnL for current week
            equity_usd: Current account equity
            now_utc: Current UTC datetime
        """
        # Update state
        self.state.current_daily_pnl = realized_pnl_today
        self.state.current_weekly_pnl = realized_pnl_week

        # Reset daily stop at UTC midnight
        if self._should_reset_daily(now_utc):
            logger.info(f"Resetting daily stop at {now_utc.isoformat()}")
            self.state.daily_stop_active = False
            self.state.last_daily_reset = now_utc

        # Reset weekly stop at Monday 00:00 UTC
        if self._should_reset_weekly(now_utc):
            logger.info(f"Resetting weekly tracking at {now_utc.isoformat()}")
            self.state.last_weekly_reset = now_utc

        # Check if weekly pause period has ended
        if self.state.weekly_pause_active and self.state.pause_end_date:
            if now_utc >= self.state.pause_end_date:
                logger.warning(
                    f"Weekly pause period ended at {now_utc.isoformat()}. "
                    f"Entering reduced risk period."
                )
                self.state.weekly_pause_active = False
                # Set reduced risk end date
                self.state.reduced_risk_end_date = now_utc + timedelta(
                    days=self.config.reduced_risk_days
                )

        # Check if reduced risk period has ended
        if self.state.reduced_risk_end_date:
            if now_utc >= self.state.reduced_risk_end_date:
                logger.info(f"Reduced risk period ended at {now_utc.isoformat()}")
                self.state.reduced_risk_end_date = None

        # Calculate PnL percentages
        daily_pnl_pct = realized_pnl_today / equity_usd if equity_usd > 0 else 0.0
        weekly_pnl_pct = realized_pnl_week / equity_usd if equity_usd > 0 else 0.0

        # Check daily stop threshold
        if daily_pnl_pct <= self.config.daily_stop_pct:
            if not self.state.daily_stop_active:
                logger.error(
                    f"⚠️ DAILY STOP TRIGGERED: PnL {daily_pnl_pct:.2%} <= {self.config.daily_stop_pct:.2%}. "
                    f"No new entries until next UTC day."
                )
                self.state.daily_stop_active = True

        # Check weekly stop threshold
        if weekly_pnl_pct <= self.config.weekly_stop_pct:
            if not self.state.weekly_pause_active:
                logger.critical(
                    f"🚨 WEEKLY STOP TRIGGERED: PnL {weekly_pnl_pct:.2%} <= {self.config.weekly_stop_pct:.2%}. "
                    f"Trading paused for {self.config.pause_days_after_weekly_stop} days."
                )
                self.state.weekly_pause_active = True
                self.state.pause_end_date = now_utc + timedelta(
                    days=self.config.pause_days_after_weekly_stop
                )

    def is_active(self) -> bool:
        """
        Check if any kill switch is currently active.

        Returns:
            True if trading should be blocked, False otherwise
        """
        return self.state.daily_stop_active or self.state.weekly_pause_active

    def get_active_reason(self) -> Optional[str]:
        """
        Get reason why kill switch is active.

        Returns:
            Description of active kill switch, or None if not active
        """
        if self.state.daily_stop_active:
            return "Daily stop loss active"
        if self.state.weekly_pause_active:
            if self.state.pause_end_date:
                return f"Weekly pause active until {self.state.pause_end_date.isoformat()}"
            return "Weekly pause active"
        return None

    def is_reduced_risk_active(self, now_utc: Optional[datetime] = None) -> bool:
        """
        Check if reduced risk period is active (after weekly pause).

        Args:
            now_utc: Current UTC datetime (defaults to utcnow())

        Returns:
            True if reduced risk period is active
        """
        if self.state.reduced_risk_end_date is None:
            return False

        if now_utc is None:
            now_utc = datetime.utcnow()

        return now_utc < self.state.reduced_risk_end_date

    def get_risk_multiplier(self, now_utc: Optional[datetime] = None) -> float:
        """
        Get risk multiplier to apply.

        Args:
            now_utc: Current UTC datetime (defaults to utcnow())

        Returns:
            1.0 for normal risk, or fraction for reduced risk period
        """
        if self.is_reduced_risk_active(now_utc):
            # Calculate ratio: reduced_risk / normal_risk
            # e.g., 0.5% / 1.0% = 0.5
            normal_risk = 0.01  # Default 1%
            reduced_risk = self.config.reduced_risk_after_pause_pct
            multiplier = reduced_risk / normal_risk
            logger.debug(f"Reduced risk multiplier: {multiplier:.2f}x")
            return multiplier
        return 1.0

    def _should_reset_daily(self, now_utc: datetime) -> bool:
        """Check if daily stop should be reset (new UTC day)"""
        if self.state.last_daily_reset is None:
            return True

        # Check if we've crossed UTC midnight
        last_reset_date = self.state.last_daily_reset.date()
        current_date = now_utc.date()

        return current_date > last_reset_date

    def _should_reset_weekly(self, now_utc: datetime) -> bool:
        """Check if weekly tracking should be reset (new week starting Monday)"""
        if self.state.last_weekly_reset is None:
            return True

        # Get ISO calendar (year, week, weekday)
        last_iso = self.state.last_weekly_reset.isocalendar()
        current_iso = now_utc.isocalendar()

        # Check if we're in a different (year, week) tuple
        return (current_iso[0], current_iso[1]) > (last_iso[0], last_iso[1])

    def get_state(self) -> KillSwitchState:
        """
        Get current kill switch state.

        Returns:
            Current state object
        """
        return self.state

    def load_state(self, state: KillSwitchState) -> None:
        """
        Load kill switch state (for recovery).

        Args:
            state: State to load
        """
        self.state = state
        logger.info(
            f"KillSwitch state loaded: "
            f"daily_active={state.daily_stop_active}, "
            f"weekly_active={state.weekly_pause_active}"
        )

    def reset(self) -> None:
        """Reset all kill switch states (use with caution)"""
        logger.warning("Resetting all kill switch states")
        self.state = KillSwitchState()
