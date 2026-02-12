"""Tests for kill switch (daily/weekly stops)"""

from datetime import datetime, timedelta

import pytest

from bot.config.models import RiskConfig
from bot.core.types import KillSwitchState
from bot.risk.kill_switch import KillSwitch


@pytest.fixture
def default_config():
    """Create default risk configuration"""
    return RiskConfig(
        risk_per_trade_pct=0.01,
        max_total_open_risk_pct=0.025,
        max_open_positions=2,
        max_same_direction_positions=2,
        daily_stop_pct=-0.04,  # -4%
        weekly_stop_pct=-0.1,  # -10%
        pause_days_after_weekly_stop=7,
        reduced_risk_after_pause_pct=0.005,  # 0.5%
        reduced_risk_days=3,
    )


@pytest.fixture
def kill_switch(default_config):
    """Create kill switch instance"""
    return KillSwitch(default_config)


class TestDailyStop:
    """Test daily stop loss functionality"""

    def test_daily_stop_not_triggered_below_threshold(self, kill_switch):
        """Test that daily stop is not triggered when loss is within limit"""
        now = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        # Loss of -3% (below -4% threshold)
        daily_pnl = -300.0
        weekly_pnl = -300.0

        kill_switch.update_pnl(daily_pnl, weekly_pnl, equity, now)

        assert kill_switch.is_active() is False
        assert kill_switch.state.daily_stop_active is False

    def test_daily_stop_triggered_at_threshold(self, kill_switch):
        """Test that daily stop triggers at exact threshold"""
        now = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        # Loss of -4% (exactly at threshold)
        daily_pnl = -400.0
        weekly_pnl = -400.0

        kill_switch.update_pnl(daily_pnl, weekly_pnl, equity, now)

        assert kill_switch.is_active() is True
        assert kill_switch.state.daily_stop_active is True
        assert "Daily stop" in kill_switch.get_active_reason()

    def test_daily_stop_triggered_beyond_threshold(self, kill_switch):
        """Test that daily stop triggers beyond threshold"""
        now = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        # Loss of -5% (beyond -4% threshold)
        daily_pnl = -500.0
        weekly_pnl = -500.0

        kill_switch.update_pnl(daily_pnl, weekly_pnl, equity, now)

        assert kill_switch.is_active() is True
        assert kill_switch.state.daily_stop_active is True

    def test_daily_stop_resets_next_day(self, kill_switch):
        """Test that daily stop resets at UTC midnight"""
        day1 = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        # Trigger daily stop on day 1
        kill_switch.update_pnl(-500.0, -500.0, equity, day1)
        assert kill_switch.is_active() is True

        # Move to next day
        day2 = datetime(2024, 1, 16, 0, 0, 1)
        kill_switch.update_pnl(-50.0, -550.0, equity, day2)

        # Daily stop should be reset
        assert kill_switch.state.daily_stop_active is False
        assert kill_switch.is_active() is False

    def test_daily_stop_stays_active_same_day(self, kill_switch):
        """Test that daily stop stays active throughout the same day"""
        morning = datetime(2024, 1, 15, 8, 0, 0)
        equity = 10000.0

        # Trigger stop in morning
        kill_switch.update_pnl(-500.0, -500.0, equity, morning)
        assert kill_switch.is_active() is True

        # Check in afternoon
        afternoon = datetime(2024, 1, 15, 16, 0, 0)
        kill_switch.update_pnl(-300.0, -500.0, equity, afternoon)

        # Should still be active
        assert kill_switch.state.daily_stop_active is True


class TestWeeklyStop:
    """Test weekly stop loss functionality"""

    def test_weekly_stop_not_triggered_below_threshold(self, kill_switch):
        """Test that weekly stop is not triggered when loss is within limit"""
        now = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        # Loss of -9% (below -10% threshold)
        daily_pnl = -100.0
        weekly_pnl = -900.0

        kill_switch.update_pnl(daily_pnl, weekly_pnl, equity, now)

        assert kill_switch.state.weekly_pause_active is False

    def test_weekly_stop_triggered_at_threshold(self, kill_switch):
        """Test that weekly stop triggers at exact threshold"""
        now = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        # Loss of -10% (exactly at threshold)
        weekly_pnl = -1000.0

        kill_switch.update_pnl(0.0, weekly_pnl, equity, now)

        assert kill_switch.is_active() is True
        assert kill_switch.state.weekly_pause_active is True
        assert kill_switch.state.pause_end_date is not None

    def test_weekly_pause_duration(self, kill_switch):
        """Test that weekly pause lasts for configured duration"""
        trigger_time = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        # Trigger weekly stop
        kill_switch.update_pnl(0.0, -1000.0, equity, trigger_time)

        expected_end = trigger_time + timedelta(days=7)
        assert kill_switch.state.pause_end_date == expected_end

    def test_weekly_pause_ends_after_duration(self, kill_switch):
        """Test that weekly pause ends after configured days"""
        trigger_time = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        # Trigger weekly stop
        kill_switch.update_pnl(0.0, -1000.0, equity, trigger_time)
        assert kill_switch.state.weekly_pause_active is True

        # Move to day 6 (still in pause)
        day6 = trigger_time + timedelta(days=6)
        kill_switch.update_pnl(0.0, -500.0, equity, day6)
        assert kill_switch.state.weekly_pause_active is True

        # Move to day 7+ (pause should end)
        day8 = trigger_time + timedelta(days=8)
        kill_switch.update_pnl(0.0, -200.0, equity, day8)
        assert kill_switch.state.weekly_pause_active is False

    def test_reduced_risk_period_after_pause(self, kill_switch):
        """Test that reduced risk period activates after weekly pause ends"""
        trigger_time = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        # Trigger weekly stop
        kill_switch.update_pnl(0.0, -1000.0, equity, trigger_time)

        # Move past pause period
        after_pause = trigger_time + timedelta(days=8)
        kill_switch.update_pnl(0.0, -200.0, equity, after_pause)

        # Reduced risk should be active
        assert kill_switch.is_reduced_risk_active(after_pause) is True
        assert kill_switch.state.reduced_risk_end_date is not None

    def test_reduced_risk_multiplier(self, kill_switch):
        """Test that risk multiplier is calculated correctly during reduced risk period"""
        trigger_time = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        # Normal period - should return 1.0
        assert kill_switch.get_risk_multiplier() == 1.0

        # Trigger and move past pause
        kill_switch.update_pnl(0.0, -1000.0, equity, trigger_time)
        after_pause = trigger_time + timedelta(days=8)
        kill_switch.update_pnl(0.0, -200.0, equity, after_pause)

        # During reduced risk period
        # Config: reduced_risk = 0.005 (0.5%), normal = 0.01 (1.0%)
        # Multiplier should be 0.5% / 1.0% = 0.5
        multiplier = kill_switch.get_risk_multiplier(after_pause)
        assert multiplier == pytest.approx(0.5)

    def test_reduced_risk_ends_after_duration(self, kill_switch):
        """Test that reduced risk period ends after configured days"""
        trigger_time = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        # Trigger and move past pause
        kill_switch.update_pnl(0.0, -1000.0, equity, trigger_time)
        after_pause = trigger_time + timedelta(days=8)
        kill_switch.update_pnl(0.0, -200.0, equity, after_pause)

        assert kill_switch.is_reduced_risk_active(after_pause) is True

        # Move past reduced risk period (3 days)
        after_reduced = trigger_time + timedelta(days=12)
        kill_switch.update_pnl(0.0, -100.0, equity, after_reduced)

        assert kill_switch.is_reduced_risk_active(after_reduced) is False
        assert kill_switch.get_risk_multiplier() == 1.0


class TestBothStops:
    """Test interaction between daily and weekly stops"""

    def test_both_stops_triggered(self, kill_switch):
        """Test when both daily and weekly stops are triggered"""
        now = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        # Trigger both stops (-10% daily and weekly)
        kill_switch.update_pnl(-1000.0, -1000.0, equity, now)

        assert kill_switch.state.daily_stop_active is True
        assert kill_switch.state.weekly_pause_active is True
        assert kill_switch.is_active() is True

    def test_daily_resets_while_weekly_active(self, kill_switch):
        """Test that daily can reset while weekly is still active"""
        day1 = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        # Trigger both stops
        kill_switch.update_pnl(-1000.0, -1000.0, equity, day1)
        assert kill_switch.state.daily_stop_active is True
        assert kill_switch.state.weekly_pause_active is True

        # Next day - daily resets but weekly still active
        day2 = datetime(2024, 1, 16, 10, 0, 0)
        kill_switch.update_pnl(-50.0, -1050.0, equity, day2)

        assert kill_switch.state.daily_stop_active is False
        assert kill_switch.state.weekly_pause_active is True
        assert kill_switch.is_active() is True  # Still blocked by weekly


class TestStateManagement:
    """Test state save/load functionality"""

    def test_get_state(self, kill_switch):
        """Test getting current state"""
        now = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        kill_switch.update_pnl(-500.0, -500.0, equity, now)

        state = kill_switch.get_state()
        assert isinstance(state, KillSwitchState)
        assert state.daily_stop_active is True
        assert state.current_daily_pnl == -500.0

    def test_load_state(self, kill_switch):
        """Test loading saved state"""
        # Create a state with daily stop active
        saved_state = KillSwitchState(
            daily_stop_active=True,
            weekly_pause_active=False,
            current_daily_pnl=-500.0,
            current_weekly_pnl=-500.0,
            last_daily_reset=datetime(2024, 1, 15, 0, 0, 0),
        )

        kill_switch.load_state(saved_state)

        assert kill_switch.state.daily_stop_active is True
        assert kill_switch.state.current_daily_pnl == -500.0
        assert kill_switch.is_active() is True

    def test_reset(self, kill_switch):
        """Test resetting all states"""
        now = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        # Trigger stops
        kill_switch.update_pnl(-1000.0, -1000.0, equity, now)
        assert kill_switch.is_active() is True

        # Reset
        kill_switch.reset()

        assert kill_switch.state.daily_stop_active is False
        assert kill_switch.state.weekly_pause_active is False
        assert kill_switch.is_active() is False


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_zero_equity(self, kill_switch):
        """Test with zero equity"""
        now = datetime(2024, 1, 15, 10, 0, 0)

        # Should not crash with zero equity
        kill_switch.update_pnl(-100.0, -100.0, 0.0, now)

        # Should not trigger (division by zero protection)
        assert kill_switch.state.daily_stop_active is False

    def test_positive_pnl(self, kill_switch):
        """Test with positive PnL (winning day)"""
        now = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        # Positive PnL
        kill_switch.update_pnl(200.0, 500.0, equity, now)

        assert kill_switch.is_active() is False

    def test_exact_threshold_negative(self, kill_switch):
        """Test that threshold uses <= not <"""
        now = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        # Exactly at -4%
        daily_pnl = -400.0
        kill_switch.update_pnl(daily_pnl, daily_pnl, equity, now)

        # Should trigger (<=)
        assert kill_switch.state.daily_stop_active is True

    def test_week_boundary_reset(self, kill_switch):
        """Test weekly reset at week boundary"""
        # Friday
        friday = datetime(2024, 1, 19, 23, 0, 0)  # Week 3
        equity = 10000.0

        kill_switch.update_pnl(-100.0, -500.0, equity, friday)

        # Before monday, should need reset
        monday = datetime(2024, 1, 22, 1, 0, 0)  # Week 4
        assert kill_switch._should_reset_weekly(monday) is True

        # After update on monday, last_weekly_reset should be updated
        kill_switch.update_pnl(-50.0, -50.0, equity, monday)

        # Verify that state was updated
        assert kill_switch.state.last_weekly_reset is not None
        # Same week check should now return False
        monday_later = datetime(2024, 1, 22, 10, 0, 0)  # Same week
        assert kill_switch._should_reset_weekly(monday_later) is False


class TestGetActiveReason:
    """Test active reason reporting"""

    def test_no_active_reason(self, kill_switch):
        """Test when no kill switch is active"""
        assert kill_switch.get_active_reason() is None

    def test_daily_stop_reason(self, kill_switch):
        """Test daily stop reason"""
        now = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        kill_switch.update_pnl(-500.0, -500.0, equity, now)

        reason = kill_switch.get_active_reason()
        assert "Daily stop" in reason

    def test_weekly_pause_reason(self, kill_switch):
        """Test weekly pause reason"""
        now = datetime(2024, 1, 15, 10, 0, 0)
        equity = 10000.0

        kill_switch.update_pnl(0.0, -1000.0, equity, now)

        reason = kill_switch.get_active_reason()
        assert "Weekly pause" in reason
        assert "until" in reason  # Should include end date
