"""Unit tests for trailing stop logic"""

import pytest
from datetime import datetime

from bot.execution.position import Position, PositionStatus, ExitReason
from bot.execution.trailing_stop import TrailingStopManager
from bot.execution.models import OrderSide


class TestTrailingStopManager:
    """Test trailing stop manager"""

    @pytest.fixture
    def trailing_manager(self):
        """Create trailing stop manager"""
        return TrailingStopManager()

    @pytest.fixture
    def long_position(self):
        """Create sample LONG position"""
        return Position(
            position_id="pos_001",
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            entry_price=50000.0,
            quantity=0.01,
            notional_usd=500.0,
            leverage=1.0,
            margin_usd=500.0,
            stop_price=49500.0,  # 1% stop
            entry_time=datetime.utcnow(),
            risk_amount_usd=5.0,  # 1% of $500
            initial_stop_price=49500.0,
            trail_after_r=1.0,  # Enable trailing after 1R
            atr_trail_mult=2.0,  # Trail at 2*ATR
            entry_order_id="entry_001",
            stop_order_id="stop_001"
        )

    @pytest.fixture
    def short_position(self):
        """Create sample SHORT position"""
        return Position(
            position_id="pos_002",
            symbol="ETHUSDT",
            side=OrderSide.SHORT,
            entry_price=3000.0,
            quantity=1.0,
            notional_usd=3000.0,
            leverage=1.0,
            margin_usd=3000.0,
            stop_price=3030.0,  # 1% stop
            entry_time=datetime.utcnow(),
            risk_amount_usd=30.0,  # 1% of $3000
            initial_stop_price=3030.0,
            trail_after_r=1.0,
            atr_trail_mult=2.0,
            entry_order_id="entry_002",
            stop_order_id="stop_002"
        )

    def test_trailing_not_enabled_initially(self, trailing_manager, long_position):
        """Test trailing is not enabled until profit threshold reached"""
        # Price hasn't moved enough for trailing
        current_price = 50100.0  # $100 profit = 0.2R (need 1.0R)
        atr = 300.0

        result = trailing_manager.update_trailing_stop(
            position=long_position,
            current_price=current_price,
            atr=atr
        )

        assert result is None
        assert long_position.trailing_enabled is False
        assert long_position.stop_price == 49500.0  # Stop unchanged

    def test_trailing_enables_at_threshold(self, trailing_manager, long_position):
        """Test trailing enables when profit >= trail_after_r"""
        # Price moved to 1R profit
        current_price = 50500.0  # $500 profit = 1.0R exactly
        atr = 300.0

        result = trailing_manager.update_trailing_stop(
            position=long_position,
            current_price=current_price,
            atr=atr
        )

        assert long_position.trailing_enabled is True
        assert result is not None  # Stop should be updated

        # Expected new stop = 50500 - (2.0 * 300) = 49900
        expected_stop = 50500.0 - (2.0 * 300.0)
        assert result == pytest.approx(expected_stop)
        assert long_position.stop_price == pytest.approx(expected_stop)

    def test_trailing_updates_on_price_increase_long(self, trailing_manager, long_position):
        """Test trailing stop updates as price rises (LONG)"""
        # Enable trailing first
        current_price = 50500.0
        atr = 300.0
        trailing_manager.update_trailing_stop(long_position, current_price, atr)
        assert long_position.trailing_enabled is True

        # Price rises further
        new_price = 51000.0
        result = trailing_manager.update_trailing_stop(long_position, new_price, atr)

        # Expected new stop = 51000 - 600 = 50400
        expected_stop = 51000.0 - 600.0
        assert result == pytest.approx(expected_stop)
        assert long_position.stop_price == pytest.approx(expected_stop)

    def test_trailing_never_moves_down_long(self, trailing_manager, long_position):
        """Test trailing stop never moves down for LONG"""
        # Enable trailing
        current_price = 50500.0
        atr = 300.0
        trailing_manager.update_trailing_stop(long_position, current_price, atr)

        current_stop = long_position.stop_price

        # Price drops
        lower_price = 50200.0
        result = trailing_manager.update_trailing_stop(long_position, lower_price, atr)

        # Stop should NOT move down
        assert result is None
        assert long_position.stop_price == current_stop

    def test_trailing_updates_on_price_decrease_short(self, trailing_manager, short_position):
        """Test trailing stop updates as price falls (SHORT)"""
        # Enable trailing first (profit for short = price decrease)
        current_price = 2970.0  # $30 profit = 1.0R
        atr = 15.0
        trailing_manager.update_trailing_stop(short_position, current_price, atr)
        assert short_position.trailing_enabled is True

        # Price falls further
        new_price = 2950.0
        result = trailing_manager.update_trailing_stop(short_position, new_price, atr)

        # Expected new stop = 2950 + (2.0 * 15) = 2980
        expected_stop = 2950.0 + 30.0
        assert result == pytest.approx(expected_stop)
        assert short_position.stop_price == pytest.approx(expected_stop)

    def test_trailing_never_moves_up_short(self, trailing_manager, short_position):
        """Test trailing stop never moves up for SHORT"""
        # Enable trailing
        current_price = 2970.0
        atr = 15.0
        trailing_manager.update_trailing_stop(short_position, current_price, atr)

        current_stop = short_position.stop_price

        # Price rises (unfavorable for short)
        higher_price = 2985.0
        result = trailing_manager.update_trailing_stop(short_position, higher_price, atr)

        # Stop should NOT move up
        assert result is None
        assert short_position.stop_price == current_stop

    def test_check_stop_hit_long(self, trailing_manager, long_position):
        """Test stop hit detection for LONG"""
        # Price above stop - not hit
        assert trailing_manager.check_stop_hit(long_position, 50000.0) is False
        assert trailing_manager.check_stop_hit(long_position, 49600.0) is False

        # Price at stop - hit
        assert trailing_manager.check_stop_hit(long_position, 49500.0) is True

        # Price below stop - hit
        assert trailing_manager.check_stop_hit(long_position, 49400.0) is True

    def test_check_stop_hit_short(self, trailing_manager, short_position):
        """Test stop hit detection for SHORT"""
        # Price below stop - not hit
        assert trailing_manager.check_stop_hit(short_position, 3000.0) is False
        assert trailing_manager.check_stop_hit(short_position, 3020.0) is False

        # Price at stop - hit
        assert trailing_manager.check_stop_hit(short_position, 3030.0) is True

        # Price above stop - hit
        assert trailing_manager.check_stop_hit(short_position, 3040.0) is True

    def test_check_tp_hit_long(self, trailing_manager, long_position):
        """Test TP hit detection for LONG"""
        long_position.tp_price = 51000.0

        # Price below TP - not hit
        assert trailing_manager.check_tp_hit(long_position, 50000.0) is False
        assert trailing_manager.check_tp_hit(long_position, 50900.0) is False

        # Price at TP - hit
        assert trailing_manager.check_tp_hit(long_position, 51000.0) is True

        # Price above TP - hit
        assert trailing_manager.check_tp_hit(long_position, 51100.0) is True

    def test_check_tp_hit_short(self, trailing_manager, short_position):
        """Test TP hit detection for SHORT"""
        short_position.tp_price = 2900.0

        # Price above TP - not hit
        assert trailing_manager.check_tp_hit(short_position, 3000.0) is False
        assert trailing_manager.check_tp_hit(short_position, 2950.0) is False

        # Price at TP - hit
        assert trailing_manager.check_tp_hit(short_position, 2900.0) is True

        # Price below TP - hit
        assert trailing_manager.check_tp_hit(short_position, 2850.0) is True

    def test_check_tp_hit_no_tp_set(self, trailing_manager, long_position):
        """Test TP hit when no TP set returns False"""
        long_position.tp_price = None

        assert trailing_manager.check_tp_hit(long_position, 60000.0) is False

    def test_get_stop_distance_r(self, trailing_manager, long_position):
        """Test calculating distance to stop in R"""
        # At entry: stop distance = -1R
        distance = trailing_manager.get_stop_distance_r(long_position, 50000.0)
        assert distance == pytest.approx(-1.0)

        # After profit: stop distance increases
        long_position.stop_price = 50200.0  # Moved stop up
        distance = trailing_manager.get_stop_distance_r(long_position, 51000.0)
        # PnL at stop = (50200 - 50000) * 0.01 = $2
        # R = 2 / 5 = 0.4R
        assert distance == pytest.approx(0.4)

    def test_get_tp_distance_r(self, trailing_manager, long_position):
        """Test calculating distance to TP in R"""
        long_position.tp_price = 50750.0

        distance = trailing_manager.get_tp_distance_r(long_position)

        # PnL at TP = (50750 - 50000) * 0.01 = $7.5
        # R = 7.5 / 5 = 1.5R
        assert distance == pytest.approx(1.5)

    def test_get_tp_distance_r_no_tp(self, trailing_manager, long_position):
        """Test TP distance when no TP set"""
        long_position.tp_price = None

        distance = trailing_manager.get_tp_distance_r(long_position)

        assert distance is None

    def test_should_update_on_price_move(self, trailing_manager, long_position):
        """Test significant price move detection"""
        long_position.trailing_enabled = True

        # 0.3% move - below threshold
        assert trailing_manager.should_update_on_price_move(
            long_position,
            price_move_pct=0.003,
            threshold_pct=0.005
        ) is False

        # 0.5% move - at threshold
        assert trailing_manager.should_update_on_price_move(
            long_position,
            price_move_pct=0.005,
            threshold_pct=0.005
        ) is True

        # 1% move - above threshold
        assert trailing_manager.should_update_on_price_move(
            long_position,
            price_move_pct=0.01,
            threshold_pct=0.005
        ) is True

    def test_should_update_not_trailing_enabled(self, trailing_manager, long_position):
        """Test no update if trailing not enabled"""
        long_position.trailing_enabled = False

        assert trailing_manager.should_update_on_price_move(
            long_position,
            price_move_pct=0.02,  # Large move
            threshold_pct=0.005
        ) is False

    def test_highest_price_tracking_long(self, long_position):
        """Test highest price tracking for LONG"""
        assert long_position.highest_price_seen == 50000.0

        # Price rises
        long_position.update_highest_price_seen(50500.0)
        assert long_position.highest_price_seen == 50500.0

        # Price falls - highest should not decrease
        long_position.update_highest_price_seen(50200.0)
        assert long_position.highest_price_seen == 50500.0

        # Price rises above previous high
        long_position.update_highest_price_seen(50800.0)
        assert long_position.highest_price_seen == 50800.0

    def test_lowest_price_tracking_short(self, short_position):
        """Test lowest price tracking for SHORT"""
        assert short_position.highest_price_seen == 3000.0  # Actually "lowest" for short

        # Price falls (good for short)
        short_position.update_highest_price_seen(2950.0)
        assert short_position.highest_price_seen == 2950.0

        # Price rises - lowest should not increase
        short_position.update_highest_price_seen(2980.0)
        assert short_position.highest_price_seen == 2950.0

        # Price falls below previous low
        short_position.update_highest_price_seen(2920.0)
        assert short_position.highest_price_seen == 2920.0


class TestPositionModel:
    """Test Position model properties and methods"""

    @pytest.fixture
    def position(self):
        """Create sample position"""
        return Position(
            position_id="test_pos",
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            notional_usd=5000.0,
            leverage=1.0,
            margin_usd=5000.0,
            stop_price=49500.0,
            entry_time=datetime.utcnow(),
            risk_amount_usd=50.0,
            initial_stop_price=49500.0,
            trail_after_r=1.0,
            atr_trail_mult=2.0,
            entry_order_id="entry_001",
            stop_order_id="stop_001"
        )

    def test_position_properties(self, position):
        """Test position boolean properties"""
        assert position.is_open is True
        assert position.is_closed is False
        assert position.is_long is True
        assert position.is_short is False

    def test_pnl_r_calculation(self, position):
        """Test PnL in R calculation"""
        # No profit yet
        assert position.pnl_r == 0.0

        # Update PnL
        position.update_unrealized_pnl(50500.0)  # $50 profit
        assert position.pnl_r == pytest.approx(1.0)  # $50 / $50 risk = 1R

        position.update_unrealized_pnl(50750.0)  # $75 profit
        assert position.pnl_r == pytest.approx(1.5)  # $75 / $50 = 1.5R

    def test_should_enable_trailing(self, position):
        """Test trailing enable check"""
        # Not profitable enough
        position.update_unrealized_pnl(50250.0)  # 0.5R
        assert position.should_enable_trailing() is False

        # Exactly at threshold
        position.update_unrealized_pnl(50500.0)  # 1.0R
        assert position.should_enable_trailing() is True

        # Above threshold
        position.update_unrealized_pnl(50750.0)  # 1.5R
        assert position.should_enable_trailing() is True

    def test_close_position(self, position):
        """Test position close"""
        position.close_position(
            exit_price=50500.0,
            exit_reason=ExitReason.TP,
            fees_paid=2.0
        )

        assert position.is_closed is True
        assert position.exit_price == 50500.0
        assert position.exit_reason == ExitReason.TP
        assert position.exit_time is not None

        # Realized PnL = (50500 - 50000) * 0.1 - 2 = 48
        assert position.realized_pnl_usd == pytest.approx(48.0)
        assert position.unrealized_pnl_usd == 0.0

    def test_position_to_dict(self, position):
        """Test position serialization"""
        position_dict = position.to_dict()

        assert position_dict["position_id"] == "test_pos"
        assert position_dict["symbol"] == "BTCUSDT"
        assert position_dict["side"] == "LONG"
        assert position_dict["entry_price"] == 50000.0
        assert position_dict["status"] == "OPEN"
        assert "pnl_r" in position_dict
        assert "holding_time_seconds" in position_dict
