"""Deterministic trailing stop simulation test."""

from datetime import datetime, timezone

from bot.backtest.engine import BacktestEngine
from bot.config.loader import load_config
from bot.core.constants import OrderSide
from bot.core.types import Candle


class DummyExchange:
    def fetch_ohlcv(self, *args, **kwargs):
        return []


def test_trailing_exit_reason_trail(tmp_path):
    config = load_config("config/config.example.json", load_env=False)
    config.logging.log_dir = str(tmp_path)
    engine = BacktestEngine(config, DummyExchange())

    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    pos = engine._account.open_position(
        symbol="BTCUSDT",
        side=OrderSide.LONG,
        entry_price_raw=100.0,
        quantity=1.0,
        notional_usd=100.0,
        margin_usd=50.0,
        risk_usd=5.0,
        stop_price=95.0,
        tp_price=None,
        entry_time=now,
        strategy="TrendPullback",
        regime="TREND",
        regime_confidence=0.7,
        trail_after_r=1.0,
        atr_trail_mult=1.0,
    )
    pos.trail_enabled = True

    # Move price to activate trailing
    candle_up = Candle(
        timestamp=int(now.timestamp() * 1000),
        open=100.0,
        high=112.0,
        low=99.5,
        close=111.0,
        volume=10.0,
    )
    engine._update_trailing_stops("BTCUSDT", candle_up, {"atr14": 5.0})
    assert pos.trailing_stop is not None

    # Retrace hits trailing stop
    candle_down = Candle(
        timestamp=int(now.timestamp() * 1000) + 300_000,
        open=111.0,
        high=111.5,
        low=105.0,
        close=106.0,
        volume=10.0,
    )
    engine._check_exits("BTCUSDT", candle_down, now)

    closed = engine._account.closed_trades
    assert closed
    assert closed[-1].exit_reason == "TRAIL"
