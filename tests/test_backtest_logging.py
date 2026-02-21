"""Integration tests for backtest logging and regime snapshot."""

from datetime import datetime, timezone, timedelta
import json

import pytest

from bot.backtest.engine import BacktestEngine
from bot.config.loader import load_config


class FakeExchange:
    """Simple exchange stub for backtest data provider."""

    def __init__(self, data_by_tf):
        self._data_by_tf = data_by_tf

    def fetch_ohlcv(self, symbol, timeframe="5m", since=None, limit=1500):
        data = self._data_by_tf.get(timeframe, [])
        if since is None:
            return data[:limit]
        filtered = [row for row in data if row[0] >= since]
        return filtered[:limit]


def _make_ohlcv_series(start: datetime, timeframe_ms: int, bars: int, start_price: float = 100.0):
    data = []
    price = start_price
    for i in range(bars):
        ts = int((start + timedelta(milliseconds=i * timeframe_ms)).timestamp() * 1000)
        # small deterministic drift
        price += 0.1
        o = price
        h = price + 0.2
        l = price - 0.2
        c = price + 0.05
        v = 10.0
        data.append([ts, o, h, l, c, v])
    return data


def test_backtest_regime_log_and_trade_snapshot(tmp_path):
    config = load_config("config/config.example.json", load_env=False)
    config.logging.log_dir = str(tmp_path)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    data_by_tf = {
        "5m": _make_ohlcv_series(start, 300_000, 200),
        "1h": _make_ohlcv_series(start, 3_600_000, 80),
        "4h": _make_ohlcv_series(start, 14_400_000, 40),
    }
    exchange = FakeExchange(data_by_tf)
    engine = BacktestEngine(config, exchange)

    result = engine.run(symbols=["BTCUSDT"], start=start, end=end)

    regime_log = tmp_path / "regime_decisions.jsonl"
    assert regime_log.exists()
    with regime_log.open() as f:
        first_line = f.readline().strip()
        assert first_line
        payload = json.loads(first_line)
        assert "regime" in payload
        assert "confidence" in payload

    if result.trades:
        assert all(t.regime for t in result.trades)
        assert all(t.regime_confidence is not None for t in result.trades)
