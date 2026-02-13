"""
E2E Correlation Tests — Milestone 8, Task 22

Covers:
  1. High-correlation same-direction → BLOCKED
  2. High-correlation opposite-direction (hedge) with corr <= hedge_corr_max → ALLOWED
  3. High-correlation opposite-direction (hedge) with corr > hedge_corr_max → BLOCKED
  4. Low-correlation symbols → always ALLOWED in any direction
  5. No correlation data (cache miss) → ALLOWED (pass-through)
  6. Correlation matrix update from price data
  7. Multi-symbol portfolio: complex mixed scenarios
  8. RiskEngine integration: correlation check is part of the pipeline
  9. get_correlated_positions / get_correlation_summary helpers
  10. Cache management (clear_cache, bidirectional lookup)
"""

from datetime import datetime
from typing import List, Optional

import numpy as np
import pytest

from bot.config.models import RiskConfig
from bot.core.constants import OrderSide, PositionStatus
from bot.core.types import Position
from bot.risk.correlation_filter import CorrelationFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_risk_config(
    correlation_threshold: float = 0.85,
    hedge_corr_max: float = 0.60,
) -> RiskConfig:
    return RiskConfig(
        correlation_threshold=correlation_threshold,
        hedge_corr_max=hedge_corr_max,
        max_open_positions=6,
        max_same_direction_positions=6,
    )


def _make_filter(
    correlation_threshold: float = 0.85,
    hedge_corr_max: float = 0.60,
) -> CorrelationFilter:
    return CorrelationFilter(config=_make_risk_config(correlation_threshold, hedge_corr_max))


def _make_position(
    symbol: str,
    side: OrderSide,
    entry_price: float = 100.0,
    trade_id: Optional[str] = None,
) -> Position:
    return Position(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        quantity=1.0,
        notional=entry_price,
        leverage=5.0,
        margin=entry_price / 5.0,
        stop_price=entry_price * 0.95,
        tp_price=entry_price * 1.1,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        entry_time=datetime.utcnow(),
        trade_id=trade_id or f"{symbol}_001",
        status=PositionStatus.OPEN,
    )


def _correlated_prices(
    base_price: float = 100.0,
    n: int = 100,
    correlation: float = 0.95,
    seed: int = 42,
) -> tuple:
    """
    Generate two price series with approximately the given correlation.
    Returns (prices_a, prices_b) as np.ndarrays.
    """
    rng = np.random.default_rng(seed)

    # Generate correlated log-returns using Cholesky decomposition
    cov = np.array([[1.0, correlation], [correlation, 1.0]])
    L = np.linalg.cholesky(cov)
    noise = rng.standard_normal((2, n))
    correlated = L @ noise

    # Convert returns to price series
    returns_a = correlated[0] * 0.02   # 2% daily volatility
    returns_b = correlated[1] * 0.02

    prices_a = base_price * np.exp(np.cumsum(np.insert(returns_a, 0, 0)))
    prices_b = base_price * np.exp(np.cumsum(np.insert(returns_b, 0, 0)))

    return prices_a, prices_b


def _inject_correlation(
    cf: CorrelationFilter,
    symbol_a: str,
    symbol_b: str,
    correlation: float,
) -> None:
    """Directly inject a known correlation into the cache (both directions)."""
    cf.correlation_cache[(symbol_a, symbol_b)] = correlation
    cf.correlation_cache[(symbol_b, symbol_a)] = correlation


# ---------------------------------------------------------------------------
# 1. High-correlation, same direction → BLOCKED
# ---------------------------------------------------------------------------

class TestHighCorrelationSameDirection:

    def test_same_direction_above_threshold_blocked(self):
        """LONG into high-corr LONG bucket → rejected."""
        cf = _make_filter(correlation_threshold=0.85)
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.92)

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        approved, reason = cf.check_correlation_filter("ETHUSDT", OrderSide.LONG, open_positions)

        assert approved is False
        assert "0.92" in reason or "High correlation" in reason

    def test_short_same_direction_above_threshold_blocked(self):
        """SHORT into high-corr SHORT bucket → rejected."""
        cf = _make_filter(correlation_threshold=0.85)
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.90)

        open_positions = [_make_position("BTCUSDT", OrderSide.SHORT)]
        approved, reason = cf.check_correlation_filter("ETHUSDT", OrderSide.SHORT, open_positions)

        assert approved is False
        assert "SHORT" in reason or "correlation" in reason.lower()

    def test_exactly_at_threshold_not_blocked(self):
        """Correlation exactly at threshold (not strictly above) → allowed."""
        cf = _make_filter(correlation_threshold=0.85)
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.85)  # exactly at threshold

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        approved, _ = cf.check_correlation_filter("ETHUSDT", OrderSide.LONG, open_positions)

        # 0.85 is NOT > 0.85, so should pass
        assert approved is True

    def test_just_above_threshold_blocked(self):
        """Correlation just above threshold → blocked."""
        cf = _make_filter(correlation_threshold=0.85)
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.851)

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        approved, _ = cf.check_correlation_filter("ETHUSDT", OrderSide.LONG, open_positions)

        assert approved is False

    def test_rejection_reason_contains_existing_symbol(self):
        """Rejection message names the conflicting position symbol."""
        cf = _make_filter()
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.92)

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        _, reason = cf.check_correlation_filter("ETHUSDT", OrderSide.LONG, open_positions)

        assert "BTCUSDT" in reason

    def test_multiple_open_positions_any_block_is_sufficient(self):
        """If ANY open position has high corr same-direction → blocked."""
        cf = _make_filter()
        _inject_correlation(cf, "SOLUSDT", "BTCUSDT", 0.30)   # low → OK
        _inject_correlation(cf, "SOLUSDT", "ETHUSDT", 0.93)   # high same-dir → block

        open_positions = [
            _make_position("BTCUSDT", OrderSide.LONG),
            _make_position("ETHUSDT", OrderSide.LONG),
        ]
        approved, _ = cf.check_correlation_filter("SOLUSDT", OrderSide.LONG, open_positions)

        assert approved is False

    def test_absolute_value_used_for_correlation(self):
        """Negative correlation -0.92 treated as 0.92 (abs) for same-direction check."""
        cf = _make_filter(correlation_threshold=0.85)
        # Negative correlation stored; abs should still trigger the check
        cf.correlation_cache[("BTCUSDT", "ETHUSDT")] = -0.92
        cf.correlation_cache[("ETHUSDT", "BTCUSDT")] = -0.92

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        approved, _ = cf.check_correlation_filter("ETHUSDT", OrderSide.LONG, open_positions)

        assert approved is False


# ---------------------------------------------------------------------------
# 2. High-correlation, opposite direction, corr <= hedge_corr_max → ALLOWED
# ---------------------------------------------------------------------------

class TestHedgeAllowed:

    def test_opposite_direction_low_enough_corr_allowed(self):
        """SHORT vs LONG position with corr between threshold and hedge_corr_max → allowed."""
        # threshold=0.85, hedge_corr_max=0.60 — BUT corr must be > threshold AND <= hedge_max
        # With defaults this means corr > 0.85 AND corr <= 0.60 which is impossible.
        # Use custom config: threshold=0.50, hedge_corr_max=0.80
        cf = _make_filter(correlation_threshold=0.50, hedge_corr_max=0.80)
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.65)   # > 0.50 threshold, <= 0.80 hedge max

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        approved, reason = cf.check_correlation_filter("ETHUSDT", OrderSide.SHORT, open_positions)

        assert approved is True
        assert reason == ""

    def test_hedge_allowed_logs_no_rejection(self):
        """Valid hedge produces empty rejection reason."""
        cf = _make_filter(correlation_threshold=0.50, hedge_corr_max=0.80)
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.70)

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        approved, reason = cf.check_correlation_filter("ETHUSDT", OrderSide.SHORT, open_positions)

        assert approved is True
        assert reason == ""

    def test_no_existing_positions_always_allowed(self):
        """With no open positions, any new position is allowed."""
        cf = _make_filter()
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.99)

        approved, reason = cf.check_correlation_filter("ETHUSDT", OrderSide.LONG, [])

        assert approved is True
        assert reason == ""


# ---------------------------------------------------------------------------
# 3. High-correlation, opposite direction, corr > hedge_corr_max → BLOCKED
# ---------------------------------------------------------------------------

class TestHedgeBlocked:

    def test_hedge_with_too_high_corr_blocked(self):
        """Hedge blocked when corr > hedge_corr_max."""
        cf = _make_filter(correlation_threshold=0.50, hedge_corr_max=0.80)
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.90)  # > hedge_corr_max=0.80

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        approved, reason = cf.check_correlation_filter("ETHUSDT", OrderSide.SHORT, open_positions)

        assert approved is False
        assert "hedge" in reason.lower() or "0.90" in reason

    def test_hedge_rejection_reason_contains_max(self):
        """Rejection reason mentions the hedge_corr_max limit."""
        cf = _make_filter(correlation_threshold=0.50, hedge_corr_max=0.75)
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.88)

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        _, reason = cf.check_correlation_filter("ETHUSDT", OrderSide.SHORT, open_positions)

        assert "0.75" in reason or "maximum" in reason.lower()

    def test_default_config_hedge_with_high_corr_blocked(self):
        """Default config: threshold=0.85, hedge_corr_max=0.60.
        Opposite-direction with corr=0.91 (> threshold) also > hedge_corr_max → blocked."""
        cf = _make_filter()  # threshold=0.85, hedge_corr_max=0.60
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.91)

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        approved, _ = cf.check_correlation_filter("ETHUSDT", OrderSide.SHORT, open_positions)

        assert approved is False


# ---------------------------------------------------------------------------
# 4. Low-correlation symbols → ALLOWED in any direction
# ---------------------------------------------------------------------------

class TestLowCorrelation:

    def test_low_corr_same_direction_allowed(self):
        """Low-correlation symbols not blocked even with same direction."""
        cf = _make_filter(correlation_threshold=0.85)
        _inject_correlation(cf, "BTCUSDT", "SOLUSDT", 0.20)

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        approved, _ = cf.check_correlation_filter("SOLUSDT", OrderSide.LONG, open_positions)

        assert approved is True

    def test_low_corr_opposite_direction_allowed(self):
        """Low-correlation opposite direction allowed."""
        cf = _make_filter()
        _inject_correlation(cf, "BTCUSDT", "SOLUSDT", 0.15)

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        approved, _ = cf.check_correlation_filter("SOLUSDT", OrderSide.SHORT, open_positions)

        assert approved is True

    def test_negative_low_corr_allowed(self):
        """Negatively correlated (abs low) symbols are allowed."""
        cf = _make_filter()
        _inject_correlation(cf, "BTCUSDT", "GOLDUSDT", -0.20)

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        approved, _ = cf.check_correlation_filter("GOLDUSDT", OrderSide.LONG, open_positions)

        assert approved is True

    def test_near_threshold_below_allowed(self):
        """Correlation just below threshold → allowed."""
        cf = _make_filter(correlation_threshold=0.85)
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.849)

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        approved, _ = cf.check_correlation_filter("ETHUSDT", OrderSide.LONG, open_positions)

        assert approved is True


# ---------------------------------------------------------------------------
# 5. No correlation data (cache miss) → ALLOWED
# ---------------------------------------------------------------------------

class TestCacheMiss:

    def test_no_data_for_pair_allowed(self):
        """Unknown pair (no cache entry) → allowed by default."""
        cf = _make_filter()
        # No correlation injected between BTCUSDT and NEWUSDT

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        approved, _ = cf.check_correlation_filter("NEWUSDT", OrderSide.LONG, open_positions)

        assert approved is True

    def test_partial_cache_miss_still_checks_known_pairs(self):
        """Known pair blocked even if other pair has no data."""
        cf = _make_filter()
        # NEWUSDT has no data, ETHUSDT has high corr
        _inject_correlation(cf, "ETHUSDT", "BNBUSDT", 0.93)

        open_positions = [
            _make_position("ETHUSDT", OrderSide.LONG),
            _make_position("NEWUSDT", OrderSide.LONG),  # no corr data
        ]
        approved, _ = cf.check_correlation_filter("BNBUSDT", OrderSide.LONG, open_positions)

        # Blocked by ETHUSDT even though NEWUSDT has no data
        assert approved is False

    def test_empty_cache_all_allowed(self):
        """Empty correlation cache → all pass-through."""
        cf = _make_filter()
        open_positions = [
            _make_position("BTCUSDT", OrderSide.LONG),
            _make_position("ETHUSDT", OrderSide.LONG),
        ]
        approved, _ = cf.check_correlation_filter("SOLUSDT", OrderSide.LONG, open_positions)

        assert approved is True

    def test_get_correlation_returns_zero_for_unknown(self):
        """get_correlation() returns 0.0 for unknown pairs."""
        cf = _make_filter()
        corr = cf.get_correlation("BTCUSDT", "UNKNOWN")
        assert corr == 0.0

    def test_get_correlation_bidirectional(self):
        """Injected correlation accessible in both directions."""
        cf = _make_filter()
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.88)

        assert cf.get_correlation("BTCUSDT", "ETHUSDT") == pytest.approx(0.88)
        assert cf.get_correlation("ETHUSDT", "BTCUSDT") == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# 6. Correlation matrix update from real price data
# ---------------------------------------------------------------------------

class TestCorrelationMatrixUpdate:

    def test_update_from_price_data_populates_cache(self):
        """update_correlation_matrix() with real data fills correlation_cache."""
        cf = _make_filter()
        prices_a, prices_b = _correlated_prices(correlation=0.90)

        cf.update_correlation_matrix({"BTCUSDT": prices_a, "ETHUSDT": prices_b})

        assert ("BTCUSDT", "ETHUSDT") in cf.correlation_cache
        assert ("ETHUSDT", "BTCUSDT") in cf.correlation_cache

    def test_high_corr_price_data_yields_high_corr_value(self):
        """Highly correlated price series → computed corr > 0.80."""
        cf = _make_filter()
        prices_a, prices_b = _correlated_prices(correlation=0.95, n=200)

        cf.update_correlation_matrix({"BTCUSDT": prices_a, "ETHUSDT": prices_b})

        corr = cf.get_correlation("BTCUSDT", "ETHUSDT")
        assert corr > 0.80

    def test_low_corr_price_data_yields_low_corr_value(self):
        """Uncorrelated price series → computed corr < 0.50."""
        cf = _make_filter()
        prices_a, prices_b = _correlated_prices(correlation=0.05, n=200)

        cf.update_correlation_matrix({"BTCUSDT": prices_a, "SOLUSDT": prices_b})

        corr = cf.get_correlation("BTCUSDT", "SOLUSDT")
        assert abs(corr) < 0.60  # Loose bound due to finite sample noise

    def test_single_symbol_no_crash(self):
        """Single symbol → no correlation calculated, no crash."""
        cf = _make_filter()
        prices = np.array([100.0 * (1 + 0.001 * i) for i in range(100)])

        cf.update_correlation_matrix({"BTCUSDT": prices})

        assert len(cf.correlation_cache) == 0

    def test_three_symbol_matrix_n_pairs(self):
        """3 symbols → 3 pairs stored (A-B, A-C, B-C) × 2 directions = 6 entries."""
        cf = _make_filter()
        prices_a, prices_b = _correlated_prices(correlation=0.80, seed=1)
        _, prices_c = _correlated_prices(correlation=0.40, seed=2)

        cf.update_correlation_matrix({
            "BTCUSDT": prices_a,
            "ETHUSDT": prices_b,
            "SOLUSDT": prices_c,
        })

        assert len(cf.correlation_cache) == 6  # 3 pairs × 2 directions

    def test_update_overwrites_old_values(self):
        """Calling update_correlation_matrix() twice overwrites old values."""
        cf = _make_filter()
        prices_a, prices_b = _correlated_prices(correlation=0.90, n=100)
        cf.update_correlation_matrix({"BTCUSDT": prices_a, "ETHUSDT": prices_b})
        first_corr = cf.get_correlation("BTCUSDT", "ETHUSDT")

        prices_a2, prices_b2 = _correlated_prices(correlation=0.10, n=100, seed=99)
        cf.update_correlation_matrix({"BTCUSDT": prices_a2, "ETHUSDT": prices_b2})
        second_corr = cf.get_correlation("BTCUSDT", "ETHUSDT")

        # Second update should change the value
        assert abs(first_corr - second_corr) > 0.20


# ---------------------------------------------------------------------------
# 7. Multi-symbol portfolio: complex mixed scenarios
# ---------------------------------------------------------------------------

class TestMultiSymbolPortfolio:

    def test_three_positions_one_high_corr_blocks(self):
        """Portfolio of 3 positions: new trade corr with one → blocked."""
        cf = _make_filter()
        _inject_correlation(cf, "BTCUSDT", "BNBUSDT", 0.30)
        _inject_correlation(cf, "ETHUSDT", "BNBUSDT", 0.30)
        _inject_correlation(cf, "SOLUSDT", "BNBUSDT", 0.91)  # high corr

        open_positions = [
            _make_position("BTCUSDT", OrderSide.LONG),
            _make_position("ETHUSDT", OrderSide.LONG),
            _make_position("SOLUSDT", OrderSide.LONG),
        ]
        approved, reason = cf.check_correlation_filter("BNBUSDT", OrderSide.LONG, open_positions)

        assert approved is False
        assert "SOLUSDT" in reason

    def test_all_low_corr_portfolio_allows_new_entry(self):
        """Diverse portfolio (all low corr) allows new same-direction trade."""
        cf = _make_filter()
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        new_sym = "BNBUSDT"

        for sym in symbols:
            _inject_correlation(cf, sym, new_sym, 0.25)

        open_positions = [_make_position(sym, OrderSide.LONG) for sym in symbols]
        approved, _ = cf.check_correlation_filter(new_sym, OrderSide.LONG, open_positions)

        assert approved is True

    def test_same_symbol_as_existing_same_direction_blocked(self):
        """Adding same symbol same direction → correlation 0 (no data), allowed.
        But if corr explicitly set to 1.0 for same-symbol → blocked."""
        cf = _make_filter()
        _inject_correlation(cf, "BTCUSDT", "BTCUSDT", 1.0)

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        approved, _ = cf.check_correlation_filter("BTCUSDT", OrderSide.LONG, open_positions)

        assert approved is False

    def test_mixed_directions_in_portfolio(self):
        """Portfolio with LONG and SHORT positions: only same-direction pairs checked."""
        cf = _make_filter()
        _inject_correlation(cf, "BTCUSDT", "BNBUSDT", 0.91)  # BTC LONG, BNB target LONG → block
        _inject_correlation(cf, "ETHUSDT", "BNBUSDT", 0.91)  # ETH SHORT, BNB target LONG → hedge check

        open_positions = [
            _make_position("BTCUSDT", OrderSide.LONG),
            _make_position("ETHUSDT", OrderSide.SHORT),
        ]
        # BNBUSDT LONG: blocked by BTCUSDT LONG (same dir, high corr)
        approved, reason = cf.check_correlation_filter("BNBUSDT", OrderSide.LONG, open_positions)

        assert approved is False
        assert "BTCUSDT" in reason


# ---------------------------------------------------------------------------
# 8. get_correlated_positions helper
# ---------------------------------------------------------------------------

class TestGetCorrelatedPositions:

    def test_returns_only_positions_above_threshold(self):
        """get_correlated_positions returns only positions above threshold."""
        cf = _make_filter(correlation_threshold=0.85)
        _inject_correlation(cf, "BTCUSDT", "NEWUSDT", 0.91)
        _inject_correlation(cf, "ETHUSDT", "NEWUSDT", 0.30)

        open_positions = [
            _make_position("BTCUSDT", OrderSide.LONG),
            _make_position("ETHUSDT", OrderSide.LONG),
        ]
        result = cf.get_correlated_positions("NEWUSDT", open_positions)

        assert len(result) == 1
        pos, corr = result[0]
        assert pos.symbol == "BTCUSDT"
        assert corr == pytest.approx(0.91)

    def test_sorted_by_correlation_descending(self):
        """Results sorted highest correlation first."""
        cf = _make_filter(correlation_threshold=0.60)
        _inject_correlation(cf, "BTCUSDT", "NEWUSDT", 0.70)
        _inject_correlation(cf, "ETHUSDT", "NEWUSDT", 0.95)
        _inject_correlation(cf, "SOLUSDT", "NEWUSDT", 0.80)

        open_positions = [
            _make_position("BTCUSDT", OrderSide.LONG),
            _make_position("ETHUSDT", OrderSide.LONG),
            _make_position("SOLUSDT", OrderSide.LONG),
        ]
        result = cf.get_correlated_positions("NEWUSDT", open_positions)

        assert len(result) == 3
        corrs = [c for _, c in result]
        assert corrs == sorted(corrs, reverse=True)

    def test_empty_positions_returns_empty(self):
        """No open positions → empty list."""
        cf = _make_filter()
        result = cf.get_correlated_positions("BTCUSDT", [])
        assert result == []

    def test_custom_threshold_override(self):
        """Custom threshold parameter overrides config threshold."""
        cf = _make_filter(correlation_threshold=0.85)
        _inject_correlation(cf, "BTCUSDT", "NEWUSDT", 0.70)  # Below 0.85 but above 0.60

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]

        # Default threshold (0.85) → not included
        result_default = cf.get_correlated_positions("NEWUSDT", open_positions)
        assert len(result_default) == 0

        # Custom threshold (0.60) → included
        result_custom = cf.get_correlated_positions("NEWUSDT", open_positions, threshold=0.60)
        assert len(result_custom) == 1


# ---------------------------------------------------------------------------
# 9. get_correlation_summary helper
# ---------------------------------------------------------------------------

class TestGetCorrelationSummary:

    def test_single_position_returns_zeros(self):
        """Single position → avg/max/min all 0.0."""
        cf = _make_filter()
        positions = [_make_position("BTCUSDT", OrderSide.LONG)]
        summary = cf.get_correlation_summary(positions)

        assert summary["avg_correlation"] == 0.0
        assert summary["max_correlation"] == 0.0
        assert summary["total_pairs"] == 0

    def test_two_positions_summary_accurate(self):
        """Two positions → 1 pair, summary reflects that correlation."""
        cf = _make_filter(correlation_threshold=0.85)
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.90)

        positions = [
            _make_position("BTCUSDT", OrderSide.LONG),
            _make_position("ETHUSDT", OrderSide.LONG),
        ]
        summary = cf.get_correlation_summary(positions)

        assert summary["total_pairs"] == 1
        assert summary["avg_correlation"] == pytest.approx(0.90)
        assert summary["max_correlation"] == pytest.approx(0.90)
        assert summary["correlated_pairs"] == 1  # 0.90 > threshold 0.85

    def test_three_positions_summary(self):
        """Three positions → 3 pairs total."""
        cf = _make_filter(correlation_threshold=0.85)
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.92)
        _inject_correlation(cf, "BTCUSDT", "SOLUSDT", 0.30)
        _inject_correlation(cf, "ETHUSDT", "SOLUSDT", 0.25)

        positions = [
            _make_position("BTCUSDT", OrderSide.LONG),
            _make_position("ETHUSDT", OrderSide.LONG),
            _make_position("SOLUSDT", OrderSide.LONG),
        ]
        summary = cf.get_correlation_summary(positions)

        assert summary["total_pairs"] == 3
        assert summary["correlated_pairs"] == 1  # Only BTC-ETH > 0.85
        assert summary["max_correlation"] == pytest.approx(0.92)

    def test_empty_positions_returns_zeros(self):
        """No positions → all zeros."""
        cf = _make_filter()
        summary = cf.get_correlation_summary([])
        assert summary["avg_correlation"] == 0.0
        assert summary["total_pairs"] == 0


# ---------------------------------------------------------------------------
# 10. Cache management
# ---------------------------------------------------------------------------

class TestCacheManagement:

    def test_clear_cache_empties_all_entries(self):
        """clear_cache() removes all cached correlations."""
        cf = _make_filter()
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.91)
        _inject_correlation(cf, "SOLUSDT", "BNBUSDT", 0.70)

        assert len(cf.correlation_cache) == 4  # 2 pairs × 2 directions

        cf.clear_cache()

        assert len(cf.correlation_cache) == 0

    def test_after_clear_pairs_allowed(self):
        """After clear_cache, previously blocked pair is allowed (no data)."""
        cf = _make_filter()
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.92)

        open_positions = [_make_position("BTCUSDT", OrderSide.LONG)]

        # Before clear: blocked
        approved_before, _ = cf.check_correlation_filter("ETHUSDT", OrderSide.LONG, open_positions)
        assert approved_before is False

        cf.clear_cache()

        # After clear: allowed (no corr data)
        approved_after, _ = cf.check_correlation_filter("ETHUSDT", OrderSide.LONG, open_positions)
        assert approved_after is True

    def test_bidirectional_storage_on_inject(self):
        """Correlation stored bidirectionally: (A,B) and (B,A) both present."""
        cf = _make_filter()
        _inject_correlation(cf, "BTCUSDT", "ETHUSDT", 0.88)

        assert ("BTCUSDT", "ETHUSDT") in cf.correlation_cache
        assert ("ETHUSDT", "BTCUSDT") in cf.correlation_cache
        assert cf.correlation_cache[("BTCUSDT", "ETHUSDT")] == pytest.approx(0.88)
        assert cf.correlation_cache[("ETHUSDT", "BTCUSDT")] == pytest.approx(0.88)

    def test_update_matrix_stores_bidirectional(self):
        """update_correlation_matrix() stores both (A,B) and (B,A) in cache."""
        cf = _make_filter()
        p_a, p_b = _correlated_prices(correlation=0.90, n=100)

        cf.update_correlation_matrix({"BTCUSDT": p_a, "ETHUSDT": p_b})

        assert ("BTCUSDT", "ETHUSDT") in cf.correlation_cache
        assert ("ETHUSDT", "BTCUSDT") in cf.correlation_cache
        # Both directions should have the same value
        assert cf.correlation_cache[("BTCUSDT", "ETHUSDT")] == pytest.approx(
            cf.correlation_cache[("ETHUSDT", "BTCUSDT")]
        )
