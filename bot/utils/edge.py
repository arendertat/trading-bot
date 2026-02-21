"""Utilities for cost-aware trade gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from bot.core.constants import OrderSide
from bot.config.models import CostGateConfig
from bot.strategies.base import FeatureSet


@dataclass
class CostGateResult:
    estimated_cost_usd: float
    estimated_cost_r: float
    setup_quality_score: float
    expected_edge_r: float


def compute_setup_quality(
    features: FeatureSet,
    side: OrderSide,
    config: CostGateConfig,
) -> float:
    """Compute setup quality score (0-1)."""
    weights = config.setup_quality_weights
    w_sum = weights.rsi + weights.ema_alignment + weights.adx + weights.trend_alignment + weights.bb_width
    if w_sum <= 0:
        return 0.0

    # RSI score: distance to ideal band center
    rsi_mid = (config.rsi_ideal_min + config.rsi_ideal_max) / 2.0
    rsi_half = max((config.rsi_ideal_max - config.rsi_ideal_min) / 2.0, 1e-9)
    rsi_dist = abs(features.rsi_5m - rsi_mid) / rsi_half
    rsi_score = max(0.0, min(1.0, 1.0 - rsi_dist))

    # EMA alignment (5m)
    ema_align = 0.0
    if side == OrderSide.LONG and features.ema20_5m >= features.ema50_5m:
        ema_align = 1.0
    if side == OrderSide.SHORT and features.ema20_5m <= features.ema50_5m:
        ema_align = 1.0

    # EMA distance from price
    ema_dist = abs(features.ema20_5m - features.ema50_5m) / max(features.ema20_5m, 1e-9)
    ema_dist_score = max(0.0, min(1.0, 1.0 - (ema_dist / max(config.ema_alignment_max_pct, 1e-9))))
    ema_score = min(ema_align, ema_dist_score)

    # ADX score
    adx_val = features.adx_5m if features.adx_5m is not None else 0.0
    adx_score = _normalize(adx_val, config.adx_min, config.adx_max)

    # 1h trend alignment
    trend_align = 0.0
    if side == OrderSide.LONG and features.ema20_1h >= features.ema50_1h:
        trend_align = 1.0
    if side == OrderSide.SHORT and features.ema20_1h <= features.ema50_1h:
        trend_align = 1.0

    # BB width score: prefer within configured band
    bb_width = features.bb_width_5m if features.bb_width_5m is not None else 0.0
    bb_score = _band_score(bb_width, config.bb_width_min, config.bb_width_max)

    score = (
        rsi_score * weights.rsi
        + ema_score * weights.ema_alignment
        + adx_score * weights.adx
        + trend_align * weights.trend_alignment
        + bb_score * weights.bb_width
    ) / w_sum
    return max(0.0, min(1.0, score))


def estimate_cost_gate(
    risk_usd: float,
    notional_usd: float,
    config: CostGateConfig,
    setup_quality_score: float,
) -> CostGateResult:
    """Estimate cost and expected edge in R terms."""
    estimated_fee_usd = notional_usd * (config.estimated_entry_fee_pct + config.estimated_exit_fee_pct)
    estimated_slippage_usd = notional_usd * config.estimated_slippage_pct * 2.0
    estimated_funding_usd = notional_usd * config.estimated_funding_pct
    estimated_cost_usd = estimated_fee_usd + estimated_slippage_usd + estimated_funding_usd
    estimated_cost_r = estimated_cost_usd / risk_usd if risk_usd > 0 else 0.0

    expected_edge_r = config.base_edge_r * setup_quality_score

    return CostGateResult(
        estimated_cost_usd=estimated_cost_usd,
        estimated_cost_r=estimated_cost_r,
        setup_quality_score=setup_quality_score,
        expected_edge_r=expected_edge_r,
    )


def _normalize(value: float, vmin: float, vmax: float) -> float:
    if vmax == vmin:
        return 0.0
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))


def _band_score(value: float, vmin: float, vmax: float) -> float:
    if vmin <= value <= vmax:
        return 1.0
    if value < vmin:
        return max(0.0, 1.0 - (vmin - value) / max(vmin, 1e-9))
    return max(0.0, 1.0 - (value - vmax) / max(vmax, 1e-9))
