"""Tests for cost gate conditions."""

from bot.config.models import CostGateConfig, GatesConfig
from bot.core.constants import OrderSide
from bot.strategies.base import FeatureSet
from bot.utils.edge import compute_setup_quality, estimate_cost_gate, passes_cost_gate


def _make_features() -> FeatureSet:
    return FeatureSet(
        rsi_5m=50.0,
        ema20_5m=105.0,
        ema50_5m=100.0,
        ema20_1h=105.0,
        ema50_1h=100.0,
        atr_5m=2.0,
        bb_upper_5m=110.0,
        bb_lower_5m=90.0,
        bb_middle_5m=100.0,
        high_20_bars=110.0,
        low_20_bars=90.0,
        volume_z_5m=0.0,
        adx_5m=30.0,
        bb_width_5m=0.03,
    )


def test_cost_gate_passes_when_net_edge_and_mult_ok():
    features = _make_features()
    cost_cfg = CostGateConfig(
        base_edge_r=1.0,
        estimated_entry_fee_pct=0.0,
        estimated_exit_fee_pct=0.0,
        estimated_slippage_pct=0.0,
        estimated_funding_pct=0.0,
    )
    gates = GatesConfig(
        cost_gate_enabled=True,
        min_net_edge_r=0.2,
        min_edge_over_cost_mult=1.5,
    )
    setup_quality = compute_setup_quality(features, OrderSide.LONG, cost_cfg)
    cost_gate = estimate_cost_gate(
        risk_usd=100.0,
        notional_usd=1000.0,
        config=cost_cfg,
        setup_quality_score=setup_quality,
    )
    assert passes_cost_gate(cost_gate, gates) is True


def test_cost_gate_fails_on_min_net_edge():
    features = _make_features()
    cost_cfg = CostGateConfig(
        base_edge_r=0.1,
        estimated_entry_fee_pct=0.0,
        estimated_exit_fee_pct=0.0,
        estimated_slippage_pct=0.0,
        estimated_funding_pct=0.0,
    )
    gates = GatesConfig(
        cost_gate_enabled=True,
        min_net_edge_r=0.2,
        min_edge_over_cost_mult=1.5,
    )
    setup_quality = compute_setup_quality(features, OrderSide.LONG, cost_cfg)
    cost_gate = estimate_cost_gate(
        risk_usd=100.0,
        notional_usd=1000.0,
        config=cost_cfg,
        setup_quality_score=setup_quality,
    )
    assert passes_cost_gate(cost_gate, gates) is False


def test_cost_gate_fails_on_edge_over_cost_multiplier():
    features = _make_features()
    cost_cfg = CostGateConfig(
        base_edge_r=1.0,
        estimated_entry_fee_pct=0.0,
        estimated_exit_fee_pct=0.0,
        estimated_slippage_pct=0.01,
        estimated_funding_pct=0.0,
    )
    gates = GatesConfig(
        cost_gate_enabled=True,
        min_net_edge_r=0.0,
        min_edge_over_cost_mult=1.5,
    )
    setup_quality = compute_setup_quality(features, OrderSide.LONG, cost_cfg)
    cost_gate = estimate_cost_gate(
        risk_usd=100.0,
        notional_usd=10000.0,
        config=cost_cfg,
        setup_quality_score=setup_quality,
    )
    assert passes_cost_gate(cost_gate, gates) is False
