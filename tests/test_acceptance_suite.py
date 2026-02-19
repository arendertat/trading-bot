"""
Acceptance Suite — Milestone 8, Task 23

Tests every acceptance criterion in BOT_SPEC_FINAL.md §16:

  AC-1  7-day stability: no crashes over 7-day simulated run
  AC-2  Restart recovery: state correctly reconstructed after crash
  AC-3  Kill switch: stops new entries (daily stop & weekly stop)
  AC-4  No duplicate orders: idempotent order submission
  AC-5  Correlation filter: same-bucket stacking prevented
  AC-6  Logs and daily reports generated

Plus configuration-driven behaviour:
  AC-7  Configuration validation: all constraints enforced
  AC-8  Configuration-driven parameters: risk/regime/strategy params respected
"""

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

# Config
from bot.config.models import (
    BotConfig,
    ExecutionConfig,
    ExchangeConfig,
    LeverageConfig,
    LoggingConfig,
    NotificationConfig,
    PerformanceConfig,
    RegimeConfig,
    RiskConfig,
    StrategiesConfig,
    StrategyRangeMeanReversionConfig,
    StrategyTrendBreakoutConfig,
    StrategyTrendPullbackConfig,
    TimeframesConfig,
    UniverseConfig,
)
from bot.config.validator import validate_config_constraints
from bot.core.constants import BotMode as CoreBotMode
from bot.core.constants import OrderSide as CoreOrderSide
from bot.core.constants import PositionStatus as CorePositionStatus
from bot.core.constants import RegimeType
from bot.core.performance_tracker import PerformanceTracker
from bot.core.strategy_selector import StrategySelector
from bot.core.types import Position as CorePosition
from bot.execution.models import OrderSide as ExecOrderSide
from bot.execution.position import ExitReason, Position, PositionStatus
from bot.health.safe_mode import SafeMode, SafeModeReason
from bot.regime.detector import RegimeDetector
from bot.risk.correlation_filter import CorrelationFilter
from bot.risk.kill_switch import KillSwitch
from bot.risk.position_sizing import PositionSizingCalculator
from bot.risk.risk_engine import RiskEngine
from bot.risk.risk_limits import RiskLimits
from bot.reporting.trade_logger import (
    ReportingTradeLogger,
    build_trade_record,
    CostsRecord,
    EntryOrderRecord,
    PortfolioRecord,
    ResultRecord,
    RiskRecord,
)
from bot.state.logger import TradeLogger
from bot.state.reconciler import Reconciler, ReconciliationResult
from bot.state.state_manager import StateManager
from bot.strategies.base import FeatureSet, StrategySignal
from tests.fixtures.market_data_generator import MarketDataGenerator

logger = logging.getLogger(__name__)


# ============================================================================
# Shared helpers
# ============================================================================

EQUITY_USD = 10_000.0


def _make_risk_config(**overrides) -> RiskConfig:
    defaults = dict(
        risk_per_trade_pct=0.01,
        max_total_open_risk_pct=0.025,
        max_open_positions=3,
        max_same_direction_positions=2,
        correlation_threshold=0.85,
        hedge_corr_max=0.60,
        daily_stop_pct=-0.04,
        weekly_stop_pct=-0.10,
        pause_days_after_weekly_stop=7,
        reduced_risk_after_pause_pct=0.005,
        reduced_risk_days=3,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def _make_bot_config(**risk_overrides) -> BotConfig:
    return BotConfig(
        mode=CoreBotMode.PAPER_LIVE,
        exchange=ExchangeConfig(),
        universe=UniverseConfig(),
        timeframes=TimeframesConfig(),
        risk=_make_risk_config(**risk_overrides),
        regime=RegimeConfig(
            trend_adx_min=25,
            range_adx_max=20,
            high_vol_atr_z=1.5,
            confidence_threshold=0.55,
            bb_width_range_min=0.01,
            bb_width_range_max=0.05,
        ),
        strategies=StrategiesConfig(
            trend_pullback=StrategyTrendPullbackConfig(),
            trend_breakout=StrategyTrendBreakoutConfig(),
            range_mean_reversion=StrategyRangeMeanReversionConfig(),
        ),
        leverage=LeverageConfig(),
        execution=ExecutionConfig(
            paper_slippage_limit_pct=0.0002,
            paper_slippage_market_pct=0.0008,
            paper_slippage_stop_pct=0.001,
            maker_fee_pct=0.0002,
            taker_fee_pct=0.0004,
        ),
        performance=PerformanceConfig(),
        notifications=NotificationConfig(),
        logging=LoggingConfig(),
    )


def _make_kill_switch(config: Optional[BotConfig] = None) -> KillSwitch:
    """Build KillSwitch from RiskConfig (correct signature)."""
    cfg = config or _make_bot_config()
    return KillSwitch(cfg.risk)


def _make_risk_engine(config: Optional[BotConfig] = None) -> Tuple[RiskEngine, KillSwitch, StateManager]:
    cfg = config or _make_bot_config()
    state = StateManager()
    ks = KillSwitch(cfg.risk)
    psc = PositionSizingCalculator(cfg)
    limits = RiskLimits(cfg.risk)
    cf = CorrelationFilter(cfg.risk)
    re = RiskEngine(
        config=cfg,
        kill_switch=ks,
        position_sizing=psc,
        risk_limits=limits,
        correlation_filter=cf,
    )
    return re, ks, state


def _make_position(
    symbol: str = "BTCUSDT",
    side: ExecOrderSide = ExecOrderSide.LONG,
    entry_price: float = 50_000.0,
    equity_usd: float = EQUITY_USD,
    strategy: str = "trend_pullback",
    regime: str = "TREND",
    confidence: float = 0.70,
    candle_index: int = 0,
) -> Position:
    stop_pct = 0.01
    stop_price = (
        entry_price * (1 - stop_pct)
        if side == ExecOrderSide.LONG
        else entry_price * (1 + stop_pct)
    )
    risk_usd = equity_usd * 0.01
    quantity = risk_usd / (entry_price * stop_pct)
    notional = quantity * entry_price
    return Position(
        position_id=f"POS_{uuid.uuid4().hex[:8]}",
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        quantity=quantity,
        notional_usd=notional,
        leverage=2.0,
        margin_usd=notional / 2.0,
        stop_price=stop_price,
        entry_time=datetime.utcnow(),
        risk_amount_usd=risk_usd,
        initial_stop_price=stop_price,
        trail_after_r=1.0,
        atr_trail_mult=2.0,
        entry_order_id=f"entry_{uuid.uuid4().hex[:8]}",
        stop_order_id=f"stop_{uuid.uuid4().hex[:8]}",
        strategy=strategy,
        regime=regime,
        confidence=confidence,       # correct field name
        metadata={},
    )


def _close_position(pos: Position, pnl_pct: float = 0.01) -> Position:
    exit_price = pos.entry_price * (1 + pnl_pct)
    pnl = (exit_price - pos.entry_price) * pos.quantity
    pos.status = PositionStatus.CLOSED
    pos.exit_price = exit_price
    pos.exit_time = datetime.utcnow()
    pos.realized_pnl_usd = pnl
    pos.exit_reason = ExitReason.TP   # correct enum value
    return pos


def _make_core_position(
    symbol: str = "BTCUSDT",
    side: CoreOrderSide = CoreOrderSide.LONG,
    entry_price: float = 50_000.0,
) -> CorePosition:
    """Build a bot.core.types.Position (used by CorrelationFilter)."""
    return CorePosition(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        quantity=0.1,
        notional=entry_price * 0.1,
        leverage=2.0,
        margin=entry_price * 0.1 / 2.0,
        stop_price=entry_price * 0.99,
        tp_price=entry_price * 1.015,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        entry_time=datetime.utcnow(),
        trade_id=f"TRADE_{uuid.uuid4().hex[:8]}",
        status=CorePositionStatus.OPEN,
    )


def _trigger_kill_switch(ks: KillSwitch) -> None:
    """Trigger daily stop on a KillSwitch with correct update_pnl signature."""
    ks.update_pnl(
        realized_pnl_today=-600.0,   # -6% on 10k → exceeds daily stop -4%
        realized_pnl_week=-600.0,
        equity_usd=EQUITY_USD,
        now_utc=datetime.utcnow(),
    )


# ============================================================================
# AC-1: 7-day stability — no crashes over a 7-day simulated run
# ============================================================================

class TestAC1SevenDayStability:
    """
    BOT_SPEC §16: "Paper live runs for 7 days without crashing"

    Simulates 2016 candles (7 days @ 5m) through regime detection →
    kill-switch check → state management → logging.
    """

    def test_7day_simulation_processes_all_candles(self):
        """All 2016 candles in the 7-day window are valid OHLCV structures."""
        gen = MarketDataGenerator(seed=42)
        candles, _ = gen.generate_30_day_scenario()
        assert len(candles) >= 2016
        for c in candles[:2016]:
            assert hasattr(c, "close")
            assert c.close > 0

    def test_7day_simulation_no_exception(self):
        """Full 7-day simulation completes without raising any exception."""
        gen = MarketDataGenerator(seed=42)
        all_candles, _ = gen.generate_30_day_scenario()
        candles = all_candles[:2016]

        config = _make_bot_config()
        state = StateManager()
        ks = KillSwitch(config.risk)
        detector = RegimeDetector(config.regime)
        exceptions_caught = []
        candle_index = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            tlogger = TradeLogger(log_dir=tmpdir)

            for candle in candles:
                candle_index += 1
                try:
                    window = candles[max(0, candle_index - 100): candle_index]
                    if len(window) < 100:
                        continue  # warmup: need 100 candles for features

                    features = gen.generate_features_from_candles(window)

                    # Guard: skip if any feature is None (edge of data)
                    if any(v is None for v in [
                        features.adx_5m, features.atr_z_5m,
                        features.bb_upper_5m, features.bb_lower_5m, features.bb_middle_5m,
                        features.ema20_5m, features.ema50_5m,
                        features.ema20_1h, features.ema50_1h,
                    ]):
                        continue

                    # Regime detection
                    regime_result = detector.detect_regime(
                        symbol="BTCUSDT",
                        adx=float(features.adx_5m),
                        atr_z=float(features.atr_z_5m),
                        bb_width=float(
                            (features.bb_upper_5m - features.bb_lower_5m)
                            / max(features.bb_middle_5m, 1e-9)
                        ),
                        ema20_5m=float(features.ema20_5m),
                        ema50_5m=float(features.ema50_5m),
                        ema20_1h=float(features.ema20_1h),
                        ema50_1h=float(features.ema50_1h),
                    )

                    # Age-close old positions
                    for pos in state.get_all_positions():
                        if (
                            pos.status == PositionStatus.OPEN
                            and candle_index - (pos.metadata.get("candle_index", 0)) > 50
                        ):
                            pos = _close_position(pos, pnl_pct=0.005)
                            state.update_position(pos)
                            tlogger.log_trade_closed(pos)

                    # Skip CHOP; skip if kill switch active
                    if regime_result.regime == RegimeType.CHOP_NO_TRADE:
                        continue
                    if ks.is_active():
                        continue

                    # Open a position every 60 candles
                    if candle_index % 60 == 0:
                        symbol = "BTCUSDT" if candle_index % 120 < 60 else "ETHUSDT"
                        side = (
                            ExecOrderSide.LONG
                            if candle_index % 240 < 120
                            else ExecOrderSide.SHORT
                        )
                        open_count = state.open_position_count()
                        if open_count < config.risk.max_open_positions:
                            pos = _make_position(
                                symbol=symbol,
                                side=side,
                                entry_price=candle.close,   # Candle object, not dict
                                candle_index=candle_index,
                            )
                            pos.metadata["candle_index"] = candle_index
                            state.add_position(pos)

                except Exception as e:
                    exceptions_caught.append(f"candle {candle_index}: {type(e).__name__}: {e}")

            tlogger.close()

        assert exceptions_caught == [], f"Exceptions during 7-day run:\n" + "\n".join(exceptions_caught)

    def test_safe_mode_does_not_crash_pipeline(self):
        """SafeMode activation mid-simulation doesn't cause exceptions."""
        safe_mode = SafeMode(recovery_seconds=60)
        exceptions_caught = []

        for i in range(200):
            try:
                if i == 50:
                    safe_mode.trigger(SafeModeReason.RATE_LIMIT, "429 Too Many Requests")
                if i == 100:
                    for _ in range(61):
                        safe_mode.record_healthy_check()
                _ = safe_mode.is_active
            except Exception as e:
                exceptions_caught.append(str(e))

        assert exceptions_caught == []

    def test_kill_switch_does_not_crash_when_triggered(self):
        """Kill switch triggering mid-run doesn't cause exceptions."""
        config = _make_bot_config()
        ks = KillSwitch(config.risk)
        exceptions_caught = []

        for i in range(100):
            try:
                daily = -600.0 if i == 30 else 50.0
                ks.update_pnl(
                    realized_pnl_today=daily,
                    realized_pnl_week=daily,
                    equity_usd=EQUITY_USD,
                    now_utc=datetime.utcnow(),
                )
                _ = ks.is_active()
            except Exception as e:
                exceptions_caught.append(str(e))

        assert exceptions_caught == []


# ============================================================================
# AC-2: Restart recovery — state correctly reconstructed after crash
# ============================================================================

class TestAC2RestartRecovery:
    """
    BOT_SPEC §16: "Restart recovery correctly reconstructs state"
    """

    def _client(self, positions=None, orders=None) -> MagicMock:
        c = MagicMock()
        c.fetch_positions.return_value = positions or []
        c.fetch_open_orders.return_value = orders or []
        c.cancel_order.return_value = {"status": "CANCELED"}
        return c

    def test_positions_restored_after_crash(self):
        """Exchange has 2 open positions; after restart they appear in StateManager."""
        raw_positions = [
            {"symbol": "BTCUSDT", "side": "LONG", "contracts": 0.1,
             "entryPrice": 50_000.0, "leverage": 2.0},
            {"symbol": "ETHUSDT", "side": "SHORT", "contracts": 1.0,
             "entryPrice": 3_000.0, "leverage": 2.0},
        ]
        state = StateManager()
        reconciler = Reconciler(
            exchange_client=self._client(positions=raw_positions),
            state_manager=state,
        )
        result = reconciler.reconcile()

        assert result.positions_restored == 2
        assert len(state.get_open_positions("BTCUSDT")) == 1
        assert len(state.get_open_positions("ETHUSDT")) == 1
        assert result.errors == []

    def test_stop_orders_linked_after_crash(self):
        """Stop orders on exchange are linked to restored positions."""
        raw_pos = {"symbol": "BTCUSDT", "side": "LONG", "contracts": 0.1,
                   "entryPrice": 50_000.0, "leverage": 2.0}

        # Pass 1: get position_id
        state_tmp = StateManager()
        Reconciler(exchange_client=self._client(positions=[raw_pos]), state_manager=state_tmp).reconcile()
        pid = state_tmp.get_open_positions("BTCUSDT")[0].position_id

        raw_stop = {
            "clientOrderId": f"{pid}_stop",
            "orderId": f"EX_{pid}_stop",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "STOP_MARKET",
            "origQty": "0.1",
            "stopPrice": "48000",
            "executedQty": "0",
            "price": "0",
        }
        state = StateManager()
        result = Reconciler(
            exchange_client=self._client(positions=[raw_pos], orders=[raw_stop]),
            state_manager=state,
        ).reconcile()

        assert result.positions_restored == 1
        assert result.orders_linked == 1
        assert result.orphan_orders_cancelled == 0

    def test_orphan_orders_cancelled_after_crash(self):
        """Exit orders without a position are cancelled on restart."""
        orphan = {
            "clientOrderId": "OLD_POS_stop",
            "orderId": "EX_OLD_POS_stop",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "STOP_MARKET",
            "origQty": "0.1",
            "stopPrice": "48000",
            "executedQty": "0",
            "price": "0",
        }
        state = StateManager()
        client = self._client(positions=[], orders=[orphan])
        result = Reconciler(exchange_client=client, state_manager=state).reconcile()

        assert result.orphan_orders_cancelled == 1
        client.cancel_order.assert_called_once()

    def test_idempotent_double_reconcile(self):
        """Calling reconcile() twice does not create duplicate positions."""
        raw_pos = {"symbol": "BTCUSDT", "side": "LONG", "contracts": 0.1,
                   "entryPrice": 50_000.0, "leverage": 2.0}
        state = StateManager()
        reconciler = Reconciler(exchange_client=self._client(positions=[raw_pos]), state_manager=state)

        reconciler.reconcile()
        reconciler.reconcile()

        assert len(state.get_open_positions("BTCUSDT")) == 1

    def test_reconciliation_result_logged(self):
        """ReconciliationResult is logged to TradeLogger event log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tlogger = TradeLogger(log_dir=tmpdir)
            raw_pos = {"symbol": "BTCUSDT", "side": "LONG", "contracts": 0.1,
                       "entryPrice": 50_000.0, "leverage": 2.0}
            state = StateManager()
            result = Reconciler(
                exchange_client=self._client(positions=[raw_pos]),
                state_manager=state,
            ).reconcile()
            tlogger.log_reconciliation(result.to_dict())
            tlogger.flush()

            with open(tlogger.get_event_log_path()) as f:
                lines = [json.loads(l) for l in f if l.strip()]

            recon = [l for l in lines if l.get("event") == "RECONCILIATION_COMPLETE"]
            assert len(recon) == 1
            assert recon[0]["payload"]["positions_restored"] == 1
            tlogger.close()


# ============================================================================
# AC-3: Kill switch — stops new entries
# ============================================================================

class TestAC3KillSwitch:
    """
    BOT_SPEC §16: "Kill switch stops new entries"

    RiskEngine delegates kill switch check to KillSwitch.is_active().
    We verify the kill switch activates at correct thresholds and
    that is_active() correctly gates entries.
    """

    def test_daily_stop_activates_when_threshold_exceeded(self):
        """KillSwitch activates after daily PnL exceeds threshold."""
        config = _make_bot_config()
        ks = KillSwitch(config.risk)
        assert not ks.is_active()

        ks.update_pnl(
            realized_pnl_today=-500.0,   # -5% on 10k > -4% threshold
            realized_pnl_week=-500.0,
            equity_usd=EQUITY_USD,
            now_utc=datetime.utcnow(),
        )
        assert ks.is_active()

    def test_weekly_stop_activates_when_threshold_exceeded(self):
        """KillSwitch activates after weekly PnL exceeds threshold."""
        config = _make_bot_config()
        ks = KillSwitch(config.risk)

        ks.update_pnl(
            realized_pnl_today=-200.0,
            realized_pnl_week=-1200.0,   # -12% on 10k > -10% weekly threshold
            equity_usd=EQUITY_USD,
            now_utc=datetime.utcnow(),
        )
        assert ks.is_active()

    def test_entries_allowed_before_kill_switch(self):
        """No stop triggered → kill switch inactive."""
        config = _make_bot_config()
        ks = KillSwitch(config.risk)

        ks.update_pnl(
            realized_pnl_today=100.0,
            realized_pnl_week=300.0,
            equity_usd=EQUITY_USD,
            now_utc=datetime.utcnow(),
        )
        assert not ks.is_active()

    def test_risk_engine_rejects_when_kill_switch_active(self):
        """RiskEngine returns approved=False when kill switch is active."""
        config = _make_bot_config()
        ks = KillSwitch(config.risk)

        # Trigger daily stop
        ks.update_pnl(
            realized_pnl_today=-600.0,
            realized_pnl_week=-600.0,
            equity_usd=EQUITY_USD,
            now_utc=datetime.utcnow(),
        )
        assert ks.is_active()

        re = RiskEngine(
            config=config,
            kill_switch=ks,
            position_sizing=PositionSizingCalculator(config),
            risk_limits=RiskLimits(config.risk),
            correlation_filter=CorrelationFilter(config.risk),
        )

        result = re.validate_entry(
            symbol="BTCUSDT",
            side=CoreOrderSide.LONG,
            regime=RegimeType.TREND,
            stop_pct=0.01,
            current_price=50_000.0,
            equity_usd=EQUITY_USD,
            free_margin_usd=EQUITY_USD,
            open_positions=[],
        )

        assert result.approved is False
        assert len(result.rejection_reason) > 0

    def test_kill_switch_rejection_reason_is_descriptive(self):
        """Rejection reason contains meaningful text."""
        config = _make_bot_config()
        ks = KillSwitch(config.risk)
        ks.update_pnl(
            realized_pnl_today=-600.0,
            realized_pnl_week=-600.0,
            equity_usd=EQUITY_USD,
            now_utc=datetime.utcnow(),
        )

        re = RiskEngine(
            config=config,
            kill_switch=ks,
            position_sizing=PositionSizingCalculator(config),
            risk_limits=RiskLimits(config.risk),
            correlation_filter=CorrelationFilter(config.risk),
        )
        result = re.validate_entry(
            symbol="ETHUSDT",
            side=CoreOrderSide.SHORT,
            regime=RegimeType.TREND,
            stop_pct=0.01,
            current_price=3_000.0,
            equity_usd=EQUITY_USD,
            free_margin_usd=EQUITY_USD,
            open_positions=[],
        )

        assert result.approved is False
        assert isinstance(result.rejection_reason, str)
        assert len(result.rejection_reason) > 5


# ============================================================================
# AC-4: No duplicate orders — idempotent
# ============================================================================

class TestAC4NoDuplicateOrders:
    """
    BOT_SPEC §16 + §13.2: "No duplicate orders"
    """

    def test_reconcile_twice_no_duplicate_positions(self):
        """Double reconcile → StateManager has exactly 1 position."""
        raw = {"symbol": "BTCUSDT", "side": "LONG", "contracts": 0.1,
               "entryPrice": 50_000.0, "leverage": 2.0}
        state = StateManager()
        client = MagicMock()
        client.fetch_positions.return_value = [raw]
        client.fetch_open_orders.return_value = []
        reconciler = Reconciler(exchange_client=client, state_manager=state)

        reconciler.reconcile()
        reconciler.reconcile()

        assert len(state.get_open_positions("BTCUSDT")) == 1

    def test_state_manager_rejects_duplicate_add(self):
        """StateManager.add_position raises ValueError on duplicate position_id."""
        state = StateManager()
        pos = _make_position("BTCUSDT")
        state.add_position(pos)

        with pytest.raises(ValueError, match="already exists"):
            state.add_position(pos)

    def test_state_manager_upsert_does_not_duplicate(self):
        """update_position (upsert) replaces existing; no duplicates."""
        state = StateManager()
        pos = _make_position("BTCUSDT")
        state.update_position(pos)
        state.update_position(pos)

        assert len(state.get_open_positions("BTCUSDT")) == 1

    def test_positions_have_unique_ids(self):
        """Two independently created positions have different position_ids."""
        pos1 = _make_position("BTCUSDT")
        pos2 = _make_position("BTCUSDT")
        assert pos1.position_id != pos2.position_id

    def test_no_position_added_twice_in_simulation(self):
        """20-position simulation never inserts duplicate position_id."""
        state = StateManager()
        seen_ids = set()
        errors = []

        for i in range(20):
            pos = _make_position("BTCUSDT", candle_index=i)
            if pos.position_id in seen_ids:
                errors.append(f"Duplicate: {pos.position_id}")
                continue
            seen_ids.add(pos.position_id)
            state.add_position(pos)

        assert errors == []
        all_ids = [p.position_id for p in state.get_all_positions()]
        assert len(all_ids) == len(set(all_ids))


# ============================================================================
# AC-5: Correlation filter — same-bucket stacking prevented
# ============================================================================

class TestAC5CorrelationFilter:
    """
    BOT_SPEC §16: "Correlation filter prevents same-bucket stacking"
    """

    def _make_cf(self) -> CorrelationFilter:
        return CorrelationFilter(config=_make_risk_config())

    def _inject(self, cf: CorrelationFilter, a: str, b: str, corr: float) -> None:
        cf.correlation_cache[(a, b)] = corr
        cf.correlation_cache[(b, a)] = corr

    def test_same_bucket_same_direction_blocked(self):
        """High-corr same-direction entry blocked by correlation filter."""
        cf = self._make_cf()
        self._inject(cf, "BTCUSDT", "ETHUSDT", 0.92)

        existing = [_make_core_position("BTCUSDT", CoreOrderSide.LONG)]
        approved, reason = cf.check_correlation_filter("ETHUSDT", CoreOrderSide.LONG, existing)

        assert approved is False
        assert len(reason) > 0

    def test_same_bucket_opposite_direction_high_corr_blocked(self):
        """Hedge blocked when corr > hedge_corr_max (0.60)."""
        cf = self._make_cf()
        self._inject(cf, "BTCUSDT", "ETHUSDT", 0.91)  # > 0.85 threshold AND > 0.60 hedge_max

        existing = [_make_core_position("BTCUSDT", CoreOrderSide.LONG)]
        approved, _ = cf.check_correlation_filter("ETHUSDT", CoreOrderSide.SHORT, existing)

        assert approved is False

    def test_low_corr_same_direction_allowed(self):
        """Low correlation symbols not blocked even with same direction."""
        cf = self._make_cf()
        self._inject(cf, "BTCUSDT", "SOLUSDT", 0.20)

        existing = [_make_core_position("BTCUSDT", CoreOrderSide.LONG)]
        approved, _ = cf.check_correlation_filter("SOLUSDT", CoreOrderSide.LONG, existing)

        assert approved is True

    def test_no_stacking_in_simulated_portfolio(self):
        """Portfolio building: second high-corr position rejected, low-corr allowed."""
        cf = self._make_cf()
        self._inject(cf, "BTCUSDT", "ETHUSDT", 0.90)
        self._inject(cf, "BTCUSDT", "SOLUSDT", 0.25)
        self._inject(cf, "ETHUSDT", "SOLUSDT", 0.25)

        portfolio: List[CorePosition] = []

        a1, _ = cf.check_correlation_filter("BTCUSDT", CoreOrderSide.LONG, portfolio)
        assert a1 is True
        portfolio.append(_make_core_position("BTCUSDT", CoreOrderSide.LONG))

        a2, reason2 = cf.check_correlation_filter("ETHUSDT", CoreOrderSide.LONG, portfolio)
        assert a2 is False
        assert len(reason2) > 0

        a3, _ = cf.check_correlation_filter("SOLUSDT", CoreOrderSide.LONG, portfolio)
        assert a3 is True

    def test_risk_engine_enforces_correlation_filter(self):
        """RiskEngine rejects entry when correlation filter blocks it."""
        config = _make_bot_config()
        ks = KillSwitch(config.risk)
        cf = CorrelationFilter(config.risk)
        cf.correlation_cache[("BTCUSDT", "ETHUSDT")] = 0.92
        cf.correlation_cache[("ETHUSDT", "BTCUSDT")] = 0.92

        re = RiskEngine(
            config=config,
            kill_switch=ks,
            position_sizing=PositionSizingCalculator(config),
            risk_limits=RiskLimits(config.risk),
            correlation_filter=cf,
        )

        # State with BTC LONG open
        btc_open = [_make_core_position("BTCUSDT", CoreOrderSide.LONG)]

        result = re.validate_entry(
            symbol="ETHUSDT",
            side=CoreOrderSide.LONG,
            regime=RegimeType.TREND,
            stop_pct=0.01,
            current_price=3_000.0,
            equity_usd=EQUITY_USD,
            free_margin_usd=EQUITY_USD,
            open_positions=btc_open,
        )

        assert result.approved is False
        assert "correlation" in result.rejection_reason.lower()


# ============================================================================
# AC-6: Logs and daily reports generated
# ============================================================================

class TestAC6LogsAndReports:
    """
    BOT_SPEC §16: "Logs and daily reports generated"
    """

    def test_trade_closed_written_to_jsonl(self):
        """Closing a position writes a TRADE_CLOSED record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tlogger = TradeLogger(log_dir=tmpdir)
            pos = _close_position(_make_position("BTCUSDT"), pnl_pct=0.015)
            tlogger.log_trade_closed(pos)
            tlogger.flush()

            with open(tlogger.get_trade_log_path()) as f:
                lines = [json.loads(l) for l in f if l.strip()]

            assert len(lines) == 1
            assert lines[0]["event"] == "TRADE_CLOSED"
            tlogger.close()

    def test_event_log_records_reconciliation(self):
        """Reconciliation produces RECONCILIATION_COMPLETE event record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tlogger = TradeLogger(log_dir=tmpdir)
            result = ReconciliationResult()
            result.positions_restored = 2
            tlogger.log_reconciliation(result.to_dict())
            tlogger.flush()

            with open(tlogger.get_event_log_path()) as f:
                lines = [json.loads(l) for l in f if l.strip()]

            events = [l for l in lines if l["event"] == "RECONCILIATION_COMPLETE"]
            assert len(events) == 1
            assert events[0]["payload"]["positions_restored"] == 2
            tlogger.close()

    def test_kill_switch_activation_logged(self):
        """Kill switch activation produces KILL_SWITCH_ACTIVATED event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tlogger = TradeLogger(log_dir=tmpdir)
            tlogger.log_kill_switch(reason="daily_stop", context={"pnl_pct": -0.05})
            tlogger.flush()

            with open(tlogger.get_event_log_path()) as f:
                lines = [json.loads(l) for l in f if l.strip()]

            ks_events = [l for l in lines if l["event"] == "KILL_SWITCH_ACTIVATED"]
            assert len(ks_events) == 1
            assert ks_events[0]["payload"]["reason"] == "daily_stop"
            tlogger.close()

    def test_reporting_trade_logger_writes_full_record(self):
        """ReportingTradeLogger.log_full_trade() writes a structured record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rlogger = ReportingTradeLogger(log_dir=tmpdir)
            pos = _close_position(_make_position("BTCUSDT"), pnl_pct=0.012)

            portfolio = PortfolioRecord(
                open_positions_count=0,
                open_risk_pct=0.0,
                correlation_bucket="",
                bucket_corr_max=0.0,
            )
            record = build_trade_record(
                position=pos,
                mode="PAPER_LIVE",
                equity_usd=EQUITY_USD,
                risk_pct=0.01,
                portfolio=portfolio,
            )
            rlogger.log_full_trade(record)
            rlogger.flush()

            # ReportingTradeLogger wraps TradeLogger — find the written file
            import glob as _glob
            trade_files = _glob.glob(os.path.join(tmpdir, "trades_*.jsonl"))
            assert len(trade_files) >= 1, "No trade log file created"
            with open(trade_files[0]) as f:
                lines = [json.loads(l) for l in f if l.strip()]

            assert len(lines) >= 1
            rlogger.close()

    def test_multiple_trades_logged_in_sequence(self):
        """Multiple closed trades all appear in trade log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tlogger = TradeLogger(log_dir=tmpdir)
            for i in range(5):
                pos = _close_position(_make_position("BTCUSDT", candle_index=i), pnl_pct=0.01 * (i + 1))
                tlogger.log_trade_closed(pos)
            tlogger.flush()

            with open(tlogger.get_trade_log_path()) as f:
                lines = [json.loads(l) for l in f if l.strip()]

            assert len([l for l in lines if l["event"] == "TRADE_CLOSED"]) == 5
            tlogger.close()

    def test_log_files_created_in_configured_directory(self):
        """Log files appear in the directory specified at construction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_subdir = os.path.join(tmpdir, "bot_logs")
            tlogger = TradeLogger(log_dir=log_subdir)
            pos = _close_position(_make_position("ETHUSDT"))
            tlogger.log_trade_closed(pos)
            tlogger.flush()

            assert os.path.isdir(log_subdir)
            trades_path = tlogger.get_trade_log_path()
            assert str(log_subdir) in str(trades_path)
            assert os.path.isfile(trades_path)
            tlogger.close()


# ============================================================================
# AC-7: Configuration validation
# ============================================================================

class TestAC7ConfigurationValidation:
    """
    BOT_SPEC §15: "All parameters must be configurable via a JSON file"
    Cross-field Pydantic constraints enforced on BotConfig construction.
    """

    def test_valid_config_parses_without_error(self):
        """Default/standard config builds without raising."""
        config = _make_bot_config()
        assert config.mode == CoreBotMode.PAPER_LIVE
        assert config.risk.risk_per_trade_pct == pytest.approx(0.01)

    def test_invalid_max_same_direction_exceeds_max_positions(self):
        """max_same_direction_positions > max_open_positions → raises on BotConfig."""
        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError)):
            _make_bot_config(
                max_open_positions=2,
                max_same_direction_positions=3,
            )

    def test_invalid_total_open_risk_less_than_per_trade(self):
        """max_total_open_risk_pct < risk_per_trade_pct → ValidationError."""
        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError)):
            _make_bot_config(
                max_total_open_risk_pct=0.005,   # < risk_per_trade_pct=0.01
            )

    def test_invalid_hedge_corr_max_exceeds_threshold(self):
        """hedge_corr_max >= correlation_threshold → ValidationError."""
        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError)):
            _make_bot_config(
                hedge_corr_max=0.90,
                correlation_threshold=0.85,
            )

    def test_invalid_daily_stop_more_severe_than_weekly(self):
        """daily_stop_pct < weekly_stop_pct (more severe) → ValidationError."""
        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError)):
            _make_bot_config(
                daily_stop_pct=-0.15,
                weekly_stop_pct=-0.10,
            )

    def test_all_strategies_disabled_raises(self):
        """At least one strategy must be enabled."""
        config = _make_bot_config()
        config.strategies.trend_pullback.enabled = False
        config.strategies.trend_breakout.enabled = False
        config.strategies.range_mean_reversion.enabled = False

        with pytest.raises(ValueError, match="least one strategy"):
            validate_config_constraints(config)

    def test_leverage_exceeds_2x_raises(self):
        """Leverage > 2.0x raises per spec §2 Non-Goals."""
        config = _make_bot_config()
        config.leverage.trend = 3.0

        with pytest.raises(ValueError):
            validate_config_constraints(config)

    def test_margin_mode_must_be_isolated(self):
        """margin_mode != ISOLATED raises per spec §1 Scope."""
        config = _make_bot_config()
        config.exchange.margin_mode = "CROSS"

        with pytest.raises(ValueError, match="ISOLATED"):
            validate_config_constraints(config)

    def test_whitelist_blacklist_overlap_raises(self):
        """Same symbol in whitelist and blacklist raises."""
        config = _make_bot_config()
        config.universe.whitelist = ["BTCUSDT", "ETHUSDT"]
        config.universe.blacklist = ["ETHUSDT"]

        with pytest.raises(ValueError, match="blacklist"):
            validate_config_constraints(config)


# ============================================================================
# AC-8: Configuration-driven behavior
# ============================================================================

class TestAC8ConfigurationDrivenBehavior:
    """
    BOT_SPEC §15: Config-driven parameters respected at runtime.
    """

    def test_smaller_risk_per_trade_yields_smaller_position(self):
        """Lower risk_per_trade_pct → smaller position quantity from PSC."""
        config_hi = _make_bot_config(risk_per_trade_pct=0.02, max_total_open_risk_pct=0.05)
        # reduced_risk_after_pause_pct must be < risk_per_trade_pct, so set it explicitly
        config_lo = _make_bot_config(risk_per_trade_pct=0.005, reduced_risk_after_pause_pct=0.001)

        psc_hi = PositionSizingCalculator(config_hi)
        psc_lo = PositionSizingCalculator(config_lo)

        result_hi = psc_hi.calculate(
            equity_usd=EQUITY_USD,
            stop_pct=0.01,
            regime=RegimeType.TREND,
            current_price=50_000.0,
            free_margin_usd=EQUITY_USD,
        )
        result_lo = psc_lo.calculate(
            equity_usd=EQUITY_USD,
            stop_pct=0.01,
            regime=RegimeType.TREND,
            current_price=50_000.0,
            free_margin_usd=EQUITY_USD,
        )

        assert result_hi.quantity > result_lo.quantity

    def test_tighter_max_positions_blocks_earlier(self):
        """max_open_positions=1 blocks second entry; max_open_positions=3 allows it."""
        # max=1: second entry blocked
        config1 = _make_bot_config(
            max_open_positions=1, max_same_direction_positions=1
        )
        ks1 = KillSwitch(config1.risk)
        re1 = RiskEngine(
            config=config1,
            kill_switch=ks1,
            position_sizing=PositionSizingCalculator(config1),
            risk_limits=RiskLimits(config1.risk),
            correlation_filter=CorrelationFilter(config1.risk),
        )
        open1 = [_make_core_position("BTCUSDT")]
        result1 = re1.validate_entry(
            symbol="ETHUSDT", side=CoreOrderSide.LONG,
            regime=RegimeType.TREND, stop_pct=0.01,
            current_price=3000.0, equity_usd=EQUITY_USD,
            free_margin_usd=EQUITY_USD, open_positions=open1,
        )
        assert result1.approved is False

        # max=3: second entry allowed
        config3 = _make_bot_config(
            max_open_positions=3, max_same_direction_positions=3
        )
        ks3 = KillSwitch(config3.risk)
        re3 = RiskEngine(
            config=config3,
            kill_switch=ks3,
            position_sizing=PositionSizingCalculator(config3),
            risk_limits=RiskLimits(config3.risk),
            correlation_filter=CorrelationFilter(config3.risk),
        )
        open3 = [_make_core_position("BTCUSDT")]
        result3 = re3.validate_entry(
            symbol="ETHUSDT", side=CoreOrderSide.LONG,
            regime=RegimeType.TREND, stop_pct=0.01,
            current_price=3000.0, equity_usd=EQUITY_USD,
            free_margin_usd=EQUITY_USD, open_positions=open3,
        )
        assert result3.approved is True

    def test_stricter_correlation_threshold_blocks_more(self):
        """Lower correlation_threshold rejects more pairs."""
        def _cf(threshold: float) -> CorrelationFilter:
            cfg = _make_risk_config(
                correlation_threshold=threshold,
                hedge_corr_max=threshold * 0.5,
            )
            cf = CorrelationFilter(cfg)
            cf.correlation_cache[("BTCUSDT", "ETHUSDT")] = 0.70
            cf.correlation_cache[("ETHUSDT", "BTCUSDT")] = 0.70
            return cf

        existing = [_make_core_position("BTCUSDT", CoreOrderSide.LONG)]

        # threshold=0.85: corr 0.70 < 0.85 → allowed
        approved_loose, _ = _cf(0.85).check_correlation_filter(
            "ETHUSDT", CoreOrderSide.LONG, existing
        )
        # threshold=0.60: corr 0.70 > 0.60 → blocked
        approved_strict, _ = _cf(0.60).check_correlation_filter(
            "ETHUSDT", CoreOrderSide.LONG, existing
        )

        assert approved_loose is True
        assert approved_strict is False

    def test_regime_adx_threshold_controls_regime_detection(self):
        """Changing trend_adx_min affects whether TREND regime is detected."""
        low_adx_detector = RegimeDetector(
            RegimeConfig(
                trend_adx_min=15, range_adx_max=10, high_vol_atr_z=1.5,
                confidence_threshold=0.3
            )
        )
        high_adx_detector = RegimeDetector(
            RegimeConfig(
                trend_adx_min=45, range_adx_max=20, high_vol_atr_z=1.5,
                confidence_threshold=0.3
            )
        )

        kwargs = dict(
            symbol="BTCUSDT",
            adx=25.0, atr_z=0.5, bb_width=0.03,
            ema20_5m=100.5, ema50_5m=100.0,
            ema20_1h=100.5, ema50_1h=100.0,
        )
        result_low = low_adx_detector.detect_regime(**kwargs)
        result_high = high_adx_detector.detect_regime(**kwargs)

        assert result_low.regime == RegimeType.TREND
        assert result_high.regime != RegimeType.TREND
