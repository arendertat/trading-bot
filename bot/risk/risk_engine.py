"""Central risk validation engine integrating all risk components"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from bot.config.models import BotConfig
from bot.core.constants import OrderSide, RegimeType
from bot.core.types import Position
from bot.risk.correlation_filter import CorrelationFilter
from bot.risk.kill_switch import KillSwitch
from bot.risk.position_sizing import PositionSizeResult, PositionSizingCalculator
from bot.risk.risk_limits import RiskLimits

logger = logging.getLogger("trading_bot.risk.risk_engine")


@dataclass
class RiskValidationResult:
    """Result of comprehensive risk validation"""
    approved: bool
    rejection_reason: str
    position_size: Optional[PositionSizeResult] = None


class RiskEngine:
    """
    Central risk validation gate called before every trade.

    Integrates:
    - Position sizing
    - Kill switches (daily/weekly stops)
    - Open risk limits
    - Max positions
    - Correlation filter
    - Direction limits
    """

    def __init__(
        self,
        config: BotConfig,
        kill_switch: KillSwitch,
        position_sizing: PositionSizingCalculator,
        risk_limits: RiskLimits,
        correlation_filter: CorrelationFilter,
    ):
        """
        Initialize risk engine with all components.

        Args:
            config: Bot configuration
            kill_switch: Kill switch manager
            position_sizing: Position sizing calculator
            risk_limits: Risk limits validator
            correlation_filter: Correlation filter
        """
        self.config = config
        self.kill_switch = kill_switch
        self.position_sizing = position_sizing
        self.risk_limits = risk_limits
        self.correlation_filter = correlation_filter

        # Özellik 10: Cooldown tracking — symbol → bars remaining
        self._cooldown_bars: dict = {}
        # Throttle insufficient margin logs — symbol -> count
        self._insufficient_margin_counts: dict = {}

        logger.info("RiskEngine initialized with all risk components")

    def record_sl_exit(self, symbol: str) -> None:
        """
        Record a stop-loss exit for cooldown tracking (Özellik 10).

        Called by BotRunner after a SL/TRAIL fill.
        Sets the cooldown counter for the symbol.
        """
        bars = self.config.risk.cooldown_after_sl_bars
        if bars > 0:
            self._cooldown_bars[symbol] = bars
            logger.info(
                f"[COOLDOWN] {symbol}: {bars}-bar cooldown started after SL"
            )

    def tick_cooldowns(self) -> None:
        """
        Decrement all active cooldown counters by 1 bar (Özellik 10).

        Called by BotRunner on every candle close.
        """
        expired = []
        for sym, bars in self._cooldown_bars.items():
            remaining = bars - 1
            if remaining <= 0:
                expired.append(sym)
                logger.info(f"[COOLDOWN] {sym}: cooldown expired")
            else:
                self._cooldown_bars[sym] = remaining
        for sym in expired:
            del self._cooldown_bars[sym]

    def validate_entry(
        self,
        symbol: str,
        side: OrderSide,
        regime: RegimeType,
        stop_pct: float,
        current_price: float,
        equity_usd: float,
        free_margin_usd: float,
        open_positions: List[Position],
        risk_per_trade_pct: Optional[float] = None,
    ) -> RiskValidationResult:
        """
        Validate all risk constraints for a new position.

        Validation order:
        1. Kill switch check (daily/weekly stops)
        2. Cooldown check (after SL, Özellik 10)
        3. Max positions check
        4. Direction limit check
        5. Position sizing calculation (HIGH_VOL risk reduction, Özellik 10)
        6. Open risk limit check
        7. Correlation filter check
        8. Net exposure check

        Args:
            symbol: Symbol to trade
            side: Order side (LONG/SHORT)
            regime: Current market regime
            stop_pct: Stop loss distance as percentage
            current_price: Current market price
            equity_usd: Total account equity
            free_margin_usd: Available margin
            open_positions: Currently open positions
            risk_per_trade_pct: Override risk percentage (for reduced risk periods)

        Returns:
            RiskValidationResult with approval status and details
        """
        logger.info(
            f"Validating entry: {symbol} {side.value}, regime={regime.value}, "
            f"stop={stop_pct:.2%}, positions={len(open_positions)}"
        )

        # 1. Kill switch check
        if self.kill_switch.is_active():
            reason = self.kill_switch.get_active_reason()
            logger.warning(f"Kill switch active: {reason}")
            return RiskValidationResult(
                approved=False,
                rejection_reason=f"Kill switch active: {reason}",
            )

        # 2. Cooldown check — symbol blocked after SL (Özellik 10)
        cooldown_remaining = self._cooldown_bars.get(symbol, 0)
        if cooldown_remaining > 0:
            logger.info(
                f"[COOLDOWN] {symbol}: {cooldown_remaining} bars remaining — entry blocked"
            )
            return RiskValidationResult(
                approved=False,
                rejection_reason=f"Cooldown active: {cooldown_remaining} bars remaining after SL",
            )

        # 3. Max positions check
        approved, reason = self.risk_limits.check_max_positions(open_positions)
        if not approved:
            logger.warning(f"Max positions check failed: {reason}")
            return RiskValidationResult(approved=False, rejection_reason=reason)

        # 3. Direction limit check
        approved, reason = self.risk_limits.check_same_direction_limit(
            new_side=side.value, open_positions=open_positions
        )
        if not approved:
            logger.warning(f"Direction limit check failed: {reason}")
            return RiskValidationResult(approved=False, rejection_reason=reason)

        # 5. Position sizing calculation
        # Özellik 10: HIGH_VOL → apply reduced risk if no explicit override
        effective_risk_pct = risk_per_trade_pct
        if effective_risk_pct is None and regime == RegimeType.HIGH_VOL:
            effective_risk_pct = self.config.risk.high_vol_risk_reduction_pct
            logger.info(
                f"[HIGH_VOL] {symbol}: reduced risk "
                f"{effective_risk_pct:.2%} (normal: {self.config.risk.risk_per_trade_pct:.2%})"
            )

        position_size = self.position_sizing.calculate(
            equity_usd=equity_usd,
            stop_pct=stop_pct,
            regime=regime,
            current_price=current_price,
            free_margin_usd=free_margin_usd,
            symbol=symbol,
            risk_per_trade_pct=effective_risk_pct,
        )

        if not position_size.approved:
            if "Insufficient margin" in position_size.rejection_reason or "INSUFFICIENT_MARGIN" in position_size.rejection_reason:
                count = self._insufficient_margin_counts.get(symbol, 0) + 1
                self._insufficient_margin_counts[symbol] = count
                every_n = self.config.risk.insufficient_margin_log_every_n
                if every_n > 0 and (count == 1 or count % every_n == 0):
                    logger.warning(
                        f"Position sizing failed: {position_size.rejection_reason} "
                        f"(count={count})"
                    )
            else:
                logger.warning(f"Position sizing failed: {position_size.rejection_reason}")
            return RiskValidationResult(
                approved=False,
                rejection_reason=position_size.rejection_reason,
                position_size=position_size,
            )

        # 6. Open risk limit check
        new_position_risk = position_size.risk_usd
        approved, reason = self.risk_limits.check_open_risk_limit(
            new_position_risk_usd=new_position_risk,
            open_positions=open_positions,
            equity_usd=equity_usd,
        )
        if not approved:
            logger.warning(f"Open risk limit check failed: {reason}")
            return RiskValidationResult(
                approved=False,
                rejection_reason=reason,
                position_size=position_size,
            )

        # 7. Correlation filter check
        approved, reason = self.correlation_filter.check_correlation_filter(
            new_symbol=symbol,
            new_side=side,
            open_positions=open_positions,
        )
        if not approved:
            logger.warning(f"Correlation filter check failed: {reason}")
            return RiskValidationResult(
                approved=False,
                rejection_reason=reason,
                position_size=position_size,
            )

        # 8. Net exposure + single-symbol concentration check (Özellik 9)
        approved, reason = self.risk_limits.check_net_exposure(
            new_side=side.value,
            new_notional_usd=position_size.notional_usd,
            new_symbol=symbol,
            open_positions=open_positions,
            equity_usd=equity_usd,
        )
        if not approved:
            logger.warning(f"Exposure limit check failed: {reason}")
            return RiskValidationResult(
                approved=False,
                rejection_reason=reason,
                position_size=position_size,
            )

        # All checks passed
        logger.info(
            f"✅ Risk validation passed: {symbol} {side.value}, "
            f"size=${position_size.notional_usd:.2f}, qty={position_size.quantity:.6f}"
        )
        return RiskValidationResult(
            approved=True,
            rejection_reason="",
            position_size=position_size,
        )

    def get_portfolio_status(
        self, open_positions: List[Position], equity_usd: float
    ) -> dict:
        """
        Get comprehensive portfolio risk status.

        Args:
            open_positions: Currently open positions
            equity_usd: Total account equity

        Returns:
            Dictionary with portfolio metrics
        """
        # Risk limits summary
        risk_summary = self.risk_limits.get_portfolio_risk_summary(
            open_positions, equity_usd
        )

        # Correlation summary
        corr_summary = self.correlation_filter.get_correlation_summary(open_positions)

        # Kill switch status
        kill_switch_state = self.kill_switch.get_state()

        return {
            # Position counts
            "total_positions": risk_summary["total_positions"],
            "long_positions": risk_summary["long_positions"],
            "short_positions": risk_summary["short_positions"],
            # Risk metrics
            "current_open_risk_usd": risk_summary["current_open_risk_usd"],
            "current_open_risk_pct": risk_summary["current_open_risk_pct"],
            "available_risk_usd": risk_summary["available_risk_usd"],
            "available_position_slots": risk_summary["available_position_slots"],
            # Correlation metrics
            "avg_correlation": corr_summary["avg_correlation"],
            "max_correlation": corr_summary["max_correlation"],
            "correlated_pairs": corr_summary["correlated_pairs"],
            # Kill switch status
            "daily_stop_active": kill_switch_state.daily_stop_active,
            "weekly_pause_active": kill_switch_state.weekly_pause_active,
            "reduced_risk_active": self.kill_switch.is_reduced_risk_active(),
            "current_daily_pnl": kill_switch_state.current_daily_pnl,
            "current_weekly_pnl": kill_switch_state.current_weekly_pnl,
        }
