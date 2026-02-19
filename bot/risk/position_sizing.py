"""Position sizing calculator for risk management"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from bot.config.models import BotConfig
from bot.core.constants import RegimeType

logger = logging.getLogger("trading_bot.risk.position_sizing")


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    notional_usd: float
    leverage: float
    margin_required_usd: float
    risk_usd: float
    quantity: float
    approved: bool
    rejection_reason: str = ""


class PositionSizingCalculator:
    """Calculate position size based on risk parameters and regime"""

    def __init__(self, config: BotConfig):
        """
        Initialize position sizing calculator.

        Args:
            config: Bot configuration
        """
        self.config = config
        self.risk_config = config.risk
        self.leverage_config = config.leverage

        # Regime-based leverage mapping
        self.regime_leverage_map: Dict[RegimeType, float] = {
            RegimeType.TREND: self.leverage_config.trend,
            RegimeType.RANGE: self.leverage_config.range,
            RegimeType.HIGH_VOL: self.leverage_config.high_vol,
            RegimeType.CHOP_NO_TRADE: 1.0,  # Minimal leverage for no-trade regime
        }

        logger.info(
            f"PositionSizingCalculator initialized: "
            f"risk_per_trade={self.risk_config.risk_per_trade_pct:.2%}, "
            f"max_open_risk={self.risk_config.max_total_open_risk_pct:.2%}, "
            f"leverage_map={self.regime_leverage_map}"
        )

    def calculate(
        self,
        equity_usd: float,
        stop_pct: float,
        regime: RegimeType,
        current_price: float,
        free_margin_usd: float,
        min_notional_usd: float = 5.0,
        risk_per_trade_pct: Optional[float] = None,
    ) -> PositionSizeResult:
        """
        Calculate position size based on risk percentage and stop distance.

        Formula:
            risk_usd = equity_usd * risk_per_trade_pct
            notional_usd = risk_usd / stop_pct
            leverage = regime_leverage_map[regime]
            margin_required_usd = notional_usd / leverage
            quantity = notional_usd / current_price

        Args:
            equity_usd: Total account equity in USD
            stop_pct: Stop loss distance as percentage (e.g., 0.01 for 1%)
            regime: Current market regime
            current_price: Current market price
            free_margin_usd: Available margin in USD
            min_notional_usd: Minimum notional order size (Binance requirement)
            risk_per_trade_pct: Risk per trade percentage (overrides config if provided)

        Returns:
            PositionSizeResult with calculated values and approval status
        """
        # Use config risk if not provided
        if risk_per_trade_pct is None:
            risk_per_trade_pct = self.risk_config.risk_per_trade_pct

        # Step 1: Calculate risk in USD
        risk_usd = equity_usd * risk_per_trade_pct

        # Step 2: Calculate notional based on stop distance
        if stop_pct <= 0:
            logger.warning(f"Invalid stop_pct: {stop_pct} (must be > 0)")
            return PositionSizeResult(
                notional_usd=0.0,
                leverage=1.0,
                margin_required_usd=0.0,
                risk_usd=0.0,
                quantity=0.0,
                approved=False,
                rejection_reason="Invalid stop_pct: must be > 0",
            )

        notional_usd = risk_usd / stop_pct

        # Step 3: Get leverage for regime
        leverage = self.regime_leverage_map.get(regime, 1.0)

        # Step 4: Calculate margin required
        margin_required_usd = notional_usd / leverage

        # Step 5: Calculate quantity
        if current_price <= 0:
            logger.warning(f"Invalid current_price: {current_price} (must be > 0)")
            return PositionSizeResult(
                notional_usd=0.0,
                leverage=1.0,
                margin_required_usd=0.0,
                risk_usd=0.0,
                quantity=0.0,
                approved=False,
                rejection_reason="Invalid current_price: must be > 0",
            )

        quantity = notional_usd / current_price

        # Validation 1: Check minimum notional
        if notional_usd < min_notional_usd:
            logger.debug(
                f"Position rejected: notional ${notional_usd:.2f} < minimum ${min_notional_usd:.2f}"
            )
            return PositionSizeResult(
                notional_usd=notional_usd,
                leverage=leverage,
                margin_required_usd=margin_required_usd,
                risk_usd=risk_usd,
                quantity=quantity,
                approved=False,
                rejection_reason=f"Notional ${notional_usd:.2f} below minimum ${min_notional_usd:.2f}",
            )

        # Validation 2: Check sufficient margin
        if margin_required_usd > free_margin_usd:
            logger.warning(
                f"Insufficient margin: required ${margin_required_usd:.2f}, "
                f"available ${free_margin_usd:.2f}"
            )
            return PositionSizeResult(
                notional_usd=notional_usd,
                leverage=leverage,
                margin_required_usd=margin_required_usd,
                risk_usd=risk_usd,
                quantity=quantity,
                approved=False,
                rejection_reason=f"Insufficient margin: required ${margin_required_usd:.2f}, available ${free_margin_usd:.2f}",
            )

        # Validation 3: Check leverage limits (Binance allows 1-125x, but we restrict to 1-2x)
        if leverage < 1.0 or leverage > 2.0:
            logger.error(
                f"Leverage {leverage:.1f}x outside allowed range [1.0, 2.0] - config error!"
            )
            return PositionSizeResult(
                notional_usd=notional_usd,
                leverage=leverage,
                margin_required_usd=margin_required_usd,
                risk_usd=risk_usd,
                quantity=quantity,
                approved=False,
                rejection_reason=f"Leverage {leverage:.1f}x outside allowed range [1.0, 2.0]",
            )

        # All validations passed
        logger.debug(
            f"Position approved: notional=${notional_usd:.2f}, "
            f"qty={quantity:.6f}, leverage={leverage:.1f}x, "
            f"margin=${margin_required_usd:.2f}, regime={regime.value}"
        )
        return PositionSizeResult(
            notional_usd=notional_usd,
            leverage=leverage,
            margin_required_usd=margin_required_usd,
            risk_usd=risk_usd,
            quantity=quantity,
            approved=True,
            rejection_reason="",
        )

    def get_leverage_for_regime(self, regime: RegimeType) -> float:
        """
        Get leverage multiplier for a specific regime.

        Args:
            regime: Market regime

        Returns:
            Leverage multiplier
        """
        return self.regime_leverage_map.get(regime, 1.0)

    def calculate_max_position_count(
        self,
        equity_usd: float,
        avg_stop_pct: float = 0.01,
        avg_regime_leverage: float = 1.5,
    ) -> int:
        """
        Estimate maximum number of positions given risk constraints.

        Args:
            equity_usd: Total account equity
            avg_stop_pct: Average stop distance (default 1%)
            avg_regime_leverage: Average leverage used (default 1.5x)

        Returns:
            Estimated max positions
        """
        # Calculate single position size
        risk_per_trade = equity_usd * self.risk_config.risk_per_trade_pct
        single_position_notional = risk_per_trade / avg_stop_pct
        single_position_margin = single_position_notional / avg_regime_leverage

        # Total margin available
        total_margin_available = equity_usd

        # Max positions by margin
        max_by_margin = int(total_margin_available / single_position_margin)

        # Max by total open risk
        max_total_risk = equity_usd * self.risk_config.max_total_open_risk_pct
        max_by_risk = int(max_total_risk / risk_per_trade)

        # Return minimum of constraints
        return min(
            max_by_margin,
            max_by_risk,
            self.risk_config.max_open_positions,
        )
