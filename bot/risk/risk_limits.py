"""Risk limit validators for portfolio-level constraints"""

import logging
from typing import List

from bot.config.models import RiskConfig
from bot.core.types import Position

logger = logging.getLogger("trading_bot.risk.risk_limits")


class RiskLimits:
    """
    Portfolio-level risk limit validators.

    Validates:
    - Total open risk limit (sum of all position risks)
    - Maximum number of open positions
    - Maximum same-direction positions
    """

    def __init__(self, config: RiskConfig):
        """
        Initialize risk limits validator.

        Args:
            config: Risk configuration
        """
        self.config = config

        logger.info(
            f"RiskLimits initialized: "
            f"max_open_risk={self.config.max_total_open_risk_pct:.2%}, "
            f"max_positions={self.config.max_open_positions}, "
            f"max_same_direction={self.config.max_same_direction_positions}"
        )

    def check_open_risk_limit(
        self,
        new_position_risk_usd: float,
        open_positions: List[Position],
        equity_usd: float,
    ) -> tuple[bool, str]:
        """
        Check if adding a new position would exceed total open risk limit.

        Open risk is defined as the sum of all position stop distances in USD:
            risk_per_position = abs(entry_price - stop_price) * quantity

        Args:
            new_position_risk_usd: Risk in USD for the new position
            open_positions: List of currently open positions
            equity_usd: Current account equity

        Returns:
            (approved, rejection_reason)
        """
        # Calculate current open risk
        current_open_risk_usd = self._calculate_total_open_risk(open_positions)

        # Calculate total open risk after adding new position
        total_open_risk_usd = current_open_risk_usd + new_position_risk_usd

        # Handle zero equity edge case
        if equity_usd <= 0:
            logger.error("Cannot check risk limit with zero or negative equity")
            return False, "Zero or negative equity"

        # Calculate percentage of equity
        total_open_risk_pct = total_open_risk_usd / equity_usd

        # Check against limit (use >= to reject at exact limit)
        if total_open_risk_pct >= self.config.max_total_open_risk_pct:
            logger.warning(
                f"Total open risk limit exceeded: "
                f"{total_open_risk_pct:.2%} > {self.config.max_total_open_risk_pct:.2%} "
                f"(${total_open_risk_usd:.2f} / ${equity_usd:.2f})"
            )
            return (
                False,
                f"Total open risk {total_open_risk_pct:.2%} exceeds limit {self.config.max_total_open_risk_pct:.2%}",
            )

        logger.debug(
            f"Open risk check passed: "
            f"{total_open_risk_pct:.2%} <= {self.config.max_total_open_risk_pct:.2%} "
            f"(current=${current_open_risk_usd:.2f}, new=${new_position_risk_usd:.2f})"
        )
        return True, ""

    def check_max_positions(self, open_positions: List[Position]) -> tuple[bool, str]:
        """
        Check if maximum number of open positions has been reached.

        Args:
            open_positions: List of currently open positions

        Returns:
            (approved, rejection_reason)
        """
        current_count = len(open_positions)

        if current_count >= self.config.max_open_positions:
            logger.warning(
                f"Max positions limit reached: {current_count} >= {self.config.max_open_positions}"
            )
            return (
                False,
                f"Max positions ({self.config.max_open_positions}) reached",
            )

        logger.debug(
            f"Max positions check passed: {current_count} < {self.config.max_open_positions}"
        )
        return True, ""

    def check_same_direction_limit(
        self, new_side: str, open_positions: List[Position]
    ) -> tuple[bool, str]:
        """
        Check if maximum same-direction positions has been reached.

        Args:
            new_side: Side of new position ('LONG' or 'SHORT')
            open_positions: List of currently open positions

        Returns:
            (approved, rejection_reason)
        """
        same_direction_count = sum(
            1 for pos in open_positions if pos.side.value == new_side
        )

        if same_direction_count >= self.config.max_same_direction_positions:
            logger.warning(
                f"Max same-direction positions limit reached: "
                f"{same_direction_count} {new_side} positions >= {self.config.max_same_direction_positions}"
            )
            return (
                False,
                f"Max {new_side} positions ({self.config.max_same_direction_positions}) reached",
            )

        logger.debug(
            f"Same-direction check passed: "
            f"{same_direction_count} {new_side} positions < {self.config.max_same_direction_positions}"
        )
        return True, ""

    def _calculate_total_open_risk(self, open_positions: List[Position]) -> float:
        """
        Calculate total open risk across all positions.

        Risk per position = abs(entry_price - stop_price) * quantity

        Args:
            open_positions: List of open positions

        Returns:
            Total open risk in USD
        """
        total_risk = 0.0

        for position in open_positions:
            # Calculate stop distance
            stop_distance = abs(position.entry_price - position.stop_price)

            # Calculate risk in USD
            position_risk = stop_distance * position.quantity

            total_risk += position_risk

        return total_risk

    def get_available_risk_budget(
        self, open_positions: List[Position], equity_usd: float
    ) -> float:
        """
        Calculate available risk budget for new positions.

        Args:
            open_positions: List of currently open positions
            equity_usd: Current account equity

        Returns:
            Available risk budget in USD
        """
        current_open_risk = self._calculate_total_open_risk(open_positions)
        max_total_risk = equity_usd * self.config.max_total_open_risk_pct
        available = max_total_risk - current_open_risk

        return max(0.0, available)

    def get_portfolio_risk_summary(
        self, open_positions: List[Position], equity_usd: float
    ) -> dict:
        """
        Get comprehensive portfolio risk summary.

        Args:
            open_positions: List of currently open positions
            equity_usd: Current account equity

        Returns:
            Dictionary with risk metrics
        """
        current_open_risk = self._calculate_total_open_risk(open_positions)
        max_total_risk = equity_usd * self.config.max_total_open_risk_pct
        available_risk = max(0.0, max_total_risk - current_open_risk)

        # Count positions by direction
        long_count = sum(1 for pos in open_positions if pos.side.value == "LONG")
        short_count = sum(1 for pos in open_positions if pos.side.value == "SHORT")

        summary = {
            "total_positions": len(open_positions),
            "long_positions": long_count,
            "short_positions": short_count,
            "current_open_risk_usd": current_open_risk,
            "current_open_risk_pct": current_open_risk / equity_usd if equity_usd > 0 else 0.0,
            "max_open_risk_usd": max_total_risk,
            "max_open_risk_pct": self.config.max_total_open_risk_pct,
            "available_risk_usd": available_risk,
            "available_risk_pct": available_risk / equity_usd if equity_usd > 0 else 0.0,
            "max_positions": self.config.max_open_positions,
            "available_position_slots": max(0, self.config.max_open_positions - len(open_positions)),
            "max_same_direction": self.config.max_same_direction_positions,
            "available_long_slots": max(0, self.config.max_same_direction_positions - long_count),
            "available_short_slots": max(0, self.config.max_same_direction_positions - short_count),
        }

        return summary
