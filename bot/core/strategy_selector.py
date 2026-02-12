"""Strategy selection based on performance metrics"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from bot.core.performance_tracker import PerformanceTracker, StrategyMetrics
from bot.core.constants import RegimeType
from bot.strategies.base import Strategy

logger = logging.getLogger("trading_bot.strategy_selector")


class StrategySelector:
    """
    Performance-based strategy selection engine.

    Selects best-performing strategy for each regime based on rolling metrics:
    - Expectancy R (expected return per trade)
    - Maximum drawdown (risk penalty)
    - Confidence scoring (metrics quality)

    Features:
    - Stability constraint: Max 1 switch per day unless expectancy < 0
    - Confidence threshold: Requires min 0.55 confidence to select
    - Fallback: Returns None if no strategy meets criteria
    """

    # Selection parameters
    DD_PENALTY_WEIGHT = 0.5  # Drawdown penalty multiplier
    CONFIDENCE_THRESHOLD = 0.55  # Min confidence to select strategy
    MIN_EXPECTANCY_EMERGENCY_SWITCH = 0.0  # Emergency switch threshold

    def __init__(
        self,
        performance_tracker: PerformanceTracker,
        strategies: Dict[str, Strategy],
        stability_hours: int = 24
    ):
        """
        Initialize strategy selector.

        Args:
            performance_tracker: Performance tracker instance
            strategies: Dict of strategy_name -> Strategy instance
            stability_hours: Hours between strategy switches (default 24)
        """
        self.performance_tracker = performance_tracker
        self.strategies = strategies
        self.stability_hours = stability_hours

        # Track last selection per regime for stability constraint
        self.last_selection_time: Dict[RegimeType, datetime] = {}
        self.current_selection: Dict[RegimeType, Optional[str]] = {}

        logger.info(
            f"StrategySelector initialized with {len(strategies)} strategies, "
            f"stability={stability_hours}h"
        )

    def select_strategy(
        self,
        regime: RegimeType,
        symbol: str
    ) -> Optional[Strategy]:
        """
        Select best strategy for current regime based on performance.

        Selection logic:
        1. Get metrics for all strategies
        2. Calculate confidence score for each
        3. Apply stability constraint (max 1 switch per stability_hours)
        4. Select highest scoring strategy above confidence threshold
        5. Return None if no strategy qualifies

        Args:
            regime: Current market regime
            symbol: Trading symbol (for logging)

        Returns:
            Best Strategy instance, or None if no strategy qualifies
        """
        # Get all strategies that can trade in this regime
        candidate_strategies = self._get_regime_compatible_strategies(regime)

        if not candidate_strategies:
            logger.warning(f"{symbol}: No strategies compatible with {regime.value} regime")
            return None

        # Calculate scores for all candidates
        scored_strategies: List[tuple[str, float, StrategyMetrics]] = []

        for strategy_name in candidate_strategies:
            metrics = self.performance_tracker.get_metrics(strategy_name)

            if metrics is None:
                # Insufficient trade history
                logger.debug(f"{strategy_name}: Insufficient trade history for scoring")
                continue

            # Calculate confidence score
            confidence = self._calculate_confidence(metrics)
            metrics.confidence = confidence

            if confidence < self.CONFIDENCE_THRESHOLD:
                logger.debug(
                    f"{strategy_name}: Confidence {confidence:.3f} < threshold {self.CONFIDENCE_THRESHOLD}"
                )
                continue

            scored_strategies.append((strategy_name, confidence, metrics))

        if not scored_strategies:
            logger.info(f"{symbol}: No strategies meet confidence threshold for {regime.value}")
            return None

        # Sort by confidence (descending)
        scored_strategies.sort(key=lambda x: x[1], reverse=True)
        best_strategy_name, best_confidence, best_metrics = scored_strategies[0]

        # Check stability constraint
        if not self._can_switch_strategy(regime, best_strategy_name, best_metrics):
            # Return current strategy if still valid
            current = self.current_selection.get(regime)
            if current and current in self.strategies:
                logger.debug(
                    f"{symbol}: Stability constraint active, keeping {current} for {regime.value}"
                )
                return self.strategies[current]
            else:
                # No current strategy, allow switch
                pass

        # Update selection tracking
        self.current_selection[regime] = best_strategy_name
        self.last_selection_time[regime] = datetime.utcnow()

        logger.info(
            f"{symbol}: Selected {best_strategy_name} for {regime.value} "
            f"(confidence={best_confidence:.3f}, expectancy_r={best_metrics.expectancy_r:.3f})"
        )

        return self.strategies[best_strategy_name]

    def _get_regime_compatible_strategies(self, regime: RegimeType) -> List[str]:
        """
        Get list of strategy names compatible with regime.

        Filters strategies based on their designed regime (from strategy class).

        Args:
            regime: Current market regime

        Returns:
            List of compatible strategy names
        """
        compatible = []

        for name, strategy in self.strategies.items():
            if not strategy.enabled:
                continue

            # Strategy compatibility check based on naming convention
            # TrendPullback/TrendBreakout -> TREND regime
            # RangeMeanReversion -> RANGE regime
            # (In production, use explicit strategy.compatible_regimes property)

            if regime == RegimeType.TREND:
                if "Trend" in name or "trend" in name.lower():
                    compatible.append(name)
            elif regime == RegimeType.RANGE:
                if "Range" in name or "range" in name.lower():
                    compatible.append(name)
            elif regime == RegimeType.HIGH_VOL:
                # HIGH_VOL can use trend strategies (typically breakout)
                if "Breakout" in name or "breakout" in name.lower():
                    compatible.append(name)

        return compatible

    def _calculate_confidence(self, metrics: StrategyMetrics) -> float:
        """
        Calculate confidence score for strategy selection.

        Score formula:
            base_score = expectancy_r - (DD_PENALTY * max_drawdown_pct)
            confidence = sigmoid(base_score) mapped to [0, 1]

        Higher expectancy_r and lower drawdown = higher confidence.

        Args:
            metrics: Strategy performance metrics

        Returns:
            Confidence score in [0, 1]
        """
        # Base score: expectancy minus drawdown penalty
        base_score = metrics.expectancy_r - (self.DD_PENALTY_WEIGHT * metrics.max_drawdown_pct)

        # Adjust based on sample size (more trades = higher confidence)
        sample_size_factor = min(1.0, metrics.total_trades / 50.0)

        # Sigmoid-like mapping to [0, 1]
        # base_score > 0.5 -> confidence > 0.6
        # base_score > 1.0 -> confidence > 0.75
        if base_score > 0:
            confidence = 0.5 + (0.5 * min(1.0, base_score / 2.0))
        else:
            confidence = 0.5 * max(0.0, 1.0 + base_score)

        # Apply sample size adjustment
        confidence = confidence * sample_size_factor

        return max(0.0, min(1.0, confidence))

    def _can_switch_strategy(
        self,
        regime: RegimeType,
        new_strategy: str,
        new_metrics: StrategyMetrics
    ) -> bool:
        """
        Check if strategy switch is allowed (stability constraint).

        Rules:
        1. If no previous selection, allow switch
        2. If expectancy < 0 (emergency), allow immediate switch
        3. If within stability window (24h), block switch
        4. If outside stability window, allow switch

        Args:
            regime: Current regime
            new_strategy: Proposed new strategy name
            new_metrics: Metrics for new strategy

        Returns:
            True if switch allowed, False otherwise
        """
        current_strategy = self.current_selection.get(regime)

        # No previous selection - allow
        if current_strategy is None:
            return True

        # Same strategy - allow (no-op)
        if current_strategy == new_strategy:
            return True

        last_switch_time = self.last_selection_time.get(regime)

        # No previous switch time recorded - allow
        if last_switch_time is None:
            return True

        # Emergency switch if current strategy has negative expectancy
        current_metrics = self.performance_tracker.get_metrics(current_strategy)
        if current_metrics and current_metrics.expectancy_r < self.MIN_EXPECTANCY_EMERGENCY_SWITCH:
            logger.warning(
                f"Emergency switch: {current_strategy} expectancy "
                f"({current_metrics.expectancy_r:.3f}) < 0"
            )
            return True

        # Check stability window
        time_since_switch = datetime.utcnow() - last_switch_time
        if time_since_switch.total_seconds() < (self.stability_hours * 3600):
            logger.debug(
                f"Stability constraint: {time_since_switch.total_seconds()/3600:.1f}h "
                f"< {self.stability_hours}h since last switch"
            )
            return False

        # Outside stability window - allow switch
        return True

    def get_current_selection(self, regime: RegimeType) -> Optional[str]:
        """
        Get currently selected strategy for regime.

        Args:
            regime: Market regime

        Returns:
            Strategy name if selected, else None
        """
        return self.current_selection.get(regime)

    def force_strategy(self, regime: RegimeType, strategy_name: str) -> bool:
        """
        Force selection of specific strategy (admin override).

        Args:
            regime: Market regime
            strategy_name: Strategy to select

        Returns:
            True if successful, False if strategy doesn't exist
        """
        if strategy_name not in self.strategies:
            logger.error(f"Cannot force strategy '{strategy_name}': not found")
            return False

        self.current_selection[regime] = strategy_name
        self.last_selection_time[regime] = datetime.utcnow()

        logger.warning(f"Forced selection: {strategy_name} for {regime.value}")
        return True

    def reset_stability_timer(self, regime: RegimeType) -> None:
        """
        Reset stability timer for regime (allow immediate switch).

        Args:
            regime: Market regime to reset
        """
        if regime in self.last_selection_time:
            del self.last_selection_time[regime]
            logger.info(f"Reset stability timer for {regime.value}")
