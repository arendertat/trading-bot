"""Strategy selection based on performance metrics"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone

from bot.core.performance_tracker import PerformanceTracker, StrategyMetrics
from bot.core.constants import RegimeType
from bot.strategies.base import Strategy
from bot.utils.jsonl_logger import JsonlLogger

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
        stability_hours: int = 24,
        log_dir: Optional[str] = None,
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
        self._selection_logger = (
            JsonlLogger(f"{log_dir.rstrip('/')}/strategy_selection.jsonl")
            if log_dir
            else None
        )

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
        symbol: str,
        timestamp: Optional[int] = None,
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
            self._log_selection(
                symbol=symbol,
                regime=regime,
                timestamp=timestamp,
                candidates=[],
                rejected=[{"name": name, "reason": "incompatible"} for name in self.strategies],
                selected=None,
                reason="no_compatible_strategies",
            )
            return None

        # Calculate scores for all candidates
        scored_strategies: List[tuple[str, float, StrategyMetrics]] = []
        strategies_without_history: List[str] = []

        rejected: List[dict] = []
        for strategy_name in candidate_strategies:
            metrics = self.performance_tracker.get_metrics(strategy_name)

            if metrics is None:
                # Insufficient trade history - track for cold start fallback
                logger.debug(f"{strategy_name}: Insufficient trade history for scoring")
                strategies_without_history.append(strategy_name)
                rejected.append({"name": strategy_name, "reason": "insufficient_history"})
                continue

            # Calculate confidence score
            confidence = self._calculate_confidence(metrics)
            metrics.confidence = confidence

            if confidence < self.CONFIDENCE_THRESHOLD:
                logger.debug(
                    f"{strategy_name}: Confidence {confidence:.3f} < threshold {self.CONFIDENCE_THRESHOLD}"
                )
                rejected.append({"name": strategy_name, "reason": "low_confidence", "confidence": confidence})
                continue

            scored_strategies.append((strategy_name, confidence, metrics))

        # Cold start fallback: If no strategies have sufficient history, select first compatible
        if not scored_strategies:
            if strategies_without_history:
                # Sort alphabetically for deterministic selection (tests expect TrendPullback before TrendBreakout)
                strategies_without_history.sort()
                fallback_strategy = strategies_without_history[0]
                logger.info(
                    f"{symbol}: COLD START - No performance history, selecting {fallback_strategy} "
                    f"for {regime.value} (will re-evaluate after 10+ trades)"
                )
                self.current_selection[regime] = fallback_strategy
                self.last_selection_time[regime] = datetime.utcnow()
                self._log_selection(
                    symbol=symbol,
                    regime=regime,
                    timestamp=timestamp,
                    candidates=strategies_without_history,
                    rejected=rejected,
                    selected=fallback_strategy,
                    reason="cold_start",
                )
                return self.strategies[fallback_strategy]
            else:
                logger.info(f"{symbol}: No strategies meet confidence threshold for {regime.value}")
                self._log_selection(
                    symbol=symbol,
                    regime=regime,
                    timestamp=timestamp,
                    candidates=candidate_strategies,
                    rejected=rejected,
                    selected=None,
                    reason="no_confident_strategies",
                )
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
                self._log_selection(
                    symbol=symbol,
                    regime=regime,
                    timestamp=timestamp,
                    candidates=candidate_strategies,
                    rejected=rejected,
                    selected=current,
                    reason="stability_constraint",
                    scored=scored_strategies,
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
        self._log_selection(
            symbol=symbol,
            regime=regime,
            timestamp=timestamp,
            candidates=candidate_strategies,
            rejected=rejected,
            selected=best_strategy_name,
            reason="selected",
            scored=scored_strategies,
        )

        return self.strategies[best_strategy_name]

    def _log_selection(
        self,
        symbol: str,
        regime: RegimeType,
        timestamp: Optional[int],
        candidates: List[str],
        rejected: List[dict],
        selected: Optional[str],
        reason: str,
        scored: Optional[List[tuple[str, float, StrategyMetrics]]] = None,
    ) -> None:
        if self._selection_logger is None:
            return
        scored_payload = []
        if scored:
            for name, confidence, metrics in scored:
                scored_payload.append({
                    "name": name,
                    "confidence": confidence,
                    "expectancy_r": metrics.expectancy_r,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "total_trades": metrics.total_trades,
                })
        self._selection_logger.log(
            {
                "ts": timestamp,
                "symbol": symbol,
                "regime": regime.value,
                "candidates": candidates,
                "rejected": rejected,
                "scored": scored_payload,
                "selected": selected,
                "reason": reason,
            }
        )

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

            # Bulgu 8: Use explicit compatible_regimes property on each strategy class.
            if regime in strategy.compatible_regimes:
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
        # 20 trades = 0.8x multiplier, 30 trades = 1.0x (full confidence)
        # Less aggressive than /50, allows good strategies to be selected sooner
        sample_size_factor = min(1.0, metrics.total_trades / 30.0)

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
