"""Correlation-based position filtering to prevent portfolio concentration"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from bot.config.models import RiskConfig
from bot.core.constants import OrderSide
from bot.core.types import Position

logger = logging.getLogger("trading_bot.risk.correlation_filter")


class CorrelationFilter:
    """
    Prevents portfolio concentration by blocking same-direction positions
    in highly correlated symbols.

    Correlation is calculated using 1h returns over a 72h lookback window.
    Symbols with correlation > threshold are considered same "bucket".
    """

    def __init__(self, config: RiskConfig):
        """
        Initialize correlation filter.

        Args:
            config: Risk configuration
        """
        self.config = config

        # Correlation matrix cache: {(symbol1, symbol2): correlation}
        self.correlation_cache: Dict[Tuple[str, str], float] = {}

        logger.info(
            f"CorrelationFilter initialized: "
            f"threshold={self.config.correlation_threshold:.2f}, "
            f"hedge_corr_max={self.config.hedge_corr_max:.2f}"
        )

    def update_correlation_matrix(
        self, price_data: Dict[str, np.ndarray]
    ) -> None:
        """
        Update correlation matrix from price data.

        Args:
            price_data: Dictionary mapping symbol -> 1h price array
                       Array should have at least 72 data points
        """
        symbols = list(price_data.keys())

        if len(symbols) < 2:
            logger.debug("Less than 2 symbols, skipping correlation calculation")
            return

        # Calculate correlations for all symbol pairs
        for i, symbol_a in enumerate(symbols):
            for symbol_b in symbols[i + 1 :]:
                prices_a = price_data[symbol_a]
                prices_b = price_data[symbol_b]

                # Calculate correlation
                corr = self._calculate_correlation(prices_a, prices_b)

                # Store in both directions
                self.correlation_cache[(symbol_a, symbol_b)] = corr
                self.correlation_cache[(symbol_b, symbol_a)] = corr

        logger.info(
            f"Updated correlation matrix: {len(self.correlation_cache)} pairs calculated"
        )

    def _calculate_correlation(
        self, prices_a: np.ndarray, prices_b: np.ndarray
    ) -> float:
        """
        Calculate rolling correlation of log returns.

        Args:
            prices_a: Price series A (1h candles, 72+ points)
            prices_b: Price series B (1h candles, 72+ points)

        Returns:
            Correlation coefficient (-1.0 to 1.0)
        """
        # Validate inputs
        if len(prices_a) < 2 or len(prices_b) < 2:
            logger.warning("Insufficient price data for correlation calculation")
            return 0.0

        if len(prices_a) != len(prices_b):
            logger.warning(
                f"Price array length mismatch: {len(prices_a)} vs {len(prices_b)}"
            )
            # Use minimum length
            min_len = min(len(prices_a), len(prices_b))
            prices_a = prices_a[-min_len:]
            prices_b = prices_b[-min_len:]

        # Calculate log returns
        returns_a = np.diff(np.log(prices_a))
        returns_b = np.diff(np.log(prices_b))

        # Handle edge cases
        if len(returns_a) == 0 or len(returns_b) == 0:
            return 0.0

        # Remove NaN/Inf values
        valid_mask = np.isfinite(returns_a) & np.isfinite(returns_b)
        returns_a = returns_a[valid_mask]
        returns_b = returns_b[valid_mask]

        if len(returns_a) < 2:
            logger.warning("Too few valid returns for correlation")
            return 0.0

        # Calculate correlation using numpy
        try:
            corr_matrix = np.corrcoef(returns_a, returns_b)
            corr = corr_matrix[0, 1]

            # Handle NaN result
            if np.isnan(corr):
                logger.debug("Correlation calculation returned NaN")
                return 0.0

            return float(corr)

        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.0

    def get_correlation(self, symbol_a: str, symbol_b: str) -> float:
        """
        Get cached correlation between two symbols.

        Args:
            symbol_a: First symbol
            symbol_b: Second symbol

        Returns:
            Correlation coefficient, or 0.0 if not cached
        """
        key = (symbol_a, symbol_b)
        return self.correlation_cache.get(key, 0.0)

    def check_correlation_filter(
        self,
        new_symbol: str,
        new_side: OrderSide,
        open_positions: List[Position],
    ) -> Tuple[bool, str]:
        """
        Check if new position passes correlation filter.

        Rules:
        1. If correlation > threshold AND same direction -> reject
        2. If correlation > threshold AND opposite direction (hedge) -> allow if corr < hedge_max
        3. If correlation <= threshold -> allow

        Args:
            new_symbol: Symbol for new position
            new_side: Side for new position (LONG/SHORT)
            open_positions: Currently open positions

        Returns:
            (approved, rejection_reason)
        """
        for position in open_positions:
            # Get correlation with existing position
            corr = abs(self.get_correlation(new_symbol, position.symbol))

            # Skip if correlation not calculated yet
            if corr == 0.0:
                logger.debug(
                    f"No correlation data for {new_symbol} vs {position.symbol}, allowing"
                )
                continue

            # Check if symbols are in same bucket (highly correlated)
            if corr > self.config.correlation_threshold:
                # Same direction - reject
                if new_side == position.side:
                    logger.warning(
                        f"Correlation filter blocked: {new_symbol} ({new_side.value}) "
                        f"corr={corr:.2f} with {position.symbol} ({position.side.value}), "
                        f"threshold={self.config.correlation_threshold:.2f}"
                    )
                    return (
                        False,
                        f"High correlation ({corr:.2f}) with existing {position.side.value} position in {position.symbol}",
                    )

                # Opposite direction (hedge) - check hedge correlation limit
                if corr > self.config.hedge_corr_max:
                    logger.warning(
                        f"Hedge correlation too high: {new_symbol} corr={corr:.2f} "
                        f"with {position.symbol}, max hedge corr={self.config.hedge_corr_max:.2f}"
                    )
                    return (
                        False,
                        f"Hedge correlation ({corr:.2f}) exceeds maximum ({self.config.hedge_corr_max:.2f}) for {position.symbol}",
                    )

                # Valid hedge
                logger.info(
                    f"Hedge position allowed: {new_symbol} ({new_side.value}) "
                    f"vs {position.symbol} ({position.side.value}), corr={corr:.2f}"
                )

        # All checks passed
        logger.debug(f"Correlation filter passed for {new_symbol} ({new_side.value})")
        return True, ""

    def get_correlated_positions(
        self, symbol: str, open_positions: List[Position], threshold: float = None
    ) -> List[Tuple[Position, float]]:
        """
        Get list of open positions correlated with given symbol.

        Args:
            symbol: Symbol to check correlations for
            open_positions: Currently open positions
            threshold: Correlation threshold (defaults to config threshold)

        Returns:
            List of (Position, correlation) tuples
        """
        if threshold is None:
            threshold = self.config.correlation_threshold

        correlated = []

        for position in open_positions:
            corr = abs(self.get_correlation(symbol, position.symbol))

            if corr > threshold:
                correlated.append((position, corr))

        # Sort by correlation (highest first)
        correlated.sort(key=lambda x: x[1], reverse=True)

        return correlated

    def get_correlation_summary(
        self, open_positions: List[Position]
    ) -> Dict[str, any]:
        """
        Get correlation summary for open positions.

        Args:
            open_positions: Currently open positions

        Returns:
            Dictionary with correlation metrics
        """
        if len(open_positions) < 2:
            return {
                "avg_correlation": 0.0,
                "max_correlation": 0.0,
                "min_correlation": 0.0,
                "correlated_pairs": 0,
                "total_pairs": 0,
            }

        correlations = []
        correlated_count = 0

        # Calculate correlations between all position pairs
        for i, pos_a in enumerate(open_positions):
            for pos_b in open_positions[i + 1 :]:
                corr = abs(self.get_correlation(pos_a.symbol, pos_b.symbol))
                correlations.append(corr)

                if corr > self.config.correlation_threshold:
                    correlated_count += 1

        if not correlations:
            return {
                "avg_correlation": 0.0,
                "max_correlation": 0.0,
                "min_correlation": 0.0,
                "correlated_pairs": 0,
                "total_pairs": 0,
            }

        return {
            "avg_correlation": float(np.mean(correlations)),
            "max_correlation": float(np.max(correlations)),
            "min_correlation": float(np.min(correlations)),
            "correlated_pairs": correlated_count,
            "total_pairs": len(correlations),
        }

    def clear_cache(self) -> None:
        """Clear correlation cache (e.g., for new trading day)"""
        logger.info("Clearing correlation cache")
        self.correlation_cache.clear()
