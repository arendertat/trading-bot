"""Range Mean Reversion Strategy implementation"""

import logging
from typing import Optional

from bot.strategies.base import Strategy, FeatureSet
from bot.core.constants import OrderSide, RegimeType

logger = logging.getLogger("trading_bot.strategies.range_mean_reversion")


class RangeMeanReversionStrategy(Strategy):
    """
    Range Mean Reversion Strategy.

    Trade extreme moves in ranging markets using RSI and Bollinger Bands.

    Entry Logic:
    - LONG: RSI < 25 (oversold) + price touches BB lower band
    - SHORT: RSI > 75 (overbought) + price touches BB upper band

    Stop Loss: Fixed 0.8% from entry (tighter than trend strategies)
    Take Profit: 1.2R (1.2x risk distance)
    Trailing: Disabled (uses fixed TP for mean reversion)

    Regime: Only trades in RANGE regime
    """

    # Bulgu 8: Explicit regime compatibility
    compatible_regimes = [RegimeType.RANGE]

    def entry_conditions(
        self,
        features: FeatureSet,
        regime_result,
        symbol: str,
        current_price: float
    ) -> tuple[bool, Optional[OrderSide], str]:
        """
        Check range mean reversion entry conditions.

        Args:
            features: Technical indicators
            regime_result: Regime detection result
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            (conditions_met, side, reason)
        """
        # Only trade in RANGE regime
        if regime_result.regime != RegimeType.RANGE:
            return False, None, "Not in RANGE regime"

        # Extract features
        rsi = features.rsi_5m
        bb_upper = features.bb_upper_5m
        bb_lower = features.bb_lower_5m

        # RSI extreme thresholds
        rsi_long_extreme = self.config.get("rsi_long_extreme", 25)
        rsi_short_extreme = self.config.get("rsi_short_extreme", 75)

        # BB touch threshold - within 0.1% of band
        bb_touch_pct = 0.001

        # Check LONG setup (oversold + BB lower touch)
        if rsi < rsi_long_extreme:
            distance_to_lower = abs(current_price - bb_lower) / bb_lower

            if distance_to_lower <= bb_touch_pct:
                reason = (
                    f"Range mean reversion LONG: RSI={rsi:.1f} (oversold < {rsi_long_extreme}), "
                    f"BB lower band touch (price={current_price:.2f}, BB_lower={bb_lower:.2f})"
                )
                logger.info(f"{symbol}: {reason}")
                return True, OrderSide.LONG, reason
            else:
                logger.debug(
                    f"{symbol}: RSI oversold ({rsi:.1f}) but not at BB lower "
                    f"(distance={distance_to_lower*100:.3f}% > {bb_touch_pct*100:.1f}%)"
                )

        # Check SHORT setup (overbought + BB upper touch)
        elif rsi > rsi_short_extreme:
            distance_to_upper = abs(current_price - bb_upper) / bb_upper

            if distance_to_upper <= bb_touch_pct:
                reason = (
                    f"Range mean reversion SHORT: RSI={rsi:.1f} (overbought > {rsi_short_extreme}), "
                    f"BB upper band touch (price={current_price:.2f}, BB_upper={bb_upper:.2f})"
                )
                logger.info(f"{symbol}: {reason}")
                return True, OrderSide.SHORT, reason
            else:
                logger.debug(
                    f"{symbol}: RSI overbought ({rsi:.1f}) but not at BB upper "
                    f"(distance={distance_to_upper*100:.3f}% > {bb_touch_pct*100:.1f}%)"
                )

        return False, None, "Mean reversion conditions not met"

    # Bulgu 6: calculate_stop_loss and calculate_take_profit use base class defaults.
    # Base reads "stop_pct" (0.008) and "target_r_multiple" (1.2) from self.config automatically.
