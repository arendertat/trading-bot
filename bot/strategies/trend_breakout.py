"""Trend Breakout Strategy implementation"""

import logging
from typing import Optional

from bot.strategies.base import Strategy, FeatureSet
from bot.core.constants import OrderSide, RegimeType

logger = logging.getLogger("trading_bot.strategies.trend_breakout")


class TrendBreakoutStrategy(Strategy):
    """
    Trend Breakout Strategy.

    Trade breakouts of recent price ranges with volume confirmation.

    Entry Logic:
    - LONG: Break above 20-bar high + volume z-score > 1.0 (bullish trend)
    - SHORT: Break below 20-bar low + volume z-score > 1.0 (bearish trend)

    Stop Loss: Fixed 1.0% from entry
    Take Profit: Very far (100R) - relies on trailing stop for exit
    Trailing: Enabled immediately, 2.5 * ATR distance

    Regime: Only trades in TREND and HIGH_VOL regimes
    """

    # Bulgu 8: Explicit regime compatibility
    compatible_regimes = [RegimeType.TREND, RegimeType.HIGH_VOL]

    def entry_conditions(
        self,
        features: FeatureSet,
        regime_result,
        symbol: str,
        current_price: float
    ) -> tuple[bool, Optional[OrderSide], str]:
        """
        Check trend breakout entry conditions.

        Args:
            features: Technical indicators
            regime_result: Regime detection result
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            (conditions_met, side, reason)
        """
        # Only trade in TREND regime
        if regime_result.regime != RegimeType.TREND:
            return False, None, "Not in TREND regime"

        # Extract features
        high_20 = features.high_20_bars
        low_20 = features.low_20_bars
        volume_z = features.volume_z_5m

        # 4h alignment check (Özellik 6) — skipped if data not yet available
        use_4h = self.config.get("use_4h_confirmation", False)
        ema20_4h = features.ema20_4h
        ema50_4h = features.ema50_4h
        has_4h_data = ema20_4h is not None and ema50_4h is not None

        # Volume confirmation threshold
        volume_z_min = self.config.get("breakout_volume_z_min", 1.0)
        volume_ok = volume_z > volume_z_min

        # Check LONG breakout (break above 20-bar high)
        if regime_result.trend_direction == "bullish" and current_price > high_20:
            if use_4h and has_4h_data and ema20_4h <= ema50_4h:
                logger.debug(
                    f"{symbol}: 4h bearish structure conflicts LONG breakout — skip"
                )
                return False, None, "4h structure conflicts LONG direction"
            if volume_ok:
                tf_note = " +4h✓" if (use_4h and has_4h_data) else ""
                reason = (
                    f"Trend breakout LONG{tf_note}: Break above 20-bar high ({high_20:.2f}), "
                    f"volume_z={volume_z:.2f}"
                )
                logger.info(f"{symbol}: {reason}")
                return True, OrderSide.LONG, reason
            else:
                logger.debug(
                    f"{symbol}: LONG breakout without volume confirmation "
                    f"(volume_z={volume_z:.2f} < {volume_z_min})"
                )

        # Check SHORT breakout (break below 20-bar low)
        elif regime_result.trend_direction == "bearish" and current_price < low_20:
            if use_4h and has_4h_data and ema20_4h >= ema50_4h:
                logger.debug(
                    f"{symbol}: 4h bullish structure conflicts SHORT breakout — skip"
                )
                return False, None, "4h structure conflicts SHORT direction"
            if volume_ok:
                tf_note = " +4h✓" if (use_4h and has_4h_data) else ""
                reason = (
                    f"Trend breakout SHORT{tf_note}: Break below 20-bar low ({low_20:.2f}), "
                    f"volume_z={volume_z:.2f}"
                )
                logger.info(f"{symbol}: {reason}")
                return True, OrderSide.SHORT, reason
            else:
                logger.debug(
                    f"{symbol}: SHORT breakout without volume confirmation "
                    f"(volume_z={volume_z:.2f} < {volume_z_min})"
                )

        return False, None, "Breakout conditions not met"

    # Bulgu 6: calculate_stop_loss uses base class default (reads "stop_pct" from config).

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_price: float,
        side: OrderSide
    ) -> float:
        """
        Calculate take profit at very far distance (relies on trailing).

        Breakout strategy doesn't use fixed TP - sets TP at 100R distance
        so trailing stop is the primary exit mechanism.

        Args:
            entry_price: Entry fill price
            stop_price: Stop loss price
            side: LONG or SHORT

        Returns:
            Take profit price (100R away)
        """
        risk_distance = abs(entry_price - stop_price)
        target_distance = risk_distance * 100  # 100R

        if side == OrderSide.LONG:
            tp_price = entry_price + target_distance
        else:  # SHORT
            tp_price = entry_price - target_distance

        return tp_price
