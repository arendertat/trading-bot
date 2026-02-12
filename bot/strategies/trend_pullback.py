"""Trend Pullback Strategy implementation"""

import logging
from typing import Optional

from bot.strategies.base import Strategy, FeatureSet
from bot.core.constants import OrderSide, RegimeType

logger = logging.getLogger("trading_bot.strategies.trend_pullback")


class TrendPullbackStrategy(Strategy):
    """
    Trend Pullback Strategy.

    Trade pullbacks in established trends using EMA structure and RSI.

    Entry Logic:
    - LONG: 1h bullish trend + 5m pullback to EMA20 + RSI 40-50
    - SHORT: 1h bearish trend + 5m pullback to EMA20 + RSI 50-60

    Stop Loss: Fixed 1.0% from entry
    Take Profit: 1.5R (1.5x risk distance)
    Trailing: Enabled after 1.0R profit, 2.0 * ATR distance

    Regime: Only trades in TREND regime
    """

    def entry_conditions(
        self,
        features: FeatureSet,
        regime_result,
        symbol: str,
        current_price: float
    ) -> tuple[bool, Optional[OrderSide], str]:
        """
        Check trend pullback entry conditions.

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
        rsi = features.rsi_5m
        ema20_5m = features.ema20_5m
        ema50_5m = features.ema50_5m

        # Pullback band - price within configured % of EMA20
        ema20_band_pct = self.config.get("ema20_band_pct", 0.002)  # 0.2%
        price_distance_from_ema20 = abs(current_price - ema20_5m) / ema20_5m
        near_ema20 = price_distance_from_ema20 <= ema20_band_pct

        # Check LONG setup
        if regime_result.trend_direction == "bullish":
            # 5m structure validation: EMA20 > EMA50 OR price > EMA50
            structure_ok = ema20_5m > ema50_5m or current_price > ema50_5m

            # RSI in pullback range
            rsi_min = self.config.get("pullback_rsi_long_min", 40)
            rsi_max = self.config.get("pullback_rsi_long_max", 50)
            rsi_ok = rsi_min <= rsi <= rsi_max

            if structure_ok and near_ema20 and rsi_ok:
                reason = (
                    f"Trend pullback LONG: 1h bullish trend, price near EMA20 "
                    f"({price_distance_from_ema20*100:.2f}% away), RSI={rsi:.1f}"
                )
                logger.info(f"{symbol}: {reason}")
                return True, OrderSide.LONG, reason

        # Check SHORT setup
        elif regime_result.trend_direction == "bearish":
            # 5m structure validation: EMA20 < EMA50 OR price < EMA50
            structure_ok = ema20_5m < ema50_5m or current_price < ema50_5m

            # RSI in pullback range
            rsi_min = self.config.get("pullback_rsi_short_min", 50)
            rsi_max = self.config.get("pullback_rsi_short_max", 60)
            rsi_ok = rsi_min <= rsi <= rsi_max

            if structure_ok and near_ema20 and rsi_ok:
                reason = (
                    f"Trend pullback SHORT: 1h bearish trend, price near EMA20 "
                    f"({price_distance_from_ema20*100:.2f}% away), RSI={rsi:.1f}"
                )
                logger.info(f"{symbol}: {reason}")
                return True, OrderSide.SHORT, reason

        return False, None, "Pullback conditions not met"

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: OrderSide,
        atr: float
    ) -> float:
        """
        Calculate stop loss at fixed percentage.

        Args:
            entry_price: Entry fill price
            side: LONG or SHORT
            atr: Average True Range (not used for fixed stop)

        Returns:
            Stop loss price
        """
        stop_pct = self.config.get("stop_pct", 0.01)  # 1.0%

        if side == OrderSide.LONG:
            stop_price = entry_price * (1 - stop_pct)
        else:  # SHORT
            stop_price = entry_price * (1 + stop_pct)

        return stop_price

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_price: float,
        side: OrderSide
    ) -> float:
        """
        Calculate take profit at fixed R multiple.

        Args:
            entry_price: Entry fill price
            stop_price: Stop loss price
            side: LONG or SHORT

        Returns:
            Take profit price
        """
        target_r = self.config.get("target_r_multiple", 1.5)

        risk_distance = abs(entry_price - stop_price)
        target_distance = risk_distance * target_r

        if side == OrderSide.LONG:
            tp_price = entry_price + target_distance
        else:  # SHORT
            tp_price = entry_price - target_distance

        return tp_price
