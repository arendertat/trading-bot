"""Regime detection engine"""

import logging
from typing import Optional

from bot.config.models import RegimeConfig
from bot.core.constants import RegimeType
from bot.regime.models import RegimeResult


logger = logging.getLogger("trading_bot.regime")


class RegimeDetector:
    """
    Rule-based regime classifier.

    Classifies market regime as:
    - TREND: Strong directional movement
    - RANGE: Sideways/consolidation
    - HIGH_VOL: High volatility
    - CHOP_NO_TRADE: Low confidence or adverse conditions
    """

    def __init__(self, config: RegimeConfig):
        """
        Initialize RegimeDetector.

        Args:
            config: Regime configuration
        """
        self.config = config
        logger.info("RegimeDetector initialized")

    def detect_regime(
        self,
        symbol: str,
        adx: float,
        atr_z: float,
        bb_width: float,
        ema20_5m: float,
        ema50_5m: float,
        ema20_1h: float,
        ema50_1h: float,
        spread_ok: bool = True,
    ) -> RegimeResult:
        """
        Detect market regime for a symbol.

        Args:
            symbol: Symbol name
            adx: ADX(14) on 5m
            atr_z: ATR z-score on 5m
            bb_width: Bollinger Band width on 5m
            ema20_5m: EMA(20) on 5m
            ema50_5m: EMA(50) on 5m
            ema20_1h: EMA(20) on 1h
            ema50_1h: EMA(50) on 1h
            spread_ok: Whether spread filter passes (default True)

        Returns:
            RegimeResult with regime classification and confidence
        """
        # If spread fails, immediately return CHOP_NO_TRADE
        if not spread_ok:
            logger.warning(f"{symbol}: CHOP_NO_TRADE (spread filter failed)")
            return RegimeResult(
                symbol=symbol,
                regime=RegimeType.CHOP_NO_TRADE,
                confidence=1.0,
                adx=adx,
                atr_z=atr_z,
                bb_width=bb_width,
                ema20_1h=ema20_1h,
                ema50_1h=ema50_1h,
                reasons=["Spread filter failed"],
                trend_direction=None,
            )

        reasons = []
        regime_scores = {
            RegimeType.HIGH_VOL: 0.0,
            RegimeType.TREND: 0.0,
            RegimeType.RANGE: 0.0,
            RegimeType.CHOP_NO_TRADE: 0.0,
        }
        trend_direction = None

        # Rule 1: HIGH_VOL if ATR_Z > threshold
        if atr_z > self.config.high_vol_atr_z:
            regime_scores[RegimeType.HIGH_VOL] = self._compute_high_vol_confidence(atr_z)
            reasons.append(f"High volatility: ATR_Z={atr_z:.2f}")

        # Rule 2: TREND if ADX > threshold and 1h trend is directional
        trend_1h_bullish = ema20_1h > ema50_1h
        trend_1h_bearish = ema20_1h < ema50_1h

        if adx > self.config.trend_adx_min and (trend_1h_bullish or trend_1h_bearish):
            regime_scores[RegimeType.TREND] = self._compute_trend_confidence(
                adx, trend_1h_bullish, trend_1h_bearish
            )
            trend_direction = "bullish" if trend_1h_bullish else "bearish"
            reasons.append(f"Trend ({trend_direction}): ADX={adx:.1f}")

        # Rule 3: RANGE if ADX < threshold and BB width within range
        if adx < self.config.range_adx_max:
            bb_in_range = self._check_bb_width_range(bb_width)
            if bb_in_range:
                regime_scores[RegimeType.RANGE] = self._compute_range_confidence(adx, bb_width)
                reasons.append(f"Range: ADX={adx:.1f}, BB_width={bb_width:.4f}")

        # Select regime with highest score
        max_regime = max(regime_scores, key=regime_scores.get)
        max_confidence = regime_scores[max_regime]

        # Rule 4: CHOP_NO_TRADE if confidence < threshold
        if max_confidence < self.config.confidence_threshold:
            regime = RegimeType.CHOP_NO_TRADE
            confidence = max_confidence
            reasons.append(f"Low confidence: {confidence:.2f} < {self.config.confidence_threshold}")
        else:
            regime = max_regime
            confidence = max_confidence

        logger.debug(
            f"{symbol}: {regime.value} (confidence={confidence:.2f}, "
            f"ADX={adx:.1f}, ATR_Z={atr_z:.2f}, BB_width={bb_width:.4f})"
        )

        return RegimeResult(
            symbol=symbol,
            regime=regime,
            confidence=confidence,
            adx=adx,
            atr_z=atr_z,
            bb_width=bb_width,
            ema20_1h=ema20_1h,
            ema50_1h=ema50_1h,
            reasons=reasons,
            trend_direction=trend_direction if regime == RegimeType.TREND else None,
        )

    def _compute_high_vol_confidence(self, atr_z: float) -> float:
        """
        Compute confidence for HIGH_VOL regime.

        Confidence increases with ATR z-score above threshold.

        Args:
            atr_z: ATR z-score

        Returns:
            Confidence score [0, 1]
        """
        if atr_z <= self.config.high_vol_atr_z:
            return 0.0

        # Linear scaling from threshold to threshold+2
        # threshold -> 0.6, threshold+2 -> 1.0
        excess = atr_z - self.config.high_vol_atr_z
        confidence = 0.6 + (excess / 2.0) * 0.4

        return max(0.0, min(1.0, confidence))

    def _compute_trend_confidence(
        self,
        adx: float,
        trend_1h_bullish: bool,
        trend_1h_bearish: bool,
    ) -> float:
        """
        Compute confidence for TREND regime.

        Confidence increases with ADX above threshold.

        Args:
            adx: ADX value
            trend_1h_bullish: 1h trend is bullish
            trend_1h_bearish: 1h trend is bearish

        Returns:
            Confidence score [0, 1]
        """
        if adx <= self.config.trend_adx_min:
            return 0.0

        if not (trend_1h_bullish or trend_1h_bearish):
            return 0.0

        # Linear scaling from trend_adx_min to trend_adx_min+20
        # trend_adx_min -> 0.6, trend_adx_min+20 -> 1.0
        excess = adx - self.config.trend_adx_min
        confidence = 0.6 + (excess / 20.0) * 0.4

        return max(0.0, min(1.0, confidence))

    def _compute_range_confidence(self, adx: float, bb_width: float) -> float:
        """
        Compute confidence for RANGE regime.

        Confidence higher when ADX is low and BB width is moderate.

        Args:
            adx: ADX value
            bb_width: Bollinger Band width

        Returns:
            Confidence score [0, 1]
        """
        if adx >= self.config.range_adx_max:
            return 0.0

        # ADX component: lower is better
        # range_adx_max -> 0.6, 0 -> 1.0
        adx_score = 0.6 + (1.0 - adx / self.config.range_adx_max) * 0.4

        # BB width component: prefer moderate width
        bb_score = 1.0
        if self.config.bb_width_range_min is not None and self.config.bb_width_range_max is not None:
            if bb_width < self.config.bb_width_range_min:
                # Too tight
                bb_score = 0.5
            elif bb_width > self.config.bb_width_range_max:
                # Too wide
                bb_score = 0.7

        confidence = adx_score * bb_score

        return max(0.0, min(1.0, confidence))

    def _check_bb_width_range(self, bb_width: float) -> bool:
        """
        Check if BB width is within acceptable range.

        Args:
            bb_width: Bollinger Band width

        Returns:
            True if in range or no range configured
        """
        if self.config.bb_width_range_min is None or self.config.bb_width_range_max is None:
            return True

        return self.config.bb_width_range_min <= bb_width <= self.config.bb_width_range_max
