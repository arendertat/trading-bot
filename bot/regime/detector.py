"""Regime detection engine"""

import logging
from collections import deque
from typing import Optional

import numpy as np

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

        # Özellik 11: Adaptive regime thresholds
        # Rolling ADX history across all symbols (combined)
        self._adx_history: deque = deque(
            maxlen=config.adaptive_adx_window
        )
        # Live adaptive thresholds (start from config defaults)
        self._adaptive_trend_adx_min: float = config.trend_adx_min
        self._adaptive_range_adx_max: float = config.range_adx_max
        self._regime_state = {}

        logger.info(
            f"RegimeDetector initialized "
            f"(adaptive={'ON' if config.adaptive_regime else 'OFF'})"
        )

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
        kaufman_er: Optional[float] = None,
        flip_rate: Optional[float] = None,
        ema1h_spread_pct: Optional[float] = None,
        bb_width_pct_rank: Optional[float] = None,
        rsi_5m: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_lower: Optional[float] = None,
        last_close_5m: Optional[float] = None,
        rsi_extreme_low: Optional[float] = None,
        rsi_extreme_high: Optional[float] = None,
        spread_ok: bool = True,
    ) -> RegimeResult:
        """
        Detect market regime for a symbol.

        Args:
            symbol: Symbol name
            adx: ADX(14) on 5m
            atr_z: ATR z-score on 5m
            bb_width: Bollinger Band width on 5m
            kaufman_er: Kaufman Efficiency Ratio on 5m
            flip_rate: Flip rate on 5m
            ema1h_spread_pct: EMA20/EMA50 spread % on 1h
            bb_width_pct_rank: BB width percentile rank (optional)
            rsi_5m: RSI(14) on 5m
            bb_upper: Bollinger upper band on 5m
            bb_lower: Bollinger lower band on 5m
            last_close_5m: Last 5m close
            rsi_extreme_low: RSI extreme low threshold
            rsi_extreme_high: RSI extreme high threshold
            ema20_5m: EMA(20) on 5m
            ema50_5m: EMA(50) on 5m
            ema20_1h: EMA(20) on 1h
            ema50_1h: EMA(50) on 1h
            spread_ok: Whether spread filter passes (default True)

        Returns:
            RegimeResult with regime classification and confidence
        """
        # Özellik 11: Update adaptive ADX thresholds from rolling history
        if self.config.adaptive_regime and adx > 0:
            self._adx_history.append(adx)
            self._update_adaptive_thresholds()

        # If spread fails, immediately return CHOP_NO_TRADE
        if not spread_ok:
            logger.warning(f"{symbol}: CHOP_NO_TRADE (spread filter failed)")
            result = RegimeResult(
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
                trend_score=0.0,
                range_score=0.0,
                high_vol_score=0.0,
                chop_score=None,
                chop_signals=None,
            )
            self._update_regime_state(symbol, result.regime, result.confidence, None)
            self._log_chop_debug(
                symbol,
                adx,
                kaufman_er,
                flip_rate,
                ema1h_spread_pct,
                bb_width,
                bb_width_pct_rank,
            )
            return result

        reasons = []
        regime_scores = {
            RegimeType.HIGH_VOL: 0.0,
            RegimeType.TREND: 0.0,
            RegimeType.RANGE: 0.0,
            RegimeType.CHOP_NO_TRADE: 0.0,
        }
        trend_direction = None

        chop_score, chop_signals = self._compute_chop_score(
            adx=adx,
            kaufman_er=kaufman_er,
            flip_rate=flip_rate,
            ema1h_spread_pct=ema1h_spread_pct,
            bb_width=bb_width,
            bb_width_pct_rank=bb_width_pct_rank,
        )

        # Rule 1: HIGH_VOL if ATR_Z > threshold
        if atr_z > self.config.high_vol_atr_z:
            regime_scores[RegimeType.HIGH_VOL] = self._compute_high_vol_confidence(atr_z)
            reasons.append(f"High volatility: ATR_Z={atr_z:.2f}")
            result = RegimeResult(
                symbol=symbol,
                regime=RegimeType.HIGH_VOL,
                confidence=regime_scores[RegimeType.HIGH_VOL],
                adx=adx,
                atr_z=atr_z,
                bb_width=bb_width,
                ema20_1h=ema20_1h,
                ema50_1h=ema50_1h,
                reasons=reasons,
                trend_direction=None,
                trend_score=0.0,
                range_score=0.0,
                high_vol_score=regime_scores[RegimeType.HIGH_VOL],
                chop_score=chop_score,
                chop_signals=chop_signals,
            )
            self._update_regime_state(symbol, result.regime, result.confidence, None)
            self._log_chop_debug(
                symbol,
                adx,
                kaufman_er,
                flip_rate,
                ema1h_spread_pct,
                bb_width,
                bb_width_pct_rank,
                chop_score=chop_score,
                chop_signals=chop_signals,
            )
            return result

        # Rule 2: CHOP_NO_TRADE if chop_score >= threshold
        if chop_score >= self.config.chop.score_threshold:
            reasons.append(
                f"Chop gate: score={chop_score} (S1..S5={chop_signals})"
            )
            result = RegimeResult(
                symbol=symbol,
                regime=RegimeType.CHOP_NO_TRADE,
                confidence=1.0,
                adx=adx,
                atr_z=atr_z,
                bb_width=bb_width,
                ema20_1h=ema20_1h,
                ema50_1h=ema50_1h,
                reasons=reasons,
                trend_direction=None,
                trend_score=0.0,
                range_score=0.0,
                high_vol_score=0.0,
                chop_score=chop_score,
                chop_signals=chop_signals,
            )
            self._update_regime_state(symbol, result.regime, result.confidence, None)
            self._log_chop_debug(
                symbol,
                adx,
                kaufman_er,
                flip_rate,
                ema1h_spread_pct,
                bb_width,
                bb_width_pct_rank,
                chop_score=chop_score,
                chop_signals=chop_signals,
            )
            return result

        # Rule 3: TREND if ADX > threshold and 1h trend is directional
        trend_1h_bullish = ema20_1h > ema50_1h
        trend_1h_bearish = ema20_1h < ema50_1h

        trend_threshold = self._adaptive_trend_adx_min
        if adx > trend_threshold and (trend_1h_bullish or trend_1h_bearish):
            regime_scores[RegimeType.TREND] = self._compute_trend_confidence(
                adx, trend_1h_bullish, trend_1h_bearish
            )
            trend_direction = "bullish" if trend_1h_bullish else "bearish"
            reasons.append(f"Trend ({trend_direction}): ADX={adx:.1f} (thresh={trend_threshold:.1f})")

        # Rule 4: RANGE if ADX < threshold and BB width within range
        range_threshold = self._adaptive_range_adx_max
        if adx < range_threshold:
            bb_in_range = self._check_bb_width_range(bb_width)
            if bb_in_range:
                extremes_ok = self._range_extremes_ok(
                    rsi_5m=rsi_5m,
                    bb_upper=bb_upper,
                    bb_lower=bb_lower,
                    last_close_5m=last_close_5m,
                    rsi_extreme_low=rsi_extreme_low,
                    rsi_extreme_high=rsi_extreme_high,
                )
                if self.config.chop.range_requires_extremes and not extremes_ok:
                    reasons.append(
                        "Range blocked: no BB touch or RSI extreme"
                    )
                else:
                    regime_scores[RegimeType.RANGE] = self._compute_range_confidence(adx, bb_width)
                    reasons.append(f"Range: ADX={adx:.1f}, BB_width={bb_width:.4f}")

        # Select regime with highest score
        max_regime = max(regime_scores, key=regime_scores.get)
        max_confidence = regime_scores[max_regime]

        # Rule 5: CHOP_NO_TRADE if confidence < threshold
        if max_confidence < self.config.confidence_threshold:
            regime = RegimeType.CHOP_NO_TRADE
            confidence = max_confidence
            reasons.append(f"Low confidence: {confidence:.2f} < {self.config.confidence_threshold}")
        else:
            regime = max_regime
            confidence = max_confidence

        debounced = False
        if regime in (RegimeType.TREND, RegimeType.RANGE):
            regime, confidence, trend_direction, debounced = self._apply_debounce(
                symbol=symbol,
                proposed_regime=regime,
                proposed_confidence=confidence,
                proposed_trend_direction=trend_direction,
            )
            if debounced:
                reasons.append("Debounce: holding previous regime")
        else:
            self._update_regime_state(symbol, regime, confidence, trend_direction)

        self._log_chop_debug(
            symbol,
            adx,
            kaufman_er,
            flip_rate,
            ema1h_spread_pct,
            bb_width,
            bb_width_pct_rank,
            chop_score=chop_score,
            chop_signals=chop_signals,
        )

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
            trend_score=regime_scores[RegimeType.TREND],
            range_score=regime_scores[RegimeType.RANGE],
            high_vol_score=regime_scores[RegimeType.HIGH_VOL],
            chop_score=chop_score,
            chop_signals=chop_signals,
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
        threshold = self._adaptive_trend_adx_min
        if adx <= threshold:
            return 0.0

        if not (trend_1h_bullish or trend_1h_bearish):
            return 0.0

        # Linear scaling from threshold to threshold+20
        excess = adx - threshold
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
        threshold = self._adaptive_range_adx_max
        if adx >= threshold:
            return 0.0

        # ADX component: lower is better
        adx_score = 0.6 + (1.0 - adx / max(threshold, 1.0)) * 0.4

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

    def _update_adaptive_thresholds(self) -> None:
        """
        Update ADX thresholds from rolling history (Özellik 11).

        Uses:
          trend_threshold = 75th percentile of recent ADX values
          range_threshold = 25th percentile of recent ADX values

        Requires at least 50 samples; falls back to config defaults otherwise.
        Enforces range_threshold < trend_threshold (min 2-point gap).
        """
        if len(self._adx_history) < 50:
            return  # Not enough data yet, keep config defaults

        arr = np.array(self._adx_history)
        trend_pct = float(np.percentile(arr, 75))
        range_pct = float(np.percentile(arr, 25))

        # Clamp to config validator bounds
        trend_pct = max(15.0, min(50.0, trend_pct))
        range_pct = max(10.0, min(30.0, range_pct))

        # Enforce gap: range must be < trend by at least 2
        if range_pct >= trend_pct - 2.0:
            range_pct = trend_pct - 2.0

        if (abs(trend_pct - self._adaptive_trend_adx_min) > 0.5 or
                abs(range_pct - self._adaptive_range_adx_max) > 0.5):
            logger.info(
                f"Adaptive ADX thresholds updated: "
                f"trend_min={trend_pct:.1f} (was {self._adaptive_trend_adx_min:.1f}), "
                f"range_max={range_pct:.1f} (was {self._adaptive_range_adx_max:.1f}) "
                f"[{len(self._adx_history)} samples]"
            )

        self._adaptive_trend_adx_min = trend_pct
        self._adaptive_range_adx_max = range_pct

    def _compute_chop_score(
        self,
        adx: float,
        kaufman_er: Optional[float],
        flip_rate: Optional[float],
        ema1h_spread_pct: Optional[float],
        bb_width: Optional[float],
        bb_width_pct_rank: Optional[float],
    ) -> tuple[int, list[bool]]:
        chop_cfg = self.config.chop
        s1 = adx < self._adaptive_range_adx_max
        s2 = kaufman_er is not None and kaufman_er <= chop_cfg.er_max
        s3 = flip_rate is not None and flip_rate >= chop_cfg.flip_rate_min
        s4 = ema1h_spread_pct is not None and ema1h_spread_pct <= chop_cfg.ema1h_spread_max

        s5 = False
        if (
            bb_width_pct_rank is not None
            and chop_cfg.bb_width_percentile_lookback is not None
            and chop_cfg.bb_width_percentile_max is not None
        ):
            s5 = bb_width_pct_rank <= chop_cfg.bb_width_percentile_max
        elif bb_width is not None and chop_cfg.bb_width_chop_max is not None:
            s5 = bb_width <= chop_cfg.bb_width_chop_max

        chop_signals = [s1, s2, s3, s4, s5]
        chop_score = sum(1 for s in chop_signals if s)
        return chop_score, chop_signals

    def _range_extremes_ok(
        self,
        rsi_5m: Optional[float],
        bb_upper: Optional[float],
        bb_lower: Optional[float],
        last_close_5m: Optional[float],
        rsi_extreme_low: Optional[float],
        rsi_extreme_high: Optional[float],
    ) -> bool:
        if last_close_5m is not None and bb_upper is not None and bb_lower is not None:
            if last_close_5m >= bb_upper or last_close_5m <= bb_lower:
                return True

        if rsi_5m is not None and rsi_extreme_low is not None and rsi_extreme_high is not None:
            if rsi_5m <= rsi_extreme_low or rsi_5m >= rsi_extreme_high:
                return True

        return False

    def _apply_debounce(
        self,
        symbol: str,
        proposed_regime: RegimeType,
        proposed_confidence: float,
        proposed_trend_direction: Optional[str],
    ) -> tuple[RegimeType, float, Optional[str], bool]:
        cfg = self.config.chop
        state = self._regime_state.get(symbol)
        if state is None:
            self._update_regime_state(
                symbol,
                proposed_regime,
                proposed_confidence,
                proposed_trend_direction,
            )
            return proposed_regime, proposed_confidence, proposed_trend_direction, False

        confirmed = state["confirmed"]
        if proposed_regime == confirmed:
            self._update_regime_state(
                symbol,
                confirmed,
                proposed_confidence,
                proposed_trend_direction,
            )
            return confirmed, proposed_confidence, proposed_trend_direction, False

        if state["cooldown"] > 0:
            state["cooldown"] = max(0, state["cooldown"] - 1)
            return confirmed, state["confidence"], state["trend_direction"], True

        if cfg.regime_persistence_bars <= 1:
            self._update_regime_state(
                symbol,
                proposed_regime,
                proposed_confidence,
                proposed_trend_direction,
                cooldown=cfg.regime_switch_cooldown_bars,
                reset_pending=True,
            )
            return proposed_regime, proposed_confidence, proposed_trend_direction, False

        if state["pending"] != proposed_regime:
            state["pending"] = proposed_regime
            state["pending_count"] = 1
        else:
            state["pending_count"] += 1

        if state["pending_count"] >= cfg.regime_persistence_bars:
            self._update_regime_state(
                symbol,
                proposed_regime,
                proposed_confidence,
                proposed_trend_direction,
                cooldown=cfg.regime_switch_cooldown_bars,
                reset_pending=True,
            )
            return proposed_regime, proposed_confidence, proposed_trend_direction, False

        return confirmed, state["confidence"], state["trend_direction"], True

    def _update_regime_state(
        self,
        symbol: str,
        regime: RegimeType,
        confidence: float,
        trend_direction: Optional[str],
        cooldown: Optional[int] = None,
        reset_pending: bool = True,
    ) -> None:
        state = self._regime_state.get(symbol)
        if state is None:
            state = {
                "confirmed": regime,
                "pending": None,
                "pending_count": 0,
                "cooldown": 0,
                "confidence": confidence,
                "trend_direction": trend_direction,
            }
            self._regime_state[symbol] = state
            return

        state["confirmed"] = regime
        if reset_pending:
            state["pending"] = None
            state["pending_count"] = 0
        if cooldown is not None:
            state["cooldown"] = cooldown
        state["confidence"] = confidence
        state["trend_direction"] = trend_direction

    def _log_chop_debug(
        self,
        symbol: str,
        adx: float,
        kaufman_er: Optional[float],
        flip_rate: Optional[float],
        ema1h_spread_pct: Optional[float],
        bb_width: Optional[float],
        bb_width_pct_rank: Optional[float],
        chop_score: Optional[int] = None,
        chop_signals: Optional[list[bool]] = None,
    ) -> None:
        if chop_score is None or chop_signals is None:
            chop_score, chop_signals = self._compute_chop_score(
                adx=adx,
                kaufman_er=kaufman_er,
                flip_rate=flip_rate,
                ema1h_spread_pct=ema1h_spread_pct,
                bb_width=bb_width,
                bb_width_pct_rank=bb_width_pct_rank,
            )
        logger.debug(
            f"{symbol}: chop_score={chop_score} "
            f"S1..S5={chop_signals} "
            f"adx={adx:.2f} "
            f"er={kaufman_er if kaufman_er is not None else 'na'} "
            f"flip={flip_rate if flip_rate is not None else 'na'} "
            f"ema1h_spread_pct={ema1h_spread_pct if ema1h_spread_pct is not None else 'na'} "
            f"bb_width={bb_width if bb_width is not None else 'na'} "
            f"bb_width_pct_rank={bb_width_pct_rank if bb_width_pct_rank is not None else 'na'}"
        )

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
