"""Data models for regime detection"""

from dataclasses import dataclass
from typing import Optional
from bot.core.constants import RegimeType


@dataclass
class RegimeResult:
    """Regime detection result for a symbol"""
    symbol: str
    regime: RegimeType
    confidence: float  # 0.0 to 1.0
    adx: float
    atr_z: float
    bb_width: float
    ema20_1h: float
    ema50_1h: float
    reasons: list[str]
    trend_direction: Optional[str] = None  # "bullish" or "bearish" if TREND
    # Diagnostics
    trend_score: Optional[float] = None
    range_score: Optional[float] = None
    high_vol_score: Optional[float] = None
    chop_score: Optional[int] = None
    chop_signals: Optional[list[bool]] = None
    gate_reason: Optional[str] = None

    def __post_init__(self):
        """Ensure confidence is clamped to [0, 1]"""
        self.confidence = max(0.0, min(1.0, self.confidence))
