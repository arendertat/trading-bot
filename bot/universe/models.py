"""Data models for universe selection"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SymbolEligibility:
    """Symbol eligibility check results"""
    symbol: str
    pass_volume: bool
    pass_spread: bool
    pass_funding: bool
    pass_atr_ratio: bool
    score: float
    reasons: List[str] = field(default_factory=list)

    @property
    def is_eligible(self) -> bool:
        """Check if symbol passes all filters"""
        return all([
            self.pass_volume,
            self.pass_spread,
            self.pass_funding,
            self.pass_atr_ratio,
        ])
