from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from canslim.config import CriteriaThresholds
from canslim.models import CriterionResult, EarningsBundle, InstitutionalSnapshot, PatternMatch, PriceFeatures


@dataclass
class CriterionContext:
    """Everything a criterion might read for one ticker."""

    ticker: str
    thresholds: CriteriaThresholds
    price_features: Optional[PriceFeatures] = None
    earnings: Optional[EarningsBundle] = None
    institutional: Optional[InstitutionalSnapshot] = None
    float_shares: Optional[float] = None
    market_cap: Optional[float] = None  # USD; gated as an early floor by the scanner, also readable by criteria
    rs_percentile: Optional[float] = None  # filled by scanner after cross-sectional rank
    patterns: list[PatternMatch] = field(default_factory=list)


class Criterion(ABC):
    letter: str
    name: str
    is_gate: bool

    @abstractmethod
    def evaluate(self, ctx: CriterionContext) -> CriterionResult:
        ...
