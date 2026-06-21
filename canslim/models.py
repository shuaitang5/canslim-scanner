from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1"


class PriceFeatures(BaseModel):
    """Derived price-bar features, computed once per ticker and shared across N/S/L criteria."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    as_of: date
    close: float
    high_52w: float
    low_52w: float
    adv10: float  # avg dollar volume last 10 sessions
    adv50: float  # avg dollar volume last 50 sessions
    avg_vol50: float  # share volume
    recent_vol_ratio: float  # last-session volume / avg_vol50
    rs_return_12m_weighted: float  # weighted 40/20/20/20 return across the 4 trailing quarters
    dist_to_52w_high_pct: float  # (high_52w - close) / high_52w


class NormalizationAdjustment(BaseModel):
    """A single one-time-items adjustment applied to reported EPS.

    Surfaced in the report so the user sees exactly what was excluded from the
    "clean" earnings number the C/A gates evaluate.
    """

    model_config = ConfigDict(frozen=True)

    rule_name: str                        # e.g. "discontinued_operations"
    description: str                      # human-readable
    period: str                           # "2025" or "2025-Q4"
    concept: str                          # SEC XBRL concept used
    dollar_amount: float                  # pre-share-division raw value
    per_share_impact: float               # per-share after applying tax haircut if needed
    direction: str                        # "subtract" (gain we remove) or "add_back" (loss we restore)
    after_tax: bool                       # True if dollar_amount was already net-of-tax in the filing
    tax_rate_assumed: Optional[float] = None  # set when we had to apply a tax haircut


class EarningsBundle(BaseModel):
    """Parsed earnings data, shared across C and A criteria.

    Both *reported* and *normalized* EPS series are carried. Normalized strips
    known one-time items (gains from divestitures, goodwill impairments, etc.)
    so the CANSLIM C and A gates evaluate the underlying operating business.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    quarterly_eps: list[float] = Field(default_factory=list)   # NORMALIZED, most recent first (used by criteria)
    quarterly_periods: list[str] = Field(default_factory=list)
    annual_eps: list[float] = Field(default_factory=list)      # NORMALIZED annual, most recent first
    annual_periods: list[str] = Field(default_factory=list)
    annual_roe_pct: list[float] = Field(default_factory=list)

    # Reported (un-normalized) series for reference in the deep-dive
    reported_quarterly_eps: list[float] = Field(default_factory=list)
    reported_annual_eps: list[float] = Field(default_factory=list)
    normalization_adjustments: list[NormalizationAdjustment] = Field(default_factory=list)


class InstitutionalSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    reported_at: date  # latest 13F filing date
    inst_own_pct: float  # 0.0-1.0
    qoq_delta_pct: Optional[float] = None
    new_positions: int = 0
    closed_positions: int = 0
    # Staleness markers set by the provider when a fresh fetch failed and we
    # had to fall back to last-known-good cached data. data_age_days==0 means
    # the value is from today; >0 means the cache TTL had expired but we kept
    # the value instead of returning None. Criteria can decide whether to
    # accept stale or treat as unavailable based on age.
    data_age_days: int = 0
    is_stale: bool = False


class CriterionResult(BaseModel):
    """Result of evaluating one CANSLIM letter against one ticker.

    `data_available=False` means we couldn't fetch the data needed to evaluate
    this criterion — distinct from "evaluated and failed". The composite-score
    aggregation and gate-pass logic both treat data-unavailable criteria as
    "abstain" rather than "fail" so missing data doesn't silently collapse
    scores or hide otherwise-strong candidates.
    """

    letter: str
    passed: bool
    is_gate: bool  # True = counts as a hard filter, False = info-only (N, M)
    score: float = 0.0  # 0..1 normalized
    value: Optional[float] = None  # primary numeric value that produced the result
    threshold: Optional[float] = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    # True when the criterion was able to evaluate against real data; False
    # when required inputs were missing (e.g., institutional fetch failed).
    # When False, `passed` is also False but the criterion ABSTAINS from
    # gate logic instead of failing it.
    data_available: bool = True


class ManagementEvent(BaseModel):
    """A management-change disclosure parsed from SEC 8-K Item 5.02 within a trailing window.

    This is the automatable slice of O'Neil's "N" (new products/management/conditions/highs) —
    we surface it as a qualitative signal on top of the mechanical 52-week-high test.
    """

    model_config = ConfigDict(frozen=True)

    filed: date  # filing date of the 8-K
    items: str  # the Item numbers present, e.g. "5.02" or "5.02,8.01"
    url: Optional[str] = None  # direct link to the primary document on EDGAR
    accession: Optional[str] = None


class FetchError(BaseModel):
    """A data-fetch failure for one (ticker, kind, provider).

    The report surfaces these so nothing is silently missing; the negative cache uses
    them to back off without refusing forever.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    kind: str  # "prices" | "fundamentals" | "institutional" | "info" | "pattern"
    provider: str  # "yfinance" | "fmp" | "sec" | pattern name
    error: str
    retryable: bool = True
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FetchSummary(BaseModel):
    """Per-kind breakdown of how data was obtained on this run."""

    kind: str
    cache_hits: int = 0
    fresh_fetches: int = 0
    failures: int = 0
    skipped_negative_cache: int = 0


class PatternMatch(BaseModel):
    """O'Neil-style chart pattern detected on daily bars."""

    model_config = ConfigDict(frozen=True)

    name: str  # "cup_with_handle" | "double_bottom" | ...
    detected: bool
    pivot: Optional[float] = None  # breakout trigger price
    confidence: float = 0.0  # 0..1 heuristic score
    started_on: Optional[date] = None
    completed_on: Optional[date] = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class ScanResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    ticker: str
    as_of: date
    passed: bool  # True when all gate criteria pass
    composite_score: float  # weighted sum of per-criterion scores
    criteria: dict[str, CriterionResult] = Field(default_factory=dict)
    patterns: list[PatternMatch] = Field(default_factory=list)
    errors: list[FetchError] = Field(default_factory=list)
    ad_grade: Optional[str] = None  # Accumulation/Distribution rating: A (strong acc) -> E (heavy dist)
    ad_ratio: Optional[float] = None  # up-flow / (up+down) flow, 0..1
    market_cap: Optional[float] = None  # USD market capitalization at scan time (None when unavailable)
    management_events_90d: list[ManagementEvent] = Field(default_factory=list)
    # Recent SEC 8-K Item 5.02 filings (officer/director changes). Populated only for top
    # candidates during the post-scan enrichment step — null/empty for most tickers.
    status: Literal["scanned", "pending_budget", "skipped_missing_data", "error"] = "scanned"
    status_reason: Optional[str] = None  # human-readable context, e.g. "no prices downloaded"
    error: Optional[str] = None


class MarketRegime(BaseModel):
    as_of: date
    spy_close: float
    spy_sma50: float
    spy_sma200: float
    uptrend: bool  # close > 200d AND 50d > 200d
    reason: str = ""


class SectorRow(BaseModel):
    """One sector's snapshot (typically 11 SPDR sectors)."""
    symbol: str
    name: str
    close: float
    return_30d_pct: float
    rs_percentile: Optional[float] = None  # 0..1 across the 11 sectors
    above_50d_ma: bool
    above_200d_ma: bool


class MarketContext(BaseModel):
    """Aggregate market-mood snapshot for the run.

    Populated post-scan from price_frames + supplementary fetches (VIX, sector ETFs).
    Renders in the HTML report's sticky header to give the user a "should I be
    greedy or scared" read alongside the M-gate's pure regime test.
    """
    as_of: date
    vix_close: Optional[float] = None
    vix_sma50: Optional[float] = None
    vix_label: Optional[str] = None  # "complacent" | "normal" | "elevated" | "fear"
    distribution_days_25: Optional[int] = None  # SPY heavy-volume down days in last 25 sessions
    pct_above_50d_ma: Optional[float] = None  # 0..1 across pre-filtered universe
    pct_above_200d_ma: Optional[float] = None
    new_highs: Optional[int] = None  # within 1% of 52w high
    new_lows: Optional[int] = None   # within 1% of 52w low
    sectors: list[SectorRow] = Field(default_factory=list)
    note: str = ""


class RunManifest(BaseModel):
    run_id: str  # YYYY-MM-DD_HHMMSS
    started_at: datetime
    finished_at: Optional[datetime] = None
    universe_name: str
    universe_size: int
    candidates_after_prefilter: int
    matches: int
    scanned: int
    pending_budget: int
    errored: int
    config_hash: str
    provider_versions: dict[str, str] = Field(default_factory=dict)
    fmp_budget_used: int = 0
    fmp_budget_remaining: Optional[int] = None
    market_regime: Optional[MarketRegime] = None
    market_context: Optional[MarketContext] = None
    fetch_summary: list[FetchSummary] = Field(default_factory=list)
    errors: list[FetchError] = Field(default_factory=list)
