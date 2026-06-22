from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    enabled: bool = True
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None  # name of env var to read the key from
    daily_budget: Optional[int] = None  # requests/day (FMP free tier = 250)
    request_timeout_s: float = 20.0
    concurrency: int = 8
    max_retries: int = 4

    def resolved_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


class CacheConfig(BaseModel):
    root: str = "cache"
    price_ttl_hours: float = 20.0  # ~daily
    fundamentals_ttl_hours: float = 24.0 * 7  # ~weekly
    institutional_ttl_hours: float = 24.0 * 7
    failure_ttl_hours: float = 2.0  # negative-cache TTL — how long to back off after a failure
    # Bounded per-ticker retry for the dedicated institutional holder endpoints.
    # A transient yfinance throttle on a few stragglers under a full us_all scan
    # gets a 2nd/3rd attempt with short backoff before we fall back to stale
    # cache. Kept LOW so it can't blow up runtime on ~1500 tickers: only the
    # institutional call retries, only when the fresh fetch came back empty, and
    # only when there's NO usable stale cache to fall back to instead.
    institutional_retry_attempts: int = 3  # total attempts incl. the first
    institutional_retry_backoff_s: float = 0.5  # base backoff; grows linearly per retry


class CriteriaThresholds(BaseModel):
    # C — current quarterly EPS
    c_min_yoy: float = 0.25  # 25%
    c_require_accelerating: bool = True

    # A — annual EPS
    a_min_annual_yoy: float = 0.25
    a_required_years: int = 3
    a_min_roe_pct: float = 0.17
    a_allow_cagr_fallback: bool = True
    # Leadership-confirmed turnaround override: when a turnaround stock has
    # top-decile RS AND a high-confidence chart pattern, allow the A gate to
    # pass even if ROE hasn't recovered yet. Captures O'Neil's "major rally
    # off a deep base after a turning point" setups (e.g., LITE 2026).
    a_leadership_override_enabled: bool = True
    a_leadership_override_min_rs: float = 0.90
    a_leadership_override_min_pattern_conf: float = 0.65

    # N — new high (info only)
    n_max_dist_to_high_pct: float = 0.15
    n_breakout_pivot_pct: float = 0.05
    n_breakout_volume_multiple: float = 1.4

    # S — supply/demand
    s_min_adv10_over_adv50: float = 1.2
    s_max_float_shares: float = 1_000_000_000.0
    # Pattern-aware relaxation: cup-with-handle, high-tight-flag, three-weeks-tight,
    # and flat-base all expect drying-up volume in the consolidation. When such a
    # pattern is detected with high confidence, treat the dry-up as constructive
    # rather than failing the volume-uptick gate. Float check still applies.
    s_pattern_override_enabled: bool = True
    s_pattern_override_min_conf: float = 0.65
    s_pattern_override_patterns: list[str] = Field(
        default_factory=lambda: ["cup_with_handle", "high_tight_flag", "three_weeks_tight", "flat_base"]
    )

    # L — leader (RS rank)
    l_min_rs_percentile: float = 0.70

    # I — institutional
    i_require_qoq_nondecrease: bool = True
    i_min_new_positions: int = 1

    # Pre-filter (before any fundamentals calls)
    prefilter_min_price: float = 5.0
    prefilter_min_adv50_usd: float = 1_000_000.0
    prefilter_max_dist_to_52w_high: float = 0.25
    # Minimum market capitalization (USD). Applied as an early gate right after
    # the cheap per-candidate info fetch (shares/float/market-cap all come from
    # one yfinance fast_info call), BEFORE the expensive fundamentals/criteria
    # stage — so sub-floor names are rejected without burning SEC/FMP calls.
    # 0 disables the floor. Default = $1B (no micro/small caps).
    prefilter_min_market_cap_usd: float = 1_000_000_000.0


class CompositeWeights(BaseModel):
    c: float = 1.0
    a: float = 1.0
    n: float = 0.5
    s: float = 1.0
    l: float = 1.0
    i: float = 1.0
    m: float = 0.5


class ScannerConfig(BaseModel):
    default_universe: str = "us_all"
    universe_file: Optional[str] = None  # for custom universe
    out_dir: str = "out"
    max_workers: int = 16
    top_n_near_matches: int = 20  # rows to show in report's "Top by composite score" section
    # Fraction of SCANNED tickers allowed to abstain (gate couldn't evaluate due
    # to a transient data-quality hiccup — e.g. a yfinance "401 Invalid Crumb"
    # on the institutional/float fetch) before the run is treated as DEGRADED.
    # Below this fraction the abstains are BENIGN: the scan completed against the
    # vast majority of tickers, so it exits 0 and publishes. At/above it the run
    # is flagged degraded (manifest warning + non-zero exit) and `publish`
    # refuses without --allow-degraded. A full-market (us_all) run touches
    # thousands of tickers; a handful of transient abstains is normal operation,
    # not a quality failure — this floor stops them from aborting the whole run.
    max_abstain_fraction: float = 0.05
    # Fraction of cap-gated candidates allowed to land in `unknown_market_cap`
    # (the crumbed yfinance market-cap fetch FAILED — not "cap known, below the
    # floor") before the run is treated as DEGRADED. A high fraction means
    # cap-fetch throttling collapsed the scanned set far below the real
    # full-market scale, so the run should retry with a warmed cap cache rather
    # than publish a thin page. Distinct from rejected_market_cap (a legitimate
    # below-$1B exclusion), which never counts here.
    max_unknown_mcap_fraction: float = 0.25
    embed_charts_base64: bool = True  # embed chart PNGs as data-URIs so report.md is self-contained
    market_index: str = "SPY"  # M-gate benchmark. Use ^HSI for Hong Kong, ^GSPTSE for Canada, etc.
    # Auto-generate a PDF of the report alongside the markdown. Requires Chrome /
    # Chromium / Brave / Edge installed (auto-detected). Falls back gracefully
    # if no browser is found. Default OFF now that HTML is the primary surface —
    # use `canslim report-pdf` for opt-in PDF generation when you need print/archive.
    generate_pdf: bool = False


class Settings(BaseModel):
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    criteria: CriteriaThresholds = Field(default_factory=CriteriaThresholds)
    weights: CompositeWeights = Field(default_factory=CompositeWeights)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)

    @classmethod
    def load(cls, path: Optional[str | Path]) -> "Settings":
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        raw: dict[str, Any] = yaml.safe_load(p.read_text()) or {}
        return cls.model_validate(raw)

    def config_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"providers"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
