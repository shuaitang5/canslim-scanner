"""Regression guard for the $1B market-cap floor — fail-closed (Beck + fix).

The chairman's #1 hard requirement is "no stocks under $1B" on an OUTWARD-FACING
page. The gate in ``scanner._evaluate_one`` now FAILS CLOSED: a name clears it
only when its cap is KNOWN and at/above the floor. When the cap is UNAVAILABLE
(yfinance returned None) the name is EXCLUDED from matches and set aside in the
``unknown_market_cap`` "needs review" bucket — it can never publish as a match.

Originally this file documented the opposite (abstain = fail-OPEN) leak: a
genuinely sub-$1B company whose ``marketCap`` field was missing — common for the
thin/illiquid small caps the floor exists to exclude — slipped through as a full
MATCH with an em-dash where the cap should be. ``test_unknown_cap_subfloor_name``
below is the load-bearing one: it now asserts the fix (``not res.passed`` +
``status == "unknown_market_cap"``), so a regression to abstain would fail it.
"""

from __future__ import annotations

import asyncio
from datetime import date

from canslim.config import CriteriaThresholds, ProviderConfig, Settings
from canslim.models import (
    EarningsBundle,
    InstitutionalSnapshot,
    MarketRegime,
    PriceFeatures,
)
from canslim.scanner import Scanner


def _price_features(ticker: str, close: float = 100.0) -> PriceFeatures:
    high = 105.0
    return PriceFeatures(
        ticker=ticker,
        as_of=date.today(),
        close=close,
        high_52w=high,
        low_52w=50.0,
        adv10=2_000_000.0,   # ADV10/ADV50 = 1.33 >= 1.2 -> S demand passes
        adv50=1_500_000.0,
        avg_vol50=50_000.0,
        recent_vol_ratio=1.5,
        rs_return_12m_weighted=0.30,
        dist_to_52w_high_pct=max(0.0, (high - close) / high),
    )


class _AllGatesPassYF:
    """yfinance stand-in where every gate has the data it needs to PASS.

    Crucially adds an institutional snapshot (the upstream test fake returned
    None, which made the I gate abstain and kept names out of "matches" for an
    unrelated reason). With I satisfied, the ONLY thing that could keep a
    sub-floor name out of the matches bucket is the market-cap gate itself.
    """

    name = "yfinance"

    def __init__(self, market_cap: dict[str, float | None]) -> None:
        self._market_cap = market_cap

    async def get_market_cap(self, ticker: str):
        return self._market_cap.get(ticker)

    async def get_shares_float(self, ticker: str):
        return 500_000_000.0  # <= s_max_float_shares -> S supply passes

    async def get_fundamentals(self, ticker: str) -> EarningsBundle:
        # Accelerating YoY (100% latest vs 50% prior) + multi-year growth + ROE
        # -> C and A gates pass.
        return EarningsBundle(
            ticker=ticker,
            quarterly_eps=[2.0, 1.5, 1.2, 1.1, 1.0, 1.0],
            quarterly_periods=[
                "2026-03-31", "2025-12-31", "2025-09-30",
                "2025-06-30", "2025-03-31", "2024-12-31",
            ],
            annual_eps=[6.0, 4.5, 3.0, 2.0],
            annual_periods=["2025", "2024", "2023", "2022"],
            annual_roe_pct=[0.25, 0.22, 0.20, 0.18],
        )

    async def get_institutional(self, ticker: str):
        # Ownership present, no 13F delta data -> I gate passes (gate-on-presence).
        return InstitutionalSnapshot(
            ticker=ticker,
            reported_at=date.today(),
            inst_own_pct=0.65,
        )


def _scanner(min_cap: float, market_cap: dict[str, float | None]) -> Scanner:
    settings = Settings(
        providers={
            "yfinance": ProviderConfig(enabled=True),
            "fmp": ProviderConfig(enabled=False),
            "sec": ProviderConfig(enabled=False),
        },
        criteria=CriteriaThresholds(prefilter_min_market_cap_usd=min_cap),
    )
    scanner = Scanner(settings)
    scanner.yf = _AllGatesPassYF(market_cap)  # type: ignore[assignment]
    scanner.sec = None
    scanner.fmp = None
    return scanner


def _evaluate(scanner: Scanner, ticker: str):
    pf = _price_features(ticker)
    regime = MarketRegime(
        as_of=date.today(), spy_close=500.0, spy_sma50=480.0, spy_sma200=450.0,
        uptrend=True, reason="test uptrend",
    )
    return asyncio.run(
        scanner._evaluate_one(ticker, pf, rs_pct=0.95, regime=regime, as_of=date.today())
    )


def test_known_subfloor_name_still_rejected_even_when_all_gates_would_pass():
    """Sanity: when the cap IS known and < floor, the early gate fires before
    any criteria run — even though this name would otherwise be a full match."""
    scanner = _scanner(min_cap=1_000_000_000.0, market_cap={"SMALL": 500_000_000.0})
    res = _evaluate(scanner, "SMALL")
    assert not res.passed
    assert res.status == "rejected_market_cap"
    assert res.criteria == {}  # never evaluated -> truly blocked


def test_above_floor_name_with_all_gates_is_a_full_match():
    """Control: a >=$1B name with the same passing data IS a full match. This
    proves the test fixture genuinely produces matches, so a None-cap leak below
    isn't just 'nothing matches anyway'."""
    scanner = _scanner(min_cap=1_000_000_000.0, market_cap={"BIG": 50_000_000_000.0})
    res = _evaluate(scanner, "BIG")
    assert res.status == "scanned"
    assert res.passed, "control large-cap should pass every gate -> full match"


def test_unknown_cap_subfloor_name_is_failed_closed_not_a_match():
    """LOAD-BEARING: a company whose yfinance cap is None must be FAILED CLOSED.
    With every other gate passing it would, under the old abstain policy, have
    become a full MATCH (passed=True) and published with cap shown as '—' — a
    genuinely sub-$1B name slipping onto the chairman's public page.

    Fixed behavior: an unknown cap cannot clear the $1B floor. The name is
    excluded from matches (passed=False) and routed to the ``unknown_market_cap``
    "needs review" bucket — visible to the chairman, never published as a match.
    The gate fires EARLY, so no criteria run."""
    scanner = _scanner(min_cap=1_000_000_000.0, market_cap={"GHOST": None})
    res = _evaluate(scanner, "GHOST")
    assert not res.passed, (
        "FAIL-CLOSED: an unknown-cap name must NOT pass the $1B floor. It cannot "
        "be proven >= $1B, so it must never publish as a match on the public page."
    )
    assert res.status == "unknown_market_cap", "unknown cap is set aside for review, not scanned"
    assert res.market_cap is None
    assert res.criteria == {}, "gate fires early; no criteria evaluated"
