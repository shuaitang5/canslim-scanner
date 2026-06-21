"""Adversarial probe of the $1B market-cap floor (Beck).

The chairman's #1 hard requirement is "no stocks under $1B". The gate in
``scanner._evaluate_one`` only rejects when market cap is KNOWN and below the
floor; an UNKNOWN cap (yfinance returned None) abstains and the name is scanned
through every CANSLIM criterion as normal.

These tests pin down the consequence of that design choice: a genuinely sub-$1B
company whose ``marketCap`` field happens to be missing from yfinance (common
for thin/illiquid names — exactly the small caps the floor is meant to exclude)
is NOT blocked by the floor. If its fundamentals/technicals otherwise pass, it
surfaces as a full MATCH on the public report with an em-dash where the cap
should be.

``test_unknown_cap_subfloor_name_leaks_into_matches`` is the load-bearing one:
it is expected to PASS today, which is exactly why it documents a requirement
violation. If the gate is later changed to EXCLUDE unknown-cap names (treat a
hard floor as fail-closed), flip the assertion.
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
    assert res.status == "skipped_missing_data"
    assert res.criteria == {}  # never evaluated -> truly blocked


def test_above_floor_name_with_all_gates_is_a_full_match():
    """Control: a >=$1B name with the same passing data IS a full match. This
    proves the test fixture genuinely produces matches, so a None-cap leak below
    isn't just 'nothing matches anyway'."""
    scanner = _scanner(min_cap=1_000_000_000.0, market_cap={"BIG": 50_000_000_000.0})
    res = _evaluate(scanner, "BIG")
    assert res.status == "scanned"
    assert res.passed, "control large-cap should pass every gate -> full match"


def test_unknown_cap_subfloor_name_leaks_into_matches():
    """LOAD-BEARING: a company that is REALLY sub-$1B but whose yfinance cap is
    None is NOT blocked by the floor. With all other gates passing it becomes a
    full MATCH (passed=True) and would be published with cap shown as '—'.

    This is the requirement violation: the chairman said "no stocks under $1B",
    but the abstain-on-unknown policy lets a sub-$1B name through whenever the
    cap field is missing. Documented here; flip to ``assert not res.passed`` if
    the gate is changed to fail-closed (exclude unknown cap)."""
    scanner = _scanner(min_cap=1_000_000_000.0, market_cap={"GHOST": None})
    res = _evaluate(scanner, "GHOST")
    assert res.status == "scanned", "unknown cap abstains rather than rejecting"
    assert res.market_cap is None
    # The leak: every gate passed, so the name is a full match despite being
    # (in reality) a sub-floor company with no cap data to prove otherwise.
    assert res.passed, (
        "CURRENT BEHAVIOR: unknown-cap name passes all gates and becomes a "
        "MATCH. A genuinely sub-$1B name with a missing cap field would be "
        "published on the chairman's public page, violating the $1B floor."
    )
