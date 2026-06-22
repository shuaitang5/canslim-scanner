from __future__ import annotations

import asyncio
from datetime import date

from canslim.config import CriteriaThresholds, ProviderConfig, Settings
from canslim.models import EarningsBundle, MarketRegime, PriceFeatures
from canslim.providers.sec_provider import _latest_dei_shares
from canslim.scanner import Scanner


def test_latest_dei_shares_picks_most_recent_end():
    facts = {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"val": 1_000, "end": "2025-03-31"},
        {"val": 2_000, "end": "2026-03-31"},
        {"val": 1_500, "end": "2024-03-31"},
    ]}}}}}
    assert _latest_dei_shares(facts) == 2_000.0


def test_latest_dei_shares_absent_concept_returns_none():
    assert _latest_dei_shares({"facts": {"dei": {}}}) is None
    assert _latest_dei_shares({}) is None
    assert _latest_dei_shares({"facts": {"dei": {"EntityCommonStockSharesOutstanding":
                                                 {"units": {"shares": []}}}}}) is None


def _price_features(ticker: str = "TEST", close: float = 100.0) -> PriceFeatures:
    high = 105.0
    return PriceFeatures(
        ticker=ticker,
        as_of=date.today(),
        close=close,
        high_52w=high,
        low_52w=50.0,
        adv10=2_000_000.0,
        adv50=1_500_000.0,
        avg_vol50=50_000.0,
        recent_vol_ratio=1.5,
        rs_return_12m_weighted=0.30,
        dist_to_52w_high_pct=max(0.0, (high - close) / high),
    )


class _FakeYF:
    """Minimal stand-in for the yfinance provider so _evaluate_one runs offline.

    Each accessor is keyed by ticker; market_cap is the variable under test.
    """

    name = "yfinance"

    def __init__(self, market_cap: dict[str, float],
                 shares_outstanding: dict[str, float] | None = None) -> None:
        self._market_cap = market_cap
        # When set for a ticker, emulates the crumbless fast_info shares count
        # the scanner uses to RECONSTRUCT cap when the direct fetch returns None.
        self._shares_outstanding = shares_outstanding or {}

    async def get_market_cap(self, ticker: str):
        return self._market_cap.get(ticker)

    async def get_shares_outstanding(self, ticker: str):
        return self._shares_outstanding.get(ticker)

    async def get_shares_float(self, ticker: str):
        return 500_000_000.0  # passes the S float cap, irrelevant to this test

    async def get_fundamentals(self, ticker: str) -> EarningsBundle:
        # Strong, ACCELERATING, multi-year-growth earnings so a >=floor name is a
        # full candidate (and we can prove the gate did NOT drop a good large cap).
        # Latest YoY = 2.0/1.0-1 = 100% > prior YoY = 1.5/1.0-1 = 50% (accelerating).
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
        return None


def _scanner(min_cap: float, market_cap: dict[str, float],
             shares_outstanding: dict[str, float] | None = None) -> Scanner:
    settings = Settings(
        providers={
            "yfinance": ProviderConfig(enabled=True),
            "fmp": ProviderConfig(enabled=False),
            "sec": ProviderConfig(enabled=False),
        },
        criteria=CriteriaThresholds(prefilter_min_market_cap_usd=min_cap),
    )
    scanner = Scanner(settings)
    scanner.yf = _FakeYF(market_cap, shares_outstanding)  # type: ignore[assignment]
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


def test_sub_floor_market_cap_is_rejected():
    scanner = _scanner(min_cap=1_000_000_000.0, market_cap={"SMALL": 500_000_000.0})
    res = _evaluate(scanner, "SMALL")
    assert not res.passed, "a $0.5B name must not pass when the floor is $1B"
    # Dedicated status: the cap is KNOWN (not "missing data") and below the floor.
    assert res.status == "rejected_market_cap"
    assert "below" in (res.status_reason or "")
    assert res.market_cap == 500_000_000.0  # cap still surfaced on the result
    # Gate is EARLY: criteria were never evaluated, so no per-letter results.
    assert res.criteria == {}


def test_at_floor_market_cap_passes_gate():
    # Exactly at the floor must NOT be rejected (gate is strict less-than).
    scanner = _scanner(min_cap=1_000_000_000.0, market_cap={"EDGE": 1_000_000_000.0})
    res = _evaluate(scanner, "EDGE")
    assert res.status == "scanned", "a name exactly at the floor must clear the gate"
    assert res.criteria != {}, "criteria should have been evaluated past the gate"
    assert res.market_cap == 1_000_000_000.0


def test_above_floor_large_cap_is_scanned_as_full_candidate():
    scanner = _scanner(min_cap=1_000_000_000.0, market_cap={"BIG": 50_000_000_000.0})
    res = _evaluate(scanner, "BIG")
    assert res.status == "scanned"
    assert res.market_cap == 50_000_000_000.0
    # The cheap large cap with strong fundamentals should clear the C and A gates.
    assert res.criteria["C"].passed
    assert res.criteria["A"].passed


def test_unknown_market_cap_fails_closed():
    # The $1B floor is a HARD outward-facing requirement, so an unavailable cap
    # FAILS CLOSED: the name cannot be proven >= floor, so it is excluded from
    # matches and set aside in the unknown_market_cap "needs review" bucket
    # (not silently dropped, but never published as a match).
    scanner = _scanner(min_cap=1_000_000_000.0, market_cap={})  # no cap for ticker
    res = _evaluate(scanner, "NOCAP")
    assert not res.passed, "unknown cap must not pass a hard floor"
    assert res.status == "unknown_market_cap"
    assert res.market_cap is None
    assert res.criteria == {}  # gate is early; no criteria evaluated


class _FakeSEC:
    """SEC stand-in exposing only the crumbless shares-outstanding source."""

    name = "sec"

    def __init__(self, shares: dict[str, float]) -> None:
        self._shares = shares

    async def get_shares_outstanding(self, ticker: str):
        return self._shares.get(ticker)


def test_sec_shares_fallback_clears_gate_when_yfinance_throttled():
    # The decisive runner fix: yfinance cap AND yfinance shares both unavailable
    # (throttled), but SEC XBRL dei shares ARE available -> scanner reconstructs
    # cap = close * SEC_shares. close=100 * 30M = $3B clears the $1B floor.
    scanner = _scanner(min_cap=1_000_000_000.0, market_cap={}, shares_outstanding={})
    scanner.sec = _FakeSEC({"SECNAME": 30_000_000.0})  # type: ignore[assignment]
    res = _evaluate(scanner, "SECNAME")
    assert res.status == "scanned", "SEC shares fallback should clear the gate"
    assert res.market_cap == 100.0 * 30_000_000.0
    assert res.criteria != {}


def test_sec_shares_fallback_preferred_over_yfinance():
    # When BOTH sources have shares, SEC (authoritative) is used.
    scanner = _scanner(
        min_cap=1_000_000_000.0,
        market_cap={},
        shares_outstanding={"BOTH": 11_000_000.0},  # yfinance would give 1.1B
    )
    scanner.sec = _FakeSEC({"BOTH": 30_000_000.0})  # SEC gives 3.0B  # type: ignore[assignment]
    res = _evaluate(scanner, "BOTH")
    assert res.market_cap == 100.0 * 30_000_000.0, "SEC shares must win"


def test_computed_cap_fallback_clears_gate_when_direct_fetch_throttled():
    # Direct cap fetch returns None (throttled), but the crumbless shares count
    # is available -> scanner reconstructs cap = close * shares. With close=100
    # and 30M shares = $3B, the name clears the $1B floor and gets scanned
    # instead of fail-closing into unknown_market_cap. This is the fix that keeps
    # a throttled runner from collapsing the scanned universe.
    scanner = _scanner(
        min_cap=1_000_000_000.0,
        market_cap={},                                  # direct fetch throttled
        shares_outstanding={"THROTTLED": 30_000_000.0},  # 100 * 30M = $3B
    )
    res = _evaluate(scanner, "THROTTLED")
    assert res.status == "scanned", "computed-cap fallback should clear the gate"
    assert res.market_cap == 100.0 * 30_000_000.0
    assert res.criteria != {}


def test_computed_cap_fallback_still_rejects_sub_floor():
    # Reconstructed cap below the floor must still REJECT — the fallback can't be
    # a backdoor around the $1B floor. close=100 * 5M shares = $0.5B < $1B.
    scanner = _scanner(
        min_cap=1_000_000_000.0,
        market_cap={},
        shares_outstanding={"SMALLCO": 5_000_000.0},
    )
    res = _evaluate(scanner, "SMALLCO")
    assert res.status == "rejected_market_cap"
    assert res.market_cap == 100.0 * 5_000_000.0


def test_unknown_cap_with_no_shares_still_fails_closed():
    # Neither a direct cap NOR a shares count -> still fail-closed. The fallback
    # only fires when it can actually compute a cap; it never weakens the floor.
    scanner = _scanner(min_cap=1_000_000_000.0, market_cap={}, shares_outstanding={})
    res = _evaluate(scanner, "GHOST")
    assert res.status == "unknown_market_cap"
    assert res.market_cap is None


def test_floor_zero_disables_gate():
    scanner = _scanner(min_cap=0.0, market_cap={"TINY": 1_000_000.0})
    res = _evaluate(scanner, "TINY")
    assert res.status == "scanned", "a $0 floor disables the market-cap gate"
    assert res.market_cap == 1_000_000.0
