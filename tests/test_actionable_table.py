"""Tests for the Actionable master table assembly + rendering.

The Actionable table is the report's primary actionable view: it unions
Full Match + Buyable + Watchlist into one row-per-ticker list (Basing and the
failed near-match runners-up are excluded), and the rendered HTML carries the
per-row Status + Pattern data attributes the client-side filter keys on. These
tests pin (1) the union/exclusion contract of ``_actionable_rows`` and (2) that
each rendered row carries status + pattern so the chairman's
"Full Match or Buyable, Cup-with-Handle" filter narrows correctly.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone

from canslim.html_report import (
    PATTERN_DISPLAY,
    _actionable_rows,
    _bucket_candidates,
    render_html,
)
from canslim.models import (
    CriterionResult,
    MarketRegime,
    PatternMatch,
    RunManifest,
    ScanResult,
)

AS_OF = date(2026, 6, 24)
LETTERS = ["C", "A", "N", "S", "L", "I", "M"]


def _gates(all_pass: bool = True, fail: str | None = None) -> dict[str, CriterionResult]:
    out: dict[str, CriterionResult] = {}
    for L in LETTERS:
        passed = all_pass and not (fail and L == fail)
        out[L] = CriterionResult(
            letter=L, passed=passed, is_gate=(L not in ("N", "M")),
            score=1.0 if passed else 0.0,
            value=(0.85 if L == "L" else None),
        )
    # N carries the close used by entry_status note formatting.
    out["N"] = CriterionResult(
        letter="N", passed=True, is_gate=False, score=1.0, evidence={"close": 100.0},
    )
    return out


def _mk(
    ticker: str, *, passed: bool, score: float,
    pattern: str | None = None, dist: float | None = None,
    fail: str | None = None, status: str = "scanned",
) -> ScanResult:
    patterns = []
    if pattern is not None:
        patterns = [PatternMatch(
            name=pattern, detected=True, pivot=100.0, confidence=0.8,
            evidence={"dist_to_pivot_pct": dist},
        )]
    return ScanResult(
        ticker=ticker, as_of=AS_OF, passed=passed, composite_score=score,
        criteria=_gates(all_pass=passed, fail=fail), patterns=patterns,
        ad_grade="B", ad_ratio=0.6, market_cap=5e9, status=status,
    )


def _manifest() -> RunManifest:
    return RunManifest(
        run_id="2026-06-24_120000",
        started_at=datetime(2026, 6, 24, 12, tzinfo=timezone.utc),
        universe_name="us_all", universe_size=500, candidates_after_prefilter=200,
        matches=2, scanned=8, pending_budget=0, errored=0, config_hash="t",
        market_regime=MarketRegime(
            as_of=AS_OF, spy_close=500.0, spy_sma50=480.0, spy_sma200=450.0,
            uptrend=True, reason="ok",
        ),
    )


# --- fixture: full matches + buyable + watchlist + basing + a failed runner-up.
def _fixture() -> list[ScanResult]:
    return [
        _mk("CUPFULL", passed=True, score=0.97, pattern="cup_with_handle", dist=0.01),
        _mk("FLATFULL", passed=True, score=0.95, pattern="flat_base", dist=-0.02),
        _mk("CUPBUY", passed=False, score=0.84, fail="S", pattern="cup_with_handle", dist=0.03),
        _mk("FLATBUY", passed=False, score=0.82, fail="C", pattern="flat_base", dist=-0.04),
        _mk("CUPWATCH", passed=False, score=0.78, fail="S", pattern="cup_with_handle", dist=0.11),
        _mk("BASENONE", passed=False, score=0.70, fail="S", pattern=None, dist=None),
        _mk("FAILRUN", passed=False, score=0.40, fail="C", pattern=None, dist=None),
    ]


def _split(results: list[ScanResult]):
    matches = [r for r in results if r.passed]
    match_set = {r.ticker for r in matches}
    pool = sorted(
        [r for r in results if r.status == "scanned" and r.ticker not in match_set],
        key=lambda r: -r.composite_score,
    )
    buyable, watchlist, _basing = _bucket_candidates(pool)
    return matches, buyable, watchlist


def test_actionable_rows_union_and_exclusions():
    matches, buyable, watchlist = _split(_fixture())
    rows = _actionable_rows(matches, buyable, watchlist)
    tickers = {row["ticker"] for row in rows}

    # Union of the three actionable buckets only.
    assert tickers == {"CUPFULL", "FLATFULL", "CUPBUY", "FLATBUY", "CUPWATCH"}
    # Basing (no pivot yet) and the failed near-match runner-up are excluded.
    assert "BASENONE" not in tickers
    assert "FAILRUN" not in tickers


def test_actionable_rows_carry_status_and_pattern():
    matches, buyable, watchlist = _split(_fixture())
    rows = _actionable_rows(matches, buyable, watchlist)
    by_ticker = {row["ticker"]: row for row in rows}

    # Every row carries its source bucket as status + a pattern (name + label).
    for row in rows:
        assert row["status"] in ("full_match", "buyable", "watchlist")
        assert "pattern_name" in row and "pattern_label" in row

    assert by_ticker["CUPFULL"]["status"] == "full_match"
    assert by_ticker["CUPBUY"]["status"] == "buyable"
    assert by_ticker["CUPWATCH"]["status"] == "watchlist"
    assert by_ticker["CUPFULL"]["pattern_name"] == "cup_with_handle"
    assert by_ticker["CUPFULL"]["pattern_label"] == "Cup-with-Handle"
    assert by_ticker["FLATFULL"]["pattern_label"] == PATTERN_DISPLAY["flat_base"]


def test_actionable_rows_sorted_by_score_desc():
    matches, buyable, watchlist = _split(_fixture())
    rows = _actionable_rows(matches, buyable, watchlist)
    scores = [row["score"] for row in rows]
    assert scores == sorted(scores, reverse=True)


def test_rendered_table_rows_have_filter_attrs():
    results = _fixture()
    html = render_html(results, _manifest(), top_n_near_matches=20)

    # Master table + both filters present.
    assert 'id="actionable-table"' in html
    assert 'id="f-status"' in html and 'id="f-pattern"' in html
    # The combined option that expresses the chairman's "Full Match OR Buyable".
    assert 'value="full_match,buyable"' in html

    # Each actionable <tr> carries data-status + data-pattern (the JS filter keys).
    tbody = re.search(
        r'id="actionable-table".*?<tbody>(.*?)</tbody>', html, re.S
    ).group(1)
    trs = re.findall(r'<tr data-status="([^"]*)" data-pattern="([^"]*)">', tbody)
    assert len(trs) == 5  # union of full + buyable + watchlist

    # Simulate the client-side filter: Full Match|Buyable AND cup_with_handle.
    filtered = [
        (s, p) for (s, p) in trs
        if s in ("full_match", "buyable") and p == "cup_with_handle"
    ]
    assert len(filtered) == 2  # CUPFULL + CUPBUY

    # Pattern column is populated, not all em-dash.
    assert "Cup-with-Handle" in html and "Flat Base" in html


def test_pattern_filter_lists_only_present_patterns():
    results = _fixture()  # has cup_with_handle + flat_base only in actionable rows
    html = render_html(results, _manifest(), top_n_near_matches=20)
    opts_block = re.search(r'id="f-pattern">(.*?)</select>', html, re.S).group(1)
    opt_values = set(re.findall(r'<option value="([^"]+)">', opts_block))

    assert "cup_with_handle" in opt_values
    assert "flat_base" in opt_values
    # A pattern not present in the data must NOT be offered as a filter option.
    assert "saucer" not in opt_values


def test_old_near_matches_heading_replaced_and_demoted():
    results = _fixture()
    html = render_html(results, _manifest(), top_n_near_matches=20)
    # The confusing original heading is gone.
    assert "by composite score (near-matches)" not in html
    # Replaced by an honest, demoted diagnostic label.
    assert "Near misses" in html and "failed gates" in html
