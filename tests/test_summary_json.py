"""Unit tests for the per-run summary.json (Part A of the dashboard merge).

summary.json is the structured, scrape-free feed the diff dashboard + ticker
history search consume. Two production paths build it:

  * canslim.report.build_run_summary  — from in-memory scan results (every run
    going forward). Bucket assignment must match the published HTML report.
  * canslim.cli._summary_from_html     — one-time bootstrap that recovers the
    named-bucket tickers from already-archived index.html.

Both must agree on: WHICH tickers land in summary.json (only the named buckets:
full_match / buyable / watchlist / basing — never the full scanned set) and the
bucket each lands in.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

from canslim.cli import _summary_from_html
from canslim.models import (
    CriterionResult,
    MarketRegime,
    PatternMatch,
    RunManifest,
    ScanResult,
)
from canslim.report import NAMED_BUCKETS, build_run_summary

AS_OF = date(2026, 6, 22)


def _gates_all_pass() -> dict[str, CriterionResult]:
    return {
        L: CriterionResult(letter=L, passed=True, is_gate=(L not in ("N", "M")), score=1.0)
        for L in ["C", "A", "N", "S", "L", "I", "M"]
    }


def _gates_fail(letter: str) -> dict[str, CriterionResult]:
    crit = _gates_all_pass()
    crit[letter] = CriterionResult(letter=letter, passed=False, is_gate=True, score=0.0)
    return crit


def _result(
    ticker: str,
    *,
    passed: bool,
    score: float,
    fail_letter: str | None = None,
    pattern_dist: float | None = None,
    ad_grade: str | None = "B",
    market_cap: float | None = 5e9,
) -> ScanResult:
    crit = _gates_all_pass() if passed else _gates_fail(fail_letter or "S")
    patterns = []
    if pattern_dist is not None:
        patterns = [
            PatternMatch(
                name="cup_with_handle",
                detected=True,
                pivot=100.0,
                confidence=0.8,
                evidence={"dist_to_pivot_pct": pattern_dist},
            )
        ]
    return ScanResult(
        ticker=ticker,
        as_of=AS_OF,
        passed=passed,
        composite_score=score,
        criteria=crit,
        patterns=patterns,
        ad_grade=ad_grade,
        market_cap=market_cap,
        status="scanned",
    )


def _manifest() -> RunManifest:
    return RunManifest(
        run_id="2026-06-22_231350",
        started_at=datetime(2026, 6, 22, 23, 13, 50, tzinfo=timezone.utc),
        universe_name="us_all",
        universe_size=1500,
        candidates_after_prefilter=300,
        matches=1,
        scanned=4,
        pending_budget=0,
        errored=0,
        config_hash="abc",
        market_regime=MarketRegime(
            as_of=AS_OF, spy_close=500.0, spy_sma50=480.0, spy_sma200=450.0,
            uptrend=True, reason="ok",
        ),
    )


def test_build_run_summary_assigns_correct_buckets():
    results = [
        # Full match: all gates pass.
        _result("FULL", passed=True, score=0.97),
        # Buyable: near-miss with a pattern within ±5% of pivot.
        _result("BUY", passed=False, score=0.80, pattern_dist=0.02),
        # Watchlist: near-miss with a pattern but extended past +5%.
        _result("WATCH", passed=False, score=0.70, pattern_dist=0.12),
        # Basing: near-miss, no pattern at all.
        _result("BASE", passed=False, score=0.60, pattern_dist=None),
        # A non-scanned reject must NEVER appear in summary.json.
        ScanResult(
            ticker="REJECT", as_of=AS_OF, passed=False, composite_score=0.0,
            criteria={}, status="rejected_market_cap",
        ),
    ]
    summary = build_run_summary(results, _manifest(), top_n_near_matches=20)

    by_ticker = {t["ticker"]: t for t in summary["tickers"]}
    assert by_ticker["FULL"]["bucket"] == "full_match"
    assert by_ticker["BUY"]["bucket"] == "buyable"
    assert by_ticker["WATCH"]["bucket"] == "watchlist"
    assert by_ticker["BASE"]["bucket"] == "basing"
    # Rejected (non-scanned) ticker is excluded entirely.
    assert "REJECT" not in by_ticker
    # Only the four named-bucket tickers are present — not the full scanned set.
    assert len(summary["tickers"]) == 4
    assert {t["bucket"] for t in summary["tickers"]} <= set(NAMED_BUCKETS)


def test_build_run_summary_entry_shape_and_metadata():
    results = [_result("BUY", passed=False, score=0.81, pattern_dist=-0.03, ad_grade="A")]
    summary = build_run_summary(results, _manifest(), top_n_near_matches=20)

    assert summary["run_id"] == "2026-06-22_231350"
    assert summary["as_of"] == "2026-06-22"
    assert summary["universe"] == "us_all"

    e = summary["tickers"][0]
    assert set(e) == {
        "ticker", "bucket", "score", "gates", "ad", "pivot", "dist",
        "market_cap", "as_of",
    }
    assert e["bucket"] == "buyable"
    assert e["score"] == 0.81
    assert e["ad"] == "A"
    assert e["pivot"] == 100.0
    assert abs(e["dist"] - (-0.03)) < 1e-9
    assert e["market_cap"] == 5e9
    # gates flag string (C A N S L I M): all pass except S (forced near-miss
    # fail) which renders lowercase.
    assert e["gates"] == "CANsLIM"
    assert e["as_of"] == "2026-06-22"


def test_build_run_summary_is_json_serializable():
    results = [_result("FULL", passed=True, score=0.9)]
    summary = build_run_summary(results, _manifest(), top_n_near_matches=20)
    # Round-trips cleanly (this is what write_run persists to disk).
    assert json.loads(json.dumps(summary, default=str))["tickers"][0]["ticker"] == "FULL"


def test_summary_from_html_backfill_bucket_assignment():
    # Minimal stand-in for the committed report markup: two bucket sections, each
    # with one candidate <details>. The bootstrap parser must map the section to
    # the canonical bucket and recover score/gates/AD/dist.
    html = """
    <section class="bucket bucket-matches" data-bucket="matches">
      <h2>Full matches</h2>
      <details class="candidate" id="c-FULL" data-ticker="FULL" data-passed="true">
        <summary>
          <span class="ticker">FULL</span>
          <span class="score mono">0.97</span>
          <span class="gates">CANSLIM</span>
          <span class="summary-meta">fails: —</span>
          <span class="summary-meta summary-meta-cap">$27.7B · AD: A</span>
        </summary>
      </details>
    </section>
    <section class="bucket bucket-buyable" data-bucket="buyable">
      <h2>Buyable now</h2>
      <details class="candidate" id="c-BUY" data-ticker="BUY" data-passed="false">
        <summary>
          <span class="ticker">BUY</span>
          <span class="score mono">0.84</span>
          <span class="gates">CANSLsIM</span>
          <span class="summary-meta">In buy zone · dist +2.3% from pivot $34.49</span>
          <span class="summary-meta summary-meta-cap">$3.1B · AD: B</span>
        </summary>
      </details>
    </section>
    <section class="bucket bucket-overrides hidden" data-bucket="overrides">
      <h2>Should be ignored — not a named bucket</h2>
    </section>
    """
    summary = _summary_from_html(html, "2026-06-22_231350", "2026-06-22", "us_all")
    by_ticker = {t["ticker"]: t for t in summary["tickers"]}

    assert by_ticker["FULL"]["bucket"] == "full_match"
    assert by_ticker["FULL"]["score"] == 0.97
    assert by_ticker["FULL"]["gates"] == "CANSLIM"
    assert by_ticker["FULL"]["ad"] == "A"

    assert by_ticker["BUY"]["bucket"] == "buyable"
    assert by_ticker["BUY"]["score"] == 0.84
    assert by_ticker["BUY"]["pivot"] == 34.49
    assert abs(by_ticker["BUY"]["dist"] - 0.023) < 1e-9
    assert by_ticker["BUY"]["ad"] == "B"

    # Only the two named-bucket candidates; the overrides section is excluded.
    assert len(summary["tickers"]) == 2
    assert summary["run_id"] == "2026-06-22_231350"
    assert summary["as_of"] == "2026-06-22"
