"""Regression guards over the COMMITTED public artifacts (PR #10 invariants).

The unit tests in test_summary_json / test_history_json exercise the builders on
synthetic fixtures. These tests instead assert that the artifacts already
committed under ``docs/`` are internally consistent — the load-bearing promise
of the dashboard merge:

  1. Every ticker in each run's ``summary.json`` lands in the SAME bucket the
     committed ``index.html`` renders for it (no backfill / regeneration drift).
  2. ``docs/history.json`` indexes ONLY named-bucket tickers — exactly the union
     of named tickers across every summary.json, never the ~1500 scanned set.

If a future re-backfill or report-layout change silently desyncs the JSON feed
from the published page, these fail. They are skipped (not errored) if the docs
tree is absent, so a sparse checkout still runs the rest of the suite.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "docs" / "runs"
HISTORY_JSON = REPO / "docs" / "history.json"

NAMED_BUCKETS = {"full_match", "buyable", "watchlist", "basing"}

# Each TOC entry in the published report is
#   <li><a href="#c-TICKER" ... data-ticker="..." ...>
#     <span class="status-dot dot-{match|buyable|watchlist|basing}" ...>
# i.e. the rendered bucket per ticker. dot-match == full_match.
_TOC_RE = re.compile(
    r'href="#c-([A-Za-z0-9.\-]+)"[^>]*data-ticker="[^"]*"[^>]*>\s*'
    r'<span class="status-dot (dot-(?:match|buyable|watchlist|basing))"'
)
_DOT_TO_BUCKET = {
    "dot-match": "full_match",
    "dot-buyable": "buyable",
    "dot-watchlist": "watchlist",
    "dot-basing": "basing",
}


def _committed_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        d for d in RUNS_DIR.iterdir()
        if (d / "summary.json").exists() and (d / "index.html").exists()
    )


def _html_bucket_map(index_html: str) -> dict[str, str]:
    return {
        t: _DOT_TO_BUCKET[dot]
        for t, dot in _TOC_RE.findall(index_html)
    }


@pytest.mark.skipif(not _committed_runs(), reason="no committed docs/runs/*")
@pytest.mark.parametrize("run_dir", _committed_runs(), ids=lambda d: d.name)
def test_committed_summary_buckets_match_published_html(run_dir):
    """summary.json bucket assignment == the published report's rendered buckets."""
    summary = json.loads((run_dir / "summary.json").read_text())
    html = (run_dir / "index.html").read_text()

    summ_map = {e["ticker"]: e["bucket"] for e in summary["tickers"]}
    html_map = _html_bucket_map(html)

    # Same ticker set...
    assert set(summ_map) == set(html_map), (
        f"{run_dir.name}: ticker set drift — "
        f"only_summary={sorted(set(summ_map) - set(html_map))} "
        f"only_html={sorted(set(html_map) - set(summ_map))}"
    )
    # ...and same bucket for each ticker.
    drift = {
        t: (html_map[t], summ_map[t])
        for t in summ_map
        if summ_map[t] != html_map[t]
    }
    assert not drift, f"{run_dir.name}: bucket drift (html, summary) = {drift}"

    # Only named buckets are ever emitted (never the raw scanned set).
    assert set(summ_map.values()) <= NAMED_BUCKETS


@pytest.mark.skipif(not HISTORY_JSON.exists(), reason="no committed docs/history.json")
def test_committed_history_scope_is_named_buckets_only():
    """history.json == exactly the union of named-bucket tickers across summaries."""
    history = json.loads(HISTORY_JSON.read_text())

    # Every appearance carries a valid named bucket.
    for sym, apps in history["tickers"].items():
        for a in apps:
            assert a["bucket"] in NAMED_BUCKETS, (
                f"{sym} @ {a.get('date')}: bucket {a['bucket']!r} not a named bucket"
            )

    # The indexed ticker set equals the union of named tickers across all runs.
    union: set[str] = set()
    for run_dir in _committed_runs():
        summary = json.loads((run_dir / "summary.json").read_text())
        union.update(e["ticker"] for e in summary["tickers"])

    indexed = set(history["tickers"])
    assert indexed == union, (
        "history scope drift — "
        f"in_history_not_summaries={sorted(indexed - union)[:20]} "
        f"in_summaries_not_history={sorted(union - indexed)[:20]}"
    )


@pytest.mark.skipif(not HISTORY_JSON.exists(), reason="no committed docs/history.json")
def test_committed_history_appearances_are_newest_first():
    """Each ticker's appearance list is ordered newest-date-first."""
    history = json.loads(HISTORY_JSON.read_text())
    for sym, apps in history["tickers"].items():
        dates = [a["date"] for a in apps]
        assert dates == sorted(dates, reverse=True), f"{sym}: appearances not newest-first"
