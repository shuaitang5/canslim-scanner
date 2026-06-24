"""Unit tests for the ticker-history inverted index (Part C of the merge).

build_history reads every committed docs/runs/<id>/summary.json and produces an
inverted index keyed by ticker, newest-first, covering ALL named buckets — and
ONLY tickers that appeared in a named bucket (never the full scanned set).
"""

from __future__ import annotations

import json

from canslim.diffboard.build_history import build_history


def _write_summary(docs_dir, run_id, as_of, regime, tickers):
    run_dir = docs_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "as_of": as_of,
                "universe": "us_all",
                "regime": regime,
                "tickers": tickers,
            }
        )
    )


def _entry(ticker, bucket, score, gates="CANSLIM", ad="A", pivot=None, dist=None):
    return {
        "ticker": ticker, "bucket": bucket, "score": score, "gates": gates,
        "ad": ad, "pivot": pivot, "dist": dist, "as_of": "x",
    }


def test_history_indexes_all_buckets_newest_first(tmp_path):
    _write_summary(
        tmp_path, "2026-06-10_120000", "2026-06-10", "UPTREND",
        [_entry("ATI", "basing", 0.90), _entry("AAA", "buyable", 0.80)],
    )
    _write_summary(
        tmp_path, "2026-06-12_120000", "2026-06-12", "UPTREND",
        [_entry("ATI", "full_match", 0.97), _entry("BBB", "watchlist", 0.70)],
    )

    h = build_history(tmp_path)

    # dates are descending
    assert h["dates"] == ["2026-06-12", "2026-06-10"]
    # ATI appears in both, newest-first, with its bucket per date
    ati = h["tickers"]["ATI"]
    assert [r["date"] for r in ati] == ["2026-06-12", "2026-06-10"]
    assert ati[0]["bucket"] == "full_match"
    assert ati[0]["score"] == 0.97
    assert ati[0]["run_id"] == "2026-06-12_120000"
    assert ati[1]["bucket"] == "basing"
    # every named bucket is indexed, not just full matches
    assert set(h["tickers"]) == {"ATI", "AAA", "BBB"}
    assert h["tickers"]["AAA"][0]["bucket"] == "buyable"
    assert h["tickers"]["BBB"][0]["bucket"] == "watchlist"


def test_history_one_report_per_date_keeps_latest_run_id(tmp_path):
    # Two runs share a data date; the later run-id wins (mirrors the publish prune).
    _write_summary(
        tmp_path, "2026-06-12_080000", "2026-06-12", "UPTREND",
        [_entry("OLD", "full_match", 0.5)],
    )
    _write_summary(
        tmp_path, "2026-06-12_230000", "2026-06-12", "UPTREND",
        [_entry("NEW", "full_match", 0.9)],
    )

    h = build_history(tmp_path)

    assert h["dates"] == ["2026-06-12"]
    # Only the latest run-id's tickers are indexed for that date.
    assert "NEW" in h["tickers"]
    assert "OLD" not in h["tickers"]
    assert h["tickers"]["NEW"][0]["run_id"] == "2026-06-12_230000"


def test_history_carries_pivot_and_dist(tmp_path):
    _write_summary(
        tmp_path, "2026-06-12_120000", "2026-06-12", "UPTREND",
        [_entry("CCC", "buyable", 0.81, pivot=100.0, dist=-0.03)],
    )
    h = build_history(tmp_path)
    row = h["tickers"]["CCC"][0]
    assert row["pivot"] == 100.0
    assert row["dist"] == -0.03
    assert set(row) == {"date", "run_id", "bucket", "score", "gates", "ad", "pivot", "dist"}


def test_history_empty_when_no_runs(tmp_path):
    h = build_history(tmp_path)
    assert h["dates"] == []
    assert h["tickers"] == {}
    assert "generated" in h
