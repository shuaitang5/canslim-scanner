"""Unit tests for the publish landing-page index helpers in canslim.cli.

These cover the pure date-derivation helper that turns a committed meta.json
dict into index-row fields, plus the manifest/HTML meta builders that feed it.
The DATA DATE shown in the index is the data/market date (``as_of``), which is
deliberately distinct from the run-id timestamp.
"""

import json

from canslim.cli import (
    _index_fields_from_meta,
    _meta_from_html,
    _meta_from_manifest,
    _plan_superseded_runs,
    _prune_superseded_runs,
    _run_as_of,
)

DASH = "—"


def _make_run(archive_dir, run_id, as_of):
    """Create a docs/runs/<run_id>/ dir with index.html and (maybe) meta.json.

    ``as_of=None`` writes NO meta.json (simulating an un-backfilled / broken run
    whose data date can't be determined).
    """
    d = archive_dir / run_id
    d.mkdir(parents=True)
    (d / "index.html").write_text(f"<html>{run_id}</html>")
    if as_of is not None:
        (d / "meta.json").write_text(
            json.dumps({"as_of": as_of, "universe": "us_all", "run_id": run_id})
        )
    return d


def test_index_fields_meta_present():
    meta = {
        "as_of": "2026-06-12",
        "universe": "us_all",
        "matches": 17,
        "scanned": 2011,
        "run_id": "2026-06-12_020555",
    }
    assert _index_fields_from_meta(meta) == {
        "as_of": "2026-06-12",
        "universe": "us_all",
        "matches": 17,
        "scanned": 2011,
    }


def test_index_fields_meta_absent():
    # No committed meta.json at all -> every field falls back to the em dash.
    assert _index_fields_from_meta(None) == {
        "as_of": DASH,
        "universe": DASH,
        "matches": DASH,
        "scanned": DASH,
    }
    assert _index_fields_from_meta({}) == {
        "as_of": DASH,
        "universe": DASH,
        "matches": DASH,
        "scanned": DASH,
    }


def test_index_fields_null_as_of_only_dashes_that_field():
    # A null as_of dashes ONLY the date column; other present fields survive.
    meta = {"as_of": None, "universe": "sp500", "matches": 3, "scanned": 100}
    fields = _index_fields_from_meta(meta)
    assert fields["as_of"] == DASH
    assert fields["universe"] == "sp500"
    assert fields["matches"] == 3
    assert fields["scanned"] == 100


def test_index_fields_zero_matches_is_not_dashed():
    # 0 matches is a real value, not "missing" — must not collapse to a dash.
    meta = {"as_of": "2026-06-18", "universe": "sp500", "matches": 0, "scanned": 82}
    assert _index_fields_from_meta(meta)["matches"] == 0


def test_meta_from_manifest_uses_regime_as_of_not_run_id():
    # The DATA DATE comes from market_regime.as_of (the prior trading day here),
    # never from the run-id timestamp.
    manifest = {
        "market_regime": {"as_of": "2026-06-18"},
        "universe_name": "sp500",
        "matches": 0,
        "scanned": 82,
        "started_at": "2026-06-21T15:48:10Z",
    }
    meta = _meta_from_manifest(manifest, "2026-06-21_154810")
    assert meta == {
        "as_of": "2026-06-18",
        "universe": "sp500",
        "matches": 0,
        "scanned": 82,
        "run_id": "2026-06-21_154810",
    }


def test_meta_from_manifest_missing_regime_yields_null_as_of():
    meta = _meta_from_manifest({"universe_name": "us_all"}, "2026-01-01_000000")
    assert meta["as_of"] is None
    assert meta["universe"] == "us_all"


def test_meta_from_html_picks_max_date_and_header_stats():
    html = (
        '<p>universe: us_all (4844 tickers)</p>'
        '<div class="stat">candidates after pre-filter <span class="v">2200</span></div>'
        '<div class="stat">scanned <span class="v">2,011</span></div>'
        '<div class="stat">full matches <span class="v">17</span></div>'
        '<svg><text>2026-06-01</text><text>2026-06-12</text></svg>'
        '<footer>generated 2026-06-12</footer>'
    )
    meta = _meta_from_html(html, "2026-06-12_020555")
    assert meta["as_of"] == "2026-06-12"  # MAX embedded date
    assert meta["universe"] == "us_all"
    assert meta["scanned"] == 2011  # comma stripped
    assert meta["matches"] == 17
    assert meta["run_id"] == "2026-06-12_020555"


def test_meta_from_html_no_dates_yields_null_as_of():
    meta = _meta_from_html("<html>no dates here</html>", "x")
    assert meta["as_of"] is None
    assert meta["universe"] is None
    assert meta["matches"] is None
    assert meta["scanned"] is None


# --- one-report-per-date prune --------------------------------------------


def test_run_as_of_reads_meta(tmp_path):
    d = _make_run(tmp_path, "2026-06-22_095053", "2026-06-18")
    assert _run_as_of(d) == "2026-06-18"


def test_run_as_of_missing_meta_is_none(tmp_path):
    d = _make_run(tmp_path, "2026-06-22_095053", None)
    assert _run_as_of(d) is None


def test_prune_keeps_latest_run_id_per_as_of(tmp_path):
    # Two data dates, multiple run-ids each (mirrors the real 6/22 pile where
    # one run-id date spans two as_of dates). Keep the LATEST run-id per as_of.
    _make_run(tmp_path, "2026-06-22_090852", "2026-06-18")
    _make_run(tmp_path, "2026-06-22_093033", "2026-06-18")
    _make_run(tmp_path, "2026-06-22_095053", "2026-06-18")  # newest for 06-18
    _make_run(tmp_path, "2026-06-22_171607", "2026-06-22")
    _make_run(tmp_path, "2026-06-22_180056", "2026-06-22")
    _make_run(tmp_path, "2026-06-22_231350", "2026-06-22")  # newest for 06-22

    deleted = _prune_superseded_runs(tmp_path)

    assert deleted == {
        "2026-06-18": ["2026-06-22_090852", "2026-06-22_093033"],
        "2026-06-22": ["2026-06-22_171607", "2026-06-22_180056"],
    }
    survivors = sorted(p.name for p in tmp_path.iterdir())
    assert survivors == ["2026-06-22_095053", "2026-06-22_231350"]


def test_prune_leaves_singletons_and_distinct_dates_untouched(tmp_path):
    # Distinct data dates are NOT dupes even if other dates pile up.
    _make_run(tmp_path, "2026-05-21_000903", "2026-05-21")
    _make_run(tmp_path, "2026-05-23_010249", "2026-05-23")
    # A genuine same-date dupe alongside the singletons.
    _make_run(tmp_path, "2026-05-27_061445", "2026-05-27")
    _make_run(tmp_path, "2026-05-27_232939", "2026-05-27")  # newest for 05-27

    deleted = _prune_superseded_runs(tmp_path)

    assert deleted == {"2026-05-27": ["2026-05-27_061445"]}
    survivors = sorted(p.name for p in tmp_path.iterdir())
    assert survivors == [
        "2026-05-21_000903",
        "2026-05-23_010249",
        "2026-05-27_232939",
    ]


def test_prune_never_touches_runs_with_unknown_as_of(tmp_path):
    # A run with no meta.json (as_of unknown) must never be grouped or deleted,
    # even if another run shares would-be characteristics.
    _make_run(tmp_path, "2026-06-10_010101", None)  # unknown as_of
    _make_run(tmp_path, "2026-06-10_020202", None)  # unknown as_of
    _make_run(tmp_path, "2026-06-11_010101", "2026-06-11")
    _make_run(tmp_path, "2026-06-11_020202", "2026-06-11")  # newest for 06-11

    deleted = _prune_superseded_runs(tmp_path)

    # Only the genuine same-as_of dupe is pruned; both no-meta runs survive.
    assert deleted == {"2026-06-11": ["2026-06-11_010101"]}
    survivors = sorted(p.name for p in tmp_path.iterdir())
    assert survivors == [
        "2026-06-10_010101",
        "2026-06-10_020202",
        "2026-06-11_020202",
    ]


def test_plan_is_non_destructive(tmp_path):
    # _plan_superseded_runs only plans — nothing is deleted by calling it.
    _make_run(tmp_path, "2026-06-22_090852", "2026-06-18")
    _make_run(tmp_path, "2026-06-22_095053", "2026-06-18")
    plan = _plan_superseded_runs(tmp_path)
    assert {k: [p.name for p in v] for k, v in plan.items()} == {
        "2026-06-18": ["2026-06-22_090852"]
    }
    # Both dirs still on disk.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "2026-06-22_090852",
        "2026-06-22_095053",
    ]
