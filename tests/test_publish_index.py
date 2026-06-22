"""Unit tests for the publish landing-page index helpers in canslim.cli.

These cover the pure date-derivation helper that turns a committed meta.json
dict into index-row fields, plus the manifest/HTML meta builders that feed it.
The DATA DATE shown in the index is the data/market date (``as_of``), which is
deliberately distinct from the run-id timestamp.
"""

from canslim.cli import (
    _index_fields_from_meta,
    _meta_from_html,
    _meta_from_manifest,
)

DASH = "—"


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
