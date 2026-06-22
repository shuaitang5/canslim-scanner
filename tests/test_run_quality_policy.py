"""Exit-code / publish policy for `canslim scan` (the us_all-blocker fix).

`_assess_run_quality` decides whether a completed scan is BENIGN (exit 0,
publishes), DEGRADED (exit 2, manifest poisoned so `publish` refuses), or FATAL
(exit 2, never publishes). The whole point: a full-market run that touches
thousands of tickers and hits a handful of transient yfinance abstains must
SUCCEED and PUBLISH — it must NOT abort the way every prior us_all runner run did.

The mapping the caller relies on:
  - health_warn non-empty  -> manifest gets `_data_quality_warnings` -> publish blocked + exit 2
  - health_warn empty       -> clean manifest -> publish allowed + exit 0
  - fatal                   -> exit 2 regardless (and health_warn is also set)
"""

import json

from typer.testing import CliRunner

from canslim.cli import _assess_run_quality, _count_abstains, app
from canslim.models import CriterionResult, ScanResult
from datetime import date as _date

runner = CliRunner()


def _result(letters_abstaining, status="scanned"):
    """Build a ScanResult whose listed gate letters abstained (data_available=False)."""
    crit = {}
    for L in ["C", "A", "S", "L", "I"]:
        crit[L] = CriterionResult(
            letter=L, passed=True, is_gate=True, score=1.0,
            data_available=(L not in letters_abstaining),
        )
    return ScanResult(
        ticker="T", as_of=_date.today(), passed=False, composite_score=0.0,
        criteria=crit, status=status,
    )


def test_count_abstains_excludes_institutional_from_gated():
    # I-only abstain -> NOT counted in gated (it's a free-stack reality), but IS
    # surfaced in the institutional count.
    results = [_result({"I"}) for _ in range(10)]
    gated, inst = _count_abstains(results)
    assert gated == 0
    assert inst == 10


def test_count_abstains_counts_fundamental_gates():
    # A C/A/S/L abstain DOES count toward the gated (degraded) figure.
    results = [_result({"C"}), _result({"S"}), _result({"I"})]
    gated, inst = _count_abstains(results)
    assert gated == 2          # C and S abstains
    assert inst == 1           # the I-abstain ticker


def test_count_abstains_ignores_non_scanned():
    results = [_result({"C"}, status="unknown_market_cap"), _result({"A"}, status="scanned")]
    gated, inst = _count_abstains(results)
    assert gated == 1          # only the scanned one

_MINIMAL_HTML = "<html><body>universe: us_all (4800 tickers)</body></html>"


def _make_run(tmp_path, run_id="2026-06-21_000000", warnings=None):
    """Create an out/runs/<id>/ with index.html + run_manifest.json for publish."""
    run_dir = tmp_path / "out" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "index.html").write_text(_MINIMAL_HTML)
    manifest = {
        "market_regime": {"as_of": "2026-06-20"},
        "universe_name": "us_all",
        "matches": 9,
        "scanned": 515,
    }
    if warnings:
        manifest["_data_quality_warnings"] = warnings
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    return run_dir

# Default full-market tolerance (mirrors ScannerConfig.max_abstain_fraction).
THRESH = 0.05


def _assess(**kw):
    base = dict(
        scanned=2000,
        universe_size=4800,
        abstained_scans=0,
        abstained_pct=0.0,
        fresh_price_failures=0,
        fresh_price_attempts=2000,
        max_abstain_fraction=THRESH,
        unknown_mcap=0,
        rejected_mcap=0,
        max_unknown_mcap_fraction=0.25,
    )
    base.update(kw)
    return _assess_run_quality(**base)


def test_clean_run_is_benign_and_publishes():
    v = _assess()
    assert v["fatal"] is False
    assert v["health_warn"] == []          # nothing blocks publish
    assert v["info_notes"] == []
    assert v["summary_color"] == "green"


def test_small_abstain_fraction_is_benign_publishes():
    # 14 of 515 abstained (2.7%) — exactly the real failed-runner case. BENIGN:
    # must NOT poison the manifest and must NOT exit non-zero.
    v = _assess(scanned=515, abstained_scans=14, abstained_pct=14 / 515)
    assert v["fatal"] is False
    assert v["health_warn"] == []          # publish allowed
    assert len(v["info_notes"]) == 1       # reported, but only as FYI
    assert "benign" in v["info_notes"][0].lower()


def test_abstain_just_below_threshold_is_benign():
    v = _assess(scanned=1000, abstained_scans=49, abstained_pct=0.049)
    assert v["health_warn"] == []
    assert v["fatal"] is False


def test_abstain_at_threshold_is_degraded():
    # >= threshold flips to DEGRADED: health_warn set -> publish blocked + exit 2.
    v = _assess(scanned=1000, abstained_scans=50, abstained_pct=0.05)
    assert v["fatal"] is False
    assert len(v["health_warn"]) == 1
    assert v["info_notes"] == []
    assert v["summary_color"] == "yellow"


def test_abstain_above_threshold_is_degraded():
    v = _assess(scanned=1000, abstained_scans=200, abstained_pct=0.20)
    assert v["health_warn"]                 # blocks publish
    assert v["fatal"] is False


def test_zero_scanned_is_fatal_blocks_publish():
    # A scan that evaluated nothing is FATAL — provider chain down. Must exit 2
    # AND block publishing an empty page (health_warn set so manifest is poisoned).
    v = _assess(scanned=0, universe_size=4800, abstained_scans=0, abstained_pct=0.0)
    assert v["fatal"] is True
    assert v["health_warn"]                 # empty-page guard still fires
    assert v["summary_color"] == "red"


def test_price_throttling_is_degraded_blocks_publish():
    # >20% of a meaningful number of fresh fetches failing = yfinance throttling.
    # Degraded: must block publish so a throttled/garbage page never goes live.
    v = _assess(
        scanned=2000,
        fresh_price_failures=600,
        fresh_price_attempts=2000,
    )
    assert v["fatal"] is False
    assert v["health_warn"]
    assert "throttl" in v["health_warn"][0].lower()


def test_price_failures_below_attempt_floor_not_degraded():
    # A tiny absolute number of price failures (<=100 attempts) is the normal
    # stale-listed us_all tail, not throttling — must stay benign.
    v = _assess(scanned=80, fresh_price_failures=40, fresh_price_attempts=80)
    assert v["health_warn"] == []
    assert v["fatal"] is False


def test_high_unknown_mcap_is_degraded():
    # Cap-fetch throttling collapsed the scanned set: 1711 of (237+1711+79)=2027
    # cap-gated candidates had an UNKNOWN cap (84%) — the real thin-page case.
    # Must be DEGRADED so the workflow retries with a warm cap cache instead of
    # publishing a 237-ticker page when the real universe is ~2000.
    v = _assess(scanned=237, unknown_mcap=1711, rejected_mcap=79,
                abstained_scans=11, abstained_pct=11 / 237)
    assert v["fatal"] is False
    assert any("UNKNOWN market cap" in w for w in v["health_warn"])
    assert v["summary_color"] == "yellow"


def test_normal_unknown_mcap_tail_is_benign():
    # A modest unknown-cap tail (thin/illiquid names whose cap genuinely isn't
    # published) below the threshold is normal full-market operation — benign.
    v = _assess(scanned=1900, unknown_mcap=200, rejected_mcap=300)
    assert v["health_warn"] == []
    assert v["fatal"] is False


def test_rejected_mcap_never_counts_as_degraded():
    # rejected_market_cap = cap KNOWN and below the $1B floor = a legitimate
    # exclusion. A universe that's mostly sub-$1B must NOT be flagged degraded.
    v = _assess(scanned=200, unknown_mcap=0, rejected_mcap=1800)
    assert v["health_warn"] == []
    assert v["fatal"] is False


def test_unknown_mcap_below_cap_gate_floor_not_degraded():
    # Tiny cap-gate population (<=100) is the small-universe case, not throttling.
    v = _assess(scanned=10, unknown_mcap=80, rejected_mcap=5)
    assert v["health_warn"] == []


def test_threshold_is_configurable():
    # Same abstain count: benign under a loose threshold, degraded under a tight one.
    loose = _assess(scanned=1000, abstained_scans=80, abstained_pct=0.08,
                    max_abstain_fraction=0.10)
    tight = _assess(scanned=1000, abstained_scans=80, abstained_pct=0.08,
                    max_abstain_fraction=0.02)
    assert loose["health_warn"] == []
    assert tight["health_warn"]


# ---- publish degraded-report guard (must STILL block) ----------------------
# The benign-abstain tolerance must not have neutered the safety guard: a run
# whose manifest carries `_data_quality_warnings` (degraded/empty) must STILL
# refuse to publish unless --allow-degraded is passed.


def test_publish_refuses_degraded_run(tmp_path):
    _make_run(tmp_path, warnings=["120/2000 price fetches failed this run — throttling"])
    result = runner.invoke(
        app,
        ["publish", "--out", str(tmp_path / "out"), "--docs", str(tmp_path / "docs")],
    )
    assert result.exit_code == 2
    assert "Refusing to publish" in result.stdout
    # And it must NOT have archived the run into docs/.
    assert not (tmp_path / "docs" / "runs").exists()


def test_publish_allows_degraded_with_override(tmp_path):
    _make_run(tmp_path, warnings=["120/2000 price fetches failed this run — throttling"])
    result = runner.invoke(
        app,
        ["publish", "--out", str(tmp_path / "out"), "--docs", str(tmp_path / "docs"),
         "--allow-degraded"],
    )
    assert result.exit_code == 0
    assert (tmp_path / "docs" / "runs").exists()


def test_publish_allows_clean_run(tmp_path):
    # A benign run (no `_data_quality_warnings`) publishes normally — the common
    # us_all case after the fix. Proves the tolerance actually lets runs through.
    _make_run(tmp_path, warnings=None)
    result = runner.invoke(
        app,
        ["publish", "--out", str(tmp_path / "out"), "--docs", str(tmp_path / "docs")],
    )
    assert result.exit_code == 0
    docs_runs = tmp_path / "docs" / "runs"
    assert docs_runs.exists()
    # meta.json written + landing page regenerated.
    metas = list(docs_runs.glob("*/meta.json"))
    assert metas
    meta = json.loads(metas[0].read_text())
    assert meta["universe"] == "us_all"
    assert (tmp_path / "docs" / "index.html").exists()
