"""Institutional fetch must survive a throttled `.info` blob.

Root cause of the published `CaNSL?M` (the I gate rendering `?`): the provider
sourced `heldPercentInstitutions` from `.info`, the quoteSummary endpoint that
gets `401 Invalid Crumb` throttled on datacenter IPs. When throttled, the
snapshot was None -> I abstained -> `?`.

The fix sources ownership from the DEDICATED `major_holders` /
`institutional_holders` endpoints, which are not throttled the same way. These
tests feed the provider a fake Ticker whose `.info` is empty (simulating the
throttle) but whose holder tables return data, and assert we still get a
populated snapshot with `inst_own_pct > 0`.
"""

from __future__ import annotations

import asyncio
import tempfile

import pandas as pd

from canslim.config import CacheConfig, ProviderConfig
from canslim.providers.cache import CacheStore
from canslim.providers.yfinance_provider import YFinanceProvider


class _FakeTicker:
    """Mimics yfinance.Ticker for the holder endpoints used by the provider."""

    def __init__(self, major_holders=None, institutional_holders=None, info=None):
        self._major_holders = major_holders
        self._institutional_holders = institutional_holders
        self._info = info or {}

    @property
    def major_holders(self):
        return self._major_holders

    @property
    def institutional_holders(self):
        return self._institutional_holders

    def get_info(self):
        return self._info

    # Some code paths poke at fast_info; keep it inert.
    @property
    def fast_info(self):
        return None


class _FakeYF:
    """Stand-in for the `yfinance` module: maps ticker -> _FakeTicker."""

    __version__ = "fake"

    def __init__(self, mapping):
        self._mapping = mapping

    def Ticker(self, sym):
        return self._mapping[sym]


def _modern_major_holders(inst_pct: float) -> pd.DataFrame:
    """The shape yfinance >=0.2 returns: metric-name index, single Value column."""
    return pd.DataFrame(
        {"Value": [0.016, inst_pct, inst_pct + 0.01, 7682.0]},
        index=[
            "insidersPercentHeld",
            "institutionsPercentHeld",
            "institutionsFloatPercentHeld",
            "institutionsCount",
        ],
    )


def _institutional_holders(pct_changes) -> pd.DataFrame:
    n = len(pct_changes)
    return pd.DataFrame(
        {
            "Date Reported": ["2026-03-31"] * n,
            "Holder": [f"Fund {i}" for i in range(n)],
            "pctHeld": [0.05] * n,
            "Shares": [1_000_000] * n,
            "Value": [1.0e8] * n,
            "pctChange": list(pct_changes),
        }
    )


def _provider(mapping) -> YFinanceProvider:
    tmp = tempfile.mkdtemp()
    cache = CacheStore(tmp)
    prov = YFinanceProvider(ProviderConfig(name="yfinance", concurrency=1), CacheConfig(), cache)
    prov._yf = _FakeYF(mapping)
    return prov


def _run(coro):
    return asyncio.run(coro)


def test_dedicated_endpoints_used_when_info_is_throttled():
    """`.info` empty (throttled) but holder tables present -> populated snapshot."""
    ticker = _FakeTicker(
        major_holders=_modern_major_holders(0.6582),
        institutional_holders=_institutional_holders([0.1, 0.2, -0.05]),
        info={},  # simulate the 401-throttled .info blob
    )
    prov = _provider({"AAPL": ticker})

    snap = _run(prov.get_institutional("AAPL"))

    assert snap is not None, "snapshot must NOT be None when holder tables return data"
    assert snap.inst_own_pct > 0
    assert abs(snap.inst_own_pct - 0.6582) < 1e-6
    # pctChange-derived deltas: 2 positive, 1 negative.
    assert snap.new_positions == 2
    assert snap.closed_positions == 1


def test_info_not_required_at_all():
    """Even if get_info would raise (hard throttle), we still get a snapshot."""

    class _ThrowingTicker(_FakeTicker):
        def get_info(self):
            raise RuntimeError("401 Invalid Crumb")

    ticker = _ThrowingTicker(
        major_holders=_modern_major_holders(0.71),
        institutional_holders=_institutional_holders([0.0, 0.0]),
    )
    prov = _provider({"NVDA": ticker})

    snap = _run(prov.get_institutional("NVDA"))
    assert snap is not None
    assert abs(snap.inst_own_pct - 0.71) < 1e-6


def test_pct_given_as_0_to_100_is_normalized():
    """Legacy/percent-scaled values (e.g. 65.82) coerced to a 0..1 fraction."""
    ticker = _FakeTicker(
        major_holders=_modern_major_holders(65.82),
        institutional_holders=None,
        info={},
    )
    prov = _provider({"MSFT": ticker})
    snap = _run(prov.get_institutional("MSFT"))
    assert snap is not None
    assert abs(snap.inst_own_pct - 0.6582) < 1e-6


def test_info_fallback_only_when_dedicated_endpoints_empty():
    """No holder data anywhere -> last-ditch `.info` still works."""
    ticker = _FakeTicker(
        major_holders=None,
        institutional_holders=None,
        info={"heldPercentInstitutions": 0.5},
    )
    prov = _provider({"XYZ": ticker})
    snap = _run(prov.get_institutional("XYZ"))
    assert snap is not None
    assert abs(snap.inst_own_pct - 0.5) < 1e-6


def test_none_everywhere_returns_none():
    ticker = _FakeTicker(major_holders=None, institutional_holders=None, info={})
    prov = _provider({"NADA": ticker})
    assert _run(prov.get_institutional("NADA")) is None


# ---- bounded retry + stale-cache-preferred-over-abstain ----------------------


class _FlakyTicker(_FakeTicker):
    """A straggler that gets throttled on the first N attempts, then succeeds.

    Simulates a transient yfinance throttle on the dedicated holder endpoints
    under a full us_all scan: the data IS gettable, the run just transiently
    fails on the first call(s). `major_holders` raises until `fail_first`
    attempts have elapsed, after which it returns real ownership data.
    """

    def __init__(self, inst_pct: float, fail_first: int):
        super().__init__(info={})
        self._inst_pct = inst_pct
        self._fail_first = fail_first
        self.calls = 0

    @property
    def major_holders(self):
        self.calls += 1
        if self.calls <= self._fail_first:
            raise RuntimeError("429 Too Many Requests (throttled)")
        return _modern_major_holders(self._inst_pct)

    @property
    def institutional_holders(self):
        # Mirror the throttle on the second endpoint too.
        if self.calls <= self._fail_first:
            return None
        return _institutional_holders([0.1, 0.2])


def test_bounded_retry_recovers_transient_throttle():
    """A throttle on the 1st attempt is recovered by the bounded retry within
    the same run — no stale cache needed, no abstain."""
    ticker = _FlakyTicker(inst_pct=0.6582, fail_first=1)  # fails once, then OK
    prov = _provider({"ACA": ticker})
    # default config -> 3 attempts; backoff is zeroed so the test stays fast
    prov.cache_cfg.institutional_retry_backoff_s = 0.0

    snap = _run(prov.get_institutional("ACA"))

    assert snap is not None, "bounded retry must recover a transient throttle"
    assert abs(snap.inst_own_pct - 0.6582) < 1e-6
    assert snap.is_stale is False  # came from a fresh fetch, not stale cache
    assert ticker.calls >= 2, "should have retried past the first throttled call"


def test_retry_attempts_are_bounded():
    """A persistently-throttled straggler with NO cache abstains after exactly
    `institutional_retry_attempts` tries — the retry can't run away."""
    ticker = _FlakyTicker(inst_pct=0.5, fail_first=99)  # never succeeds
    prov = _provider({"PHIN": ticker})
    prov.cache_cfg.institutional_retry_attempts = 3
    prov.cache_cfg.institutional_retry_backoff_s = 0.0

    snap = _run(prov.get_institutional("PHIN"))

    assert snap is None, "no data ever existed -> abstain is the correct last resort"
    assert ticker.calls == 3, "must stop at the bounded attempt count, not loop forever"


def test_stale_cache_preferred_over_abstain():
    """A throttled straggler that HAS a persisted last-known-good value renders
    from stale cache instead of abstaining (`?`). This is the runner path once
    the cache persists across runs."""
    import json
    from pathlib import Path

    # A ticker whose dedicated endpoints + info are all throttled/empty now.
    ticker = _FakeTicker(major_holders=None, institutional_holders=None, info={})
    prov = _provider({"PHIN": ticker})

    # Seed a stale-but-good institutional snapshot from a prior run, then
    # back-date its mtime well beyond the TTL so the fresh path is forced.
    prov.cache.write_json(
        "institutional",
        prov.name,
        "PHIN",
        {
            "ticker": "PHIN",
            "reported_at": "2026-03-31",
            "inst_own_pct": 0.842,
            "qoq_delta_pct": None,
            "new_positions": 0,
            "closed_positions": 0,
        },
    )
    p = Path(prov.cache.root) / "institutional" / prov.name / "PHIN.json"
    old = p.stat().st_mtime - (prov.cache_cfg.institutional_ttl_hours + 48) * 3600
    import os as _os
    _os.utime(p, (old, old))

    snap = _run(prov.get_institutional("PHIN"))

    assert snap is not None, "stale cache must be preferred over abstaining"
    assert abs(snap.inst_own_pct - 0.842) < 1e-6
    assert snap.is_stale is True, "snapshot must be flagged as last-known-good"
    assert snap.data_age_days >= 7


def test_stale_cache_skips_extra_retries():
    """When a usable stale value exists, a throttled fetch falls back to it after
    a SINGLE attempt — it does NOT burn the full retry budget (runtime guard for
    a cold-but-persisted cache over ~1500 tickers)."""
    import os as _os
    from pathlib import Path

    ticker = _FlakyTicker(inst_pct=0.5, fail_first=99)  # always throttled
    prov = _provider({"WMT": ticker})
    prov.cache_cfg.institutional_retry_attempts = 3
    prov.cache_cfg.institutional_retry_backoff_s = 0.0

    prov.cache.write_json(
        "institutional", prov.name, "WMT",
        {"ticker": "WMT", "reported_at": "2026-03-31", "inst_own_pct": 0.77,
         "qoq_delta_pct": None, "new_positions": 0, "closed_positions": 0},
    )
    p = Path(prov.cache.root) / "institutional" / prov.name / "WMT.json"
    old = p.stat().st_mtime - (prov.cache_cfg.institutional_ttl_hours + 1) * 3600
    _os.utime(p, (old, old))

    snap = _run(prov.get_institutional("WMT"))

    assert snap is not None and snap.is_stale is True
    assert ticker.calls == 1, "stale value present -> only ONE fetch attempt, no retry storm"


def test_snapshot_makes_i_gate_pass():
    """End-to-end: the populated snapshot must make the I criterion PASS
    (uppercase I), not abstain — that's what kills the report's `?`."""
    from canslim.criteria.base import CriterionContext
    from canslim.criteria.i_institutional import Institutional
    from canslim.config import CriteriaThresholds

    ticker = _FakeTicker(
        major_holders=_modern_major_holders(0.6582),
        institutional_holders=_institutional_holders([0.1, 0.2]),
        info={},
    )
    prov = _provider({"AAPL": ticker})
    snap = _run(prov.get_institutional("AAPL"))

    ctx = CriterionContext(ticker="AAPL", thresholds=CriteriaThresholds(), institutional=snap)
    res = Institutional().evaluate(ctx)
    assert res.data_available is True
    assert res.passed is True
