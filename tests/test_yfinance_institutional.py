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
