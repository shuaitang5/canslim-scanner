from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from canslim.config import CacheConfig, ProviderConfig
from canslim.models import EarningsBundle, InstitutionalSnapshot
from canslim.providers.base import DataProvider, ProviderError
from canslim.providers.cache import CacheStore

log = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class YFinanceProvider(DataProvider):
    """Price-focused provider backed by `yfinance`.

    `yfinance` is synchronous; we run its blocking calls in a thread pool. The main
    win is the batch `yf.download(tickers=..., group_by='ticker')` call, which we use
    in the pre-filter stage to cut the 6000-ticker universe in one shot.
    """

    name = "yfinance"

    def __init__(self, cfg: ProviderConfig, cache_cfg: CacheConfig, cache: CacheStore) -> None:
        self.cfg = cfg
        self.cache_cfg = cache_cfg
        self.cache = cache
        self._sem = asyncio.Semaphore(max(1, cfg.concurrency))
        # Import lazily so `canslim check-providers` can fail gracefully if yfinance isn't installed
        self._yf = __import__("yfinance")

    async def health_check(self) -> dict[str, str]:
        def _ping() -> str:
            t = self._yf.Ticker("SPY")
            hist = t.history(period="5d", auto_adjust=False)
            if hist is None or hist.empty:
                raise ProviderError("yfinance returned no data for SPY — network or upstream issue")
            return str(hist.index.max().date())
        latest = await asyncio.to_thread(_ping)
        return {"provider": self.name, "latest_spy_date": latest, "yfinance_version": self._yf.__version__}

    # ---- prices

    async def get_prices(
        self,
        tickers: list[str],
        start: Optional[date] = None,
        end: Optional[date] = None,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        if not tickers:
            return {}
        today = date.today()
        end = end or today
        start = start or (today - timedelta(days=380))

        # Track per-ticker outcome so the scanner can surface misses
        self.last_fetch_stats = {"cache_hits": 0, "fresh_fetches": 0, "failures": 0, "skipped_negative": 0}
        self.last_missing: list[tuple[str, str]] = []  # (ticker, reason)

        out: dict[str, pd.DataFrame] = {}
        to_fetch: list[str] = []
        for t in tickers:
            if not force_refresh and self.cache.is_fresh("prices", self.name, t, self.cache_cfg.price_ttl_hours):
                df = self.cache.read_df("prices", self.name, t)
                if df is not None and not df.empty:
                    out[t] = df
                    self.last_fetch_stats["cache_hits"] += 1
                    continue
            if not force_refresh and self.cache.is_failure_fresh(
                "prices", self.name, t, self.cache_cfg.failure_ttl_hours
            ):
                fail = self.cache.read_failure("prices", self.name, t) or {}
                self.last_missing.append((t, f"negative-cache: {fail.get('error', 'prior failure')[:120]}"))
                self.last_fetch_stats["skipped_negative"] += 1
                continue
            to_fetch.append(t)

        if to_fetch:
            fetched = await self._download_batch(to_fetch, start, end)
            for t in to_fetch:
                df = fetched.get(t)
                if df is None or df.empty:
                    reason = "yfinance returned no bars in batch"
                    self.cache.record_failure("prices", self.name, t, reason)
                    self.last_missing.append((t, reason))
                    self.last_fetch_stats["failures"] += 1
                    continue
                self.cache.write_df("prices", self.name, t, df)
                self.cache.clear_failure("prices", self.name, t)
                out[t] = df
                self.last_fetch_stats["fresh_fetches"] += 1
        return out

    async def _download_batch(
        self, tickers: list[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        async with self._sem:
            return await asyncio.to_thread(self._download_sync, tickers, start, end)

    def _download_sync(self, tickers: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
        # yfinance handles batching internally; cap at 400 per call to avoid URL size issues
        results: dict[str, pd.DataFrame] = {}
        BATCH = 400
        for i in range(0, len(tickers), BATCH):
            chunk = tickers[i : i + BATCH]
            try:
                raw = self._yf.download(
                    tickers=chunk,
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    threads=True,
                    group_by="ticker",
                )
            except Exception as e:  # pragma: no cover - upstream variability
                log.warning("yfinance batch download failed for %d tickers: %s", len(chunk), e)
                continue
            if raw is None or raw.empty:
                continue
            if len(chunk) == 1:
                t = chunk[0]
                results[t] = _normalize_bars(raw)
            else:
                for t in chunk:
                    if t not in raw.columns.get_level_values(0):
                        continue
                    df = raw[t]
                    results[t] = _normalize_bars(df)
        return results

    # ---- per-ticker info (float)

    async def get_shares_float(self, ticker: str) -> Optional[float]:
        cached = self.cache.read_json("info", self.name, ticker)
        cached_float: Optional[float] = (
            cached.get("float_shares") if cached is not None else None
        )
        # Critical: only short-circuit on a fresh cache that ACTUALLY has the value.
        # Previously we'd accept `float_shares: null` from a partial earlier fetch,
        # which silently abstained on S forever until cache expired (7 days).
        if cached_float is not None and self.cache.is_json_fresh(
            "info", self.name, ticker, self.cache_cfg.fundamentals_ttl_hours
        ):
            return cached_float

        # Cache is missing the value (or stale) — try a fresh fetch.
        async with self._sem:
            info = await asyncio.to_thread(self._get_info_sync, ticker)
        if info is not None and info.get("float_shares") is not None:
            self.cache.write_json("info", self.name, ticker, info)
            return info["float_shares"]

        # Fresh fetch returned nothing useful. Stale-data fallback: float-share
        # counts barely change day-to-day, so a known-good value from a previous
        # successful fetch is better than abstaining on S indefinitely.
        if cached_float is not None:
            age_h = self.cache.json_age_hours("info", self.name, ticker) or 0.0
            log.debug("Float fetch failed for %s — using stale cache (%.1fh old)", ticker, age_h)
            return cached_float
        return None

    async def get_market_cap(self, ticker: str) -> Optional[float]:
        """Latest market capitalization (USD), reusing the same `info` cache blob
        that `get_shares_float` populates — `_get_info_sync` already pulls
        `marketCap` from yfinance `fast_info` (no crumb, survives 401s), so a
        cache hit costs zero extra network calls. Mirrors `get_shares_float`'s
        fresh-fetch + stale-fallback policy (market cap drifts slowly day to day,
        so last-known-good beats abstaining)."""
        cached = self.cache.read_json("info", self.name, ticker)
        cached_cap: Optional[float] = (
            cached.get("market_cap") if cached is not None else None
        )
        if cached_cap is not None and self.cache.is_json_fresh(
            "info", self.name, ticker, self.cache_cfg.fundamentals_ttl_hours
        ):
            return cached_cap

        async with self._sem:
            info = await asyncio.to_thread(self._get_info_sync, ticker)
        if info is not None and info.get("market_cap") is not None:
            # Merge into the existing blob so we don't clobber float_shares etc.
            merged = dict(cached or {})
            merged.update({k: v for k, v in info.items() if v is not None})
            self.cache.write_json("info", self.name, ticker, merged)
            return info["market_cap"]

        if cached_cap is not None:
            age_h = self.cache.json_age_hours("info", self.name, ticker) or 0.0
            log.debug("Market-cap fetch failed for %s — using stale cache (%.1fh old)", ticker, age_h)
            return cached_cap
        return None

    async def get_shares_outstanding(self, ticker: str) -> Optional[float]:
        """Shares outstanding, reusing the crumbless `fast_info`/cached info blob.

        Used by the scanner's computed-cap fallback (cap = close * shares) when
        the direct marketCap fetch is throttled. `fast_info.shares` comes from
        the spark-chart endpoint (no crumb), so this survives the 401 storms that
        knock out `get_info`. Mirrors get_market_cap's fresh + stale-cache policy."""
        cached = self.cache.read_json("info", self.name, ticker)
        cached_so: Optional[float] = (
            cached.get("shares_outstanding") if cached is not None else None
        )
        if cached_so is not None and self.cache.is_json_fresh(
            "info", self.name, ticker, self.cache_cfg.fundamentals_ttl_hours
        ):
            return cached_so

        async with self._sem:
            info = await asyncio.to_thread(self._get_info_sync, ticker)
        if info is not None and info.get("shares_outstanding") is not None:
            merged = dict(cached or {})
            merged.update({k: v for k, v in info.items() if v is not None})
            self.cache.write_json("info", self.name, ticker, merged)
            return info["shares_outstanding"]

        if cached_so is not None:
            return cached_so
        return None

    def _get_info_sync(self, ticker: str) -> Optional[dict]:
        """Resilient info lookup.

        Strategy: `fast_info` first — uses the spark-chart endpoint and doesn't need a crumb,
        so it survives the 401 "Invalid Crumb" batch failures. Fall back to `get_info()` with
        one retry (clearing the session) only if fast_info missed the field we need.
        """
        t = self._yf.Ticker(ticker)
        out: dict = {}

        fi = _try_fast_info(t)
        if fi:
            out["float_shares"] = _as_float(fi.get("floatShares") or fi.get("shares"))
            out["shares_outstanding"] = _as_float(fi.get("shares"))
            out["market_cap"] = _as_float(fi.get("marketCap") or fi.get("market_cap"))

        # Only escalate to get_info (crumbed, flaky) if we still lack critical fields.
        needs_fallback = out.get("float_shares") is None or out.get("shares_outstanding") is None
        if needs_fallback:
            info = _try_get_info_with_retry(t, attempts=2)
            if info:
                out.setdefault("float_shares", _as_float(info.get("floatShares")))
                out.setdefault("shares_outstanding", _as_float(info.get("sharesOutstanding")))
                out.setdefault("market_cap", _as_float(info.get("marketCap")))
                out["short_name"] = info.get("shortName")
                out["held_percent_institutions"] = _as_float(info.get("heldPercentInstitutions"))

        return out if out else None

    # ---- fallback fundamentals (used when FMP is unavailable/out of budget)

    async def get_fundamentals(self, ticker: str) -> EarningsBundle:
        async with self._sem:
            payload = await asyncio.to_thread(self._get_fundamentals_sync, ticker)
        return payload

    def _get_fundamentals_sync(self, ticker: str) -> EarningsBundle:
        try:
            t = self._yf.Ticker(ticker)
            q = t.quarterly_income_stmt
            a = t.income_stmt
            info = t.get_info() or {}
        except Exception as e:
            log.debug("yfinance fundamentals failed for %s: %s", ticker, e)
            return EarningsBundle(ticker=ticker)

        q_eps, q_periods = _extract_eps_row(q)
        a_eps, a_periods = _extract_eps_row(a)
        roe = _as_float(info.get("returnOnEquity"))
        a_roe = [roe or 0.0] * len(a_periods)
        return EarningsBundle(
            ticker=ticker,
            quarterly_eps=q_eps,
            quarterly_periods=q_periods,
            annual_eps=a_eps,
            annual_periods=a_periods,
            annual_roe_pct=a_roe,
        )

    async def get_institutional(self, ticker: str) -> Optional[InstitutionalSnapshot]:
        cached = self.cache.read_json("institutional", self.name, ticker)
        # Same fix as get_shares_float: only short-circuit on a fresh cache
        # that actually has the value we need.
        if cached is not None and cached.get("inst_own_pct") is not None and self.cache.is_json_fresh(
            "institutional", self.name, ticker, self.cache_cfg.institutional_ttl_hours
        ):
            return _snap_from_cache(cached, age_hours=0.0)
        async with self._sem:
            snap = await asyncio.to_thread(self._get_institutional_sync, ticker)
        if snap is not None:
            self.cache.write_json(
                "institutional",
                self.name,
                ticker,
                {
                    "ticker": snap.ticker,
                    "reported_at": snap.reported_at.isoformat(),
                    "inst_own_pct": snap.inst_own_pct,
                    "qoq_delta_pct": snap.qoq_delta_pct,
                    "new_positions": snap.new_positions,
                    "closed_positions": snap.closed_positions,
                },
            )
            return snap
        # Stale-data fallback: fresh fetch failed but we have a cached value
        # from before. Institutional ownership barely changes day-to-day; using
        # last-known-good data is far better than failing the I criterion silently.
        if cached is not None and cached.get("inst_own_pct") is not None:
            age_h = self.cache.json_age_hours("institutional", self.name, ticker) or 0.0
            log.debug("Institutional fetch failed for %s — using stale cache (%.1fh old)", ticker, age_h)
            return _snap_from_cache(cached, age_hours=age_h)
        return None

    def _get_institutional_sync(self, ticker: str) -> Optional[InstitutionalSnapshot]:
        t = self._yf.Ticker(ticker)
        info = _try_get_info_with_retry(t, attempts=2)
        if not info:
            return None
        pct = _as_float(info.get("heldPercentInstitutions"))
        if pct is None:
            return None
        return InstitutionalSnapshot(
            ticker=ticker,
            reported_at=date.today(),
            inst_own_pct=pct,
            qoq_delta_pct=None,
        )


def _normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance with group_by='ticker' returns MultiIndex columns even for single tickers;
    # drop the ticker level if present.
    if isinstance(df.columns, pd.MultiIndex):
        levels = df.columns.nlevels
        if levels >= 2:
            # Keep the last level (price field names); if only one ticker value, droplevel(0).
            try:
                df = df.droplevel(0, axis=1)
            except Exception:
                df.columns = df.columns.get_level_values(-1)
    rename = {c: str(c).lower().replace(" ", "_") for c in df.columns}
    out = df.rename(columns=rename)
    for col in _REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[_REQUIRED_COLUMNS].copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    out.index = idx
    out.index.name = "date"
    out["fetched_at"] = pd.Timestamp.now(tz="UTC")
    return out.dropna(subset=["close"])


def _extract_eps_row(df) -> tuple[list[float], list[str]]:
    if df is None or df.empty:
        return [], []
    # yfinance uses "Diluted EPS" or "Basic EPS" as index entries
    candidates = ["Diluted EPS", "Basic EPS"]
    row = None
    for c in candidates:
        if c in df.index:
            row = df.loc[c]
            break
    if row is None:
        return [], []
    ordered = row.dropna().sort_index(ascending=False)
    vals = [float(v) for v in ordered.tolist()]
    periods = [pd.Timestamp(p).date().isoformat() for p in ordered.index]
    return vals, periods


def _try_fast_info(t) -> Optional[dict]:
    """yfinance.fast_info is a lazy dict-like backed by the spark-chart endpoint (no crumb)."""
    try:
        fi = t.fast_info
        if fi is None:
            return None
        # Materialize a dict so we can introspect keys safely
        out: dict = {}
        for key in ("shares", "floatShares", "marketCap", "market_cap", "last_price", "year_high"):
            try:
                v = fi[key] if hasattr(fi, "__getitem__") else getattr(fi, key, None)
            except Exception:
                v = None
            if v is not None:
                out[key] = v
        return out or None
    except Exception as e:
        log.debug("fast_info failed: %s", e)
        return None


def _try_get_info_with_retry(t, attempts: int = 2) -> Optional[dict]:
    """Call .get_info() with a tiny retry. On 401/Invalid Crumb we clear the session
    so yfinance refreshes its crumb cookie, then try once more.
    """
    import time as _time
    last_exc: Optional[Exception] = None
    for i in range(attempts):
        try:
            info = t.get_info()
            if info:
                return info
        except Exception as e:
            last_exc = e
            msg = str(e)
            # Reset session so yfinance re-crumbs next call; swallow if session attr isn't there.
            if "401" in msg or "Crumb" in msg or "Unauthorized" in msg:
                try:
                    sess = getattr(t, "_session", None) or getattr(t, "session", None)
                    if sess is not None and hasattr(sess, "cookies"):
                        sess.cookies.clear()
                except Exception:
                    pass
                _time.sleep(0.4 * (i + 1))
                continue
            # Non-auth failure: don't waste attempts
            break
    if last_exc is not None:
        log.debug("get_info failed after %d attempts: %s", attempts, last_exc)
    return None


def _as_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _snap_from_cache(d: dict, age_hours: float = 0.0) -> InstitutionalSnapshot:
    """Hydrate an InstitutionalSnapshot from cached JSON. `age_hours>0` flags
    the snapshot as stale (fresh fetch failed, we're using last-known-good)."""
    days = int(age_hours / 24)
    return InstitutionalSnapshot(
        ticker=d["ticker"],
        reported_at=date.fromisoformat(d["reported_at"]),
        inst_own_pct=float(d["inst_own_pct"]),
        qoq_delta_pct=d.get("qoq_delta_pct"),
        new_positions=int(d.get("new_positions", 0)),
        closed_positions=int(d.get("closed_positions", 0)),
        data_age_days=days,
        is_stale=age_hours > 0,
    )
