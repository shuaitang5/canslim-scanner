from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any, Optional

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from canslim.config import CacheConfig, ProviderConfig
from canslim.models import EarningsBundle, InstitutionalSnapshot
from canslim.providers.base import BudgetExhausted, DataProvider, ProviderError, RateLimited
from canslim.providers.cache import CacheStore

log = logging.getLogger(__name__)

BASE_URL = "https://financialmodelingprep.com/stable"
BUDGET_META = "fmp_budget"


class FMPProvider(DataProvider):
    """Financial Modeling Prep provider for quarterly/annual fundamentals + institutional.

    Free tier is 250 requests/day. We persist a per-day counter to `cache/meta/fmp_budget.json`
    and raise `BudgetExhausted` when depleted. Scanner treats that as `status=pending_budget`.
    """

    name = "fmp"

    def __init__(self, cfg: ProviderConfig, cache_cfg: CacheConfig, cache: CacheStore) -> None:
        self.cfg = cfg
        self.cache_cfg = cache_cfg
        self.cache = cache
        self._api_key = cfg.resolved_api_key()
        self._client = httpx.AsyncClient(timeout=cfg.request_timeout_s)
        self._sem = asyncio.Semaphore(max(1, cfg.concurrency or 2))

    async def close(self) -> None:
        await self._client.aclose()

    # ---- budget

    def _today_key(self) -> str:
        return date.today().isoformat()

    def _budget_state(self) -> dict[str, Any]:
        meta = self.cache.read_meta(BUDGET_META)
        today = self._today_key()
        if meta.get("date") != today:
            meta = {"date": today, "used": 0}
            self.cache.write_meta(BUDGET_META, meta)
        return meta

    def budget_remaining(self) -> Optional[int]:
        if self.cfg.daily_budget is None:
            return None
        state = self._budget_state()
        return max(0, self.cfg.daily_budget - int(state.get("used", 0)))

    def _check_budget(self, cost: int = 1) -> None:
        remaining = self.budget_remaining()
        if remaining is not None and remaining < cost:
            raise BudgetExhausted(f"FMP daily budget exhausted ({self.cfg.daily_budget} req/day)")

    def _charge(self, cost: int = 1) -> None:
        if self.cfg.daily_budget is None:
            return
        state = self._budget_state()
        state["used"] = int(state.get("used", 0)) + cost
        self.cache.write_meta(BUDGET_META, state)

    # ---- HTTP

    async def _get(self, path: str, params: Optional[dict] = None, cost: int = 1) -> Any:
        if not self._api_key:
            raise ProviderError("FMP API key is not set (providers.fmp.api_key or env)")
        self._check_budget(cost)
        p = dict(params or {})
        p["apikey"] = self._api_key
        url = f"{BASE_URL}{path}"

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.cfg.max_retries),
            wait=wait_exponential_jitter(initial=1.0, max=10.0),
            retry=retry_if_exception_type((RateLimited, httpx.TransportError, httpx.ReadTimeout)),
            reraise=True,
        ):
            with attempt:
                async with self._sem:
                    resp = await self._client.get(url, params=p)
                if resp.status_code == 429:
                    raise RateLimited(f"FMP rate limited: {resp.text[:200]}")
                if resp.status_code >= 500:
                    raise httpx.TransportError(f"FMP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 400:
                    raise ProviderError(f"FMP HTTP {resp.status_code}: {resp.text[:200]}")
                self._charge(cost)
                return resp.json()
        return None  # unreachable

    async def health_check(self) -> dict[str, str]:
        if not self._api_key:
            raise ProviderError("FMP API key missing (providers.fmp.api_key or env var)")
        data = await self._get("/profile", params={"symbol": "AAPL"}, cost=1)
        if not data:
            raise ProviderError("FMP profile endpoint returned empty for AAPL")
        remaining = self.budget_remaining()
        return {
            "provider": self.name,
            "aapl_name": data[0].get("companyName", "?"),
            "budget_remaining": str(remaining) if remaining is not None else "unmetered",
        }

    # ---- fundamentals

    async def get_fundamentals(self, ticker: str) -> EarningsBundle:
        if self.cache.is_json_fresh("fundamentals", self.name, ticker, self.cache_cfg.fundamentals_ttl_hours):
            blob = self.cache.read_json("fundamentals", self.name, ticker)
            if blob:
                return _bundle_from_cache(blob)
        quarters = await self._get(
            "/income-statement", params={"symbol": ticker, "period": "quarter", "limit": 12}, cost=1
        )
        annuals = await self._get(
            "/income-statement", params={"symbol": ticker, "period": "annual", "limit": 6}, cost=1
        )
        key_metrics = await self._get(
            "/key-metrics", params={"symbol": ticker, "period": "annual", "limit": 6}, cost=1
        )

        q_eps, q_periods = _eps_from_statements(quarters or [])
        a_eps, a_periods = _eps_from_statements(annuals or [])
        a_roe = _roe_series(key_metrics or [], a_periods)

        bundle = EarningsBundle(
            ticker=ticker,
            quarterly_eps=q_eps,
            quarterly_periods=q_periods,
            annual_eps=a_eps,
            annual_periods=a_periods,
            annual_roe_pct=a_roe,
        )
        self.cache.write_json("fundamentals", self.name, ticker, bundle.model_dump(mode="json"))
        return bundle

    # ---- institutional

    async def get_institutional(self, ticker: str) -> Optional[InstitutionalSnapshot]:
        cached = self.cache.read_json("institutional", self.name, ticker)
        if self.cache.is_json_fresh("institutional", self.name, ticker, self.cache_cfg.institutional_ttl_hours):
            if cached:
                return _snap_from_cache(cached)
        # Stable API: /institutional-ownership/symbol-ownership?symbol=AAPL
        holders = await self._get(
            "/institutional-ownership/symbol-ownership",
            params={"symbol": ticker, "includeCurrentQuarter": "false"},
            cost=1,
        )

        if not holders:
            # Stale-data fallback: fresh FMP fetch returned nothing. Use last
            # cached institutional snapshot if we have it.
            if cached:
                return _snap_from_cache(cached)
            return None
        # Aggregate naive signals: count rows with `change > 0` vs `change < 0`
        total_shares = 0.0
        new_positions = 0
        closed_positions = 0
        latest = None
        for h in holders:
            try:
                total_shares += float(h.get("shares") or 0.0)
                change = float(h.get("change") or 0.0)
            except (TypeError, ValueError):
                change = 0.0
            if change > 0:
                new_positions += 1
            elif change < 0:
                closed_positions += 1
            d = h.get("dateReported") or h.get("date")
            if d and (latest is None or d > latest):
                latest = d
        reported_at = date.fromisoformat(latest[:10]) if latest else date.today()
        # inst_own_pct: FMP profile carries `floatShares`; but without it we leave as 0 and let caller fill.
        snap = InstitutionalSnapshot(
            ticker=ticker,
            reported_at=reported_at,
            inst_own_pct=0.0,  # will be enriched by combining with shares outstanding if needed
            qoq_delta_pct=None,
            new_positions=new_positions,
            closed_positions=closed_positions,
        )
        # Stash aggregated shares for optional enrichment
        blob = snap.model_dump(mode="json")
        blob["aggregated_inst_shares"] = total_shares
        self.cache.write_json("institutional", self.name, ticker, blob)
        return snap


def _eps_from_statements(rows: list[dict]) -> tuple[list[float], list[str]]:
    out: list[tuple[str, float]] = []
    for r in rows:
        eps = r.get("epsdiluted")
        if eps is None:
            eps = r.get("eps")
        if eps is None:
            continue
        period = r.get("period")
        year = r.get("calendarYear") or (r.get("date") or "")[:4]
        if period and period != "FY" and year:
            key = f"{year}-{period}"
        elif year:
            key = str(year)
        else:
            key = r.get("date", "")
        try:
            out.append((key, float(eps)))
        except (TypeError, ValueError):
            continue
    # FMP returns newest first already; enforce it
    out.sort(key=lambda x: x[0], reverse=True)
    return [v for _, v in out], [k for k, _ in out]


def _roe_series(key_metrics: list[dict], annual_periods: list[str]) -> list[float]:
    by_year: dict[str, float] = {}
    for km in key_metrics:
        y = str(km.get("calendarYear") or (km.get("date") or "")[:4])
        roe = km.get("roe")
        try:
            if roe is not None:
                by_year[y] = float(roe)
        except (TypeError, ValueError):
            continue
    return [by_year.get(p[:4], 0.0) for p in annual_periods]


def _bundle_from_cache(d: dict) -> EarningsBundle:
    return EarningsBundle.model_validate(d)


def _snap_from_cache(d: dict) -> InstitutionalSnapshot:
    return InstitutionalSnapshot(
        ticker=d["ticker"],
        reported_at=date.fromisoformat(d["reported_at"][:10]) if isinstance(d["reported_at"], str) else d["reported_at"],
        inst_own_pct=float(d.get("inst_own_pct", 0.0)),
        qoq_delta_pct=d.get("qoq_delta_pct"),
        new_positions=int(d.get("new_positions", 0)),
        closed_positions=int(d.get("closed_positions", 0)),
    )
