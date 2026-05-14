"""SEC EDGAR provider — free, unlimited, authoritative US fundamentals.

Uses the XBRL `companyfacts` JSON endpoint which returns every reported
concept for a company in one call: quarterly + annual EPS, net income,
shareholders equity (for ROE), revenue, etc.

Rate limit: SEC's fair-access policy allows up to 10 req/sec with a proper
User-Agent header. We default to Semaphore(5) and require a descriptive UA.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Any, Optional

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from canslim.config import CacheConfig, ProviderConfig
from canslim.models import EarningsBundle, InstitutionalSnapshot, NormalizationAdjustment
from canslim.normalization import apply_rules, default_rules
from canslim.providers.base import DataProvider, ProviderError, RateLimited
from canslim.providers.cache import CacheStore

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# Domestic issuers file 10-K/10-Q with us-gaap taxonomy; foreign private issuers file 20-F with ifrs-full.
# Try us-gaap first (quarterly data available), fall back to ifrs-full (typically annual-only).
_TAXONOMIES = (
    {
        "name": "us-gaap",
        "eps": ("EarningsPerShareDiluted", "EarningsPerShareBasic"),
        "net_income": ("NetIncomeLoss", "ProfitLoss"),
        "equity": (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
    },
    {
        "name": "ifrs-full",
        "eps": ("DilutedEarningsLossPerShare", "BasicEarningsLossPerShare"),
        "net_income": ("ProfitLoss", "ProfitLossAttributableToOwnersOfParent"),
        "equity": ("Equity", "EquityAttributableToOwnersOfParent"),
    },
)


class SECProvider(DataProvider):
    name = "sec"

    def __init__(
        self,
        cfg: ProviderConfig,
        cache_cfg: CacheConfig,
        cache: CacheStore,
        user_agent: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.cache_cfg = cache_cfg
        self.cache = cache
        ua = user_agent or cfg.api_key or "canslim-scanner contact@example.com"
        if "@" not in ua:
            # SEC requires contact info in UA; append a sentinel if the user gave just a name
            ua = f"{ua} contact@example.com"
        self._client = httpx.AsyncClient(
            timeout=cfg.request_timeout_s,
            headers={"User-Agent": ua, "Accept": "application/json"},
        )
        self._sem = asyncio.Semaphore(max(1, min(cfg.concurrency, 5)))  # cap at 5 rps
        self._ticker_to_cik: Optional[dict[str, str]] = None
        self._ticker_map_lock = asyncio.Lock()

    async def close(self) -> None:
        await self._client.aclose()

    # ---- HTTP

    async def _get_json(self, url: str) -> Any:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.cfg.max_retries),
            wait=wait_exponential_jitter(initial=1.0, max=10.0),
            retry=retry_if_exception_type((RateLimited, httpx.TransportError, httpx.ReadTimeout)),
            reraise=True,
        ):
            with attempt:
                async with self._sem:
                    resp = await self._client.get(url)
                if resp.status_code == 429:
                    raise RateLimited(f"SEC rate limited: {resp.text[:200]}")
                if resp.status_code == 404:
                    raise ProviderError(f"SEC 404: {url}")
                if resp.status_code >= 500:
                    raise httpx.TransportError(f"SEC {resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 400:
                    raise ProviderError(f"SEC HTTP {resp.status_code}: {resp.text[:200]}")
                return resp.json()
        return None  # unreachable

    # ---- health

    async def health_check(self) -> dict[str, str]:
        await self._ensure_ticker_map()
        # confirm we can fetch AAPL facts
        cik = await self.cik_for("AAPL")
        if cik is None:
            raise ProviderError("SEC ticker map loaded but AAPL CIK missing")
        data = await self._get_json(FACTS_URL.format(cik=cik))
        name = data.get("entityName", "?") if isinstance(data, dict) else "?"
        return {
            "provider": self.name,
            "aapl_name": name,
            "ticker_map_size": str(len(self._ticker_to_cik or {})),
        }

    # ---- ticker → CIK

    async def _ensure_ticker_map(self) -> dict[str, str]:
        if self._ticker_to_cik is not None:
            return self._ticker_to_cik
        async with self._ticker_map_lock:
            if self._ticker_to_cik is not None:
                return self._ticker_to_cik
            cached = self.cache.read_json("ticker_map", self.name, "_all")
            if cached and self.cache.is_json_fresh("ticker_map", self.name, "_all", ttl_hours=24 * 7):
                self._ticker_to_cik = {k: v for k, v in cached.items() if not k.startswith("_")}
                return self._ticker_to_cik
            raw = await self._get_json(TICKERS_URL)
            mapping: dict[str, str] = {}
            if isinstance(raw, dict):
                for entry in raw.values():
                    try:
                        t = str(entry["ticker"]).upper().replace(".", "-")
                        cik = str(int(entry["cik_str"])).zfill(10)
                        mapping[t] = cik
                    except (KeyError, ValueError, TypeError):
                        continue
            self._ticker_to_cik = mapping
            self.cache.write_json("ticker_map", self.name, "_all", mapping)
            return mapping

    async def cik_for(self, ticker: str) -> Optional[str]:
        m = await self._ensure_ticker_map()
        return m.get(ticker.upper())

    # ---- fundamentals

    async def get_fundamentals(self, ticker: str) -> EarningsBundle:
        if self.cache.is_json_fresh("fundamentals", self.name, ticker, self.cache_cfg.fundamentals_ttl_hours):
            blob = self.cache.read_json("fundamentals", self.name, ticker)
            if blob:
                return EarningsBundle.model_validate(
                    {k: v for k, v in blob.items() if not k.startswith("_")}
                )
        cik = await self.cik_for(ticker)
        if cik is None:
            # Ticker not in SEC map (e.g. ETF, foreign, or delisted) — no EDGAR data
            return EarningsBundle(ticker=ticker)
        try:
            facts = await self._get_json(FACTS_URL.format(cik=cik))
            bundle = _bundle_from_facts(ticker, facts)
            self.cache.write_json("fundamentals", self.name, ticker, bundle.model_dump(mode="json"))
            return bundle
        except Exception as e:
            # Stale-data fallback: fresh fetch failed (SEC throttled, network, etc.).
            # SEC fundamentals barely change between scans (quarterly filings) — return
            # the last cached version rather than letting C/A criteria silently fail.
            blob = self.cache.read_json("fundamentals", self.name, ticker)
            if blob:
                log.debug("SEC fundamentals fetch failed for %s — using stale cache: %s", ticker, e)
                return EarningsBundle.model_validate(
                    {k: v for k, v in blob.items() if not k.startswith("_")}
                )
            raise

    async def get_institutional(self, ticker: str) -> Optional[InstitutionalSnapshot]:
        # EDGAR has 13F filings but aggregation is non-trivial; defer.
        return None

    async def get_recent_management_events(self, ticker: str, within_days: int = 90) -> list[dict]:
        """Return SEC 8-K Item 5.02 filings in the trailing window.

        Item 5.02 = "Departure of Directors or Certain Officers; Election of Directors;
        Appointment of Certain Officers". Captures O'Neil's "new management" sub-signal
        of the N gate in a clean structured way.

        Cost: one SEC API call per ticker. Called only for top candidates via
        the scanner's enrichment step, never for the full universe.
        """
        cik = await self.cik_for(ticker)
        if cik is None:
            return []
        # No separate cache — submissions data updates daily, and we only call for top tickers
        data = await self._get_json(SUBMISSIONS_URL.format(cik=cik))
        if not isinstance(data, dict):
            return []
        filings = (data.get("filings", {}) or {}).get("recent", {}) or {}
        forms = filings.get("form", []) or []
        dates = filings.get("filingDate", []) or []
        items_list = filings.get("items", []) or []
        accession = filings.get("accessionNumber", []) or []
        primary_doc = filings.get("primaryDocument", []) or []

        cutoff = date.today() - timedelta(days=within_days)
        out: list[dict] = []
        for i, form in enumerate(forms):
            if form != "8-K":
                continue
            if i >= len(items_list) or not items_list[i]:
                continue
            if "5.02" not in items_list[i]:
                continue
            try:
                filed = date.fromisoformat(dates[i])
            except (ValueError, IndexError):
                continue
            if filed < cutoff:
                continue
            acc_raw = accession[i] if i < len(accession) else ""
            acc_clean = acc_raw.replace("-", "") if acc_raw else ""
            primary = primary_doc[i] if i < len(primary_doc) else ""
            url_filing = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{primary}"
                if acc_clean and primary else ""
            )
            out.append({
                "filed": filed,
                "items": items_list[i],
                "accession": acc_raw,
                "url": url_filing,
            })
        return out


def _bundle_from_facts(ticker: str, facts: dict) -> EarningsBundle:
    """Parse XBRL facts JSON into quarterly/annual EPS + annual ROE, with
    normalization applied to strip one-time items (gains, impairments, etc.).

    Tries us-gaap first (domestic issuers, has quarterly data).
    Falls back to ifrs-full (foreign private issuers filing 20-F, typically annual-only).
    """
    if not isinstance(facts, dict):
        return EarningsBundle(ticker=ticker)
    all_facts = facts.get("facts", {})

    eps_entries: list[dict] = []
    ni_entries: list[dict] = []
    eq_entries: list[dict] = []
    active_taxonomy: Optional[dict] = None
    for tax in _TAXONOMIES:
        root = all_facts.get(tax["name"], {})
        if not root:
            continue
        eps_entries = _collect_concept_best_unit(root, tax["eps"])
        ni_entries = _collect_concept_best_unit(root, tax["net_income"])
        eq_entries = _collect_concept_best_unit(root, tax["equity"])
        if eps_entries:
            active_taxonomy = tax
            break

    # Quarterly EPS: entries where fp in {Q1, Q2, Q3, Q4} — Q4 is inferred below.
    # SEC convention: FY reports have fp='FY' (cumulative annual). Quarterly 10-Qs report fp Q1-Q3.
    # Key by `end` date (period ending), not `fy` (filing year) — FPIs report prior-year
    # comparatives tagged with the current filing year, which would overwrite data otherwise.
    q_eps_by_period: dict[str, float] = {}
    a_eps_by_period: dict[str, float] = {}
    for e in eps_entries:
        fp = e.get("fp")
        val = e.get("val")
        end = e.get("end")
        if fp is None or val is None or end is None:
            continue
        end_year = str(end)[:4]
        if fp == "FY":
            a_eps_by_period[end_year] = float(val)
        elif fp in ("Q1", "Q2", "Q3"):
            q_eps_by_period[f"{end_year}-{fp}"] = float(val)
        elif fp == "Q4":
            q_eps_by_period[f"{end_year}-Q4"] = float(val)

    # Derive Q4 where missing: FY − (Q1 + Q2 + Q3), if all three present
    for yr_str, fy_val in a_eps_by_period.items():
        q4_key = f"{yr_str}-Q4"
        if q4_key in q_eps_by_period:
            continue
        q1 = q_eps_by_period.get(f"{yr_str}-Q1")
        q2 = q_eps_by_period.get(f"{yr_str}-Q2")
        q3 = q_eps_by_period.get(f"{yr_str}-Q3")
        if q1 is not None and q2 is not None and q3 is not None:
            q_eps_by_period[q4_key] = fy_val - q1 - q2 - q3

    # Sort quarters newest-first
    def _q_sort_key(k: str) -> tuple[int, int]:
        fy, fp = k.split("-")
        return (int(fy), int(fp[1:]))

    q_periods = sorted(q_eps_by_period.keys(), key=_q_sort_key, reverse=True)
    q_eps = [q_eps_by_period[k] for k in q_periods]

    a_periods = sorted(a_eps_by_period.keys(), reverse=True)
    a_eps = [a_eps_by_period[k] for k in a_periods]

    # Annual ROE: latest fiscal-year net income / prior year-end equity (simple)
    ni_annual = _annual_by_fy(ni_entries)
    eq_annual = _annual_by_fy(eq_entries)
    roe_series: list[float] = []
    for fy_str in a_periods:
        fy = int(fy_str)
        ni = ni_annual.get(fy)
        eq = eq_annual.get(fy)
        if ni is not None and eq is not None and eq > 0:
            roe_series.append(ni / eq)
        else:
            roe_series.append(0.0)

    # === NORMALIZATION ==================================================
    # Apply pluggable one-time-items rules only on us-gaap filings (IFRS
    # concept names differ; would need a separate rule set). For now, we
    # pass the adjustments through when we have us-gaap data.
    adjustments: list[NormalizationAdjustment] = []
    normalized_a_eps = list(a_eps)
    normalized_q_eps = list(q_eps)
    if active_taxonomy and active_taxonomy["name"] == "us-gaap":
        usgaap_root = all_facts.get("us-gaap", {})
        # Compute shares outstanding by period so we can convert dollars → per-share
        a_shares = _shares_by_period(ni_annual, a_eps_by_period)  # annual
        q_shares = _quarterly_shares(usgaap_root, q_eps_by_period)  # quarterly
        tax_rates = _annual_tax_rates(usgaap_root)

        rules = default_rules()
        annual_adj = apply_rules(rules, usgaap_root, a_shares, tax_rates, periodicity_filter="annual")
        quarterly_adj = apply_rules(rules, usgaap_root, q_shares, tax_rates, periodicity_filter="quarterly")
        adjustments = annual_adj + quarterly_adj

        # Apply annual adjustments
        net_annual: dict[str, float] = {p: 0.0 for p in a_periods}
        for adj in annual_adj:
            if adj.period not in net_annual:
                continue
            delta = -adj.per_share_impact if adj.direction == "subtract" else adj.per_share_impact
            net_annual[adj.period] += delta
        normalized_a_eps = [round(a_eps_by_period[p] + net_annual.get(p, 0.0), 4) for p in a_periods]

        # Apply quarterly adjustments
        net_quarterly: dict[str, float] = {p: 0.0 for p in q_periods}
        for adj in quarterly_adj:
            if adj.period not in net_quarterly:
                continue
            delta = -adj.per_share_impact if adj.direction == "subtract" else adj.per_share_impact
            net_quarterly[adj.period] += delta
        normalized_q_eps = [round(q_eps_by_period[p] + net_quarterly.get(p, 0.0), 4) for p in q_periods]

    return EarningsBundle(
        ticker=ticker,
        quarterly_eps=normalized_q_eps,
        quarterly_periods=q_periods,
        annual_eps=normalized_a_eps,
        annual_periods=a_periods,
        annual_roe_pct=roe_series,
        reported_quarterly_eps=q_eps,
        reported_annual_eps=a_eps,
        normalization_adjustments=adjustments,
    )


def _shares_by_period(ni_annual: dict[int, float], a_eps_by_period: dict[str, float]) -> dict[str, float]:
    """Derive shares outstanding per annual period: NI / EPS."""
    out: dict[str, float] = {}
    for period, eps in a_eps_by_period.items():
        try:
            fy = int(period)
        except (ValueError, TypeError):
            continue
        ni = ni_annual.get(fy)
        if ni and eps and eps != 0:
            out[period] = abs(ni / eps)
    return out


def _quarterly_shares(usgaap_root: dict, q_eps_by_period: dict[str, float]) -> dict[str, float]:
    """Derive quarterly shares outstanding using WeightedAverageNumberOfDilutedSharesOutstanding if present,
    else fall back to net-income / quarterly-eps."""
    out: dict[str, float] = {}
    shares_node = usgaap_root.get("WeightedAverageNumberOfDilutedSharesOutstanding") or usgaap_root.get(
        "WeightedAverageNumberOfSharesOutstandingBasic"
    )
    if shares_node:
        units = shares_node.get("units", {})
        entries = units.get("shares") or (next(iter(units.values())) if units else [])
        for e in entries:
            fp = e.get("fp")
            end = e.get("end", "")
            val = e.get("val")
            if fp in ("Q1", "Q2", "Q3", "Q4") and end and val:
                key = f"{end[:4]}-{fp}"
                if key in q_eps_by_period:
                    out[key] = float(val)
    # Fallback using net income for any period still missing
    ni_node = usgaap_root.get("NetIncomeLoss")
    if ni_node:
        units = ni_node.get("units", {})
        entries = units.get("USD") or (next(iter(units.values())) if units else [])
        for e in entries:
            fp = e.get("fp")
            end = e.get("end", "")
            val = e.get("val")
            if fp in ("Q1", "Q2", "Q3", "Q4") and end and val:
                key = f"{end[:4]}-{fp}"
                if key in q_eps_by_period and key not in out:
                    eps = q_eps_by_period[key]
                    if eps and eps != 0:
                        out[key] = abs(val / eps)
    return out


def _annual_tax_rates(usgaap_root: dict) -> dict[str, float]:
    """Compute effective tax rate per FY = tax_expense / pretax_income. Fallback: 0.21."""
    tax_node = usgaap_root.get("IncomeTaxExpenseBenefit")
    pre_node = (
        usgaap_root.get("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest")
        or usgaap_root.get("IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments")
    )
    if not tax_node or not pre_node:
        return {}
    tax_by_fy: dict[str, float] = {}
    pre_by_fy: dict[str, float] = {}
    for e in (tax_node.get("units", {}).get("USD") or []):
        if e.get("fp") == "FY" and e.get("end"):
            tax_by_fy[e["end"][:4]] = float(e["val"])
    for e in (pre_node.get("units", {}).get("USD") or []):
        if e.get("fp") == "FY" and e.get("end"):
            pre_by_fy[e["end"][:4]] = float(e["val"])
    out: dict[str, float] = {}
    for y, pre in pre_by_fy.items():
        if pre and pre > 0:
            rate = tax_by_fy.get(y, 0.0) / pre
            if 0.0 <= rate <= 0.5:  # sanity bounds
                out[y] = rate
    return out


def _collect_concept(
    taxonomy_root: dict,
    tag_candidates: tuple[str, ...],
    prefer_units: tuple[str, ...],
) -> list[dict]:
    """Walk the first tag that exists, returning its unit entries in a flat list."""
    for tag in tag_candidates:
        node = taxonomy_root.get(tag)
        if not node:
            continue
        units = node.get("units", {})
        for u in prefer_units:
            if u in units:
                return list(units[u])
        # fallback to any unit
        for entries in units.values():
            return list(entries)
    return []


def _collect_concept_best_unit(taxonomy_root, tag_candidates: tuple[str, ...]) -> list[dict]:
    """Return entries from the unit with the most distinct end-years.

    Handles FPIs that switched reporting currency mid-life (e.g. SGML: CAD/shares for 2020-2023,
    USD/shares for 2024-2025). We prefer the longest single-currency series so growth calcs are
    apples-to-apples, rather than mixing units.
    """
    for tag in tag_candidates:
        node = taxonomy_root.get(tag)
        if not node:
            continue
        units = node.get("units", {})
        if not units:
            continue
        best_unit = None
        best_years: set[str] = set()
        for u, entries in units.items():
            years = {str(e.get("end", ""))[:4] for e in entries if e.get("end") and e.get("fp") == "FY"}
            if len(years) > len(best_years):
                best_years = years
                best_unit = u
        if best_unit is not None:
            return list(units[best_unit])
        # No FY entries — return whichever unit has the most rows
        return list(max(units.values(), key=len))
    return []


def _annual_by_fy(entries: list[dict]) -> dict[int, float]:
    """Return {fy: value} taking the FY-filed value for each fiscal year."""
    out: dict[int, float] = {}
    for e in entries:
        if e.get("fp") != "FY":
            continue
        fy = e.get("fy")
        val = e.get("val")
        if fy is None or val is None:
            continue
        try:
            out[int(fy)] = float(val)
        except (TypeError, ValueError):
            continue
    return out
