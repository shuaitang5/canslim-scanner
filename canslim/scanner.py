from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from canslim.config import Settings
from canslim.criteria import discover
from canslim.criteria.base import CriterionContext
from canslim.models import (
    CriterionResult,
    EarningsBundle,
    FetchError,
    FetchSummary,
    ManagementEvent,
    MarketRegime,
    PatternMatch,
    PriceFeatures,
    RunManifest,
    ScanResult,
)
from canslim.accumulation import compute_ad_rating
from canslim.patterns import default_patterns, detect_all
from canslim.providers.base import BudgetExhausted, DataProvider, ProviderError
from canslim.providers.cache import CacheStore
from canslim.providers.fmp_provider import FMPProvider
from canslim.providers.sec_provider import SECProvider
from canslim.providers.yfinance_provider import YFinanceProvider

log = logging.getLogger(__name__)

class Scanner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache = CacheStore(settings.cache.root)
        self.yf = self._build_yfinance()
        self.fmp = self._build_fmp()
        self.sec = self._build_sec()
        self.criteria = discover()
        self.chart_patterns = default_patterns()
        self.market_index = settings.scanner.market_index

    def _build_yfinance(self) -> YFinanceProvider:
        cfg = self.settings.providers.get("yfinance")
        if cfg is None:
            from canslim.config import ProviderConfig
            cfg = ProviderConfig()
        return YFinanceProvider(cfg, self.settings.cache, self.cache)

    def _build_fmp(self) -> Optional[FMPProvider]:
        cfg = self.settings.providers.get("fmp")
        if cfg is None or not cfg.enabled:
            return None
        return FMPProvider(cfg, self.settings.cache, self.cache)

    def _build_sec(self) -> Optional[SECProvider]:
        cfg = self.settings.providers.get("sec")
        if cfg is None or not cfg.enabled:
            return None
        return SECProvider(cfg, self.settings.cache, self.cache)

    # ---- public entry

    async def health_check(self) -> dict[str, dict[str, str]]:
        report: dict[str, dict[str, str]] = {}
        yf_task = asyncio.create_task(self.yf.health_check())
        fmp_task = asyncio.create_task(self.fmp.health_check()) if self.fmp else None
        sec_task = asyncio.create_task(self.sec.health_check()) if self.sec else None
        try:
            report["yfinance"] = await yf_task
        except Exception as e:
            report["yfinance"] = {"status": "error", "error": str(e)}
        if fmp_task is not None:
            try:
                report["fmp"] = await fmp_task
            except Exception as e:
                report["fmp"] = {"status": "error", "error": str(e)}
        else:
            report["fmp"] = {"status": "disabled"}
        if sec_task is not None:
            try:
                report["sec"] = await sec_task
            except Exception as e:
                report["sec"] = {"status": "error", "error": str(e)}
        else:
            report["sec"] = {"status": "disabled"}
        return report

    async def close(self) -> None:
        await self.yf.close()
        if self.fmp:
            await self.fmp.close()
        if self.sec:
            await self.sec.close()

    async def scan(
        self, tickers: list[str], dry_run: bool = False, force_refresh: bool = False
    ) -> tuple[list[ScanResult], RunManifest]:
        started = datetime.now(timezone.utc)
        today = date.today()
        run_id = started.strftime("%Y-%m-%d_%H%M%S")

        regime = await self.evaluate_market_regime()

        # Stage 1: batch price download + pre-filter
        log.info("Stage 1: downloading prices for %d tickers", len(tickers))
        run_errors: list[FetchError] = []
        price_frames = await self.yf.get_prices(tickers + [self.market_index], force_refresh=force_refresh)
        market_df = price_frames.pop(self.market_index, None)
        features_by_ticker = self._compute_price_features(price_frames, market_df, as_of=today)
        # Keep price frames around so per-candidate pattern detection can reuse them
        self._price_frames = price_frames
        # Capture per-ticker price-fetch misses so the report surfaces them
        price_miss_reasons: dict[str, str] = {}
        for t, reason in getattr(self.yf, "last_missing", []):
            if t == self.market_index:
                continue
            price_miss_reasons[t] = reason
            run_errors.append(FetchError(ticker=t, kind="prices", provider=self.yf.name, error=reason))

        # Cross-sectional RS percentile over pre-filtered set
        prefiltered = self._prefilter(features_by_ticker)
        rs_percentiles = self._rs_percentiles({t: f for t, f in features_by_ticker.items() if t in prefiltered})

        candidates = sorted(prefiltered, key=lambda t: -rs_percentiles.get(t, 0.0))
        log.info("Pre-filter: %d -> %d candidates", len(tickers), len(candidates))

        if dry_run:
            fmp_needed = len(candidates) * 3  # fundamentals (3 calls) + 1 institutional = ~4
            budget_remaining = self.fmp.budget_remaining() if self.fmp else None
            log.info(
                "[dry-run] candidates=%d fmp_needed~%d fmp_budget_remaining=%s",
                len(candidates), fmp_needed, budget_remaining,
            )
            manifest = RunManifest(
                run_id=run_id,
                started_at=started,
                finished_at=datetime.now(timezone.utc),
                universe_name="<dry-run>",
                universe_size=len(tickers),
                candidates_after_prefilter=len(candidates),
                matches=0,
                scanned=0,
                pending_budget=0,
                errored=0,
                config_hash=self.settings.config_hash(),
                provider_versions=self._provider_versions(),
                fmp_budget_used=0,
                fmp_budget_remaining=budget_remaining,
                market_regime=regime,
            )
            return [], manifest

        # Stage 2: per-candidate fundamentals + institutional + criteria
        results: list[ScanResult] = []
        sem = asyncio.Semaphore(self.settings.scanner.max_workers)

        fmp_used_start = self._fmp_used_today()

        async def process(t: str) -> ScanResult:
            async with sem:
                return await self._evaluate_one(
                    t,
                    features_by_ticker.get(t),
                    rs_percentiles.get(t),
                    regime,
                    today,
                )

        tasks = [asyncio.create_task(process(t)) for t in candidates]
        for fut in asyncio.as_completed(tasks):
            try:
                res = await fut
            except Exception as e:
                log.warning("Unexpected error in worker: %s", e)
                continue
            results.append(res)

        # Add skipped tickers with explicit reasons
        scanned_set = {r.ticker for r in results}
        for t in tickers:
            if t in scanned_set or t in candidates:
                continue
            reason = price_miss_reasons.get(t)
            if not reason:
                if t not in features_by_ticker:
                    reason = "insufficient price history (<60 sessions)"
                else:
                    reason = "filtered out by pre-filter thresholds"
            results.append(ScanResult(
                ticker=t, as_of=today, passed=False, composite_score=0.0,
                status="skipped_missing_data",
                status_reason=reason,
            ))

        # Enrich top-candidates with SEC 8-K Item 5.02 (management-change) events.
        # Runs AFTER the main loop so it touches only the ~60-100 interesting tickers,
        # not all 2000+ scanned. ~1 SEC call per candidate.
        await self._enrich_management_events(results)

        matches = sum(1 for r in results if r.passed)
        pending = sum(1 for r in results if r.status == "pending_budget")
        errored = sum(1 for r in results if r.status == "error")
        scanned = sum(1 for r in results if r.status == "scanned")
        fmp_budget_remaining = self.fmp.budget_remaining() if self.fmp else None

        # Aggregate per-ticker errors onto the run
        for r in results:
            run_errors.extend(r.errors)

        price_stats = getattr(self.yf, "last_fetch_stats", {}) or {}
        fetch_summary = [
            FetchSummary(
                kind="prices",
                cache_hits=int(price_stats.get("cache_hits", 0)),
                fresh_fetches=int(price_stats.get("fresh_fetches", 0)),
                failures=int(price_stats.get("failures", 0)),
                skipped_negative_cache=int(price_stats.get("skipped_negative", 0)),
            )
        ]

        # Market context: VIX, sector standings, distribution days, breadth
        market_context = await self._compute_market_context(market_df, price_frames, force_refresh)

        manifest = RunManifest(
            run_id=run_id,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
            universe_name="",  # CLI fills this in
            universe_size=len(tickers),
            candidates_after_prefilter=len(candidates),
            matches=matches,
            scanned=scanned,
            pending_budget=pending,
            errored=errored,
            config_hash=self.settings.config_hash(),
            provider_versions=self._provider_versions(),
            fmp_budget_used=max(0, self._fmp_used_today() - fmp_used_start),
            fmp_budget_remaining=fmp_budget_remaining,
            market_regime=regime,
            market_context=market_context,
            fetch_summary=fetch_summary,
            errors=run_errors,
        )
        return results, manifest

    async def _compute_market_context(self, spy_df, price_frames, force_refresh: bool):
        """Fetch ^VIX + 11 SPDRs, then compose MarketContext. Best-effort.

        Side effect: stashes the SPY + VIX dataframes onto self for the report
        renderer to use for market overview charts.
        """
        try:
            from canslim.market_context import compose_context, SPDR_SECTORS
            sector_symbols = [s for s, _ in SPDR_SECTORS]
            extras = await self.yf.get_prices(["^VIX"] + sector_symbols, force_refresh=force_refresh)
            vix_df = extras.get("^VIX")
            sector_frames = {s: extras[s] for s, _ in SPDR_SECTORS if s in extras}
            # Stash for the report renderer
            self._market_overview_frames = {self.market_index: spy_df, "^VIX": vix_df}
            return compose_context(spy_df, vix_df, sector_frames, price_frames)
        except Exception as e:  # pragma: no cover — defensive
            log.debug("market_context computation failed: %s", e)
            return None

    # ---- market regime (M)

    async def evaluate_market_regime(self) -> MarketRegime:
        today = date.today()
        frames = await self.yf.get_prices([self.market_index])
        df = frames.get(self.market_index)
        if df is None or df.empty or len(df) < 200:
            return MarketRegime(
                as_of=today, spy_close=0, spy_sma50=0, spy_sma200=0, uptrend=False,
                reason="insufficient SPY history",
            )
        close = df["close"].astype(float)
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1]
        last = close.iloc[-1]
        uptrend = bool(last > sma200 and sma50 > sma200)
        return MarketRegime(
            as_of=df.index[-1].date() if hasattr(df.index[-1], "date") else today,
            spy_close=float(last),
            spy_sma50=float(sma50),
            spy_sma200=float(sma200),
            uptrend=uptrend,
            reason=("SPY above 200d and 50d>200d" if uptrend else "SPY in downtrend or 50d<=200d"),
        )

    # ---- features + pre-filter

    def _compute_price_features(
        self,
        frames: dict[str, pd.DataFrame],
        market_df: Optional[pd.DataFrame],
        as_of: date,
    ) -> dict[str, PriceFeatures]:
        out: dict[str, PriceFeatures] = {}
        for ticker, df in frames.items():
            if df is None or df.empty or len(df) < 60:
                continue
            try:
                close = df["close"].astype(float)
                vol = df["volume"].astype(float)
                last_close = float(close.iloc[-1])
                if not np.isfinite(last_close) or last_close <= 0:
                    continue
                last_252 = close.tail(252)
                high_52w = float(last_252.max())
                low_52w = float(last_252.min())
                dist = max(0.0, (high_52w - last_close) / high_52w) if high_52w > 0 else 1.0

                adv50 = float((close.tail(50) * vol.tail(50)).mean())
                adv10 = float((close.tail(10) * vol.tail(10)).mean())
                avg_vol50 = float(vol.tail(50).mean())
                recent_vol_ratio = float(vol.iloc[-1] / avg_vol50) if avg_vol50 > 0 else 0.0

                rs_weighted = _weighted_12m_return(close)

                out[ticker] = PriceFeatures(
                    ticker=ticker,
                    as_of=as_of,
                    close=last_close,
                    high_52w=high_52w,
                    low_52w=low_52w,
                    adv10=adv10,
                    adv50=adv50,
                    avg_vol50=avg_vol50,
                    recent_vol_ratio=recent_vol_ratio,
                    rs_return_12m_weighted=rs_weighted,
                    dist_to_52w_high_pct=dist,
                )
            except Exception as e:  # pragma: no cover
                log.debug("feature calc failed for %s: %s", ticker, e)
                continue
        return out

    def _prefilter(self, feats: dict[str, PriceFeatures]) -> set[str]:
        th = self.settings.criteria
        keep: set[str] = set()
        for t, f in feats.items():
            if f.close < th.prefilter_min_price:
                continue
            if f.adv50 < th.prefilter_min_adv50_usd:
                continue
            if f.dist_to_52w_high_pct > th.prefilter_max_dist_to_52w_high:
                continue
            keep.add(t)
        return keep

    def _rs_percentiles(self, feats: dict[str, PriceFeatures]) -> dict[str, float]:
        if not feats:
            return {}
        tickers = list(feats.keys())
        values = np.array([feats[t].rs_return_12m_weighted for t in tickers])
        if len(values) == 1:
            return {tickers[0]: 1.0}
        ranks = values.argsort().argsort()  # 0..n-1
        pct = ranks / (len(values) - 1)
        return {t: float(pct[i]) for i, t in enumerate(tickers)}

    # ---- per-ticker evaluation

    async def _evaluate_one(
        self,
        ticker: str,
        pf: Optional[PriceFeatures],
        rs_pct: Optional[float],
        regime: MarketRegime,
        as_of: date,
    ) -> ScanResult:
        # Fetch fundamentals + institutional + float + market cap (concurrent).
        # Float and market cap both come from the SAME cached yfinance info blob
        # (one fast_info call), so adding the cap fetch costs no extra network I/O.
        fundamentals_task = self._get_fundamentals(ticker)
        institutional_task = self._get_institutional(ticker)
        float_task = self.yf.get_shares_float(ticker)
        mcap_task = self.yf.get_market_cap(ticker)
        try:
            eb, inst, fshares, mcap = await asyncio.gather(
                fundamentals_task, institutional_task, float_task, mcap_task,
                return_exceptions=True,
            )
        except Exception as e:
            return ScanResult(
                ticker=ticker, as_of=as_of, passed=False, composite_score=0.0,
                status="error", error=str(e),
            )

        # Handle budget-exhausted gracefully
        if isinstance(eb, BudgetExhausted) or isinstance(inst, BudgetExhausted):
            return ScanResult(
                ticker=ticker, as_of=as_of, passed=False, composite_score=0.0,
                status="pending_budget",
                error="FMP daily budget exhausted",
            )
        if isinstance(eb, Exception):
            eb = None
        if isinstance(inst, Exception):
            inst = None
        if isinstance(fshares, Exception):
            fshares = None
        if isinstance(mcap, Exception):
            mcap = None
        market_cap = float(mcap) if isinstance(mcap, (int, float)) else None

        if not isinstance(eb, EarningsBundle) and eb is not None:
            eb = None

        # Early market-cap gate: enforce the $1B floor right after the cheap info
        # fetch, BEFORE running any criteria. The floor is the chairman's #1 hard
        # requirement ("no stocks under $1B") on an OUTWARD-FACING page, so it
        # FAILS CLOSED — a name can only clear the gate when its cap is KNOWN and
        # at/above the floor.
        #   - cap KNOWN, < floor   -> reject  (status="rejected_market_cap")
        #   - cap UNKNOWN/None      -> exclude (status="unknown_market_cap"): we
        #     can't prove it's >=$1B, so it must NOT publish as a match. It is set
        #     aside in its own "needs review" bucket (not silently dropped), but
        #     passed stays False so it never reaches the public Matches section.
        #   - cap KNOWN, >= floor   -> pass through to the criteria below.
        # Missing-cap is most common for thin/illiquid small caps — exactly the
        # names the floor exists to exclude — so abstaining here would fail OPEN.
        min_cap = self.settings.criteria.prefilter_min_market_cap_usd
        if min_cap > 0:
            if market_cap is None:
                return ScanResult(
                    ticker=ticker, as_of=as_of, passed=False, composite_score=0.0,
                    market_cap=None,
                    status="unknown_market_cap",
                    status_reason=(
                        f"market cap unavailable; cannot confirm >= ${min_cap/1e9:.2f}B "
                        f"floor (fail-closed, set aside for review)"
                    ),
                )
            if market_cap < min_cap:
                return ScanResult(
                    ticker=ticker, as_of=as_of, passed=False, composite_score=0.0,
                    market_cap=market_cap,
                    status="rejected_market_cap",
                    status_reason=(
                        f"market cap ${market_cap/1e9:.2f}B below ${min_cap/1e9:.2f}B floor"
                    ),
                )

        # Detect patterns up-front so the A criterion can use them for the
        # leadership-turnaround override.
        patterns, pattern_errors = self._detect_patterns(ticker)
        per_ticker_errors: list[FetchError] = list(pattern_errors)

        ctx = CriterionContext(
            ticker=ticker,
            thresholds=self.settings.criteria,
            price_features=pf,
            earnings=eb if isinstance(eb, EarningsBundle) else None,
            institutional=inst if inst is not None and not isinstance(inst, Exception) else None,
            float_shares=fshares if isinstance(fshares, float) else (float(fshares) if fshares else None),
            market_cap=market_cap,
            rs_percentile=rs_pct,
            patterns=patterns,
        )

        criteria_results: dict[str, CriterionResult] = {}
        gate_pass_all = True
        composite = 0.0
        weights = self.settings.weights.model_dump()
        total_w = 0.0

        for letter, crit in self.criteria.items():
            if letter == "m":
                # Fill market regime onto per-result criteria
                res = CriterionResult(
                    letter="M",
                    passed=regime.uptrend,
                    is_gate=False,
                    score=1.0 if regime.uptrend else 0.0,
                    evidence={
                        "spy_close": regime.spy_close,
                        "spy_sma50": regime.spy_sma50,
                        "spy_sma200": regime.spy_sma200,
                        "as_of": regime.as_of.isoformat(),
                    },
                    reason=regime.reason,
                )
            else:
                try:
                    res = crit.evaluate(ctx)
                except Exception as e:
                    log.debug("criterion %s failed for %s: %s", letter, ticker, e)
                    res = CriterionResult(
                        letter=letter.upper(), passed=False, is_gate=crit.is_gate,
                        reason=f"error: {e}", data_available=False,
                    )
            criteria_results[letter.upper()] = res
            w = float(weights.get(letter, 0.0))
            # Only criteria that actually evaluated against real data contribute
            # to the composite score. Otherwise a single fetch failure (e.g.,
            # institutional data missing for VICR) would drag a perfect-1.00
            # score down to 0.79 and silently drop the ticker from rankings.
            if w > 0 and res.data_available:
                composite += res.score * w
                total_w += w
            # Gate logic: a gate that couldn't be evaluated is an "abstain", not
            # a "fail". The ticker can't be a full match (we don't know), but it
            # shouldn't fall into "near miss with N gates failed" — it goes into
            # a separate incomplete-data bucket via abstained_gates.
            if res.is_gate and not res.passed and res.data_available:
                gate_pass_all = False

        # Count abstained gates — used by downstream report bucketing
        abstained_gates = [L for L, r in criteria_results.items() if r.is_gate and not r.data_available]
        if abstained_gates:
            # If any gate abstained we can't claim full match (don't know enough),
            # but we shouldn't treat as failed either. Mark gate_pass_all False
            # so it doesn't surface as "MATCH"; the report will distinguish
            # abstain vs fail via the criterion's data_available flag.
            gate_pass_all = False

        composite = composite / total_w if total_w else 0.0
        ad = self._compute_ad(ticker)
        # Surface fundamentals/institutional/float errors onto the result
        if not isinstance(eb, EarningsBundle) or not (eb.quarterly_eps or eb.annual_eps):
            per_ticker_errors.append(FetchError(
                ticker=ticker, kind="fundamentals", provider="chain",
                error="no fundamentals returned by SEC/FMP/yfinance chain",
                retryable=True,
            ))
        if inst is None:
            per_ticker_errors.append(FetchError(
                ticker=ticker, kind="institutional", provider="chain",
                error="no institutional snapshot from any provider",
                retryable=True,
            ))

        return ScanResult(
            ticker=ticker,
            as_of=as_of,
            passed=gate_pass_all,
            composite_score=round(composite, 4),
            criteria=criteria_results,
            patterns=patterns,
            errors=per_ticker_errors,
            ad_grade=ad.grade if ad else None,
            ad_ratio=ad.ratio if ad else None,
            market_cap=market_cap,
            status="scanned",
        )

    async def _enrich_management_events(self, results: list[ScanResult]) -> None:
        """For the top candidates (full matches + pass-4/5 + high-quality buyable zone),
        fetch recent 8-K Item 5.02 (management change) filings and attach to the result.
        """
        if self.sec is None:
            return

        targets = [r for r in results if r.status == "scanned" and _is_top_candidate(r)]
        if not targets:
            return

        log.info("Enrichment: fetching SEC 8-K Item 5.02 for %d top candidates", len(targets))
        sem = asyncio.Semaphore(5)  # match SEC rate-limit floor

        async def fetch_one(r: ScanResult) -> None:
            try:
                async with sem:
                    events_raw = await self.sec.get_recent_management_events(r.ticker)
            except Exception as e:
                log.debug("8-K fetch failed for %s: %s", r.ticker, e)
                return
            if not events_raw:
                return
            events = [ManagementEvent(**e) for e in events_raw]
            # ScanResult isn't frozen; mutate the list directly
            r.management_events_90d = events

        await asyncio.gather(*[fetch_one(r) for r in targets], return_exceptions=True)

    def _compute_ad(self, ticker: str):
        frames = getattr(self, "_price_frames", None)
        if not frames:
            return None
        df = frames.get(ticker)
        if df is None or df.empty:
            return None
        return compute_ad_rating(df)

    def _detect_patterns(self, ticker: str) -> tuple[list[PatternMatch], list[FetchError]]:
        frames = getattr(self, "_price_frames", None)
        if not frames:
            return [], []
        df = frames.get(ticker)
        if df is None or df.empty:
            return [], []
        return detect_all(self.chart_patterns, df, ticker=ticker)

    async def _get_fundamentals(self, ticker: str) -> EarningsBundle:
        # Priority: SEC (free, authoritative) -> FMP (paid, clean) -> yfinance (scraped fallback)
        if self.sec is not None:
            try:
                bundle = await self.sec.get_fundamentals(ticker)
                if bundle.quarterly_eps or bundle.annual_eps:
                    return bundle
            except ProviderError:
                pass
        if self.fmp is not None:
            try:
                return await self.fmp.get_fundamentals(ticker)
            except BudgetExhausted:
                raise
            except ProviderError:
                pass
        return await self.yf.get_fundamentals(ticker)

    async def _get_institutional(self, ticker: str):
        if self.fmp is not None:
            try:
                snap = await self.fmp.get_institutional(ticker)
                if snap is not None:
                    return snap
            except BudgetExhausted:
                raise
            except ProviderError:
                pass
        return await self.yf.get_institutional(ticker)

    def _fmp_used_today(self) -> int:
        if self.fmp is None:
            return 0
        meta = self.cache.read_meta("fmp_budget")
        if meta.get("date") != date.today().isoformat():
            return 0
        return int(meta.get("used", 0))

    def _provider_versions(self) -> dict[str, str]:
        out: dict[str, str] = {}
        try:
            import yfinance
            out["yfinance"] = yfinance.__version__
        except Exception:
            pass
        try:
            import httpx
            out["httpx"] = httpx.__version__
        except Exception:
            pass
        return out


def _is_top_candidate(r: ScanResult) -> bool:
    """Same inclusion rule used by deepdive.py — keeps the enrichment scope tight."""
    if r.passed:
        return True
    gate_results = [cr for cr in r.criteria.values() if cr.is_gate]
    gate_passes = sum(1 for cr in gate_results if cr.passed)
    if gate_results and gate_passes >= len(gate_results) - 1:
        return True
    if r.composite_score < 0.75 or gate_passes < 3:
        return False
    for p in r.patterns:
        if p.confidence < 0.55:
            continue
        dist = p.evidence.get("dist_to_pivot_pct")
        if isinstance(dist, (int, float)) and -0.05 <= dist <= 0.05:
            return True
    return False


def _weighted_12m_return(close: pd.Series) -> float:
    """Weighted 40/20/20/20 return over the 4 trailing quarters (most recent first)."""
    if len(close) < 252:
        if len(close) < 20:
            return 0.0
        return float(close.iloc[-1] / close.iloc[0] - 1.0)
    quarters = [close.iloc[-63:], close.iloc[-126:-63], close.iloc[-189:-126], close.iloc[-252:-189]]
    returns = [float(q.iloc[-1] / q.iloc[0] - 1.0) for q in quarters if len(q) > 1 and q.iloc[0] > 0]
    weights = [0.4, 0.2, 0.2, 0.2][: len(returns)]
    if not returns:
        return 0.0
    total = sum(r * w for r, w in zip(returns, weights))
    return total
