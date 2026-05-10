"""Market-context computation: VIX, breadth, distribution days, sector standings.

Called post-scan to enrich `RunManifest.market_context` with a "should I be
greedy or scared" snapshot beyond the binary M-gate uptrend/caution flag.

Inputs:
  * price_frames: dict of ticker -> DataFrame (OHLCV) for the scan universe
  * a price provider for ad-hoc fetches (^VIX + 11 SPDR sector ETFs)

Outputs:
  * MarketContext dataclass with VIX level + label, distribution days,
    breadth metrics, and per-sector RS/return.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Awaitable, Callable, Optional

import numpy as np
import pandas as pd

from canslim.models import MarketContext, SectorRow

log = logging.getLogger(__name__)


# 11 GICS sectors via SPDR ETFs. We fetch these and compute their own RS within the set.
SPDR_SECTORS = [
    ("XLK", "Technology"),
    ("XLF", "Financials"),
    ("XLE", "Energy"),
    ("XLV", "Health Care"),
    ("XLP", "Consumer Staples"),
    ("XLY", "Consumer Discretionary"),
    ("XLI", "Industrials"),
    ("XLU", "Utilities"),
    ("XLB", "Materials"),
    ("XLRE", "Real Estate"),
    ("XLC", "Communication Services"),
]


def classify_vix(vix: Optional[float]) -> Optional[str]:
    if vix is None:
        return None
    if vix < 15:
        return "complacent"
    if vix < 20:
        return "normal"
    if vix < 30:
        return "elevated"
    return "fear"


def count_distribution_days(spy: pd.DataFrame, sessions: int = 25) -> int:
    """Count down days on heavier volume than the prior session in the last N sessions.

    O'Neil's distribution-day signal: ≥5 over 4 weeks = caution, ≥7 = warning.
    """
    if spy is None or len(spy) < sessions + 1:
        return 0
    window = spy.tail(sessions + 1)
    closes = window["close"].astype(float).values
    vols = window["volume"].astype(float).values
    count = 0
    for i in range(1, len(closes)):
        # Heavier than the prior bar (rough O'Neil rule), and meaningful drop
        if vols[i] > vols[i - 1] and closes[i] < closes[i - 1] * 0.998:
            count += 1
    return count


def compute_breadth(price_frames: dict[str, pd.DataFrame]) -> tuple[Optional[float], Optional[float], int, int]:
    """Aggregate breadth across the universe.

    Returns (pct_above_50d_ma, pct_above_200d_ma, new_highs, new_lows).
    new_highs = stocks within 1% of their 52w high.
    new_lows = stocks within 1% of their 52w low.
    """
    if not price_frames:
        return None, None, 0, 0
    above_50, above_200, total_50, total_200 = 0, 0, 0, 0
    nh = nl = 0
    for ticker, df in price_frames.items():
        if df is None or len(df) < 50:
            continue
        close = df["close"].astype(float)
        last = float(close.iloc[-1])
        if len(close) >= 50:
            ma50 = float(close.tail(50).mean())
            if ma50 > 0:
                total_50 += 1
                if last > ma50:
                    above_50 += 1
        if len(close) >= 200:
            ma200 = float(close.tail(200).mean())
            if ma200 > 0:
                total_200 += 1
                if last > ma200:
                    above_200 += 1
        # 52w extremes
        win = df.tail(252)
        h52 = float(win["high"].astype(float).max())
        l52 = float(win["low"].astype(float).min())
        if h52 > 0 and last >= h52 * 0.99:
            nh += 1
        if l52 > 0 and last <= l52 * 1.01:
            nl += 1
    pct_50 = above_50 / total_50 if total_50 else None
    pct_200 = above_200 / total_200 if total_200 else None
    return pct_50, pct_200, nh, nl


def compute_sector_rows(sector_frames: dict[str, pd.DataFrame]) -> list[SectorRow]:
    """Given the 11 SPDR ETF DataFrames, compute per-sector snapshot + RS within the cohort."""
    rows: list[tuple[str, str, float, float, bool, bool]] = []
    for symbol, name in SPDR_SECTORS:
        df = sector_frames.get(symbol)
        if df is None or len(df) < 50:
            continue
        close = df["close"].astype(float)
        last = float(close.iloc[-1])
        # 30-day return
        if len(close) >= 30:
            ret_30d = (last / float(close.iloc[-30]) - 1.0) * 100
        else:
            ret_30d = 0.0
        ma50 = float(close.tail(50).mean())
        ma200 = float(close.tail(200).mean()) if len(close) >= 200 else None
        rows.append((symbol, name, last, ret_30d, last > ma50, ma200 is None or last > ma200))

    if not rows:
        return []

    # RS percentile across the 11 sectors based on 30d return
    returns = np.array([r[3] for r in rows])
    ranks = returns.argsort().argsort()  # 0 = lowest, len-1 = highest
    pcts = ranks / max(len(rows) - 1, 1)
    out: list[SectorRow] = []
    for (symbol, name, last, r30, above50, above200), pct in zip(rows, pcts):
        out.append(SectorRow(
            symbol=symbol, name=name, close=last,
            return_30d_pct=r30, rs_percentile=float(pct),
            above_50d_ma=above50, above_200d_ma=above200,
        ))
    out.sort(key=lambda r: -r.return_30d_pct)
    return out


def build_note(
    vix_label: Optional[str],
    dist_days: int,
    pct_50: Optional[float],
    nh: int,
    nl: int,
) -> str:
    """One-line stockpicker's verdict combining the quantitative signals."""
    parts = []
    if vix_label in ("fear", "elevated"):
        parts.append(f"VIX {vix_label}")
    if dist_days >= 5:
        parts.append(f"{dist_days} distribution days — caution")
    if pct_50 is not None and pct_50 < 0.30:
        parts.append("breadth weak (<30% above 50d)")
    if nh and nl and nh > 2 * nl:
        parts.append(f"new-high breadth strong ({nh}:{nl})")
    elif nl > nh:
        parts.append(f"more new lows than highs ({nl}:{nh})")
    if not parts:
        return "Backdrop supports buying breakouts; pursue strongest setups in leading sectors."
    return " · ".join(parts)


# === Async wrapper for the scanner ===

async def fetch_supplementary_prices(
    fetch_one: Callable[[str], Awaitable[Optional[pd.DataFrame]]],
    symbols: list[str],
) -> dict[str, pd.DataFrame]:
    """Fetch a list of symbols using the provided async fetcher; ignore failures."""
    import asyncio
    tasks = [fetch_one(s) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: dict[str, pd.DataFrame] = {}
    for sym, res in zip(symbols, results):
        if isinstance(res, Exception):
            continue
        if isinstance(res, pd.DataFrame) and not res.empty:
            out[sym] = res
    return out


def compose_context(
    spy: Optional[pd.DataFrame],
    vix: Optional[pd.DataFrame],
    sector_frames: dict[str, pd.DataFrame],
    price_frames: dict[str, pd.DataFrame],
    as_of: Optional[date] = None,
) -> MarketContext:
    """Assemble a MarketContext from the input frames."""
    vix_close = vix_sma50 = None
    if vix is not None and not vix.empty:
        try:
            vix_close = float(vix["close"].astype(float).iloc[-1])
            if len(vix) >= 50:
                vix_sma50 = float(vix["close"].astype(float).tail(50).mean())
        except Exception:
            pass

    dist_days = count_distribution_days(spy) if spy is not None else 0
    pct_50, pct_200, nh, nl = compute_breadth(price_frames)
    sectors = compute_sector_rows(sector_frames)

    if as_of is None:
        if spy is not None and not spy.empty:
            try:
                as_of = spy.index[-1].date()
            except Exception:
                from datetime import date as _date
                as_of = _date.today()
        else:
            from datetime import date as _date
            as_of = _date.today()

    return MarketContext(
        as_of=as_of,
        vix_close=vix_close,
        vix_sma50=vix_sma50,
        vix_label=classify_vix(vix_close),
        distribution_days_25=dist_days,
        pct_above_50d_ma=pct_50,
        pct_above_200d_ma=pct_200,
        new_highs=nh,
        new_lows=nl,
        sectors=sectors,
        note=build_note(classify_vix(vix_close), dist_days, pct_50, nh, nl),
    )
