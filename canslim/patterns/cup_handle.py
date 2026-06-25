"""Cup-with-Handle detector (O'Neil's canonical base pattern).

Pivot anchors to the FIRST recovery into the left-rim zone (not the global max),
so the buy point sits at the real base rim instead of drifting to whatever the
latest high is. Evidence carries pivot DATES (left_peak_date, cup_bottom_date,
right_peak_date, handle_start_date). numpy + pandas only. Kept in sync with the
vendored copy in stock-quickview/patterns/cup_handle.py.

Rules encoded from *How to Make Money in Stocks*:

  * Prior uptrend of ~30% before the cup forms (we relax to "price near 52w high").
  * Cup depth 12-33% from the left peak; 15-20% is classic.
  * Cup duration 7 weeks minimum (~35 sessions); rounded bottom, not V-shaped.
  * Right side of cup recovers to within ~5% of the left peak.
  * Handle: 1-2 week pullback of 5-15% on lighter volume, drifting downward.
  * Handle must be in the upper half of the cup (bullish structure).
  * Pivot = handle high + $0.10.

All thresholds are configurable via the class constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from canslim.patterns.base import ChartPattern
from canslim.models import PatternMatch


@dataclass
class CupHandleParams:
    lookback_sessions: int = 150  # ~30 weeks
    min_cup_sessions: int = 35  # 7 weeks
    min_cup_depth: float = 0.12
    max_cup_depth: float = 0.35
    max_right_side_gap: float = 0.08  # right peak must be within 8% of left peak (lower bound)
    # Upper bound: right peak can exceed left peak by up to this much. Larger values
    # accept small earnings-gap variants but quickly degrade into "stock in sustained
    # uptrend past its real base" (e.g., AMD 35% above left peak = NOT a cup).
    # 0.10 keeps the pattern conservative; setups that gapped >10% past prior peak
    # should be classified as post-earnings continuation, not cup-with-handle.
    max_right_side_overshoot: float = 0.10
    min_handle_sessions: int = 5  # 1 week
    max_handle_sessions: int = 25  # 5 weeks
    min_handle_depth: float = 0.03
    max_handle_depth: float = 0.18
    handle_upper_half_only: bool = True
    # O'Neil's "handle in the upper half of the cup" is a GUIDELINE, not a razor.
    # We require the handle low to sit at/above this fraction of the cup height
    # (bottom + frac*(left_peak - bottom)). 0.50 is the textbook midpoint; we
    # relax to 0.45 so a handle that misses the midpoint by a hair (e.g. UAL,
    # handle low $101.77 vs cup_mid $101.92 — a $0.15 miss) still passes, while a
    # handle that sags into the genuine lower third (< 45% of cup height) is
    # still rejected. Tunable so the gate can be tightened/loosened per use.
    handle_min_cup_fraction: float = 0.45
    # Opt-in: when True, an already-EXTENDED base (current price > pivot * 1.20)
    # is RETURNED instead of rejected, tagged via evidence["extended"]=True so a
    # consumer (e.g. the quickview overlay) can still draw the cup and flag that
    # the entry is gone. DEFAULT False — the canslim SCANNER must keep rejecting
    # extended cups (returns None), so this stays off for screening.
    include_extended: bool = False


class CupWithHandle(ChartPattern):
    name = "cup_with_handle"

    def __init__(self, params: Optional[CupHandleParams] = None) -> None:
        self.params = params or CupHandleParams()

    def detect(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        p = self.params
        if df is None or df.empty or len(df) < p.min_cup_sessions + p.max_handle_sessions:
            return None

        window = df.tail(p.lookback_sessions).copy()
        if len(window) < p.min_cup_sessions + p.min_handle_sessions:
            return None

        close = window["close"].astype(float).values
        high = window.get("high", window["close"]).astype(float).values
        low = window.get("low", window["close"]).astype(float).values
        volume = window["volume"].astype(float).values
        index = window.index

        # Find left peak in the first ~60% of the window
        left_cutoff = int(len(window) * 0.4)
        if left_cutoff < 5:
            return None
        left_idx = int(np.argmax(high[:left_cutoff]))
        left_peak = float(high[left_idx])

        # Cup bottom = min after left peak, excluding the trailing handle window
        bottom_search_end = len(window) - p.min_handle_sessions
        if bottom_search_end <= left_idx + p.min_cup_sessions // 2:
            return None
        bottom_region = low[left_idx:bottom_search_end]
        if bottom_region.size == 0:
            return None
        bottom_rel = int(np.argmin(bottom_region))
        bottom_idx = left_idx + bottom_rel
        bottom = float(low[bottom_idx])

        depth = (left_peak - bottom) / left_peak if left_peak > 0 else 0.0
        if not (p.min_cup_depth <= depth <= p.max_cup_depth):
            return None

        # Cup must be rounded: reject V-shape by requiring at least N sessions in the trough
        cup_duration = bottom_idx - left_idx
        if cup_duration < p.min_cup_sessions // 2:
            return None

        # --- RIGHT RIM: first recovery into the left-rim zone (NOT the global max) ---
        # The naive "argmax(high) on the right side" grabs whatever the highest bar
        # is — for a base that has since broken out, that is the BREAKOUT bar, not
        # the rim. Instead we locate the rim the way a human reads the chart:
        #   1. Scan FORWARD from the cup bottom for the FIRST bar whose high enters
        #      the left-rim zone: within `max_right_side_gap` below the left peak
        #      (lower bound) and not more than `max_right_side_overshoot` above it.
        #   2. From that entry, the rim is the FIRST recovery PEAK — walk forward
        #      tracking the running max of the high; the rim locks once a genuine
        #      handle-sized pullback follows it (a bar whose high sits
        #      >= `min_handle_depth` below the running max), or once price runs away
        #      above the overshoot ceiling. This gives the leftmost true rim, so the
        #      handle that follows is the real pullback — not a later breakout leg.
        rim_lo = left_peak * (1.0 - p.max_right_side_gap)
        rim_hi = left_peak * (1.0 + p.max_right_side_overshoot)

        entry_idx = None
        for i in range(bottom_idx + 1, len(window)):
            if rim_lo <= high[i] <= rim_hi:
                entry_idx = i
                break
        # No recovery into the rim zone at all => stock never made it back => no cup.
        if entry_idx is None:
            return None

        right_idx = entry_idx
        right_peak = float(high[entry_idx])
        for i in range(entry_idx + 1, len(window)):
            if high[i] > rim_hi:
                # Runaway breakout above the rim zone before any handle formed.
                break
            if high[i] >= right_peak:
                right_peak = float(high[i])
                right_idx = i  # rim advances with each fresh recovery high
            elif (right_peak - high[i]) / right_peak >= p.min_handle_depth:
                # A genuine pullback has begun => the rim is locked at right_idx.
                break

        # Right side must recover close to the left peak (lower bound) but may exceed
        # it by up to `max_right_side_overshoot`. The zone scan above already enforces
        # this, but recompute the canonical gap for the confidence/recovery score.
        recovery_gap = (left_peak - right_peak) / left_peak if left_peak > 0 else 1.0
        if recovery_gap > p.max_right_side_gap or recovery_gap < -p.max_right_side_overshoot:
            return None

        # Total cup width must be sufficient
        if (right_idx - left_idx) < p.min_cup_sessions:
            return None

        # --- HANDLE: the pullback that IMMEDIATELY FOLLOWS the right rim ---
        # The handle window runs from the bar after the rim up to (but not into) the
        # bar where price reclaims the rim high (the breakout), capped at
        # `max_handle_sessions`. Existing handle validation (depth, upper-half,
        # duration) applies to THIS window — not "everything to the series end".
        reclaim_idx = len(window)
        for i in range(right_idx + 1, len(window)):
            if high[i] > right_peak:
                reclaim_idx = i
                break
        handle_end_idx = min(reclaim_idx - 1, right_idx + p.max_handle_sessions)
        if handle_end_idx <= right_idx:
            return None

        # Window spanning the rim through the end of the handle (inclusive).
        handle_region = window.iloc[right_idx : handle_end_idx + 1]
        handle_duration = handle_end_idx - right_idx
        if not (p.min_handle_sessions <= handle_duration <= p.max_handle_sessions):
            return None

        handle_low = float(handle_region["low"].min() if "low" in handle_region else handle_region["close"].min())
        handle_depth = (right_peak - handle_low) / right_peak if right_peak > 0 else 0.0
        if not (p.min_handle_depth <= handle_depth <= p.max_handle_depth):
            return None

        # Handle should sit in the upper portion of the cup. TOLERANCE band, not a
        # razor: the threshold is bottom + handle_min_cup_fraction * cup_height
        # (default 0.45 ~= "upper 55%"), so a handle that misses the strict 0.50
        # midpoint by a hair still passes, but one that sags into the genuine
        # lower third is still rejected.
        cup_mid = bottom + 0.5 * (left_peak - bottom)  # retained for reference/scoring
        handle_floor = bottom + p.handle_min_cup_fraction * (left_peak - bottom)
        if p.handle_upper_half_only and handle_low < handle_floor:
            return None

        # Volume check: handle volume should be lighter than cup-advance volume
        cup_advance_vol = float(np.mean(volume[bottom_idx:right_idx]) or 0.0) if right_idx > bottom_idx else 0.0
        handle_vol = float(handle_region["volume"].mean() or 0.0)
        light_handle_volume = handle_vol <= cup_advance_vol * 1.1

        # Pivot = handle's highest high + small buffer (now anchored to the CORRECT
        # handle window, so it sits at the real rim, not a later breakout bar).
        handle_high = float(handle_region["high"].max() if "high" in handle_region else handle_region["close"].max())
        pivot = handle_high + 0.10
        # "Stale"/extended cup-with-handle: price already >20% past pivot — the base
        # broke out and ran, so the buy point is gone. Now that the rim/handle anchors
        # are correct, this guard operates on the true pivot.
        #   * include_extended False (DEFAULT) -> reject (return None). Scanner path.
        #   * include_extended True            -> return the match, tagged extended,
        #     so the overlay can still draw the (historical) cup.
        last_close_now = float(close[-1])
        extension_pct = (last_close_now - pivot) / pivot if pivot > 0 else 0.0
        extended = pivot > 0 and last_close_now > pivot * 1.20
        if extended and not p.include_extended:
            return None

        # Confidence blends: depth in classic range, handle depth, volume quality, right-side recovery
        ideal_depth = 0.20
        depth_score = 1.0 - min(1.0, abs(depth - ideal_depth) / 0.20)
        handle_score = 1.0 - min(1.0, abs(handle_depth - 0.08) / 0.12)
        vol_score = 1.0 if light_handle_volume else 0.5
        recovery_score = 1.0 - min(1.0, max(0.0, recovery_gap) / p.max_right_side_gap)
        confidence = float(max(0.0, min(1.0, 0.35 * depth_score + 0.25 * handle_score + 0.2 * vol_score + 0.2 * recovery_score)))

        started_on = _as_date(index[left_idx])
        completed_on = _as_date(index[-1])

        # Tier-3 overlay needs the pivot DATES, not just the prices. The indices
        # already exist above — map them through `index[...]` to ISO strings.
        # handle_start == right_peak bar (the handle begins at the right peak).
        # handle_low / handle_end let the overlay draw the handle as ONE short
        # segment from the rim down to the real handle low (the handle no longer
        # runs to the series end now that the rim is the May recovery, not the
        # latest breakout bar).
        handle_low_idx = int(handle_region["low"].values.argmin()) + right_idx if "low" in handle_region else right_idx
        left_peak_date = _as_iso(index[left_idx])
        cup_bottom_date = _as_iso(index[bottom_idx])
        right_peak_date = _as_iso(index[right_idx])
        handle_start_date = right_peak_date
        handle_low_date = _as_iso(index[handle_low_idx])
        handle_end_date = _as_iso(index[handle_end_idx])

        return PatternMatch(
            name=self.name,
            detected=True,
            pivot=round(pivot, 2),
            confidence=round(confidence, 3),
            started_on=started_on,
            completed_on=completed_on,
            evidence={
                "left_peak": round(left_peak, 2),
                "cup_bottom": round(bottom, 2),
                "right_peak": round(right_peak, 2),
                "cup_depth_pct": round(depth, 4),
                "cup_duration_sessions": int(right_idx - left_idx),
                "handle_duration_sessions": int(handle_duration),
                "handle_depth_pct": round(handle_depth, 4),
                "handle_low": round(handle_low, 2),
                "handle_high": round(handle_high, 2),
                "handle_volume_over_cup": round(handle_vol / cup_advance_vol, 3) if cup_advance_vol else None,
                "light_handle_volume": light_handle_volume,
                "current_close": round(float(close[-1]), 2),
                "dist_to_pivot_pct": round((pivot - float(close[-1])) / pivot, 4) if pivot else None,
                # Extended flag: True when price has already run >20% past pivot
                # (only reachable when include_extended=True). extension_pct is the
                # signed distance (last_close - pivot)/pivot — negative/small for a
                # live base, ~+0.23 for an extended one.
                "extended": bool(extended),
                "extension_pct": round(extension_pct, 4),
                # Pivot DATES (ISO) for Tier-3 overlay drawing.
                "left_peak_date": left_peak_date,
                "cup_bottom_date": cup_bottom_date,
                "right_peak_date": right_peak_date,
                "handle_start_date": handle_start_date,
                "handle_low_date": handle_low_date,
                "handle_end_date": handle_end_date,
            },
        )


def _as_date(idx_value) -> Optional[date]:
    try:
        return idx_value.date() if hasattr(idx_value, "date") else None
    except Exception:
        return None


def _as_iso(idx_value) -> Optional[str]:
    d = _as_date(idx_value)
    return d.isoformat() if d is not None else None
