# vendored from canslim-scanner/canslim/patterns/double_bottom.py on 2026-06-24; DO NOT import canslim at runtime
"""Double-Bottom ("W") detector.

Vendored from canslim-scanner/canslim/patterns/double_bottom.py. Two LOCAL
improvements over the prior canslim logic:

  FIX 1 — "second low = last bar" flaw. The source sets second_low to the
  global argmin of low[second_search_start:], which on a stock making FRESH
  lows grabs the LAST bar (an active decline mislabeled as a completed W). We
  now require the second low to be a CONFIRMED SWING LOW: at least
  MIN_BARS_AFTER_SECOND_LOW bars must follow it AND price must have recovered
  off it (post-low high exceeds the low by min_recovery_off_second_low, and the
  last close sits above the low). If the "second low" hugs the last bar with no
  recovery, the pattern is INCOMPLETE -> return None.

  FIX 2 — emit anchor DATES. The source emits only prices + start/end dates. We
  add first_low_date / middle_peak_date / second_low_date (ISO strings, mapped
  through index[...]) so the quickview frontend can anchor the drawn W to real
  bars. Mirrors how cup_handle.py emits left_peak_date etc.

Rules (O'Neil):
  * Two distinct lows separated by ~7 weeks minimum (35 sessions).
  * Middle peak rises 5-15% from the lows.
  * Second low is at or below first low (undercut is ideal).
  * Pattern forms after a prior decline of 8%+ (or near a 52-week high recovery).
  * Pivot = middle peak high + $0.10.

numpy + pandas only.
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
class DoubleBottomParams:
    lookback_sessions: int = 120
    min_separation_sessions: int = 30  # ~6 weeks
    max_low_mismatch_pct: float = 0.05  # second low within 5% of first
    min_middle_peak_rise: float = 0.05
    max_middle_peak_rise: float = 0.20
    require_second_undercut: bool = False  # strict O'Neil prefers this; off by default for more hits
    # --- FIX 1 (quickview only): require the second low to be a CONFIRMED swing
    # low with a recovery, so an active fresh-low decline is NOT mislabeled as a
    # completed W. Both gates are tunable.
    min_bars_after_second_low: int = 5  # >= this many bars must follow second low
    min_recovery_off_second_low: float = 0.03  # post-low high must exceed low by >= 3%
    # Outer-leg lookback: how many bars before the first low to search for the
    # left-top (the high price fell from into the W). Descriptive only.
    outer_leg_lookback: int = 45


class DoubleBottom(ChartPattern):
    name = "double_bottom"

    def __init__(self, params: Optional[DoubleBottomParams] = None) -> None:
        self.params = params or DoubleBottomParams()

    def detect(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        p = self.params
        if df is None or df.empty or len(df) < p.lookback_sessions:
            return None

        window = df.tail(p.lookback_sessions).copy()
        if len(window) < p.min_separation_sessions + 10:
            return None

        low = window.get("low", window["close"]).astype(float).values
        high = window.get("high", window["close"]).astype(float).values
        close = window["close"].astype(float).values
        index = window.index

        # First low = min in first half
        first_half_end = len(window) // 2
        if first_half_end < 5:
            return None
        first_low_idx = int(np.argmin(low[:first_half_end]))
        first_low = float(low[first_low_idx])

        # Second low = min in a window after enough separation
        second_search_start = first_low_idx + p.min_separation_sessions
        if second_search_start >= len(window) - 5:
            return None
        second_low_rel = int(np.argmin(low[second_search_start:]))
        second_low_idx = second_search_start + second_low_rel
        second_low = float(low[second_low_idx])

        # --- FIX 1 (quickview only): the second low must be a CONFIRMED swing low.
        # 1) at least N bars must FOLLOW it (an undercut on the last bar means the
        #    decline is still in progress — not a completed W).
        bars_after = (len(window) - 1) - second_low_idx
        if bars_after < p.min_bars_after_second_low:
            return None
        # 2) price must have RECOVERED off the low: the max high after the low must
        #    exceed the low by >= min_recovery_off_second_low, AND the last close
        #    must sit above the low (a small epsilon guards exact-equality noise).
        post_low_high = float(np.max(high[second_low_idx + 1:])) if second_low_idx + 1 < len(window) else second_low
        recovery_off_low = (post_low_high - second_low) / second_low if second_low > 0 else 0.0
        last_close = float(close[-1])
        if recovery_off_low < p.min_recovery_off_second_low:
            return None
        if last_close <= second_low * (1.0 + 1e-4):
            return None

        # Lows must be comparable
        low_ref = min(first_low, second_low)
        if low_ref <= 0:
            return None
        mismatch = abs(first_low - second_low) / first_low
        if mismatch > p.max_low_mismatch_pct:
            return None

        if p.require_second_undercut and second_low > first_low:
            return None

        # Middle peak between the two lows
        if second_low_idx - first_low_idx < 5:
            return None
        middle_region = high[first_low_idx:second_low_idx]
        if middle_region.size == 0:
            return None
        middle_peak_rel = int(np.argmax(middle_region))
        middle_peak_idx = first_low_idx + middle_peak_rel
        middle_peak = float(middle_region[middle_peak_rel])

        rise_from_first = (middle_peak - first_low) / first_low if first_low > 0 else 0.0
        if not (p.min_middle_peak_rise <= rise_from_first <= p.max_middle_peak_rise):
            return None

        # --- Outer W legs (quickview overlay): the two arms OUTSIDE the inner
        # first_low -> middle_peak -> second_low checkmark, so the drawn shape is a
        # full "W" rather than just its center. These are descriptive anchors only;
        # they do NOT gate detection.
        #   (1) LEFT TOP  = highest high in the run-up BEFORE the first low (the
        #       level price fell from into the W's left leg).
        #   (5) RIGHT TOP = highest high AFTER the second low (the recovery that
        #       carries toward the pivot / breakout — the W's right leg).
        left_lookback = max(0, first_low_idx - p.outer_leg_lookback)
        if first_low_idx > left_lookback:
            left_top_idx = left_lookback + int(np.argmax(high[left_lookback:first_low_idx]))
        else:
            left_top_idx = first_low_idx
        left_top = float(high[left_top_idx])

        if second_low_idx + 1 < len(window):
            right_top_idx = second_low_idx + 1 + int(np.argmax(high[second_low_idx + 1:]))
        else:
            right_top_idx = second_low_idx
        right_top = float(high[right_top_idx])

        # Pivot = middle peak high + small buffer
        pivot = middle_peak + 0.10
        # Reject "stale" patterns where the breakout already happened and price
        # has run far past the pivot — the base is no longer actionable as an
        # entry. O'Neil's buy zone is pivot to pivot+5%; >20% above pivot is
        # firmly in "extended, chasing" territory.
        if pivot > 0 and last_close > pivot * 1.20:
            return None
        dist_to_pivot = (pivot - last_close) / pivot if pivot > 0 else None

        # Confidence: balance mismatch quality + peak rise centered on ~10%
        mismatch_score = 1.0 - min(1.0, mismatch / p.max_low_mismatch_pct)
        rise_score = 1.0 - min(1.0, abs(rise_from_first - 0.10) / 0.10)
        undercut_bonus = 0.15 if second_low <= first_low else 0.0
        confidence = float(min(1.0, 0.5 * mismatch_score + 0.35 * rise_score + undercut_bonus))

        return PatternMatch(
            name=self.name,
            detected=True,
            pivot=round(pivot, 2),
            confidence=round(confidence, 3),
            started_on=_as_date(index[first_low_idx]),
            completed_on=_as_date(index[-1]),
            evidence={
                "first_low": round(first_low, 2),
                "second_low": round(second_low, 2),
                "low_mismatch_pct": round(mismatch, 4),
                "middle_peak": round(middle_peak, 2),
                "middle_peak_rise_pct": round(rise_from_first, 4),
                "separation_sessions": int(second_low_idx - first_low_idx),
                "second_undercuts_first": bool(second_low <= first_low),
                "current_close": round(last_close, 2),
                "dist_to_pivot_pct": round(dist_to_pivot, 4) if dist_to_pivot is not None else None,
                # FIX 1 evidence: swing-low confirmation metrics.
                "bars_after_second_low": int(bars_after),
                "recovery_off_second_low_pct": round(recovery_off_low, 4),
                # FIX 2: anchor DATES (ISO) for the frontend W overlay.
                "first_low_date": _as_iso(index[first_low_idx]),
                "middle_peak_date": _as_iso(index[middle_peak_idx]),
                "second_low_date": _as_iso(index[second_low_idx]),
                # Outer W legs (full-W overlay): left-top run-up + right-top recovery.
                "left_top": round(left_top, 2),
                "left_top_date": _as_iso(index[left_top_idx]),
                "right_top": round(right_top, 2),
                "right_top_date": _as_iso(index[right_top_idx]),
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
