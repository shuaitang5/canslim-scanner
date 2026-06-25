# vendored from canslim-scanner/canslim/patterns/ascending_triangle.py on 2026-06-25; DO NOT import canslim at runtime
"""Ascending Triangle detector.

Bullish continuation: a FLAT upper resistance (≈ equal swing highs) with a
RISING lower support (swing lows stepping up), converging toward an upside
breakout.

numpy + pandas only.

FIX vs. the prior canslim logic: the source validated "rising support"
using ONLY (last_low - first_low)/first_low ≥ 2%. That accepts garbage:
  * V-bottoms where the middle low is HIGHER than the last (not monotonic),
  * near-vertical recoveries (e.g. +45% "support") that aren't a triangle base.
We now require the bottom swing lows to be (a) roughly MONOTONICALLY rising
(each swing low ≥ the prior minus a small tolerance) and (b) within a SANE
slope band (rise between min and max %), so the lower trendline is an actual
gently-rising support, not a ramp.

FIX 2 — emit anchor coords so the frontend can DRAW the two trendlines:
  top_left_date / top_right_date  (flat resistance spans these, at top_max)
  support_first_date+price / support_last_date+price (rising support line)
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
class AscendingTriangleParams:
    min_sessions: int = 20   # ~4 weeks
    max_sessions: int = 60   # ~12 weeks
    max_top_variation: float = 0.04    # flat resistance: ≤4% variation across highs
    min_lows_slope_pct: float = 0.02   # rising support: last swing low ≥2% above first
    # FIX (quickview): cap the rise so a near-vertical recovery isn't called a
    # triangle base, and require the lows to actually step up in order.
    max_lows_slope_pct: float = 0.20   # last swing low ≤20% above first (else it's a ramp, not support)
    monotonic_tolerance: float = 0.02  # each swing low must be ≥ prior * (1 - this); allows minor noise
    peak_prominence: float = 0.02      # swing must be 2%+ above neighbors
    # FIX 3: the top-swings and bottom-swings must overlap in time by at least
    # this many sessions (so resistance + support describe ONE triangle, not two
    # disjoint periods).
    min_overlap_sessions: int = 10
    # FIX 4: a FLAT top must connect >=2 NEAR-EQUAL peaks ("tested at least
    # twice"). A peak counts as a resistance touch only if it's within this tight
    # band of the top. (max_top_variation gates the overall pattern; this tighter
    # band picks the actual anchor peaks that define the horizontal line.)
    flat_top_band: float = 0.02       # peaks within 2% of top_max are "touches"
    min_resistance_touches: int = 2   # the flat line must be tested >= twice


class AscendingTriangle(ChartPattern):
    name = "ascending_triangle"

    def __init__(self, params: Optional[AscendingTriangleParams] = None) -> None:
        self.params = params or AscendingTriangleParams()

    def detect(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        p = self.params
        if df is None or df.empty or len(df) < p.min_sessions:
            return None

        for size in range(min(p.max_sessions, len(df)), p.min_sessions - 1, -1):
            window = df.tail(size)
            highs = window["high"].astype(float).values
            lows = window["low"].astype(float).values
            closes = window["close"].astype(float).values
            index = window.index

            top_level, top_indices = _top_swings(highs, p.peak_prominence)
            if top_level is None or len(top_indices) < 2:
                continue
            top_max = float(np.max(highs[top_indices]))
            top_min = float(np.min(highs[top_indices]))
            if top_max <= 0:
                continue
            variation = (top_max - top_min) / top_max
            if variation > p.max_top_variation:
                continue

            # --- FIX 4: a flat top must connect >=2 NEAR-EQUAL peaks. Keep only the
            # swing highs within `flat_top_band` of top_max — those are the genuine
            # resistance "touches". A line anchored to a single peak (with the other
            # 2%+ below it) is not a flat top. ---
            touch_indices = [i for i in top_indices
                             if (top_max - float(highs[i])) / top_max <= p.flat_top_band]
            if len(touch_indices) < p.min_resistance_touches:
                continue

            bot_indices = _bottom_swings(lows, p.peak_prominence)
            if len(bot_indices) < 2:
                continue
            bot_lows = [float(lows[i]) for i in bot_indices]
            first_low = bot_lows[0]
            last_low = bot_lows[-1]
            if first_low <= 0:
                continue

            # --- FIX: rising support must actually RISE, in order, at a sane slope. ---
            # (a) Monotonic-ish: each swing low ≥ the previous one minus a small
            #     tolerance (so a single noisy dip doesn't disqualify, but a
            #     middle low far below the first — a V-bottom — does).
            monotonic = all(
                bot_lows[k] >= bot_lows[k - 1] * (1.0 - p.monotonic_tolerance)
                for k in range(1, len(bot_lows))
            )
            if not monotonic:
                continue
            # (b) Slope within a sane band: rising, but not a near-vertical ramp.
            slope_pct = (last_low - first_low) / first_low
            if not (p.min_lows_slope_pct <= slope_pct <= p.max_lows_slope_pct):
                continue

            # --- FIX 3: the flat top and the rising support must form ONE triangle
            # over a SHARED span and actually CONVERGE — not a June ceiling glued
            # to a May floor. ---
            # Resistance anchors = the genuine near-equal touch peaks (FIX 4), so
            # the flat line connects >=2 tested peaks, not a single high.
            top_first_i, top_last_i = touch_indices[0], touch_indices[-1]
            sup_first_i, sup_last_i = bot_indices[0], bot_indices[-1]
            # (c) Time overlap: the two trendlines' [start,end] spans must overlap
            #     by at least `min_overlap_frac` of the window — they describe the
            #     SAME consolidation, not two separate periods.
            overlap = (min(top_last_i, sup_last_i) - max(top_first_i, sup_first_i))
            if overlap < p.min_overlap_sessions:
                continue
            # (d) Convergence: the rising support must sit BELOW the flat top across
            #     the pattern (a triangle narrows). Require the last support low to
            #     still be a real gap below top_max (not already through it) AND the
            #     support to be rising toward it (slope already positive from (b)).
            if last_low >= top_max:
                continue
            gap_first = (top_max - first_low) / top_max
            gap_last = (top_max - last_low) / top_max
            # The wedge must NARROW: the later gap to resistance must be smaller
            # than the earlier gap (support climbing toward the flat ceiling).
            if not (gap_last < gap_first):
                continue

            pivot = top_max + 0.10
            last_close = float(closes[-1])
            # Reject "stale" triangle: price >20% past pivot — already broken out and run.
            if pivot > 0 and last_close > pivot * 1.20:
                continue
            dist_to_pivot = (pivot - last_close) / pivot if pivot > 0 else None
            # Confidence: tight top + strong rising support + enough swings
            tight_score = 1.0 - min(1.0, variation / p.max_top_variation)
            slope_score = min(1.0, slope_pct / 0.10)
            swings_score = min(1.0, (len(top_indices) + len(bot_indices)) / 6.0)
            confidence = float(0.45 * tight_score + 0.35 * slope_score + 0.2 * swings_score)

            # (top_first_i/top_last_i/sup_first_i/sup_last_i computed above for the
            # overlap/convergence checks — reused here as the trendline anchors.)

            return PatternMatch(
                name=self.name,
                detected=True,
                pivot=round(pivot, 2),
                confidence=round(confidence, 3),
                started_on=_as_date(index[0]),
                completed_on=_as_date(index[-1]),
                evidence={
                    "sessions": int(size),
                    "top_max": round(top_max, 2),
                    "top_variation_pct": round(variation, 4),
                    "rising_support_pct": round(slope_pct, 4),
                    "top_touches": int(len(top_indices)),
                    # FIX 4: how many of those peaks actually touch the flat line.
                    "resistance_touches": int(len(touch_indices)),
                    "bottom_touches": int(len(bot_indices)),
                    "current_close": round(last_close, 2),
                    "dist_to_pivot_pct": round(dist_to_pivot, 4) if dist_to_pivot is not None else None,
                    # FIX 2: anchor coords (ISO dates + prices) for the trendlines.
                    "top_left_date": _as_iso(index[top_first_i]),
                    "top_right_date": _as_iso(index[top_last_i]),
                    # FIX 4: the actual tested peaks (ISO date + price), so the
                    # frontend can mark each touch on the flat resistance line.
                    "resistance_touch_dates": [_as_iso(index[i]) for i in touch_indices],
                    "resistance_touch_prices": [round(float(highs[i]), 2) for i in touch_indices],
                    "support_first_date": _as_iso(index[sup_first_i]),
                    "support_first_price": round(first_low, 2),
                    "support_last_date": _as_iso(index[sup_last_i]),
                    "support_last_price": round(last_low, 2),
                },
            )
        return None


def _top_swings(highs: np.ndarray, prominence: float) -> tuple[Optional[float], list[int]]:
    """Find local maxima that rise `prominence` above neighbors."""
    n = len(highs)
    out: list[int] = []
    for i in range(2, n - 2):
        if highs[i] <= 0:
            continue
        left = max(highs[i - 2], highs[i - 1])
        right = max(highs[i + 1], highs[i + 2])
        if highs[i] > left * (1 + prominence * 0.5) and highs[i] > right * (1 + prominence * 0.5):
            out.append(i)
    return (max(highs[out]) if out else None), out


def _bottom_swings(lows: np.ndarray, prominence: float) -> list[int]:
    n = len(lows)
    out: list[int] = []
    for i in range(2, n - 2):
        if lows[i] <= 0:
            continue
        left = min(lows[i - 2], lows[i - 1])
        right = min(lows[i + 1], lows[i + 2])
        if lows[i] < left * (1 - prominence * 0.5) and lows[i] < right * (1 - prominence * 0.5):
            out.append(i)
    return out


def _as_date(idx_value) -> Optional[date]:
    try:
        return idx_value.date() if hasattr(idx_value, "date") else None
    except Exception:
        return None


def _as_iso(idx_value) -> Optional[str]:
    try:
        return idx_value.date().isoformat() if hasattr(idx_value, "date") else None
    except Exception:
        return None
