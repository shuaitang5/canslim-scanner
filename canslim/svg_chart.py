"""Inline SVG chart renderer for the HTML report.

Generates a single self-contained SVG element per ticker with:
  * Price candles (or line) with 10/21/50/200 day moving averages
  * Volume bars in a sub-panel below the price
  * Horizontal pivot line + tinted buy zone (pivot to +5%)
  * Horizontal stop level (-7% from pivot)
  * Hover crosshair via vanilla JS (script appended once per page)
  * <title> elements per session for native browser tooltips

Output is pure SVG markup (string). The caller embeds inline in <body>.
Coordinate system is fully self-contained (viewBox), so the chart scales to
whatever container width the parent CSS sets.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from canslim.models import ScanResult


# Layout constants — viewBox dimensions (units, not pixels)
SVG_W = 800
SVG_H = 400
MARGIN_LEFT = 50
MARGIN_RIGHT = 96   # right gutter holds price labels e.g. "pivot $1234.56" (~14 chars @ font-size 10 monospace ≈ 84u, drawn from SVG_W-MARGIN_RIGHT+4)
MARGIN_TOP = 16
MARGIN_BOTTOM = 22
VOLUME_BAND_H = 80     # height reserved for volume sub-panel
PRICE_BAND_H = SVG_H - MARGIN_TOP - MARGIN_BOTTOM - VOLUME_BAND_H

VISIBLE_SESSIONS = 130  # ~6 months


def render_svg(
    ticker: str,
    df: pd.DataFrame,
    result: Optional[ScanResult] = None,
    *,
    horizontal_lines: Optional[list[tuple[float, str, str]]] = None,
    vertical_markers: Optional[list[tuple[int, str]]] = None,
    height: int = SVG_H,
) -> str:
    """Return an SVG <svg>...</svg> element for the ticker, or empty string on failure.

    `result` is optional — when omitted (e.g. for market overview charts), the
    pivot/buy-zone/stop overlays are skipped, but MA + candles + volume render.

    `horizontal_lines` is a list of (price, color, label) tuples drawn as
    dashed horizontal lines (useful for VIX threshold lines like 15/20/30).

    `vertical_markers` is a list of (session_index, color) tuples drawn as
    vertical dashed lines (useful for distribution-day markers on SPY).

    `height` overrides SVG_H — useful for compact mini-charts.
    """
    if df is None or df.empty:
        return ""
    window = df.tail(VISIBLE_SESSIONS).copy()
    if len(window) < 5:
        return ""

    close = window["close"].astype(float).values
    open_ = window.get("open", window["close"]).astype(float).values
    high = window.get("high", window["close"]).astype(float).values
    low = window.get("low", window["close"]).astype(float).values
    volume = window["volume"].astype(float).values
    dates = window.index

    # Moving averages computed on the full series so values are correct at the left edge
    full_close = df["close"].astype(float)
    ma10 = full_close.rolling(10).mean().loc[window.index].values
    ma21 = full_close.rolling(21).mean().loc[window.index].values
    ma50 = full_close.rolling(50).mean().loc[window.index].values
    ma200 = full_close.rolling(200).mean().loc[window.index].values

    # Pattern overlays from highest-confidence pattern with a numeric pivot
    pivot, buy_zone_high, stop_loss = None, None, None
    pattern_label = None
    if result and result.patterns:
        candidates = [p for p in result.patterns if p.pivot is not None]
        if candidates:
            p = max(candidates, key=lambda x: x.confidence)
            pivot = float(p.pivot or 0)
            if pivot > 0:
                buy_zone_high = pivot * 1.05
                stop_loss = pivot * 0.93
                pattern_label = p.name
    extra_h_lines = horizontal_lines or []
    extra_v_markers = vertical_markers or []

    # Compute layout — allow caller to override height for compact charts
    svg_h = height
    price_band_h = svg_h - MARGIN_TOP - MARGIN_BOTTOM - VOLUME_BAND_H

    # Y-axis range: include MAs + price + pivot + stop + extra h-lines
    y_values = [v for v in [
        np.nanmin(low), np.nanmax(high),
        float(np.nanmin(ma200[~np.isnan(ma200)])) if not np.all(np.isnan(ma200)) else None,
        float(np.nanmax(ma200[~np.isnan(ma200)])) if not np.all(np.isnan(ma200)) else None,
        pivot, buy_zone_high, stop_loss,
    ] if v is not None and not (isinstance(v, float) and math.isnan(v))]
    for hp, _, _ in extra_h_lines:
        y_values.append(hp)
    y_min = min(y_values) * 0.98
    y_max = max(y_values) * 1.02

    n = len(close)
    # X-coordinate per session
    plot_w = SVG_W - MARGIN_LEFT - MARGIN_RIGHT
    x_step = plot_w / max(n - 1, 1)

    def x_of(i: int) -> float:
        return MARGIN_LEFT + i * x_step

    def y_price(p: float) -> float:
        return MARGIN_TOP + (1 - (p - y_min) / (y_max - y_min)) * price_band_h

    # Volume band
    vol_max = float(np.nanmax(volume)) if len(volume) else 1.0
    vol_band_top = MARGIN_TOP + price_band_h + 2
    vol_band_bottom = vol_band_top + VOLUME_BAND_H

    def y_volume(v: float) -> float:
        if vol_max <= 0 or math.isnan(v):
            return vol_band_bottom
        return vol_band_bottom - (v / vol_max) * VOLUME_BAND_H

    # === Build SVG layers ===
    parts: list[str] = []
    parts.append(
        f'<svg class="canslim-chart" viewBox="0 0 {SVG_W} {svg_h}" '
        f'preserveAspectRatio="xMidYMid meet" data-ticker="{ticker}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    )

    # Background (light) for volume band
    parts.append(
        f'<rect x="{MARGIN_LEFT}" y="{vol_band_top}" '
        f'width="{plot_w}" height="{VOLUME_BAND_H}" fill="#fafafa" stroke="none"/>'
    )

    # Y-axis grid lines (5 horizontal lines)
    parts.append('<g class="grid" stroke="#eaecef" stroke-width="0.5">')
    for k in range(5):
        gy = MARGIN_TOP + (price_band_h * k / 4)
        parts.append(f'<line x1="{MARGIN_LEFT}" y1="{gy:.1f}" x2="{SVG_W - MARGIN_RIGHT}" y2="{gy:.1f}"/>')
    parts.append('</g>')

    # Vertical markers (e.g., distribution days on SPY)
    if extra_v_markers:
        parts.append('<g class="v-markers">')
        for idx, color in extra_v_markers:
            if 0 <= idx < n:
                vx = x_of(idx)
                parts.append(
                    f'<line x1="{vx:.1f}" y1="{MARGIN_TOP}" '
                    f'x2="{vx:.1f}" y2="{vol_band_bottom}" '
                    f'stroke="{color}" stroke-width="0.8" stroke-dasharray="2 2" '
                    f'opacity="0.6"/>'
                )
        parts.append('</g>')

    # Horizontal threshold lines (e.g., VIX 15/20/30)
    if extra_h_lines:
        parts.append('<g class="h-lines">')
        for hp, hcolor, hlabel in extra_h_lines:
            try:
                hpv = float(hp)
            except (TypeError, ValueError):
                continue
            hy = y_price(hpv)
            parts.append(
                f'<line x1="{MARGIN_LEFT}" y1="{hy:.1f}" '
                f'x2="{SVG_W - MARGIN_RIGHT}" y2="{hy:.1f}" '
                f'stroke="{hcolor}" stroke-width="0.8" stroke-dasharray="3 3" opacity="0.7"/>'
                f'<text x="{SVG_W - MARGIN_RIGHT + 4}" y="{hy + 3:.1f}" '
                f'font-size="10" fill="{hcolor}" font-family="ui-monospace,Menlo,monospace">'
                f'{_escape(hlabel)}</text>'
            )
        parts.append('</g>')

    # Buy zone band (tinted green between pivot and pivot*1.05)
    if pivot and buy_zone_high:
        y_top = y_price(buy_zone_high)
        y_bot = y_price(pivot)
        parts.append(
            f'<rect class="buy-zone" x="{MARGIN_LEFT}" y="{y_top:.1f}" '
            f'width="{plot_w}" height="{(y_bot - y_top):.1f}" '
            f'fill="rgba(46, 125, 50, 0.12)" stroke="none"/>'
        )

    # Pivot line (red dashed)
    if pivot:
        py = y_price(pivot)
        parts.append(
            f'<line class="pivot" x1="{MARGIN_LEFT}" y1="{py:.1f}" '
            f'x2="{SVG_W - MARGIN_RIGHT}" y2="{py:.1f}" '
            f'stroke="#c62828" stroke-width="1.2" stroke-dasharray="6 3"/>'
            f'<text x="{SVG_W - MARGIN_RIGHT + 4}" y="{py + 3:.1f}" '
            f'font-size="10" fill="#c62828" font-family="ui-monospace,Menlo,monospace">'
            f'pivot ${pivot:.2f}</text>'
        )

    # Stop loss line (yellow dashed)
    if stop_loss:
        sy = y_price(stop_loss)
        parts.append(
            f'<line class="stop" x1="{MARGIN_LEFT}" y1="{sy:.1f}" '
            f'x2="{SVG_W - MARGIN_RIGHT}" y2="{sy:.1f}" '
            f'stroke="#e9740b" stroke-width="1" stroke-dasharray="3 3"/>'
            f'<text x="{SVG_W - MARGIN_RIGHT + 4}" y="{sy + 3:.1f}" '
            f'font-size="10" fill="#e9740b" font-family="ui-monospace,Menlo,monospace">'
            f'stop ${stop_loss:.2f}</text>'
        )

    # Volume bars
    parts.append('<g class="volume">')
    for i, v in enumerate(volume):
        if math.isnan(v) or v <= 0:
            continue
        bar_x = x_of(i) - x_step * 0.4
        bar_w = max(x_step * 0.8, 0.5)
        bar_y = y_volume(v)
        bar_h = vol_band_bottom - bar_y
        # color: green if up day, red if down day
        is_up = (close[i] >= open_[i]) if not (math.isnan(open_[i]) or math.isnan(close[i])) else True
        color = "#a5d6a7" if is_up else "#ef9a9a"
        parts.append(
            f'<rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="{bar_w:.2f}" '
            f'height="{bar_h:.2f}" fill="{color}" stroke="none"/>'
        )
    parts.append('</g>')

    # MA lines
    for ma_arr, color, label, width in [
        (ma200, "#9e9e9e", "200d", 1.0),
        (ma50, "#1976d2", "50d", 1.2),
        (ma21, "#7b1fa2", "21d", 1.0),
        (ma10, "#00897b", "10d", 0.8),
    ]:
        pts = []
        for i, v in enumerate(ma_arr):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            pts.append(f"{x_of(i):.1f},{y_price(v):.1f}")
        if not pts:
            continue
        parts.append(
            f'<polyline class="ma ma-{label}" fill="none" '
            f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
            f'stroke-linejoin="round" points="{" ".join(pts)}"/>'
        )

    # Candles (wicks + bodies)
    parts.append('<g class="candles">')
    for i in range(n):
        if math.isnan(close[i]) or math.isnan(open_[i]):
            continue
        cx = x_of(i)
        is_up = close[i] >= open_[i]
        body_color = "#2e7d32" if is_up else "#c62828"
        # Wick (high-low)
        if not math.isnan(high[i]) and not math.isnan(low[i]):
            parts.append(
                f'<line x1="{cx:.1f}" y1="{y_price(high[i]):.1f}" '
                f'x2="{cx:.1f}" y2="{y_price(low[i]):.1f}" '
                f'stroke="{body_color}" stroke-width="0.6"/>'
            )
        # Body (open-close)
        body_top = y_price(max(open_[i], close[i]))
        body_bot = y_price(min(open_[i], close[i]))
        bw = max(x_step * 0.6, 0.8)
        # Tooltip via SVG <title>
        date_label = dates[i].strftime("%Y-%m-%d") if hasattr(dates[i], "strftime") else str(dates[i])[:10]
        title = (
            f"{date_label}: O ${open_[i]:.2f} "
            f"H ${high[i]:.2f} L ${low[i]:.2f} C ${close[i]:.2f} "
            f"vol {int(volume[i]):,}"
        )
        parts.append(
            f'<rect x="{cx - bw/2:.2f}" y="{body_top:.2f}" '
            f'width="{bw:.2f}" height="{max(body_bot - body_top, 0.5):.2f}" '
            f'fill="{body_color}" stroke="none">'
            f'<title>{_escape(title)}</title>'
            f'</rect>'
        )
    parts.append('</g>')

    # Y-axis price labels (5 ticks)
    parts.append('<g class="y-labels" font-size="10" fill="#5b6473" font-family="ui-monospace,Menlo,monospace">')
    for k in range(5):
        py = MARGIN_TOP + (price_band_h * k / 4)
        price_at = y_max - (k / 4) * (y_max - y_min)
        parts.append(
            f'<text x="{MARGIN_LEFT - 4}" y="{py + 3:.1f}" text-anchor="end">${price_at:.2f}</text>'
        )
    parts.append('</g>')

    # X-axis date labels (~5 ticks across the window)
    parts.append('<g class="x-labels" font-size="9" fill="#5b6473" font-family="ui-monospace,Menlo,monospace">')
    n_ticks = min(5, n)
    for k in range(n_ticks):
        idx = int(k * (n - 1) / (n_ticks - 1)) if n_ticks > 1 else 0
        d = dates[idx]
        label = d.strftime("%m/%d") if hasattr(d, "strftime") else str(d)[5:10]
        anchor = "start" if k == 0 else ("end" if k == n_ticks - 1 else "middle")
        parts.append(
            f'<text x="{x_of(idx):.1f}" y="{svg_h - 6}" text-anchor="{anchor}">{label}</text>'
        )
    parts.append('</g>')

    # Legend (top-left, compact)
    parts.append(
        f'<g class="legend" font-size="9.5" font-family="-apple-system,system-ui,sans-serif">'
        f'<text x="{MARGIN_LEFT + 4}" y="{MARGIN_TOP + 12}">'
        f'<tspan fill="#00897b">10d</tspan> '
        f'<tspan fill="#7b1fa2">21d</tspan> '
        f'<tspan fill="#1976d2">50d</tspan> '
        f'<tspan fill="#9e9e9e">200d</tspan>'
        f'{" · pattern: " + pattern_label if pattern_label else ""}'
        f'</text></g>'
    )

    # Plot border
    parts.append(
        f'<rect x="{MARGIN_LEFT}" y="{MARGIN_TOP}" '
        f'width="{plot_w}" height="{price_band_h + VOLUME_BAND_H + 2}" '
        f'fill="none" stroke="#d8dde3" stroke-width="0.5"/>'
    )

    parts.append('</svg>')
    return "\n".join(parts)


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
