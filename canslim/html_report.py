"""HTML report renderer — single self-contained file with rich interactivity.

Produces a single index.html per scan run with:
- Sticky header (market regime + breadth — Phase 3 will fill VIX/sectors)
- Tab navigation (full-matches / buyable / watchlist / basing / override)
- Per-candidate <details> blocks (collapsed by default, click to expand)
- Sticky TOC sidebar with ticker list
- Vanilla JS search/filter/sort (no framework)
- Inline SVG charts (Phase 2 will replace embedded PNGs)

Design language: minimal/professional. High data density, semantic colors,
monospace for numerics, system sans-serif for prose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import jinja2
import pandas as pd

from canslim.models import RunManifest, ScanResult
from canslim.report import _fmt_mktcap


LETTERS = ["C", "A", "N", "S", "L", "I", "M"]


# ---------- Public API ----------

def render_html(
    results: list[ScanResult],
    manifest: RunManifest,
    top_n_near_matches: int = 20,
    chart_paths: Optional[dict[str, Path]] = None,
    embed_base64: bool = True,
    price_frames: Optional[dict[str, "pd.DataFrame"]] = None,
) -> str:
    """Render results to a single self-contained HTML string.

    If `price_frames` is provided, inline SVG charts are rendered per ticker
    (preferred). Falls back to embedded PNG `<img>` tags from `chart_paths`
    for any ticker missing a price frame.
    """
    chart_paths = chart_paths or {}
    price_frames = price_frames or {}
    chart_data_uris = _build_chart_data_uris(chart_paths, embed_base64=embed_base64)
    inline_svgs = _build_inline_svgs(results, price_frames)
    market_charts = _build_market_overview_svgs(price_frames, manifest)

    matches = sorted([r for r in results if r.passed], key=lambda r: -r.composite_score)
    scanned = sorted(
        [r for r in results if r.status == "scanned"], key=lambda r: -r.composite_score
    )
    near_matches = [r for r in scanned if not r.passed][:top_n_near_matches]

    match_set = {r.ticker for r in matches}
    candidate_pool = [r for r in scanned if r.ticker not in match_set][:top_n_near_matches]
    buyable, watchlist, basing = _bucket_candidates(candidate_pool)

    # Override watchlist: any ticker where A or S used override
    overrides = []
    for r in scanned:
        used = _override_reasons(r)
        if used:
            overrides.append((r, used))
    overrides.sort(key=lambda pair: (-_gate_pass_count(pair[0]), -pair[0].composite_score))

    pattern_hits = sorted(
        [r for r in scanned if r.patterns],
        key=lambda r: -max((p.confidence for p in r.patterns), default=0.0),
    )[:top_n_near_matches]

    # Render context
    ctx = {
        "manifest": manifest,
        "regime": manifest.market_regime,
        "regime_index_label": _regime_index_label(manifest),
        "matches": matches,
        "near_matches": near_matches,
        "buyable": buyable,
        "watchlist": watchlist,
        "basing": basing,
        "overrides": overrides,
        "pattern_hits": pattern_hits,
        "chart_uri": chart_data_uris,
        "inline_svg": inline_svgs,
        "market_charts": market_charts,
        "letters": LETTERS,
        "helpers": {
            "gate_flags": _gate_flags,
            "failed_gates": _failed_gates,
            "num": _num,
            "format_value_for_letter": _format_value_for_letter,
            "patterns_label": _patterns_label,
            "ad_grade_label": _ad_grade_label,
            "override_reasons": _override_reasons,
            "gate_pass_count": _gate_pass_count,
            "entry_status": _entry_status,
            "primary_pattern": _primary_pattern,
        },
    }

    env = _jinja_env()
    return env.get_template("report.html").render(**ctx)


# ---------- Helpers used by the template (also exported so tests can hit them) ----------

def _gate_flags(r: ScanResult) -> str:
    """Compact CANSLIM string: UPPER=pass, lower=fail, ?=abstain (no data)."""
    out: list[str] = []
    for L in LETTERS:
        cr = r.criteria.get(L)
        if cr is None:
            out.append("?")
            continue
        if not cr.data_available:
            out.append("?")
            continue
        out.append(L if cr.passed else L.lower())
    return "".join(out)


def _failed_gates(r: ScanResult) -> str:
    out: list[str] = []
    for L in LETTERS:
        cr = r.criteria.get(L)
        if cr is None or not cr.is_gate:
            continue
        if not cr.passed:
            out.append(L)
    return "".join(out)


def _gate_pass_count(r: ScanResult) -> int:
    return sum(
        1
        for L in LETTERS
        if r.criteria.get(L) and r.criteria[L].is_gate and r.criteria[L].passed
    )


def _num(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v) if v == v else 0.0  # NaN check
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _format_value_for_letter(letter: str, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return f"[{value}]"
    if isinstance(value, float) and value != value:  # NaN
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if letter in ("C", "A", "I"):
        return f"[{v:.1%}]"
    if letter in ("L", "S"):
        return f"[{v:.2f}]"
    return ""


def _patterns_label(patterns: list) -> str:
    if not patterns:
        return ""
    parts = []
    for p in sorted(patterns, key=lambda x: -x.confidence)[:3]:
        short = {
            "cup_with_handle": "cup+H",
            "high_tight_flag": "HTF",
            "three_weeks_tight": "3wk",
            "ascending_triangle": "asc△",
            "double_bottom": "2bot",
            "flat_base": "flat",
            "consolidation": "box",
            "saucer": "sauc",
        }.get(p.name, p.name)
        parts.append(f"{short}({p.confidence:.2f})")
    return ", ".join(parts)


def _ad_grade_label(grade: Optional[str]) -> str:
    return {
        "A": "heavy accumulation",
        "B": "moderate accumulation",
        "C": "neutral / balanced",
        "D": "moderate distribution",
        "E": "heavy distribution",
    }.get(grade or "", "—")


def _override_reasons(r: ScanResult) -> list[str]:
    used: list[str] = []
    a = r.criteria.get("A")
    if a and isinstance(a.evidence, dict) and a.evidence.get("override_used"):
        used.append("A:leadership")
    s = r.criteria.get("S")
    if s and isinstance(s.evidence, dict) and s.evidence.get("pattern_override"):
        used.append("S:pattern")
    return used


def _primary_pattern(r: ScanResult):
    if not r.patterns:
        return None
    candidates = [p for p in r.patterns if p.pivot is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.confidence)


def _entry_status(r: ScanResult) -> dict:
    """Return entry-plan dict for template rendering, or empty dict if not applicable."""
    p = _primary_pattern(r)
    if not p or not p.pivot:
        return {}
    pivot = float(p.pivot)
    dist = p.evidence.get("dist_to_pivot_pct") if isinstance(p.evidence, dict) else None
    n = r.criteria.get("N")
    n_ev = n.evidence if (n is not None and isinstance(n.evidence, dict)) else {}
    close = n_ev.get("close")

    buy_zone_low = pivot
    buy_zone_high = pivot * 1.05
    stop_loss = pivot * 0.93
    invalidation = pivot * 0.95

    if isinstance(dist, (int, float)):
        if dist > 0.05:
            status_class, status_label = "forming", "Setup forming"
            note = f"Close ${close:,.2f} is {dist:+.1%} below pivot. Set price alert at ${pivot:,.2f}."
        elif dist > 0.0:
            status_class, status_label = "approaching", "Approaching pivot"
            note = f"Close ${close:,.2f} is {dist:+.1%} below pivot. Watch for break above ${pivot:,.2f} on heavy volume."
        elif dist >= -0.05:
            status_class, status_label = "buyzone", "In buy zone"
            note = f"Close ${close:,.2f} is {-dist:+.1%} past pivot ${pivot:,.2f}. Textbook O'Neil entry zone."
        else:
            status_class, status_label = "extended", "Extended"
            note = f"Close ${close:,.2f} is {-dist:.1%} past pivot ${pivot:,.2f}, outside +5% buy zone. Chase risk."
    else:
        status_class, status_label, note = "unknown", "Pattern detected", ""

    if r.passed:
        sizing = "full position (all gates pass + pattern confirmed)"
    elif p.confidence >= 0.75:
        sizing = "full position (high pattern confidence; near-miss is on a soft gate)"
    elif p.confidence >= 0.60:
        sizing = "half position (moderate pattern confidence)"
    else:
        sizing = "watchlist only (low pattern confidence)"

    return {
        "pivot": pivot,
        "dist": dist,
        "buy_zone_low": buy_zone_low,
        "buy_zone_high": buy_zone_high,
        "stop_loss": stop_loss,
        "invalidation": invalidation,
        "sizing": sizing,
        "status_class": status_class,
        "status_label": status_label,
        "note": note,
        "pattern_name": p.name,
    }


# ---------- Internal ----------

# _fmt_mktcap is the single canonical implementation in canslim.report; imported
# above and registered as a Jinja filter below.


def _jinja_env() -> jinja2.Environment:
    """Jinja2 environment that loads templates from canslim/templates/."""
    template_dir = Path(__file__).parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    # Useful filters
    env.filters["fmt_pct"] = lambda v: f"{v:+.1%}" if isinstance(v, (int, float)) else "—"
    env.filters["fmt_pct_abs"] = lambda v: f"{v:.1%}" if isinstance(v, (int, float)) else "—"
    env.filters["fmt_money"] = lambda v: f"${v:,.2f}" if isinstance(v, (int, float)) else "—"
    env.filters["fmt_int"] = lambda v: f"{int(v):,}" if isinstance(v, (int, float)) else "—"
    env.filters["fmt_ratio"] = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "—"
    env.filters["fmt_mktcap"] = _fmt_mktcap
    env.filters["tojson_safe"] = lambda v: json.dumps(v, default=str)
    return env


def _build_market_overview_svgs(
    price_frames: dict, manifest: RunManifest,
) -> dict[str, str]:
    """Render SPY and VIX charts for the market context panel.

    SPY chart shows the regime visually, with distribution days highlighted.
    VIX chart shows fear gauge with horizontal threshold lines (15/20/30).
    """
    out: dict[str, str] = {}
    if not price_frames:
        return out
    from canslim.svg_chart import render_svg, VISIBLE_SESSIONS

    # SPY (or whatever the market index is — try common ones)
    for idx_sym in ("SPY", "^GSPC", "^HSI"):
        spy_df = price_frames.get(idx_sym)
        if spy_df is not None and not spy_df.empty:
            # Mark distribution days as red vertical lines in the visible window
            v_markers = []
            try:
                window = spy_df.tail(VISIBLE_SESSIONS)
                closes = window["close"].astype(float).values
                vols = window["volume"].astype(float).values
                for i in range(1, len(closes)):
                    if vols[i] > vols[i - 1] and closes[i] < closes[i - 1] * 0.998:
                        v_markers.append((i, "#c62828"))
            except Exception:
                pass
            try:
                svg = render_svg(idx_sym, spy_df, None, vertical_markers=v_markers, height=300)
                if svg:
                    out["__market__"] = svg
            except Exception:
                pass
            break

    # VIX with threshold lines at 15 / 20 / 30
    vix_df = price_frames.get("^VIX")
    if vix_df is not None and not vix_df.empty:
        try:
            svg = render_svg(
                "^VIX", vix_df, None,
                horizontal_lines=[
                    (15.0, "#0277bd", "complacent <15"),
                    (20.0, "#e9740b", "elevated >20"),
                    (30.0, "#c62828", "fear >30"),
                ],
                height=260,
            )
            if svg:
                out["__vix__"] = svg
        except Exception:
            pass

    return out


def _build_inline_svgs(
    results: list[ScanResult], price_frames: dict
) -> dict[str, str]:
    """Render inline SVG markup per ticker using svg_chart.render_svg."""
    if not price_frames:
        return {}
    from canslim.svg_chart import render_svg
    out: dict[str, str] = {}
    for r in results:
        df = price_frames.get(r.ticker)
        if df is None:
            continue
        try:
            svg = render_svg(r.ticker, df, r)
            if svg:
                out[r.ticker] = svg
        except Exception:
            continue
    return out


def _build_chart_data_uris(
    chart_paths: dict[str, Path], embed_base64: bool
) -> dict[str, str]:
    """Build {ticker: data:image/png;base64,...} for embedding."""
    if not embed_base64:
        return {t: str(p) for t, p in chart_paths.items()}
    import base64
    out = {}
    for ticker, path in chart_paths.items():
        try:
            data = path.read_bytes()
            encoded = base64.b64encode(data).decode("ascii")
            out[ticker] = f"data:image/png;base64,{encoded}"
        except Exception:
            continue
    return out


def _bucket_candidates(
    candidates: list[ScanResult],
) -> tuple[list[ScanResult], list[ScanResult], list[ScanResult]]:
    """Classify near-match candidates by actionability."""
    buyable: list[ScanResult] = []
    watchlist: list[ScanResult] = []
    basing: list[ScanResult] = []
    for r in candidates:
        # Find primary pattern with a pivot and dist
        p = _primary_pattern(r)
        if p is None:
            basing.append(r)
            continue
        dist = p.evidence.get("dist_to_pivot_pct") if isinstance(p.evidence, dict) else None
        if not isinstance(dist, (int, float)):
            basing.append(r)
            continue
        if -0.05 <= dist <= 0.05:
            buyable.append(r)
        else:
            watchlist.append(r)
    return buyable, watchlist, basing


def _regime_index_label(manifest: RunManifest) -> str:
    uname = (manifest.universe_name or "").lower()
    if uname.startswith("hk_"):
        return "HSI"
    if uname.startswith("us_") or uname == "sp500":
        return "SPY"
    return "Index"
