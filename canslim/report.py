from __future__ import annotations

import base64
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from canslim.charts import render_chart
from canslim.deepdive import emit_deepdives
from canslim.models import RunManifest, ScanResult

LETTERS = ["C", "A", "N", "S", "L", "I", "M"]


def write_run(
    out_dir: str | Path,
    results: list[ScanResult],
    manifest: RunManifest,
    universe_snapshot: list[str],
    top_n_near_matches: int = 20,
    price_frames: Optional[dict[str, pd.DataFrame]] = None,
    embed_charts_base64: bool = True,
    generate_pdf: bool = True,
) -> Path:
    run_dir = Path(out_dir) / "runs" / manifest.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # universe snapshot
    (run_dir / "universe.csv").write_text(
        "ticker\n" + "\n".join(universe_snapshot) + ("\n" if universe_snapshot else "")
    )

    # manifest
    (run_dir / "run_manifest.json").write_text(
        manifest.model_dump_json(indent=2)
    )

    # parquet for downstream consumers
    _write_results_parquet(run_dir / "results.parquet", results)

    # per-ticker charts for matches + top near-matches
    charts_dir = run_dir / "charts"
    chart_paths = _render_charts(results, price_frames, charts_dir, top_n_near_matches)

    # markdown report
    report_path = run_dir / "report.md"
    report_path.write_text(_render_markdown(
        results, manifest,
        top_n_near_matches=top_n_near_matches,
        chart_paths=chart_paths,
        embed_base64=embed_charts_base64,
    ))

    # Per-ticker deep-dive .md files for each full match + pass-4/5 + buyable-zone candidate
    emit_deepdives(
        results, manifest, run_dir,
        price_frames=price_frames,
        chart_paths=chart_paths,
        embed_base64=embed_charts_base64,
    )

    # HTML rendering (primary output going forward; markdown remains for grep)
    try:
        from canslim.html_report import render_html
        html = render_html(
            results, manifest,
            top_n_near_matches=top_n_near_matches,
            chart_paths=chart_paths,
            embed_base64=embed_charts_base64,
            price_frames=price_frames,
        )
        (run_dir / "index.html").write_text(html)
    except Exception as e:  # pragma: no cover — defensive
        import logging
        logging.getLogger("canslim.report").warning("HTML generation failed: %s", e)

    # PDF rendering (opt-in; not auto-emitted now that HTML is primary)
    if generate_pdf:
        try:
            from canslim.pdf import render_pdf
            render_pdf(report_path)
        except Exception as e:  # pragma: no cover — defensive
            import logging
            logging.getLogger("canslim.report").warning("PDF generation failed: %s", e)

    return report_path


def _render_charts(
    results: list[ScanResult],
    price_frames: Optional[dict[str, pd.DataFrame]],
    charts_dir: Path,
    top_n: int,
) -> dict[str, Path]:
    """Render charts for full matches + top N near-matches. Returns {ticker: png_path}."""
    if price_frames is None:
        return {}
    scanned = [r for r in results if r.status == "scanned"]
    scanned.sort(key=lambda r: -r.composite_score)
    matches = [r for r in scanned if r.passed]
    near = [r for r in scanned if not r.passed][:top_n]
    targets = list({r.ticker: r for r in matches + near}.values())

    out: dict[str, Path] = {}
    for r in targets:
        df = price_frames.get(r.ticker)
        if df is None:
            continue
        png = render_chart(r.ticker, df, r, charts_dir)
        if png is not None:
            out[r.ticker] = png
    return out


def _write_results_parquet(path: Path, results: list[ScanResult]) -> None:
    if not results:
        path.write_bytes(b"")
        return
    rows = []
    for r in results:
        row = {
            "schema_version": r.schema_version,
            "ticker": r.ticker,
            "as_of": r.as_of.isoformat(),
            "passed": r.passed,
            "composite_score": r.composite_score,
            "status": r.status,
            "error": r.error,
            "market_cap": r.market_cap,
        }
        for letter in LETTERS:
            cr = r.criteria.get(letter)
            if cr is None:
                continue
            row[f"{letter}_passed"] = cr.passed
            row[f"{letter}_score"] = cr.score
            row[f"{letter}_value"] = cr.value
            row[f"{letter}_reason"] = cr.reason
            row[f"{letter}_evidence"] = json.dumps(cr.evidence, default=str)
        row["patterns"] = ",".join(p.name for p in r.patterns) if r.patterns else ""
        row["patterns_detail"] = json.dumps(
            [p.model_dump(mode="json") for p in r.patterns], default=str
        ) if r.patterns else ""
        row["ad_grade"] = r.ad_grade
        row["ad_ratio"] = r.ad_ratio
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def _render_markdown(
    results: list[ScanResult],
    manifest: RunManifest,
    top_n_near_matches: int = 20,
    chart_paths: Optional[dict[str, Path]] = None,
    embed_base64: bool = True,
) -> str:
    chart_paths = chart_paths or {}
    chart_refs = _build_chart_refs(chart_paths, embed_base64=embed_base64)
    matches = [r for r in results if r.passed]
    matches.sort(key=lambda r: -r.composite_score)

    regime = manifest.market_regime
    # Label the regime line with the actual benchmark for this run (SPY for US, ^HSI for HK, etc.)
    # The regime carries the numbers; the manifest carries the benchmark ticker via provider_versions or universe.
    # Fall back to "Index" if we can't determine it.
    idx_label = "Index"
    uname = (manifest.universe_name or "").lower()
    if uname.startswith("hk_"):
        idx_label = "HSI"
    elif uname.startswith("us_") or uname == "sp500":
        idx_label = "SPY"
    regime_line = (
        f"- **Market (M):** {'UPTREND' if regime and regime.uptrend else 'CAUTION'} — "
        f"{idx_label} close {regime.spy_close:.2f}, 50d {regime.spy_sma50:.2f}, 200d {regime.spy_sma200:.2f} "
        f"({regime.reason})" if regime else "- **Market (M):** unknown"
    )

    lines: list[str] = []
    lines.append(f"# CANSLIM Scan — {manifest.run_id}")
    lines.append("")
    lines.append(f"- Universe: **{manifest.universe_name}** ({manifest.universe_size} tickers)")
    lines.append(f"- Candidates after pre-filter: {manifest.candidates_after_prefilter}")
    lines.append(f"- Matches: **{manifest.matches}** | scanned: {manifest.scanned} | pending budget: {manifest.pending_budget} | errors: {manifest.errored}")
    if manifest.fmp_budget_remaining is not None:
        lines.append(f"- FMP budget used this run: {manifest.fmp_budget_used} | remaining today: {manifest.fmp_budget_remaining}")
    lines.append(regime_line)
    lines.append(f"- Config hash: `{manifest.config_hash}`")
    lines.append("")

    if matches:
        lines.append("## Matches (all gates passed)")
        lines.append("")
        lines.append("| Ticker | Score | Passes | Close | Mkt Cap | 52w-high dist | C YoY | 3y CAGR | RS %ile | Inst % |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in matches:
            lines.append(_summary_row(r))
        lines.append("")
    else:
        lines.append("_No tickers passed all gate criteria in this run._")
        lines.append("")

    # Near-matches — always show regardless of full-match count
    lines.extend(_near_matches_section(results, top_n=top_n_near_matches, exclude_full_matches=bool(matches)))

    # Leadership-override watchlist — tickers where a gate passed via conditional
    # relaxation (turnaround + RS/pattern, or pattern-aware S dry-up). These are
    # below the strict-CANSLIM cutoff but match O'Neil's spirit-of-the-rules setups.
    lines.extend(_leadership_override_section(results))

    # Bucket scanned tickers by actionability (buyable / watchlist / basing)
    scanned_sorted = sorted([r for r in results if r.status == "scanned"], key=lambda r: -r.composite_score)
    match_set = {r.ticker for r in matches}
    candidates_pool = [r for r in scanned_sorted if r.ticker not in match_set][:top_n_near_matches]
    buyable, watchlist, basing = _bucket_candidates(candidates_pool)

    if matches or buyable or watchlist or basing:
        lines.append("## Candidate detail — sorted by actionability")
        lines.append("")
        lines.append("_**Buyable now**: pattern present and price within -5% to +5% of the pivot._")
        lines.append("_**Watchlist**: setup exists but extended past pivot, or waiting for volume confirmation._")
        lines.append("_**Basing**: strong fundamentals, no clean pivot yet — re-check in 2-4 weeks._")
        lines.append("")

    def _emit(section_name: str, rows: list[ScanResult]) -> None:
        if not rows:
            return
        lines.append(f"### {section_name} ({len(rows)})")
        lines.append("")
        for r in rows:
            lines.extend(_candidate_block(r, chart_refs.get(r.ticker)))

    _emit("Full matches", matches)
    _emit("Buyable now", buyable)
    _emit("Watchlist", watchlist)
    _emit("Basing (no pivot yet)", basing)

    pending = [r for r in results if r.status == "pending_budget"]
    if pending:
        lines.append("## Skipped (pending budget)")
        lines.append("")
        lines.append(", ".join(sorted({r.ticker for r in pending})))
        lines.append("")

    # Unknown market cap — fail-closed bucket. These names cleared everything the
    # scanner could check, but we could NOT confirm they clear the $1B floor (the
    # cap field was unavailable). They are NEVER published as matches; surfaced
    # here so the chairman can eyeball what was set aside rather than silently
    # dropped.
    needs_review = [r for r in results if r.status == "unknown_market_cap"]
    if needs_review:
        lines.append(f"## Unknown market cap — needs review ({len(needs_review)})")
        lines.append("")
        lines.append("_Excluded from Matches (cannot confirm >= $1B floor). Verify cap manually._")
        lines.append("")
        lines.append(", ".join(sorted({r.ticker for r in needs_review})))
        lines.append("")

    lines.extend(_data_integrity_section(results, manifest))

    return "\n".join(lines)


def _data_integrity_section(results: list[ScanResult], manifest: RunManifest) -> list[str]:
    # "skipped_missing_data" = no scan possible (no prices / pre-filtered out).
    # "rejected_market_cap"  = scan skipped on purpose: cap KNOWN and below floor.
    # Both belong in the set-aside table; their status_reason already reads right.
    skipped = [r for r in results if r.status in ("skipped_missing_data", "rejected_market_cap")]
    errors = list(manifest.errors)
    if not errors and not skipped and not manifest.fetch_summary:
        return []

    lines = ["## Data integrity", ""]
    if manifest.fetch_summary:
        lines.append("| Kind | Cache hits | Fresh fetches | Failures | Skipped (neg. cache) |")
        lines.append("|---|---|---|---|---|")
        for fs in manifest.fetch_summary:
            lines.append(
                f"| {fs.kind} | {fs.cache_hits} | {fs.fresh_fetches} | {fs.failures} | {fs.skipped_negative_cache} |"
            )
        lines.append("")

    if errors:
        by_kind = Counter((e.kind, e.provider) for e in errors)
        lines.append("**Errors by (kind, provider):**")
        lines.append("")
        lines.append("| Kind | Provider | Count |")
        lines.append("|---|---|---|")
        for (kind, provider), n in sorted(by_kind.items(), key=lambda x: -x[1]):
            lines.append(f"| {kind} | {provider} | {n} |")
        lines.append("")

        # Sample up to 30 specific errors
        sample = errors[:30]
        lines.append("**Sample errors (first 30):**")
        lines.append("")
        lines.append("| Ticker | Kind | Provider | Error |")
        lines.append("|---|---|---|---|")
        for e in sample:
            err_short = (e.error[:120] + "…") if len(e.error) > 120 else e.error
            lines.append(f"| {e.ticker} | {e.kind} | {e.provider} | {err_short} |")
        if len(errors) > 30:
            lines.append(f"| … | | | {len(errors) - 30} more errors elided |")
        lines.append("")

    if skipped:
        reason_counts = Counter(r.status_reason or "unspecified" for r in skipped)
        lines.append("**Skipped tickers (no scan performed):**")
        lines.append("")
        lines.append("| Reason | Count |")
        lines.append("|---|---|")
        for reason, n in reason_counts.most_common():
            lines.append(f"| {reason} | {n} |")
        lines.append("")

        # Show tickers that were skipped with a real (non-prefilter) reason
        real_misses = [r for r in skipped if r.status_reason and "pre-filter" not in (r.status_reason or "")]
        if real_misses:
            lines.append("_Tickers skipped due to data issues (not just pre-filter):_")
            lines.append("")
            lines.append(", ".join(sorted(r.ticker for r in real_misses[:60])))
            if len(real_misses) > 60:
                lines.append(f"… and {len(real_misses) - 60} more.")
            lines.append("")

    lines.append("_Re-run with `--force-refresh` to bypass both positive and negative caches for all tickers._")
    lines.append("")
    return lines


def _near_matches_section(results: list[ScanResult], top_n: int, exclude_full_matches: bool) -> list[str]:
    scanned = [r for r in results if r.status == "scanned"]
    if exclude_full_matches:
        scanned = [r for r in scanned if not r.passed]
    scanned.sort(key=lambda r: -r.composite_score)
    top = scanned[:top_n]
    if not top:
        return []

    lines = [f"## Top {len(top)} by composite score" + (" (near-matches)" if exclude_full_matches else ""), ""]
    lines.append("| # | Ticker | Score | Mkt Cap | C A N S L I M | Gates failed | C YoY | 3y CAGR | RS %ile | Patterns | S reason |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for idx, r in enumerate(top, start=1):
        flags = _gate_flags(r)
        failed = _failed_gates(r)
        c = r.criteria.get("C")
        a = r.criteria.get("A")
        l = r.criteria.get("L")
        s = r.criteria.get("S")
        c_yoy_raw = c.evidence.get("latest_yoy") if c else None
        c_yoy_s = "turnaround" if c_yoy_raw == "turnaround" else f"{_num(c_yoy_raw):.1%}"
        cagr_raw = a.evidence.get("cagr") if a else None
        cagr_s = "n/a" if cagr_raw is None else f"{_num(cagr_raw):.1%}"
        rs = _num(l.value if l else None)
        s_reason = (s.reason if s else "") or ""
        patterns_cell = _patterns_cell(r.patterns) or "—"
        lines.append(
            f"| {idx} | {r.ticker} | {r.composite_score:.2f} | {_fmt_mktcap(r.market_cap)} | {flags} | {failed or '—'} | "
            f"{c_yoy_s} | {cagr_s} | {rs:.2f} | {patterns_cell} | {s_reason[:40]} |"
        )
    lines.append("")
    lines.append("_Legend: uppercase letter = gate passed, lowercase = failed. N/M are info-only._")
    lines.append("")

    # Dedicated patterns section for any scanned ticker with ≥1 detected pattern
    with_patterns = [r for r in scanned if r.patterns]
    if with_patterns:
        with_patterns.sort(key=lambda r: -max((p.confidence for p in r.patterns), default=0.0))
        lines.append("## Detected chart patterns")
        lines.append("")
        lines.append("| Ticker | Score | Pattern | Confidence | Pivot | Dist to pivot | Evidence |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in with_patterns[:top_n]:
            for p in sorted(r.patterns, key=lambda x: -x.confidence):
                dist = p.evidence.get("dist_to_pivot_pct")
                dist_s = f"{dist:+.1%}" if isinstance(dist, (int, float)) else "—"
                pivot_s = f"{p.pivot:.2f}" if p.pivot is not None else "—"
                ev = {k: v for k, v in p.evidence.items() if k not in ("dist_to_pivot_pct",)}
                lines.append(
                    f"| {r.ticker} | {r.composite_score:.2f} | {p.name} | {p.confidence:.2f} | "
                    f"{pivot_s} | {dist_s} | `{json.dumps(ev, default=str)[:100]}` |"
                )
        lines.append("")

    return lines


def _override_reasons(r: ScanResult) -> list[str]:
    """Return the override types fired for this result, e.g. ['A:leadership', 'S:pattern']."""
    used: list[str] = []
    a = r.criteria.get("A")
    if a and isinstance(a.evidence, dict) and a.evidence.get("override_used"):
        used.append("A:leadership")
    s = r.criteria.get("S")
    if s and isinstance(s.evidence, dict) and s.evidence.get("pattern_override"):
        used.append("S:pattern")
    return used


def _leadership_override_section(results: list[ScanResult]) -> list[str]:
    """List tickers where any gate passed via leadership/pattern override.

    These are O'Neil-spirit candidates that the strict gates would otherwise hide:
    turnaround stocks with top-decile RS + pattern, or constructive-base stocks
    with drying volume. Sorted by how many gates pass overall, then composite.
    """
    overrides = [
        (r, _override_reasons(r))
        for r in results
        if r.status == "scanned"
    ]
    overrides = [(r, used) for r, used in overrides if used]
    if not overrides:
        return []

    def _gate_pass_count(r: ScanResult) -> int:
        return sum(1 for L in LETTERS if r.criteria.get(L) and r.criteria[L].is_gate and r.criteria[L].passed)

    overrides.sort(key=lambda pair: (-_gate_pass_count(pair[0]), -pair[0].composite_score))

    lines: list[str] = ["## Leadership-override watchlist", ""]
    lines.append(
        "_Tickers where a strict gate failed but a relaxation path applies — "
        "turnaround names with top-decile RS + a high-confidence pattern (A:leadership), "
        "or constructive-base names with drying volume (S:pattern). These match O'Neil's "
        "\"rally off a turning point\" setups that the strict gates would otherwise hide._"
    )
    lines.append("")
    lines.append("| Ticker | Score | Gates | C A N S L I M | Override(s) | RS %ile | Pattern | A reason |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r, used in overrides:
        flags = _gate_flags(r)
        gates_passed = _gate_pass_count(r)
        l = r.criteria.get("L")
        rs = _num(l.value if l else None)
        a = r.criteria.get("A")
        a_reason = (a.reason if a else "")[:60] or "—"
        # Show the highest-confidence pattern from any override that fired
        pattern_cell = _patterns_cell(r.patterns) or "—"
        lines.append(
            f"| {r.ticker} | {r.composite_score:.2f} | {gates_passed}/5 | {flags} | "
            f"{', '.join(used)} | {rs:.2f} | {pattern_cell} | {a_reason} |"
        )
    lines.append("")
    return lines


def _bucket_candidates(candidates: list[ScanResult]) -> tuple[list[ScanResult], list[ScanResult], list[ScanResult]]:
    """Classify near-match candidates by actionability today.

    Returns (buyable, watchlist, basing). Ordering within each bucket is by composite_score desc.

    Rules:
      * Buyable: has ≥1 detected pattern whose dist_to_pivot_pct is between -0.05 and +0.05.
        Price is either just below the trigger ("about to fire") or within O'Neil's
        max-chase-zone (within 5% past pivot, still allowable).
      * Watchlist: has a pattern but price is either >5% past pivot (extended, don't chase)
        or further than -5% below pivot with volume not yet surging.
      * Basing: no actionable pattern detected, but still on the near-match list (e.g. passes
        several gates). Worth watching for a base to form.
    """
    buyable: list[ScanResult] = []
    watchlist: list[ScanResult] = []
    basing: list[ScanResult] = []

    for r in candidates:
        if not r.patterns:
            basing.append(r)
            continue
        top = max(r.patterns, key=lambda p: p.confidence)
        dist = top.evidence.get("dist_to_pivot_pct")
        if dist is None:
            watchlist.append(r)
        elif -0.05 <= dist <= 0.05:
            buyable.append(r)
        else:
            watchlist.append(r)

    return buyable, watchlist, basing


def _build_chart_refs(chart_paths: dict[str, Path], embed_base64: bool) -> dict[str, str]:
    """Return {ticker: markdown-image-src}. Data URIs when embed_base64=True, else relative paths."""
    out: dict[str, str] = {}
    for ticker, png_path in chart_paths.items():
        if embed_base64:
            try:
                data = png_path.read_bytes()
            except Exception:
                out[ticker] = f"charts/{png_path.name}"
                continue
            encoded = base64.b64encode(data).decode("ascii")
            out[ticker] = f"data:image/png;base64,{encoded}"
        else:
            out[ticker] = f"charts/{png_path.name}"
    return out


def _patterns_cell(patterns) -> str:
    if not patterns:
        return ""
    shorten = {"cup_with_handle": "cup+H", "double_bottom": "2bot", "flat_base": "flat"}
    return ", ".join(
        f"{shorten.get(p.name, p.name)}({p.confidence:.2f})"
        for p in sorted(patterns, key=lambda x: -x.confidence)[:3]
    )


def _gate_flags(r: ScanResult) -> str:
    """Compact CANSLIM letters: UPPER=pass, lower=fail, ?=abstain (no data)."""
    out = []
    for letter in LETTERS:
        cr = r.criteria.get(letter)
        if cr is None:
            out.append("·")
        elif not cr.data_available:
            out.append("?")
        elif cr.passed:
            out.append(letter)
        else:
            out.append(letter.lower())
    return " ".join(out)


def _failed_gates(r: ScanResult) -> str:
    return "".join(
        letter for letter in LETTERS
        if (cr := r.criteria.get(letter)) is not None and cr.is_gate and not cr.passed
    )


def _summary_row(r: ScanResult) -> str:
    passes = "".join(
        letter if (r.criteria.get(letter) and r.criteria[letter].passed) else letter.lower()
        for letter in LETTERS
    )
    c = r.criteria.get("C")
    a = r.criteria.get("A")
    n = r.criteria.get("N")
    l = r.criteria.get("L")
    i = r.criteria.get("I")
    close = _num(n.evidence.get("close") if n else None)
    dist = _num(n.evidence.get("dist_to_52w_high_pct") if n else None)
    c_yoy_raw = c.evidence.get("latest_yoy") if c else None
    c_yoy_s = "turnaround" if c_yoy_raw == "turnaround" else f"{_num(c_yoy_raw):.1%}"
    cagr_raw = a.evidence.get("cagr") if a else None
    cagr_s = "n/a" if cagr_raw is None else f"{_num(cagr_raw):.1%}"
    rs_pct = _num(l.value if l else None)
    inst_pct = _num(i.value if i else None)
    return (
        f"| {r.ticker} | {r.composite_score:.2f} | `{passes}` | {close:.2f} | "
        f"{_fmt_mktcap(r.market_cap)} | "
        f"{dist:.1%} | {c_yoy_s} | {cagr_s} | {rs_pct:.2f} | {inst_pct:.1%} |"
    )


def _fmt_mktcap(v) -> str:
    """Compact USD market cap: $2.1B, $340B, $1.2T. '—' when unknown.

    Canonical implementation — html_report imports this (deduped, was copied).
    """
    if not isinstance(v, (int, float)) or v != v or v <= 0:
        return "—"
    if v >= 1e12:
        return f"${v / 1e12:.2f}T"
    return f"${v / 1e9:.1f}B"


def _num(v) -> float:
    """Coerce any value (including 'turnaround' sentinel or None) to a safe float for formatting."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v) if v == v else 0.0  # NaN check
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _candidate_block(r: ScanResult, chart_ref: Optional[str]) -> list[str]:
    status_tag = "MATCH" if r.passed else "near-miss"
    gate_flags = _gate_flags(r)
    failed = _failed_gates(r) or "—"
    parts: list[str] = []
    parts.append(f"### {r.ticker}  ·  score {r.composite_score:.2f}  ·  `{gate_flags}`  ·  {status_tag}  ·  fails: {failed}")
    parts.append("")
    if chart_ref is not None:
        parts.append(f"![{r.ticker} chart]({chart_ref})")
        parts.append("")

    if r.ad_grade is not None:
        ad_descr = {
            "A": "heavy accumulation", "B": "moderate accumulation",
            "C": "neutral / balanced", "D": "moderate distribution", "E": "heavy distribution",
        }.get(r.ad_grade, "?")
        parts.append(f"**Accumulation/Distribution: `{r.ad_grade}`** ({ad_descr}, up/down flow ratio {r.ad_ratio:.2f})")
        parts.append("")

    parts.append("**CANSLIM brief**")
    parts.append("")
    for letter in LETTERS:
        cr = r.criteria.get(letter)
        if cr is None:
            continue
        gate = "GATE" if cr.is_gate else "info"
        check = "PASS" if cr.passed else "fail"
        value_s = _format_value_for_letter(letter, cr.value)
        parts.append(
            f"- **{letter}** ({gate}, {check}) {value_s}— {cr.reason or '(ok)'}"
        )
    parts.append("")

    if r.patterns:
        parts.append("**Detected patterns**")
        parts.append("")
        for p in sorted(r.patterns, key=lambda x: -x.confidence):
            dist = p.evidence.get("dist_to_pivot_pct")
            dist_s = f" (dist {dist:+.1%})" if isinstance(dist, (int, float)) else ""
            pivot_s = f" pivot {p.pivot:.2f}" if p.pivot is not None else ""
            parts.append(f"- {p.name}{pivot_s}{dist_s}, confidence {p.confidence:.2f}")
        parts.append("")
    else:
        parts.append("_No chart pattern detected above confidence threshold._")
        parts.append("")

    parts.extend(_volume_price_narrative(r))
    parts.extend(_entry_plan(r))

    if r.errors:
        parts.append(f"_Data warnings: {len(r.errors)} fetch issue(s) during this scan._")
        parts.append("")
    parts.append("")
    return parts


def _volume_price_narrative(r: ScanResult) -> list[str]:
    """Render a 'Price & volume action' subsection summarizing recent tape behavior."""
    n = r.criteria.get("N")
    s = r.criteria.get("S")
    if n is None and s is None:
        return []

    n_ev = n.evidence if n and isinstance(n.evidence, dict) else {}
    s_ev = s.evidence if s and isinstance(s.evidence, dict) else {}

    close = n_ev.get("close")
    high_52w = n_ev.get("high_52w")
    dist_high = n_ev.get("dist_to_52w_high_pct")
    recent_vol_ratio = n_ev.get("recent_vol_ratio")
    adv10 = s_ev.get("adv10")
    adv50 = s_ev.get("adv50")
    ratio = s_ev.get("ratio")
    breakout_flag = n_ev.get("breakout")

    lines: list[str] = ["**Price & volume action**", ""]
    bullets: list[str] = []

    if isinstance(close, (int, float)) and isinstance(high_52w, (int, float)) and high_52w > 0:
        if isinstance(dist_high, (int, float)):
            label = "**at 52w high**" if dist_high < 0.005 else f"{dist_high:.1%} off 52w high"
            bullets.append(f"Close ${close:,.2f} — {label} (52w peak ${high_52w:,.2f})")

    if isinstance(adv10, (int, float)) and isinstance(adv50, (int, float)) and adv50 > 0:
        adv_ratio = adv10 / adv50
        if adv_ratio < 0.95:
            adv_label = "**drying up** (constructive in a base/handle)"
        elif adv_ratio >= 1.20:
            adv_label = "**elevated** (institutional accumulation)"
        else:
            adv_label = "near 50-day average"
        bullets.append(
            f"Volume trend: ADV10 / ADV50 = {adv_ratio:.2f}× — {adv_label}"
        )
        if isinstance(recent_vol_ratio, (int, float)):
            mark = " ✓ confirmed" if recent_vol_ratio >= 1.4 else " — below 1.4× confirmation threshold"
            bullets.append(
                f"Latest session: {recent_vol_ratio:.2f}× ADV50{mark}"
            )

    if breakout_flag:
        bullets.append(
            "**N-criterion flagged BREAKOUT**: within 5% of 52w high AND volume ≥1.4× ADV50"
        )

    if r.ad_grade in ("A", "B"):
        bullets.append(
            f"Accumulation/distribution `{r.ad_grade}` (up/down flow {r.ad_ratio:.2f}) — "
            f"{'institutions buying' if r.ad_grade == 'A' else 'mild buying pressure'}"
        )
    elif r.ad_grade in ("D", "E"):
        bullets.append(
            f"⚠️ Accumulation/distribution `{r.ad_grade}` — institutions selling; pattern at risk"
        )

    if not bullets:
        return []

    for b in bullets:
        lines.append(f"- {b}")
    lines.append("")
    return lines


def _entry_plan(r: ScanResult) -> list[str]:
    """Render an 'Entry plan' subsection — actionable trigger / stop / sizing for buyable setups.

    Only emitted when the candidate has a detected pattern with a numeric pivot.
    Logic adapts to where the stock sits relative to pivot:
      * dist > +5%: setup forming, wait for breakout
      * 0% < dist <= +5%: approaching pivot, set alert
      * -5% <= dist <= 0%: in buy zone post-breakout (textbook entry)
      * dist < -5%: extended past +5% buy zone, chase risk
    """
    if not r.patterns:
        return []
    # Use highest-confidence pattern with a numeric pivot
    candidates = [p for p in r.patterns if p.pivot is not None]
    if not candidates:
        return []
    p = max(candidates, key=lambda x: x.confidence)
    pivot = float(p.pivot or 0)
    if pivot <= 0:
        return []
    dist = p.evidence.get("dist_to_pivot_pct") if isinstance(p.evidence, dict) else None
    n_crit = r.criteria.get("N")
    n_ev = n_crit.evidence if (n_crit is not None and isinstance(n_crit.evidence, dict)) else {}
    close = n_ev.get("close")

    buy_zone_low = pivot
    buy_zone_high = pivot * 1.05
    stop_loss = pivot * 0.93  # -7% from pivot

    lines: list[str] = ["**Entry plan**", ""]

    # Status banner based on dist
    if isinstance(dist, (int, float)):
        if dist > 0.05:
            banner = (
                f"📍 **Setup forming** — close ${close:,.2f} is {dist:+.1%} below pivot. "
                f"Set price alert at ${pivot:,.2f}."
            )
        elif dist > 0.0:
            banner = (
                f"⏳ **Approaching pivot** — close ${close:,.2f} is {dist:+.1%} below pivot. "
                f"Watch for break above ${pivot:,.2f} on heavy volume."
            )
        elif dist >= -0.05:
            banner = (
                f"✅ **In buy zone** — close ${close:,.2f} is {-dist:+.1%} past pivot ${pivot:,.2f}. "
                f"Textbook O'Neil entry zone."
            )
        else:
            banner = (
                f"⚠️ **Extended** — close ${close:,.2f} is {-dist:.1%} past pivot ${pivot:,.2f}, "
                f"outside the +5% buy zone. Chase risk: wait for first pullback to the 21-day MA, "
                f"or for a new base to form."
            )
        lines.append(banner)
        lines.append("")

    # Concrete trigger / zone / stop
    lines.append(f"- **Trigger**: close above ${pivot:,.2f} on volume ≥ 1.4× ADV50")
    lines.append(
        f"- **Buy zone**: ${buy_zone_low:,.2f} → ${buy_zone_high:,.2f} (pivot to +5% past pivot)"
    )
    lines.append(
        f"- **Initial stop**: ${stop_loss:,.2f} (-7% from pivot) — non-negotiable per O'Neil rule"
    )

    # Position sizing hint based on gate quality + pattern confidence
    if r.passed:
        sizing = "full position (all gates pass + pattern confirmed)"
    elif p.confidence >= 0.75:
        sizing = "full position (high pattern confidence; near-miss is on a soft gate)"
    elif p.confidence >= 0.60:
        sizing = "half position (moderate pattern confidence)"
    else:
        sizing = "watchlist only (low pattern confidence; wait for second signal)"
    lines.append(f"- **Position size**: {sizing}")

    # Invalidation level
    lines.append(
        f"- **Invalidation**: close below ${pivot * 0.95:,.2f} (-5% from pivot) on heavy volume "
        f"= setup failed; do not re-enter without a new base"
    )
    lines.append("")
    return lines


def _format_value_for_letter(letter: str, value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):  # e.g. "turnaround" sentinel
        return f"[{value}] "
    if isinstance(value, float) and value != value:  # NaN
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if letter in ("C", "A", "I"):
        return f"[{v:.1%}] "
    if letter in ("L", "S"):
        return f"[{v:.2f}] "
    return ""


def _brief(evidence: dict) -> str:
    if not evidence:
        return ""
    short = {}
    for k, v in evidence.items():
        if isinstance(v, float):
            short[k] = round(v, 4)
        elif isinstance(v, (list, tuple)):
            short[k] = [round(x, 4) if isinstance(x, float) else x for x in v][:6]
        else:
            short[k] = v
    return json.dumps(short, default=str)


__all__ = ["write_run"]
