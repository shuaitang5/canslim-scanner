from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from canslim.config import Settings
from canslim.dashboard import render_dashboard
from canslim.monitor import evaluate_positions, render_monitor_report, snapshot_dict
from canslim.positions import PositionsFile
from canslim.report import write_run
from canslim.scanner import Scanner
from canslim.universe import load_universe

app = typer.Typer(add_completion=False, help="CANSLIM stock scanner.")
console = Console()
log = logging.getLogger("canslim")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load(config: Optional[Path]) -> Settings:
    try:
        return Settings.load(config)
    except FileNotFoundError as e:
        console.print(f"[red]config error:[/red] {e}")
        raise typer.Exit(code=2)


def _count_abstains(results) -> tuple[int, int]:
    """Count scanned tickers with a gate abstain, split by gating relevance.

    Returns ``(gated_abstains, inst_abstains)``:
      - ``gated_abstains`` counts a scanned ticker if ANY gate OTHER THAN
        institutional (I) abstained (data_available False). This is the figure
        the degraded threshold uses — it reflects genuine fundamental/float data
        gaps that should gate publishing.
      - ``inst_abstains`` counts tickers whose I-gate abstained, surfaced for
        transparency only. The I-signal comes from yfinance's crumbed get_info,
        which is heavily throttled on a shared datacenter IP, so I-abstains are a
        normal free-data-stack reality — a ticker that abstains on I still gets
        scanned and just can't be a *full* match. Gating on I-abstains would
        block a perfectly valid full-market page, so they are excluded above.
    """
    gated = 0
    inst = 0
    for r in results:
        if getattr(r, "status", None) != "scanned":
            continue
        crit = r.criteria
        if any(
            cr.is_gate and not cr.data_available and L != "I"
            for L, cr in crit.items()
        ):
            gated += 1
        i = crit.get("I")
        if i is not None and i.is_gate and not i.data_available:
            inst += 1
    return gated, inst


def _assess_run_quality(
    *,
    scanned: int,
    universe_size: int,
    abstained_scans: int,
    abstained_pct: float,
    fresh_price_failures: int,
    fresh_price_attempts: int,
    max_abstain_fraction: float,
    unknown_mcap: int = 0,
    rejected_mcap: int = 0,
    max_unknown_mcap_fraction: float = 0.25,
) -> dict:
    """Decide the run's quality verdict (exit-code / publish policy).

    Pure + filesystem-free so the exit-code policy can be unit-tested directly.
    Returns a dict: ``{"fatal": bool, "health_warn": [str], "info_notes": [str],
    "summary_color": str}``.

    Three outcomes (see also: the GitHub workflow scan step):

      FATAL    -> ``fatal=True`` and a ``health_warn``. A real scan did NOT
                  happen (0 tickers evaluated -> provider chain down). `publish`
                  refuses regardless; the run exits 2.
      DEGRADED -> a ``health_warn`` (no ``fatal``). A scan ran but a MEANINGFUL
                  fraction is suspect: in-run price-fetch throttling, OR abstains
                  at/above ``max_abstain_fraction``. The warning is written into
                  the manifest so `publish` refuses without --allow-degraded —
                  this is the guard that stops a throttled/empty page going live.
                  Exits 2.
      BENIGN   -> only ``info_notes`` (no ``health_warn``, no ``fatal``). A real
                  scan completed against the vast majority of tickers and only a
                  small fraction (below ``max_abstain_fraction``) abstained on a
                  transient hiccup (e.g. a yfinance "401 Invalid Crumb"). The run
                  SUCCEEDS and PUBLISHES — the common us_all case where a few of
                  thousands of tickers hit a transient data-quality quirk.

    Anything in ``health_warn`` blocks publish (via the manifest) AND triggers a
    non-zero exit. ``info_notes`` does neither.
    """
    health_warn: list[str] = []
    info_notes: list[str] = []
    fatal = False
    summary_color = "green"

    price_throttled = (
        fresh_price_attempts > 100
        and fresh_price_failures / max(fresh_price_attempts, 1) > 0.20
    )

    # FATAL: a "scan" that evaluated nothing isn't a real scan. NOTHING reached
    # the criteria stage -> the provider chain was down. (Distinct from a
    # universe that legitimately filtered everything out via cap/pre-filter,
    # which still produces scanned > 0 results.)
    if scanned == 0:
        fatal = True
        summary_color = "red"
        health_warn.append(
            f"scan evaluated 0 tickers (universe_size={universe_size}) — "
            "provider chain likely down; refusing to publish an empty page"
        )

    # DEGRADED: in-run price-fetch throttling on a meaningful fraction.
    if price_throttled:
        summary_color = "yellow" if not fatal else summary_color
        health_warn.append(
            f"{fresh_price_failures}/{fresh_price_attempts} price fetches failed this run — "
            "yfinance throttling likely; consider re-running with --force-refresh"
        )

    # DEGRADED: cap-fetch throttling collapsed the scanned set. A name lands in
    # `unknown_market_cap` ONLY when the (crumbed) market-cap fetch failed — the
    # same yfinance 401 throttling that hits a cold runner IP — so a large
    # `unknown_mcap` fraction of the cap-gate population means we couldn't even
    # determine eligibility for most of the universe and `scanned` collapsed far
    # below the real full-market scale. This is distinct from `rejected_mcap`
    # (cap KNOWN and below the $1B floor — a legitimate exclusion). Flagging it
    # degraded makes the workflow retry with a warmed cap cache, recovering the
    # universe instead of silently publishing a thin page.
    cap_gate_pop = scanned + unknown_mcap + rejected_mcap
    if cap_gate_pop > 100 and unknown_mcap / cap_gate_pop > max_unknown_mcap_fraction:
        summary_color = "yellow" if not fatal else summary_color
        health_warn.append(
            f"{unknown_mcap}/{cap_gate_pop} candidates had an UNKNOWN market cap "
            f"(>{max_unknown_mcap_fraction:.0%}) — yfinance cap-fetch throttling collapsed "
            f"the scanned set to {scanned}; re-run with warm caches to recover the universe"
        )

    # Abstains: benign below the threshold, degraded at/above it.
    if abstained_scans > 0:
        msg = (
            f"{abstained_scans} of {scanned} scanned tickers "
            f"({abstained_pct:.1%}) had gates abstain due to missing data "
            f"(institutional/fundamentals/float)."
        )
        if abstained_pct >= max_abstain_fraction:
            summary_color = "yellow" if not fatal else summary_color
            health_warn.append(msg + " Re-run with --force-refresh to retry.")
        else:
            info_notes.append(
                msg + f" Within tolerance ({max_abstain_fraction:.0%}) — treated as benign."
            )

    return {
        "fatal": fatal,
        "health_warn": health_warn,
        "info_notes": info_notes,
        "summary_color": summary_color,
    }


@app.command()
def scan(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to canslim.yaml"),
    universe: Optional[str] = typer.Option(None, "--universe", "-u", help="Universe name (sp500, us_all, custom)"),
    out_dir: Optional[Path] = typer.Option(None, "--out", "-o", help="Output dir override"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate API calls without spending budget"),
    force_refresh: bool = typer.Option(
        False, "--force-refresh",
        help="Bypass positive + negative caches; re-fetch every ticker from upstream.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a full scan and write report + parquet + manifest."""
    _setup_logging(verbose)
    settings = _load(config)
    u_name = universe or settings.scanner.default_universe
    out = str(out_dir or settings.scanner.out_dir)

    tickers = load_universe(u_name, settings)
    console.print(f"Loaded universe [bold]{u_name}[/bold] ({len(tickers)} tickers)")

    async def _run():
        scanner = Scanner(settings)
        try:
            results, manifest = await scanner.scan(tickers, dry_run=dry_run, force_refresh=force_refresh)
            return (
                results, manifest,
                getattr(scanner, "_price_frames", {}),
                getattr(scanner, "_market_overview_frames", {}),
            )
        finally:
            await scanner.close()

    results, manifest, price_frames, overview_frames = asyncio.run(_run())

    manifest.universe_name = u_name
    # Merge market overview frames (SPY, ^VIX) into price_frames so the HTML
    # report can render market-wide charts alongside per-candidate charts.
    if overview_frames:
        for k, v in overview_frames.items():
            if v is not None and k not in price_frames:
                price_frames[k] = v
    report_path = write_run(
        out, results, manifest, tickers,
        top_n_near_matches=settings.scanner.top_n_near_matches,
        price_frames=price_frames,
        embed_charts_base64=settings.scanner.embed_charts_base64,
        generate_pdf=settings.scanner.generate_pdf,
    )
    n_errors = len(manifest.errors)
    n_skipped_data = sum(1 for r in results if r.status == "skipped_missing_data")
    n_rejected_mcap = sum(1 for r in results if r.status == "rejected_market_cap")
    # Fail-closed $1B floor: cap unavailable -> excluded from matches, set aside.
    n_unknown_mcap = sum(1 for r in results if r.status == "unknown_market_cap")

    # Loud data-quality summary so silent fetch failures don't slip past.
    # The benign/degraded/fatal decision is made by `_assess_run_quality` below.
    #
    # The abstain rate measures genuine data-quality gaps that should gate
    # publishing — but it deliberately EXCLUDES the institutional (I) gate. The
    # I-signal comes from yfinance's crumbed `get_info` (heldPercentInstitutions),
    # which is heavily throttled on a shared datacenter IP, so a large fraction
    # of I-abstains is the normal reality of the free-data stack, NOT a broken
    # page: a ticker whose I-gate abstained still gets scanned, and it simply
    # can't become a *full* match (gate_pass_all stays False) — it falls into the
    # report's incomplete-data bucket. The matches that DO surface always have
    # full data. Counting I-abstains toward the degraded threshold would block a
    # perfectly valid full-market page just because institutional sponsorship
    # couldn't be confirmed for some names. Abstains on C/A/S/L (fundamentals/
    # float) DO count — those signal real fundamental-data problems.
    abstained_scans, inst_abstains = _count_abstains(results)
    abstained_pct = abstained_scans / max(manifest.scanned or 1, 1)

    # Distinguish "yfinance failed to fetch during this run" from "this ticker has
    # no recent prices ever" — only the former indicates a degraded run. The
    # us_all universe always carries a long tail of delisted/thinly-traded tickers
    # whose cached prices are empty; those aren't a quality signal.
    fresh_price_failures = 0
    fresh_price_attempts = 0
    for fs in manifest.fetch_summary or []:
        if fs.kind == "prices":
            fresh_price_failures += int(fs.failures or 0) + int(fs.skipped_negative_cache or 0)
            fresh_price_attempts += int(fs.fresh_fetches or 0) + int(fs.failures or 0)

    # ---- Exit-code / degrade policy ----------------------------------------
    # Decided by the pure `_assess_run_quality` helper so it's unit-testable
    # without live fetches. See that helper for the FATAL / DEGRADED / BENIGN
    # contract. `health_warn` is the degrade channel (blocks publish + exit 2);
    # `info_notes` is the benign FYI channel (never blocks, never exits non-zero).
    verdict = _assess_run_quality(
        scanned=manifest.scanned,
        universe_size=manifest.universe_size,
        abstained_scans=abstained_scans,
        abstained_pct=abstained_pct,
        fresh_price_failures=fresh_price_failures,
        fresh_price_attempts=fresh_price_attempts,
        max_abstain_fraction=settings.scanner.max_abstain_fraction,
        unknown_mcap=n_unknown_mcap,
        rejected_mcap=n_rejected_mcap,
        max_unknown_mcap_fraction=settings.scanner.max_unknown_mcap_fraction,
    )
    summary_color = verdict["summary_color"]
    health_warn = verdict["health_warn"]
    info_notes = verdict["info_notes"]
    fatal = verdict["fatal"]

    console.print(
        f"[{summary_color}]done[/{summary_color}] — matches={manifest.matches} scanned={manifest.scanned} "
        f"pending={manifest.pending_budget} errors={manifest.errored} "
        f"fetch_errors={n_errors} skipped_missing={n_skipped_data} "
        f"rejected_mcap={n_rejected_mcap} unknown_mcap={n_unknown_mcap} "
        f"abstained={abstained_scans} inst_abstained={inst_abstains}"
    )
    for w in health_warn:
        console.print(f"[yellow]⚠ data quality:[/yellow] {w}")
    for note in info_notes:
        console.print(f"[dim]· {note}[/dim]")

    html_path = report_path.parent / "index.html"
    if html_path.exists():
        console.print(f"html:   {html_path}")
    console.print(f"report: {report_path}")
    pdf_path = report_path.with_suffix(".pdf")
    if pdf_path.exists():
        console.print(f"pdf:    {pdf_path}")

    # Annotate manifest with degraded flag so `canslim publish` can refuse
    # without --allow-degraded. Done by re-writing run_manifest.json with
    # a non-schema extra field for now (avoids breaking pydantic strict mode).
    # Only DEGRADE/FATAL signals land here — benign abstains never poison the
    # manifest, so they never block publishing.
    if health_warn:
        manifest_path = report_path.parent / "run_manifest.json"
        try:
            import json as _json
            m = _json.loads(manifest_path.read_text())
            m["_data_quality_warnings"] = health_warn
            manifest_path.write_text(_json.dumps(m, indent=2, default=str))
        except Exception:
            pass

    # Exit non-zero ONLY for genuinely fatal/degraded conditions:
    #   - fatal: a real scan didn't happen (0 tickers evaluated), or
    #   - degraded: a meaningful fraction is suspect (price throttling, or
    #     abstains >= max_abstain_fraction).
    # A scan that completed with only a benign sub-threshold sliver of abstains
    # exits 0 and publishes — that's the whole point of the tolerance. No
    # blanket `|| true` is needed in the workflow; the degraded-report guard in
    # `publish` still blocks empty/throttled pages because those set health_warn.
    if fatal or health_warn:
        raise typer.Exit(code=2)


@app.command("check-providers")
def check_providers(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Ping each provider and print health."""
    _setup_logging(verbose)
    settings = _load(config)
    scanner = Scanner(settings)

    async def _run():
        try:
            return await scanner.health_check()
        finally:
            await scanner.close()

    report = asyncio.run(_run())
    table = Table(title="Providers")
    table.add_column("Provider")
    table.add_column("Status")
    table.add_column("Detail")
    any_bad = False
    for name, info in report.items():
        status = info.get("status") or "ok"
        if status == "error":
            any_bad = True
            table.add_row(name, "[red]error[/red]", info.get("error", ""))
        elif status == "disabled":
            table.add_row(name, "[yellow]disabled[/yellow]", "")
        else:
            detail = ", ".join(f"{k}={v}" for k, v in info.items() if k != "provider")
            table.add_row(name, "[green]ok[/green]", detail)
    console.print(table)
    if any_bad:
        raise typer.Exit(code=1)


@app.command("monitor")
def monitor(
    positions: Path = typer.Option(..., "--positions", "-p", help="Path to positions.yaml"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write markdown report to this path (default: print)"),
    archive: Optional[Path] = typer.Option(
        None, "--archive", "-a",
        help="Directory to append timestamped snapshot (md + json) for history/dashboard. Creates dir if missing.",
    ),
    force_refresh: bool = typer.Option(False, "--force-refresh"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate held positions against O'Neil's sell rules and emit a position report."""
    import json as _json
    from datetime import datetime as _dt, timezone as _tz
    _setup_logging(verbose)
    settings = _load(config)
    pos_file = PositionsFile.load(positions)
    if not pos_file.positions:
        console.print("[yellow]No positions in file — nothing to evaluate.[/yellow]")
        raise typer.Exit(code=0)

    async def _run():
        return await evaluate_positions(pos_file.positions, settings, force_refresh=force_refresh)

    evaluations, market_alerts = asyncio.run(_run())
    report = render_monitor_report(evaluations, market_alerts)

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        console.print(f"Wrote monitor report to {out}")
    elif not archive:
        console.print(report)

    if archive:
        archive.mkdir(parents=True, exist_ok=True)
        ts = _dt.now(_tz.utc).strftime("%Y-%m-%d_%H%M%S")
        (archive / f"{ts}.md").write_text(report)
        snap = snapshot_dict(evaluations, market_alerts)
        (archive / f"{ts}.json").write_text(_json.dumps(snap, indent=2, default=str))
        console.print(f"Archived snapshot to {archive}/{ts}.{{md,json}}")

    # Exit 1 if any critical alerts — useful for cron / CI integration
    has_critical = any(a.severity == "critical" for ev in evaluations for a in ev.alerts) or any(
        a.severity == "critical" for a in market_alerts
    )
    raise typer.Exit(code=1 if has_critical else 0)


@app.command("dashboard")
def dashboard(
    history: Path = typer.Option(..., "--history", "-h", help="Directory with *.json monitor snapshots"),
    out: Path = typer.Option(Path("out/monitor/dashboard.html"), "--out", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Render a self-contained HTML dashboard from archived monitor snapshots."""
    _setup_logging(verbose)
    if not history.exists():
        console.print(f"[red]History dir not found:[/red] {history}")
        raise typer.Exit(code=2)
    path = render_dashboard(history, out)
    console.print(f"[green]Dashboard written to[/green] {path}")
    console.print(f"Open in browser: file://{path.resolve()}")


@app.command("report-pdf")
def report_pdf(
    path: Optional[Path] = typer.Argument(
        None,
        help="Path to a report.md (or its run dir). Defaults to the most recent run in ./out/runs/.",
    ),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output PDF path (default: alongside the .md)"),
) -> None:
    """Render a scan report.md to PDF (Chrome / Chromium / Brave / Edge required).

    Convenience: pass either the report.md directly, a run directory, or
    nothing — in which case the most recent run under `out/runs/` is used.
    """
    from canslim.pdf import render_pdf as _render

    md_path: Optional[Path] = None
    if path is None:
        runs_dir = Path("out/runs")
        if not runs_dir.exists():
            console.print(f"[red]No runs dir found:[/red] {runs_dir}")
            raise typer.Exit(code=2)
        candidates = sorted(
            (p for p in runs_dir.iterdir() if (p / "report.md").exists()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            console.print(f"[red]No report.md found under[/red] {runs_dir}")
            raise typer.Exit(code=2)
        md_path = candidates[0] / "report.md"
        console.print(f"[dim]Using most recent run: {md_path.parent.name}[/dim]")
    elif path.is_dir():
        md_path = path / "report.md"
        if not md_path.exists():
            console.print(f"[red]No report.md in[/red] {path}")
            raise typer.Exit(code=2)
    else:
        md_path = path

    pdf_path = _render(md_path, out)
    if pdf_path is None:
        console.print(
            "[yellow]PDF generation skipped or failed (see warning above). "
            "HTML intermediate is still in place; install Chrome / Chromium to enable PDF.[/yellow]"
        )
        raise typer.Exit(code=1)
    console.print(f"[green]PDF written:[/green] {pdf_path}")


_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _index_fields_from_meta(meta: Optional[dict]) -> dict:
    """Pure helper: derive the index-row fields from a parsed meta.json dict.

    Returns a dict with keys ``as_of``, ``universe``, ``matches``, ``scanned``.
    Each value is the recovered value, or ``"—"`` when the meta is absent or the
    field is missing/null. Filesystem-free so it can be unit-tested directly.

    ``as_of`` is the data/market date (``market_regime.as_of``), which is
    deliberately distinct from the run-id timestamp — never substitute one for
    the other here.
    """
    dash = "—"
    if not meta:
        return {"as_of": dash, "universe": dash, "matches": dash, "scanned": dash}

    def _val(key: str):
        v = meta.get(key)
        return dash if v is None else v

    return {
        "as_of": _val("as_of"),
        "universe": _val("universe"),
        "matches": _val("matches"),
        "scanned": _val("scanned"),
    }


def _meta_from_manifest(manifest: dict, run_id: str) -> dict:
    """Build the committed meta.json payload from a run's manifest dict."""
    regime = manifest.get("market_regime") or {}
    return {
        "as_of": regime.get("as_of"),
        "universe": manifest.get("universe_name"),
        "matches": manifest.get("matches"),
        "scanned": manifest.get("scanned"),
        "run_id": run_id,
    }


def _meta_from_html(html: str, run_id: str) -> dict:
    """Recover a meta.json payload from a committed run's index.html (backfill).

    ``as_of`` = the MAX ``YYYY-MM-DD`` date string embedded anywhere in the file
    (charts end on the as-of bar — verified to match the scan/commit date).
    ``universe`` / ``scanned`` / ``matches`` are read from the header stat block
    when cheaply parseable; any field that can't be recovered is left ``None``.
    """
    dates = _DATE_RE.findall(html)
    as_of = max(dates) if dates else None

    uni_m = re.search(r"universe:\s*([\w-]+)", html)
    scanned_m = re.search(r'scanned\s*<span class="v">\s*([\d,]+)', html)
    matches_m = re.search(r'full matches\s*<span class="v">\s*([\d,]+)', html)

    def _to_int(m):
        if not m:
            return None
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            return None

    return {
        "as_of": as_of,
        "universe": uni_m.group(1) if uni_m else None,
        "matches": _to_int(matches_m),
        "scanned": _to_int(scanned_m),
        "run_id": run_id,
    }


# --- summary.json backfill from committed HTML (one-time bootstrap) --------
#
# Going forward summary.json is generated from the in-memory scan results
# (canslim.report.build_run_summary). The ONLY sanctioned HTML parse is this
# bootstrap, which recovers the named-bucket tickers for the ~20 runs that were
# archived before summary.json existed. It keys off the stable report markup:
# each candidate renders as a <details class="candidate" data-ticker=...> inside
# its <section class="bucket bucket-<name>">, with score/gates/AD/dist in the
# <summary>. _candidate.html / report.html are the source of truth for these.

_BUCKET_SECTION_RE = re.compile(
    r'<section class="bucket bucket-(matches|buyable|watchlist|basing)"[^>]*>(.*?)</section>',
    re.DOTALL,
)
_CANDIDATE_RE = re.compile(
    r'<details class="candidate"[^>]*data-ticker="([^"]+)"[^>]*>\s*<summary>(.*?)</summary>',
    re.DOTALL,
)
_SUM_SCORE_RE = re.compile(r'<span class="score mono">\s*([0-9.]+)\s*</span>')
_SUM_GATES_RE = re.compile(r'<span class="gates">\s*([^<]*?)\s*</span>')
_SUM_AD_RE = re.compile(r'AD:\s*([A-E])')
# Entry-plan meta carries "dist +1.2% from pivot $34.50" when a pattern exists.
_SUM_DIST_PIVOT_RE = re.compile(
    r'dist\s*([+\-]?[0-9.]+)%\s*from pivot\s*\$([0-9.,]+)'
)
# Sticky-header market regime badge: <span class="regime-badge regime-uptrend">UPTREND</span>
_SUM_REGIME_RE = re.compile(
    r'<span class="regime-badge[^"]*"[^>]*>\s*([A-Z]+)\s*</span>'
)

# section name in the HTML -> canonical bucket name used everywhere else
_HTML_BUCKET_MAP = {
    "matches": "full_match",
    "buyable": "buyable",
    "watchlist": "watchlist",
    "basing": "basing",
}


def _summary_entry_from_summary_html(summary_html: str, ticker: str, bucket: str,
                                     as_of: Optional[str]) -> dict:
    """Parse one candidate <summary> block into a summary.json entry dict."""
    score_m = _SUM_SCORE_RE.search(summary_html)
    gates_m = _SUM_GATES_RE.search(summary_html)
    ad_m = _SUM_AD_RE.search(summary_html)
    dp_m = _SUM_DIST_PIVOT_RE.search(summary_html)

    pivot = None
    dist = None
    if dp_m:
        try:
            dist = round(float(dp_m.group(1)) / 100.0, 6)
        except ValueError:
            dist = None
        try:
            pivot = round(float(dp_m.group(2).replace(",", "")), 4)
        except ValueError:
            pivot = None
    return {
        "ticker": ticker,
        "bucket": bucket,
        "score": float(score_m.group(1)) if score_m else None,
        "gates": gates_m.group(1).strip() if gates_m else "",
        "ad": ad_m.group(1) if ad_m else None,
        "pivot": pivot,
        "dist": dist,
        "market_cap": None,  # not reliably recoverable from the summary text
        "as_of": as_of,
    }


def _summary_from_html(html: str, run_id: str, as_of: Optional[str],
                       universe: Optional[str]) -> dict:
    """Recover a summary.json payload from a committed run's index.html.

    Bootstrap-only (see module note above). Walks each named bucket section and
    its candidate <details> blocks, in render order, so the recovered bucket
    assignment matches exactly what the page shows. market_cap is left null
    (the rendered summary text isn't a reliable numeric source for it).
    """
    tickers: list[dict] = []
    for sec_m in _BUCKET_SECTION_RE.finditer(html):
        html_bucket = sec_m.group(1)
        bucket = _HTML_BUCKET_MAP[html_bucket]
        body = sec_m.group(2)
        for cand_m in _CANDIDATE_RE.finditer(body):
            ticker = cand_m.group(1)
            summary_html = cand_m.group(2)
            tickers.append(
                _summary_entry_from_summary_html(summary_html, ticker, bucket, as_of)
            )
    regime_m = _SUM_REGIME_RE.search(html)
    return {
        "run_id": run_id,
        "as_of": as_of,
        "universe": universe,
        "regime": regime_m.group(1) if regime_m else None,
        "generated": None,  # backfilled from HTML, not a live scan
        "tickers": tickers,
    }


def _run_as_of(run_dir: Path) -> Optional[str]:
    """Read a committed run's data date (``as_of``) from its meta.json.

    Returns the ``as_of`` string, or ``None`` when the run has no meta.json or
    the file is unreadable / missing the field. ``None`` means "as_of unknown"
    — callers MUST treat such a run as non-prunable (never delete a dir whose
    data date we can't establish).
    """
    meta_p = run_dir / "meta.json"
    if not meta_p.exists():
        return None
    try:
        meta = json.loads(meta_p.read_text())
    except Exception:
        return None
    as_of = meta.get("as_of")
    return as_of if isinstance(as_of, str) and as_of else None


def _plan_superseded_runs(archive_dir: Path) -> dict[str, list[Path]]:
    """Group archived run dirs by data date and pick which older dupes to prune.

    "One report per date" is keyed on the DATA DATE (``as_of`` from meta.json),
    NOT the run-id timestamp. For every ``as_of`` shared by more than one run,
    the run with the LATEST run-id (lexicographic max of the dir name, which is
    a sortable ``YYYY-MM-DD_HHMMSS`` stamp) is kept; the rest are returned for
    deletion.

    Runs whose ``as_of`` can't be determined (no/unreadable meta.json) are NEVER
    grouped or pruned — each is left strictly alone. Returns a mapping
    ``{as_of: [run_dirs_to_delete]}`` containing only dates that actually had a
    prunable dupe (kept run excluded). Pure planning — performs no deletion.
    """
    from collections import defaultdict

    by_as_of: dict[str, list[Path]] = defaultdict(list)
    for p in archive_dir.iterdir():
        if not p.is_dir():
            continue
        as_of = _run_as_of(p)
        if as_of is None:
            continue  # unknown data date -> untouchable
        by_as_of[as_of].append(p)

    plan: dict[str, list[Path]] = {}
    for as_of, runs in by_as_of.items():
        if len(runs) < 2:
            continue
        keep = max(runs, key=lambda p: p.name)
        plan[as_of] = sorted((p for p in runs if p != keep), key=lambda p: p.name)
    return plan


def _prune_superseded_runs(archive_dir: Path) -> dict[str, list[str]]:
    """Delete older same-``as_of`` run dirs, keeping only the latest per date.

    Wraps :func:`_plan_superseded_runs` and removes the planned directories. Only
    runs that share a data date with a newer run are deleted — runs of distinct
    dates, and runs whose ``as_of`` is unknown, are never touched. Returns
    ``{as_of: [deleted_run_ids]}`` for logging.
    """
    import shutil

    plan = _plan_superseded_runs(archive_dir)
    deleted: dict[str, list[str]] = {}
    for as_of, dirs in plan.items():
        for d in dirs:
            shutil.rmtree(d)
        deleted[as_of] = [d.name for d in dirs]
    return deleted


@app.command("publish")
def publish(
    run: Optional[Path] = typer.Argument(
        None,
        help="Run directory or report.md to publish. Defaults to most recent run in ./out/runs/.",
    ),
    docs_dir: Path = typer.Option(Path("docs"), "--docs", help="Output docs directory (default: ./docs)"),
    out_dir: Path = typer.Option(Path("out"), "--out", help="Scan output dir to source from"),
    allow_degraded: bool = typer.Option(
        False, "--allow-degraded",
        help="Publish even if the source scan was flagged with data-quality warnings",
    ),
    backfill: bool = typer.Option(
        False, "--backfill",
        help="Write missing docs/runs/<id>/meta.json for already-archived runs "
             "by parsing their committed index.html, then regenerate the landing page. "
             "Does not publish a new run.",
    ),
) -> None:
    """Publish a scan report to ./docs/runs/<run-id>/ for GitHub Pages.

    Each published run is archived under `docs/runs/<run-id>/`, and the top-level
    `docs/index.html` is regenerated as a landing page listing all archived runs
    with metadata (data date, universe, # matches). Push the docs/ commit to your
    repo to update the live site.

    Run with no argument to publish the most recent scan.
    """
    import shutil

    archive_dir = docs_dir / "runs"

    # --backfill: write missing meta.json for already-archived runs by parsing
    # their committed index.html, then regenerate the landing page. No new run.
    if backfill:
        if not archive_dir.exists():
            console.print(f"[red]No archived runs dir found:[/red] {archive_dir}")
            raise typer.Exit(code=2)
        archived_runs = sorted(
            [p for p in archive_dir.iterdir() if (p / "index.html").exists()],
            key=lambda p: p.name,
            reverse=True,
        )
        written = skipped = 0
        summary_written = summary_skipped = 0
        for p in archived_runs:
            html = (p / "index.html").read_text()
            meta_p = p / "meta.json"
            if meta_p.exists():
                skipped += 1
                try:
                    meta = json.loads(meta_p.read_text())
                except Exception:
                    meta = _meta_from_html(html, p.name)
            else:
                meta = _meta_from_html(html, p.name)
                meta_p.write_text(json.dumps(meta, indent=2))
                written += 1
                console.print(f"[green]Backfilled meta:[/green] {meta_p} (as_of={meta.get('as_of')})")

            # Bootstrap summary.json from the committed HTML (one-time). Going
            # forward it's emitted from in-memory results by write_run.
            summary_p = p / "summary.json"
            if summary_p.exists():
                summary_skipped += 1
            else:
                summary = _summary_from_html(
                    html, p.name, meta.get("as_of"), meta.get("universe")
                )
                summary_p.write_text(json.dumps(summary, indent=2))
                summary_written += 1
                console.print(
                    f"[green]Backfilled summary:[/green] {summary_p} "
                    f"({len(summary['tickers'])} named-bucket ticker(s))"
                )
        console.print(
            f"[green]Backfill complete:[/green] meta {written} written / {skipped} present, "
            f"summary {summary_written} written / {summary_skipped} present "
            f"({len(archived_runs)} archived run(s))"
        )
        # Now that every backfilled run has a meta.json (its as_of), collapse any
        # same-data-date dupes down to the newest run-id per date.
        pruned = _prune_superseded_runs(archive_dir)
        for as_of, run_ids in pruned.items():
            console.print(
                f"[yellow]Pruned {len(run_ids)} superseded run(s) for {as_of}:[/yellow] "
                + ", ".join(run_ids)
            )
        _regenerate_landing_page(docs_dir, archive_dir)
        return

    runs_dir = out_dir / "runs"
    if run is None:
        if not runs_dir.exists():
            console.print(f"[red]No runs dir found:[/red] {runs_dir}")
            raise typer.Exit(code=2)
        candidates = sorted(
            (p for p in runs_dir.iterdir() if (p / "index.html").exists()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            console.print(f"[red]No runs with index.html found under[/red] {runs_dir}")
            raise typer.Exit(code=2)
        run = candidates[0]
        console.print(f"[dim]Using most recent run: {run.name}[/dim]")
    elif run.is_file():
        run = run.parent

    src_html = run / "index.html"
    if not src_html.exists():
        console.print(f"[red]No index.html in[/red] {run} — run a scan first or pass an explicit path")
        raise typer.Exit(code=2)

    # Refuse to publish degraded scans unless explicitly allowed.
    manifest_p = run / "run_manifest.json"
    manifest_data: Optional[dict] = None
    if manifest_p.exists():
        try:
            manifest_data = json.loads(manifest_p.read_text())
            m = manifest_data
            warnings = m.get("_data_quality_warnings") or []
            if warnings and not allow_degraded:
                console.print(f"[red]Refusing to publish {run.name} — data quality is degraded:[/red]")
                for w in warnings:
                    console.print(f"  • {w}")
                console.print("\n[dim]Re-run the scan with --force-refresh, or use[/dim]")
                console.print(f"[dim]  canslim publish {run.name} --allow-degraded[/dim]")
                console.print("[dim]if you want to publish anyway.[/dim]")
                raise typer.Exit(code=2)
        except FileNotFoundError:
            pass

    # Archive the run under docs/runs/<run-id>/
    run_id = run.name
    dest = docs_dir / "runs" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_html, dest / "index.html")
    console.print(f"[green]Archived:[/green] {dest / 'index.html'}")

    # Carry the structured per-run summary.json alongside the HTML. write_run
    # emits it into the source run dir (out/runs/<id>/); the dashboard + ticker
    # history search read it from docs/runs/<id>/ — clean JSON, NO HTML scraping.
    src_summary = run / "summary.json"
    if src_summary.exists():
        shutil.copy2(src_summary, dest / "summary.json")
        console.print(f"[green]Archived:[/green] {dest / 'summary.json'}")
    else:
        console.print(
            f"[yellow]No summary.json in {run.name} — dashboard/search will lack this run "
            f"until it is backfilled or re-scanned.[/yellow]"
        )

    # Write a COMMITTED meta.json next to the archived HTML so the index can read
    # the true data date (and headline counts) without the gitignored out/ dir.
    # Source from the run's manifest if present; fall back to parsing the HTML.
    if manifest_data is not None:
        meta = _meta_from_manifest(manifest_data, run_id)
    else:
        meta = _meta_from_html(src_html.read_text(), run_id)
    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    console.print(f"[green]Wrote meta:[/green] {dest / 'meta.json'} (as_of={meta.get('as_of')})")

    # One report per DATA DATE: prune any OLDER run dirs that share this run's
    # as_of (keep only the newest run-id per data date). A daily cron plus ad-hoc
    # manual dispatches must not repile multiple reports for the same trading
    # day. Only same-as_of older runs are removed; other dates are never touched,
    # and a run whose as_of can't be determined is left alone.
    pruned = _prune_superseded_runs(archive_dir)
    for as_of, run_ids in pruned.items():
        console.print(
            f"[yellow]Pruned {len(run_ids)} superseded run(s) for {as_of}:[/yellow] "
            + ", ".join(run_ids)
        )

    n = _regenerate_landing_page(docs_dir, archive_dir)
    console.print(f"[green]Landing page:[/green] {docs_dir / 'index.html'} ({n} run(s) listed)")
    console.print()
    console.print("[dim]To publish to your live site:[/dim]")
    console.print(f"[dim]  git add {docs_dir}/ && git commit -m 'publish {run_id}' && git push[/dim]")


def _regenerate_landing_page(docs_dir: Path, archive_dir: Path) -> int:
    """Rebuild docs/index.html from committed docs/runs/<id>/meta.json files.

    Reads each archived run's committed meta.json (never the gitignored out/
    manifest), derives index-row fields via the pure helper, and writes the
    landing page + .nojekyll. The ticker-history search panel's data
    (history index + company names) is inlined into the page so it works on
    plain static hosting with no fetch / backend. Returns the number of runs
    listed.
    """
    archived = sorted(
        [p for p in archive_dir.iterdir() if (p / "index.html").exists()],
        key=lambda p: p.name,
        reverse=True,
    )
    rows: list[str] = []
    for p in archived:
        meta_p = p / "meta.json"
        meta: Optional[dict] = None
        if meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text())
            except Exception:
                meta = None
        f = _index_fields_from_meta(meta)
        rows.append(
            f'<tr>'
            f'<td><a href="runs/{p.name}/">{f["as_of"]}</a></td>'
            f'<td class="mono">{p.name}</td>'
            f'<td>{f["universe"]}</td>'
            f'<td class="num">{f["matches"]}</td>'
            f'<td class="num">{f["scanned"]}</td>'
            f'</tr>'
        )

    # Inline the search panel's data: the ticker-history inverted index (built
    # fresh from the committed summary.json files) plus the company-name cache.
    from canslim.diffboard.build_history import build_history

    try:
        history = build_history(docs_dir)
    except Exception:
        history = {"generated": "", "dates": [], "tickers": {}}
    companies: dict = {}
    companies_file = Path(__file__).resolve().parent / "diffboard" / "companies.json"
    if companies_file.exists():
        try:
            companies = json.loads(companies_file.read_text())
        except Exception:
            companies = {}

    landing_html = _build_landing_page(rows, history, companies)
    (docs_dir / "index.html").write_text(landing_html)
    (docs_dir / ".nojekyll").touch()
    return len(archived)


def _build_landing_page(
    rows: list[str],
    history: Optional[dict] = None,
    companies: Optional[dict] = None,
) -> str:
    """Static landing page: ticker-history search panel + the archived-runs table.

    The search panel (autocomplete, range filter, per-ticker timeline) is the
    ticker-history feature that used to live on a dedicated search.html page;
    it now lives here on Reports. ``history`` (the inverted index) and
    ``companies`` (name cache) are inlined as <script type="application/json">
    blocks so the page is fully client-side on plain static hosting.
    """
    body = "\n".join(rows) if rows else '<tr><td colspan="5">No runs archived yet.</td></tr>'
    history_json = json.dumps(history or {"dates": [], "tickers": {}})
    companies_json = json.dumps(companies or {})
    return (
        _LANDING_TEMPLATE
        .replace("__RUNS_TBODY__", body)
        .replace("__HISTORY_JSON__", history_json)
        .replace("__COMPANIES_JSON__", companies_json)
    )


# Static landing-page template. Search panel (rehomed from the former
# search.html) sits above the archived-runs table; unified top-right nav
# (Reports active · Dashboard) matches the report + dashboard pages. Timeline
# rows link to runs/<id>/#c-TICKER at LANDING depth (no ../) in the SAME tab.
_LANDING_TEMPLATE = r"""<!DOCTYPE html><html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CANSLIM Scan Reports</title>
<style>
  :root { --text:#1a1f2b; --muted:#5b6473; --border:#d8dde3; --accent:#1a4480; --bg:#fff; --bg-alt:#f7f8f9;
          --pass:#2e7d32; --info:#0277bd; --warn:#e9740b;
          --mono: ui-monospace,"SF Mono",Menlo,monospace; }
  * { box-sizing: border-box; }
  body { font: 14px/1.5 -apple-system, "Inter", system-ui, sans-serif;
          color: var(--text); margin: 0 auto; padding: 24px; max-width: 900px; }
  .page-head { display: flex; align-items: baseline; justify-content: space-between;
               gap: 12px; flex-wrap: wrap; }
  h1 { font-size: 20px; margin: 0 0 4px 0; }
  .nav-links { margin-left: auto; display: flex; gap: 14px; font-size: 13px; }
  .nav-links a { color: var(--accent); text-decoration: none; }
  .nav-links a:hover { text-decoration: underline; }
  .nav-links a.active { color: var(--text); font-weight: 600; text-decoration: none; cursor: default; }
  .lede { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
  .lede a { color: var(--accent); text-decoration: none; }
  .lede a:hover { text-decoration: underline; }

  .search { margin: 4px 0 24px; padding: 16px; border: 1px solid var(--border);
            border-radius: 8px; background: var(--bg-alt); }
  .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .search-wrap { position: relative; flex: 1; min-width: 200px; }
  #q { width: 100%; font: 15px var(--mono); padding: 9px 12px; border: 1px solid var(--border);
       border-radius: 6px; text-transform: uppercase; background: var(--bg); }
  #q:focus { outline: none; border-color: var(--accent); }
  .ac { position: absolute; left: 0; right: 0; top: calc(100% + 2px); background: var(--bg);
        border: 1px solid var(--border); border-radius: 6px; box-shadow: 0 4px 14px rgba(0,0,0,.08);
        max-height: 260px; overflow-y: auto; z-index: 20; display: none; }
  .ac.open { display: block; }
  .ac-item { padding: 7px 12px; font-family: var(--mono); cursor: pointer; display: flex;
             justify-content: space-between; gap: 10px; }
  .ac-item .co { font-family: -apple-system, "Inter", system-ui, sans-serif; color: var(--muted);
                 font-size: 12px; font-weight: 400; }
  .ac-item.active, .ac-item:hover { background: var(--bg-alt); }
  .ranges { display: flex; gap: 4px; }
  .ranges button { font: inherit; font-size: 12px; padding: 7px 11px; border: 1px solid var(--border);
                   background: var(--bg); border-radius: 6px; cursor: pointer; color: var(--muted); }
  .ranges button.active { background: var(--accent); color: #fff; border-color: var(--accent); }

  .pager { display: flex; align-items: center; gap: 10px; margin-top: 12px; }
  .pager button { font: inherit; font-size: 12px; padding: 7px 11px; border: 1px solid var(--border);
                  background: var(--bg); border-radius: 6px; cursor: pointer; color: var(--muted); }
  .pager button:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
  .pager button:disabled { opacity: 0.4; cursor: default; }
  .pager .page-label { font-size: 12px; color: var(--muted); }

  #result { margin-top: 16px; }
  .result-head .sym { font: 700 22px var(--mono); }
  .result-head .co { color: var(--muted); margin-left: 10px; font-size: 14px; }
  .summary-line { color: var(--muted); font-size: 13px; margin: 6px 0 12px; }
  .summary-line .stat { color: var(--text); font-weight: 600; }
  #result table { margin-top: 0; }
  td.date a { font-family: var(--mono); }
  td.tl-num { font-family: var(--mono); text-align: right; }
  td.gates { font-family: var(--mono); }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px;
         vertical-align: middle; }
  .b-full_match .dot { background: var(--pass); }
  .b-buyable   .dot { background: var(--info); }
  .b-watchlist .dot { background: var(--warn); }
  .b-basing    .dot { background: var(--muted); }
  .bucket-label { text-transform: capitalize; }
  .empty { color: var(--muted); padding: 24px 0; text-align: center; font-style: italic; }
  .hint { color: var(--muted); font-size: 13px; padding: 18px 0; text-align: center; }

  table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 13px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
  th { background: var(--bg-alt); font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.05em; color: var(--muted); }
  td.num { text-align: right; font-family: var(--mono); }
  td.mono { font-family: var(--mono); font-size: 11px; color: var(--muted); }
  td a { color: var(--accent); text-decoration: none; font-weight: 600; }
  td a:hover { text-decoration: underline; }
  #result td a { font-weight: 400; }
  tbody tr:hover { background: var(--bg-alt); }
  .footer { color: var(--muted); font-size: 11px; margin-top: 24px; padding-top: 12px;
             border-top: 1px solid var(--border); }
  .footer a { color: var(--accent); }
  @media (max-width: 600px) {
    body { padding: 14px; }
    table { font-size: 12px; }
    th, td { padding: 6px 6px; }
    td.mono { font-size: 10px; }
    .nav-links { gap: 10px; }
    .search { padding: 12px; }
  }
</style>
</head>
<body>
<div class="page-head">
  <h1>CANSLIM Scan Reports</h1>
  <span class="nav-links">
    <a href="dashboard/">Daily Ranking Change</a>
  </span>
</div>
<p class="lede">
  Daily scans of the US equity universe against William O'Neil's CANSLIM framework
  (with leadership-override paths for turnaround setups). Click any data date below
  to view the full report — full matches, near-misses, override watchlist, market
  context (VIX/breadth/sectors), and per-ticker entry plans.
  Source: <a href="https://github.com/shuaitang5/canslim-scanner">github.com/shuaitang5/canslim-scanner</a>
</p>

<section class="search">
  <div class="controls">
    <div class="search-wrap">
      <input id="q" type="text" placeholder="Search ticker history (e.g. ATI)" autocomplete="off" spellcheck="false">
      <div id="ac" class="ac"></div>
    </div>
    <div class="ranges" id="ranges">
      <button data-range="3">3mo</button>
      <button data-range="6">6mo</button>
      <button data-range="12">1yr</button>
      <button data-range="all" class="active">all</button>
    </div>
  </div>
  <div id="result"></div>
</section>

<table>
  <thead>
    <tr>
      <th>Data date</th>
      <th>Run ID</th>
      <th>Universe</th>
      <th class="num">Matches</th>
      <th class="num">Scanned</th>
    </tr>
  </thead>
  <tbody id="reports-tbody">
__RUNS_TBODY__
  </tbody>
</table>
<div id="reports-pager" class="pager" hidden></div>
<p class="footer">
  Most recent at top. Each report is self-contained HTML with inline SVG charts.
  See <a href="https://github.com/shuaitang5/canslim-scanner">README</a> for methodology.
</p>

<script id="data" type="application/json">__HISTORY_JSON__</script>
<script id="companies" type="application/json">__COMPANIES_JSON__</script>
<script>
(function () {
  const HISTORY = JSON.parse(document.getElementById("data").textContent);
  const COMPANIES = JSON.parse(document.getElementById("companies").textContent);
  const TICKERS = Object.keys(HISTORY.tickers || {}).sort();
  const ALL_DATES = HISTORY.dates || [];
  const NUM_REPORTS = ALL_DATES.length;
  const PAGE_SIZE = 20;

  // Render a prev/next + "page N of M" pager into `el` for `totalItems` rows.
  // `onGo(page)` is called (1-based) when the user clicks prev/next. The pager
  // hides itself when there's a single page. Accessible: real <button>s with
  // aria-labels, disabled at the ends, plus an aria-live page label.
  function renderPager(el, totalItems, page, onGo) {
    const pages = Math.max(1, Math.ceil(totalItems / PAGE_SIZE));
    if (pages <= 1) { el.hidden = true; el.innerHTML = ""; return; }
    el.hidden = false;
    el.innerHTML =
      '<button type="button" class="pg-prev" aria-label="Previous page"' +
        (page <= 1 ? ' disabled' : '') + '>&larr; Prev</button>' +
      '<span class="page-label" aria-live="polite">Page ' + page + ' of ' + pages + '</span>' +
      '<button type="button" class="pg-next" aria-label="Next page"' +
        (page >= pages ? ' disabled' : '') + '>Next &rarr;</button>';
    const prev = el.querySelector(".pg-prev");
    const next = el.querySelector(".pg-next");
    if (prev) prev.addEventListener("click", () => { if (page > 1) onGo(page - 1); });
    if (next) next.addEventListener("click", () => { if (page < pages) onGo(page + 1); });
  }

  const BUCKET_LABEL = {
    full_match: "Full Match", buyable: "Buyable",
    watchlist: "Watchlist", basing: "Basing",
  };

  const qEl = document.getElementById("q");
  const acEl = document.getElementById("ac");
  const resultEl = document.getElementById("result");
  const rangesEl = document.getElementById("ranges");

  let rangeMonths = "all";
  let acIndex = -1;
  let acMatches = [];

  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
      .replace(/"/g,"&quot;").replace(/'/g,"&#39;");
  }

  // Cutoff date string for the active range, relative to the newest report date
  // (not "today" — keeps the filter meaningful on a static archive).
  function cutoffDate() {
    if (rangeMonths === "all" || !ALL_DATES.length) return null;
    const newest = new Date(ALL_DATES[0] + "T00:00:00Z");
    const c = new Date(newest);
    c.setUTCMonth(c.getUTCMonth() - parseInt(rangeMonths, 10));
    return c.toISOString().slice(0, 10);
  }

  function filterByRange(rows) {
    const cut = cutoffDate();
    if (!cut) return rows;
    return rows.filter(r => r.date >= cut);
  }

  // ---- autocomplete ----
  function renderAc(matches) {
    acMatches = matches;
    acIndex = -1;
    if (!matches.length) { acEl.classList.remove("open"); acEl.innerHTML = ""; return; }
    acEl.innerHTML = matches.slice(0, 30).map((t) => {
      const co = COMPANIES[t];
      const name = co && co.name && co.name !== t ? '<span class="co">' + escapeHtml(co.name) + '</span>' : '';
      return '<div class="ac-item" data-t="' + t + '"><span>' + t + '</span>' + name + '</div>';
    }).join("");
    acEl.classList.add("open");
  }

  function updateAc() {
    const v = qEl.value.trim().toUpperCase();
    if (!v) { acEl.classList.remove("open"); return; }
    const starts = TICKERS.filter(t => t.startsWith(v));
    const contains = TICKERS.filter(t => !t.startsWith(v) && t.includes(v));
    renderAc(starts.concat(contains));
  }

  qEl.addEventListener("input", () => { updateAc(); });

  // Select-all on focus so typing a new ticker replaces the old one (no
  // backspacing). A plain click fires focus (which selects) then a mouseup
  // that collapses the selection to the caret; swallow that one mouseup so a
  // click keeps the full text selected.
  let justFocused = false;
  qEl.addEventListener("focus", () => { justFocused = true; qEl.select(); });
  qEl.addEventListener("mouseup", (e) => {
    if (justFocused) { e.preventDefault(); justFocused = false; }
  });
  qEl.addEventListener("blur", () => { justFocused = false; });

  qEl.addEventListener("keydown", (e) => {
    const items = Array.from(acEl.querySelectorAll(".ac-item"));
    if (e.key === "ArrowDown") { e.preventDefault(); acIndex = Math.min(acIndex + 1, items.length - 1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); acIndex = Math.max(acIndex - 1, 0); }
    else if (e.key === "Enter") {
      e.preventDefault();
      const pick = (acIndex >= 0 && items[acIndex]) ? items[acIndex].dataset.t
                 : (qEl.value.trim().toUpperCase() || null);
      if (pick) selectTicker(pick);
      return;
    } else if (e.key === "Escape") { acEl.classList.remove("open"); return; }
    else { return; }
    items.forEach((el, i) => el.classList.toggle("active", i === acIndex));
  });

  acEl.addEventListener("click", (e) => {
    const item = e.target.closest(".ac-item");
    if (item) selectTicker(item.dataset.t);
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".search-wrap")) acEl.classList.remove("open");
  });

  rangesEl.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    rangeMonths = btn.dataset.range;
    rangesEl.querySelectorAll("button").forEach(b => b.classList.toggle("active", b === btn));
    resultPage = 1;          // range change reshapes the result set → page 1
    if (currentTicker) render(currentTicker);
  });

  // ---- result render ----
  let currentTicker = null;
  let resultPage = 1;        // current page of the per-ticker timeline
  let resultRows = [];       // range-filtered appearances backing the timeline

  function selectTicker(t) {
    t = t.toUpperCase();
    qEl.value = t;
    acEl.classList.remove("open");
    resultPage = 1;          // reset to page 1 on every new search
    render(t);
  }

  function bucketCounts(rows) {
    const c = {};
    rows.forEach(r => { c[r.bucket] = (c[r.bucket] || 0) + 1; });
    return c;
  }

  function fmtScore(v) { return (typeof v === "number") ? v.toFixed(2) : "—"; }

  function render(ticker) {
    currentTicker = ticker;
    const co = COMPANIES[ticker];
    const allRows = (HISTORY.tickers || {})[ticker];

    if (!allRows) {
      resultEl.innerHTML =
        '<div class="result-head"><span class="sym">' + escapeHtml(ticker) + '</span></div>' +
        '<div class="empty">' + escapeHtml(ticker) +
        ' never appeared in any report (' + NUM_REPORTS + ' reports scanned).</div>';
      return;
    }

    const rows = filterByRange(allRows);
    const coName = (co && co.name && co.name !== ticker)
      ? '<span class="co">' + escapeHtml(co.name) + '</span>' : '';

    if (!rows.length) {
      resultEl.innerHTML =
        '<div class="result-head"><span class="sym">' + escapeHtml(ticker) + '</span>' + coName + '</div>' +
        '<div class="empty">No appearances in the selected date range (' +
        allRows.length + ' total over all history).</div>';
      return;
    }

    const counts = bucketCounts(rows);
    const countParts = ["full_match","buyable","watchlist","basing"]
      .filter(b => counts[b])
      .map(b => BUCKET_LABEL[b] + " " + counts[b] + "x");
    const first = rows[rows.length - 1].date;
    const last = rows[0].date;
    const summary =
      'appeared in <span class="stat">' + rows.length + '</span> of ' + NUM_REPORTS + ' reports' +
      ' &nbsp;·&nbsp; first <span class="stat">' + first + '</span>, last <span class="stat">' + last + '</span>' +
      (countParts.length ? ' &nbsp;·&nbsp; ' + countParts.join(", ") : "");

    // Paginate the timeline 20/page. Clamp the page in case the result set
    // shrank (e.g. range change); summary still reflects the full count.
    resultRows = rows;
    const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
    resultPage = Math.min(Math.max(1, resultPage), pages);
    const start = (resultPage - 1) * PAGE_SIZE;
    const pageRows = rows.slice(start, start + PAGE_SIZE);

    const trs = pageRows.map(r => {
      const label = BUCKET_LABEL[r.bucket] || r.bucket || "—";
      return '<tr>' +
        '<td class="date"><a href="runs/' + r.run_id + '/#c-' + ticker + '">' + r.date + '</a></td>' +
        '<td class="b-' + r.bucket + '"><span class="dot"></span><span class="bucket-label">' + escapeHtml(label) + '</span></td>' +
        '<td class="tl-num">' + fmtScore(r.score) + '</td>' +
        '<td class="gates">' + escapeHtml(r.gates || "") + '</td>' +
        '<td>' + (r.ad || "—") + '</td>' +
      '</tr>';
    }).join("");

    resultEl.innerHTML =
      '<div class="result-head"><span class="sym">' + escapeHtml(ticker) + '</span>' + coName + '</div>' +
      '<div class="summary-line">' + summary + '</div>' +
      '<table><thead><tr><th>Date</th><th>Bucket</th><th class="tl-num">Score</th><th>Gates</th><th>AD</th></tr></thead>' +
      '<tbody>' + trs + '</tbody></table>' +
      '<div id="result-pager" class="pager" hidden></div>';

    renderPager(document.getElementById("result-pager"), rows.length, resultPage, (p) => {
      resultPage = p;
      render(currentTicker);
    });
  }

  // ---- reports-table pagination ----
  // Rows are server-rendered into #reports-tbody; paginate by showing/hiding
  // 20-row ranges. No pager (and all rows visible) when <=20 reports.
  const reportsBody = document.getElementById("reports-tbody");
  const reportsPager = document.getElementById("reports-pager");
  const reportRows = reportsBody
    ? Array.from(reportsBody.querySelectorAll("tr")).filter(tr => !tr.querySelector("td[colspan]"))
    : [];
  let reportsPage = 1;

  function showReportsPage(page) {
    const pages = Math.max(1, Math.ceil(reportRows.length / PAGE_SIZE));
    reportsPage = Math.min(Math.max(1, page), pages);
    const start = (reportsPage - 1) * PAGE_SIZE;
    const end = start + PAGE_SIZE;
    reportRows.forEach((tr, i) => { tr.hidden = (i < start || i >= end); });
    renderPager(reportsPager, reportRows.length, reportsPage, showReportsPage);
  }
  if (reportRows.length) showReportsPage(1);

  // Deep-link support: index.html?t=ATI
  const params = new URLSearchParams(location.search);
  const initial = (params.get("t") || "").toUpperCase();
  if (initial) { selectTicker(initial); }
  else {
    resultEl.innerHTML = '<div class="hint">Type a ticker to see every report it appeared in. ' +
      TICKERS.length + ' tickers indexed across ' + NUM_REPORTS + ' reports.</div>';
  }
})();
</script>
</body>
</html>"""


@app.command("serve")
def serve(
    out_dir: Path = typer.Option(Path("out"), "--out", "-o", help="Output dir to serve"),
    port: int = typer.Option(8765, "--port", "-p", help="Port for the local HTTP server"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the index page in your browser on start"),
) -> None:
    """Serve scan reports locally over HTTP.

    Generates an index page listing all historical runs (newest first), then
    starts a Python http.server rooted at the output directory. Each run's
    index.html is browsable at http://localhost:<port>/runs/<run_id>/index.html.
    """
    import http.server
    import socketserver
    import threading
    import webbrowser

    runs_dir = out_dir / "runs"
    if not runs_dir.exists():
        console.print(f"[red]No runs dir found:[/red] {runs_dir}")
        raise typer.Exit(code=2)

    # Generate a simple index page listing all runs newest-first
    runs = sorted(
        [p for p in runs_dir.iterdir() if (p / "index.html").exists() or (p / "report.md").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    index_html = _build_runs_index(runs, out_dir)
    (out_dir / "index.html").write_text(index_html)
    console.print(f"[green]Wrote runs index:[/green] {out_dir / 'index.html'}")

    handler = http.server.SimpleHTTPRequestHandler

    class _ReuseTCP(socketserver.TCPServer):
        allow_reuse_address = True

    serve_dir = str(out_dir.resolve())

    def _serve_forever():
        import os
        os.chdir(serve_dir)
        with _ReuseTCP(("127.0.0.1", port), handler) as httpd:
            httpd.serve_forever()

    url = f"http://127.0.0.1:{port}/index.html"
    console.print(f"[green]Serving[/green] {serve_dir} at {url}")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        _serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]stopped.[/yellow]")


def _build_runs_index(runs: list[Path], out_dir: Path) -> str:
    """Build a minimal HTML index page listing all scan runs."""
    rows = []
    for run in runs:
        target = "index.html" if (run / "index.html").exists() else "report.md"
        href = f"runs/{run.name}/{target}"
        # Try to read manifest for headline numbers
        m_path = run / "run_manifest.json"
        meta = {}
        if m_path.exists():
            try:
                import json as _json
                meta = _json.loads(m_path.read_text())
            except Exception:
                pass
        matches = meta.get("matches", "?")
        scanned = meta.get("scanned", "?")
        universe = meta.get("universe_name", "?")
        rows.append(
            f'<tr><td><a href="{href}">{run.name}</a></td>'
            f'<td>{universe}</td>'
            f'<td class="num">{matches}</td>'
            f'<td class="num">{scanned}</td></tr>'
        )
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>CANSLIM Scan Runs</title>
<style>
  body {{ font: 14px/1.5 -apple-system, system-ui, sans-serif; margin: 24px; max-width: 900px; color: #1a1f2b; }}
  h1 {{ font-size: 18px; margin-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #d8dde3; }}
  th {{ background: #f7f8f9; font-size: 12px; }}
  td.num {{ text-align: right; font-family: ui-monospace, "SF Mono", Menlo, monospace; }}
  a {{ color: #1a4480; text-decoration: none; font-family: ui-monospace, "SF Mono", Menlo, monospace; }}
  a:hover {{ text-decoration: underline; }}
  .meta {{ color: #5b6473; font-size: 12px; }}
</style></head><body>
<h1>CANSLIM Scan Runs</h1>
<p class="meta">{len(runs)} runs · serving from <code>{out_dir.resolve()}</code></p>
<table>
  <thead><tr><th>Run</th><th>Universe</th><th class="num">Matches</th><th class="num">Scanned</th></tr></thead>
  <tbody>
    {''.join(rows) if rows else '<tr><td colspan="4">No runs found.</td></tr>'}
  </tbody>
</table>
</body></html>"""


@app.command("list-universe")
def list_universe(
    name: str = typer.Argument(..., help="Universe name (sp500, us_all, custom)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    limit: int = typer.Option(0, "--limit", "-n", help="Print at most N tickers (0 = all)"),
) -> None:
    """Print tickers in a universe."""
    settings = _load(config)
    tickers = load_universe(name, settings)
    if limit > 0:
        tickers = tickers[:limit]
    for t in tickers:
        typer.echo(t)
    console.print(f"[dim]{len(tickers)} tickers[/dim]")


if __name__ == "__main__":
    app()
