from __future__ import annotations

import asyncio
import logging
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
    console.print(
        f"[green]done[/green] — matches={manifest.matches} scanned={manifest.scanned} "
        f"pending={manifest.pending_budget} errors={manifest.errored} "
        f"fetch_errors={n_errors} skipped_missing={n_skipped_data}"
    )
    html_path = report_path.parent / "index.html"
    if html_path.exists():
        console.print(f"html:   {html_path}")
    console.print(f"report: {report_path}")
    pdf_path = report_path.with_suffix(".pdf")
    if pdf_path.exists():
        console.print(f"pdf:    {pdf_path}")


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


@app.command("publish")
def publish(
    run: Optional[Path] = typer.Argument(
        None,
        help="Run directory or report.md to publish. Defaults to most recent run in ./out/runs/.",
    ),
    docs_dir: Path = typer.Option(Path("docs"), "--docs", help="Output docs directory (default: ./docs)"),
    out_dir: Path = typer.Option(Path("out"), "--out", help="Scan output dir to source from"),
) -> None:
    """Publish a scan report to ./docs/runs/<run-id>/ for GitHub Pages.

    Each published run is archived under `docs/runs/<run-id>/`, and the top-level
    `docs/index.html` is regenerated as a landing page listing all archived runs
    with metadata (data date, universe, # matches). Push the docs/ commit to your
    repo to update the live site.

    Run with no argument to publish the most recent scan.
    """
    import json as _json
    import shutil

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

    # Archive the run under docs/runs/<run-id>/
    run_id = run.name
    dest = docs_dir / "runs" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_html, dest / "index.html")
    console.print(f"[green]Archived:[/green] {dest / 'index.html'}")

    # Regenerate docs/index.html as a landing page listing every archived run
    archive_dir = docs_dir / "runs"
    archived = sorted(
        [p for p in archive_dir.iterdir() if (p / "index.html").exists()],
        key=lambda p: p.name,
        reverse=True,
    )
    rows: list[str] = []
    for p in archived:
        manifest_p = out_dir / "runs" / p.name / "run_manifest.json"
        as_of = matches = scanned = universe = "—"
        if manifest_p.exists():
            try:
                m = _json.loads(manifest_p.read_text())
                regime = m.get("market_regime") or {}
                as_of = regime.get("as_of") or m.get("started_at", "—")[:10]
                matches = m.get("matches", "—")
                scanned = m.get("scanned", "—")
                universe = m.get("universe_name", "—")
            except Exception:
                pass
        rows.append(
            f'<tr>'
            f'<td><a href="runs/{p.name}/">{as_of}</a></td>'
            f'<td class="mono">{p.name}</td>'
            f'<td>{universe}</td>'
            f'<td class="num">{matches}</td>'
            f'<td class="num">{scanned}</td>'
            f'</tr>'
        )

    landing_html = _build_landing_page(rows)
    (docs_dir / "index.html").write_text(landing_html)
    (docs_dir / ".nojekyll").touch()
    console.print(f"[green]Landing page:[/green] {docs_dir / 'index.html'} ({len(archived)} run(s) listed)")
    console.print()
    console.print("[dim]To publish to your live site:[/dim]")
    console.print(f"[dim]  git add {docs_dir}/ && git commit -m 'publish {run_id}' && git push[/dim]")


def _build_landing_page(rows: list[str]) -> str:
    """Static landing page listing all archived scan runs."""
    body = "\n".join(rows) if rows else '<tr><td colspan="5">No runs archived yet.</td></tr>'
    return f"""<!DOCTYPE html><html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CANSLIM Scan Reports</title>
<style>
  :root {{ --text:#1a1f2b; --muted:#5b6473; --border:#d8dde3; --accent:#1a4480; --bg-alt:#f7f8f9; }}
  * {{ box-sizing: border-box; }}
  body {{ font: 14px/1.5 -apple-system, "Inter", system-ui, sans-serif;
          color: var(--text); margin: 0 auto; padding: 24px; max-width: 900px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px 0; }}
  .lede {{ color: var(--muted); font-size: 13px; margin-bottom: 20px; }}
  .lede a {{ color: var(--accent); text-decoration: none; }}
  .lede a:hover {{ text-decoration: underline; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
  th {{ background: var(--bg-alt); font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.05em; color: var(--muted); }}
  td.num {{ text-align: right; font-family: ui-monospace, "SF Mono", Menlo, monospace; }}
  td.mono {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11px; color: var(--muted); }}
  td a {{ color: var(--accent); text-decoration: none; font-weight: 600; }}
  td a:hover {{ text-decoration: underline; }}
  tbody tr:hover {{ background: var(--bg-alt); }}
  .footer {{ color: var(--muted); font-size: 11px; margin-top: 24px; padding-top: 12px;
             border-top: 1px solid var(--border); }}
  .footer a {{ color: var(--accent); }}
  @media (max-width: 600px) {{
    body {{ padding: 14px; }}
    table {{ font-size: 12px; }}
    th, td {{ padding: 6px 6px; }}
    td.mono {{ font-size: 10px; }}
  }}
</style>
</head>
<body>
<h1>CANSLIM Scan Reports</h1>
<p class="lede">
  Daily scans of the US equity universe against William O'Neil's CANSLIM framework
  (with leadership-override paths for turnaround setups). Click any data date below
  to view the full report — full matches, near-misses, override watchlist, market
  context (VIX/breadth/sectors), and per-ticker entry plans.
  Source: <a href="https://github.com/zhoutongchar/canslim-scanner">github.com/zhoutongchar/canslim-scanner</a>
</p>
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
  <tbody>
{body}
  </tbody>
</table>
<p class="footer">
  Most recent at top. Each report is self-contained HTML with inline SVG charts.
  See <a href="https://github.com/zhoutongchar/canslim-scanner">README</a> for methodology.
</p>
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
