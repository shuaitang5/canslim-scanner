"""Build the CANSLIM day-over-day diff dashboard.

Reads each archived run's structured ``docs/runs/<id>/summary.json`` (the clean
per-run feed written by the scanner — see ``canslim.report.build_run_summary``),
derives the day-over-day full-match diff (rank deltas vs the prior report), and
emits a self-contained static page:

  - docs/dashboard/data.json   : the parsed/merged data (one entry per date)
  - docs/dashboard/index.html  : self-contained static page (data inlined)

NO HTML scraping and NO network access — everything comes from the committed
summary.json files in the same repo. This replaces the old standalone
canslim-dashboard repo, which regex-scraped the published report HTML.

Usage:
    python -m canslim.diffboard.build_dashboard            # repo-root relative
    python -m canslim.diffboard.build_dashboard --docs docs --out docs/dashboard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPANIES_FILE = HERE / "companies.json"


def _load_summaries(runs_dir: Path) -> list[dict]:
    """Load every committed summary.json under ``runs_dir`` (docs/runs/)."""
    out: list[dict] = []
    if not runs_dir.exists():
        return out
    for run_dir in sorted(runs_dir.iterdir()):
        sp = run_dir / "summary.json"
        if not sp.exists():
            continue
        try:
            data = json.loads(sp.read_text())
        except Exception as e:  # pragma: no cover — defensive
            print(f"[warn] unreadable {sp}: {e}", file=sys.stderr)
            continue
        data.setdefault("run_id", run_dir.name)
        out.append(data)
    return out


def _full_matches(summary: dict) -> list[dict]:
    """Return full-match entries for a run, ranked by score desc -> [{rank,...}]."""
    fm = [t for t in summary.get("tickers", []) if t.get("bucket") == "full_match"]
    fm.sort(key=lambda t: -(t.get("score") or 0.0))
    items = []
    for idx, t in enumerate(fm, start=1):
        items.append(
            {
                "rank": idx,
                "ticker": t["ticker"],
                "score": t.get("score"),
                "gates": t.get("gates") or "",
                "ad": t.get("ad") or "",
            }
        )
    return items


def build_data(docs_dir: Path) -> dict:
    """Build the dashboard data model from committed summary.json files.

    One report per DATA DATE (``as_of``); if two runs share a date the
    latest-run-id wins (matches the scanner's one-report-per-date prune). Each
    report carries a relative ``source_url`` to that run's report so the
    dashboard links stay on the scanner's own GitHub Pages.
    """
    runs_dir = docs_dir / "runs"
    summaries = _load_summaries(runs_dir)
    print(f"[discover] {len(summaries)} summary.json file(s) under {runs_dir}",
          file=sys.stderr)

    by_date: dict[str, dict] = {}
    for s in summaries:
        date = s.get("as_of")
        run_id = s.get("run_id")
        if not date or not run_id:
            continue
        # later run-id wins for the same data date (lexicographic compare)
        existing = by_date.get(date)
        if existing and existing["run_id"] >= run_id:
            continue
        by_date[date] = {
            "date": date,
            "run_id": run_id,
            # Relative to docs/dashboard/index.html -> ../runs/<id>/
            "source_url": f"../runs/{run_id}/",
            "regime": s.get("regime") or "",
            "full_matches": _full_matches(s),
        }

    for date, rpt in by_date.items():
        rpt["full_matches_total"] = len(rpt["full_matches"])

    dates_desc = sorted(by_date.keys(), reverse=True)
    return {"dates": dates_desc, "reports": by_date}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CANSLIM full-match dashboard</title>
<style>
  :root {
    --bg: #ffffff; --bg-alt: #f7f8f9; --bg-dark: #eceef0;
    --border: #d8dde3; --text: #1a1f2b; --muted: #5b6473;
    --accent: #1a4480; --pass: #2e7d32; --fail: #c62828;
    --warn: #e9740b; --info: #0277bd;
    --mono: ui-monospace, "SF Mono", Menlo, monospace;
    --sans: -apple-system, "Inter", "Segoe UI", system-ui, sans-serif;
  }
  * { box-sizing: border-box; }
  body { font: 13px/1.5 var(--sans); color: var(--text); background: var(--bg); margin: 0; }
  header { padding: 14px 20px; border-bottom: 1px solid var(--border);
           display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
           position: sticky; top: 0; background: var(--bg); z-index: 10; }
  header h1 { margin: 0; font-size: 16px; font-weight: 600; }
  .regime { display: inline-block; padding: 2px 10px; border-radius: 3px;
            font-size: 11px; font-weight: 700; letter-spacing: 0.05em; }
  .regime-UPTREND { background: var(--pass); color: white; }
  .regime-CAUTION { background: var(--warn); color: white; }
  .regime-DOWN    { background: var(--fail); color: white; }
  header select { font: inherit; padding: 4px 8px; border: 1px solid var(--border);
                  border-radius: 3px; background: white; }
  header a { color: var(--accent); text-decoration: none; font-family: var(--mono); font-size: 12px; }
  header a:hover { text-decoration: underline; }
  .compare-note { color: var(--muted); font-size: 12px; }
  .nav-links { margin-left: auto; display: flex; gap: 14px; font-size: 13px; }
  .nav-links a.active { color: var(--text); font-weight: 600; text-decoration: none; cursor: default; }

  main { display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
         padding: 20px; max-width: 1700px; margin: 0 auto; }
  .badge-short { display: none; }
  .badge-long { display: inline; }
  .panel h2 { margin: 0 0 4px 0; font-size: 14px; }
  .panel .sub { color: var(--muted); font-size: 12px; margin-bottom: 10px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 6px 8px; border-bottom: 1px solid var(--border); text-align: left; font-size: 13px; }
  th { background: var(--bg-alt); font-size: 11px; text-transform: uppercase;
       letter-spacing: 0.04em; color: var(--muted); font-weight: 600; }
  td.rank { font-family: var(--mono); color: var(--muted); width: 28px; vertical-align: top; }
  td.ticker { vertical-align: top; min-width: 240px; }
  td.ticker .sym { font-family: var(--mono); font-weight: 700; }
  td.ticker .sym a { color: var(--accent); text-decoration: none; }
  td.ticker .sym a:hover { text-decoration: underline; }
  td.num a { color: var(--accent); text-decoration: none; }
  td.num a:hover { text-decoration: underline; }
  td.ticker .co { color: var(--text); margin-left: 6px; font-size: 12px; }
  td.ticker .industry { color: var(--muted); font-size: 11px; margin-left: 6px; }
  tbody tr.main-row { cursor: pointer; }
  tbody tr.blurb-row { display: none; }
  tbody tr.main-row.expanded + tr.blurb-row { display: table-row; }
  tbody tr.main-row.expanded td { background: var(--bg-alt); }
  td.blurb-cell { padding: 6px 12px 10px; background: var(--bg-alt);
                  border-bottom: 1px solid var(--border); }
  td.blurb-cell .blurb { color: var(--muted); font-size: 11.5px;
                         line-height: 1.5; max-width: none; }
  td.num { font-family: var(--mono); text-align: right; vertical-align: top; }
  td.ad { font-family: var(--mono); text-align: center; width: 34px; vertical-align: top; }
  td.delta { text-align: right; width: 110px; vertical-align: top; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
           font-size: 11px; font-weight: 700; font-family: var(--mono); }
  .b-new   { background: #e1f5fe; color: var(--info); }
  .b-up    { background: #c8e6c9; color: var(--pass); }
  .b-down  { background: #ffcdd2; color: var(--fail); }
  .b-same  { background: var(--bg-dark); color: var(--muted); }
  .b-drop  { background: #ffcdd2; color: var(--fail); text-decoration: line-through; }
  tr.dropped td { color: var(--muted); }
  .empty { color: var(--muted); padding: 20px; text-align: center; font-style: italic; }

  @media (max-width: 760px) {
    main { gap: 6px; padding: 6px; grid-template-columns: 1fr 1fr; }
    body { font-size: 11px; }
    .panel { min-width: 0; overflow: hidden; }
    table { table-layout: fixed; width: 100%; }
    th, td { padding: 3px 3px; font-size: 10.5px; white-space: nowrap;
             overflow: hidden; text-overflow: ellipsis; }
    td.ticker { white-space: normal; overflow: visible; text-overflow: clip; }
    td.ticker .industry { white-space: normal; overflow: visible; text-overflow: clip; }
    td.blurb-cell { padding: 4px 6px 8px; white-space: normal; overflow: visible;
                    text-overflow: clip; }
    td.blurb-cell .blurb { font-size: 10.5px; line-height: 1.45;
                           white-space: normal; overflow: visible; text-overflow: clip; }
    td.rank, th:first-child { width: 16px; padding-right: 0; padding-left: 3px; }
    td.ticker { min-width: 0; }
    td.ticker .co { display: none !important; }
    td.ticker .industry { display: block; margin-left: 0; margin-top: 2px;
                          font-size: 9.5px; white-space: normal; }
    td.ticker .industry .bullet { display: none; }
    th.num, td.num { width: 28px; font-size: 9.5px; padding-left: 1px; padding-right: 1px; }
    th.ad-col, td.ad { width: 22px; font-size: 9.5px; padding-left: 1px; padding-right: 1px; }
    th.delta, td.delta { width: 38px; padding-left: 1px; padding-right: 2px; font-size: 9.5px; }
    th { font-size: 9px; padding: 3px 2px; letter-spacing: 0; }
    .badge { padding: 1px 3px; font-size: 9.5px; }
    .badge-long { display: none !important; }
    .badge-short { display: inline !important; }
    .panel h2 { font-size: 11.5px; }
    .panel .sub { display: none; }
    header { padding: 8px 10px; gap: 8px; }
    header h1 { font-size: 13px; }
    .nav-links { margin-left: 0; gap: 10px; }
  }
</style>
</head>
<body>

<header>
  <h1>CANSLIM full matches</h1>
  <span id="regime" class="regime"></span>
  <label class="compare-note">Date <select id="dateSelect"></select></label>
  <span class="compare-note" id="compareNote"></span>
  <span class="nav-links">
    <a href="../">All Reports</a>
    <a class="active" href="index.html">Daily Ranking Change</a>
  </span>
</header>

<main>
  <section class="panel">
    <h2 id="leftTitle">Today</h2>
    <div class="sub" id="leftSub"></div>
    <table id="leftTable">
      <thead>
        <tr><th>#</th><th>Ticker</th><th class="num">Score</th><th class="ad-col">AD</th><th class="delta">Δ vs prior</th></tr>
      </thead>
      <tbody></tbody>
    </table>
  </section>
  <section class="panel">
    <h2 id="rightTitle">Prior day</h2>
    <div class="sub" id="rightSub"></div>
    <table id="rightTable">
      <thead>
        <tr><th>#</th><th>Ticker</th><th class="num">Score</th><th class="ad-col">AD</th><th class="delta">Status today</th></tr>
      </thead>
      <tbody></tbody>
    </table>
  </section>
</main>

<script id="data" type="application/json">__DATA_JSON__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById("data").textContent);
  const dates = DATA.dates;
  const reports = DATA.reports;
  const companies = DATA.companies || {};

  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  const sel = document.getElementById("dateSelect");
  // Cap the dropdown at the 100 most-recent dates (dates is newest-first) so a
  // large archive doesn't overload option rendering. The diff is day-over-day,
  // so older dates aren't useful to select here; the underlying reports map is
  // unaffected.
  const DROPDOWN_LIMIT = 100;
  const dropdownDates = dates.slice(0, DROPDOWN_LIMIT);
  dropdownDates.forEach((d, i) => {
    const o = document.createElement("option");
    o.value = d; o.textContent = d + (i === 0 ? "  (latest)" : "");
    sel.appendChild(o);
  });
  if (dates.length > DROPDOWN_LIMIT) {
    const note = document.createElement("option");
    note.disabled = true;
    note.textContent = "… older " + (dates.length - DROPDOWN_LIMIT) + " date(s) hidden";
    sel.appendChild(note);
  }
  sel.value = dates[0];
  sel.addEventListener("change", render);

  function rankMap(report) {
    const m = new Map();
    (report.full_matches || []).forEach(it => m.set(it.ticker, it));
    return m;
  }

  function deltaBadge(primaryItem, priorMap) {
    if (!priorMap) return { cls: "b-same", text: "—" };
    const prior = priorMap.get(primaryItem.ticker);
    if (!prior) return { cls: "b-new", text: "NEW" };
    const delta = prior.rank - primaryItem.rank;
    if (delta === 0) return { cls: "b-same", text: "=" };
    if (delta > 0) return { cls: "b-up", text: "↑" + delta };
    return { cls: "b-down", text: "↓" + Math.abs(delta) };
  }

  function priorStatus(priorItem, primaryMap) {
    if (!primaryMap) return { cls: "b-same", text: "—", short: "—" };
    const today = primaryMap.get(priorItem.ticker);
    if (!today) return { cls: "b-drop", text: "DROPPED", short: "drop" };
    const delta = priorItem.rank - today.rank;
    if (delta === 0) return { cls: "b-same", text: "held #" + today.rank, short: "h#" + today.rank };
    if (delta > 0) return { cls: "b-up", text: "↑ to #" + today.rank, short: "↑#" + today.rank };
    return { cls: "b-down", text: "↓ to #" + today.rank, short: "↓#" + today.rank };
  }

  function renderTable(tbody, items, deltaFn, options={}) {
    tbody.innerHTML = "";
    if (!items.length) {
      const tr = document.createElement("tr");
      tr.innerHTML = '<td colspan="5" class="empty">No full matches.</td>';
      tbody.appendChild(tr);
      return;
    }
    items.forEach(it => {
      const tr = document.createElement("tr");
      tr.className = "main-row";
      const d = deltaFn(it);
      if (options.droppedCheck && options.droppedCheck(it)) {
        tr.classList.add("dropped");
      }
      const co = companies[it.ticker] || {};
      const nameHtml = co.name     ? '<span class="co">' + escapeHtml(co.name) + '</span>' : '';
      const indHtml  = co.industry ? '<span class="industry"><span class="bullet">· </span>' + escapeHtml(co.industry) + '</span>' : '';
      tr.innerHTML =
        '<td class="rank">' + it.rank + '</td>' +
        '<td class="ticker">' +
          '<span class="sym"><a href="https://ticker-quickview.onrender.com/#' + it.ticker + '" target="_blank" rel="noopener">' + it.ticker + '</a></span>' +
          nameHtml + indHtml +
        '</td>' +
        '<td class="num"><a href="' + options.sourceUrl + '#c-' + it.ticker + '">' + (it.score != null ? it.score.toFixed(2) : '—') + '</a></td>' +
        '<td class="ad">' + (it.ad || '—') + '</td>' +
        '<td class="delta"><span class="badge ' + d.cls + '">' +
          '<span class="badge-long">' + d.text + '</span>' +
          '<span class="badge-short">' + (d.short || d.text) + '</span>' +
        '</span></td>';
      tbody.appendChild(tr);

      const blurbTr = document.createElement("tr");
      blurbTr.className = "blurb-row";
      const blurbHtml = co.blurb ? escapeHtml(co.blurb) : '';
      blurbTr.innerHTML =
        '<td colspan="5" class="blurb-cell"><div class="blurb">' + blurbHtml + '</div></td>';
      tbody.appendChild(blurbTr);
    });
  }

  function render() {
    const primaryDate = sel.value;
    const idx = dates.indexOf(primaryDate);
    const priorDate = dates[idx + 1] || null;
    const primary = reports[primaryDate];
    const prior = priorDate ? reports[priorDate] : null;

    const regimeEl = document.getElementById("regime");
    regimeEl.textContent = primary.regime || "UNKNOWN";
    regimeEl.className = "regime regime-" + (primary.regime || "UNKNOWN");

    document.getElementById("compareNote").textContent =
      prior ? "comparing " + primaryDate + "  vs  " + priorDate
            : "no prior report available for comparison";

    document.getElementById("leftTitle").textContent =
      primaryDate + " — " + primary.full_matches_total + " full matches";
    document.getElementById("leftSub").textContent =
      "Full CANSLIM matches ranked by composite score.";

    document.getElementById("rightTitle").textContent =
      prior ? priorDate + " — " + prior.full_matches_total + " full matches"
            : "No prior report";
    document.getElementById("rightSub").textContent =
      prior ? "Prior trading day — used as the comparison baseline." : "";

    const primaryTop = primary.full_matches || [];
    const priorTop   = prior ? (prior.full_matches || []) : [];

    const primaryMap = rankMap(primary);
    const priorMap   = prior ? rankMap(prior) : null;

    renderTable(
      document.querySelector("#leftTable tbody"),
      primaryTop,
      it => deltaBadge(it, priorMap),
      { sourceUrl: primary.source_url }
    );
    renderTable(
      document.querySelector("#rightTable tbody"),
      priorTop,
      it => priorStatus(it, primaryMap),
      {
        sourceUrl: prior ? prior.source_url : "#",
        droppedCheck: it => primaryMap && !primaryMap.has(it.ticker),
      }
    );
  }

  render();

  function onRowClick(e) {
    if (e.target.closest("a")) return;
    const tr = e.target.closest("tr.main-row");
    if (!tr || tr.querySelector(".empty")) return;
    const wasExpanded = tr.classList.contains("expanded");
    document.querySelectorAll("tbody tr.main-row.expanded").forEach(r => r.classList.remove("expanded"));
    if (!wasExpanded) tr.classList.add("expanded");
  }
  document.querySelector("#leftTable tbody").addEventListener("click", onRowClick);
  document.querySelector("#rightTable tbody").addEventListener("click", onRowClick);
})();
</script>

</body>
</html>
"""


def build(docs_dir: Path, out_dir: Path) -> dict:
    data = build_data(docs_dir)
    if COMPANIES_FILE.exists():
        data["companies"] = json.loads(COMPANIES_FILE.read_text())
        print(f"[merge] {len(data['companies'])} company entries from "
              f"{COMPANIES_FILE.name}", file=sys.stderr)
    else:
        data["companies"] = {}
        print("[warn] companies.json not found — run enrich_companies.py",
              file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(data, indent=2))
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data))
    (out_dir / "index.html").write_text(html)
    print(f"[write] {out_dir / 'data.json'}", file=sys.stderr)
    print(f"[write] {out_dir / 'index.html'}", file=sys.stderr)
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the CANSLIM diff dashboard + ticker-history index from summary.json files."
    )
    ap.add_argument("--docs", type=Path, default=Path("docs"),
                    help="docs/ dir holding runs/<id>/summary.json (default: docs)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir for the dashboard (default: <docs>/dashboard)")
    args = ap.parse_args(argv)
    out_dir = args.out or (args.docs / "dashboard")

    # 1) diff dashboard (also returns the merged data incl. companies)
    build(args.docs, out_dir)
    # 2) ticker-history inverted index at docs/history.json — the search panel
    #    on the landing page (Reports) inlines this at publish time. No more
    #    standalone search.html page.
    from canslim.diffboard.build_history import build as build_history_file
    build_history_file(args.docs, args.docs / "history.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
