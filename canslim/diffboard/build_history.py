"""Build docs/history.json — the ticker-history inverted index.

Reads every committed ``docs/runs/<id>/summary.json`` and emits a single
inverted index keyed by ticker:

    {
      "generated": "<iso8601>",
      "dates": ["2026-06-23", "2026-06-22", ...],   # all report dates, desc
      "tickers": {
        "ATI": [
          {"date","bucket","score","gates","ad","pivot","dist","run_id"},  # newest first
          ...
        ],
        ...
      }
    }

Only tickers that appeared in a NAMED bucket (full_match / buyable / watchlist /
basing) are indexed — NOT the full ~1500 scanned set — so the file stays small
(dozens-to-hundreds of tickers). The search page (search.html) is pure static
client-side JS that reads this file.

Usage:
    python -m canslim.diffboard.build_history             # docs/ relative
    python -m canslim.diffboard.build_history --docs docs --out docs/history.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def build_history(docs_dir: Path) -> dict:
    """Build the inverted ticker-history index from all summary.json files."""
    runs_dir = docs_dir / "runs"
    summaries: list[dict] = []
    if runs_dir.exists():
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
            summaries.append(data)

    # One report per data date: keep the latest run-id per as_of (matches the
    # scanner's one-report-per-date prune so history mirrors what is published).
    by_date: dict[str, dict] = {}
    for s in summaries:
        date = s.get("as_of")
        run_id = s.get("run_id")
        if not date or not run_id:
            continue
        existing = by_date.get(date)
        if existing and existing["run_id"] >= run_id:
            continue
        by_date[date] = s

    dates_desc = sorted(by_date.keys(), reverse=True)

    tickers: dict[str, list[dict]] = {}
    for date in dates_desc:  # iterate newest-first so each list is newest-first
        s = by_date[date]
        run_id = s["run_id"]
        for t in s.get("tickers", []):
            sym = t.get("ticker")
            if not sym:
                continue
            tickers.setdefault(sym, []).append(
                {
                    "date": date,
                    "run_id": run_id,
                    "bucket": t.get("bucket"),
                    "score": t.get("score"),
                    "gates": t.get("gates") or "",
                    "ad": t.get("ad"),
                    "pivot": t.get("pivot"),
                    "dist": t.get("dist"),
                }
            )

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "dates": dates_desc,
        "tickers": tickers,
    }


def build(docs_dir: Path, out_path: Path) -> dict:
    history = build_history(docs_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(history, indent=2))
    print(
        f"[write] {out_path} — {len(history['tickers'])} ticker(s) across "
        f"{len(history['dates'])} report date(s)",
        file=sys.stderr,
    )
    return history


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the ticker-history inverted index (history.json).")
    ap.add_argument("--docs", type=Path, default=Path("docs"),
                    help="docs/ dir holding runs/<id>/summary.json (default: docs)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: <docs>/history.json)")
    args = ap.parse_args(argv)
    out_path = args.out or (args.docs / "history.json")
    build(args.docs, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
