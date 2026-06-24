"""Pull company name + short blurb for every named-bucket ticker.

Discovers tickers from the committed per-run ``docs/runs/<id>/summary.json``
files (ALL named buckets — full_match / buyable / watchlist / basing — so every
ticker the dashboard or search page can show gets a name/sector/blurb). Uses
yfinance's ``info``; results are cached to companies.json so repeat runs are
free and robust to Yahoo rate limits.

Usage:
    python -m canslim.diffboard.enrich_companies             # missing only
    python -m canslim.diffboard.enrich_companies --refresh   # re-fetch all
    python -m canslim.diffboard.enrich_companies --docs docs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_FILE = HERE / "companies.json"


def first_two_sentences(text: str) -> str:
    """Return first 1-2 sentences, capped at ~320 chars for UI tidy-ness."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = parts[0] if parts else text
    if len(parts) > 1 and len(out) < 180:
        out += " " + parts[1]
    if len(out) > 320:
        out = out[:317].rstrip() + "…"
    return out


def fetch_one(ticker: str) -> dict:
    import yfinance as yf

    info = yf.Ticker(ticker).info or {}
    name = info.get("longName") or info.get("shortName") or ticker
    industry = info.get("industry") or ""
    sector = info.get("sector") or ""
    blurb = first_two_sentences(info.get("longBusinessSummary") or "")
    return {
        "ticker": ticker,
        "name": name,
        "industry": industry,
        "sector": sector,
        "blurb": blurb,
    }


def discover_tickers(docs_dir: Path) -> list[str]:
    """All tickers that ever appeared in a NAMED bucket across summary.json files."""
    runs_dir = docs_dir / "runs"
    tickers: set[str] = set()
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            sp = run_dir / "summary.json"
            if not sp.exists():
                continue
            try:
                data = json.loads(sp.read_text())
            except Exception:
                continue
            for t in data.get("tickers", []):
                if t.get("ticker"):
                    tickers.add(t["ticker"])
    return sorted(tickers)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch every ticker, ignoring cache")
    ap.add_argument("--docs", type=Path, default=Path("docs"),
                    help="docs/ dir holding runs/<id>/summary.json (default: docs)")
    args = ap.parse_args(argv)

    tickers = discover_tickers(args.docs)
    print(f"[discover] {len(tickers)} unique named-bucket ticker(s)", file=sys.stderr)

    cache: dict[str, dict] = {}
    if CACHE_FILE.exists() and not args.refresh:
        cache = json.loads(CACHE_FILE.read_text())
        print(f"[cache] {len(cache)} cached entries", file=sys.stderr)

    todo = [t for t in tickers if t not in cache]
    print(f"[todo] fetching {len(todo)} ticker(s)", file=sys.stderr)

    for t in todo:
        try:
            cache[t] = fetch_one(t)
            print(f"  [ok] {t} — {cache[t]['name']}", file=sys.stderr)
        except Exception as e:
            print(f"  [err] {t}: {e}", file=sys.stderr)
            cache[t] = {
                "ticker": t, "name": t, "industry": "", "sector": "", "blurb": "",
            }
        time.sleep(0.2)  # be nice to Yahoo

    CACHE_FILE.write_text(json.dumps(cache, indent=2))
    print(f"[write] {CACHE_FILE}", file=sys.stderr)

    blanks = [t for t, v in cache.items() if not v.get("blurb")]
    if blanks:
        print(f"[warn] no blurb for: {blanks} (will render with name only)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
