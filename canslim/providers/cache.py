from __future__ import annotations

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class CacheStore:
    """Parquet-backed cache partitioned by (provider, kind, ticker).

    Layout:
      cache/
        prices/yfinance/AAPL.parquet        (tabular cache + fetched_at column)
        fundamentals/fmp/AAPL.parquet
        institutional/fmp/AAPL.parquet
        meta/fmp_budget.json
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- path helpers

    def _path(self, kind: str, provider: str, ticker: str) -> Path:
        safe = ticker.replace("/", "-").upper()
        d = self.root / kind / provider
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{safe}.parquet"

    def _meta_path(self, name: str) -> Path:
        d = self.root / "meta"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{name}.json"

    # ---- DataFrame cache (prices, fundamentals)

    def read_df(self, kind: str, provider: str, ticker: str) -> Optional[pd.DataFrame]:
        p = self._path(kind, provider, ticker)
        if not p.exists():
            return None
        try:
            return pd.read_parquet(p)
        except Exception:
            return None

    def write_df(self, kind: str, provider: str, ticker: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        p = self._path(kind, provider, ticker)
        out = df.copy()
        if "fetched_at" not in out.columns:
            out["fetched_at"] = pd.Timestamp.now(tz="UTC")
        out.to_parquet(p, index=True)

    def age_hours(self, kind: str, provider: str, ticker: str) -> Optional[float]:
        p = self._path(kind, provider, ticker)
        if not p.exists():
            return None
        mtime = p.stat().st_mtime
        return (time.time() - mtime) / 3600.0

    def is_fresh(self, kind: str, provider: str, ticker: str, ttl_hours: float) -> bool:
        age = self.age_hours(kind, provider, ticker)
        return age is not None and age <= ttl_hours

    # ---- JSON blob cache (institutional snapshot, small objects)

    def read_json(self, kind: str, provider: str, ticker: str) -> Optional[dict[str, Any]]:
        safe = ticker.replace("/", "-").upper()
        d = self.root / kind / provider
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{safe}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def write_json(self, kind: str, provider: str, ticker: str, data: dict[str, Any]) -> None:
        safe = ticker.replace("/", "-").upper()
        d = self.root / kind / provider
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{safe}.json"
        payload = dict(data)
        payload.setdefault("_fetched_at", datetime.now(timezone.utc).isoformat())
        p.write_text(json.dumps(payload, default=str))

    def is_json_fresh(self, kind: str, provider: str, ticker: str, ttl_hours: float) -> bool:
        safe = ticker.replace("/", "-").upper()
        p = self.root / kind / provider / f"{safe}.json"
        if not p.exists():
            return False
        age = (time.time() - p.stat().st_mtime) / 3600.0
        return age <= ttl_hours

    def json_age_hours(self, kind: str, provider: str, ticker: str) -> Optional[float]:
        """Age of JSON cache file in hours; None if no cache exists."""
        safe = ticker.replace("/", "-").upper()
        p = self.root / kind / provider / f"{safe}.json"
        if not p.exists():
            return None
        return (time.time() - p.stat().st_mtime) / 3600.0

    # ---- negative cache (records failed fetches so re-runs back off briefly)

    def _neg_path(self, kind: str, provider: str, ticker: str) -> Path:
        safe = ticker.replace("/", "-").upper()
        d = self.root / "negative" / kind / provider
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{safe}.json"

    def record_failure(self, kind: str, provider: str, ticker: str, error: str) -> None:
        p = self._neg_path(kind, provider, ticker)
        prev = {}
        if p.exists():
            try:
                prev = json.loads(p.read_text())
            except Exception:
                prev = {}
        payload = {
            "ticker": ticker,
            "kind": kind,
            "provider": provider,
            "error": error[:500],
            "first_seen": prev.get("first_seen") or datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "retry_count": int(prev.get("retry_count", 0)) + 1,
        }
        p.write_text(json.dumps(payload, default=str, indent=2))

    def clear_failure(self, kind: str, provider: str, ticker: str) -> None:
        p = self._neg_path(kind, provider, ticker)
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    def read_failure(self, kind: str, provider: str, ticker: str) -> Optional[dict[str, Any]]:
        p = self._neg_path(kind, provider, ticker)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def is_failure_fresh(self, kind: str, provider: str, ticker: str, ttl_hours: float) -> bool:
        p = self._neg_path(kind, provider, ticker)
        if not p.exists():
            return False
        age = (time.time() - p.stat().st_mtime) / 3600.0
        return age <= ttl_hours

    def list_failures(self) -> list[dict[str, Any]]:
        """Walk the negative cache and return all recorded failures."""
        root = self.root / "negative"
        if not root.exists():
            return []
        out: list[dict[str, Any]] = []
        for f in root.rglob("*.json"):
            try:
                out.append(json.loads(f.read_text()))
            except Exception:
                continue
        return out

    # ---- budget meta

    def read_meta(self, name: str) -> dict[str, Any]:
        p = self._meta_path(name)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}

    def write_meta(self, name: str, data: dict[str, Any]) -> None:
        p = self._meta_path(name)
        p.write_text(json.dumps(data, default=str, indent=2))
