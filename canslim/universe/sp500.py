from __future__ import annotations

import io
import logging
from typing import Optional

import httpx
import pandas as pd

from canslim.universe._http import BROWSER_UA
from canslim.universe.base import Universe

log = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# GitHub-hosted constituent CSV (datasets/s-and-p-500-companies). Served from the
# same network as the GitHub Actions runner, so it isn't subject to Wikipedia's
# datacenter-IP/User-Agent 403s. Used as a fallback when the live Wikipedia
# scrape fails so a transient block doesn't nuke the nightly run.
FALLBACK_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "main/data/constituents.csv"
)


class SP500Universe(Universe):
    name = "sp500"

    def __init__(
        self,
        url: Optional[str] = None,
        fallback_url: Optional[str] = None,
        timeout: float = 20.0,
    ) -> None:
        self.url = url or WIKI_URL
        self.fallback_url = fallback_url or FALLBACK_CSV_URL
        self.timeout = timeout

    def load(self) -> list[str]:
        try:
            return self._load_wikipedia()
        except Exception as e:
            log.warning(
                "S&P 500 Wikipedia fetch failed (%s); falling back to %s",
                e,
                self.fallback_url,
            )
            return self._load_fallback()

    def _load_wikipedia(self) -> list[str]:
        # Wikipedia 403s empty/library User-Agents from datacenter IPs (the GitHub
        # runner), so send a real browser UA string.
        with httpx.Client(
            timeout=self.timeout, follow_redirects=True, headers={"User-Agent": BROWSER_UA}
        ) as c:
            resp = c.get(self.url)
            resp.raise_for_status()
            html = resp.text
        tables = pd.read_html(io.StringIO(html))
        df = tables[0]
        return self._tickers_from_df(df)

    def _load_fallback(self) -> list[str]:
        with httpx.Client(
            timeout=self.timeout, follow_redirects=True, headers={"User-Agent": BROWSER_UA}
        ) as c:
            resp = c.get(self.fallback_url)
            resp.raise_for_status()
            csv = resp.text
        df = pd.read_csv(io.StringIO(csv))
        return self._tickers_from_df(df)

    @staticmethod
    def _tickers_from_df(df: pd.DataFrame) -> list[str]:
        sym_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = df[sym_col].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
        return sorted({t for t in tickers if t and t != "nan"})
