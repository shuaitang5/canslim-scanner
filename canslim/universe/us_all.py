from __future__ import annotations

import io
import logging
from typing import Optional

import httpx
import pandas as pd

from canslim.universe._http import BROWSER_UA
from canslim.universe.base import Universe

log = logging.getLogger(__name__)

# Nasdaq Trader posts daily listing files for all NASDAQ and non-NASDAQ US securities
NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


class USAllUniverse(Universe):
    """All US common stocks from NASDAQ + NYSE/AMEX listing files.

    Filters out ETFs, test issues, when-issued, preferred shares, warrants.
    """

    name = "us_all"

    def __init__(
        self,
        nasdaq_url: Optional[str] = None,
        other_url: Optional[str] = None,
        timeout: float = 20.0,
    ) -> None:
        self.nasdaq_url = nasdaq_url or NASDAQ_LISTED
        self.other_url = other_url or OTHER_LISTED
        self.timeout = timeout

    def load(self) -> list[str]:
        tickers: set[str] = set()
        # Send a real browser UA so a strict datacenter-IP filter on the runner
        # can't 403 the listing files (same precaution as the sp500 loader).
        with httpx.Client(
            timeout=self.timeout, follow_redirects=True, headers={"User-Agent": BROWSER_UA}
        ) as client:
            tickers.update(self._parse(self._fetch(client, self.nasdaq_url), source="nasdaq"))
            tickers.update(self._parse(self._fetch(client, self.other_url), source="other"))
        return sorted(tickers)

    def _fetch(self, client: httpx.Client, url: str) -> str:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text

    def _parse(self, text: str, source: str) -> list[str]:
        # Pipe-delimited; last line is a footer "File Creation Time..."
        lines = [ln for ln in text.splitlines() if ln and not ln.lower().startswith("file creation")]
        if not lines:
            return []
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")
        sym_col = "Symbol" if "Symbol" in df.columns else ("ACT Symbol" if "ACT Symbol" in df.columns else df.columns[0])
        df = df[df[sym_col].astype(str).str.match(r"^[A-Z][A-Z0-9.\-]{0,5}$", na=False)]
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"] != "Y"]
        if "ETF" in df.columns:
            df = df[df["ETF"] != "Y"]
        if "Financial Status" in df.columns:  # NASDAQ only — drop deficient/delinquent/bankrupt
            df = df[df["Financial Status"].isin(["N", "H", ""])]

        # Filter non-common-stock instruments by Security Name. NASDAQ suffixes common stock
        # with "- Common Stock"; warrants/rights/units/preferred/notes all carry distinctive
        # security-name fragments we can reject. Before this filter ~3,770 non-stocks leaked
        # through and failed in yfinance batch downloads.
        name_col = "Security Name" if "Security Name" in df.columns else None
        if name_col:
            non_common = (
                r"Warrant|Right|Units?\b|When Issued|Depositary|Depositary Shares?|"
                r"\bPreferred\b|Preferred Stock|Preference|"
                r"\bNotes?\b|\bBonds?\b|Debenture|Subordinated|"
                r"Trust Units?|Beneficial Interest|"
                r"Closed End|Closed-End|Closed End Fund|Investment Trust|"
                # ETNs (e.g. "iPath Bloomberg Commodity Index Total Return ETN") sneak past
                # the ETF column flag — Nasdaq Trader marks them ETF=N. Filter by name.
                r"\bETNs?\b|"
                # Closed-end funds nearly always have "Fund" or "Trust" in their name
                # (BlackRock/Nuveen/Virtus/Cohen & Steers/iShares). Operating growth
                # companies almost never do, so a whole-word match is safe.
                r"\bFund\b|\bTrust\b"
            )
            df = df[~df[name_col].astype(str).str.contains(non_common, case=False, regex=True, na=False)]

        syms = df[sym_col].astype(str).str.replace(".", "-", regex=False)
        # Symbol-level safety net: tickers with $ = (warrants, depository) or ending in -W/-U/-R
        syms = syms[~syms.str.contains(r"[\$=]", na=False)]
        syms = syms[~syms.str.match(r"^[A-Z]+-[WUR]$", na=False)]
        log.info("us_all: %d common stocks from %s", len(syms), source)
        return syms.tolist()
