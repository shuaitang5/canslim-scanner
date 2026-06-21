from __future__ import annotations

import httpx
import pytest

from canslim.universe._http import BROWSER_UA
from canslim.universe.sp500 import SP500Universe
from canslim.universe.us_all import USAllUniverse

WIKI_HTML = """
<table class="wikitable">
<tr><th>Symbol</th><th>Security</th></tr>
<tr><td>AAPL</td><td>Apple</td></tr>
<tr><td>BRK.B</td><td>Berkshire</td></tr>
<tr><td>MSFT</td><td>Microsoft</td></tr>
</table>
"""

FALLBACK_CSV = "Symbol,Security\nAAPL,Apple\nBRK.B,Berkshire\nMSFT,Microsoft\n"

NASDAQ_TXT = (
    "Symbol|Security Name|Test Issue|Financial Status|ETF\n"
    "AAPL|Apple Inc. - Common Stock|N|N|N\n"
    "File Creation Time: 0101202600:00\n"
)
OTHER_TXT = (
    "ACT Symbol|Security Name|Test Issue|ETF\n"
    "BRK.B|Berkshire Hathaway - Common Stock|N|N\n"
    "File Creation Time: 0101202600:00\n"
)


def _patch_client(monkeypatch, handler):
    """Force every httpx.Client created in the loaders to use a MockTransport."""
    orig_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)


def test_sp500_sends_browser_ua(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers.get("User-Agent")
        return httpx.Response(200, text=WIKI_HTML)

    _patch_client(monkeypatch, handler)
    tickers = SP500Universe().load()

    assert seen["ua"] == BROWSER_UA
    # "Mozilla" alone isn't enough — must not be the old library UA Wikipedia 403s
    assert "(canslim-scanner)" not in seen["ua"]
    # "." normalized to "-" for yfinance compatibility
    assert tickers == ["AAPL", "BRK-B", "MSFT"]


def test_sp500_falls_back_on_403(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "wikipedia.org" in request.url.host:
            return httpx.Response(403, text="Forbidden")
        return httpx.Response(200, text=FALLBACK_CSV)

    _patch_client(monkeypatch, handler)
    tickers = SP500Universe().load()

    # Hit Wikipedia (403) then the GitHub-hosted fallback CSV
    assert any("wikipedia.org" in u for u in calls)
    assert any("githubusercontent.com" in u for u in calls)
    assert tickers == ["AAPL", "BRK-B", "MSFT"]


def test_sp500_tries_fallback_before_giving_up(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return httpx.Response(403, text="Forbidden")

    _patch_client(monkeypatch, handler)
    # Only when BOTH the live source and the fallback are down does it raise —
    # a single-source 403 must NOT short-circuit before the fallback is tried.
    with pytest.raises(httpx.HTTPStatusError):
        SP500Universe().load()
    assert any("wikipedia.org" in h for h in calls)
    assert any("githubusercontent.com" in h for h in calls)


def test_us_all_sends_browser_ua(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("User-Agent"))
        body = NASDAQ_TXT if "nasdaqlisted" in str(request.url) else OTHER_TXT
        return httpx.Response(200, text=body)

    _patch_client(monkeypatch, handler)
    tickers = USAllUniverse().load()

    assert seen and all(ua == BROWSER_UA for ua in seen)
    assert tickers == ["AAPL", "BRK-B"]
