from __future__ import annotations

# A real browser User-Agent string. Some public data sources (notably Wikipedia)
# return 403 Forbidden to empty or library-style User-Agents requested from
# datacenter IP ranges such as the GitHub Actions runner. Sending a normal
# browser UA keeps scheduled scans working. Shared across universe loaders so
# every outbound fetch presents a consistent, non-bot UA.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
