# Beck — adversarial review of PR #1 (feature/market-cap-1b-and-hosting)

Reviewed at HEAD 5064f31. Base main c0b8344 (unmerged). Verified independently, not from the summary.

## VERDICT: APPROVED WITH NITS — one fix recommended before merge

The $1B gate is correctly implemented for the common case, the 64-test suite passes
(67 with my added tests), the 0-matches is genuinely environmental (not an over-rejection
bug), and the publish pipeline has a real degraded-report guard that blocks a throttled
runner from publishing garbage to the public page. One design hole on a hard floor and a
couple of nits below.

---

## 1. The $1B gate — verified 3 ways

**(a) Gate code** (`scanner._evaluate_one`, scanner.py:416-430): early reject right after the
cheap info fetch, before any criteria. Rejects ONLY when `market_cap is not None and < floor`.
Config knob `prefilter_min_market_cap_usd=1e9` (config.py:88). Cap surfaced in markdown,
HTML, parquet. Correct and clean. `>= floor` passes (strict less-than). `floor=0` disables.

**(b) Suite:** `.venv/bin/pytest -m "not slow"` → **64 passed** (confirmed). With my 3 added
tests → 67 passed.

**(c) Load-bearing adversarial check — unknown-cap leak (the design hole):**
Market cap is NOT a CANSLIM gate criterion — it's a separate early reject. When yfinance
returns `None` for the cap, the name ABSTAINS: `status="scanned"`, and it proceeds through
all gates (C/A/I/L/S). If those pass, `passed=True` and it lands in the public
"## Matches (all gates passed)" section with `Mkt Cap = —`.

I wrote `tests/test_market_cap_gate_leak.py` to prove it. `test_unknown_cap_subfloor_name_leaks_into_matches`
PASSES today: a genuinely sub-$1B company whose cap field is missing becomes a full MATCH.
Control test (`BIG` $50B, identical passing data) proves the fixture really produces matches,
so this isn't a "nothing matches anyway" artifact.

**Why this matters:** the chairman's #1 hard requirement is "no stocks under $1B" on an
OUTWARD-FACING page. Missing-cap is most common precisely for thin/illiquid small caps —
exactly the names the floor exists to exclude. Abstain = fail-OPEN on a hard floor.

**My call: abstain is wrong for a HARD floor.** For a soft filter, abstain is reasonable.
For a "no exceptions" public floor, the gate should fail-CLOSED: exclude unknown-cap names
from matches, and surface them in a separate "unknown market cap — needs review" bucket so
they're not silently dropped (preserving the no-silent-drop philosophy) but never published
as matches. In practice the leak is narrow — it needs cap=None while fundamentals/inst/float
are all present — and my live sp500 run had 0 leaks (all 3 matches had caps >=$1B). But it
is structurally possible and it's the chairman's explicit must-have, so I'm flagging it as
the single most important fix.

Nit: sub-floor rejects use `status="skipped_missing_data"` — semantically odd (the cap is
KNOWN, not missing). A dedicated `rejected_market_cap` status would read better in the
data-integrity table. Cosmetic.

## 2. 0-matches — pressure-tested, it's ENVIRONMENTAL (not a code bug)

I ran a real cached `sp500` scan (no --force-refresh). Result: **matches=3, scanned=376,
abstained=244, fetch_errors=247, EXIT=2.** I hit the same yfinance `401 Invalid Crumb`
throttling Linus described — yet still got 3 valid matches (FTNT $106B, GEV $298B, LLY $980B),
all caps populated, all >=$1B, 0 leaks. The gate did NOT tank the scan; names with cached
data cleared normally. Linus's 0 was a colder cache, not over-rejection. Explanation holds.

Cache evidence: 210 info blobs, 200 with market_cap, 51 sub-$1B — matches his "51/51 rejected"
claim exactly.

## 3. Hosting / public-page safety — SAFE, guard is present

- **Degraded-report guard EXISTS and works end-to-end** (this was the brief's biggest worry):
  `canslim scan` exits code 2 on degradation and stamps `_data_quality_warnings` in the
  manifest. GH Actions runs `run:` under `bash -e`, so exit 2 fails the Scan step → job stops
  → Publish + commit steps never run. Independently, `canslim publish` REFUSES a degraded
  manifest unless `--allow-degraded` (workflow does not pass it). Verified live: my throttled
  scan exited 2 with warnings stamped. A nightly throttled run will NOT publish a 0-match page.
  Linus's claim that he didn't commit a degraded report checks out — his degraded runs are all
  flagged `warnings=True`; main + the 21 historical docs/runs are untouched.
- SEC User-Agent IS set on the runner (committed in canslim.yaml). SEC fair-use satisfied.
- No secret leakage: only default `GITHUB_TOKEN` via checkout + `permissions: contents: write`.
- Cron `0 22 * * 1-5` (after US close, weekdays) sane; `concurrency` group prevents double commits.

**Nit (privacy):** the committed SEC User-Agent exposes the chairman's real email
`tangshua@amazon.com` in a PUBLIC repo (scrapeable → spam). Suggest a role/alias address or a
non-personal contact. Not a blocker.

## 4. Merge state — clean
main == origin/main == c0b8344 (NOT merged). docs/ untouched on the branch. 21 historical
runs intact on both branches. Working tree clean (only my new untracked test).

## Nits (non-blocking)
- `_fmt_mktcap` is DUPLICATED in both report.py (line ~544) and html_report.py (line ~281).
  Consolidate to one.
- `status="skipped_missing_data"` reused for a known-too-small cap (see §1).

## Recommended fix before merge (Elon's call)
Make the floor fail-CLOSED for unknown cap: exclude unknown-cap names from the Matches bucket
(route them to a separate review list), so a sub-$1B name with a missing cap field can never
be published as a match on the public page. Small change in `_evaluate_one` (don't let
`passed=True` when `market_cap is None and min_cap > 0`) plus a report bucket. My leak test
flips to `assert not res.passed` once fixed.
