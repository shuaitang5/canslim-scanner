from __future__ import annotations

from typing import Optional

from canslim.criteria.base import Criterion, CriterionContext
from canslim.models import CriterionResult

# Sentinel for "turnaround": prior-period EPS was ≤ 0 and current is > 0.
# O'Neil explicitly includes this in CANSLIM — "turning from a loss to a
# substantial profit" counts as a valid C signal, even though % growth is
# mathematically undefined (can't divide by a negative base).
TURNAROUND = float("inf")


class CurrentEarnings(Criterion):
    """C: current quarterly EPS YoY growth ≥ threshold AND accelerating vs prior quarter.

    Turnaround handling: if year-ago EPS was ≤ 0 and latest is > 0, that's a pass
    (O'Neil's "turning from loss to profit" clause). We encode this as yoy = +inf
    so it automatically clears any positive threshold and any acceleration check.
    """

    letter = "C"
    name = "Current Earnings"
    is_gate = True

    def evaluate(self, ctx: CriterionContext) -> CriterionResult:
        eb = ctx.earnings
        th = ctx.thresholds
        if eb is None:
            return CriterionResult(
                letter=self.letter, passed=False, is_gate=True,
                reason="no earnings data available (fundamentals fetch failed)",
                threshold=th.c_min_yoy,
                data_available=False,
            )
        if len(eb.quarterly_eps) < 5:
            # Distinct from "fetch failed": data was fetched but there aren't enough
            # quarters of public history (e.g., recent IPOs like CRCL, RKLB).
            # Real evaluation — just fails the eligibility check.
            return CriterionResult(
                letter=self.letter, passed=False, is_gate=True,
                reason=f"insufficient quarterly EPS history ({len(eb.quarterly_eps)} of 5 needed)",
                threshold=th.c_min_yoy,
            )
        # quarterly_eps[0] = most recent, index 4 = same quarter a year ago
        latest = eb.quarterly_eps[0]
        yago = eb.quarterly_eps[4]
        prior_latest = eb.quarterly_eps[1] if len(eb.quarterly_eps) > 1 else None
        prior_yago = eb.quarterly_eps[5] if len(eb.quarterly_eps) > 5 else None

        latest_yoy = _yoy(latest, yago)
        prior_yoy = _yoy(prior_latest, prior_yago) if prior_latest is not None and prior_yago is not None else None
        turnaround = latest_yoy == TURNAROUND

        passed_threshold = latest_yoy is not None and latest_yoy >= th.c_min_yoy
        accelerating = True
        if th.c_require_accelerating and prior_yoy is not None and latest_yoy is not None:
            accelerating = latest_yoy >= prior_yoy

        passed = passed_threshold and accelerating
        score = 0.0
        if latest_yoy is not None:
            # Cap score at 1.0; turnaround gets the max score
            score = 1.0 if turnaround else max(0.0, min(1.0, latest_yoy / max(th.c_min_yoy, 1e-6)))

        if latest_yoy is None:
            reason = "both latest and year-ago EPS non-positive — no growth signal"
        elif turnaround:
            reason = f"turnaround: ${yago:.2f} → ${latest:.2f} (loss → profit)"
        elif not passed_threshold:
            reason = f"latest YoY {latest_yoy:.1%} below threshold {th.c_min_yoy:.0%}"
        elif not accelerating:
            reason = f"not accelerating: latest {latest_yoy:.1%} < prior {prior_yoy:.1%}" if prior_yoy else "not accelerating"
        else:
            reason = ""

        return CriterionResult(
            letter=self.letter,
            passed=passed,
            is_gate=True,
            score=score,
            value=None if turnaround else latest_yoy,  # inf doesn't roundtrip cleanly through parquet
            threshold=th.c_min_yoy,
            evidence={
                "latest_period": eb.quarterly_periods[0] if eb.quarterly_periods else None,
                "latest_eps": latest,
                "year_ago_eps": yago,
                "latest_yoy": "turnaround" if turnaround else latest_yoy,
                "prior_yoy": prior_yoy if prior_yoy != TURNAROUND else "turnaround",
                "turnaround": turnaround,
            },
            reason=reason,
        )


def _yoy(latest: Optional[float], yago: Optional[float]) -> Optional[float]:
    """Return YoY growth, TURNAROUND sentinel for loss→profit, or None when both sides are non-positive."""
    if latest is None or yago is None:
        return None
    if yago <= 0 and latest > 0:
        return TURNAROUND
    if yago <= 0:  # both non-positive — no growth signal
        return None
    return (latest - yago) / yago
