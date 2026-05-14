from __future__ import annotations

from canslim.criteria.base import Criterion, CriterionContext
from canslim.models import CriterionResult


class Institutional(Criterion):
    """I: institutional ownership present and at least one new position, net nondecreasing QoQ.

    13F lags by ~45 days; we surface `reported_at` in evidence so users can judge freshness.
    """

    letter = "I"
    name = "Institutional Sponsorship"
    is_gate = True

    def evaluate(self, ctx: CriterionContext) -> CriterionResult:
        th = ctx.thresholds
        snap = ctx.institutional
        if snap is None:
            return CriterionResult(
                letter=self.letter, passed=False, is_gate=True,
                reason="no institutional snapshot available (fetch failed and no cache fallback)",
                data_available=False,
            )

        has_ownership = snap.inst_own_pct > 0 or snap.new_positions > 0
        # Detect "delta data unavailable" — the data source gave us ownership but no 13F delta.
        # When that's the case, don't penalize: gate only on ownership presence.
        deltas_available = snap.new_positions > 0 or snap.closed_positions > 0 or snap.qoq_delta_pct is not None

        if deltas_available:
            new_positions_ok = snap.new_positions >= th.i_min_new_positions
            qoq_ok = True
            if th.i_require_qoq_nondecrease and snap.qoq_delta_pct is not None:
                qoq_ok = snap.qoq_delta_pct >= 0
        else:
            new_positions_ok = True
            qoq_ok = True

        passed = has_ownership and new_positions_ok and qoq_ok

        score = 0.5 if has_ownership else 0.0
        if new_positions_ok:
            score += 0.3
        if qoq_ok:
            score += 0.2
        score = min(1.0, score)

        reasons = []
        if not has_ownership:
            reasons.append("no institutional ownership reported")
        if deltas_available and not new_positions_ok:
            reasons.append(f"new positions {snap.new_positions} < {th.i_min_new_positions}")
        if deltas_available and not qoq_ok:
            reasons.append(f"QoQ delta {snap.qoq_delta_pct}")
        if not deltas_available and has_ownership:
            reasons.append(f"inst. ownership {snap.inst_own_pct:.1%} (no 13F delta data)")

        return CriterionResult(
            letter=self.letter,
            passed=passed,
            is_gate=True,
            score=score,
            value=snap.inst_own_pct,
            threshold=None,
            evidence={
                "inst_own_pct": snap.inst_own_pct,
                "qoq_delta_pct": snap.qoq_delta_pct,
                "new_positions": snap.new_positions,
                "closed_positions": snap.closed_positions,
                "reported_at": snap.reported_at.isoformat(),
            },
            reason="; ".join(reasons) or "institutional sponsorship present",
        )
