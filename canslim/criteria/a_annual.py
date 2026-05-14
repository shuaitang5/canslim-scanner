from __future__ import annotations

from typing import Optional

from canslim.criteria.base import Criterion, CriterionContext
from canslim.models import CriterionResult

TURNAROUND = float("inf")


class AnnualEarnings(Criterion):
    """A: each of last N years EPS grew ≥ threshold AND ROE ≥ min. 3y CAGR fallback.

    Turnaround handling: if a prior year's EPS was ≤ 0 and the next year's is > 0,
    we treat that year as a pass (O'Neil's "turning from loss to profit" clause).
    CAGR is undefined when start is non-positive; in that case the turnaround flag
    on the most recent transition year is what lets the gate pass.
    """

    letter = "A"
    name = "Annual Earnings"
    is_gate = True

    def evaluate(self, ctx: CriterionContext) -> CriterionResult:
        eb = ctx.earnings
        th = ctx.thresholds
        required = th.a_required_years

        if eb is None:
            return CriterionResult(
                letter=self.letter, passed=False, is_gate=True,
                reason="no earnings data available (fundamentals fetch failed)",
                threshold=th.a_min_annual_yoy,
                data_available=False,
            )
        if len(eb.annual_eps) < required + 1:
            return CriterionResult(
                letter=self.letter, passed=False, is_gate=True,
                reason=f"need {required + 1} years of annual EPS, got {len(eb.annual_eps)}",
                threshold=th.a_min_annual_yoy,
            )

        # annual_eps[0] is most recent. Compute YoY for each of the required years.
        yoys: list[Optional[float]] = []
        for i in range(required):
            yoys.append(_yoy(eb.annual_eps[i], eb.annual_eps[i + 1]))

        # A year "passes" if it's ≥ threshold OR it's a turnaround year (loss → profit)
        def _year_ok(y: Optional[float]) -> bool:
            return y is not None and (y == TURNAROUND or y >= th.a_min_annual_yoy)

        per_year_pass = all(_year_ok(y) for y in yoys)
        turnaround_years = sum(1 for y in yoys if y == TURNAROUND)

        cagr = _cagr(eb.annual_eps[0], eb.annual_eps[required], required)
        cagr_pass = cagr is not None and cagr >= th.a_min_annual_yoy

        roe_latest = eb.annual_roe_pct[0] if eb.annual_roe_pct else None
        roe_pass = roe_latest is not None and roe_latest >= th.a_min_roe_pct

        # If the transition year is a turnaround, CAGR is undefined (start ≤ 0) but the
        # stock still qualifies. Allow the turnaround branch to pass A.
        turnaround_path = turnaround_years > 0 and eb.annual_eps[0] > 0 and roe_pass

        # Leadership-confirmed turnaround override: when a turnaround stock has top-decile
        # RS AND a high-confidence chart pattern, the market's vote can substitute for the
        # ROE proof. Latest YoY must still be a turnaround or ≥ threshold so that an old
        # turnaround with recent decline can't sneak through.
        latest_yoy_ok = _year_ok(yoys[0] if yoys else None)
        rs_ok = (ctx.rs_percentile or 0.0) >= th.a_leadership_override_min_rs
        pattern_ok = any(
            (p.confidence or 0.0) >= th.a_leadership_override_min_pattern_conf
            for p in (ctx.patterns or [])
        )
        override_path = (
            th.a_leadership_override_enabled
            and turnaround_years > 0
            and eb.annual_eps[0] > 0
            and latest_yoy_ok
            and rs_ok
            and pattern_ok
        )

        passed = (
            (per_year_pass and roe_pass)
            or (th.a_allow_cagr_fallback and cagr_pass and roe_pass)
            or turnaround_path
            or override_path
        )

        score = 0.0
        if cagr is not None:
            score = min(1.0, max(0.0, cagr / max(th.a_min_annual_yoy, 1e-6)))
        elif turnaround_path or override_path:
            score = 0.8  # turnaround is valid but harder to quantify; give a strong but not max score

        reasons = []
        if turnaround_years > 0:
            reasons.append(f"turnaround detected ({turnaround_years} year(s) flipped loss→profit)")
        if not per_year_pass and not turnaround_path and not override_path:
            reasons.append("not every year met threshold")
        if not roe_pass:
            roe_str = f"{roe_latest:.3f}" if roe_latest is not None else "n/a"
            reasons.append(f"ROE {roe_str} < {th.a_min_roe_pct}")
        if not per_year_pass and th.a_allow_cagr_fallback and cagr_pass:
            reasons.append("fallback via CAGR")
        if override_path and not turnaround_path:
            reasons.append(
                f"leadership override (RS {ctx.rs_percentile:.2f} ≥ {th.a_leadership_override_min_rs}, pattern conf ≥ {th.a_leadership_override_min_pattern_conf})"
            )

        return CriterionResult(
            letter=self.letter,
            passed=passed,
            is_gate=True,
            score=score,
            value=cagr,
            threshold=th.a_min_annual_yoy,
            evidence={
                "annual_eps": eb.annual_eps[: required + 1],
                "annual_periods": eb.annual_periods[: required + 1],
                "yoys": ["turnaround" if y == TURNAROUND else y for y in yoys],
                "cagr": cagr,
                "roe_latest": roe_latest,
                "turnaround_years": turnaround_years,
                "override_used": bool(override_path and not turnaround_path),
            },
            reason="; ".join(reasons),
        )


def _yoy(latest: Optional[float], prior: Optional[float]) -> Optional[float]:
    """Return YoY, TURNAROUND for loss→profit, or None when both sides are non-positive."""
    if latest is None or prior is None:
        return None
    if prior <= 0 and latest > 0:
        return TURNAROUND
    if prior <= 0:
        return None
    return (latest - prior) / prior


def _cagr(end: Optional[float], start: Optional[float], years: int) -> Optional[float]:
    if end is None or start is None or start <= 0 or end <= 0 or years <= 0:
        return None
    return (end / start) ** (1.0 / years) - 1.0
