from __future__ import annotations

from datetime import date

from canslim.config import CriteriaThresholds
from canslim.criteria.a_annual import AnnualEarnings
from canslim.criteria.base import CriterionContext
from canslim.criteria.c_current import CurrentEarnings
from canslim.criteria.i_institutional import Institutional
from canslim.criteria.l_leader import Leader
from canslim.criteria.n_new_high import NewHigh
from canslim.criteria.s_supply_demand import SupplyDemand
from canslim.models import EarningsBundle, InstitutionalSnapshot, PatternMatch, PriceFeatures
from canslim.providers.sec_provider import _bundle_from_facts


def _ctx(**kwargs) -> CriterionContext:
    return CriterionContext(ticker="TEST", thresholds=CriteriaThresholds(), **kwargs)


def _price_features(
    close: float = 100.0,
    high_52w: float = 105.0,
    adv10: float = 2_000_000.0,
    adv50: float = 1_500_000.0,
    avg_vol50: float = 50_000.0,
    recent_vol_ratio: float = 1.5,
    rs_weighted: float = 0.30,
) -> PriceFeatures:
    return PriceFeatures(
        ticker="TEST",
        as_of=date.today(),
        close=close,
        high_52w=high_52w,
        low_52w=50.0,
        adv10=adv10,
        adv50=adv50,
        avg_vol50=avg_vol50,
        recent_vol_ratio=recent_vol_ratio,
        rs_return_12m_weighted=rs_weighted,
        dist_to_52w_high_pct=max(0.0, (high_52w - close) / high_52w),
    )


class TestCurrent:
    def test_passes_with_accelerating_growth(self):
        eb = EarningsBundle(
            ticker="T",
            quarterly_eps=[1.50, 1.40, 1.30, 1.20, 1.00, 1.00, 1.00, 1.00, 1.00],
            quarterly_periods=["2026-Q1", "2025-Q4", "2025-Q3", "2025-Q2", "2025-Q1", "2024-Q4", "2024-Q3", "2024-Q2", "2024-Q1"],
        )
        res = CurrentEarnings().evaluate(_ctx(earnings=eb))
        assert res.passed
        assert res.value == 0.5  # (1.50 - 1.00) / 1.00

    def test_fails_below_threshold(self):
        eb = EarningsBundle(
            ticker="T",
            quarterly_eps=[1.10, 1.05, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
            quarterly_periods=["q"] * 9,
        )
        res = CurrentEarnings().evaluate(_ctx(earnings=eb))
        assert not res.passed

    def test_fails_if_decelerating(self):
        # latest_yoy = 25%, prior_yoy = 40% -> accelerating check fails
        eb = EarningsBundle(
            ticker="T",
            quarterly_eps=[1.25, 1.40, 1.30, 1.20, 1.00, 1.00, 1.00, 1.00],
            quarterly_periods=["q"] * 8,
        )
        res = CurrentEarnings().evaluate(_ctx(earnings=eb))
        assert not res.passed
        assert "not accelerating" in res.reason

    def test_abstains_when_no_earnings_bundle(self):
        # Fundamentals fetch failed entirely — eb=None must mark data_available=False
        res = CurrentEarnings().evaluate(_ctx(earnings=None))
        assert res.data_available is False
        assert not res.passed
        assert "no earnings data available" in res.reason

    def test_evaluates_when_insufficient_quarters(self):
        # Recent IPO: data was fetched but only 3 quarters exist. Real evaluation
        # (not a fetch failure) — data_available stays True, just fails the gate.
        eb = EarningsBundle(
            ticker="T",
            quarterly_eps=[1.0, 0.8, 0.5],
            quarterly_periods=["2025-Q3", "2025-Q2", "2025-Q1"],
        )
        res = CurrentEarnings().evaluate(_ctx(earnings=eb))
        assert res.data_available is True  # not a fetch failure
        assert not res.passed
        assert "insufficient quarterly EPS history" in res.reason

    def test_turnaround_passes_gate(self):
        # Q4 year ago was a loss, this Q4 is a solid profit. O'Neil: that's a C pass.
        eb = EarningsBundle(
            ticker="T",
            quarterly_eps=[0.50, 0.45, 0.40, 0.35, -0.30, -0.40, -0.20, -0.10, 0.05],
            quarterly_periods=["q"] * 9,
        )
        res = CurrentEarnings().evaluate(_ctx(earnings=eb))
        assert res.passed, f"turnaround should pass, got reason: {res.reason}"
        assert "turnaround" in res.reason.lower()
        assert res.evidence["turnaround"] is True


class TestAnnual:
    def test_passes_with_3yr_growth_and_roe(self):
        eb = EarningsBundle(
            ticker="T",
            annual_eps=[4.0, 3.0, 2.0, 1.5],
            annual_periods=["2025", "2024", "2023", "2022"],
            annual_roe_pct=[0.20, 0.18, 0.17, 0.15],
        )
        res = AnnualEarnings().evaluate(_ctx(earnings=eb))
        assert res.passed

    def test_fails_if_roe_too_low(self):
        eb = EarningsBundle(
            ticker="T",
            annual_eps=[4.0, 3.0, 2.0, 1.5],
            annual_periods=["2025", "2024", "2023", "2022"],
            annual_roe_pct=[0.10, 0.10, 0.10, 0.10],
        )
        res = AnnualEarnings().evaluate(_ctx(earnings=eb))
        assert not res.passed

    def test_turnaround_year_passes(self):
        # 2024→2025 flipped loss to profit, ROE fine. Should pass A via turnaround path.
        eb = EarningsBundle(
            ticker="T",
            annual_eps=[2.50, -0.30, -0.40, -0.50],
            annual_periods=["2025", "2024", "2023", "2022"],
            annual_roe_pct=[0.20, -0.05, -0.05, -0.05],
        )
        res = AnnualEarnings().evaluate(_ctx(earnings=eb))
        assert res.passed, f"turnaround should pass, got reason: {res.reason}"
        assert res.evidence["turnaround_years"] >= 1

    def test_leadership_override_passes_when_low_roe_but_high_rs_and_pattern(self):
        # LITE-shaped: turnaround year exists, latest EPS positive, but ROE far below
        # threshold (2.3% < 17%). Top-decile RS + a high-confidence pattern should
        # let the leadership override path pass A.
        eb = EarningsBundle(
            ticker="LITE",
            annual_eps=[0.79, -7.09, -1.43, 2.99],
            annual_periods=["2025", "2024", "2023", "2022"],
            annual_roe_pct=[0.023, -0.20, -0.05, 0.10],
        )
        pat = PatternMatch(name="high_tight_flag", detected=True, pivot=1021.10, confidence=0.73)
        res = AnnualEarnings().evaluate(_ctx(earnings=eb, rs_percentile=0.99, patterns=[pat]))
        assert res.passed, f"LITE-shaped override should pass, got reason: {res.reason}"
        assert res.evidence["override_used"] is True
        assert "leadership override" in res.reason

    def test_leadership_override_rejects_deteriorating_turnaround(self):
        # Turnaround happened years ago and the stock is now deteriorating
        # (latest YoY -40%). Even with high RS + pattern, override must NOT pass.
        eb = EarningsBundle(
            ticker="DETER",
            annual_eps=[0.30, 0.50, 1.00, -5.00],
            annual_periods=["2025", "2024", "2023", "2022"],
            annual_roe_pct=[0.04, 0.06, 0.10, -0.10],
        )
        pat = PatternMatch(name="cup_with_handle", detected=True, pivot=10.0, confidence=0.80)
        res = AnnualEarnings().evaluate(_ctx(earnings=eb, rs_percentile=0.95, patterns=[pat]))
        assert not res.passed, f"deteriorating turnaround must not pass override, got: {res.reason}"
        assert res.evidence["override_used"] is False

    def test_leadership_override_requires_market_confirmation(self):
        # Same shape as LITE but no high-RS / no pattern. Override must NOT pass.
        eb = EarningsBundle(
            ticker="NOMKT",
            annual_eps=[0.79, -7.09, -1.43, 2.99],
            annual_periods=["2025", "2024", "2023", "2022"],
            annual_roe_pct=[0.023, -0.20, -0.05, 0.10],
        )
        # Mid-RS, no pattern
        res = AnnualEarnings().evaluate(_ctx(earnings=eb, rs_percentile=0.55, patterns=[]))
        assert not res.passed, "must not pass without market confirmation"
        # High-RS but no pattern
        res2 = AnnualEarnings().evaluate(_ctx(earnings=eb, rs_percentile=0.99, patterns=[]))
        assert not res2.passed, "must not pass without a pattern"
        # Pattern but RS too low
        pat = PatternMatch(name="cup_with_handle", detected=True, confidence=0.80)
        res3 = AnnualEarnings().evaluate(_ctx(earnings=eb, rs_percentile=0.85, patterns=[pat]))
        assert not res3.passed, "must not pass when RS below override threshold"

    def test_leadership_override_can_be_disabled(self):
        # When the flag is off, LITE-shaped override path must NOT pass even with full
        # market confirmation.
        from canslim.config import CriteriaThresholds
        eb = EarningsBundle(
            ticker="LITE",
            annual_eps=[0.79, -7.09, -1.43, 2.99],
            annual_periods=["2025", "2024", "2023", "2022"],
            annual_roe_pct=[0.023, -0.20, -0.05, 0.10],
        )
        pat = PatternMatch(name="high_tight_flag", detected=True, pivot=1021.10, confidence=0.73)
        thresholds = CriteriaThresholds(a_leadership_override_enabled=False)
        ctx = CriterionContext(
            ticker="LITE",
            thresholds=thresholds,
            earnings=eb,
            rs_percentile=0.99,
            patterns=[pat],
        )
        res = AnnualEarnings().evaluate(ctx)
        assert not res.passed, "override disabled flag must be honored"


class TestNewHigh:
    def test_info_only_never_gate(self):
        pf = _price_features(close=100, high_52w=105)
        res = NewHigh().evaluate(_ctx(price_features=pf))
        assert res.is_gate is False
        assert res.passed  # within 5% of high
        assert res.evidence["breakout"] is True  # 4.76% dist and 1.5x volume


class TestSupplyDemand:
    def test_passes_with_volume_uptick(self):
        pf = _price_features(adv10=2_000_000, adv50=1_500_000)
        res = SupplyDemand().evaluate(_ctx(price_features=pf, float_shares=500_000_000))
        assert res.passed

    def test_fails_if_float_too_big(self):
        pf = _price_features(adv10=2_000_000, adv50=1_500_000)
        res = SupplyDemand().evaluate(_ctx(price_features=pf, float_shares=5_000_000_000))
        assert not res.passed

    def test_abstains_when_float_shares_missing(self):
        # Previously a missing float_shares silently passed the supply check —
        # the new behavior marks data_available=False and refuses to pass.
        pf = _price_features(adv10=2_000_000, adv50=1_500_000)  # demand passes
        res = SupplyDemand().evaluate(_ctx(price_features=pf, float_shares=None))
        assert res.data_available is False, (
            f"missing float should abstain, got data_available={res.data_available}"
        )
        assert not res.passed, "abstained criterion must not pass the gate"
        assert "float-shares fetch unavailable" in res.reason

    def test_pattern_override_passes_with_dry_up_volume(self):
        # ADV10/ADV50 = 0.94 (drying volume) — would normally fail, but a high-confidence
        # high_tight_flag is detected. Should pass via pattern override.
        pf = _price_features(adv10=940_000, adv50=1_000_000)
        pat = PatternMatch(name="high_tight_flag", detected=True, pivot=100.0, confidence=0.73)
        res = SupplyDemand().evaluate(_ctx(price_features=pf, float_shares=500_000_000, patterns=[pat]))
        assert res.passed, f"flag pattern + dry-up should pass, got: {res.reason}"
        assert res.evidence["pattern_override"] == "high_tight_flag"
        assert "pattern override" in res.reason

    def test_pattern_override_does_not_bypass_float_cap(self):
        # Mega-cap with a flag pattern still fails on float, not volume.
        pf = _price_features(adv10=940_000, adv50=1_000_000)
        pat = PatternMatch(name="cup_with_handle", detected=True, confidence=0.80)
        res = SupplyDemand().evaluate(_ctx(price_features=pf, float_shares=5_000_000_000, patterns=[pat]))
        assert not res.passed
        assert "float" in res.reason

    def test_pattern_override_ignores_low_confidence_patterns(self):
        # Pattern is below confidence threshold — must NOT trigger override.
        pf = _price_features(adv10=940_000, adv50=1_000_000)
        pat = PatternMatch(name="cup_with_handle", detected=True, confidence=0.40)
        res = SupplyDemand().evaluate(_ctx(price_features=pf, float_shares=500_000_000, patterns=[pat]))
        assert not res.passed
        assert "ADV10/ADV50" in res.reason

    def test_pattern_override_only_for_constructive_patterns(self):
        # double_bottom is not in the allowed override list — must NOT trigger.
        pf = _price_features(adv10=940_000, adv50=1_000_000)
        pat = PatternMatch(name="double_bottom", detected=True, confidence=0.80)
        res = SupplyDemand().evaluate(_ctx(price_features=pf, float_shares=500_000_000, patterns=[pat]))
        assert not res.passed
        assert "ADV10/ADV50" in res.reason


class TestLeader:
    def test_passes_at_or_above_percentile(self):
        pf = _price_features()
        res = Leader().evaluate(_ctx(price_features=pf, rs_percentile=0.75))
        assert res.passed

    def test_fails_below(self):
        pf = _price_features()
        res = Leader().evaluate(_ctx(price_features=pf, rs_percentile=0.50))
        assert not res.passed


class TestInstitutional:
    def test_passes_with_ownership_and_new_positions(self):
        snap = InstitutionalSnapshot(
            ticker="T", reported_at=date.today(), inst_own_pct=0.55,
            qoq_delta_pct=0.02, new_positions=3, closed_positions=1,
        )
        res = Institutional().evaluate(_ctx(institutional=snap))
        assert res.passed

    def test_fails_if_qoq_decreases(self):
        snap = InstitutionalSnapshot(
            ticker="T", reported_at=date.today(), inst_own_pct=0.55,
            qoq_delta_pct=-0.05, new_positions=3,
        )
        res = Institutional().evaluate(_ctx(institutional=snap))
        assert not res.passed

    def test_abstains_when_no_snapshot(self):
        # Today's VICR case: institutional fetch returned None — criterion must
        # mark data_available=False so the composite score & gate logic abstain
        # rather than dragging the ticker out of rankings.
        res = Institutional().evaluate(_ctx(institutional=None))
        assert res.data_available is False
        assert not res.passed
        assert "no institutional snapshot available" in res.reason


class TestSECParser:
    def test_us_gaap_domestic_issuer_annual_series(self):
        # Synthetic companyfacts payload: 2 years of annual EPS from a domestic 10-K filer
        facts = {
            "facts": {
                "us-gaap": {
                    "EarningsPerShareDiluted": {
                        "units": {
                            "USD/shares": [
                                {"fp": "FY", "fy": 2024, "end": "2023-12-31", "val": 1.50},
                                {"fp": "FY", "fy": 2024, "end": "2024-12-31", "val": 2.00},
                            ]
                        }
                    }
                }
            }
        }
        eb = _bundle_from_facts("TEST", facts)
        # Two distinct end-years preserved (bug where fy-keyed parse overwrote the prior-year comparative)
        assert eb.annual_periods == ["2024", "2023"]
        assert eb.annual_eps == [2.0, 1.5]

    def test_ifrs_foreign_issuer_picks_best_unit(self):
        # SGML-like: CAD/shares has more years than USD/shares. Expect CAD-years returned.
        facts = {
            "facts": {
                "ifrs-full": {
                    "DilutedEarningsLossPerShare": {
                        "units": {
                            "CAD/shares": [
                                {"fp": "FY", "fy": 2023, "end": "2020-12-31", "val": -0.02},
                                {"fp": "FY", "fy": 2023, "end": "2021-12-31", "val": -0.25},
                                {"fp": "FY", "fy": 2023, "end": "2022-12-31", "val": -1.26},
                                {"fp": "FY", "fy": 2023, "end": "2023-12-31", "val": -0.35},
                            ],
                            "USD/shares": [
                                {"fp": "FY", "fy": 2025, "end": "2024-12-31", "val": -0.46},
                                {"fp": "FY", "fy": 2025, "end": "2025-12-31", "val": -0.45},
                            ],
                        }
                    }
                }
            }
        }
        eb = _bundle_from_facts("SGML", facts)
        # CAD/shares had 4 distinct end-years vs USD/shares with 2 — prefer the longer series
        assert eb.annual_periods == ["2023", "2022", "2021", "2020"]
        assert eb.annual_eps == [-0.35, -1.26, -0.25, -0.02]
