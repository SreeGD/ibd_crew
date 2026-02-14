"""
Agent 04: Sector Strategist — Schema Tests
Level 1: Pure Pydantic validation, no LLM, no file I/O.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ibd_agents.schemas.strategy_output import (
    DEFENSIVE_THEMES,
    GROWTH_THEMES,
    REGIME_ACTIONS,
    SECTOR_LIMITS,
    THEME_ETFS,
    TIER_ALLOCATION_TARGETS,
    VALID_CONVICTION_LEVELS,
    RotationActionSignal,
    SectorAllocationPlan,
    SectorStrategyOutput,
    ThemeRecommendation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tier_alloc(sectors: dict[str, float] | None = None) -> dict[str, float]:
    """Make a tier allocation dict that sums to ~100%."""
    if sectors:
        return sectors
    return {
        "CHIPS": 15.0, "SOFTWARE": 12.0, "MINING": 10.0,
        "AEROSPACE": 10.0, "MEDICAL": 10.0, "BANKS": 10.0,
        "INTERNET": 8.0, "ENERGY": 8.0, "CONSUMER": 7.0,
        "BUILDING": 5.0, "FINANCE": 5.0,
    }


def _make_overall_alloc() -> dict[str, float]:
    """Make an overall allocation with 11 sectors summing to ~100%."""
    return {
        "CHIPS": 14.0, "SOFTWARE": 11.0, "MINING": 10.0,
        "AEROSPACE": 9.0, "MEDICAL": 9.0, "BANKS": 9.0,
        "INTERNET": 8.0, "ENERGY": 7.0, "CONSUMER": 7.0,
        "BUILDING": 6.0, "FINANCE": 5.0, "ELECTRONICS": 5.0,
    }


def _make_allocation_plan(**kw) -> SectorAllocationPlan:
    return SectorAllocationPlan(
        tier_1_allocation=kw.get("tier_1_allocation", _make_tier_alloc()),
        tier_2_allocation=kw.get("tier_2_allocation", _make_tier_alloc()),
        tier_3_allocation=kw.get("tier_3_allocation", _make_tier_alloc()),
        overall_allocation=kw.get("overall_allocation", _make_overall_alloc()),
        tier_targets=kw.get("tier_targets", {"T1": 39.0, "T2": 37.0, "T3": 22.0, "Cash": 2.0}),
        cash_recommendation=kw.get("cash_recommendation", 2.0),
        rationale=kw.get("rationale", "Bull regime with no rotation: maintaining baseline allocation targets across diversified sectors"),
    )


def _make_theme_rec(**kw) -> ThemeRecommendation:
    return ThemeRecommendation(
        theme=kw.get("theme", "Artificial Intelligence"),
        recommended_etfs=kw.get("recommended_etfs", ["BOTZ", "AIQ"]),
        tier_fit=kw.get("tier_fit", 1),
        allocation_suggestion=kw.get("allocation_suggestion", "3-5% of T1"),
        conviction=kw.get("conviction", "HIGH"),
        rationale=kw.get("rationale", "AI theme aligns with strong CHIPS/SOFTWARE sectors in destination"),
    )


def _make_rotation_signal(**kw) -> RotationActionSignal:
    return RotationActionSignal(
        action=kw.get("action", "Maintain current sector allocations within existing leadership"),
        trigger=kw.get("trigger", "Rotation verdict changes from NONE to EMERGING or ACTIVE"),
        confirmation=kw.get("confirmation", ["2+ rotation signals trigger"]),
        invalidation=kw.get("invalidation", ["RS gap narrows below 10"]),
    )


def _make_output(**kw) -> SectorStrategyOutput:
    return SectorStrategyOutput(
        rotation_response=kw.get("rotation_response", "No rotation detected: maintaining current sector outlook and optimizing within existing leadership"),
        regime_adjustment=kw.get("regime_adjustment", "Bull regime: T1 target increased to 44%, T3 reduced to 17%, favoring momentum exposure"),
        sector_allocations=kw.get("sector_allocations", _make_allocation_plan()),
        theme_recommendations=kw.get("theme_recommendations", [_make_theme_rec()]),
        rotation_signals=kw.get("rotation_signals", [_make_rotation_signal()]),
        analysis_date=kw.get("analysis_date", "2026-02-10"),
        summary=kw.get("summary", "Sector Strategy: bull regime, no rotation. Maintaining baseline 39/37/22 allocation with momentum bias. 11 sectors allocated."),
    )


# ---------------------------------------------------------------------------
# SectorAllocationPlan Tests
# ---------------------------------------------------------------------------

class TestSectorAllocationPlan:

    @pytest.mark.schema
    def test_valid_allocation_plan(self):
        plan = _make_allocation_plan()
        assert plan.cash_recommendation == 2.0

    @pytest.mark.schema
    def test_no_sector_exceeds_40pct(self):
        bad_overall = _make_overall_alloc()
        bad_overall["CHIPS"] = 45.0
        with pytest.raises(ValidationError, match="exceeds max 40%"):
            _make_allocation_plan(overall_allocation=bad_overall)

    @pytest.mark.schema
    def test_min_8_sectors(self):
        few_sectors = {"CHIPS": 30.0, "SOFTWARE": 20.0, "MINING": 15.0,
                       "BANKS": 10.0, "MEDICAL": 10.0, "ENERGY": 8.0,
                       "AEROSPACE": 7.0}  # Only 7
        with pytest.raises(ValidationError, match="minimum is 8"):
            _make_allocation_plan(overall_allocation=few_sectors)

    @pytest.mark.schema
    def test_tier_targets_sum_to_100(self):
        plan = _make_allocation_plan()
        total = sum(plan.tier_targets.values())
        assert 95.0 <= total <= 105.0

    @pytest.mark.schema
    def test_tier_targets_sum_invalid(self):
        with pytest.raises(ValidationError, match="tier_targets sum"):
            _make_allocation_plan(
                tier_targets={"T1": 40.0, "T2": 40.0, "T3": 20.0, "Cash": 10.0}
            )

    @pytest.mark.schema
    def test_tier_1_range(self):
        with pytest.raises(ValidationError, match="T1"):
            _make_allocation_plan(
                tier_targets={"T1": 20.0, "T2": 50.0, "T3": 25.0, "Cash": 5.0}
            )

    @pytest.mark.schema
    def test_tier_3_range(self):
        with pytest.raises(ValidationError, match="T3"):
            _make_allocation_plan(
                tier_targets={"T1": 39.0, "T2": 37.0, "T3": 14.0, "Cash": 10.0}
            )

    @pytest.mark.schema
    def test_cash_range_low(self):
        with pytest.raises(ValidationError):
            _make_allocation_plan(cash_recommendation=1.0)

    @pytest.mark.schema
    def test_cash_range_high(self):
        with pytest.raises(ValidationError):
            _make_allocation_plan(cash_recommendation=20.0)

    @pytest.mark.schema
    def test_tier_allocation_sums_95_to_105(self):
        bad_t1 = {"CHIPS": 80.0, "SOFTWARE": 30.0}  # 110%
        with pytest.raises(ValidationError, match="tier_1_allocation sums"):
            _make_allocation_plan(tier_1_allocation=bad_t1)

    @pytest.mark.schema
    def test_rationale_min_length(self):
        with pytest.raises(ValidationError):
            _make_allocation_plan(rationale="Too short")


# ---------------------------------------------------------------------------
# ThemeRecommendation Tests
# ---------------------------------------------------------------------------

class TestThemeRecommendation:

    @pytest.mark.schema
    def test_valid_theme_recommendation(self):
        tr = _make_theme_rec()
        assert tr.theme == "Artificial Intelligence"
        assert tr.conviction == "HIGH"

    @pytest.mark.schema
    def test_theme_must_be_in_mapping(self):
        with pytest.raises(ValidationError, match="not in THEME_ETFS"):
            _make_theme_rec(theme="Invalid Theme")

    @pytest.mark.schema
    def test_etfs_must_exist_in_theme(self):
        with pytest.raises(ValidationError, match="not in THEME_ETFS"):
            _make_theme_rec(recommended_etfs=["FAKE_ETF"])

    @pytest.mark.schema
    def test_growth_theme_tier_fit_1_or_2(self):
        with pytest.raises(ValidationError, match="tier_fit 1 or 2"):
            _make_theme_rec(theme="Cybersecurity", recommended_etfs=["CIBR"], tier_fit=3)

    @pytest.mark.schema
    def test_defensive_theme_tier_fit_2_or_3(self):
        with pytest.raises(ValidationError, match="tier_fit 2 or 3"):
            _make_theme_rec(
                theme="Healthcare Innovation",
                recommended_etfs=["XBI"],
                tier_fit=1,
            )

    @pytest.mark.schema
    def test_conviction_valid_values(self):
        for conv in ("HIGH", "MEDIUM", "LOW"):
            tr = _make_theme_rec(conviction=conv)
            assert tr.conviction == conv

    @pytest.mark.schema
    def test_conviction_invalid(self):
        with pytest.raises(ValidationError, match="conviction"):
            _make_theme_rec(conviction="VERY_HIGH")

    @pytest.mark.schema
    def test_empty_etfs_allowed(self):
        """Themes with no ETFs (like Aging Population) can have empty list."""
        tr = _make_theme_rec(
            theme="Aging Population",
            recommended_etfs=[],
            tier_fit=3,
        )
        assert tr.recommended_etfs == []


# ---------------------------------------------------------------------------
# RotationActionSignal Tests
# ---------------------------------------------------------------------------

class TestRotationActionSignal:

    @pytest.mark.schema
    def test_valid_signal(self):
        sig = _make_rotation_signal()
        assert len(sig.confirmation) >= 1

    @pytest.mark.schema
    def test_action_min_length(self):
        with pytest.raises(ValidationError):
            _make_rotation_signal(action="Short")

    @pytest.mark.schema
    def test_confirmation_required(self):
        with pytest.raises(ValidationError):
            _make_rotation_signal(confirmation=[])

    @pytest.mark.schema
    def test_invalidation_required(self):
        with pytest.raises(ValidationError):
            _make_rotation_signal(invalidation=[])


# ---------------------------------------------------------------------------
# SectorStrategyOutput Tests
# ---------------------------------------------------------------------------

class TestSectorStrategyOutput:

    @pytest.mark.schema
    def test_valid_output(self):
        output = _make_output()
        assert output.analysis_date == "2026-02-10"

    @pytest.mark.schema
    def test_summary_min_length(self):
        with pytest.raises(ValidationError):
            _make_output(summary="Too short")

    @pytest.mark.schema
    def test_analysis_date_format(self):
        with pytest.raises(ValidationError):
            _make_output(analysis_date="Feb 10, 2026")

    @pytest.mark.schema
    def test_rotation_response_min_length(self):
        with pytest.raises(ValidationError):
            _make_output(rotation_response="Short")

    @pytest.mark.schema
    def test_regime_adjustment_min_length(self):
        with pytest.raises(ValidationError):
            _make_output(regime_adjustment="Short")


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    @pytest.mark.schema
    def test_theme_etfs_contains_16_themes(self):
        assert len(THEME_ETFS) == 16

    @pytest.mark.schema
    def test_regime_actions_has_3_regimes(self):
        assert set(REGIME_ACTIONS.keys()) == {"bull", "bear", "neutral"}

    @pytest.mark.schema
    def test_tier_allocation_targets_sum_100(self):
        total = sum(t["target_pct"] for t in TIER_ALLOCATION_TARGETS.values())
        assert total == 100

    @pytest.mark.schema
    def test_growth_and_defensive_themes_no_overlap(self):
        assert GROWTH_THEMES & DEFENSIVE_THEMES == set()

    @pytest.mark.schema
    def test_all_themes_classified(self):
        """Every theme is either growth or defensive."""
        all_themes = set(THEME_ETFS.keys())
        classified = GROWTH_THEMES | DEFENSIVE_THEMES
        assert all_themes == classified
