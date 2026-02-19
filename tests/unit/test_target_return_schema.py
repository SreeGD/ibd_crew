"""
Tests for Agent 14: Target Return Portfolio Constructor — Schema Validation
Level 1: Pure Pydantic validation, no LLM, no file I/O.
"""

from __future__ import annotations

import pytest

from ibd_agents.schemas.target_return_output import (
    ACHIEVABILITY_RATINGS,
    CONVICTION_LEVELS,
    DEFAULT_FRICTION,
    DEFAULT_HORIZON_MONTHS,
    DEFAULT_TARGET_RETURN,
    DEFAULT_TOTAL_CAPITAL,
    ENTRY_STRATEGIES,
    MAX_POSITIONS,
    MAX_PROB_ACHIEVE,
    MAX_SECTOR_CONCENTRATION_PCT,
    MAX_SINGLE_POSITION_PCT,
    MIN_POSITIONS,
    REGIME_GUIDANCE,
    STANDARD_DISCLAIMER,
    AlternativePortfolio,
    ProbabilityAssessment,
    RiskDisclosure,
    ScenarioAnalysis,
    SectorWeight,
    TargetPosition,
    TargetReturnOutput,
    TargetReturnScenario,
    TargetTierAllocation,
    TransitionAction,
    TransitionPlan,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tier_allocation():
    return TargetTierAllocation(
        t1_momentum_pct=55.0,
        t2_quality_growth_pct=35.0,
        t3_defensive_pct=10.0,
        rationale="T1=55%/T2=35%/T3=10% optimized for 30% target in bull regime",
    )


@pytest.fixture
def sample_position():
    return TargetPosition(
        ticker="NVDA",
        company_name="NVIDIA",
        tier=1,
        allocation_pct=12.0,
        dollar_amount=30000.0,
        shares=40,
        entry_strategy="market",
        target_entry_price=750.0,
        stop_loss_price=585.0,
        stop_loss_pct=22.0,
        expected_return_contribution_pct=4.2,
        conviction_level="HIGH",
        selection_rationale="NVDA: Tier 1 with Composite 98, RS 81. HIGH conviction in CHIPS sector.",
        composite_score=98,
        eps_rating=99,
        rs_rating=81,
        sector="CHIPS",
        sector_rank=1,
        multi_source_count=5,
    )


@pytest.fixture
def sample_probability():
    return ProbabilityAssessment(
        prob_achieve_target=0.48,
        prob_positive_return=0.85,
        prob_beat_sp500=0.65,
        prob_beat_nasdaq=0.45,
        prob_beat_dow=0.70,
        expected_return_pct=27.5,
        median_return_pct=25.0,
        p10_return_pct=-5.0,
        p25_return_pct=10.0,
        p50_return_pct=25.0,
        p75_return_pct=40.0,
        p90_return_pct=55.0,
        key_assumptions=[
            "Bull regime continues",
            "IBD momentum persists",
            "No black swan events",
        ],
        key_risks=[
            "Sudden bear shift",
            "Sector rotation",
            "Concentration risk",
        ],
    )


@pytest.fixture
def sample_scenario():
    return TargetReturnScenario(
        name="Bull",
        probability_pct=55.0,
        portfolio_return_pct=38.0,
        sp500_return_pct=22.0,
        nasdaq_return_pct=30.0,
        dow_return_pct=18.0,
        alpha_vs_sp500=16.0,
        max_drawdown_pct=-12.0,
        description="Bull scenario: 38.0% return driven by T1 momentum leaders in favorable conditions.",
        top_contributors=["NVDA", "AVGO", "CLS"],
        biggest_drags=["GS", "TPR"],
        stops_triggered=0,
    )


@pytest.fixture
def sample_risk_disclosure():
    return RiskDisclosure(
        achievability_rating="REALISTIC",
        achievability_rationale="In a bull market regime, a 30% target is achievable with proper construction.",
        max_expected_drawdown_pct=17.5,
        recovery_time_months=4.0,
        conditions_for_success=[
            "Bull regime continues",
            "Leading sectors maintain RS",
            "Selected stocks show strong EPS growth",
        ],
        conditions_for_failure=[
            "Sudden bear regime shift",
            "Sector rotation away from positions",
            "Multiple earnings misses",
        ],
    )


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestTargetReturnConstants:
    def test_default_target_return(self):
        assert DEFAULT_TARGET_RETURN == 30.0

    def test_default_capital(self):
        assert DEFAULT_TOTAL_CAPITAL == 250_000.0

    def test_default_horizon(self):
        assert DEFAULT_HORIZON_MONTHS == 12

    def test_achievability_ratings(self):
        assert len(ACHIEVABILITY_RATINGS) == 4
        assert "REALISTIC" in ACHIEVABILITY_RATINGS
        assert "IMPROBABLE" in ACHIEVABILITY_RATINGS

    def test_conviction_levels(self):
        assert len(CONVICTION_LEVELS) == 3
        assert "HIGH" in CONVICTION_LEVELS

    def test_entry_strategies(self):
        assert len(ENTRY_STRATEGIES) == 3
        assert "market" in ENTRY_STRATEGIES

    def test_max_prob_achieve(self):
        assert MAX_PROB_ACHIEVE == 0.85

    def test_position_limits(self):
        assert MIN_POSITIONS == 5
        assert MAX_POSITIONS == 20

    def test_concentration_limits(self):
        assert MAX_SINGLE_POSITION_PCT == 15.0
        assert MAX_SECTOR_CONCENTRATION_PCT == 40.0

    def test_regime_guidance_keys(self):
        assert "bull" in REGIME_GUIDANCE
        assert "neutral" in REGIME_GUIDANCE
        assert "bear" in REGIME_GUIDANCE


# ---------------------------------------------------------------------------
# Tier Allocation Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestTargetTierAllocation:
    def test_valid_allocation(self, sample_tier_allocation):
        assert sample_tier_allocation.t1_momentum_pct == 55.0
        assert sample_tier_allocation.t2_quality_growth_pct == 35.0
        assert sample_tier_allocation.t3_defensive_pct == 10.0

    def test_allocation_sums_to_100(self):
        alloc = TargetTierAllocation(
            t1_momentum_pct=50.0,
            t2_quality_growth_pct=30.0,
            t3_defensive_pct=20.0,
            rationale="Valid allocation summing to 100%",
        )
        total = alloc.t1_momentum_pct + alloc.t2_quality_growth_pct + alloc.t3_defensive_pct
        assert 99.0 <= total <= 101.0

    def test_allocation_rejects_wrong_sum(self):
        with pytest.raises(Exception):
            TargetTierAllocation(
                t1_momentum_pct=60.0,
                t2_quality_growth_pct=30.0,
                t3_defensive_pct=20.0,
                rationale="Invalid: sums to 110%",
            )


# ---------------------------------------------------------------------------
# Position Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestTargetPosition:
    def test_valid_position(self, sample_position):
        assert sample_position.ticker == "NVDA"
        assert sample_position.tier == 1
        assert sample_position.allocation_pct == 12.0

    def test_stop_below_entry(self, sample_position):
        assert sample_position.stop_loss_price < sample_position.target_entry_price

    def test_rejects_stop_above_entry(self):
        with pytest.raises(Exception):
            TargetPosition(
                ticker="BAD",
                company_name="Bad Corp",
                tier=1,
                allocation_pct=10.0,
                dollar_amount=25000.0,
                shares=100,
                entry_strategy="market",
                target_entry_price=100.0,
                stop_loss_price=110.0,  # above entry
                stop_loss_pct=10.0,
                expected_return_contribution_pct=3.0,
                conviction_level="HIGH",
                selection_rationale="Bad stop loss test",
                composite_score=95,
                eps_rating=90,
                rs_rating=85,
                sector="CHIPS",
                sector_rank=1,
                multi_source_count=1,
            )

    def test_rejects_invalid_conviction(self):
        with pytest.raises(Exception):
            TargetPosition(
                ticker="BAD",
                company_name="Bad Corp",
                tier=1,
                allocation_pct=10.0,
                dollar_amount=25000.0,
                shares=100,
                entry_strategy="market",
                target_entry_price=100.0,
                stop_loss_price=78.0,
                stop_loss_pct=22.0,
                expected_return_contribution_pct=3.0,
                conviction_level="INVALID",
                selection_rationale="Invalid conviction test",
                composite_score=95,
                eps_rating=90,
                rs_rating=85,
                sector="CHIPS",
                sector_rank=1,
                multi_source_count=1,
            )

    def test_rejects_invalid_entry_strategy(self):
        with pytest.raises(Exception):
            TargetPosition(
                ticker="BAD",
                company_name="Bad Corp",
                tier=1,
                allocation_pct=10.0,
                dollar_amount=25000.0,
                shares=100,
                entry_strategy="yolo_buy",
                target_entry_price=100.0,
                stop_loss_price=78.0,
                stop_loss_pct=22.0,
                expected_return_contribution_pct=3.0,
                conviction_level="HIGH",
                selection_rationale="Invalid strategy test",
                composite_score=95,
                eps_rating=90,
                rs_rating=85,
                sector="CHIPS",
                sector_rank=1,
                multi_source_count=1,
            )


# ---------------------------------------------------------------------------
# Probability Assessment Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestProbabilityAssessment:
    def test_valid_assessment(self, sample_probability):
        assert sample_probability.prob_achieve_target <= MAX_PROB_ACHIEVE

    def test_percentiles_ordered(self, sample_probability):
        assert sample_probability.p10_return_pct <= sample_probability.p25_return_pct
        assert sample_probability.p25_return_pct <= sample_probability.p50_return_pct
        assert sample_probability.p50_return_pct <= sample_probability.p75_return_pct
        assert sample_probability.p75_return_pct <= sample_probability.p90_return_pct

    def test_overconfidence_cap(self):
        pa = ProbabilityAssessment(
            prob_achieve_target=0.95,  # above cap
            prob_positive_return=0.85,
            prob_beat_sp500=0.65,
            prob_beat_nasdaq=0.45,
            prob_beat_dow=0.70,
            expected_return_pct=27.5,
            median_return_pct=25.0,
            p10_return_pct=-5.0,
            p25_return_pct=10.0,
            p50_return_pct=25.0,
            p75_return_pct=40.0,
            p90_return_pct=55.0,
            key_assumptions=["A", "B", "C"],
            key_risks=["X", "Y", "Z"],
        )
        assert pa.prob_achieve_target <= MAX_PROB_ACHIEVE

    def test_rejects_disordered_percentiles(self):
        with pytest.raises(Exception):
            ProbabilityAssessment(
                prob_achieve_target=0.50,
                prob_positive_return=0.85,
                prob_beat_sp500=0.65,
                prob_beat_nasdaq=0.45,
                prob_beat_dow=0.70,
                expected_return_pct=27.5,
                median_return_pct=25.0,
                p10_return_pct=50.0,  # higher than p90!
                p25_return_pct=10.0,
                p50_return_pct=25.0,
                p75_return_pct=40.0,
                p90_return_pct=30.0,
                key_assumptions=["A", "B", "C"],
                key_risks=["X", "Y", "Z"],
            )

    def test_requires_min_assumptions(self):
        with pytest.raises(Exception):
            ProbabilityAssessment(
                prob_achieve_target=0.50,
                prob_positive_return=0.85,
                prob_beat_sp500=0.65,
                prob_beat_nasdaq=0.45,
                prob_beat_dow=0.70,
                expected_return_pct=27.5,
                median_return_pct=25.0,
                p10_return_pct=-5.0,
                p25_return_pct=10.0,
                p50_return_pct=25.0,
                p75_return_pct=40.0,
                p90_return_pct=55.0,
                key_assumptions=["A"],  # need 3
                key_risks=["X", "Y", "Z"],
            )


# ---------------------------------------------------------------------------
# Scenario Analysis Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestScenarioAnalysis:
    def test_valid_scenarios(self, sample_scenario):
        bull = sample_scenario
        base = TargetReturnScenario(
            name="Base", probability_pct=35.0, portfolio_return_pct=18.0,
            sp500_return_pct=10.5, nasdaq_return_pct=14.0, dow_return_pct=9.0,
            alpha_vs_sp500=7.5, max_drawdown_pct=-15.0,
            description="Base scenario: 18.0% return under normal conditions.",
            top_contributors=["NVDA"], biggest_drags=["GS"], stops_triggered=1,
        )
        bear = TargetReturnScenario(
            name="Bear", probability_pct=10.0, portfolio_return_pct=-6.0,
            sp500_return_pct=-15.0, nasdaq_return_pct=-22.0, dow_return_pct=-12.0,
            alpha_vs_sp500=9.0, max_drawdown_pct=-20.0,
            description="Bear scenario: -6.0% return in adverse conditions.",
            top_contributors=["GS"], biggest_drags=["NVDA"], stops_triggered=3,
        )
        sa = ScenarioAnalysis(
            bull_scenario=bull, base_scenario=base, bear_scenario=bear,
        )
        total_prob = (
            sa.bull_scenario.probability_pct
            + sa.base_scenario.probability_pct
            + sa.bear_scenario.probability_pct
        )
        assert 98.0 <= total_prob <= 102.0

    def test_rejects_wrong_probability_sum(self, sample_scenario):
        bull = sample_scenario
        base = TargetReturnScenario(
            name="Base", probability_pct=55.0, portfolio_return_pct=18.0,
            sp500_return_pct=10.5, nasdaq_return_pct=14.0, dow_return_pct=9.0,
            alpha_vs_sp500=7.5, max_drawdown_pct=-15.0,
            description="Base scenario: 18.0% return under normal conditions.",
            top_contributors=["NVDA"], biggest_drags=["GS"], stops_triggered=1,
        )
        bear = TargetReturnScenario(
            name="Bear", probability_pct=55.0, portfolio_return_pct=-6.0,
            sp500_return_pct=-15.0, nasdaq_return_pct=-22.0, dow_return_pct=-12.0,
            alpha_vs_sp500=9.0, max_drawdown_pct=-20.0,
            description="Bear scenario: -6.0% return in adverse conditions.",
            top_contributors=["GS"], biggest_drags=["NVDA"], stops_triggered=3,
        )
        with pytest.raises(Exception):
            ScenarioAnalysis(
                bull_scenario=bull, base_scenario=base, bear_scenario=bear,
            )


# ---------------------------------------------------------------------------
# Risk Disclosure Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestRiskDisclosure:
    def test_valid_disclosure(self, sample_risk_disclosure):
        assert sample_risk_disclosure.achievability_rating == "REALISTIC"
        assert sample_risk_disclosure.disclaimer == STANDARD_DISCLAIMER

    def test_rejects_invalid_rating(self):
        with pytest.raises(Exception):
            RiskDisclosure(
                achievability_rating="MAYBE",
                achievability_rationale="Invalid rating test - this is not a valid achievability rating.",
                max_expected_drawdown_pct=17.5,
                recovery_time_months=4.0,
                conditions_for_success=["A", "B", "C"],
                conditions_for_failure=["X", "Y", "Z"],
            )

    def test_requires_min_conditions(self):
        with pytest.raises(Exception):
            RiskDisclosure(
                achievability_rating="REALISTIC",
                achievability_rationale="Missing conditions test - this is a test.",
                max_expected_drawdown_pct=17.5,
                recovery_time_months=4.0,
                conditions_for_success=["A"],  # need 3
                conditions_for_failure=["X", "Y", "Z"],
            )


# ---------------------------------------------------------------------------
# Alternative Portfolio Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestAlternativePortfolio:
    def test_valid_alternative(self):
        alt = AlternativePortfolio(
            name="Higher Probability (20% target)",
            target_return_pct=20.0,
            prob_achieve_target=0.65,
            position_count=12,
            t1_pct=40.0,
            t2_pct=40.0,
            t3_pct=20.0,
            max_drawdown_pct=15.0,
            key_difference="Lower target allows more diversification and lower risk",
            tradeoff="Gain: higher probability. Give up: 10% potential return",
        )
        assert alt.position_count >= MIN_POSITIONS
        assert alt.position_count <= MAX_POSITIONS


# ---------------------------------------------------------------------------
# Transition Plan Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestTransitionPlan:
    def test_valid_plan(self):
        plan = TransitionPlan(
            positions_to_sell=[
                TransitionAction(
                    ticker="OLD", action="SELL_FULL",
                    current_allocation_pct=5.0, target_allocation_pct=0.0,
                    dollar_amount=12500.0, priority=1,
                    rationale="Not in target portfolio",
                ),
            ],
            positions_to_buy=[
                TransitionAction(
                    ticker="NEW", action="BUY_NEW",
                    current_allocation_pct=0.0, target_allocation_pct=10.0,
                    dollar_amount=25000.0, priority=2,
                    rationale="New target position",
                ),
            ],
            positions_to_keep=["NVDA", "AVGO"],
            transition_urgency="phase_over_1_week",
            transition_sequence=[
                "Step 1: Sell OLD to free capital",
                "Step 2: Buy NEW at target allocation",
            ],
        )
        assert len(plan.positions_to_sell) == 1
        assert len(plan.positions_to_buy) == 1
        assert len(plan.positions_to_keep) == 2
