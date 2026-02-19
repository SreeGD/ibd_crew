"""Agent 14: Target Return Constructor — Pipeline & Behavioral Tests
Level 2: Deterministic pipeline tests with mock data.
Level 3: Behavioral boundary tests.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from ibd_agents.agents.analyst_agent import run_analyst_pipeline
from ibd_agents.agents.portfolio_manager import run_portfolio_pipeline
from ibd_agents.agents.returns_projector import run_returns_projector_pipeline
from ibd_agents.agents.risk_officer import run_risk_pipeline
from ibd_agents.agents.rotation_detector import run_rotation_pipeline
from ibd_agents.agents.sector_strategist import run_strategist_pipeline
from ibd_agents.agents.target_return_constructor import run_target_return_pipeline
from ibd_agents.schemas.research_output import (
    ResearchOutput,
    ResearchStock,
    compute_preliminary_tier,
    is_ibd_keep_candidate,
)
from ibd_agents.schemas.target_return_output import (
    ACHIEVABILITY_RATINGS,
    DEFAULT_TARGET_RETURN,
    MAX_POSITIONS,
    MAX_PROB_ACHIEVE,
    MAX_SECTOR_CONCENTRATION_PCT,
    MAX_SINGLE_POSITION_PCT,
    MIN_POSITIONS,
    REGIME_GUIDANCE,
    TargetReturnOutput,
)
from ibd_agents.tools.candidate_ranker import rank_candidates, select_positions
from ibd_agents.tools.constraint_validator_target import validate_constraints
from ibd_agents.tools.monte_carlo_engine import (
    compute_benchmark_beat_probability,
    run_monte_carlo,
)
from ibd_agents.tools.tier_mix_optimizer import (
    get_regime_guidance,
    solve_tier_mix,
)

from tests.fixtures.conftest import PADDING_STOCKS, SAMPLE_IBD_STOCKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_research_output(stocks_data: list[dict]) -> ResearchOutput:
    research_stocks = []
    for s in stocks_data:
        tier = compute_preliminary_tier(
            s.get("composite_rating"), s.get("rs_rating"), s.get("eps_rating")
        )
        keep = is_ibd_keep_candidate(s.get("composite_rating"), s.get("rs_rating"))
        has_ratings = sum(1 for r in [s.get("composite_rating"), s.get("rs_rating"),
                                       s.get("eps_rating"), s.get("smr_rating"),
                                       s.get("acc_dis_rating")] if r is not None)
        confidence = min(1.0, 0.3 + (has_ratings * 0.14))

        research_stocks.append(ResearchStock(
            symbol=s["symbol"], company_name=s.get("company_name", s["symbol"]),
            sector=s.get("sector", "UNKNOWN"), security_type="stock",
            composite_rating=s.get("composite_rating"), rs_rating=s.get("rs_rating"),
            eps_rating=s.get("eps_rating"), smr_rating=s.get("smr_rating"),
            acc_dis_rating=s.get("acc_dis_rating"), ibd_lists=["IBD 50"],
            schwab_themes=[], fool_status=None, other_ratings={},
            validation_score=3, validation_providers=1, is_multi_source_validated=False,
            is_ibd_keep_candidate=keep, preliminary_tier=tier, sources=["test_data.xls"],
            confidence=confidence, reasoning="Test stock for target return pipeline testing",
        ))

    return ResearchOutput(
        stocks=research_stocks, etfs=[], sector_patterns=[],
        data_sources_used=["test_data.xls"], data_sources_failed=[],
        total_securities_scanned=len(research_stocks),
        ibd_keep_candidates=[s.symbol for s in research_stocks if s.is_ibd_keep_candidate],
        multi_source_validated=[],
        analysis_date=date.today().isoformat(),
        summary="Test research output for target return pipeline testing with sample data",
    )


def _build_full_pipeline():
    """Build upstream agents 01-07 for target return constructor input."""
    research = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
    analyst = run_analyst_pipeline(research)
    rotation = run_rotation_pipeline(analyst, research)
    strategy = run_strategist_pipeline(rotation, analyst)
    portfolio = run_portfolio_pipeline(strategy, analyst)
    risk = run_risk_pipeline(portfolio, strategy)
    returns = run_returns_projector_pipeline(portfolio, rotation, risk, analyst)
    return analyst, rotation, strategy, risk, returns


_cached = None


def _get_pipeline_outputs():
    global _cached
    if _cached is None:
        _cached = _build_full_pipeline()
    return _cached


# ---------------------------------------------------------------------------
# Tool Function Tests
# ---------------------------------------------------------------------------

class TestToolFunctions:

    @pytest.mark.schema
    def test_tier_mix_optimizer_returns_solutions(self):
        """solve_tier_mix returns at least 1 solution."""
        solutions = solve_tier_mix(target_return=30.0, regime="bull")
        assert len(solutions) >= 1

    @pytest.mark.schema
    def test_tier_mix_optimizer_top_5(self):
        """solve_tier_mix returns at most 5 solutions."""
        solutions = solve_tier_mix(target_return=30.0, regime="bull")
        assert len(solutions) <= 5

    @pytest.mark.schema
    def test_tier_mix_weights_sum_to_100(self):
        """Each solution's w1 + w2 + w3 sums to 100%."""
        solutions = solve_tier_mix(target_return=30.0, regime="bull")
        for sol in solutions:
            total = sol["w1"] + sol["w2"] + sol["w3"]
            assert abs(total - 100.0) < 0.1, f"Weights sum to {total}"

    @pytest.mark.schema
    def test_tier_mix_sorted_by_probability(self):
        """Solutions are sorted by prob_achieve descending."""
        solutions = solve_tier_mix(target_return=30.0, regime="bull")
        for i in range(len(solutions) - 1):
            assert solutions[i]["prob_achieve"] >= solutions[i + 1]["prob_achieve"]

    @pytest.mark.schema
    def test_tier_mix_prob_capped_at_085(self):
        """No solution's prob_achieve exceeds 0.85."""
        solutions = solve_tier_mix(target_return=10.0, regime="bull")
        for sol in solutions:
            assert sol["prob_achieve"] <= MAX_PROB_ACHIEVE

    @pytest.mark.schema
    def test_regime_guidance_bull(self):
        """Bull regime returns REALISTIC achievability."""
        guidance = get_regime_guidance("bull")
        assert guidance["achievability"] == "REALISTIC"

    @pytest.mark.schema
    def test_regime_guidance_bear(self):
        """Bear regime returns IMPROBABLE achievability."""
        guidance = get_regime_guidance("bear")
        assert guidance["achievability"] == "IMPROBABLE"

    @pytest.mark.schema
    def test_regime_guidance_unknown_falls_back(self):
        """Unknown regime falls back to neutral guidance."""
        guidance = get_regime_guidance("unknown_regime")
        assert guidance["achievability"] == REGIME_GUIDANCE["neutral"]["achievability"]

    @pytest.mark.schema
    def test_candidate_ranker_respects_tier(self):
        """rank_candidates only returns stocks matching requested tier."""
        analyst, *_ = _get_pipeline_outputs()
        candidates = rank_candidates(analyst.rated_stocks, tier=1)
        for c in candidates:
            assert c["tier"] == 1

    @pytest.mark.schema
    def test_candidate_ranker_respects_exclusions(self):
        """rank_candidates excludes specified tickers."""
        analyst, *_ = _get_pipeline_outputs()
        excluded = ["MU", "CLS"]
        candidates = rank_candidates(
            analyst.rated_stocks, tier=1, excluded_stocks=excluded,
        )
        tickers = {c["ticker"] for c in candidates}
        for ex in excluded:
            assert ex not in tickers

    @pytest.mark.schema
    def test_candidate_ranker_sorted_by_score(self):
        """Candidates are sorted by candidate_score descending (non-required first)."""
        analyst, *_ = _get_pipeline_outputs()
        candidates = rank_candidates(analyst.rated_stocks, tier=1)
        for i in range(len(candidates) - 1):
            assert candidates[i]["candidate_score"] >= candidates[i + 1]["candidate_score"]

    @pytest.mark.schema
    def test_select_positions_count(self):
        """select_positions returns requested count (or fewer if not enough candidates)."""
        analyst, *_ = _get_pipeline_outputs()
        candidates = rank_candidates(analyst.rated_stocks, tier=1)
        positions = select_positions(candidates, tier_weight_pct=50.0, position_count=3, total_capital=250_000)
        assert len(positions) <= 3
        assert len(positions) >= 1

    @pytest.mark.schema
    def test_select_positions_allocation_within_bounds(self):
        """No position exceeds MAX_SINGLE_POSITION_PCT."""
        analyst, *_ = _get_pipeline_outputs()
        candidates = rank_candidates(analyst.rated_stocks, tier=1)
        positions = select_positions(candidates, tier_weight_pct=50.0, position_count=5, total_capital=250_000)
        for pos in positions:
            assert pos["allocation_pct"] <= MAX_SINGLE_POSITION_PCT + 0.1

    @pytest.mark.schema
    def test_monte_carlo_deterministic(self):
        """Same seed produces same results."""
        positions = [
            {"tier": 1, "allocation_pct": 50.0, "stop_loss_pct": 22.0},
            {"tier": 2, "allocation_pct": 30.0, "stop_loss_pct": 18.0},
            {"tier": 3, "allocation_pct": 20.0, "stop_loss_pct": 12.0},
        ]
        tier_pcts = {1: 50.0, 2: 30.0, 3: 20.0}
        r1 = run_monte_carlo(positions, tier_pcts, "bull", 30.0, seed=42)
        r2 = run_monte_carlo(positions, tier_pcts, "bull", 30.0, seed=42)
        assert r1["prob_achieve_target"] == r2["prob_achieve_target"]
        assert r1["expected_return_pct"] == r2["expected_return_pct"]

    @pytest.mark.schema
    def test_monte_carlo_prob_capped(self):
        """prob_achieve_target never exceeds MAX_PROB_ACHIEVE."""
        positions = [
            {"tier": 1, "allocation_pct": 100.0, "stop_loss_pct": 22.0},
        ]
        tier_pcts = {1: 100.0, 2: 0.0, 3: 0.0}
        result = run_monte_carlo(positions, tier_pcts, "bull", 5.0, seed=42)
        assert result["prob_achieve_target"] <= MAX_PROB_ACHIEVE

    @pytest.mark.schema
    def test_monte_carlo_percentiles_ordered(self):
        """p10 <= p25 <= p50 <= p75 <= p90."""
        positions = [
            {"tier": 1, "allocation_pct": 50.0, "stop_loss_pct": 22.0},
            {"tier": 2, "allocation_pct": 30.0, "stop_loss_pct": 18.0},
            {"tier": 3, "allocation_pct": 20.0, "stop_loss_pct": 12.0},
        ]
        tier_pcts = {1: 50.0, 2: 30.0, 3: 20.0}
        result = run_monte_carlo(positions, tier_pcts, "bull", 30.0, seed=42)
        assert result["p10"] <= result["p25"] <= result["p50"] <= result["p75"] <= result["p90"]

    @pytest.mark.schema
    def test_monte_carlo_bull_higher_expected_than_bear(self):
        """Bull regime produces higher expected returns than bear."""
        positions = [
            {"tier": 1, "allocation_pct": 50.0, "stop_loss_pct": 22.0},
            {"tier": 2, "allocation_pct": 30.0, "stop_loss_pct": 18.0},
            {"tier": 3, "allocation_pct": 20.0, "stop_loss_pct": 12.0},
        ]
        tier_pcts = {1: 50.0, 2: 30.0, 3: 20.0}
        bull = run_monte_carlo(positions, tier_pcts, "bull", 30.0, seed=42)
        bear = run_monte_carlo(positions, tier_pcts, "bear", 30.0, seed=42)
        assert bull["expected_return_pct"] > bear["expected_return_pct"]

    @pytest.mark.schema
    def test_benchmark_beat_probability_range(self):
        """Benchmark beat probability is between 0 and 1."""
        positions = [
            {"tier": 1, "allocation_pct": 50.0, "stop_loss_pct": 22.0},
            {"tier": 2, "allocation_pct": 30.0, "stop_loss_pct": 18.0},
            {"tier": 3, "allocation_pct": 20.0, "stop_loss_pct": 12.0},
        ]
        tier_pcts = {1: 50.0, 2: 30.0, 3: 20.0}
        result = run_monte_carlo(positions, tier_pcts, "bull", 30.0, seed=42)
        prob = compute_benchmark_beat_probability(result["returns_distribution"], "SPY", "bull")
        assert 0.0 <= prob <= 1.0

    @pytest.mark.schema
    def test_constraint_validator_passes_valid(self):
        """Valid portfolio passes constraint validation."""
        positions = [
            {"ticker": f"T{i}", "allocation_pct": 8.0, "sector": f"SEC{i}",
             "stop_loss_price": 90.0, "target_entry_price": 100.0}
            for i in range(10)
        ]
        result = validate_constraints(positions, cash_reserve_pct=20.0)
        assert result["passed"] is True

    @pytest.mark.schema
    def test_constraint_validator_catches_over_concentration(self):
        """Validator flags single position exceeding 15%."""
        positions = [
            {"ticker": "BIG", "allocation_pct": 20.0, "sector": "SEC1",
             "stop_loss_price": 90.0, "target_entry_price": 100.0},
        ] + [
            {"ticker": f"T{i}", "allocation_pct": 8.0, "sector": f"SEC{i}",
             "stop_loss_price": 90.0, "target_entry_price": 100.0}
            for i in range(8)
        ]
        result = validate_constraints(positions, cash_reserve_pct=16.0)
        assert result["passed"] is False
        assert any("BIG" in v for v in result["violations"])

    @pytest.mark.schema
    def test_constraint_validator_catches_stop_above_entry(self):
        """Validator flags stop_loss_price >= target_entry_price."""
        positions = [
            {"ticker": "BAD", "allocation_pct": 10.0, "sector": "SEC1",
             "stop_loss_price": 110.0, "target_entry_price": 100.0},
        ] + [
            {"ticker": f"T{i}", "allocation_pct": 10.0, "sector": f"SEC{i}",
             "stop_loss_price": 90.0, "target_entry_price": 100.0}
            for i in range(7)
        ]
        result = validate_constraints(positions, cash_reserve_pct=20.0)
        assert result["passed"] is False
        assert any("BAD" in v for v in result["violations"])


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

class TestTargetReturnPipeline:

    @pytest.fixture
    def pipeline_output(self):
        analyst, rotation, strategy, risk, returns = _get_pipeline_outputs()
        return run_target_return_pipeline(analyst, rotation, strategy, risk, returns)

    @pytest.mark.schema
    def test_pipeline_produces_valid_output(self, pipeline_output):
        assert isinstance(pipeline_output, TargetReturnOutput)

    @pytest.mark.schema
    def test_position_count_in_range(self, pipeline_output):
        n = len(pipeline_output.positions)
        assert MIN_POSITIONS <= n <= MAX_POSITIONS, f"Got {n} positions"

    @pytest.mark.schema
    def test_allocations_sum_to_100(self, pipeline_output):
        pos_sum = sum(p.allocation_pct for p in pipeline_output.positions)
        total = pos_sum + pipeline_output.cash_reserve_pct
        assert 97.0 <= total <= 103.0, f"Total allocation {total:.1f}%"

    @pytest.mark.schema
    def test_no_position_exceeds_max(self, pipeline_output):
        for p in pipeline_output.positions:
            assert p.allocation_pct <= MAX_SINGLE_POSITION_PCT + 0.1, (
                f"{p.ticker} allocation {p.allocation_pct:.1f}%"
            )

    @pytest.mark.schema
    def test_no_sector_exceeds_max(self, pipeline_output):
        for sw in pipeline_output.sector_weights:
            assert sw.weight_pct <= MAX_SECTOR_CONCENTRATION_PCT + 0.1, (
                f"Sector {sw.sector} weight {sw.weight_pct:.1f}%"
            )

    @pytest.mark.schema
    def test_prob_achieve_target_capped(self, pipeline_output):
        assert pipeline_output.probability_assessment.prob_achieve_target <= MAX_PROB_ACHIEVE

    @pytest.mark.schema
    def test_percentiles_ordered(self, pipeline_output):
        pa = pipeline_output.probability_assessment
        assert pa.p10_return_pct <= pa.p25_return_pct <= pa.p50_return_pct
        assert pa.p50_return_pct <= pa.p75_return_pct <= pa.p90_return_pct

    @pytest.mark.schema
    def test_scenario_probabilities_sum_to_100(self, pipeline_output):
        s = pipeline_output.scenarios
        total = (
            s.bull_scenario.probability_pct
            + s.base_scenario.probability_pct
            + s.bear_scenario.probability_pct
        )
        assert 98.0 <= total <= 102.0, f"Scenario probabilities sum to {total:.1f}%"

    @pytest.mark.schema
    def test_bull_return_gt_base_gt_bear(self, pipeline_output):
        s = pipeline_output.scenarios
        assert s.bull_scenario.portfolio_return_pct > s.base_scenario.portfolio_return_pct
        assert s.base_scenario.portfolio_return_pct > s.bear_scenario.portfolio_return_pct

    @pytest.mark.schema
    def test_alternatives_count(self, pipeline_output):
        assert 2 <= len(pipeline_output.alternatives) <= 3

    @pytest.mark.schema
    def test_risk_disclosure_present(self, pipeline_output):
        rd = pipeline_output.risk_disclosure
        assert rd.achievability_rating in ACHIEVABILITY_RATINGS
        assert len(rd.conditions_for_success) >= 3
        assert len(rd.conditions_for_failure) >= 3
        assert len(rd.disclaimer) > 0

    @pytest.mark.schema
    def test_construction_rationale_min_length(self, pipeline_output):
        assert len(pipeline_output.construction_rationale) >= 100

    @pytest.mark.schema
    def test_stop_loss_below_entry_for_all_positions(self, pipeline_output):
        for p in pipeline_output.positions:
            assert p.stop_loss_price < p.target_entry_price, (
                f"{p.ticker}: stop={p.stop_loss_price} >= entry={p.target_entry_price}"
            )

    @pytest.mark.schema
    def test_all_positions_have_rationale(self, pipeline_output):
        for p in pipeline_output.positions:
            assert len(p.selection_rationale) >= 10

    @pytest.mark.schema
    def test_return_contributions_near_target(self, pipeline_output):
        contrib_sum = sum(p.expected_return_contribution_pct for p in pipeline_output.positions)
        target = pipeline_output.target_return_pct
        assert abs(contrib_sum - target) <= 5.0, (
            f"Contribution sum {contrib_sum:.1f}% vs target {target:.1f}%"
        )

    @pytest.mark.schema
    def test_tier_allocation_sums_to_100(self, pipeline_output):
        ta = pipeline_output.tier_allocation
        total = ta.t1_momentum_pct + ta.t2_quality_growth_pct + ta.t3_defensive_pct
        assert 98.0 <= total <= 102.0

    @pytest.mark.schema
    def test_target_return_is_default(self, pipeline_output):
        assert pipeline_output.target_return_pct == DEFAULT_TARGET_RETURN

    @pytest.mark.schema
    def test_key_assumptions_min_3(self, pipeline_output):
        assert len(pipeline_output.probability_assessment.key_assumptions) >= 3

    @pytest.mark.schema
    def test_key_risks_min_3(self, pipeline_output):
        assert len(pipeline_output.probability_assessment.key_risks) >= 3

    @pytest.mark.schema
    def test_sector_weights_present(self, pipeline_output):
        assert len(pipeline_output.sector_weights) >= 1

    @pytest.mark.schema
    def test_analysis_date_is_today(self, pipeline_output):
        assert pipeline_output.analysis_date == date.today().isoformat()


# ---------------------------------------------------------------------------
# Behavioral Boundary Tests
# ---------------------------------------------------------------------------

class TestBehavioralBoundaries:

    @pytest.fixture
    def target_output(self):
        analyst, rotation, strategy, risk, returns = _get_pipeline_outputs()
        return run_target_return_pipeline(analyst, rotation, strategy, risk, returns)

    @pytest.mark.behavior
    def test_no_buy_sell_recommendations(self, target_output):
        """Summary should not contain advisory language."""
        text = target_output.summary.lower()
        for phrase in ["you should buy", "recommend selling", "must purchase"]:
            assert phrase not in text

    @pytest.mark.behavior
    def test_no_guarantees(self, target_output):
        """Summary and rationale should not make guarantee claims."""
        for text_source in [target_output.summary, target_output.construction_rationale]:
            text = text_source.lower()
            for phrase in ["we guarantee", "guaranteed returns", "results are guaranteed"]:
                assert phrase not in text

    @pytest.mark.behavior
    def test_disclaimer_present(self, target_output):
        """Risk disclosure must include a disclaimer."""
        disclaimer = target_output.risk_disclosure.disclaimer.lower()
        assert "not investment advice" in disclaimer or "past performance" in disclaimer

    @pytest.mark.behavior
    def test_bull_regime_realistic_achievability(self):
        """Bull regime → achievability should be REALISTIC."""
        analyst, rotation, strategy, risk, returns = _get_pipeline_outputs()
        output = run_target_return_pipeline(analyst, rotation, strategy, risk, returns)
        # Default pipeline uses bull regime from test data
        if output.market_regime == "bull":
            assert output.risk_disclosure.achievability_rating == "REALISTIC"

    @pytest.mark.behavior
    def test_probability_not_100_percent(self, target_output):
        """No probability should be exactly 1.0 (overconfident)."""
        pa = target_output.probability_assessment
        assert pa.prob_achieve_target < 1.0
        assert pa.prob_positive_return < 1.0

    @pytest.mark.behavior
    def test_ranges_not_points(self, target_output):
        """p10 should differ from p90 (ranges, not point estimates)."""
        pa = target_output.probability_assessment
        assert pa.p10_return_pct != pa.p90_return_pct

    @pytest.mark.behavior
    def test_alternatives_differ_from_primary(self, target_output):
        """Each alternative has a different target return than primary."""
        primary_target = target_output.target_return_pct
        for alt in target_output.alternatives:
            assert alt.target_return_pct != primary_target


# ---------------------------------------------------------------------------
# End-to-End Chain Test
# ---------------------------------------------------------------------------

class TestEndToEndChain:

    @pytest.mark.integration
    def test_full_chain_to_target_return(self):
        """Run full pipeline from research through target return constructor."""
        research = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
        analyst = run_analyst_pipeline(research)
        rotation = run_rotation_pipeline(analyst, research)
        strategy = run_strategist_pipeline(rotation, analyst)
        portfolio = run_portfolio_pipeline(strategy, analyst)
        risk = run_risk_pipeline(portfolio, strategy)
        returns = run_returns_projector_pipeline(portfolio, rotation, risk, analyst)
        target = run_target_return_pipeline(analyst, rotation, strategy, risk, returns)

        assert isinstance(target, TargetReturnOutput)
        assert MIN_POSITIONS <= len(target.positions) <= MAX_POSITIONS
        assert target.probability_assessment.prob_achieve_target <= MAX_PROB_ACHIEVE
        assert 2 <= len(target.alternatives) <= 3
        assert target.risk_disclosure.achievability_rating in ACHIEVABILITY_RATINGS


# ---------------------------------------------------------------------------
# Golden Dataset Test
# ---------------------------------------------------------------------------

class TestGoldenDataset:

    @pytest.mark.schema
    def test_golden_target_return_output(self):
        golden_path = Path(__file__).parent.parent.parent / "golden_datasets" / "target_return_golden.json"
        if not golden_path.exists():
            pytest.skip("Golden dataset not yet created")

        with open(golden_path) as f:
            golden = json.load(f)

        expected = golden["expected"]
        analyst, rotation, strategy, risk, returns = _get_pipeline_outputs()
        output = run_target_return_pipeline(analyst, rotation, strategy, risk, returns)

        # Position count in expected range
        n = len(output.positions)
        assert expected["position_count_range"][0] <= n <= expected["position_count_range"][1], (
            f"Position count {n} not in {expected['position_count_range']}"
        )

        # T1 allocation in expected range
        t1 = output.tier_allocation.t1_momentum_pct
        assert expected["t1_allocation_range"][0] <= t1 <= expected["t1_allocation_range"][1], (
            f"T1 allocation {t1:.1f}% not in {expected['t1_allocation_range']}"
        )

        # Probability in expected range
        prob = output.probability_assessment.prob_achieve_target
        assert expected["prob_achieve_target_range"][0] <= prob <= expected["prob_achieve_target_range"][1], (
            f"Prob {prob:.4f} not in {expected['prob_achieve_target_range']}"
        )

        # Achievability rating
        assert output.risk_disclosure.achievability_rating == expected["achievability_rating"]

        # Alternatives count in range
        n_alts = len(output.alternatives)
        assert expected["alternatives_count_range"][0] <= n_alts <= expected["alternatives_count_range"][1]

        # Cash reserve in range
        cash = output.cash_reserve_pct
        assert expected["cash_reserve_range"][0] <= cash <= expected["cash_reserve_range"][1]

        # Construction rationale min length
        assert len(output.construction_rationale) >= expected["construction_rationale_min_length"]

        # Scenario count
        assert expected["scenario_count"] == 3  # always 3

        # Sector weights present
        assert len(output.sector_weights) >= expected["sector_weights_min"]

        # Key assumptions/risks minimums
        assert len(output.probability_assessment.key_assumptions) >= expected["key_assumptions_min"]
        assert len(output.probability_assessment.key_risks) >= expected["key_risks_min"]
