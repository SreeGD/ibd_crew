"""Agent 07: Returns Projector — Pipeline & Behavioral Tests
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
from ibd_agents.schemas.research_output import (
    ResearchOutput,
    ResearchStock,
    compute_preliminary_tier,
    is_ibd_keep_candidate,
)
from ibd_agents.schemas.returns_projection_output import (
    BENCHMARK_RETURNS,
    BENCHMARKS,
    MAX_PORTFOLIO_LOSS_WITH_STOPS,
    RISK_FREE_RATE,
    SCENARIO_WEIGHTS,
    SCENARIOS,
    STOP_LOSS_BY_TIER,
    TIER_HISTORICAL_RETURNS,
    ReturnsProjectionOutput,
)
from ibd_agents.tools.alpha_calculator import (
    compute_alpha,
    compute_expected_alpha,
    decompose_alpha,
    estimate_outperform_probability,
)
from ibd_agents.tools.benchmark_fetcher import (
    compute_benchmark_expected,
    get_benchmark_stats,
)
from ibd_agents.tools.confidence_builder import (
    build_confidence_intervals,
    build_percentile_range,
)
from ibd_agents.tools.drawdown_estimator import (
    compute_drawdown_with_stops,
    compute_drawdown_without_stops,
    estimate_stop_triggers,
)
from ibd_agents.tools.scenario_weighter import (
    compute_expected,
    get_scenario_weights,
)
from ibd_agents.tools.tier_return_calculator import (
    compute_blended_return,
    compute_tier_contribution,
    compute_tier_return,
    scale_to_horizon,
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
            confidence=confidence, reasoning="Test stock for returns projector pipeline testing",
        ))

    return ResearchOutput(
        stocks=research_stocks, etfs=[], sector_patterns=[],
        data_sources_used=["test_data.xls"], data_sources_failed=[],
        total_securities_scanned=len(research_stocks),
        ibd_keep_candidates=[s.symbol for s in research_stocks if s.is_ibd_keep_candidate],
        multi_source_validated=[],
        analysis_date=date.today().isoformat(),
        summary="Test research output for returns projector pipeline testing with sample data",
    )


def _build_full_pipeline():
    research = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
    analyst = run_analyst_pipeline(research)
    rotation = run_rotation_pipeline(analyst, research)
    strategy = run_strategist_pipeline(rotation, analyst)
    portfolio = run_portfolio_pipeline(strategy, analyst)
    risk = run_risk_pipeline(portfolio, strategy)
    return portfolio, rotation, risk, analyst


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
    def test_tier_return_bull_gt_base_gt_bear(self):
        """For each tier, bull return > base return > bear return."""
        for tier in (1, 2, 3):
            bull = compute_tier_return(tier, "bull")
            base = compute_tier_return(tier, "base")
            bear = compute_tier_return(tier, "bear")
            assert bull > base > bear, (
                f"Tier {tier}: bull={bull}, base={base}, bear={bear}"
            )

    @pytest.mark.schema
    def test_tier_return_t1_highest_in_bull(self):
        """T1 bull return > T2 bull return > T3 bull return."""
        t1 = compute_tier_return(1, "bull")
        t2 = compute_tier_return(2, "bull")
        t3 = compute_tier_return(3, "bull")
        assert t1 > t2 > t3, f"T1={t1}, T2={t2}, T3={t3}"

    @pytest.mark.schema
    def test_blended_return_matches_manual(self):
        """compute_blended_return matches manual weighted calculation."""
        tier_pcts = {1: 39.0, 2: 37.0, 3: 22.0}
        tier_returns = {1: 18.0, 2: 14.0, 3: 10.0}
        cash_pct = 2.0

        result = compute_blended_return(tier_pcts, tier_returns, cash_pct)

        manual = (
            0.39 * 18.0
            + 0.37 * 14.0
            + 0.22 * 10.0
            + 0.02 * RISK_FREE_RATE
        )
        assert abs(result - manual) < 0.01, f"result={result}, manual={manual}"

    @pytest.mark.schema
    def test_scale_to_3month(self):
        """3-month return = annual * 0.25."""
        annual_return = 20.0
        annual_std = 14.0
        ret_3m, std_3m = scale_to_horizon(annual_return, annual_std, "3_month")
        assert abs(ret_3m - annual_return * 0.25) < 0.01

    @pytest.mark.schema
    def test_benchmark_stats_spy_base(self):
        """SPY base scenario: mean=10.5, std=14.0."""
        stats = get_benchmark_stats("SPY", "base")
        assert stats["mean"] == 10.5
        assert stats["std"] == 14.0

    @pytest.mark.schema
    def test_scenario_weights_sum_to_1(self):
        """For each regime, scenario weights sum to 1.0."""
        for regime in ("bull", "neutral", "bear"):
            weights = get_scenario_weights(regime)
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.001, (
                f"Regime {regime}: weights sum to {total}"
            )

    @pytest.mark.schema
    def test_alpha_is_difference(self):
        """compute_alpha(15, 10.5) == 4.5."""
        assert compute_alpha(15.0, 10.5) == 4.5

    @pytest.mark.schema
    def test_drawdown_with_stops_le_without(self):
        """Drawdown with stops < drawdown without stops."""
        tier_pcts = {1: 39.0, 2: 37.0, 3: 22.0}
        dd_with = compute_drawdown_with_stops(tier_pcts)
        dd_without = compute_drawdown_without_stops(tier_pcts)
        assert dd_with < dd_without, (
            f"with_stops={dd_with}, without_stops={dd_without}"
        )

    @pytest.mark.schema
    def test_drawdown_with_stops_expected_value(self):
        """For pcts {1:39, 2:37, 3:22}, drawdown_with_stops ~ 17.88."""
        tier_pcts = {1: 39.0, 2: 37.0, 3: 22.0}
        dd_with = compute_drawdown_with_stops(tier_pcts)
        assert abs(dd_with - MAX_PORTFOLIO_LOSS_WITH_STOPS) < 0.01, (
            f"dd_with={dd_with}, expected={MAX_PORTFOLIO_LOSS_WITH_STOPS}"
        )

    @pytest.mark.schema
    def test_percentile_ordering(self):
        """build_percentile_range results: p10 < p25 < p50 < p75 < p90."""
        result = build_percentile_range(mean=12.0, std=10.0, portfolio_value=1_500_000)
        assert result["p10"] < result["p25"] < result["p50"] < result["p75"] < result["p90"]

    @pytest.mark.schema
    def test_outperform_probability_range(self):
        """estimate_outperform_probability result in [0.01, 0.99]."""
        prob = estimate_outperform_probability(
            p_mean=15.0, p_std=12.0, b_mean=10.5, b_std=14.0
        )
        assert 0.01 <= prob <= 0.99


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

class TestReturnsProjectorPipeline:

    @pytest.fixture
    def pipeline_output(self):
        portfolio, rotation, risk, analyst = _get_pipeline_outputs()
        return run_returns_projector_pipeline(portfolio, rotation, risk, analyst)

    @pytest.mark.schema
    def test_pipeline_produces_valid_output(self, pipeline_output):
        assert isinstance(pipeline_output, ReturnsProjectionOutput)

    @pytest.mark.schema
    def test_3_scenarios(self, pipeline_output):
        assert len(pipeline_output.scenarios) == 3

    @pytest.mark.schema
    def test_3_benchmarks(self, pipeline_output):
        assert len(pipeline_output.benchmark_comparisons) == 3

    @pytest.mark.schema
    def test_scenario_probabilities_sum_to_1(self, pipeline_output):
        total = sum(s.probability for s in pipeline_output.scenarios)
        assert abs(total - 1.0) < 0.01, f"Scenario probabilities sum to {total}"

    @pytest.mark.schema
    def test_bull_gt_base_gt_bear(self, pipeline_output):
        """Bull 12m return > base > bear."""
        by_scenario = {s.scenario: s for s in pipeline_output.scenarios}
        assert by_scenario["bull"].portfolio_return_12m > by_scenario["base"].portfolio_return_12m
        assert by_scenario["base"].portfolio_return_12m > by_scenario["bear"].portfolio_return_12m

    @pytest.mark.schema
    def test_t1_highest_contribution_bull(self, pipeline_output):
        """T1 contribution > T2 > T3 in bull scenario."""
        bull = next(s for s in pipeline_output.scenarios if s.scenario == "bull")
        assert bull.tier_1_contribution > bull.tier_2_contribution > bull.tier_3_contribution

    @pytest.mark.schema
    def test_expected_return_between_scenarios(self, pipeline_output):
        """bear 12m < expected_12m < bull 12m."""
        by_scenario = {s.scenario: s for s in pipeline_output.scenarios}
        expected = pipeline_output.expected_return.expected_12m
        assert by_scenario["bear"].portfolio_return_12m < expected
        assert expected < by_scenario["bull"].portfolio_return_12m

    @pytest.mark.schema
    def test_alpha_vs_spy_positive_base(self, pipeline_output):
        """Alpha vs SPY in base scenario should be > 0."""
        spy_comparison = next(
            b for b in pipeline_output.benchmark_comparisons if b.symbol == "SPY"
        )
        assert spy_comparison.alpha_base_12m > 0

    @pytest.mark.schema
    def test_drawdown_ordering(self, pipeline_output):
        """max_drawdown_with_stops < max_drawdown_without_stops."""
        rm = pipeline_output.risk_metrics
        assert rm.max_drawdown_with_stops < rm.max_drawdown_without_stops

    @pytest.mark.schema
    def test_sharpe_positive_bull(self, pipeline_output):
        """Portfolio Sharpe ratio in bull scenario should be > 0."""
        assert pipeline_output.risk_metrics.portfolio_sharpe_bull > 0

    @pytest.mark.schema
    def test_confidence_intervals_ordered(self, pipeline_output):
        """p10 < p25 < p50 < p75 < p90 for 12m horizon."""
        h12 = pipeline_output.confidence_intervals.horizon_12m
        assert h12.p10 < h12.p25 < h12.p50 < h12.p75 < h12.p90

    @pytest.mark.schema
    def test_analysis_date_is_today(self, pipeline_output):
        assert pipeline_output.analysis_date == date.today().isoformat()

    @pytest.mark.schema
    def test_summary_min_length(self, pipeline_output):
        assert len(pipeline_output.summary) >= 50

    @pytest.mark.schema
    def test_caveats_non_empty(self, pipeline_output):
        assert len(pipeline_output.caveats) >= 1

    @pytest.mark.schema
    def test_3_tier_mixes(self, pipeline_output):
        assert len(pipeline_output.tier_mix_comparison) == 3


# ---------------------------------------------------------------------------
# Behavioral Boundary Tests
# ---------------------------------------------------------------------------

class TestBehavioralBoundaries:

    @pytest.fixture
    def projector_output(self):
        portfolio, rotation, risk, analyst = _get_pipeline_outputs()
        return run_returns_projector_pipeline(portfolio, rotation, risk, analyst)

    @pytest.mark.behavior
    def test_no_buy_sell_recommendations(self, projector_output):
        """Summary should not contain advisory language."""
        text = projector_output.summary.lower()
        for phrase in ["you should buy", "recommend selling", "must purchase"]:
            assert phrase not in text

    @pytest.mark.behavior
    def test_no_guarantees(self, projector_output):
        """Summary should not make affirmative guarantee claims."""
        text = projector_output.summary.lower()
        for phrase in ["we guarantee", "guaranteed returns", "results are guaranteed"]:
            assert phrase not in text

    @pytest.mark.behavior
    def test_no_specific_dollar_amounts_in_summary(self, projector_output):
        """Summary should not expose specific dollar amounts like $X,XXX."""
        assert not re.search(r"\$\d{1,3}(,\d{3})+", projector_output.summary)

    @pytest.mark.behavior
    def test_caveats_contain_disclaimers(self, projector_output):
        """At least one caveat mentions 'past performance' or 'not guarantee'."""
        caveats_text = " ".join(projector_output.caveats).lower()
        has_past_perf = "past performance" in caveats_text
        has_not_guarantee = "not guarantee" in caveats_text
        assert has_past_perf or has_not_guarantee, (
            "Caveats must mention 'past performance' or 'not guarantee'"
        )

    @pytest.mark.behavior
    def test_ranges_not_points(self, projector_output):
        """Confidence intervals have p10 != p90 (ranges, not point estimates)."""
        h12 = projector_output.confidence_intervals.horizon_12m
        assert h12.p10 != h12.p90, "p10 and p90 should differ (range, not point)"


# ---------------------------------------------------------------------------
# End-to-End Chain Test
# ---------------------------------------------------------------------------

class TestEndToEndChain:

    @pytest.mark.integration
    def test_full_chain_to_returns_projector(self):
        """Run full pipeline from research through returns projector."""
        research = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
        analyst = run_analyst_pipeline(research)
        rotation = run_rotation_pipeline(analyst, research)
        strategy = run_strategist_pipeline(rotation, analyst)
        portfolio = run_portfolio_pipeline(strategy, analyst)
        risk = run_risk_pipeline(portfolio, strategy)
        projection = run_returns_projector_pipeline(portfolio, rotation, risk, analyst)

        assert isinstance(projection, ReturnsProjectionOutput)
        assert len(projection.scenarios) == 3
        assert len(projection.benchmark_comparisons) == 3
        assert len(projection.tier_mix_comparison) == 3
        assert projection.risk_metrics.max_drawdown_with_stops < projection.risk_metrics.max_drawdown_without_stops


# ---------------------------------------------------------------------------
# Golden Dataset Test
# ---------------------------------------------------------------------------

class TestGoldenDataset:

    @pytest.mark.schema
    def test_golden_returns_projector_output(self):
        golden_path = Path(__file__).parent.parent.parent / "golden_datasets" / "returns_projection_golden.json"
        if not golden_path.exists():
            pytest.skip("Golden dataset not yet created")

        with open(golden_path) as f:
            golden = json.load(f)

        portfolio, rotation, risk, analyst = _get_pipeline_outputs()
        output = run_returns_projector_pipeline(portfolio, rotation, risk, analyst)

        assert len(output.scenarios) == golden["scenario_count"]
        assert len(output.benchmark_comparisons) == golden["benchmark_count"]
        assert len(output.caveats) >= golden["caveats_min"]
        assert output.risk_metrics.max_drawdown_with_stops <= golden["drawdown_with_stops_max"]
