"""
Agent 10: Executive Summary Synthesizer — Pipeline & Behavioral Tests
Level 2: Deterministic pipeline tests with mock data.
Level 3: Behavioral boundary tests.
"""

from __future__ import annotations

import re
from datetime import date

import pytest

from ibd_agents.agents.analyst_agent import run_analyst_pipeline
from ibd_agents.agents.educator_agent import run_educator_pipeline
from ibd_agents.agents.executive_synthesizer import (
    _build_action_items,
    _build_cross_agent_connections,
    _detect_contradictions,
    _extract_key_numbers,
    run_synthesizer_pipeline,
)
from ibd_agents.agents.portfolio_manager import run_portfolio_pipeline
from ibd_agents.agents.portfolio_reconciler import run_reconciler_pipeline
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
from ibd_agents.schemas.synthesis_output import SynthesisOutput

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
            confidence=confidence, reasoning="Test stock for synthesizer pipeline testing",
        ))

    return ResearchOutput(
        stocks=research_stocks, etfs=[], sector_patterns=[],
        data_sources_used=["test_data.xls"], data_sources_failed=[],
        total_securities_scanned=len(research_stocks),
        ibd_keep_candidates=[s.symbol for s in research_stocks if s.is_ibd_keep_candidate],
        multi_source_validated=[],
        analysis_date=date.today().isoformat(),
        summary="Test research output for synthesizer pipeline testing with sample data",
    )


def _build_full_pipeline():
    """Build all 9 agent outputs for testing."""
    research = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
    analyst = run_analyst_pipeline(research)
    rotation = run_rotation_pipeline(analyst, research)
    strategy = run_strategist_pipeline(rotation, analyst)
    portfolio = run_portfolio_pipeline(strategy, analyst)
    risk = run_risk_pipeline(portfolio, strategy)
    returns = run_returns_projector_pipeline(portfolio, rotation, risk, analyst)
    recon = run_reconciler_pipeline(portfolio, analyst)
    educator = run_educator_pipeline(
        portfolio, analyst, rotation, strategy, risk, recon
    )
    return (research, analyst, rotation, strategy, portfolio,
            risk, returns, recon, educator)


_cached = None


def _get_pipeline_outputs():
    global _cached
    if _cached is None:
        _cached = _build_full_pipeline()
    return _cached


_cached_synthesis = None


def _get_cached_synthesis():
    global _cached_synthesis
    if _cached_synthesis is None:
        (research, analyst, rotation, strategy, portfolio,
         risk, returns, recon, educator) = _get_pipeline_outputs()
        _cached_synthesis = run_synthesizer_pipeline(
            research, analyst, rotation, strategy, portfolio,
            risk, returns, recon, educator,
        )
    return _cached_synthesis


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

class TestSynthesizerPipeline:

    @pytest.mark.schema
    def test_pipeline_produces_valid_output(self):
        output = _get_cached_synthesis()
        assert isinstance(output, SynthesisOutput)

    @pytest.mark.schema
    def test_synthesis_source_is_template(self):
        """Without anthropic SDK configured, should use template fallback."""
        output = _get_cached_synthesis()
        assert output.synthesis_source == "template"

    @pytest.mark.schema
    def test_investment_thesis_200_chars(self):
        output = _get_cached_synthesis()
        assert len(output.investment_thesis) >= 200

    @pytest.mark.schema
    def test_portfolio_narrative_100_chars(self):
        output = _get_cached_synthesis()
        assert len(output.portfolio_narrative) >= 100

    @pytest.mark.schema
    def test_risk_reward_assessment_100_chars(self):
        output = _get_cached_synthesis()
        assert len(output.risk_reward_assessment) >= 100

    @pytest.mark.schema
    def test_market_context_100_chars(self):
        output = _get_cached_synthesis()
        assert len(output.market_context) >= 100

    @pytest.mark.schema
    def test_minimum_3_connections(self):
        output = _get_cached_synthesis()
        assert len(output.cross_agent_connections) >= 3

    @pytest.mark.schema
    def test_action_items_3_to_5(self):
        output = _get_cached_synthesis()
        assert 3 <= len(output.action_items) <= 5

    @pytest.mark.schema
    def test_analysis_date_format(self):
        output = _get_cached_synthesis()
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", output.analysis_date)

    @pytest.mark.schema
    def test_summary_50_chars(self):
        output = _get_cached_synthesis()
        assert len(output.summary) >= 50


# ---------------------------------------------------------------------------
# Key Numbers Extraction Tests
# ---------------------------------------------------------------------------

class TestKeyNumbersExtraction:

    @pytest.mark.schema
    def test_key_numbers_match_source_outputs(self):
        (research, analyst, rotation, strategy, portfolio,
         risk, returns, recon, _) = _get_pipeline_outputs()
        output = _get_cached_synthesis()
        kn = output.key_numbers

        # Research
        assert kn.total_stocks_scanned == research.total_securities_scanned
        assert kn.stocks_in_universe == len(research.stocks)

        # Analyst
        assert kn.stocks_rated == len(analyst.rated_stocks)
        assert kn.tier_1_count == analyst.tier_distribution.tier_1_count
        assert kn.tier_2_count == analyst.tier_distribution.tier_2_count
        assert kn.tier_3_count == analyst.tier_distribution.tier_3_count

        # Rotation
        assert kn.rotation_verdict == rotation.verdict.value
        assert kn.rotation_confidence == rotation.confidence
        assert kn.market_regime == rotation.market_regime.regime

        # Portfolio
        assert kn.portfolio_positions == portfolio.total_positions
        assert kn.stock_count == portfolio.stock_count
        assert kn.etf_count == portfolio.etf_count
        assert kn.cash_pct == portfolio.cash_pct

        # Risk
        assert kn.risk_status == risk.overall_status
        assert kn.sleep_well_score == risk.sleep_well_scores.overall_score

        # Reconciler
        assert kn.turnover_pct == recon.transformation_metrics.turnover_pct
        assert kn.actions_count == len(recon.actions)


# ---------------------------------------------------------------------------
# Cross-Agent Connections Tests
# ---------------------------------------------------------------------------

class TestCrossAgentConnections:

    @pytest.mark.schema
    def test_minimum_5_connections(self):
        output = _get_cached_synthesis()
        assert len(output.cross_agent_connections) >= 5

    @pytest.mark.schema
    def test_connections_reference_multiple_agents(self):
        output = _get_cached_synthesis()
        agents_mentioned = set()
        for conn in output.cross_agent_connections:
            agents_mentioned.add(conn.from_agent)
            agents_mentioned.add(conn.to_agent)
        assert len(agents_mentioned) >= 4

    @pytest.mark.schema
    def test_connections_have_substantive_content(self):
        output = _get_cached_synthesis()
        for conn in output.cross_agent_connections:
            assert len(conn.connection) >= 10
            assert len(conn.implication) >= 10


# ---------------------------------------------------------------------------
# Contradiction Detection Tests
# ---------------------------------------------------------------------------

class TestContradictionDetection:

    @pytest.mark.schema
    def test_contradictions_are_valid_objects(self):
        output = _get_cached_synthesis()
        for c in output.contradictions:
            assert len(c.agent_a) >= 1
            assert len(c.agent_b) >= 1
            assert len(c.finding_a) >= 10
            assert len(c.finding_b) >= 10
            assert len(c.resolution) >= 10

    @pytest.mark.schema
    def test_contradiction_agents_differ(self):
        output = _get_cached_synthesis()
        for c in output.contradictions:
            assert c.agent_a != c.agent_b


# ---------------------------------------------------------------------------
# Action Items Tests
# ---------------------------------------------------------------------------

class TestActionItems:

    @pytest.mark.schema
    def test_action_items_are_substantive(self):
        output = _get_cached_synthesis()
        for item in output.action_items:
            assert len(item) >= 10

    @pytest.mark.schema
    def test_action_items_reference_numbers(self):
        """Action items should include concrete numbers (counts, percentages)."""
        output = _get_cached_synthesis()
        has_number = any(
            any(c.isdigit() for c in item)
            for item in output.action_items
        )
        assert has_number, "At least one action item should contain a number"


# ---------------------------------------------------------------------------
# Behavioral Boundary Tests
# ---------------------------------------------------------------------------

class TestBehavioralBoundaries:

    @pytest.fixture
    def synthesis_output(self):
        return _get_cached_synthesis()

    @pytest.mark.behavior
    def test_no_buy_sell_recommendations(self, synthesis_output):
        """Synthesis should never recommend specific trades."""
        forbidden_patterns = [
            r"\bbuy\s+(now|immediately|today)\b",
            r"\bsell\s+(now|immediately|today)\b",
            r"\bstrong\s+buy\b",
            r"\bmust\s+buy\b",
        ]
        text = (
            synthesis_output.investment_thesis
            + synthesis_output.portfolio_narrative
            + synthesis_output.risk_reward_assessment
            + synthesis_output.market_context
        ).lower()
        for pattern in forbidden_patterns:
            assert not re.search(pattern, text), \
                f"Found forbidden pattern '{pattern}' in synthesis text"

    @pytest.mark.behavior
    def test_no_specific_dollar_amounts_in_thesis(self, synthesis_output):
        """Investment thesis should not mention specific dollar amounts."""
        thesis = synthesis_output.investment_thesis
        dollar_pattern = r"\$[\d,]+\.?\d*"
        matches = re.findall(dollar_pattern, thesis)
        assert len(matches) == 0, \
            f"Investment thesis contains dollar amounts: {matches}"

    @pytest.mark.behavior
    def test_no_position_sizing_in_thesis(self, synthesis_output):
        """Investment thesis should not contain position sizing guidance."""
        forbidden = ["% of portfolio", "position size", "allocate exactly"]
        thesis_lower = synthesis_output.investment_thesis.lower()
        for phrase in forbidden:
            assert phrase not in thesis_lower, \
                f"Thesis contains position sizing: '{phrase}'"
