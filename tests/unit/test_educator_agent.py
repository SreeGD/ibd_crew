"""
Agent 09: Educator — Pipeline & Behavioral Tests
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
from ibd_agents.agents.educator_agent import run_educator_pipeline
from ibd_agents.agents.portfolio_manager import run_portfolio_pipeline
from ibd_agents.agents.portfolio_reconciler import run_reconciler_pipeline
from ibd_agents.agents.risk_officer import run_risk_pipeline
from ibd_agents.agents.rotation_detector import run_rotation_pipeline
from ibd_agents.agents.sector_strategist import run_strategist_pipeline
from ibd_agents.schemas.educator_output import (
    MIN_ACTION_ITEMS,
    MIN_CONCEPT_LESSONS,
    MIN_GLOSSARY_ENTRIES,
    MIN_STOCK_EXPLANATIONS,
    REQUIRED_GLOSSARY_TERMS,
    EducatorOutput,
)
from ibd_agents.schemas.portfolio_output import ALL_KEEPS, KEEP_METADATA
from ibd_agents.schemas.research_output import (
    ResearchOutput,
    ResearchStock,
    compute_preliminary_tier,
    is_ibd_keep_candidate,
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
            confidence=confidence, reasoning="Test stock for educator pipeline testing",
        ))

    return ResearchOutput(
        stocks=research_stocks, etfs=[], sector_patterns=[],
        data_sources_used=["test_data.xls"], data_sources_failed=[],
        total_securities_scanned=len(research_stocks),
        ibd_keep_candidates=[s.symbol for s in research_stocks if s.is_ibd_keep_candidate],
        multi_source_validated=[],
        analysis_date=date.today().isoformat(),
        summary="Test research output for educator pipeline testing with sample data",
    )


def _build_full_pipeline():
    research = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
    analyst = run_analyst_pipeline(research)
    rotation = run_rotation_pipeline(analyst, research)
    strategy = run_strategist_pipeline(rotation, analyst)
    portfolio = run_portfolio_pipeline(strategy, analyst)
    risk = run_risk_pipeline(portfolio, strategy)
    recon = run_reconciler_pipeline(portfolio, analyst)
    return portfolio, analyst, rotation, strategy, risk, recon, research


_cached = None


def _get_pipeline_outputs():
    global _cached
    if _cached is None:
        _cached = _build_full_pipeline()
    return _cached


def _get_educator_output():
    portfolio, analyst, rotation, strategy, risk, recon, _ = _get_pipeline_outputs()
    return run_educator_pipeline(portfolio, analyst, rotation, strategy, risk, recon)


_cached_educator = None


def _get_cached_educator():
    global _cached_educator
    if _cached_educator is None:
        _cached_educator = _get_educator_output()
    return _cached_educator


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

class TestEducatorPipeline:

    @pytest.mark.schema
    def test_pipeline_produces_valid_output(self):
        output = _get_cached_educator()
        assert isinstance(output, EducatorOutput)

    @pytest.mark.schema
    def test_executive_summary_200_chars(self):
        output = _get_cached_educator()
        assert len(output.executive_summary) >= 200

    @pytest.mark.schema
    def test_minimum_stock_explanations(self):
        output = _get_cached_educator()
        assert len(output.stock_explanations) >= MIN_STOCK_EXPLANATIONS

    @pytest.mark.schema
    def test_minimum_concept_lessons(self):
        output = _get_cached_educator()
        assert len(output.concept_lessons) >= MIN_CONCEPT_LESSONS

    @pytest.mark.schema
    def test_minimum_glossary_entries(self):
        output = _get_cached_educator()
        assert len(output.glossary) >= MIN_GLOSSARY_ENTRIES

    @pytest.mark.schema
    def test_minimum_action_items(self):
        output = _get_cached_educator()
        assert len(output.action_items) >= MIN_ACTION_ITEMS

    @pytest.mark.schema
    def test_rotation_explanation_100_chars(self):
        output = _get_cached_educator()
        assert len(output.rotation_explanation) >= 100

    @pytest.mark.schema
    def test_risk_explainer_100_chars(self):
        output = _get_cached_educator()
        assert len(output.risk_explainer) >= 100

    @pytest.mark.schema
    def test_tier_guide_complete(self):
        output = _get_cached_educator()
        guide = output.tier_guide
        assert len(guide.overview) >= 50
        assert len(guide.tier_1_description) >= 30
        assert len(guide.tier_2_description) >= 30
        assert len(guide.tier_3_description) >= 30

    @pytest.mark.schema
    def test_keep_explanations_complete(self):
        output = _get_cached_educator()
        keeps = output.keep_explanations
        assert keeps.total_keeps == 14
        assert len(keeps.fundamental_keeps) >= 30
        assert len(keeps.ibd_keeps) >= 30

    @pytest.mark.schema
    def test_stock_explanations_have_tiers(self):
        output = _get_cached_educator()
        for stock in output.stock_explanations:
            assert stock.tier in (1, 2, 3), f"Stock {stock.symbol} has invalid tier {stock.tier}"

    @pytest.mark.schema
    def test_concept_lessons_have_examples(self):
        output = _get_cached_educator()
        for lesson in output.concept_lessons:
            assert len(lesson.example_from_analysis) >= 20, (
                f"Concept '{lesson.concept}' lacks example"
            )

    @pytest.mark.schema
    def test_transition_guide_present(self):
        output = _get_cached_educator()
        guide = output.transition_guide
        assert len(guide.overview) >= 100
        assert len(guide.money_flow_explained) >= 100
        assert len(guide.what_to_sell) >= 30
        assert len(guide.what_to_buy) >= 30

    @pytest.mark.schema
    def test_analysis_date_is_today(self):
        output = _get_cached_educator()
        assert output.analysis_date == date.today().isoformat()

    @pytest.mark.schema
    def test_summary_min_length(self):
        output = _get_cached_educator()
        assert len(output.summary) >= 50

    @pytest.mark.schema
    def test_stock_symbols_match_portfolio(self):
        """Explained stocks should come from the portfolio."""
        portfolio, analyst, rotation, strategy, risk, recon, _ = _get_pipeline_outputs()
        output = run_educator_pipeline(portfolio, analyst, rotation, strategy, risk, recon)
        portfolio_syms = set()
        for tier in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]:
            for pos in tier.positions:
                portfolio_syms.add(pos.symbol)
        for stock_exp in output.stock_explanations:
            assert stock_exp.symbol in portfolio_syms, (
                f"Explained stock {stock_exp.symbol} not in portfolio"
            )


# ---------------------------------------------------------------------------
# Behavioral Boundary Tests
# ---------------------------------------------------------------------------

class TestBehavioralBoundaries:

    @pytest.fixture
    def educator_output(self):
        return _get_cached_educator()

    @pytest.mark.behavior
    def test_no_investment_decisions(self, educator_output):
        """Educator should not make investment decisions."""
        text = educator_output.model_dump_json().lower()
        for phrase in ["you should buy", "you must sell", "recommend purchasing", "i advise"]:
            assert phrase not in text, f"Found advisory language: '{phrase}'"

    @pytest.mark.behavior
    def test_no_modified_analysis(self, educator_output):
        """Educator should not modify other agents' analysis."""
        text = educator_output.model_dump_json().lower()
        for phrase in ["i would change", "override the", "adjusting the tier"]:
            assert phrase not in text

    @pytest.mark.behavior
    def test_no_personal_advice(self, educator_output):
        """Educator should not give personalized advice."""
        text = educator_output.summary.lower()
        assert "based on your situation" not in text
        assert "in your case" not in text

    @pytest.mark.behavior
    def test_no_return_guarantees(self, educator_output):
        text = educator_output.model_dump_json().lower()
        assert "guaranteed" not in text
        assert "will return" not in text

    @pytest.mark.behavior
    def test_keep_categories_valid(self, educator_output):
        """Every keep_category is fundamental/ibd/None."""
        for stock in educator_output.stock_explanations:
            if stock.keep_category is not None:
                assert stock.keep_category in ("fundamental", "ibd"), (
                    f"Stock {stock.symbol} has invalid keep_category: {stock.keep_category}"
                )

    @pytest.mark.behavior
    def test_keeps_have_correct_categories(self, educator_output):
        """Stocks in KEEP_METADATA should have the right category."""
        for stock in educator_output.stock_explanations:
            if stock.symbol in KEEP_METADATA:
                expected = KEEP_METADATA[stock.symbol]["category"]
                assert stock.keep_category == expected, (
                    f"{stock.symbol}: expected category '{expected}', got '{stock.keep_category}'"
                )


# ---------------------------------------------------------------------------
# End-to-End Chain Test
# ---------------------------------------------------------------------------

class TestEndToEndChain:

    @pytest.mark.integration
    def test_full_chain_to_educator(self):
        research = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
        analyst = run_analyst_pipeline(research)
        rotation = run_rotation_pipeline(analyst, research)
        strategy = run_strategist_pipeline(rotation, analyst)
        portfolio = run_portfolio_pipeline(strategy, analyst)
        risk = run_risk_pipeline(portfolio, strategy)
        recon = run_reconciler_pipeline(portfolio, analyst)
        educator = run_educator_pipeline(portfolio, analyst, rotation, strategy, risk, recon)

        assert isinstance(educator, EducatorOutput)
        assert len(educator.stock_explanations) >= 15
        assert len(educator.concept_lessons) >= 5
        assert len(educator.glossary) >= 15
        assert educator.keep_explanations.total_keeps == 14


# ---------------------------------------------------------------------------
# Golden Dataset Test
# ---------------------------------------------------------------------------

class TestGoldenDataset:

    @pytest.mark.schema
    def test_golden_educator_output(self):
        golden_path = Path(__file__).parent.parent.parent / "golden_datasets" / "educator_golden.json"
        if not golden_path.exists():
            pytest.skip("Golden dataset not yet created")

        with open(golden_path) as f:
            golden = json.load(f)

        output = _get_cached_educator()

        assert len(output.stock_explanations) >= golden["min_stock_explanations"]
        assert len(output.concept_lessons) >= golden["min_concept_lessons"]
        assert len(output.glossary) >= golden["min_glossary_entries"]
        assert len(output.action_items) >= golden["min_action_items"]
        assert output.keep_explanations.total_keeps == golden["keeps_total"]
