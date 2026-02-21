"""
Agent 06: Risk Officer — Pipeline & Behavioral Tests
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
from ibd_agents.agents.risk_officer import run_risk_pipeline
from ibd_agents.agents.rotation_detector import run_rotation_pipeline
from ibd_agents.agents.sector_strategist import run_strategist_pipeline
from ibd_agents.schemas.portfolio_output import ALL_KEEPS, PortfolioOutput
from ibd_agents.schemas.research_output import (
    ResearchOutput,
    ResearchStock,
    compute_preliminary_tier,
    is_ibd_keep_candidate,
)
from ibd_agents.schemas.risk_output import RiskAssessment
from ibd_agents.tools.risk_analyzer import (
    check_keeps,
    check_position_sizing,
    check_sector_concentration,
    compute_sleep_well_scores,
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
            confidence=confidence, reasoning="Test stock for risk pipeline testing",
        ))

    return ResearchOutput(
        stocks=research_stocks, etfs=[], sector_patterns=[],
        data_sources_used=["test_data.xls"], data_sources_failed=[],
        total_securities_scanned=len(research_stocks),
        ibd_keep_candidates=[s.symbol for s in research_stocks if s.is_ibd_keep_candidate],
        multi_source_validated=[],
        analysis_date=date.today().isoformat(),
        summary="Test research output for risk officer pipeline testing with sample data",
    )


def _build_full_pipeline():
    research = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
    analyst = run_analyst_pipeline(research)
    rotation = run_rotation_pipeline(analyst, research)
    strategy = run_strategist_pipeline(rotation, analyst)
    portfolio = run_portfolio_pipeline(strategy, analyst)
    return portfolio, strategy, analyst, rotation, research


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
    def test_check_position_sizing_pass(self):
        portfolio, _, _, _, _ = _get_pipeline_outputs()
        result = check_position_sizing(portfolio)
        assert result.status == "PASS"

    @pytest.mark.schema
    def test_check_sector_concentration_pass(self):
        portfolio, _, _, _, _ = _get_pipeline_outputs()
        result = check_sector_concentration(portfolio)
        assert result.status == "PASS"

    @pytest.mark.schema
    def test_check_keeps_pass(self):
        portfolio, _, _, _, _ = _get_pipeline_outputs()
        result = check_keeps(portfolio)
        assert result.status == "PASS"

    @pytest.mark.schema
    def test_sleep_well_scores_in_range(self):
        portfolio, _, _, _, _ = _get_pipeline_outputs()
        checks = [check_position_sizing(portfolio)]
        scores = compute_sleep_well_scores(checks, portfolio)
        assert 1 <= scores.overall_score <= 10
        assert 1 <= scores.tier_1_score <= 10


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

class TestRiskPipeline:

    @pytest.fixture
    def pipeline_outputs(self):
        return _get_pipeline_outputs()

    @pytest.mark.schema
    def test_pipeline_produces_valid_output(self, pipeline_outputs):
        portfolio, strategy, _, _, _ = pipeline_outputs
        output = run_risk_pipeline(portfolio, strategy)
        assert isinstance(output, RiskAssessment)

    @pytest.mark.schema
    def test_11_checks_present(self, pipeline_outputs):
        portfolio, strategy, _, _, _ = pipeline_outputs
        output = run_risk_pipeline(portfolio, strategy)
        assert len(output.check_results) == 11

    @pytest.mark.schema
    def test_all_check_names_present(self, pipeline_outputs):
        portfolio, strategy, _, _, _ = pipeline_outputs
        output = run_risk_pipeline(portfolio, strategy)
        names = {c.check_name for c in output.check_results}
        from ibd_agents.schemas.risk_output import CHECK_NAMES
        assert names == set(CHECK_NAMES)

    @pytest.mark.schema
    def test_sleep_well_1_to_10(self, pipeline_outputs):
        portfolio, strategy, _, _, _ = pipeline_outputs
        output = run_risk_pipeline(portfolio, strategy)
        for score in [
            output.sleep_well_scores.tier_1_score,
            output.sleep_well_scores.tier_2_score,
            output.sleep_well_scores.tier_3_score,
            output.sleep_well_scores.overall_score,
        ]:
            assert 1 <= score <= 10

    @pytest.mark.schema
    def test_stress_tests_3_scenarios(self, pipeline_outputs):
        portfolio, strategy, _, _, _ = pipeline_outputs
        output = run_risk_pipeline(portfolio, strategy)
        assert len(output.stress_test_results.scenarios) >= 3

    @pytest.mark.schema
    def test_keeps_validated(self, pipeline_outputs):
        portfolio, strategy, _, _, _ = pipeline_outputs
        output = run_risk_pipeline(portfolio, strategy)
        assert output.keep_validation.total_keeps_found == 14

    @pytest.mark.schema
    def test_overall_status_valid(self, pipeline_outputs):
        portfolio, strategy, _, _, _ = pipeline_outputs
        output = run_risk_pipeline(portfolio, strategy)
        assert output.overall_status in ("APPROVED", "CONDITIONAL", "REJECTED")

    @pytest.mark.schema
    def test_analysis_date_is_today(self, pipeline_outputs):
        portfolio, strategy, _, _, _ = pipeline_outputs
        output = run_risk_pipeline(portfolio, strategy)
        assert output.analysis_date == date.today().isoformat()

    @pytest.mark.schema
    def test_summary_min_length(self, pipeline_outputs):
        portfolio, strategy, _, _, _ = pipeline_outputs
        output = run_risk_pipeline(portfolio, strategy)
        assert len(output.summary) >= 50

    @pytest.mark.schema
    def test_status_consistency(self, pipeline_outputs):
        """Status matches vetoes/warnings."""
        portfolio, strategy, _, _, _ = pipeline_outputs
        output = run_risk_pipeline(portfolio, strategy)
        if output.vetoes:
            assert output.overall_status == "REJECTED"
        elif output.warnings:
            assert output.overall_status == "CONDITIONAL"
        else:
            assert output.overall_status == "APPROVED"


# ---------------------------------------------------------------------------
# Behavioral Boundary Tests
# ---------------------------------------------------------------------------

class TestBehavioralBoundaries:

    @pytest.fixture
    def risk_output(self):
        portfolio, strategy, _, _, _ = _get_pipeline_outputs()
        return run_risk_pipeline(portfolio, strategy)

    @pytest.mark.behavior
    def test_no_stock_picks(self, risk_output):
        text = risk_output.model_dump_json().lower()
        for phrase in ["buy nvda", "sell aapl", "purchase tsla"]:
            assert phrase not in text

    @pytest.mark.behavior
    def test_no_position_sizes(self, risk_output):
        text = risk_output.model_dump_json()
        assert not re.search(r"\$\d+[,.]?\d*\s*(shares|position|worth)", text.lower())

    @pytest.mark.behavior
    def test_no_return_projections(self, risk_output):
        text = risk_output.summary.lower()
        assert "will return" not in text
        assert "guaranteed" not in text

    @pytest.mark.behavior
    def test_no_pm_override(self, risk_output):
        text = risk_output.model_dump_json().lower()
        assert "you must buy" not in text
        assert "you must sell" not in text


# ---------------------------------------------------------------------------
# End-to-End Chain Test
# ---------------------------------------------------------------------------

class TestEndToEndChain:

    @pytest.mark.integration
    def test_full_chain_to_risk(self):
        research = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
        analyst = run_analyst_pipeline(research)
        rotation = run_rotation_pipeline(analyst, research)
        strategy = run_strategist_pipeline(rotation, analyst)
        portfolio = run_portfolio_pipeline(strategy, analyst)
        risk = run_risk_pipeline(portfolio, strategy)

        assert isinstance(risk, RiskAssessment)
        assert len(risk.check_results) == 11
        assert risk.sleep_well_scores.overall_score >= 1


# ---------------------------------------------------------------------------
# Golden Dataset Test
# ---------------------------------------------------------------------------

class TestGoldenDataset:

    @pytest.mark.schema
    def test_golden_risk_output(self):
        golden_path = Path(__file__).parent.parent.parent / "golden_datasets" / "risk_golden.json"
        if not golden_path.exists():
            pytest.skip("Golden dataset not yet created")

        with open(golden_path) as f:
            golden = json.load(f)

        portfolio, strategy, _, _, _ = _get_pipeline_outputs()
        output = run_risk_pipeline(portfolio, strategy)

        assert len(output.check_results) == golden["check_count"]
        assert len(output.stress_test_results.scenarios) >= golden["stress_scenario_count"]
        assert golden["sleep_well_min"] <= output.sleep_well_scores.overall_score <= golden["sleep_well_max"]


# ---------------------------------------------------------------------------
# Stop-Loss Tuning Pipeline Tests
# ---------------------------------------------------------------------------

class TestStopLossPipeline:

    @pytest.fixture
    def pipeline_outputs(self):
        return _get_pipeline_outputs()

    @pytest.mark.schema
    def test_stop_loss_source_set(self, pipeline_outputs):
        """After pipeline, stop_loss_source is set."""
        portfolio, strategy, _, _, _ = pipeline_outputs
        output = run_risk_pipeline(portfolio, strategy)
        # Without API key, should be "deterministic"
        assert output.stop_loss_source in ("llm", "deterministic")

    @pytest.mark.schema
    def test_stop_loss_recommendations_type(self, pipeline_outputs):
        """stop_loss_recommendations is a list."""
        portfolio, strategy, _, _, _ = pipeline_outputs
        output = run_risk_pipeline(portfolio, strategy)
        assert isinstance(output.stop_loss_recommendations, list)


class TestSleepWellConviction:
    """Test Sleep Well scores with analyst conviction/volatility data."""

    @pytest.fixture
    def pipeline_outputs(self):
        return _get_pipeline_outputs()

    @pytest.mark.schema
    def test_sleep_well_without_analyst_unchanged(self, pipeline_outputs):
        """Without analyst_output, sleep well scores unchanged from before."""
        portfolio, strategy, _, _, _ = pipeline_outputs
        output_without = run_risk_pipeline(portfolio, strategy)
        output_with_none = run_risk_pipeline(portfolio, strategy, analyst_output=None)
        assert output_without.sleep_well_scores.overall_score == output_with_none.sleep_well_scores.overall_score

    @pytest.mark.schema
    def test_sleep_well_with_analyst_output(self, pipeline_outputs):
        """With analyst_output, sleep well scores still in valid range."""
        portfolio, strategy, analyst, _, _ = pipeline_outputs
        output = run_risk_pipeline(portfolio, strategy, analyst_output=analyst)
        for score in [
            output.sleep_well_scores.tier_1_score,
            output.sleep_well_scores.tier_2_score,
            output.sleep_well_scores.tier_3_score,
            output.sleep_well_scores.overall_score,
        ]:
            assert 1 <= score <= 10

    @pytest.mark.schema
    def test_sleep_well_factors_include_data_note(self, pipeline_outputs):
        """When analyst_output provided, factors mention data incorporation."""
        portfolio, strategy, analyst, _, _ = pipeline_outputs
        output = run_risk_pipeline(portfolio, strategy, analyst_output=analyst)
        factors = output.sleep_well_scores.factors
        has_data_factor = any(
            "conviction" in f.lower() or "volatility" in f.lower()
            for f in factors
        )
        assert has_data_factor, f"Expected conviction/volatility factor in: {factors}"


# ---------------------------------------------------------------------------
# Pattern Quality Check Tests
# ---------------------------------------------------------------------------

from ibd_agents.tools.risk_analyzer import check_pattern_quality
from ibd_agents.schemas.pattern_output import (
    PortfolioPatternOutput,
    PortfolioPatternSummary,
    EnhancedStockAnalysis,
    BaseScoreBreakdown,
    PatternAlert,
)


def _make_enhanced_stock(symbol: str, enhanced_score: int, tier: int = 1) -> EnhancedStockAnalysis:
    """Helper to create a minimal EnhancedStockAnalysis for testing."""
    base = BaseScoreBreakdown(
        ibd_score=min(40.0, enhanced_score * 0.4),
        analyst_score=min(30.0, enhanced_score * 0.3),
        risk_score=min(30.0, enhanced_score * 0.3),
        base_total=enhanced_score,
    )
    label = "Review" if enhanced_score < 70 else ("Monitor" if enhanced_score < 85 else (
        "Notable" if enhanced_score < 100 else "Favorable"))
    stars = "★" if enhanced_score < 70 else ("★★" if enhanced_score < 85 else (
        "★★★" if enhanced_score < 100 else "★★★★"))
    return EnhancedStockAnalysis(
        symbol=symbol, company_name=f"{symbol} Corp", sector="TECH",
        tier=tier, base_score=base, pattern_score=None,
        enhanced_score=enhanced_score, enhanced_rating=stars,
        enhanced_rating_label=label, pattern_alerts=[],
        scoring_source="deterministic",
    )


def _make_pattern_output(
    stocks: list[EnhancedStockAnalysis],
    alerts: list[PatternAlert] | None = None,
) -> PortfolioPatternOutput:
    """Helper to create a PortfolioPatternOutput for testing."""
    return PortfolioPatternOutput(
        stock_analyses=stocks,
        pattern_alerts=alerts or [],
        portfolio_summary=PortfolioPatternSummary(
            total_stocks_scored=len(stocks),
            stocks_with_patterns=0,
            avg_enhanced_score=sum(s.enhanced_score for s in stocks) / max(len(stocks), 1),
            tier_1_candidates=0, category_kings=0, inflection_alerts=0,
            disruption_risks=sum(1 for a in (alerts or []) if a.alert_type == "Disruption Risk"),
        ),
        scoring_source="deterministic",
        methodology_notes="Test pattern output",
        analysis_date="2026-02-20",
        summary="Test pattern output for risk officer pattern quality check testing with sample data",
    )


class TestPatternQualityCheck:
    """Tests for the check_pattern_quality() risk check."""

    @pytest.fixture
    def portfolio(self):
        portfolio, _, _, _, _ = _get_pipeline_outputs()
        return portfolio

    @pytest.mark.schema
    def test_pass_no_data(self, portfolio):
        """pattern_output=None → PASS (backward compatible)."""
        result = check_pattern_quality(portfolio, pattern_output=None)
        assert result.status == "PASS"
        assert result.check_name == "pattern_quality"
        assert "no pattern data" in result.findings.lower()

    @pytest.mark.schema
    def test_pass_strong_scores(self, portfolio):
        """All stocks have enhanced_score >= 60 → PASS."""
        # Create stocks matching portfolio symbols with strong scores
        symbols = set()
        for tier in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]:
            for p in tier.positions:
                symbols.add(p.symbol)

        stocks = [_make_enhanced_stock(sym, 85) for sym in list(symbols)[:5]]
        pattern = _make_pattern_output(stocks)
        result = check_pattern_quality(portfolio, pattern_output=pattern)
        assert result.status == "PASS"

    @pytest.mark.schema
    def test_warning_weak_scores(self, portfolio):
        """3+ portfolio positions with enhanced_score < 60 → WARNING."""
        # Get actual portfolio symbols
        symbols = []
        for tier in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]:
            for p in tier.positions:
                symbols.append(p.symbol)

        # Create 3 weak stocks using portfolio symbols
        stocks = [_make_enhanced_stock(sym, 40) for sym in symbols[:3]]
        # Add some strong stocks too
        stocks.extend([_make_enhanced_stock(sym, 90) for sym in symbols[3:6]])
        pattern = _make_pattern_output(stocks)
        result = check_pattern_quality(portfolio, pattern_output=pattern)
        assert result.status == "WARNING"
        assert "weak" in result.findings.lower()

    @pytest.mark.schema
    def test_warning_disruption_alerts(self, portfolio):
        """2+ Disruption Risk alerts → WARNING."""
        symbols = []
        for tier in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]:
            for p in tier.positions:
                symbols.append(p.symbol)

        stocks = [_make_enhanced_stock(sym, 90) for sym in symbols[:5]]
        alerts = [
            PatternAlert(
                alert_type="Disruption Risk", symbol=symbols[0],
                description="Self-cannibalization score is zero indicating disruption vulnerability",
                pattern_name="Self-Cannibalization", pattern_score=0,
            ),
            PatternAlert(
                alert_type="Disruption Risk", symbol=symbols[1],
                description="Self-cannibalization score is zero indicating disruption vulnerability",
                pattern_name="Self-Cannibalization", pattern_score=0,
            ),
        ]
        pattern = _make_pattern_output(stocks, alerts)
        result = check_pattern_quality(portfolio, pattern_output=pattern)
        assert result.status == "WARNING"
        assert "disruption" in result.findings.lower()

    @pytest.mark.schema
    def test_pass_below_thresholds(self, portfolio):
        """2 weak stocks and 1 disruption alert → still PASS."""
        symbols = []
        for tier in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]:
            for p in tier.positions:
                symbols.append(p.symbol)

        stocks = [_make_enhanced_stock(sym, 40) for sym in symbols[:2]]
        stocks.extend([_make_enhanced_stock(sym, 90) for sym in symbols[2:6]])
        alerts = [
            PatternAlert(
                alert_type="Disruption Risk", symbol=symbols[0],
                description="Self-cannibalization score is zero indicating disruption vulnerability",
                pattern_name="Self-Cannibalization", pattern_score=0,
            ),
        ]
        pattern = _make_pattern_output(stocks, alerts)
        result = check_pattern_quality(portfolio, pattern_output=pattern)
        assert result.status == "PASS"

    @pytest.mark.schema
    def test_pipeline_includes_pattern_check(self):
        """Full pipeline includes pattern_quality in check results."""
        portfolio, strategy, _, _, _ = _get_pipeline_outputs()
        output = run_risk_pipeline(portfolio, strategy)
        names = {c.check_name for c in output.check_results}
        assert "pattern_quality" in names
