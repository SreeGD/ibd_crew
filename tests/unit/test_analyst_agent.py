"""
Agent 02: Analyst Agent — Pipeline Tests
Level 2: Deterministic pipeline tests with mock data.

Tests the full run_analyst_pipeline() flow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ibd_agents.agents.analyst_agent import run_analyst_pipeline
from ibd_agents.agents.research_agent import run_research_pipeline
from ibd_agents.schemas.analyst_output import AnalystOutput
from ibd_agents.schemas.research_output import (
    ResearchOutput,
    ResearchStock,
    compute_preliminary_tier,
    is_ibd_keep_candidate,
)

# Import shared fixtures
from tests.fixtures.conftest import PADDING_STOCKS, SAMPLE_IBD_STOCKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_research_output(stocks_data: list[dict]) -> ResearchOutput:
    """Build a valid ResearchOutput from raw stock dicts for testing."""
    from datetime import date

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
            symbol=s["symbol"],
            company_name=s.get("company_name", s["symbol"]),
            sector=s.get("sector", "UNKNOWN"),
            security_type="stock",
            composite_rating=s.get("composite_rating"),
            rs_rating=s.get("rs_rating"),
            eps_rating=s.get("eps_rating"),
            smr_rating=s.get("smr_rating"),
            acc_dis_rating=s.get("acc_dis_rating"),
            ibd_lists=s.get("ibd_lists", ["IBD 50"]),
            schwab_themes=s.get("schwab_themes", []),
            fool_status=None,
            other_ratings={},
            validation_score=3,
            validation_providers=1,
            is_multi_source_validated=False,
            is_ibd_keep_candidate=keep,
            preliminary_tier=tier,
            sources=["test_data.xls"],
            confidence=confidence,
            reasoning="Test stock from golden dataset — awaiting deeper analysis from Analyst Agent",
        ))

    keep_candidates = [s.symbol for s in research_stocks if s.is_ibd_keep_candidate]
    multi_validated = [s.symbol for s in research_stocks if s.is_multi_source_validated]

    return ResearchOutput(
        stocks=research_stocks,
        etfs=[],
        sector_patterns=[],
        data_sources_used=["test_data.xls"],
        data_sources_failed=[],
        total_securities_scanned=len(research_stocks),
        ibd_keep_candidates=keep_candidates,
        multi_source_validated=multi_validated,
        analysis_date=date.today().isoformat(),
        summary="Test research output for analyst pipeline testing with sample IBD stock data and ratings",
    )


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

class TestAnalystPipeline:

    @pytest.fixture
    def research_output(self) -> ResearchOutput:
        """Build ResearchOutput from sample stocks + padding."""
        return _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)

    @pytest.mark.schema
    def test_pipeline_produces_valid_output(self, research_output):
        """Pipeline produces valid AnalystOutput."""
        output = run_analyst_pipeline(research_output)
        assert isinstance(output, AnalystOutput)
        assert len(output.rated_stocks) > 0

    @pytest.mark.schema
    def test_top_50_selection(self, research_output):
        """Top 50 are the highest-scored stocks (T1 first, then T2, etc)."""
        output = run_analyst_pipeline(research_output)
        tiers = [s.tier for s in output.rated_stocks]
        # T1 stocks should come before T2
        t1_indices = [i for i, t in enumerate(tiers) if t == 1]
        t2_indices = [i for i, t in enumerate(tiers) if t == 2]
        if t1_indices and t2_indices:
            assert max(t1_indices) < min(t2_indices)

    @pytest.mark.schema
    def test_sector_rankings_present(self, research_output):
        """Sector rankings cover sectors with ratable stocks."""
        output = run_analyst_pipeline(research_output)
        assert len(output.sector_rankings) > 0
        # Ranks should be sequential from 1
        ranks = [sr.rank for sr in output.sector_rankings]
        assert ranks == list(range(1, len(ranks) + 1))

    @pytest.mark.schema
    def test_sector_rankings_sorted_by_score(self, research_output):
        """Sector rankings sorted by sector_score descending."""
        output = run_analyst_pipeline(research_output)
        scores = [sr.sector_score for sr in output.sector_rankings]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.schema
    def test_ibd_keeps_consistent(self, research_output):
        """All IBD keeps have Comp>=93 AND RS>=90."""
        output = run_analyst_pipeline(research_output)
        for keep in output.ibd_keeps:
            assert keep.composite_rating >= 93
            assert keep.rs_rating >= 90

    @pytest.mark.schema
    def test_tier_distribution_adds_up(self, research_output):
        """Tier counts should add up to total stocks."""
        output = run_analyst_pipeline(research_output)
        td = output.tier_distribution
        total = (td.tier_1_count + td.tier_2_count + td.tier_3_count
                 + td.below_threshold_count + td.unrated_count)
        assert total == len(research_output.stocks)

    @pytest.mark.schema
    def test_elite_filter_summary(self, research_output):
        """Elite filter summary has valid counts."""
        output = run_analyst_pipeline(research_output)
        ef = output.elite_filter_summary
        assert ef.total_screened > 0
        assert ef.passed_all_four <= ef.total_screened

    @pytest.mark.schema
    def test_unrated_stocks_separated(self):
        """Stocks without ratings go to unrated."""
        # Add a stock with no ratings
        stocks = SAMPLE_IBD_STOCKS[:5] + [
            {"symbol": "NORATINGCO", "company_name": "No Rating Inc",
             "sector": "UNKNOWN"},
        ]
        ro = _build_research_output(stocks)
        output = run_analyst_pipeline(ro)
        unrated_symbols = [s.symbol for s in output.unrated_stocks]
        assert "NORATINGCO" in unrated_symbols

    @pytest.mark.schema
    def test_conviction_in_valid_range(self, research_output):
        """All convictions are 1-10."""
        output = run_analyst_pipeline(research_output)
        for s in output.rated_stocks:
            assert 1 <= s.conviction <= 10

    @pytest.mark.schema
    def test_strengths_weaknesses_present(self, research_output):
        """Each rated stock has strengths and weaknesses."""
        output = run_analyst_pipeline(research_output)
        for s in output.rated_stocks:
            assert 1 <= len(s.strengths) <= 5
            assert 1 <= len(s.weaknesses) <= 5
            assert len(s.catalyst) > 0

    @pytest.mark.schema
    def test_reasoning_min_length(self, research_output):
        """Each rated stock has reasoning >= 50 chars."""
        output = run_analyst_pipeline(research_output)
        for s in output.rated_stocks:
            assert len(s.reasoning) >= 50

    @pytest.mark.schema
    def test_methodology_notes_present(self, research_output):
        """Output has methodology notes."""
        output = run_analyst_pipeline(research_output)
        assert "Elite Screening" in output.methodology_notes
        assert "Tier" in output.methodology_notes

    @pytest.mark.schema
    def test_analysis_date_format(self, research_output):
        """analysis_date is YYYY-MM-DD."""
        output = run_analyst_pipeline(research_output)
        import re
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", output.analysis_date)


# ---------------------------------------------------------------------------
# Golden Dataset Verification
# ---------------------------------------------------------------------------

class TestGoldenDataset:

    @pytest.fixture
    def golden(self) -> dict:
        golden_path = Path(__file__).parent.parent.parent / "golden_datasets" / "analyst_golden.json"
        return json.loads(golden_path.read_text())

    @pytest.fixture
    def research_output(self) -> ResearchOutput:
        return _build_research_output(SAMPLE_IBD_STOCKS)

    @pytest.mark.schema
    def test_tier_assignments_match_golden(self, research_output, golden):
        """Tier assignments match golden dataset."""
        output = run_analyst_pipeline(research_output)
        expected = golden["expected_tier_assignments"]

        # Build lookup from output
        rated_map = {s.symbol: s.tier for s in output.rated_stocks}

        for sym, data in expected.items():
            if sym in rated_map:
                assert rated_map[sym] == data["tier"], (
                    f"{sym}: expected tier {data['tier']}, got {rated_map[sym]}"
                )

    @pytest.mark.schema
    def test_ibd_keeps_match_golden(self, research_output, golden):
        """IBD keeps match golden dataset."""
        output = run_analyst_pipeline(research_output)
        keep_symbols = {k.symbol for k in output.ibd_keeps}
        expected = set(golden["expected_ibd_keeps"])
        assert keep_symbols == expected, (
            f"Missing keeps: {expected - keep_symbols}, "
            f"Extra keeps: {keep_symbols - expected}"
        )


# ---------------------------------------------------------------------------
# Valuation Metrics Pipeline Tests
# ---------------------------------------------------------------------------

class TestValuationInPipeline:

    @pytest.fixture
    def research_output(self) -> ResearchOutput:
        return _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)

    @pytest.mark.schema
    def test_pipeline_populates_valuation_metrics(self, research_output):
        """All rated stocks have valuation metrics populated."""
        output = run_analyst_pipeline(research_output)
        for s in output.rated_stocks:
            assert s.estimated_pe is not None, f"{s.symbol} missing estimated_pe"
            assert s.pe_category is not None
            assert s.estimated_beta is not None
            assert s.sharpe_ratio is not None
            assert s.risk_rating is not None

    @pytest.mark.schema
    def test_pipeline_valuation_summary_present(self, research_output):
        """Output has valuation summary with all sections."""
        output = run_analyst_pipeline(research_output)
        vs = output.valuation_summary
        assert vs is not None
        assert vs.market_context.sp500_forward_pe == 23.3
        assert len(vs.tier_pe_stats) > 0
        assert vs.pe_distribution.total > 0
        assert vs.peg_analysis is not None

    @pytest.mark.schema
    def test_pipeline_sharpe_all_positive(self, research_output):
        """For high-RS universe, all Sharpe ratios should be positive."""
        output = run_analyst_pipeline(research_output)
        for s in output.rated_stocks:
            if s.sharpe_ratio is not None:
                assert s.sharpe_ratio > 0, f"{s.symbol} has non-positive Sharpe {s.sharpe_ratio}"

    @pytest.mark.schema
    def test_pipeline_sector_rankings_have_valuation(self, research_output):
        """Sector rankings have avg_pe, avg_beta, avg_volatility."""
        output = run_analyst_pipeline(research_output)
        for sr in output.sector_rankings:
            assert sr.avg_pe is not None, f"{sr.sector} missing avg_pe"
            assert sr.avg_beta is not None, f"{sr.sector} missing avg_beta"
            assert sr.avg_volatility is not None, f"{sr.sector} missing avg_volatility"


class TestCatalystPipeline:
    """Catalyst enrichment field tests."""

    @pytest.fixture
    def research_output(self) -> ResearchOutput:
        return _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)

    @pytest.mark.schema
    def test_catalyst_source_set_on_all_stocks(self, research_output):
        """After pipeline, every rated stock has catalyst_source set."""
        output = run_analyst_pipeline(research_output)
        for s in output.rated_stocks:
            assert s.catalyst_source in ("llm", "template"), \
                f"{s.symbol} has catalyst_source={s.catalyst_source}"

    @pytest.mark.schema
    def test_catalyst_conviction_adjustment_default(self, research_output):
        """Without LLM, catalyst_conviction_adjustment defaults to 0."""
        output = run_analyst_pipeline(research_output)
        # In test mode (no API key), all should be template with adjustment 0
        template_stocks = [s for s in output.rated_stocks if s.catalyst_source == "template"]
        for s in template_stocks:
            assert s.catalyst_conviction_adjustment == 0, \
                f"{s.symbol} has unexpected catalyst_conviction_adjustment={s.catalyst_conviction_adjustment}"


class TestCapSizePipeline:

    @pytest.mark.schema
    def test_pipeline_propagates_cap_size(self):
        """cap_size set on ResearchStock propagates to RatedStock."""
        stocks_data = list(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
        # Set cap_size on the first stock (MU)
        stocks_data[0] = {**stocks_data[0], "cap_size": "large"}
        ro = _build_research_output(stocks_data)
        # Manually set cap_size on the ResearchStock for MU
        for s in ro.stocks:
            if s.symbol == "MU":
                s.cap_size = "large"
                break
        output = run_analyst_pipeline(ro)
        mu_rated = [s for s in output.rated_stocks if s.symbol == "MU"]
        assert len(mu_rated) == 1
        assert mu_rated[0].cap_size == "large"


class TestEndToEndWithMockFiles:
    """Test full pipeline: Research -> Analyst using mock data files."""

    @pytest.mark.schema
    def test_research_to_analyst(self, mock_data_dir):
        """Run Research pipeline then Analyst pipeline."""
        research_output = run_research_pipeline(str(mock_data_dir))
        analyst_output = run_analyst_pipeline(research_output)

        assert isinstance(analyst_output, AnalystOutput)
        assert len(analyst_output.rated_stocks) > 0
        assert analyst_output.tier_distribution.tier_1_count > 0
