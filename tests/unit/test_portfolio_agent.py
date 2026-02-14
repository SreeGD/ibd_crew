"""
Agent 05: Portfolio Manager — Pipeline & Behavioral Tests
Level 2: Deterministic pipeline tests with mock data.
Level 3: Behavioral boundary tests.

Tests the full run_portfolio_pipeline() flow + tool functions.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from ibd_agents.agents.analyst_agent import run_analyst_pipeline
from ibd_agents.agents.portfolio_manager import run_portfolio_pipeline
from ibd_agents.agents.rotation_detector import run_rotation_pipeline
from ibd_agents.agents.sector_strategist import run_strategist_pipeline
from ibd_agents.schemas.analyst_output import AnalystOutput
from ibd_agents.schemas.portfolio_output import (
    ALL_KEEPS,
    ETF_LIMITS,
    FUNDAMENTAL_KEEPS,
    IBD_KEEPS,
    STOCK_LIMITS,
    TRAILING_STOPS,
    PortfolioOutput,
)
from ibd_agents.schemas.research_output import (
    ResearchOutput,
    ResearchStock,
    compute_preliminary_tier,
    is_ibd_keep_candidate,
)
from ibd_agents.schemas.strategy_output import SectorStrategyOutput
from ibd_agents.tools.keeps_manager import place_keeps, build_keep_positions
from ibd_agents.tools.position_sizer import (
    compute_max_loss,
    compute_trailing_stop,
    size_position,
)

# Import shared fixtures
from tests.fixtures.conftest import PADDING_STOCKS, SAMPLE_IBD_STOCKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_research_output(stocks_data: list[dict]) -> ResearchOutput:
    """Build a valid ResearchOutput from raw stock dicts for testing."""
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
            reasoning="Test stock for portfolio pipeline testing",
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
        summary="Test research output for portfolio manager pipeline testing with sample data",
    )


def _build_full_pipeline():
    """Run Research → Analyst → Rotation → Strategy → return all for PM testing."""
    research_output = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
    analyst_output = run_analyst_pipeline(research_output)
    rotation_output = run_rotation_pipeline(analyst_output, research_output)
    strategy_output = run_strategist_pipeline(rotation_output, analyst_output)
    return analyst_output, strategy_output, rotation_output, research_output


# Cache outputs (deterministic)
_cached_outputs = None


def _get_pipeline_outputs():
    global _cached_outputs
    if _cached_outputs is None:
        _cached_outputs = _build_full_pipeline()
    return _cached_outputs


# ---------------------------------------------------------------------------
# Tool Function Tests
# ---------------------------------------------------------------------------

class TestToolFunctions:

    @pytest.mark.schema
    def test_size_position_t1_stock(self):
        """T1 stock sizing respects 5% max."""
        pct = size_position(1, "stock", 10, is_keep=True)
        assert 0.0 < pct <= 5.0

    @pytest.mark.schema
    def test_size_position_t2_stock(self):
        """T2 stock sizing respects 4% max."""
        pct = size_position(2, "stock", 10, is_keep=False)
        assert 0.0 < pct <= 4.0

    @pytest.mark.schema
    def test_size_position_t3_stock(self):
        """T3 stock sizing respects 3% max."""
        pct = size_position(3, "stock", 10, is_keep=False)
        assert 0.0 < pct <= 3.0

    @pytest.mark.schema
    def test_size_position_conviction_scaling(self):
        """Higher conviction → larger position."""
        low = size_position(1, "stock", 2)
        high = size_position(1, "stock", 9)
        assert high > low

    @pytest.mark.schema
    def test_trailing_stop_initial(self):
        """Initial stops: T1=22, T2=18, T3=12."""
        assert compute_trailing_stop(1) == 22.0
        assert compute_trailing_stop(2) == 18.0
        assert compute_trailing_stop(3) == 12.0

    @pytest.mark.schema
    def test_trailing_stop_tightens(self):
        """Stop tightens after gains."""
        initial = compute_trailing_stop(1, gain_pct=0)
        after_10 = compute_trailing_stop(1, gain_pct=12)
        after_20 = compute_trailing_stop(1, gain_pct=25)
        assert after_10 < initial
        assert after_20 < after_10

    @pytest.mark.schema
    def test_max_loss_within_limits(self):
        """Max loss doesn't exceed framework limits."""
        loss = compute_max_loss(1, "stock", 5.0)
        assert loss <= 1.10

    @pytest.mark.schema
    def test_place_keeps_14_total(self):
        """Keeps placement has 14 total."""
        analyst, _, _, _ = _get_pipeline_outputs()
        kp = place_keeps(analyst.rated_stocks)
        assert kp.total_keeps == 14

    @pytest.mark.schema
    def test_keep_positions_built(self):
        """All 14 keep positions built successfully."""
        analyst, _, _, _ = _get_pipeline_outputs()
        kp = place_keeps(analyst.rated_stocks)
        positions = build_keep_positions(kp, analyst.rated_stocks)
        assert len(positions) == 14
        for p in positions:
            assert p.keep_category in ("fundamental", "ibd")


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

class TestPortfolioPipeline:

    @pytest.fixture
    def pipeline_outputs(self):
        return _get_pipeline_outputs()

    @pytest.mark.schema
    def test_pipeline_produces_valid_output(self, pipeline_outputs):
        """Pipeline produces valid PortfolioOutput."""
        analyst, strategy, _, _ = pipeline_outputs
        output = run_portfolio_pipeline(strategy, analyst)
        assert isinstance(output, PortfolioOutput)

    @pytest.mark.schema
    def test_14_keeps_placed(self, pipeline_outputs):
        """All 14 keeps are placed."""
        analyst, strategy, _, _ = pipeline_outputs
        output = run_portfolio_pipeline(strategy, analyst)
        assert output.keeps_placement.total_keeps == 14

    @pytest.mark.schema
    def test_keeps_categories_correct(self, pipeline_outputs):
        """5 fundamental + 9 IBD keeps."""
        analyst, strategy, _, _ = pipeline_outputs
        output = run_portfolio_pipeline(strategy, analyst)
        assert len(output.keeps_placement.fundamental_keeps) == 5
        assert len(output.keeps_placement.ibd_keeps) == 9

    @pytest.mark.schema
    def test_tier_allocation_sums_near_100(self, pipeline_outputs):
        """T1 + T2 + T3 + Cash ≈ 100%."""
        analyst, strategy, _, _ = pipeline_outputs
        output = run_portfolio_pipeline(strategy, analyst)
        total = (
            output.tier_1.actual_pct
            + output.tier_2.actual_pct
            + output.tier_3.actual_pct
            + output.cash_pct
        )
        assert 95.0 <= total <= 105.0, f"Total = {total}"

    @pytest.mark.schema
    def test_no_stock_exceeds_tier_limit(self, pipeline_outputs):
        """No stock position exceeds its tier-specific limit."""
        analyst, strategy, _, _ = pipeline_outputs
        output = run_portfolio_pipeline(strategy, analyst)
        for tier in [output.tier_1, output.tier_2, output.tier_3]:
            for p in tier.positions:
                if p.asset_type == "stock":
                    limit = STOCK_LIMITS[p.tier]
                    assert p.target_pct <= limit + 0.01, (
                        f"{p.symbol}: {p.target_pct}% > {limit}% for T{p.tier} stock"
                    )

    @pytest.mark.schema
    def test_no_etf_exceeds_tier_limit(self, pipeline_outputs):
        """No ETF position exceeds its tier-specific limit."""
        analyst, strategy, _, _ = pipeline_outputs
        output = run_portfolio_pipeline(strategy, analyst)
        for tier in [output.tier_1, output.tier_2, output.tier_3]:
            for p in tier.positions:
                if p.asset_type == "etf":
                    limit = ETF_LIMITS[p.tier]
                    assert p.target_pct <= limit + 0.01, (
                        f"{p.symbol}: {p.target_pct}% > {limit}% for T{p.tier} ETF"
                    )

    @pytest.mark.schema
    def test_min_8_sectors(self, pipeline_outputs):
        """At least 8 sectors in portfolio."""
        analyst, strategy, _, _ = pipeline_outputs
        output = run_portfolio_pipeline(strategy, analyst)
        assert len(output.sector_exposure) >= 8

    @pytest.mark.schema
    def test_stock_etf_ratio_in_range(self, pipeline_outputs):
        """Stock/ETF ratio within 35-65% when both types present."""
        analyst, strategy, _, _ = pipeline_outputs
        output = run_portfolio_pipeline(strategy, analyst)
        total = output.stock_count + output.etf_count
        if output.stock_count > 0 and output.etf_count > 0:
            stock_pct = (output.stock_count / total) * 100
            assert 35.0 <= stock_pct <= 65.0, f"Stock ratio = {stock_pct:.1f}%"

    @pytest.mark.schema
    def test_cash_in_range(self, pipeline_outputs):
        """Cash 2-15%."""
        analyst, strategy, _, _ = pipeline_outputs
        output = run_portfolio_pipeline(strategy, analyst)
        assert 2.0 <= output.cash_pct <= 15.0

    @pytest.mark.schema
    def test_implementation_plan_4_weeks(self, pipeline_outputs):
        """Implementation plan has 4 weeks."""
        analyst, strategy, _, _ = pipeline_outputs
        output = run_portfolio_pipeline(strategy, analyst)
        assert len(output.implementation_plan) == 4
        weeks = [w.week for w in output.implementation_plan]
        assert sorted(weeks) == [1, 2, 3, 4]

    @pytest.mark.schema
    def test_analysis_date_is_today(self, pipeline_outputs):
        """Analysis date matches today."""
        analyst, strategy, _, _ = pipeline_outputs
        output = run_portfolio_pipeline(strategy, analyst)
        assert output.analysis_date == date.today().isoformat()

    @pytest.mark.schema
    def test_summary_min_length(self, pipeline_outputs):
        """Summary must be at least 50 characters."""
        analyst, strategy, _, _ = pipeline_outputs
        output = run_portfolio_pipeline(strategy, analyst)
        assert len(output.summary) >= 50


# ---------------------------------------------------------------------------
# Behavioral Boundary Tests
# ---------------------------------------------------------------------------

class TestBehavioralBoundaries:

    @pytest.fixture
    def portfolio_output(self):
        analyst, strategy, _, _ = _get_pipeline_outputs()
        return run_portfolio_pipeline(strategy, analyst)

    @pytest.mark.behavior
    def test_no_dollar_amounts(self, portfolio_output):
        """No dollar amounts in output text."""
        text = portfolio_output.model_dump_json()
        assert not re.search(r"\$\d+[,.]?\d*\s*(shares|position|worth)", text.lower())

    @pytest.mark.behavior
    def test_no_buy_sell_language(self, portfolio_output):
        """No direct buy/sell language in summary/methodology."""
        for text in [
            portfolio_output.summary,
            portfolio_output.construction_methodology,
        ]:
            text_lower = text.lower()
            for word in ["you should buy", "sell immediately", "purchase now"]:
                assert word not in text_lower, f"Found '{word}' in: {text}"

    @pytest.mark.behavior
    def test_no_price_targets(self, portfolio_output):
        """No specific price targets."""
        text = portfolio_output.model_dump_json().lower()
        assert "price target" not in text
        assert "target price" not in text

    @pytest.mark.behavior
    def test_no_pm_override(self, portfolio_output):
        """Output should not override PM discretion."""
        text = portfolio_output.model_dump_json().lower()
        assert "you must" not in text
        assert "you should immediately" not in text

    @pytest.mark.behavior
    def test_keeps_have_reasoning(self, portfolio_output):
        """All keep positions have reasoning."""
        all_positions = (
            portfolio_output.tier_1.positions
            + portfolio_output.tier_2.positions
            + portfolio_output.tier_3.positions
        )
        keeps = [p for p in all_positions if p.keep_category is not None]
        for k in keeps:
            assert len(k.reasoning) >= 10, f"{k.symbol} has short reasoning"


# ---------------------------------------------------------------------------
# End-to-End Chain Test
# ---------------------------------------------------------------------------

class TestEndToEndChain:

    @pytest.mark.integration
    def test_research_to_portfolio_chain(self):
        """Full Research → Analyst → Rotation → Strategy → Portfolio chain."""
        research_output = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
        analyst_output = run_analyst_pipeline(research_output)
        rotation_output = run_rotation_pipeline(analyst_output, research_output)
        strategy_output = run_strategist_pipeline(rotation_output, analyst_output)
        portfolio_output = run_portfolio_pipeline(strategy_output, analyst_output)

        assert isinstance(portfolio_output, PortfolioOutput)
        assert portfolio_output.keeps_placement.total_keeps == 14
        assert len(portfolio_output.sector_exposure) >= 8
        assert 2.0 <= portfolio_output.cash_pct <= 15.0


# ---------------------------------------------------------------------------
# Golden Dataset Test
# ---------------------------------------------------------------------------

class TestGoldenDataset:

    @pytest.mark.schema
    def test_golden_portfolio_output(self):
        """Verify portfolio output against golden dataset expectations."""
        golden_path = Path(__file__).parent.parent.parent / "golden_datasets" / "portfolio_golden.json"
        if not golden_path.exists():
            pytest.skip("Golden dataset not yet created")

        with open(golden_path) as f:
            golden = json.load(f)

        analyst, strategy, _, _ = _get_pipeline_outputs()
        output = run_portfolio_pipeline(strategy, analyst)

        assert output.keeps_placement.total_keeps == golden["keeps_count"]
        assert len(output.sector_exposure) >= golden["min_sectors"]
        assert golden["cash_min"] <= output.cash_pct <= golden["cash_max"]

        total = (
            output.tier_1.actual_pct
            + output.tier_2.actual_pct
            + output.tier_3.actual_pct
            + output.cash_pct
        )
        assert golden["total_min"] <= total <= golden["total_max"]


# ---------------------------------------------------------------------------
# Dynamic Sizing Pipeline Tests
# ---------------------------------------------------------------------------

class TestDynamicSizingPipeline:

    @pytest.fixture
    def pipeline_outputs(self):
        return _get_pipeline_outputs()

    @pytest.mark.schema
    def test_sizing_source_set_on_all_positions(self, pipeline_outputs):
        """After pipeline, every position has sizing_source set."""
        analyst, strategy, _, _ = pipeline_outputs
        output = run_portfolio_pipeline(strategy, analyst)
        for tier in [output.tier_1, output.tier_2, output.tier_3]:
            for p in tier.positions:
                assert p.sizing_source in ("llm", "deterministic"), \
                    f"{p.symbol} has sizing_source={p.sizing_source}"

    @pytest.mark.schema
    def test_volatility_adjustment_in_range(self, pipeline_outputs):
        """All volatility adjustments are in [-2.0, 2.0]."""
        analyst, strategy, _, _ = pipeline_outputs
        output = run_portfolio_pipeline(strategy, analyst)
        for tier in [output.tier_1, output.tier_2, output.tier_3]:
            for p in tier.positions:
                assert -2.0 <= p.volatility_adjustment <= 2.0, \
                    f"{p.symbol} has volatility_adjustment={p.volatility_adjustment}"

    @pytest.mark.schema
    def test_default_sizing_source_deterministic(self, pipeline_outputs):
        """Without LLM, sizing_source defaults to 'deterministic'."""
        analyst, strategy, _, _ = pipeline_outputs
        output = run_portfolio_pipeline(strategy, analyst)
        det_positions = []
        for tier in [output.tier_1, output.tier_2, output.tier_3]:
            det_positions.extend([p for p in tier.positions if p.sizing_source == "deterministic"])
        # Without API key, all should be deterministic
        assert len(det_positions) > 0
