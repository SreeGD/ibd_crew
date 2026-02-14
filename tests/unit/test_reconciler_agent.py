"""
Agent 08: Portfolio Reconciler — Pipeline & Behavioral Tests
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
from ibd_agents.agents.portfolio_reconciler import run_reconciler_pipeline
from ibd_agents.agents.rotation_detector import run_rotation_pipeline
from ibd_agents.agents.sector_strategist import run_strategist_pipeline
from ibd_agents.schemas.portfolio_output import ALL_KEEPS
from ibd_agents.schemas.reconciliation_output import (
    ACTION_TYPES,
    IMPLEMENTATION_PHASES,
    ReconciliationOutput,
)
from ibd_agents.schemas.research_output import (
    ResearchOutput,
    ResearchStock,
    compute_preliminary_tier,
    is_ibd_keep_candidate,
)
from ibd_agents.tools.portfolio_reader import build_mock_current_holdings
from ibd_agents.tools.position_differ import (
    compute_money_flow,
    diff_positions,
    verify_keeps,
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
            confidence=confidence, reasoning="Test stock for reconciler pipeline testing",
        ))

    return ResearchOutput(
        stocks=research_stocks, etfs=[], sector_patterns=[],
        data_sources_used=["test_data.xls"], data_sources_failed=[],
        total_securities_scanned=len(research_stocks),
        ibd_keep_candidates=[s.symbol for s in research_stocks if s.is_ibd_keep_candidate],
        multi_source_validated=[],
        analysis_date=date.today().isoformat(),
        summary="Test research output for reconciler pipeline testing with sample data",
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
    def test_mock_holdings_has_keeps(self):
        _, _, analyst, _, _ = _get_pipeline_outputs()
        holdings = build_mock_current_holdings(analyst.rated_stocks)
        symbols = {h.symbol for h in holdings.holdings}
        for keep in ALL_KEEPS:
            assert keep in symbols, f"Keep {keep} not in mock holdings"

    @pytest.mark.schema
    def test_mock_holdings_has_non_recommended(self):
        _, _, analyst, _, _ = _get_pipeline_outputs()
        holdings = build_mock_current_holdings(analyst.rated_stocks)
        symbols = {h.symbol for h in holdings.holdings}
        # Non-recommended stocks from portfolio_reader.py
        for sym in ["CAT", "IDXX", "AMD", "PFE", "DIS"]:
            assert sym in symbols, f"Non-recommended {sym} not in mock holdings"

    @pytest.mark.schema
    def test_mock_holdings_3_accounts(self):
        _, _, analyst, _, _ = _get_pipeline_outputs()
        holdings = build_mock_current_holdings(analyst.rated_stocks)
        assert holdings.account_count == 3

    @pytest.mark.schema
    def test_diff_produces_all_action_types(self):
        portfolio, _, analyst, _, _ = _get_pipeline_outputs()
        holdings = build_mock_current_holdings(analyst.rated_stocks)
        actions = diff_positions(holdings, portfolio)
        types_seen = {a.action_type for a in actions}
        # Must have at least SELL (non-recommended) and BUY (new recommended)
        assert "SELL" in types_seen, "No SELL actions generated"
        assert "BUY" in types_seen, "No BUY actions generated"

    @pytest.mark.schema
    def test_diff_sells_non_recommended(self):
        portfolio, _, analyst, _, _ = _get_pipeline_outputs()
        holdings = build_mock_current_holdings(analyst.rated_stocks)
        actions = diff_positions(holdings, portfolio)
        sell_syms = {a.symbol for a in actions if a.action_type == "SELL"}
        # Non-recommended stocks should be sold
        for sym in ["CAT", "IDXX", "AMD", "PFE", "DIS"]:
            assert sym in sell_syms, f"Non-recommended {sym} not marked for SELL"

    @pytest.mark.schema
    def test_verify_keeps_finds_all_14(self):
        _, _, analyst, _, _ = _get_pipeline_outputs()
        holdings = build_mock_current_holdings(analyst.rated_stocks)
        verification = verify_keeps(holdings)
        assert len(verification.keeps_in_current) == 14

    @pytest.mark.schema
    def test_money_flow_balanced(self):
        portfolio, _, analyst, _, _ = _get_pipeline_outputs()
        holdings = build_mock_current_holdings(analyst.rated_stocks)
        actions = diff_positions(holdings, portfolio)
        flow = compute_money_flow(actions, holdings.total_value)
        expected_net = (flow.sell_proceeds + flow.trim_proceeds) - (flow.buy_cost + flow.add_cost)
        assert abs(flow.net_cash_change - expected_net) < 1.0


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

class TestReconcilerPipeline:

    @pytest.fixture
    def pipeline_outputs(self):
        return _get_pipeline_outputs()

    @pytest.mark.schema
    def test_pipeline_produces_valid_output(self, pipeline_outputs):
        portfolio, _, analyst, _, _ = pipeline_outputs
        output = run_reconciler_pipeline(portfolio, analyst)
        assert isinstance(output, ReconciliationOutput)

    @pytest.mark.schema
    def test_4_week_plan(self, pipeline_outputs):
        portfolio, _, analyst, _, _ = pipeline_outputs
        output = run_reconciler_pipeline(portfolio, analyst)
        assert len(output.implementation_plan) == 4
        weeks = sorted(w.week for w in output.implementation_plan)
        assert weeks == [1, 2, 3, 4]

    @pytest.mark.schema
    def test_week_1_is_liquidation(self, pipeline_outputs):
        portfolio, _, analyst, _, _ = pipeline_outputs
        output = run_reconciler_pipeline(portfolio, analyst)
        w1 = next(w for w in output.implementation_plan if w.week == 1)
        assert "LIQUIDATION" in w1.phase_name.upper()

    @pytest.mark.schema
    def test_week_phases_match(self, pipeline_outputs):
        portfolio, _, analyst, _, _ = pipeline_outputs
        output = run_reconciler_pipeline(portfolio, analyst)
        for w in output.implementation_plan:
            expected_phase = IMPLEMENTATION_PHASES[w.week]
            assert expected_phase in w.phase_name.upper()

    @pytest.mark.schema
    def test_all_action_types_valid(self, pipeline_outputs):
        portfolio, _, analyst, _, _ = pipeline_outputs
        output = run_reconciler_pipeline(portfolio, analyst)
        for a in output.actions:
            assert a.action_type in ACTION_TYPES

    @pytest.mark.schema
    def test_keeps_verified(self, pipeline_outputs):
        portfolio, _, analyst, _, _ = pipeline_outputs
        output = run_reconciler_pipeline(portfolio, analyst)
        assert len(output.keep_verification.keeps_in_current) == 14

    @pytest.mark.schema
    def test_transformation_metrics_present(self, pipeline_outputs):
        portfolio, _, analyst, _, _ = pipeline_outputs
        output = run_reconciler_pipeline(portfolio, analyst)
        assert output.transformation_metrics.before_position_count > 0
        assert output.transformation_metrics.after_position_count > 0
        assert 0 <= output.transformation_metrics.turnover_pct <= 100

    @pytest.mark.schema
    def test_money_flow_has_sell_proceeds(self, pipeline_outputs):
        portfolio, _, analyst, _, _ = pipeline_outputs
        output = run_reconciler_pipeline(portfolio, analyst)
        # Must have sell proceeds from non-recommended positions
        assert output.money_flow.sell_proceeds > 0

    @pytest.mark.schema
    def test_analysis_date_is_today(self, pipeline_outputs):
        portfolio, _, analyst, _, _ = pipeline_outputs
        output = run_reconciler_pipeline(portfolio, analyst)
        assert output.analysis_date == date.today().isoformat()

    @pytest.mark.schema
    def test_summary_min_length(self, pipeline_outputs):
        portfolio, _, analyst, _, _ = pipeline_outputs
        output = run_reconciler_pipeline(portfolio, analyst)
        assert len(output.summary) >= 50

    @pytest.mark.schema
    def test_sells_in_week_1(self, pipeline_outputs):
        """SELL actions should be assigned to week 1."""
        portfolio, _, analyst, _, _ = pipeline_outputs
        output = run_reconciler_pipeline(portfolio, analyst)
        for a in output.actions:
            if a.action_type == "SELL":
                assert a.week == 1, f"SELL for {a.symbol} in week {a.week}, expected 1"

    @pytest.mark.schema
    def test_current_holdings_populated(self, pipeline_outputs):
        portfolio, _, analyst, _, _ = pipeline_outputs
        output = run_reconciler_pipeline(portfolio, analyst)
        assert len(output.current_holdings.holdings) > 0
        assert output.current_holdings.total_value > 0


# ---------------------------------------------------------------------------
# Behavioral Boundary Tests
# ---------------------------------------------------------------------------

class TestBehavioralBoundaries:

    @pytest.fixture
    def reconciler_output(self):
        portfolio, _, analyst, _, _ = _get_pipeline_outputs()
        return run_reconciler_pipeline(portfolio, analyst)

    @pytest.mark.behavior
    def test_no_dollar_amounts_in_summary(self, reconciler_output):
        """Summary should not expose specific dollar amounts."""
        assert not re.search(r"\$\d+[,.]?\d*", reconciler_output.summary)

    @pytest.mark.behavior
    def test_no_buy_sell_recommendations(self, reconciler_output):
        """Summary should not contain advisory language."""
        text = reconciler_output.summary.lower()
        for phrase in ["you should buy", "recommend selling", "must purchase"]:
            assert phrase not in text

    @pytest.mark.behavior
    def test_no_price_targets(self, reconciler_output):
        text = reconciler_output.model_dump_json().lower()
        assert "price target" not in text
        assert "fair value" not in text

    @pytest.mark.behavior
    def test_actions_have_rationale(self, reconciler_output):
        """Every action must have a rationale >= 10 chars."""
        for a in reconciler_output.actions:
            assert len(a.rationale) >= 10, f"Action {a.symbol} rationale too short"


# ---------------------------------------------------------------------------
# End-to-End Chain Test
# ---------------------------------------------------------------------------

class TestEndToEndChain:

    @pytest.mark.integration
    def test_full_chain_to_reconciler(self):
        research = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
        analyst = run_analyst_pipeline(research)
        rotation = run_rotation_pipeline(analyst, research)
        strategy = run_strategist_pipeline(rotation, analyst)
        portfolio = run_portfolio_pipeline(strategy, analyst)
        reconciliation = run_reconciler_pipeline(portfolio, analyst)

        assert isinstance(reconciliation, ReconciliationOutput)
        assert len(reconciliation.implementation_plan) == 4
        assert len(reconciliation.keep_verification.keeps_in_current) == 14
        assert reconciliation.money_flow.sell_proceeds > 0


# ---------------------------------------------------------------------------
# Golden Dataset Test
# ---------------------------------------------------------------------------

class TestGoldenDataset:

    @pytest.mark.schema
    def test_golden_reconciler_output(self):
        golden_path = Path(__file__).parent.parent.parent / "golden_datasets" / "reconciler_golden.json"
        if not golden_path.exists():
            pytest.skip("Golden dataset not yet created")

        with open(golden_path) as f:
            golden = json.load(f)

        portfolio, _, analyst, _, _ = _get_pipeline_outputs()
        output = run_reconciler_pipeline(portfolio, analyst)

        assert len(output.implementation_plan) == golden["implementation_weeks"]
        assert len(output.keep_verification.keeps_in_current) == golden["keeps_found"]
        assert output.transformation_metrics.turnover_pct >= golden["min_turnover_pct"]
        assert output.money_flow.sell_proceeds > golden["min_sell_proceeds"]


# ---------------------------------------------------------------------------
# Rationale Source Pipeline Tests
# ---------------------------------------------------------------------------

class TestRationaleSourcePipeline:

    @pytest.fixture
    def pipeline_outputs(self):
        return _get_pipeline_outputs()

    @pytest.mark.schema
    def test_rationale_source_set(self, pipeline_outputs):
        """After pipeline, rationale_source is set."""
        portfolio, _, analyst, _, _ = pipeline_outputs
        output = run_reconciler_pipeline(portfolio, analyst)
        # Without API key, should be "deterministic"
        assert output.rationale_source in ("llm", "deterministic")

    @pytest.mark.schema
    def test_pipeline_accepts_context_params(self, pipeline_outputs):
        """Pipeline accepts rotation_output, strategy_output, risk_output."""
        portfolio, strategy, analyst, rotation, _ = pipeline_outputs
        from ibd_agents.agents.risk_officer import run_risk_pipeline
        risk = run_risk_pipeline(portfolio, strategy)
        output = run_reconciler_pipeline(
            portfolio, analyst,
            rotation_output=rotation,
            strategy_output=strategy,
            risk_output=risk,
        )
        assert isinstance(output, ReconciliationOutput)
        assert output.rationale_source in ("llm", "deterministic")
