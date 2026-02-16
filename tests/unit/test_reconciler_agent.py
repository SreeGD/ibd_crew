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


# ---------------------------------------------------------------------------
# Sell Quality Gate Tests
# ---------------------------------------------------------------------------

from ibd_agents.schemas.reconciliation_output import PositionAction
from ibd_agents.tools.position_differ import (
    apply_sell_quality_gate,
    build_sell_quality_context,
    SellQualityContext,
    BEAR_HEDGE_SYMBOLS,
    HIGH_CONVICTION_THRESHOLD,
    STRONG_ETF_RANK_THRESHOLD,
    STRONG_ETF_SCORE_THRESHOLD,
    QUALITY_COMPOSITE_THRESHOLD,
    QUALITY_RS_THRESHOLD,
    QUALITY_SHARPE_THRESHOLD,
    MAX_TURNOVER_RATIO,
    GRADUATED_TRIM_TARGET_PCT,
    GRADUATED_TRIM_WEEK,
)


def _make_sell_action(symbol: str, current_pct: float = 1.5) -> PositionAction:
    return PositionAction(
        symbol=symbol,
        action_type="SELL",
        current_pct=current_pct,
        target_pct=0.0,
        dollar_change=round(-current_pct / 100 * 1_520_000, 2),
        priority="HIGH" if current_pct >= 1.0 else ("MEDIUM" if current_pct >= 0.4 else "LOW"),
        week=1,
        rationale=f"Position not in recommended portfolio — liquidate {current_pct:.1f}%",
    )


def _make_buy_action(symbol: str, target_pct: float = 2.0) -> PositionAction:
    return PositionAction(
        symbol=symbol,
        action_type="BUY",
        current_pct=0.0,
        target_pct=target_pct,
        dollar_change=round(target_pct / 100 * 1_520_000, 2),
        priority="MEDIUM",
        week=2,
        rationale=f"New T1 position at {target_pct:.1f}% allocation",
    )


class TestSellQualityGate:
    """Tests for the sell quality gate that converts aggressive SELLs to TRIMs."""

    @pytest.mark.schema
    def test_empty_context_no_conversions(self):
        """With no analyst data, no SELLs are converted."""
        actions = [_make_sell_action("AAPL"), _make_sell_action("MSFT")]
        ctx = SellQualityContext()
        result = apply_sell_quality_gate(actions, ctx)
        assert all(a.action_type == "SELL" for a in result)

    @pytest.mark.schema
    def test_high_conviction_t1_shielded(self):
        """T1 stock with conviction >= 8 converts SELL to TRIM."""
        actions = [_make_sell_action("WDC")]
        ctx = SellQualityContext(
            stock_map={"WDC": {"tier": 1, "conviction": 9, "composite_rating": 99,
                               "rs_rating": 99, "sharpe_ratio": 1.5, "sector": "CHIPS"}},
        )
        result = apply_sell_quality_gate(actions, ctx)
        assert result[0].action_type == "TRIM"
        assert result[0].target_pct == GRADUATED_TRIM_TARGET_PCT
        assert result[0].week == GRADUATED_TRIM_WEEK

    @pytest.mark.schema
    def test_high_conviction_t2_not_shielded(self):
        """T2 stock with high conviction is NOT shielded by rule 1 (T1 only)."""
        actions = [_make_sell_action("XYZ")]
        ctx = SellQualityContext(
            stock_map={"XYZ": {"tier": 2, "conviction": 9, "composite_rating": 90,
                               "rs_rating": 80, "sharpe_ratio": 0.5, "sector": "RETAIL"}},
        )
        result = apply_sell_quality_gate(actions, ctx)
        assert result[0].action_type == "SELL"

    @pytest.mark.schema
    def test_strong_etf_rank_shielded(self):
        """ETF with rank in top 10 converts SELL to TRIM."""
        actions = [_make_sell_action("EWY")]
        ctx = SellQualityContext(
            etf_map={"EWY": {"tier": 2, "etf_rank": 2, "etf_score": 85.0, "conviction": 7}},
        )
        result = apply_sell_quality_gate(actions, ctx)
        assert result[0].action_type == "TRIM"

    @pytest.mark.schema
    def test_strong_etf_score_shielded(self):
        """ETF with score >= 70 converts SELL to TRIM."""
        actions = [_make_sell_action("ILF")]
        ctx = SellQualityContext(
            etf_map={"ILF": {"tier": 2, "etf_rank": 15, "etf_score": 72.0, "conviction": 5}},
        )
        result = apply_sell_quality_gate(actions, ctx)
        assert result[0].action_type == "TRIM"

    @pytest.mark.schema
    def test_weak_etf_not_shielded(self):
        """ETF with low rank and score is NOT shielded."""
        actions = [_make_sell_action("ARKK")]
        ctx = SellQualityContext(
            etf_map={"ARKK": {"tier": 3, "etf_rank": 25, "etf_score": 40.0, "conviction": 3}},
        )
        result = apply_sell_quality_gate(actions, ctx)
        assert result[0].action_type == "SELL"

    @pytest.mark.schema
    def test_bear_hedge_in_bear_regime(self):
        """GLD preserved as TRIM in bear regime."""
        actions = [_make_sell_action("GLD")]
        ctx = SellQualityContext(regime="bear")
        result = apply_sell_quality_gate(actions, ctx)
        assert result[0].action_type == "TRIM"
        assert "bear" in result[0].rationale.lower()

    @pytest.mark.schema
    def test_bear_hedge_in_bull_regime_not_shielded(self):
        """GLD NOT shielded in bull regime (no analyst data either)."""
        actions = [_make_sell_action("GLD")]
        ctx = SellQualityContext(regime="bull")
        result = apply_sell_quality_gate(actions, ctx)
        assert result[0].action_type == "SELL"

    @pytest.mark.schema
    def test_quality_gate_composite_rs_sharpe(self):
        """Stock with Comp>=95, RS>=90, Sharpe>=1.0 shielded."""
        actions = [_make_sell_action("ADI")]
        ctx = SellQualityContext(
            stock_map={"ADI": {"tier": 2, "conviction": 6, "composite_rating": 99,
                               "rs_rating": 92, "sharpe_ratio": 1.2, "sector": "CHIPS"}},
        )
        result = apply_sell_quality_gate(actions, ctx)
        assert result[0].action_type == "TRIM"

    @pytest.mark.schema
    def test_quality_gate_low_sharpe_not_shielded(self):
        """Stock with high comp/rs but low Sharpe is NOT shielded by rule 4."""
        actions = [_make_sell_action("XYZ")]
        ctx = SellQualityContext(
            stock_map={"XYZ": {"tier": 2, "conviction": 5, "composite_rating": 97,
                               "rs_rating": 91, "sharpe_ratio": 0.7, "sector": "RETAIL"}},
        )
        result = apply_sell_quality_gate(actions, ctx)
        assert result[0].action_type == "SELL"

    @pytest.mark.schema
    def test_turnover_cap_converts_excess_sells(self):
        """When SELL+BUY > 50% of actions, low-priority SELLs convert to TRIMs."""
        # 6 sells + 6 buys = 12/14 = 86% churn > 50% cap
        actions = (
            [_make_sell_action(f"S{i}", 0.3) for i in range(6)]
            + [_make_buy_action(f"B{i}") for i in range(6)]
            + [PositionAction(symbol="KEEP1", action_type="KEEP", current_pct=3.0,
                              target_pct=3.0, dollar_change=0.0, priority="LOW",
                              week=1, rationale="Position approximately at target (3.0% vs 3.0%)")]
            + [PositionAction(symbol="KEEP2", action_type="KEEP", current_pct=2.0,
                              target_pct=2.0, dollar_change=0.0, priority="LOW",
                              week=1, rationale="Position approximately at target (2.0% vs 2.0%)")]
        )
        ctx = SellQualityContext(regime="neutral")  # non-empty context so gate runs
        result = apply_sell_quality_gate(actions, ctx)
        sell_count = sum(1 for a in result if a.action_type == "SELL")
        buy_count = sum(1 for a in result if a.action_type == "BUY")
        total = len(result)
        assert (sell_count + buy_count) / total <= MAX_TURNOVER_RATIO + 0.08

    @pytest.mark.schema
    def test_converted_trim_has_correct_target(self):
        """Converted TRIMs have GRADUATED_TRIM_TARGET_PCT target."""
        actions = [_make_sell_action("WDC")]
        ctx = SellQualityContext(
            stock_map={"WDC": {"tier": 1, "conviction": 9, "composite_rating": 99,
                               "rs_rating": 99, "sharpe_ratio": 2.0, "sector": "CHIPS"}},
        )
        result = apply_sell_quality_gate(actions, ctx)
        assert result[0].target_pct == GRADUATED_TRIM_TARGET_PCT
        assert result[0].week == GRADUATED_TRIM_WEEK
        assert result[0].dollar_change < 0  # Still reducing position

    @pytest.mark.schema
    def test_converted_trim_rationale_explains_reason(self):
        """Converted TRIM rationale contains the shield reason."""
        actions = [_make_sell_action("GLD")]
        ctx = SellQualityContext(regime="bear")
        result = apply_sell_quality_gate(actions, ctx)
        assert "bear" in result[0].rationale.lower()
        assert "reduced" in result[0].rationale.lower()

    @pytest.mark.schema
    def test_non_sell_actions_unchanged(self):
        """BUY, KEEP, ADD, TRIM actions pass through unchanged."""
        actions = [
            _make_buy_action("NVDA"),
            PositionAction(symbol="AAPL", action_type="KEEP", current_pct=3.0,
                          target_pct=3.0, dollar_change=0.0, priority="LOW",
                          week=1, rationale="Position approximately at target (3.0% vs 3.0%)"),
        ]
        ctx = SellQualityContext(
            stock_map={"NVDA": {"tier": 1, "conviction": 10, "composite_rating": 99,
                                "rs_rating": 99, "sharpe_ratio": 2.0, "sector": "CHIPS"}},
        )
        result = apply_sell_quality_gate(actions, ctx)
        assert result[0].action_type == "BUY"
        assert result[1].action_type == "KEEP"
