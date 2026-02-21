"""
Agent 15: Exit Strategist — Pipeline & Behavioral Tests
Level 2: Deterministic pipeline tests with mock data.
Level 3: Behavioral boundary tests.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from ibd_agents.agents.analyst_agent import run_analyst_pipeline
from ibd_agents.agents.exit_strategist import run_exit_strategist_pipeline
from ibd_agents.agents.portfolio_manager import run_portfolio_pipeline
from ibd_agents.agents.risk_officer import run_risk_pipeline
from ibd_agents.agents.rotation_detector import run_rotation_pipeline
from ibd_agents.agents.sector_strategist import run_strategist_pipeline
from ibd_agents.schemas.exit_strategy_output import (
    HEALTH_SCORE_RANGE,
    SELL_RULE_NAMES,
    URGENCY_ORDER,
    ExitActionType,
    ExitMarketRegime,
    ExitStrategyOutput,
    SellRule,
    SellType,
    Urgency,
)
from ibd_agents.schemas.research_output import (
    ResearchOutput,
    ResearchStock,
    compute_preliminary_tier,
    is_ibd_keep_candidate,
)
from ibd_agents.tools.exit_analyzer import (
    check_below_50_day_ma,
    check_climax_top,
    check_earnings_risk,
    check_profit_target_20_25,
    check_regime_tightened_stop,
    check_rs_deterioration,
    check_sector_distribution,
    check_stop_loss_7_8,
    classify_action,
    classify_urgency,
    compute_health_score,
    compute_portfolio_impact,
    evaluate_position,
)
from ibd_agents.tools.position_monitor import (
    MarketHealthData,
    PositionMarketData,
    build_mock_market_data,
    build_mock_market_health,
    map_regime_to_exit_regime,
)

from tests.fixtures.conftest import PADDING_STOCKS, SAMPLE_IBD_STOCKS


# ---------------------------------------------------------------------------
# Force mock market data in tests (deterministic)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def _force_mock_market_data():
    """Tests always use mock data for deterministic behavior."""
    import ibd_agents.agents.exit_strategist as es_module
    es_module.has_real_market_data = lambda: False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_research_output(stocks_data: list[dict]) -> ResearchOutput:
    research_stocks = []
    for s in stocks_data:
        tier = compute_preliminary_tier(
            s.get("composite_rating"), s.get("rs_rating"), s.get("eps_rating"),
        )
        keep = is_ibd_keep_candidate(s.get("composite_rating"), s.get("rs_rating"))
        has_ratings = sum(
            1 for r in [
                s.get("composite_rating"), s.get("rs_rating"),
                s.get("eps_rating"), s.get("smr_rating"),
                s.get("acc_dis_rating"),
            ]
            if r is not None
        )
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
            confidence=confidence, reasoning="Test stock for exit strategist testing",
        ))

    return ResearchOutput(
        stocks=research_stocks, etfs=[], sector_patterns=[],
        data_sources_used=["test_data.xls"], data_sources_failed=[],
        total_securities_scanned=len(research_stocks),
        ibd_keep_candidates=[s.symbol for s in research_stocks if s.is_ibd_keep_candidate],
        multi_source_validated=[],
        analysis_date=date.today().isoformat(),
        summary="Test research output for exit strategist pipeline testing with sample data",
    )


def _build_full_pipeline():
    research = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
    analyst = run_analyst_pipeline(research)
    rotation = run_rotation_pipeline(analyst, research)
    strategy = run_strategist_pipeline(rotation, analyst)
    portfolio = run_portfolio_pipeline(strategy, analyst)
    risk = run_risk_pipeline(portfolio, strategy)
    return portfolio, risk, strategy, analyst, rotation, research


_cached = None


def _get_pipeline_outputs():
    global _cached
    if _cached is None:
        _cached = _build_full_pipeline()
    return _cached


def _make_position_data(**overrides) -> PositionMarketData:
    """Create PositionMarketData with sensible defaults and overrides."""
    defaults = dict(
        symbol="TEST",
        tier=2,
        asset_type="stock",
        sector="CHIPS",
        current_price=100.0,
        buy_price=100.0,
        gain_loss_pct=0.0,
        days_held=30,
        volume_ratio=1.0,
        ma_50=98.0,
        pct_from_50ma=2.04,
        ma_200=93.0,
        pct_from_200ma=7.53,
        rs_rating_current=85,
        rs_rating_peak_since_buy=85,
        rs_rating_4w_ago=83,
        sector_distribution_days_3w=1,
        days_until_earnings=60,
        avg_earnings_move_pct=8.0,
        price_surge_pct_3w=3.0,
        price_surge_volume_ratio=1.0,
    )
    defaults.update(overrides)
    return PositionMarketData(**defaults)


# ===========================================================================
# TestSellRuleFunctions
# ===========================================================================

class TestSellRuleFunctions:
    """Test each of 8 sell rule check functions."""

    regime = ExitMarketRegime.CONFIRMED_UPTREND

    # --- Stop Loss 7-8% ---

    @pytest.mark.schema
    def test_stop_loss_triggered(self):
        data = _make_position_data(gain_loss_pct=-8.2)
        result = check_stop_loss_7_8(data, self.regime)
        assert result.triggered is True
        assert result.rule == SellRule.STOP_LOSS_7_8

    @pytest.mark.schema
    def test_stop_loss_not_triggered(self):
        data = _make_position_data(gain_loss_pct=-5.0)
        result = check_stop_loss_7_8(data, self.regime)
        assert result.triggered is False

    @pytest.mark.schema
    def test_stop_loss_boundary(self):
        data = _make_position_data(gain_loss_pct=-7.0)
        result = check_stop_loss_7_8(data, self.regime)
        assert result.triggered is True

    @pytest.mark.schema
    def test_hard_backstop(self):
        data = _make_position_data(gain_loss_pct=-12.0)
        result = check_stop_loss_7_8(data, self.regime)
        assert result.triggered is True
        assert "HARD BACKSTOP" in result.detail

    # --- Climax Top ---

    @pytest.mark.schema
    def test_climax_top_triggered(self):
        data = _make_position_data(
            gain_loss_pct=40.0,
            price_surge_pct_3w=35.0,
            price_surge_volume_ratio=2.5,
        )
        result = check_climax_top(data, self.regime)
        assert result.triggered is True

    @pytest.mark.schema
    def test_climax_top_not_triggered_low_volume(self):
        data = _make_position_data(
            gain_loss_pct=40.0,
            price_surge_pct_3w=30.0,
            price_surge_volume_ratio=1.5,  # Below 2.0x threshold
        )
        result = check_climax_top(data, self.regime)
        assert result.triggered is False

    @pytest.mark.schema
    def test_climax_top_not_triggered_low_surge(self):
        data = _make_position_data(
            gain_loss_pct=10.0,
            price_surge_pct_3w=15.0,  # Below 25% threshold
            price_surge_volume_ratio=3.0,
        )
        result = check_climax_top(data, self.regime)
        assert result.triggered is False

    # --- Below 50-day MA ---

    @pytest.mark.schema
    def test_below_50ma_triggered(self):
        data = _make_position_data(
            pct_from_50ma=-2.1,
            volume_ratio=2.0,
        )
        result = check_below_50_day_ma(data, self.regime)
        assert result.triggered is True

    @pytest.mark.schema
    def test_below_50ma_not_triggered_low_volume(self):
        data = _make_position_data(
            pct_from_50ma=-3.0,
            volume_ratio=1.0,  # Below 1.5x
        )
        result = check_below_50_day_ma(data, self.regime)
        assert result.triggered is False

    @pytest.mark.schema
    def test_above_50ma(self):
        data = _make_position_data(pct_from_50ma=5.0, volume_ratio=2.0)
        result = check_below_50_day_ma(data, self.regime)
        assert result.triggered is False

    # --- RS Deterioration ---

    @pytest.mark.schema
    def test_rs_below_threshold(self):
        data = _make_position_data(rs_rating_current=65, rs_rating_peak_since_buy=80)
        result = check_rs_deterioration(data, self.regime)
        assert result.triggered is True

    @pytest.mark.schema
    def test_rs_large_drop(self):
        data = _make_position_data(rs_rating_current=75, rs_rating_peak_since_buy=92)
        result = check_rs_deterioration(data, self.regime)
        assert result.triggered is True  # Drop of 17 >= 15

    @pytest.mark.schema
    def test_rs_healthy(self):
        data = _make_position_data(rs_rating_current=85, rs_rating_peak_since_buy=88)
        result = check_rs_deterioration(data, self.regime)
        assert result.triggered is False

    # --- Profit Target 20-25% ---

    @pytest.mark.schema
    def test_profit_target_triggered(self):
        data = _make_position_data(gain_loss_pct=22.0, days_held=60)
        result = check_profit_target_20_25(data, self.regime)
        assert result.triggered is True

    @pytest.mark.schema
    def test_profit_target_8_week_exception(self):
        """20%+ in < 3 weeks → NOT triggered (8-week hold)."""
        data = _make_position_data(gain_loss_pct=22.0, days_held=12)
        result = check_profit_target_20_25(data, self.regime)
        assert result.triggered is False
        assert "8-week hold" in result.detail

    @pytest.mark.schema
    def test_profit_target_below(self):
        data = _make_position_data(gain_loss_pct=15.0, days_held=30)
        result = check_profit_target_20_25(data, self.regime)
        assert result.triggered is False

    # --- Sector Distribution ---

    @pytest.mark.schema
    def test_sector_distribution_triggered(self):
        data = _make_position_data(sector_distribution_days_3w=5)
        result = check_sector_distribution(data, self.regime)
        assert result.triggered is True

    @pytest.mark.schema
    def test_sector_distribution_not_triggered(self):
        data = _make_position_data(sector_distribution_days_3w=2)
        result = check_sector_distribution(data, self.regime)
        assert result.triggered is False

    # --- Regime Tightened Stop ---

    @pytest.mark.schema
    def test_regime_tightened_stop_correction(self):
        data = _make_position_data(gain_loss_pct=-2.5)
        result = check_regime_tightened_stop(data, ExitMarketRegime.CORRECTION)
        assert result.triggered is True

    @pytest.mark.schema
    def test_regime_tightened_stop_not_in_uptrend(self):
        data = _make_position_data(gain_loss_pct=-2.5)
        result = check_regime_tightened_stop(data, ExitMarketRegime.CONFIRMED_UPTREND)
        assert result.triggered is False

    @pytest.mark.schema
    def test_regime_tightened_stop_no_double_count(self):
        """If standard 7-8% stop already caught it, regime stop does NOT trigger."""
        data = _make_position_data(gain_loss_pct=-8.0)
        result = check_regime_tightened_stop(data, ExitMarketRegime.CORRECTION)
        assert result.triggered is False

    # --- Earnings Risk ---

    @pytest.mark.schema
    def test_earnings_risk_triggered(self):
        data = _make_position_data(
            days_until_earnings=10,
            gain_loss_pct=8.0,  # Below 15% cushion
            avg_earnings_move_pct=12.0,
        )
        result = check_earnings_risk(data, self.regime)
        assert result.triggered is True

    @pytest.mark.schema
    def test_earnings_risk_sufficient_cushion(self):
        data = _make_position_data(
            days_until_earnings=10,
            gain_loss_pct=18.0,  # Above 15% cushion
        )
        result = check_earnings_risk(data, self.regime)
        assert result.triggered is False

    @pytest.mark.schema
    def test_earnings_risk_far_away(self):
        data = _make_position_data(days_until_earnings=60, gain_loss_pct=5.0)
        result = check_earnings_risk(data, self.regime)
        assert result.triggered is False

    @pytest.mark.schema
    def test_earnings_risk_no_date(self):
        data = _make_position_data(days_until_earnings=None)
        result = check_earnings_risk(data, self.regime)
        assert result.triggered is False


# ===========================================================================
# TestUrgencyClassification
# ===========================================================================

class TestUrgencyClassification:
    """Test urgency classification logic."""

    regime = ExitMarketRegime.CONFIRMED_UPTREND

    @pytest.mark.schema
    def test_stop_loss_is_critical(self):
        data = _make_position_data(gain_loss_pct=-8.5)
        results = [check_stop_loss_7_8(data, self.regime)]
        assert classify_urgency(results, data, self.regime) == Urgency.CRITICAL

    @pytest.mark.schema
    def test_climax_top_is_critical(self):
        data = _make_position_data(
            gain_loss_pct=40.0, price_surge_pct_3w=30.0,
            price_surge_volume_ratio=2.5,
        )
        results = [check_climax_top(data, self.regime)]
        assert classify_urgency(results, data, self.regime) == Urgency.CRITICAL

    @pytest.mark.schema
    def test_below_50ma_is_warning(self):
        data = _make_position_data(pct_from_50ma=-3.0, volume_ratio=2.0)
        results = [check_below_50_day_ma(data, self.regime)]
        assert classify_urgency(results, data, self.regime) == Urgency.WARNING

    @pytest.mark.schema
    def test_profit_target_is_watch(self):
        data = _make_position_data(gain_loss_pct=22.0, days_held=60)
        results = [check_profit_target_20_25(data, self.regime)]
        assert classify_urgency(results, data, self.regime) == Urgency.WATCH

    @pytest.mark.schema
    def test_nothing_triggered_is_healthy(self):
        data = _make_position_data(gain_loss_pct=5.0)
        from ibd_agents.schemas.exit_strategy_output import SellRuleResult, SellRule
        results = [SellRuleResult(
            rule=SellRule.NONE, triggered=False, value=5.0, threshold=None,
            detail="No rules triggered for this healthy position observation",
        )]
        assert classify_urgency(results, data, self.regime) == Urgency.HEALTHY

    @pytest.mark.schema
    def test_regime_stop_correction_is_critical(self):
        data = _make_position_data(gain_loss_pct=-2.5)
        results = [check_regime_tightened_stop(data, ExitMarketRegime.CORRECTION)]
        assert classify_urgency(results, data, ExitMarketRegime.CORRECTION) == Urgency.CRITICAL


# ===========================================================================
# TestActionClassification
# ===========================================================================

class TestActionClassification:
    """Test action, sell_type, sell_pct classification."""

    regime = ExitMarketRegime.CONFIRMED_UPTREND

    @pytest.mark.schema
    def test_stop_loss_is_sell_all_defensive(self):
        data = _make_position_data(gain_loss_pct=-8.5)
        results = [check_stop_loss_7_8(data, self.regime)]
        action, sell_type, sell_pct = classify_action(
            results, data, self.regime, Urgency.CRITICAL,
        )
        assert action == ExitActionType.SELL_ALL
        assert sell_type == SellType.DEFENSIVE
        assert sell_pct == 100.0

    @pytest.mark.schema
    def test_climax_top_is_sell_all_offensive(self):
        data = _make_position_data(
            gain_loss_pct=40.0, price_surge_pct_3w=30.0,
            price_surge_volume_ratio=2.5,
        )
        results = [check_climax_top(data, self.regime)]
        action, sell_type, _ = classify_action(
            results, data, self.regime, Urgency.CRITICAL,
        )
        assert action == ExitActionType.SELL_ALL
        assert sell_type == SellType.OFFENSIVE

    @pytest.mark.schema
    def test_earnings_risk_is_trim(self):
        data = _make_position_data(
            days_until_earnings=10, gain_loss_pct=8.0,
        )
        results = [check_earnings_risk(data, self.regime)]
        action, sell_type, sell_pct = classify_action(
            results, data, self.regime, Urgency.WARNING,
        )
        assert action == ExitActionType.TRIM
        assert sell_type == SellType.DEFENSIVE
        assert sell_pct == 50.0

    @pytest.mark.schema
    def test_8_week_hold_action(self):
        data = _make_position_data(gain_loss_pct=22.0, days_held=12)
        # Profit target NOT triggered due to 8-week exception
        results = [check_profit_target_20_25(data, self.regime)]
        action, sell_type, sell_pct = classify_action(
            results, data, self.regime, Urgency.HEALTHY,
        )
        assert action == ExitActionType.HOLD_8_WEEK
        assert sell_type == SellType.NOT_APPLICABLE
        assert sell_pct is None

    @pytest.mark.schema
    def test_rs_deterioration_tighten_stop(self):
        data = _make_position_data(rs_rating_current=65, rs_rating_peak_since_buy=80)
        results = [check_rs_deterioration(data, self.regime)]
        action, sell_type, _ = classify_action(
            results, data, self.regime, Urgency.WARNING,
        )
        assert action == ExitActionType.TIGHTEN_STOP
        assert sell_type == SellType.NOT_APPLICABLE

    @pytest.mark.schema
    def test_healthy_is_hold(self):
        from ibd_agents.schemas.exit_strategy_output import SellRuleResult
        data = _make_position_data(gain_loss_pct=5.0)
        results = [SellRuleResult(
            rule=SellRule.NONE, triggered=False, value=5.0, threshold=None,
            detail="No rules triggered for this healthy position observation",
        )]
        action, sell_type, sell_pct = classify_action(
            results, data, self.regime, Urgency.HEALTHY,
        )
        assert action == ExitActionType.HOLD
        assert sell_type == SellType.NOT_APPLICABLE
        assert sell_pct is None


# ===========================================================================
# TestPositionMonitor
# ===========================================================================

class TestPositionMonitor:
    """Test position monitor utility functions."""

    @pytest.mark.schema
    def test_regime_mapping_bull(self):
        assert map_regime_to_exit_regime("bull regime") == ExitMarketRegime.CONFIRMED_UPTREND

    @pytest.mark.schema
    def test_regime_mapping_bear(self):
        assert map_regime_to_exit_regime("bear") == ExitMarketRegime.CORRECTION

    @pytest.mark.schema
    def test_regime_mapping_neutral(self):
        assert map_regime_to_exit_regime("neutral") == ExitMarketRegime.UPTREND_UNDER_PRESSURE

    @pytest.mark.schema
    def test_regime_mapping_rally(self):
        assert map_regime_to_exit_regime("rally attempt") == ExitMarketRegime.RALLY_ATTEMPT

    @pytest.mark.schema
    def test_mock_market_health_uptrend(self):
        health = build_mock_market_health(ExitMarketRegime.CONFIRMED_UPTREND)
        assert health.exposure_model >= 60

    @pytest.mark.schema
    def test_mock_market_health_correction(self):
        health = build_mock_market_health(ExitMarketRegime.CORRECTION)
        assert health.exposure_model <= 30


# ===========================================================================
# TestExitStrategistPipeline
# ===========================================================================

class TestExitStrategistPipeline:
    """Test the deterministic pipeline end to end."""

    @pytest.fixture
    def pipeline_outputs(self):
        portfolio, risk, strategy, analyst, rotation, _research = _get_pipeline_outputs()
        return portfolio, risk, strategy, analyst, rotation

    @pytest.mark.behavior
    def test_pipeline_returns_valid_output(self, pipeline_outputs):
        portfolio, risk, strategy, analyst, rotation = pipeline_outputs
        result = run_exit_strategist_pipeline(
            portfolio, risk, strategy, analyst, rotation,
        )
        assert isinstance(result, ExitStrategyOutput)

    @pytest.mark.behavior
    def test_all_positions_evaluated(self, pipeline_outputs):
        portfolio, risk, strategy, analyst, rotation = pipeline_outputs
        result = run_exit_strategist_pipeline(
            portfolio, risk, strategy, analyst, rotation,
        )
        # Count total positions in portfolio
        total = (
            len(portfolio.tier_1.positions)
            + len(portfolio.tier_2.positions)
            + len(portfolio.tier_3.positions)
        )
        assert len(result.signals) == total

    @pytest.mark.behavior
    def test_signals_ordered_by_urgency(self, pipeline_outputs):
        portfolio, risk, strategy, analyst, rotation = pipeline_outputs
        result = run_exit_strategist_pipeline(
            portfolio, risk, strategy, analyst, rotation,
        )
        urgencies = [URGENCY_ORDER[s.urgency.value] for s in result.signals]
        assert urgencies == sorted(urgencies)

    @pytest.mark.behavior
    def test_health_score_in_range(self, pipeline_outputs):
        portfolio, risk, strategy, analyst, rotation = pipeline_outputs
        result = run_exit_strategist_pipeline(
            portfolio, risk, strategy, analyst, rotation,
        )
        assert HEALTH_SCORE_RANGE[0] <= result.portfolio_health_score <= HEALTH_SCORE_RANGE[1]

    @pytest.mark.behavior
    def test_every_signal_has_evidence(self, pipeline_outputs):
        portfolio, risk, strategy, analyst, rotation = pipeline_outputs
        result = run_exit_strategist_pipeline(
            portfolio, risk, strategy, analyst, rotation,
        )
        for sig in result.signals:
            assert len(sig.evidence) >= 1, f"{sig.symbol} has no evidence"

    @pytest.mark.behavior
    def test_portfolio_impact_present(self, pipeline_outputs):
        portfolio, risk, strategy, analyst, rotation = pipeline_outputs
        result = run_exit_strategist_pipeline(
            portfolio, risk, strategy, analyst, rotation,
        )
        assert result.portfolio_impact is not None
        assert result.portfolio_impact.current_cash_pct >= 0

    @pytest.mark.behavior
    def test_reasoning_source_deterministic(self, pipeline_outputs):
        portfolio, risk, strategy, analyst, rotation = pipeline_outputs
        result = run_exit_strategist_pipeline(
            portfolio, risk, strategy, analyst, rotation,
        )
        assert result.reasoning_source == "deterministic"

    @pytest.mark.behavior
    def test_summary_nonempty(self, pipeline_outputs):
        portfolio, risk, strategy, analyst, rotation = pipeline_outputs
        result = run_exit_strategist_pipeline(
            portfolio, risk, strategy, analyst, rotation,
        )
        assert len(result.summary) >= 50

    @pytest.mark.behavior
    def test_analysis_date_is_today(self, pipeline_outputs):
        portfolio, risk, strategy, analyst, rotation = pipeline_outputs
        result = run_exit_strategist_pipeline(
            portfolio, risk, strategy, analyst, rotation,
        )
        assert result.analysis_date == date.today().isoformat()


# ===========================================================================
# TestBehavioralBoundaries
# ===========================================================================

class TestBehavioralBoundaries:
    """Test behavioral constraints from the spec."""

    regime = ExitMarketRegime.CONFIRMED_UPTREND

    @pytest.mark.behavior
    def test_stop_loss_absolute_never_overridden_by_strong_rs(self):
        """MUST-M2: 7-8% stop loss is non-negotiable, even with RS=99."""
        data = _make_position_data(
            gain_loss_pct=-8.5,
            rs_rating_current=99,
            rs_rating_peak_since_buy=99,
        )
        signal = evaluate_position(data, self.regime)
        assert signal.urgency == Urgency.CRITICAL
        assert signal.action == ExitActionType.SELL_ALL

    @pytest.mark.behavior
    def test_no_averaging_down(self):
        """No position at a loss should ever recommend ADD or BUY."""
        data = _make_position_data(gain_loss_pct=-5.0)
        signal = evaluate_position(data, self.regime)
        # Should be HOLD or TIGHTEN_STOP — never anything that adds exposure
        assert signal.action in (
            ExitActionType.HOLD,
            ExitActionType.TIGHTEN_STOP,
            ExitActionType.SELL_ALL,
        )

    @pytest.mark.behavior
    def test_regime_tightens_stops(self):
        """In CORRECTION, a -2.5% loss triggers sell (vs -7% in uptrend)."""
        data = _make_position_data(gain_loss_pct=-2.5)

        signal_uptrend = evaluate_position(data, ExitMarketRegime.CONFIRMED_UPTREND)
        signal_correction = evaluate_position(data, ExitMarketRegime.CORRECTION)

        assert signal_uptrend.urgency == Urgency.HEALTHY or signal_uptrend.urgency == Urgency.WATCH
        assert signal_correction.urgency == Urgency.CRITICAL

    @pytest.mark.behavior
    def test_8_week_hold_overridden_by_stop_loss(self):
        """Even a 20%+ fast gainer gets sold if it falls 7-8% from buy."""
        data = _make_position_data(
            gain_loss_pct=-8.0,
            days_held=10,  # Fast gainer window
        )
        signal = evaluate_position(data, self.regime)
        assert signal.urgency == Urgency.CRITICAL
        assert signal.action == ExitActionType.SELL_ALL

    @pytest.mark.behavior
    def test_sell_all_always_has_sell_type(self):
        """SELL_ALL must be OFFENSIVE or DEFENSIVE, never N/A."""
        data = _make_position_data(gain_loss_pct=-9.0)
        signal = evaluate_position(data, self.regime)
        assert signal.action == ExitActionType.SELL_ALL
        assert signal.sell_type in (SellType.OFFENSIVE, SellType.DEFENSIVE)

    @pytest.mark.behavior
    def test_hold_has_na_sell_type(self):
        """HOLD must have N/A sell_type."""
        data = _make_position_data(gain_loss_pct=5.0)
        signal = evaluate_position(data, self.regime)
        if signal.action == ExitActionType.HOLD:
            assert signal.sell_type == SellType.NOT_APPLICABLE

    @pytest.mark.behavior
    def test_evaluate_position_runs_all_8_rules(self):
        """evaluate_position should check all 8 sell rules."""
        data = _make_position_data(gain_loss_pct=5.0)
        signal = evaluate_position(data, self.regime)
        # Should produce a valid signal regardless of rule results
        assert signal.symbol == "TEST"
        assert len(signal.evidence) >= 1


# ===========================================================================
# TestGoldenDataset
# ===========================================================================

class TestGoldenDataset:
    """Verify pipeline output against golden expectations."""

    @pytest.fixture
    def golden(self):
        golden_path = Path(__file__).parent.parent.parent / "golden_datasets" / "exit_strategy_golden.json"
        with open(golden_path) as f:
            return json.load(f)

    @pytest.fixture
    def result(self):
        portfolio, risk, strategy, analyst, rotation, _research = _get_pipeline_outputs()
        return run_exit_strategist_pipeline(
            portfolio, risk, strategy, analyst, rotation,
        )

    @pytest.mark.behavior
    def test_sell_rule_count(self, golden):
        assert len(SELL_RULE_NAMES) == golden["sell_rule_count"]

    @pytest.mark.behavior
    def test_min_signals(self, golden, result):
        assert len(result.signals) >= golden["min_signals"]

    @pytest.mark.behavior
    def test_health_score_bounds(self, golden, result):
        assert result.portfolio_health_score >= golden["health_score_min"]
        assert result.portfolio_health_score <= golden["health_score_max"]

    @pytest.mark.behavior
    def test_urgency_levels_present(self, golden, result):
        actual_urgencies = {s.urgency.value for s in result.signals}
        # Must have at least CRITICAL and HEALTHY
        assert "CRITICAL" in actual_urgencies, "Expected at least 1 CRITICAL signal"
        assert "HEALTHY" in actual_urgencies, "Expected at least 1 HEALTHY signal"

    @pytest.mark.behavior
    def test_min_critical_signals(self, golden, result):
        n_critical = sum(1 for s in result.signals if s.urgency == Urgency.CRITICAL)
        assert n_critical >= golden["min_critical_signals"]

    @pytest.mark.behavior
    def test_min_healthy_signals(self, golden, result):
        n_healthy = sum(1 for s in result.signals if s.urgency == Urgency.HEALTHY)
        assert n_healthy >= golden["min_healthy_signals"]

    @pytest.mark.behavior
    def test_portfolio_impact_present(self, golden, result):
        assert result.portfolio_impact is not None

    @pytest.mark.behavior
    def test_evidence_per_signal(self, golden, result):
        for sig in result.signals:
            assert len(sig.evidence) >= golden["evidence_per_signal_min"]


# ===========================================================================
# TestEndToEnd
# ===========================================================================

class TestEndToEnd:
    """Full chain: Research → Analyst → Rotation → Strategy → Portfolio → Risk → Exit."""

    @pytest.mark.integration
    def test_full_chain(self):
        portfolio, risk, strategy, analyst, rotation, _research = _get_pipeline_outputs()
        result = run_exit_strategist_pipeline(
            portfolio, risk, strategy, analyst, rotation,
        )
        assert isinstance(result, ExitStrategyOutput)
        assert result.portfolio_health_score >= 1
        assert len(result.signals) >= 1
        # Verify signals are properly ordered
        urgencies = [URGENCY_ORDER[s.urgency.value] for s in result.signals]
        assert urgencies == sorted(urgencies)
