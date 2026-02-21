"""
Agent 17: Earnings Risk Analyst — Pipeline & Behavioral Tests
Level 2: Deterministic pipeline tests with mock data.
Level 3: Behavioral boundary tests.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from ibd_agents.agents.analyst_agent import run_analyst_pipeline
from ibd_agents.agents.earnings_risk_analyst import run_earnings_risk_pipeline
from ibd_agents.agents.portfolio_manager import run_portfolio_pipeline
from ibd_agents.agents.rotation_detector import run_rotation_pipeline
from ibd_agents.agents.sector_strategist import run_strategist_pipeline
from ibd_agents.schemas.earnings_risk_output import (
    ADD_MAX_POSITION_PCT,
    ADD_MIN_BEAT_RATE,
    ADD_MIN_CUSHION_RATIO,
    CUSHION_COMFORTABLE,
    CUSHION_MANAGEABLE_LOW,
    CUSHION_THIN_LOW,
    FORBIDDEN_PREDICTIONS,
    LOOKFORWARD_DAYS,
    WEAK_REGIMES,
    CushionCategory,
    EarningsRisk,
    EarningsRiskOutput,
    EstimateRevision,
    ImpliedVolSignal,
    StrategyType,
)
from ibd_agents.schemas.portfolio_output import PortfolioOutput
from ibd_agents.schemas.research_output import (
    ResearchOutput,
    ResearchStock,
    compute_preliminary_tier,
    is_ibd_keep_candidate,
)
from ibd_agents.tools.earnings_data_fetcher import (
    apply_strategy_modifiers,
    assess_concentration,
    build_scenario_outcomes,
    build_strategy_options,
    classify_risk_level,
    compute_cushion_ratio,
    determine_base_strategy,
    fetch_earnings_calendar_mock,
    fetch_historical_earnings_mock,
    HistoricalEarnings,
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
            confidence=confidence, reasoning="Test stock for earnings risk pipeline testing",
        ))

    return ResearchOutput(
        stocks=research_stocks, etfs=[], sector_patterns=[],
        data_sources_used=["test_data.xls"], data_sources_failed=[],
        total_securities_scanned=len(research_stocks),
        ibd_keep_candidates=[s.symbol for s in research_stocks if s.is_ibd_keep_candidate],
        multi_source_validated=[],
        analysis_date=date.today().isoformat(),
        summary="Test research output for earnings risk pipeline testing with sample data",
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


def _make_historical(beat_count=6, miss_count=2, avg_move=8.0):
    total = beat_count + miss_count
    return HistoricalEarnings(
        quarters_analyzed=total,
        beat_count=beat_count,
        miss_count=miss_count,
        beat_rate_pct=round(beat_count / total * 100, 1) if total > 0 else 0.0,
        avg_move_pct=avg_move,
        avg_gap_up_pct=10.0,
        avg_gap_down_pct=-12.0,
        max_adverse_move_pct=-18.0,
        move_std_dev=4.0,
        recent_trend="consistent",
    )


# ---------------------------------------------------------------------------
# Tool Function Tests: Cushion Ratio
# ---------------------------------------------------------------------------

class TestCushionRatio:

    @pytest.mark.schema
    def test_comfortable(self):
        ratio, cat = compute_cushion_ratio(20.0, 8.0)
        assert ratio == 2.5
        assert cat == CushionCategory.COMFORTABLE

    @pytest.mark.schema
    def test_manageable(self):
        ratio, cat = compute_cushion_ratio(12.0, 8.0)
        assert ratio == 1.5
        assert cat == CushionCategory.MANAGEABLE

    @pytest.mark.schema
    def test_thin(self):
        ratio, cat = compute_cushion_ratio(5.0, 8.0)
        assert ratio == 0.62
        assert cat == CushionCategory.THIN

    @pytest.mark.schema
    def test_insufficient(self):
        ratio, cat = compute_cushion_ratio(2.0, 8.0)
        assert ratio == 0.25
        assert cat == CushionCategory.INSUFFICIENT

    @pytest.mark.schema
    def test_zero_avg_move_positive_gain(self):
        ratio, cat = compute_cushion_ratio(20.0, 0.0)
        assert ratio == 999.0
        assert cat == CushionCategory.COMFORTABLE

    @pytest.mark.schema
    def test_zero_avg_move_negative_gain(self):
        ratio, cat = compute_cushion_ratio(-5.0, 0.0)
        assert ratio == 0.0
        assert cat == CushionCategory.INSUFFICIENT

    @pytest.mark.schema
    def test_negative_gain(self):
        ratio, cat = compute_cushion_ratio(-3.0, 8.0)
        assert ratio < 0
        assert cat == CushionCategory.INSUFFICIENT

    @pytest.mark.schema
    def test_exact_threshold_comfortable(self):
        ratio, cat = compute_cushion_ratio(16.0, 8.0)
        assert ratio == 2.0
        assert cat == CushionCategory.MANAGEABLE  # > not >=

    @pytest.mark.schema
    def test_exact_threshold_manageable(self):
        ratio, cat = compute_cushion_ratio(8.0, 8.0)
        assert ratio == 1.0
        assert cat == CushionCategory.MANAGEABLE

    @pytest.mark.schema
    def test_exact_threshold_thin(self):
        ratio, cat = compute_cushion_ratio(4.0, 8.0)
        assert ratio == 0.5
        assert cat == CushionCategory.THIN


# ---------------------------------------------------------------------------
# Tool Function Tests: Risk Classification
# ---------------------------------------------------------------------------

class TestRiskClassification:

    @pytest.mark.schema
    def test_comfortable_low_risk(self):
        level, factors = classify_risk_level(
            2.5, CushionCategory.COMFORTABLE, 75.0, 3.0,
            "CONFIRMED_UPTREND", EstimateRevision.NEUTRAL,
            ImpliedVolSignal.NORMAL, 8,
        )
        assert level == EarningsRisk.LOW

    @pytest.mark.schema
    def test_insufficient_critical_risk(self):
        level, factors = classify_risk_level(
            0.3, CushionCategory.INSUFFICIENT, 50.0, 3.0,
            "CONFIRMED_UPTREND", EstimateRevision.NEUTRAL,
            ImpliedVolSignal.NORMAL, 8,
        )
        assert level == EarningsRisk.CRITICAL

    @pytest.mark.schema
    def test_large_position_amplifies(self):
        level, factors = classify_risk_level(
            2.5, CushionCategory.COMFORTABLE, 75.0, 12.0,  # >10%
            "CONFIRMED_UPTREND", EstimateRevision.NEUTRAL,
            ImpliedVolSignal.NORMAL, 8,
        )
        assert level == EarningsRisk.MODERATE
        assert any("Large position" in f for f in factors)

    @pytest.mark.schema
    def test_insufficient_history_amplifies(self):
        level, factors = classify_risk_level(
            2.5, CushionCategory.COMFORTABLE, 75.0, 3.0,
            "CONFIRMED_UPTREND", EstimateRevision.NEUTRAL,
            ImpliedVolSignal.NORMAL, 2,  # < 4
        )
        assert level == EarningsRisk.HIGH
        assert any("Insufficient history" in f for f in factors)

    @pytest.mark.schema
    def test_low_beat_rate_amplifies(self):
        level, factors = classify_risk_level(
            2.5, CushionCategory.COMFORTABLE, 40.0, 3.0,
            "CONFIRMED_UPTREND", EstimateRevision.NEUTRAL,
            ImpliedVolSignal.NORMAL, 8,
        )
        assert level == EarningsRisk.MODERATE
        assert any("Low beat rate" in f for f in factors)

    @pytest.mark.schema
    def test_negative_estimate_amplifies(self):
        level, factors = classify_risk_level(
            1.5, CushionCategory.MANAGEABLE, 75.0, 3.0,
            "CONFIRMED_UPTREND", EstimateRevision.NEGATIVE,
            ImpliedVolSignal.NORMAL, 8,
        )
        assert level == EarningsRisk.HIGH
        assert any("Estimates revised down" in f for f in factors)

    @pytest.mark.schema
    def test_weak_regime_amplifies(self):
        level, factors = classify_risk_level(
            1.5, CushionCategory.MANAGEABLE, 75.0, 3.0,
            "CORRECTION", EstimateRevision.NEUTRAL,
            ImpliedVolSignal.NORMAL, 8,
        )
        assert level == EarningsRisk.HIGH
        assert any("Weak market regime" in f for f in factors)

    @pytest.mark.schema
    def test_risk_factors_always_present(self):
        level, factors = classify_risk_level(
            2.5, CushionCategory.COMFORTABLE, 75.0, 3.0,
            "CONFIRMED_UPTREND", EstimateRevision.NEUTRAL,
            ImpliedVolSignal.NORMAL, 8,
        )
        assert len(factors) >= 1


# ---------------------------------------------------------------------------
# Tool Function Tests: Strategy Decision Matrix
# ---------------------------------------------------------------------------

class TestStrategyMatrix:

    @pytest.mark.schema
    def test_correction_insufficient_exits(self):
        strat = determine_base_strategy(CushionCategory.INSUFFICIENT, "CORRECTION")
        assert strat == StrategyType.EXIT_BEFORE_EARNINGS

    @pytest.mark.schema
    def test_correction_comfortable_trims(self):
        strat = determine_base_strategy(CushionCategory.COMFORTABLE, "CORRECTION")
        assert strat == StrategyType.TRIM_TO_HALF

    @pytest.mark.schema
    def test_uptrend_comfortable_holds(self):
        strat = determine_base_strategy(CushionCategory.COMFORTABLE, "CONFIRMED_UPTREND")
        assert strat == StrategyType.HOLD_FULL

    @pytest.mark.schema
    def test_uptrend_insufficient_trims(self):
        strat = determine_base_strategy(CushionCategory.INSUFFICIENT, "CONFIRMED_UPTREND")
        assert strat == StrategyType.TRIM_TO_HALF

    @pytest.mark.schema
    def test_pressure_comfortable_holds(self):
        strat = determine_base_strategy(CushionCategory.COMFORTABLE, "UPTREND_UNDER_PRESSURE")
        assert strat == StrategyType.HOLD_FULL

    @pytest.mark.schema
    def test_pressure_insufficient_exits(self):
        strat = determine_base_strategy(CushionCategory.INSUFFICIENT, "UPTREND_UNDER_PRESSURE")
        assert strat == StrategyType.EXIT_BEFORE_EARNINGS

    @pytest.mark.schema
    def test_rally_maps_to_correction(self):
        strat = determine_base_strategy(CushionCategory.INSUFFICIENT, "RALLY_ATTEMPT")
        assert strat == StrategyType.EXIT_BEFORE_EARNINGS

    @pytest.mark.schema
    def test_unknown_regime_defaults(self):
        strat = determine_base_strategy(CushionCategory.COMFORTABLE, "UNKNOWN_REGIME")
        assert strat == StrategyType.HOLD_FULL  # Default to CONFIRMED_UPTREND


# ---------------------------------------------------------------------------
# Tool Function Tests: Strategy Modifiers
# ---------------------------------------------------------------------------

class TestStrategyModifiers:

    @pytest.mark.schema
    def test_large_position_shifts_conservative(self):
        result = apply_strategy_modifiers(
            StrategyType.HOLD_FULL, 12.0, 75.0,
            EstimateRevision.NEUTRAL, ImpliedVolSignal.NORMAL,
            None, 8.0, 1,
        )
        # Large position shifts -1: HOLD_FULL → HEDGE_WITH_PUT
        assert result == StrategyType.HEDGE_WITH_PUT

    @pytest.mark.schema
    def test_high_beat_rate_shifts_aggressive(self):
        result = apply_strategy_modifiers(
            StrategyType.TRIM_TO_HALF, 2.0, 90.0,
            EstimateRevision.NEUTRAL, ImpliedVolSignal.NORMAL,
            None, 8.0, 1,
        )
        # High beat rate +1, small position (<3%) +1 → shifts +2
        assert result in (StrategyType.HOLD_FULL, StrategyType.HOLD_AND_ADD)

    @pytest.mark.schema
    def test_negative_estimate_shifts_conservative(self):
        result = apply_strategy_modifiers(
            StrategyType.HOLD_FULL, 5.0, 75.0,
            EstimateRevision.NEGATIVE, ImpliedVolSignal.NORMAL,
            None, 8.0, 1,
        )
        assert result == StrategyType.HEDGE_WITH_PUT

    @pytest.mark.schema
    def test_multiple_conservative_modifiers(self):
        result = apply_strategy_modifiers(
            StrategyType.HOLD_FULL, 12.0, 40.0,
            EstimateRevision.NEGATIVE, ImpliedVolSignal.ELEVATED_EXPECTATIONS,
            None, 8.0, 3,
        )
        # -1 large, -1 low beat, -1 neg estimate, -1 elevated IV, -1 concentration = -5
        assert result == StrategyType.EXIT_BEFORE_EARNINGS

    @pytest.mark.schema
    def test_does_not_go_below_exit(self):
        result = apply_strategy_modifiers(
            StrategyType.EXIT_BEFORE_EARNINGS, 12.0, 40.0,
            EstimateRevision.NEGATIVE, ImpliedVolSignal.NORMAL,
            None, 8.0, 1,
        )
        assert result == StrategyType.EXIT_BEFORE_EARNINGS

    @pytest.mark.schema
    def test_does_not_go_above_add(self):
        result = apply_strategy_modifiers(
            StrategyType.HOLD_AND_ADD, 2.0, 95.0,
            EstimateRevision.POSITIVE, ImpliedVolSignal.NORMAL,
            5.0, 8.0, 1,  # implied < avg → +1
        )
        assert result == StrategyType.HOLD_AND_ADD


# ---------------------------------------------------------------------------
# Tool Function Tests: Scenario Outcomes
# ---------------------------------------------------------------------------

class TestScenarioOutcomes:

    @pytest.mark.schema
    def test_three_scenarios(self):
        outcomes = build_scenario_outcomes(
            shares=100, current_price=500.0, buy_price=400.0,
            gain_loss_pct=25.0, move_pcts=(10.0, -5.0, -15.0),
        )
        assert len(outcomes) == 3
        assert outcomes[0].scenario == "BEST"
        assert outcomes[1].scenario == "BASE"
        assert outcomes[2].scenario == "WORST"

    @pytest.mark.schema
    def test_math_strings_present(self):
        outcomes = build_scenario_outcomes(
            shares=100, current_price=500.0, buy_price=400.0,
            gain_loss_pct=25.0, move_pcts=(10.0, -5.0, -15.0),
        )
        for sc in outcomes:
            assert len(sc.math) >= 5
            assert "shares" in sc.math

    @pytest.mark.schema
    def test_fractional_position(self):
        outcomes = build_scenario_outcomes(
            shares=100, current_price=500.0, buy_price=400.0,
            gain_loss_pct=25.0, move_pcts=(10.0, -5.0, -15.0),
            position_fraction=0.5,
        )
        # With 50% fraction, impact should be half
        full = build_scenario_outcomes(
            shares=100, current_price=500.0, buy_price=400.0,
            gain_loss_pct=25.0, move_pcts=(10.0, -5.0, -15.0),
            position_fraction=1.0,
        )
        assert abs(outcomes[0].dollar_impact) < abs(full[0].dollar_impact)


# ---------------------------------------------------------------------------
# Tool Function Tests: Strategy Options Builder
# ---------------------------------------------------------------------------

class TestStrategyOptions:

    @pytest.mark.schema
    def test_min_2_strategies(self):
        hist = _make_historical()
        strategies, rec, rationale = build_strategy_options(
            shares=100, current_price=500.0, buy_price=400.0,
            gain_loss_pct=25.0, portfolio_pct=3.0,
            historical=hist, cushion_ratio=3.0,
            cushion_category=CushionCategory.COMFORTABLE,
            regime="CONFIRMED_UPTREND",
            estimate_revision=EstimateRevision.NEUTRAL,
            implied_vol_signal=ImpliedVolSignal.NORMAL,
            implied_move_pct=None,
            risk_level=EarningsRisk.LOW,
            same_week_count=1,
        )
        assert len(strategies) >= 2

    @pytest.mark.schema
    def test_3_scenarios_per_strategy(self):
        hist = _make_historical()
        strategies, _, _ = build_strategy_options(
            shares=100, current_price=500.0, buy_price=400.0,
            gain_loss_pct=25.0, portfolio_pct=3.0,
            historical=hist, cushion_ratio=3.0,
            cushion_category=CushionCategory.COMFORTABLE,
            regime="CONFIRMED_UPTREND",
            estimate_revision=EstimateRevision.NEUTRAL,
            implied_vol_signal=ImpliedVolSignal.NORMAL,
            implied_move_pct=None,
            risk_level=EarningsRisk.LOW,
            same_week_count=1,
        )
        for s in strategies:
            assert len(s.scenarios) == 3

    @pytest.mark.schema
    def test_must_not_s1_enforced(self):
        """No HOLD_FULL when insufficient cushion + weak market."""
        hist = _make_historical()
        strategies, rec, _ = build_strategy_options(
            shares=100, current_price=500.0, buy_price=490.0,
            gain_loss_pct=2.0, portfolio_pct=3.0,
            historical=hist, cushion_ratio=0.25,
            cushion_category=CushionCategory.INSUFFICIENT,
            regime="CORRECTION",
            estimate_revision=EstimateRevision.NEUTRAL,
            implied_vol_signal=ImpliedVolSignal.NORMAL,
            implied_move_pct=None,
            risk_level=EarningsRisk.CRITICAL,
            same_week_count=1,
        )
        strategy_types = {s.strategy for s in strategies}
        assert StrategyType.HOLD_FULL not in strategy_types
        assert StrategyType.HOLD_AND_ADD not in strategy_types
        assert rec not in (StrategyType.HOLD_FULL, StrategyType.HOLD_AND_ADD)

    @pytest.mark.schema
    def test_must_not_s2_enforced(self):
        """HOLD_AND_ADD not offered when conditions not met."""
        hist = _make_historical(beat_count=5, miss_count=3)  # 62.5%
        strategies, rec, _ = build_strategy_options(
            shares=100, current_price=500.0, buy_price=400.0,
            gain_loss_pct=25.0, portfolio_pct=3.0,
            historical=hist, cushion_ratio=3.0,
            cushion_category=CushionCategory.COMFORTABLE,
            regime="CONFIRMED_UPTREND",
            estimate_revision=EstimateRevision.NEUTRAL,
            implied_vol_signal=ImpliedVolSignal.NORMAL,
            implied_move_pct=None,
            risk_level=EarningsRisk.LOW,
            same_week_count=1,
        )
        strategy_types = {s.strategy for s in strategies}
        assert StrategyType.HOLD_AND_ADD not in strategy_types

    @pytest.mark.schema
    def test_no_prediction_in_rationale(self):
        hist = _make_historical()
        _, _, rationale = build_strategy_options(
            shares=100, current_price=500.0, buy_price=400.0,
            gain_loss_pct=25.0, portfolio_pct=3.0,
            historical=hist, cushion_ratio=3.0,
            cushion_category=CushionCategory.COMFORTABLE,
            regime="CONFIRMED_UPTREND",
            estimate_revision=EstimateRevision.NEUTRAL,
            implied_vol_signal=ImpliedVolSignal.NORMAL,
            implied_move_pct=None,
            risk_level=EarningsRisk.LOW,
            same_week_count=1,
        )
        lower = rationale.lower()
        for phrase in FORBIDDEN_PREDICTIONS:
            assert phrase not in lower, f"Found forbidden prediction: '{phrase}'"


# ---------------------------------------------------------------------------
# Tool Function Tests: Concentration Assessment
# ---------------------------------------------------------------------------

class TestConcentration:

    @pytest.mark.schema
    def test_empty_analyses(self):
        conc = assess_concentration([])
        assert conc.total_positions_approaching == 0
        assert conc.concentration_risk == "LOW"

    @pytest.mark.schema
    def test_low_concentration(self):
        analyses = [
            {"ticker": "NVDA", "earnings_date": "2026-03-10", "portfolio_pct": 3.0},
            {"ticker": "CRM", "earnings_date": "2026-03-12", "portfolio_pct": 2.5},
        ]
        conc = assess_concentration(analyses)
        assert conc.total_positions_approaching == 2
        assert conc.concentration_risk == "LOW"

    @pytest.mark.schema
    def test_high_concentration(self):
        from datetime import date, timedelta
        base = date.today() + timedelta(days=10)
        analyses = [
            {"ticker": f"S{i}", "earnings_date": base, "portfolio_pct": 8.0}
            for i in range(5)  # 40% in one week
        ]
        conc = assess_concentration(analyses)
        assert conc.concentration_risk in ("HIGH", "CRITICAL")

    @pytest.mark.schema
    def test_critical_concentration(self):
        from datetime import date, timedelta
        base = date.today() + timedelta(days=10)
        analyses = [
            {"ticker": f"S{i}", "earnings_date": base, "portfolio_pct": 9.0}
            for i in range(6)  # 54% in one week
        ]
        conc = assess_concentration(analyses)
        assert conc.concentration_risk == "CRITICAL"

    @pytest.mark.schema
    def test_weekly_grouping(self):
        from datetime import date, timedelta
        monday = date.today() + timedelta(days=(7 - date.today().weekday()) % 7)
        # Two dates in same week
        analyses = [
            {"ticker": "A", "earnings_date": monday, "portfolio_pct": 5.0},
            {"ticker": "B", "earnings_date": monday + timedelta(days=2), "portfolio_pct": 5.0},
        ]
        conc = assess_concentration(analyses)
        assert len(conc.earnings_calendar) == 1
        assert len(conc.earnings_calendar[0].positions_reporting) == 2


# ---------------------------------------------------------------------------
# Mock Data Tests
# ---------------------------------------------------------------------------

class TestMockData:

    @pytest.mark.schema
    def test_mock_calendar_returns_data(self):
        cal = fetch_earnings_calendar_mock(["NVDA", "CRM", "PANW"])
        assert "NVDA" in cal
        assert "CRM" in cal
        for sym, data in cal.items():
            assert "earnings_date" in data
            assert "days_until_earnings" in data

    @pytest.mark.schema
    def test_mock_calendar_etfs_excluded(self):
        cal = fetch_earnings_calendar_mock(["GLD", "SCHD", "VEA"])
        assert len(cal) == 0

    @pytest.mark.schema
    def test_mock_historical_known_stock(self):
        hist = fetch_historical_earnings_mock("NVDA")
        assert hist["beat_count"] == 7
        assert hist["miss_count"] == 1
        assert hist["quarters_analyzed"] == 8

    @pytest.mark.schema
    def test_mock_historical_unknown_stock(self):
        hist = fetch_historical_earnings_mock("UNKNOWN_STOCK_XYZ")
        assert hist["quarters_analyzed"] == 8
        assert hist["beat_count"] + hist["miss_count"] == 8

    @pytest.mark.schema
    def test_mock_historical_deterministic(self):
        h1 = fetch_historical_earnings_mock("UNKNOWN_STOCK_XYZ")
        h2 = fetch_historical_earnings_mock("UNKNOWN_STOCK_XYZ")
        assert h1 == h2


# ---------------------------------------------------------------------------
# MUST_NOT Enforcement Tests
# ---------------------------------------------------------------------------

class TestMustNotEnforcement:

    @pytest.mark.behavior
    def test_must_not_s1_comprehensive(self):
        """No HOLD/ADD in any weak regime with thin-or-less cushion."""
        for regime in WEAK_REGIMES:
            hist = _make_historical()
            strategies, rec, _ = build_strategy_options(
                shares=100, current_price=100.0, buy_price=98.0,
                gain_loss_pct=2.0, portfolio_pct=3.0,
                historical=hist, cushion_ratio=0.25,
                cushion_category=CushionCategory.INSUFFICIENT,
                regime=regime,
                estimate_revision=EstimateRevision.NEUTRAL,
                implied_vol_signal=ImpliedVolSignal.NORMAL,
                implied_move_pct=None,
                risk_level=EarningsRisk.CRITICAL,
                same_week_count=1,
            )
            forbidden = {StrategyType.HOLD_FULL, StrategyType.HOLD_AND_ADD}
            for s in strategies:
                assert s.strategy not in forbidden, (
                    f"MUST_NOT-S1 violated: {s.strategy} offered in {regime}"
                )
            assert rec not in forbidden, (
                f"MUST_NOT-S1 violated: {rec} recommended in {regime}"
            )

    @pytest.mark.behavior
    def test_must_not_s6_no_predictions(self):
        """Rationale from pipeline must never contain prediction language."""
        hist = _make_historical()
        for cushion, cat in [
            (3.0, CushionCategory.COMFORTABLE),
            (1.5, CushionCategory.MANAGEABLE),
            (0.7, CushionCategory.THIN),
            (0.2, CushionCategory.INSUFFICIENT),
        ]:
            _, _, rationale = build_strategy_options(
                shares=100, current_price=500.0, buy_price=400.0,
                gain_loss_pct=25.0, portfolio_pct=3.0,
                historical=hist, cushion_ratio=cushion,
                cushion_category=cat,
                regime="CONFIRMED_UPTREND",
                estimate_revision=EstimateRevision.NEUTRAL,
                implied_vol_signal=ImpliedVolSignal.NORMAL,
                implied_move_pct=None,
                risk_level=EarningsRisk.LOW,
                same_week_count=1,
            )
            lower = rationale.lower()
            for phrase in FORBIDDEN_PREDICTIONS:
                assert phrase not in lower


# ---------------------------------------------------------------------------
# Full Pipeline Test
# ---------------------------------------------------------------------------

class TestPipelineExecution:

    @pytest.mark.schema
    def test_pipeline_produces_valid_output(self):
        portfolio, _, analyst, _, _ = _get_pipeline_outputs()
        output = run_earnings_risk_pipeline(portfolio, analyst_output=analyst)
        assert isinstance(output, EarningsRiskOutput)
        assert output.lookforward_days == LOOKFORWARD_DAYS
        assert output.data_source == "mock"

    @pytest.mark.schema
    def test_pipeline_has_analyses_or_clear(self):
        portfolio, _, analyst, _, _ = _get_pipeline_outputs()
        output = run_earnings_risk_pipeline(portfolio, analyst_output=analyst)
        # Must have at least some positions
        total = len(output.analyses) + len(output.positions_clear)
        assert total > 0

    @pytest.mark.schema
    def test_pipeline_analyses_ordered(self):
        portfolio, _, analyst, _, _ = _get_pipeline_outputs()
        output = run_earnings_risk_pipeline(portfolio, analyst_output=analyst)
        dates = [a.earnings_date for a in output.analyses]
        assert dates == sorted(dates)

    @pytest.mark.schema
    def test_pipeline_min_2_strategies_per_analysis(self):
        portfolio, _, analyst, _, _ = _get_pipeline_outputs()
        output = run_earnings_risk_pipeline(portfolio, analyst_output=analyst)
        for a in output.analyses:
            assert len(a.strategies) >= 2

    @pytest.mark.schema
    def test_pipeline_3_scenarios_per_strategy(self):
        portfolio, _, analyst, _, _ = _get_pipeline_outputs()
        output = run_earnings_risk_pipeline(portfolio, analyst_output=analyst)
        for a in output.analyses:
            for s in a.strategies:
                assert len(s.scenarios) == 3
                names = {sc.scenario for sc in s.scenarios}
                assert names == {"BEST", "BASE", "WORST"}

    @pytest.mark.schema
    def test_pipeline_concentration_matches(self):
        portfolio, _, analyst, _, _ = _get_pipeline_outputs()
        output = run_earnings_risk_pipeline(portfolio, analyst_output=analyst)
        assert output.concentration.total_positions_approaching == len(output.analyses)

    @pytest.mark.schema
    def test_pipeline_executive_summary(self):
        portfolio, _, analyst, _, _ = _get_pipeline_outputs()
        output = run_earnings_risk_pipeline(portfolio, analyst_output=analyst)
        assert len(output.executive_summary) >= 50

    @pytest.mark.schema
    def test_pipeline_no_predictions(self):
        portfolio, _, analyst, _, _ = _get_pipeline_outputs()
        output = run_earnings_risk_pipeline(portfolio, analyst_output=analyst)
        for a in output.analyses:
            lower = a.recommendation_rationale.lower()
            for phrase in FORBIDDEN_PREDICTIONS:
                assert phrase not in lower


# ---------------------------------------------------------------------------
# Golden Dataset Test
# ---------------------------------------------------------------------------

class TestGoldenDataset:

    @pytest.mark.schema
    def test_golden_expectations(self):
        golden_path = Path(__file__).parent.parent.parent / "golden_datasets" / "earnings_risk_golden.json"
        with open(golden_path) as f:
            golden = json.load(f)

        portfolio, _, analyst, _, _ = _get_pipeline_outputs()
        output = run_earnings_risk_pipeline(portfolio, analyst_output=analyst)

        # Verify golden expectations
        assert output.lookforward_days == golden["lookforward_days"]
        assert (output.concentration is not None) == golden["has_concentration"]
        assert (len(output.executive_summary) >= 50) == golden["has_executive_summary"]

        for a in output.analyses:
            assert len(a.strategies) >= golden["min_strategies_per_analysis"]
            for s in a.strategies:
                assert len(s.scenarios) == golden["scenarios_per_strategy"]
            assert a.risk_level.value in golden["valid_risk_levels"]
            assert a.cushion_category.value in golden["valid_cushion_categories"]
            assert a.recommended_strategy.value in golden["valid_strategy_types"]

        if output.data_source is not None:
            assert output.data_source in golden["valid_data_sources"]

        # No forbidden predictions
        for a in output.analyses:
            lower = a.recommendation_rationale.lower()
            for phrase in golden["forbidden_prediction_words"]:
                assert phrase not in lower


# ---------------------------------------------------------------------------
# End-to-End Chain Test
# ---------------------------------------------------------------------------

class TestEndToEndChain:

    @pytest.mark.integration
    def test_research_to_earnings_risk(self):
        """Full pipeline: Research → Analyst → Rotation → Strategy → Portfolio → Earnings Risk."""
        portfolio, strategy, analyst, rotation, research = _get_pipeline_outputs()
        output = run_earnings_risk_pipeline(
            portfolio, analyst_output=analyst,
        )
        assert isinstance(output, EarningsRiskOutput)
        assert output.market_regime == "CONFIRMED_UPTREND"  # Default
        assert output.data_source == "mock"
