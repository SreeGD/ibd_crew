"""
Agent 17: Earnings Risk Analyst — Schema Tests
Level 1: Pure Pydantic validation, no LLM, no file I/O.
"""

from __future__ import annotations

import pytest

from ibd_agents.schemas.earnings_risk_output import (
    ADD_MAX_POSITION_PCT,
    ADD_MIN_BEAT_RATE,
    ADD_MIN_CUSHION_RATIO,
    CONCENTRATION_CRITICAL_PCT,
    CONCENTRATION_MODERATE_PCT,
    CUSHION_COMFORTABLE,
    CUSHION_MANAGEABLE_LOW,
    CUSHION_THIN_LOW,
    FORBIDDEN_PREDICTIONS,
    LARGE_POSITION_PCT,
    LOOKFORWARD_DAYS,
    MIN_CONFIDENT_QUARTERS,
    RISK_ORDER,
    STRATEGY_LADDER,
    WEAK_REGIMES,
    CushionCategory,
    EarningsRisk,
    EarningsRiskOutput,
    EarningsWeek,
    EstimateRevision,
    HistoricalEarnings,
    ImpliedVolSignal,
    PortfolioEarningsConcentration,
    PositionEarningsAnalysis,
    ScenarioOutcome,
    StrategyOption,
    StrategyType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scenario(scenario: str = "BEST", move: float = 10.0) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario=scenario,
        expected_move_pct=move,
        resulting_gain_loss_pct=15.0,
        dollar_impact=5000.0,
        math=f"100 shares x $50.00 gain = $5,000.00",
    )


def _make_scenarios() -> list[ScenarioOutcome]:
    return [
        _make_scenario("BEST", 10.0),
        _make_scenario("BASE", -5.0),
        _make_scenario("WORST", -15.0),
    ]


def _make_strategy(
    strategy: StrategyType = StrategyType.HOLD_FULL,
    scenarios: list[ScenarioOutcome] | None = None,
) -> StrategyOption:
    return StrategyOption(
        strategy=strategy,
        description="Hold all 100 shares through earnings with full exposure.",
        shares_to_sell=None,
        scenarios=scenarios or _make_scenarios(),
        risk_reward_summary="Risk $7,500 to make $5,000",
    )


def _make_historical(
    beat_count: int = 6,
    miss_count: int = 2,
    avg_move: float = 8.0,
) -> HistoricalEarnings:
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


def _make_analysis(
    ticker: str = "NVDA",
    earnings_date: str = "2026-03-10",
    cushion_ratio: float = 2.5,
    cushion_category: CushionCategory = CushionCategory.COMFORTABLE,
    risk_level: EarningsRisk = EarningsRisk.LOW,
    recommended: StrategyType = StrategyType.HOLD_FULL,
    strategies: list[StrategyOption] | None = None,
    historical: HistoricalEarnings | None = None,
    portfolio_pct: float = 3.5,
    gain_loss_pct: float = 20.0,
    rationale: str = "Cushion ratio of 2.5 provides substantial buffer against average earnings moves.",
) -> PositionEarningsAnalysis:
    strats = strategies or [
        _make_strategy(StrategyType.HOLD_FULL),
        _make_strategy(StrategyType.TRIM_TO_HALF),
    ]
    return PositionEarningsAnalysis(
        ticker=ticker,
        account="Schwab",
        earnings_date=earnings_date,
        days_until_earnings=14,
        reporting_time="AFTER_CLOSE",
        shares=100,
        current_price=500.0,
        buy_price=400.0,
        gain_loss_pct=gain_loss_pct,
        position_value=50000.0,
        portfolio_pct=portfolio_pct,
        historical=historical or _make_historical(),
        cushion_ratio=cushion_ratio,
        cushion_category=cushion_category,
        estimate_revision=EstimateRevision.NEUTRAL,
        estimate_revision_detail="No recent revision data available",
        implied_vol_signal=ImpliedVolSignal.NORMAL,
        implied_move_pct=None,
        risk_level=risk_level,
        risk_factors=["Comfortable cushion (ratio 2.5)"],
        strategies=strats,
        recommended_strategy=recommended,
        recommendation_rationale=rationale,
    )


def _make_week(
    week_start: str = "2026-03-09",
    tickers: list[str] | None = None,
    pct: float = 8.0,
    flag: str | None = None,
) -> EarningsWeek:
    return EarningsWeek(
        week_start=week_start,
        week_label="Week of Mar 9",
        positions_reporting=tickers or ["NVDA"],
        aggregate_portfolio_pct=pct,
        concentration_flag=flag,
    )


def _make_concentration(
    count: int = 2,
    pct: float = 10.0,
    weeks: list[EarningsWeek] | None = None,
    risk: str = "LOW",
) -> PortfolioEarningsConcentration:
    return PortfolioEarningsConcentration(
        total_positions_approaching=count,
        total_portfolio_pct_exposed=pct,
        earnings_calendar=weeks or [_make_week(tickers=["NVDA", "CRM"])],
        concentration_risk=risk,
    )


def _make_output(
    analyses: list[PositionEarningsAnalysis] | None = None,
    regime: str = "CONFIRMED_UPTREND",
    concentration: PortfolioEarningsConcentration | None = None,
) -> EarningsRiskOutput:
    a = analyses if analyses is not None else [
        _make_analysis("NVDA", "2026-03-10"),
        _make_analysis("CRM", "2026-03-12"),
    ]
    conc = concentration or _make_concentration(count=len(a))
    return EarningsRiskOutput(
        analysis_date="2026-02-21",
        market_regime=regime,
        lookforward_days=LOOKFORWARD_DAYS,
        analyses=a,
        positions_clear=["AAPL", "MSFT"],
        concentration=conc,
        executive_summary=(
            "Earnings Risk Analysis: 2 positions have earnings within 21 days "
            "in CONFIRMED_UPTREND regime. Risk breakdown: 0 critical, 0 high, 0 moderate, 2 low."
        ),
        data_source="mock",
    )


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    @pytest.mark.schema
    def test_lookforward_days(self):
        assert LOOKFORWARD_DAYS == 21

    @pytest.mark.schema
    def test_cushion_thresholds(self):
        assert CUSHION_COMFORTABLE == 2.0
        assert CUSHION_MANAGEABLE_LOW == 1.0
        assert CUSHION_THIN_LOW == 0.5

    @pytest.mark.schema
    def test_concentration_thresholds(self):
        assert CONCENTRATION_MODERATE_PCT == 30.0
        assert CONCENTRATION_CRITICAL_PCT == 50.0

    @pytest.mark.schema
    def test_large_position_pct(self):
        assert LARGE_POSITION_PCT == 10.0

    @pytest.mark.schema
    def test_add_conditions(self):
        assert ADD_MIN_BEAT_RATE == 87.5
        assert ADD_MIN_CUSHION_RATIO == 2.0
        assert ADD_MAX_POSITION_PCT == 5.0

    @pytest.mark.schema
    def test_min_confident_quarters(self):
        assert MIN_CONFIDENT_QUARTERS == 4

    @pytest.mark.schema
    def test_strategy_ladder_order(self):
        assert STRATEGY_LADDER[0] == "EXIT_BEFORE_EARNINGS"
        assert STRATEGY_LADDER[-1] == "HOLD_AND_ADD"
        assert len(STRATEGY_LADDER) == 6

    @pytest.mark.schema
    def test_risk_order(self):
        assert RISK_ORDER["CRITICAL"] < RISK_ORDER["HIGH"] < RISK_ORDER["MODERATE"] < RISK_ORDER["LOW"]

    @pytest.mark.schema
    def test_weak_regimes(self):
        assert "CORRECTION" in WEAK_REGIMES
        assert "UPTREND_UNDER_PRESSURE" in WEAK_REGIMES
        assert "RALLY_ATTEMPT" in WEAK_REGIMES
        assert "CONFIRMED_UPTREND" not in WEAK_REGIMES

    @pytest.mark.schema
    def test_forbidden_predictions(self):
        assert len(FORBIDDEN_PREDICTIONS) >= 5
        for phrase in FORBIDDEN_PREDICTIONS:
            assert isinstance(phrase, str)


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------

class TestEnums:

    @pytest.mark.schema
    def test_earnings_risk_values(self):
        assert set(EarningsRisk) == {
            EarningsRisk.LOW, EarningsRisk.MODERATE,
            EarningsRisk.HIGH, EarningsRisk.CRITICAL,
        }

    @pytest.mark.schema
    def test_cushion_category_values(self):
        assert set(CushionCategory) == {
            CushionCategory.COMFORTABLE, CushionCategory.MANAGEABLE,
            CushionCategory.THIN, CushionCategory.INSUFFICIENT,
        }

    @pytest.mark.schema
    def test_estimate_revision_values(self):
        assert set(EstimateRevision) == {
            EstimateRevision.POSITIVE, EstimateRevision.NEUTRAL,
            EstimateRevision.NEGATIVE,
        }

    @pytest.mark.schema
    def test_implied_vol_signal_values(self):
        assert set(ImpliedVolSignal) == {
            ImpliedVolSignal.NORMAL, ImpliedVolSignal.ELEVATED_EXPECTATIONS,
        }

    @pytest.mark.schema
    def test_strategy_type_values(self):
        assert len(StrategyType) == 6


# ---------------------------------------------------------------------------
# ScenarioOutcome Tests
# ---------------------------------------------------------------------------

class TestScenarioOutcome:

    @pytest.mark.schema
    def test_valid_scenario(self):
        s = _make_scenario("BEST", 10.0)
        assert s.scenario == "BEST"
        assert s.expected_move_pct == 10.0

    @pytest.mark.schema
    def test_invalid_scenario_name(self):
        with pytest.raises(Exception):
            ScenarioOutcome(
                scenario="NEUTRAL",
                expected_move_pct=0.0,
                resulting_gain_loss_pct=0.0,
                dollar_impact=0.0,
                math="100 shares x $0.00 = $0.00",
            )

    @pytest.mark.schema
    def test_math_min_length(self):
        with pytest.raises(Exception):
            ScenarioOutcome(
                scenario="BEST",
                expected_move_pct=10.0,
                resulting_gain_loss_pct=15.0,
                dollar_impact=5000.0,
                math="hi",  # Too short
            )


# ---------------------------------------------------------------------------
# StrategyOption Tests
# ---------------------------------------------------------------------------

class TestStrategyOption:

    @pytest.mark.schema
    def test_valid_strategy(self):
        s = _make_strategy()
        assert s.strategy == StrategyType.HOLD_FULL
        assert len(s.scenarios) == 3

    @pytest.mark.schema
    def test_requires_3_scenarios(self):
        with pytest.raises(Exception):
            StrategyOption(
                strategy=StrategyType.HOLD_FULL,
                description="Hold all shares through earnings with full exposure.",
                scenarios=[_make_scenario("BEST")],
                risk_reward_summary="Risk $7,500 to make $5,000",
            )

    @pytest.mark.schema
    def test_requires_all_3_scenario_names(self):
        with pytest.raises(Exception):
            StrategyOption(
                strategy=StrategyType.HOLD_FULL,
                description="Hold all shares through earnings with full exposure.",
                scenarios=[
                    _make_scenario("BEST"),
                    _make_scenario("BEST"),
                    _make_scenario("WORST"),
                ],
                risk_reward_summary="Risk $7,500 to make $5,000",
            )


# ---------------------------------------------------------------------------
# HistoricalEarnings Tests
# ---------------------------------------------------------------------------

class TestHistoricalEarnings:

    @pytest.mark.schema
    def test_valid_historical(self):
        h = _make_historical()
        assert h.quarters_analyzed == 8
        assert h.beat_count + h.miss_count == h.quarters_analyzed

    @pytest.mark.schema
    def test_count_mismatch_rejected(self):
        with pytest.raises(Exception):
            HistoricalEarnings(
                quarters_analyzed=8,
                beat_count=5,
                miss_count=2,  # 5+2 != 8
                beat_rate_pct=62.5,
                avg_move_pct=8.0,
                avg_gap_up_pct=10.0,
                avg_gap_down_pct=-12.0,
                max_adverse_move_pct=-18.0,
                move_std_dev=4.0,
                recent_trend="consistent",
            )

    @pytest.mark.schema
    def test_invalid_trend_rejected(self):
        with pytest.raises(Exception):
            _make_historical()
            HistoricalEarnings(
                quarters_analyzed=8,
                beat_count=6,
                miss_count=2,
                beat_rate_pct=75.0,
                avg_move_pct=8.0,
                avg_gap_up_pct=10.0,
                avg_gap_down_pct=-12.0,
                max_adverse_move_pct=-18.0,
                move_std_dev=4.0,
                recent_trend="unknown",
            )

    @pytest.mark.schema
    def test_beat_rate_bounded(self):
        with pytest.raises(Exception):
            HistoricalEarnings(
                quarters_analyzed=8,
                beat_count=6,
                miss_count=2,
                beat_rate_pct=150.0,  # Over 100
                avg_move_pct=8.0,
                avg_gap_up_pct=10.0,
                avg_gap_down_pct=-12.0,
                max_adverse_move_pct=-18.0,
                move_std_dev=4.0,
                recent_trend="consistent",
            )


# ---------------------------------------------------------------------------
# PositionEarningsAnalysis Tests
# ---------------------------------------------------------------------------

class TestPositionAnalysis:

    @pytest.mark.schema
    def test_valid_analysis(self):
        a = _make_analysis()
        assert a.ticker == "NVDA"
        assert len(a.strategies) >= 2

    @pytest.mark.schema
    def test_ticker_uppercased(self):
        a = _make_analysis(ticker="nvda")
        assert a.ticker == "NVDA"

    @pytest.mark.schema
    def test_min_2_strategies(self):
        with pytest.raises(Exception):
            _make_analysis(strategies=[_make_strategy()])

    @pytest.mark.schema
    def test_recommended_must_be_in_strategies(self):
        with pytest.raises(Exception):
            _make_analysis(
                recommended=StrategyType.EXIT_BEFORE_EARNINGS,
                strategies=[
                    _make_strategy(StrategyType.HOLD_FULL),
                    _make_strategy(StrategyType.TRIM_TO_HALF),
                ],
            )

    @pytest.mark.schema
    def test_invalid_reporting_time(self):
        with pytest.raises(Exception):
            _make_analysis()
            PositionEarningsAnalysis(
                ticker="TEST",
                account="Schwab",
                earnings_date="2026-03-10",
                days_until_earnings=14,
                reporting_time="DURING_HOURS",
                shares=100,
                current_price=500.0,
                buy_price=400.0,
                gain_loss_pct=25.0,
                position_value=50000.0,
                portfolio_pct=3.5,
                historical=_make_historical(),
                cushion_ratio=2.5,
                cushion_category=CushionCategory.COMFORTABLE,
                estimate_revision=EstimateRevision.NEUTRAL,
                estimate_revision_detail="No recent data",
                implied_vol_signal=ImpliedVolSignal.NORMAL,
                risk_level=EarningsRisk.LOW,
                risk_factors=["test"],
                strategies=[
                    _make_strategy(StrategyType.HOLD_FULL),
                    _make_strategy(StrategyType.TRIM_TO_HALF),
                ],
                recommended_strategy=StrategyType.HOLD_FULL,
                recommendation_rationale="Test rationale for this position with adequate cushion.",
            )


# ---------------------------------------------------------------------------
# EarningsWeek Tests
# ---------------------------------------------------------------------------

class TestEarningsWeek:

    @pytest.mark.schema
    def test_valid_week(self):
        w = _make_week()
        assert w.week_start == "2026-03-09"

    @pytest.mark.schema
    def test_invalid_flag_rejected(self):
        with pytest.raises(Exception):
            _make_week(flag="WARNING")

    @pytest.mark.schema
    def test_valid_flags(self):
        w1 = _make_week(flag="CONCENTRATED")
        assert w1.concentration_flag == "CONCENTRATED"
        w2 = _make_week(flag="CRITICAL")
        assert w2.concentration_flag == "CRITICAL"
        w3 = _make_week(flag=None)
        assert w3.concentration_flag is None


# ---------------------------------------------------------------------------
# PortfolioEarningsConcentration Tests
# ---------------------------------------------------------------------------

class TestConcentration:

    @pytest.mark.schema
    def test_valid_concentration(self):
        c = _make_concentration()
        assert c.total_positions_approaching == 2

    @pytest.mark.schema
    def test_invalid_risk_level(self):
        with pytest.raises(Exception):
            _make_concentration(risk="EXTREME")


# ---------------------------------------------------------------------------
# EarningsRiskOutput Validator Tests
# ---------------------------------------------------------------------------

class TestOutputValidators:

    @pytest.mark.schema
    def test_valid_output(self):
        output = _make_output()
        assert len(output.analyses) == 2
        assert output.data_source == "mock"

    @pytest.mark.schema
    def test_analyses_ordered_by_date(self):
        """Analyses must be sorted by earnings_date ascending."""
        with pytest.raises(Exception, match="ordered by earnings_date"):
            _make_output(
                analyses=[
                    _make_analysis("CRM", "2026-03-15"),
                    _make_analysis("NVDA", "2026-03-10"),
                ],
            )

    @pytest.mark.schema
    def test_concentration_count_mismatch(self):
        """Concentration count must equal len(analyses)."""
        with pytest.raises(Exception, match="total_positions_approaching"):
            _make_output(
                analyses=[_make_analysis("NVDA", "2026-03-10")],
                concentration=_make_concentration(count=5),
            )

    @pytest.mark.schema
    def test_must_not_s1_weak_market_insufficient_cushion(self):
        """MUST_NOT-S1: No HOLD_FULL in weak regime with insufficient cushion."""
        with pytest.raises(Exception, match="MUST_NOT-S1"):
            _make_output(
                regime="CORRECTION",
                analyses=[
                    _make_analysis(
                        ticker="CRM",
                        earnings_date="2026-03-10",
                        cushion_ratio=0.3,
                        cushion_category=CushionCategory.INSUFFICIENT,
                        risk_level=EarningsRisk.CRITICAL,
                        recommended=StrategyType.HOLD_FULL,
                        strategies=[
                            _make_strategy(StrategyType.HOLD_FULL),
                            _make_strategy(StrategyType.EXIT_BEFORE_EARNINGS),
                        ],
                    ),
                ],
                concentration=_make_concentration(count=1),
            )

    @pytest.mark.schema
    def test_must_not_s1_allows_hold_in_strong_market(self):
        """HOLD_FULL allowed in CONFIRMED_UPTREND even with low cushion."""
        output = _make_output(
            regime="CONFIRMED_UPTREND",
            analyses=[
                _make_analysis(
                    ticker="CRM",
                    earnings_date="2026-03-10",
                    cushion_ratio=0.3,
                    cushion_category=CushionCategory.INSUFFICIENT,
                    risk_level=EarningsRisk.CRITICAL,
                    recommended=StrategyType.HOLD_FULL,
                    strategies=[
                        _make_strategy(StrategyType.HOLD_FULL),
                        _make_strategy(StrategyType.EXIT_BEFORE_EARNINGS),
                    ],
                ),
            ],
            concentration=_make_concentration(count=1),
        )
        assert output.analyses[0].recommended_strategy == StrategyType.HOLD_FULL

    @pytest.mark.schema
    def test_must_not_s2_invalid_hold_and_add(self):
        """MUST_NOT-S2: HOLD_AND_ADD requires all 4 conditions."""
        # Missing: beat_rate < 87.5%
        with pytest.raises(Exception, match="MUST_NOT-S2"):
            _make_output(
                regime="CONFIRMED_UPTREND",
                analyses=[
                    _make_analysis(
                        ticker="CRM",
                        earnings_date="2026-03-10",
                        cushion_ratio=2.5,
                        cushion_category=CushionCategory.COMFORTABLE,
                        risk_level=EarningsRisk.LOW,
                        recommended=StrategyType.HOLD_AND_ADD,
                        strategies=[
                            _make_strategy(StrategyType.HOLD_AND_ADD),
                            _make_strategy(StrategyType.HOLD_FULL),
                        ],
                        historical=_make_historical(beat_count=5, miss_count=3),  # 62.5%
                        portfolio_pct=3.0,
                    ),
                ],
                concentration=_make_concentration(count=1),
            )

    @pytest.mark.schema
    def test_must_not_s2_valid_hold_and_add(self):
        """HOLD_AND_ADD allowed when all 4 conditions met."""
        output = _make_output(
            regime="CONFIRMED_UPTREND",
            analyses=[
                _make_analysis(
                    ticker="AXON",
                    earnings_date="2026-03-10",
                    cushion_ratio=3.0,
                    cushion_category=CushionCategory.COMFORTABLE,
                    risk_level=EarningsRisk.LOW,
                    recommended=StrategyType.HOLD_AND_ADD,
                    strategies=[
                        _make_strategy(StrategyType.HOLD_AND_ADD),
                        _make_strategy(StrategyType.HOLD_FULL),
                    ],
                    historical=_make_historical(beat_count=7, miss_count=1),  # 87.5%
                    portfolio_pct=4.0,
                ),
            ],
            concentration=_make_concentration(count=1),
        )
        assert output.analyses[0].recommended_strategy == StrategyType.HOLD_AND_ADD

    @pytest.mark.schema
    def test_must_not_s2_wrong_regime(self):
        """HOLD_AND_ADD rejected in non-CONFIRMED_UPTREND."""
        with pytest.raises(Exception, match="MUST_NOT-S2"):
            _make_output(
                regime="CORRECTION",
                analyses=[
                    _make_analysis(
                        ticker="AXON",
                        earnings_date="2026-03-10",
                        cushion_ratio=3.0,
                        cushion_category=CushionCategory.COMFORTABLE,
                        risk_level=EarningsRisk.LOW,
                        recommended=StrategyType.HOLD_AND_ADD,
                        strategies=[
                            _make_strategy(StrategyType.HOLD_AND_ADD),
                            _make_strategy(StrategyType.HOLD_FULL),
                        ],
                        historical=_make_historical(beat_count=7, miss_count=1),
                        portfolio_pct=4.0,
                    ),
                ],
                concentration=_make_concentration(count=1),
            )

    @pytest.mark.schema
    def test_must_not_s2_too_large_position(self):
        """HOLD_AND_ADD rejected when portfolio_pct >= 5%."""
        with pytest.raises(Exception, match="MUST_NOT-S2"):
            _make_output(
                regime="CONFIRMED_UPTREND",
                analyses=[
                    _make_analysis(
                        ticker="AXON",
                        earnings_date="2026-03-10",
                        cushion_ratio=3.0,
                        cushion_category=CushionCategory.COMFORTABLE,
                        risk_level=EarningsRisk.LOW,
                        recommended=StrategyType.HOLD_AND_ADD,
                        strategies=[
                            _make_strategy(StrategyType.HOLD_AND_ADD),
                            _make_strategy(StrategyType.HOLD_FULL),
                        ],
                        historical=_make_historical(beat_count=7, miss_count=1),
                        portfolio_pct=6.0,  # Too big
                    ),
                ],
                concentration=_make_concentration(count=1),
            )

    @pytest.mark.schema
    def test_must_not_s6_no_predictions(self):
        """MUST_NOT-S6: No prediction language in rationales."""
        with pytest.raises(Exception, match="MUST_NOT-S6"):
            _make_output(
                analyses=[
                    _make_analysis(
                        ticker="NVDA",
                        earnings_date="2026-03-10",
                        rationale="NVDA will beat earnings based on the strong cushion ratio.",
                    ),
                    _make_analysis("CRM", "2026-03-12"),
                ],
            )

    @pytest.mark.schema
    def test_must_not_s6_likely_to_beat(self):
        with pytest.raises(Exception, match="MUST_NOT-S6"):
            _make_output(
                analyses=[
                    _make_analysis(
                        ticker="NVDA",
                        earnings_date="2026-03-10",
                        rationale="NVDA is likely to beat consensus estimates given strong trends.",
                    ),
                    _make_analysis("CRM", "2026-03-12"),
                ],
            )

    @pytest.mark.schema
    def test_data_source_valid(self):
        output = _make_output()
        assert output.data_source in ("real", "mock")

    @pytest.mark.schema
    def test_data_source_invalid(self):
        with pytest.raises(Exception):
            EarningsRiskOutput(
                analysis_date="2026-02-21",
                market_regime="CONFIRMED_UPTREND",
                lookforward_days=21,
                analyses=[],
                positions_clear=[],
                concentration=_make_concentration(count=0),
                executive_summary="No positions have earnings within the 21-day lookforward window. All positions are clear.",
                data_source="unknown",
            )

    @pytest.mark.schema
    def test_empty_analyses_valid(self):
        """Empty analyses with matching concentration count is valid."""
        output = EarningsRiskOutput(
            analysis_date="2026-02-21",
            market_regime="CONFIRMED_UPTREND",
            lookforward_days=21,
            analyses=[],
            positions_clear=["AAPL", "MSFT"],
            concentration=_make_concentration(count=0, weeks=[]),
            executive_summary="No positions have earnings within the 21-day lookforward window. All positions are clear.",
            data_source="mock",
        )
        assert len(output.analyses) == 0
        assert len(output.positions_clear) == 2
