"""
Agent 07: Returns Projector — Schema Tests
Level 1: Pure Pydantic validation, no LLM, no file I/O.
"""

from __future__ import annotations

import pytest

from ibd_agents.schemas.returns_projection_output import (
    BENCHMARK_RETURNS,
    BENCHMARKS,
    DEFAULT_PORTFOLIO_VALUE,
    SCENARIOS,
    SCENARIO_WEIGHTS,
    STANDARD_CAVEATS,
    STOP_LOSS_BY_TIER,
    TIER_HISTORICAL_RETURNS,
    TIME_HORIZONS,
    AlphaAnalysis,
    AlphaSource,
    BenchmarkComparison,
    ConfidenceIntervals,
    ExpectedReturn,
    PercentileRange,
    ReturnsProjectionOutput,
    RiskMetrics,
    ScenarioProjection,
    TierAllocation,
    TierMixScenario,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tier_allocation(**overrides) -> TierAllocation:
    defaults = dict(
        tier_1_pct=40.0,
        tier_2_pct=30.0,
        tier_3_pct=20.0,
        cash_pct=10.0,
        tier_1_value=600_000.0,
        tier_2_value=450_000.0,
        tier_3_value=300_000.0,
        cash_value=150_000.0,
    )
    defaults.update(overrides)
    return TierAllocation(**defaults)


def _make_scenario(scenario: str, probability: float) -> ScenarioProjection:
    returns_by_scenario = {
        "bull": {"t1": 35.0, "t2": 25.0, "t3": 15.0, "3m": 7.0, "6m": 14.0, "12m": 27.0},
        "base": {"t1": 18.0, "t2": 14.0, "t3": 10.0, "3m": 3.5, "6m": 7.5, "12m": 15.0},
        "bear": {"t1": -8.0, "t2": -5.0, "t3": 2.0, "3m": -1.5, "6m": -3.0, "12m": -4.0},
    }
    r = returns_by_scenario[scenario]
    portfolio_gain_12m = DEFAULT_PORTFOLIO_VALUE * (r["12m"] / 100.0)
    ending_value = DEFAULT_PORTFOLIO_VALUE + portfolio_gain_12m
    return ScenarioProjection(
        scenario=scenario,
        probability=probability,
        tier_1_return_pct=r["t1"],
        tier_2_return_pct=r["t2"],
        tier_3_return_pct=r["t3"],
        portfolio_return_3m=r["3m"],
        portfolio_return_6m=r["6m"],
        portfolio_return_12m=r["12m"],
        portfolio_gain_12m=portfolio_gain_12m,
        ending_value_12m=ending_value,
        tier_1_contribution=r["t1"] * 0.40,
        tier_2_contribution=r["t2"] * 0.30,
        tier_3_contribution=r["t3"] * 0.20,
        reasoning=f"The {scenario} scenario projects returns based on tier historical performance and market conditions.",
    )


def _make_expected_return() -> ExpectedReturn:
    return ExpectedReturn(
        expected_3m=3.9,
        expected_6m=8.2,
        expected_12m=16.1,
    )


def _make_benchmark(symbol: str, name: str) -> BenchmarkComparison:
    return BenchmarkComparison(
        symbol=symbol,
        name=name,
        benchmark_return_3m=2.5,
        benchmark_return_6m=5.0,
        benchmark_return_12m=10.5,
        alpha_bull_12m=16.5,
        alpha_base_12m=4.5,
        alpha_bear_12m=6.5,
        expected_alpha_12m=5.6,
        outperform_probability=0.72,
    )


def _make_alpha_analysis() -> AlphaAnalysis:
    return AlphaAnalysis(
        primary_alpha_sources=[
            AlphaSource(
                source="Momentum factor exposure",
                contribution_pct=4.5,
                confidence="high",
                reasoning="IBD T1 stocks historically capture strong momentum premiums over benchmarks.",
            ),
        ],
        primary_alpha_drags=[
            AlphaSource(
                source="Transaction costs and slippage",
                contribution_pct=-0.5,
                confidence="medium",
                reasoning="Trailing stop rebalancing incurs transaction costs reducing net alpha.",
            ),
        ],
        net_expected_alpha_vs_spy=5.6,
        net_expected_alpha_vs_dia=7.1,
        net_expected_alpha_vs_qqq=2.1,
        alpha_persistence_note="Alpha persistence is moderate; momentum factor can reverse in sharp regime changes.",
    )


def _make_risk_metrics(**overrides) -> RiskMetrics:
    defaults = dict(
        max_drawdown_with_stops=17.88,
        max_drawdown_without_stops=35.0,
        max_drawdown_dollar=268_200.0,
        projected_annual_volatility=18.5,
        spy_annual_volatility=15.0,
        portfolio_sharpe_bull=1.69,
        portfolio_sharpe_base=0.73,
        portfolio_sharpe_bear=-0.57,
        spy_sharpe_base=0.40,
        return_per_unit_risk=0.87,
        probability_any_stop_triggers_12m=0.45,
        expected_stops_triggered_12m=3,
        whipsaw_risk_note="Trailing stops may trigger during volatile pullbacks causing premature exits.",
    )
    defaults.update(overrides)
    return RiskMetrics(**defaults)


def _make_tier_mix(label: str) -> TierMixScenario:
    mixes = {
        "Aggressive": dict(tier_1_pct=60, tier_2_pct=25, tier_3_pct=10, cash_pct=5,
                           expected_return_12m=20.0, bull_return_12m=33.0, bear_return_12m=-6.0,
                           max_drawdown_with_stops=19.5, sharpe_ratio_base=0.68),
        "Balanced": dict(tier_1_pct=40, tier_2_pct=30, tier_3_pct=20, cash_pct=10,
                         expected_return_12m=16.0, bull_return_12m=27.0, bear_return_12m=-4.0,
                         max_drawdown_with_stops=17.88, sharpe_ratio_base=0.73),
        "Conservative": dict(tier_1_pct=20, tier_2_pct=30, tier_3_pct=35, cash_pct=15,
                             expected_return_12m=11.0, bull_return_12m=18.0, bear_return_12m=0.5,
                             max_drawdown_with_stops=12.0, sharpe_ratio_base=0.81),
    }
    m = mixes[label]
    return TierMixScenario(
        label=label,
        comparison_note=f"The {label} mix trades off upside for downside protection compared to the recommended mix.",
        **m,
    )


def _make_percentile_range(offset: float = 0.0) -> PercentileRange:
    return PercentileRange(
        p10=-5.0 + offset,
        p25=2.0 + offset,
        p50=10.0 + offset,
        p75=18.0 + offset,
        p90=26.0 + offset,
        p10_dollar=DEFAULT_PORTFOLIO_VALUE * (1 + (-5.0 + offset) / 100),
        p50_dollar=DEFAULT_PORTFOLIO_VALUE * (1 + (10.0 + offset) / 100),
        p90_dollar=DEFAULT_PORTFOLIO_VALUE * (1 + (26.0 + offset) / 100),
    )


def _make_confidence_intervals() -> ConfidenceIntervals:
    return ConfidenceIntervals(
        horizon_3m=_make_percentile_range(offset=-7.0),
        horizon_6m=_make_percentile_range(offset=-3.0),
        horizon_12m=_make_percentile_range(offset=0.0),
    )


def _make_full_output(**overrides) -> ReturnsProjectionOutput:
    defaults = dict(
        portfolio_value=DEFAULT_PORTFOLIO_VALUE,
        analysis_date="2025-01-15",
        market_regime="neutral",
        tier_allocation=_make_tier_allocation(),
        scenarios=[
            _make_scenario("bull", 0.25),
            _make_scenario("base", 0.50),
            _make_scenario("bear", 0.25),
        ],
        expected_return=_make_expected_return(),
        benchmark_comparisons=[
            _make_benchmark("SPY", "S&P 500"),
            _make_benchmark("DIA", "Dow Jones"),
            _make_benchmark("QQQ", "NASDAQ-100"),
        ],
        alpha_analysis=_make_alpha_analysis(),
        risk_metrics=_make_risk_metrics(),
        tier_mix_comparison=[
            _make_tier_mix("Aggressive"),
            _make_tier_mix("Balanced"),
            _make_tier_mix("Conservative"),
        ],
        confidence_intervals=_make_confidence_intervals(),
        summary=(
            "Returns Projector: neutral regime. Expected 12-month return 16.1% "
            "with 72% probability of outperforming SPY. Portfolio Sharpe 0.73 base case."
        ),
        caveats=list(STANDARD_CAVEATS),
    )
    defaults.update(overrides)
    return ReturnsProjectionOutput(**defaults)


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------

class TestConstants:

    @pytest.mark.schema
    def test_tier_historical_returns_structure(self):
        """3 tiers, each has 3 scenarios, each has mean and std."""
        assert len(TIER_HISTORICAL_RETURNS) == 3
        for tier in (1, 2, 3):
            tier_data = TIER_HISTORICAL_RETURNS[tier]
            assert set(tier_data.keys()) == {"bull", "base", "bear"}
            for scenario_data in tier_data.values():
                assert "mean" in scenario_data
                assert "std" in scenario_data

    @pytest.mark.schema
    def test_benchmark_returns_structure(self):
        """3 benchmarks (SPY/DIA/QQQ), each has 3 scenarios."""
        assert set(BENCHMARK_RETURNS.keys()) == {"SPY", "DIA", "QQQ"}
        for sym in BENCHMARKS:
            bench_data = BENCHMARK_RETURNS[sym]
            assert set(bench_data.keys()) == {"bull", "base", "bear"}
            for scenario_data in bench_data.values():
                assert "mean" in scenario_data
                assert "std" in scenario_data

    @pytest.mark.schema
    def test_scenario_weights_sum(self):
        """Each regime's scenario weights must sum to 1.0."""
        for regime, weights in SCENARIO_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, (
                f"Regime '{regime}' weights sum to {total}, expected 1.0"
            )

    @pytest.mark.schema
    def test_stop_loss_by_tier_complete(self):
        """All 3 tiers present in STOP_LOSS_BY_TIER."""
        assert set(STOP_LOSS_BY_TIER.keys()) == {1, 2, 3}
        for tier, pct in STOP_LOSS_BY_TIER.items():
            assert pct > 0.0, f"Tier {tier} stop loss must be positive"

    @pytest.mark.schema
    def test_time_horizons_complete(self):
        """3 horizons present with fraction and volatility_scale."""
        assert len(TIME_HORIZONS) == 3
        for horizon, data in TIME_HORIZONS.items():
            assert "fraction" in data
            assert "volatility_scale" in data
            assert data["fraction"] > 0.0
            assert data["volatility_scale"] > 0.0

    @pytest.mark.schema
    def test_standard_caveats_non_empty(self):
        """At least 3 caveats defined."""
        assert len(STANDARD_CAVEATS) >= 3
        for caveat in STANDARD_CAVEATS:
            assert isinstance(caveat, str)
            assert len(caveat) > 10


# ---------------------------------------------------------------------------
# TestTierAllocation
# ---------------------------------------------------------------------------

class TestTierAllocation:

    @pytest.mark.schema
    def test_valid_tier_allocation(self):
        """Create valid TierAllocation, verify it passes."""
        ta = _make_tier_allocation()
        assert ta.tier_1_pct == 40.0
        assert ta.tier_2_pct == 30.0
        assert ta.tier_3_pct == 20.0
        assert ta.cash_pct == 10.0

    @pytest.mark.schema
    def test_tier_allocation_rejects_bad_sum(self):
        """Percentages that don't sum to ~100% should fail."""
        with pytest.raises(ValueError, match="sum to"):
            _make_tier_allocation(
                tier_1_pct=50.0,
                tier_2_pct=50.0,
                tier_3_pct=50.0,
                cash_pct=10.0,
            )


# ---------------------------------------------------------------------------
# TestScenarioProjection
# ---------------------------------------------------------------------------

class TestScenarioProjection:

    @pytest.mark.schema
    def test_valid_scenario_projection(self):
        """Create valid ScenarioProjection for each scenario."""
        for name in SCENARIOS:
            sp = _make_scenario(name, 0.33)
            assert sp.scenario == name

    @pytest.mark.schema
    def test_invalid_scenario_name(self):
        """Scenario not in bull/base/bear should fail."""
        with pytest.raises(ValueError):
            ScenarioProjection(
                scenario="sideways",
                probability=0.25,
                tier_1_return_pct=10.0,
                tier_2_return_pct=8.0,
                tier_3_return_pct=5.0,
                portfolio_return_3m=2.0,
                portfolio_return_6m=4.0,
                portfolio_return_12m=8.0,
                portfolio_gain_12m=120_000.0,
                ending_value_12m=1_620_000.0,
                tier_1_contribution=4.0,
                tier_2_contribution=2.4,
                tier_3_contribution=1.0,
                reasoning="Sideways market scenario with limited directional movement across all tiers.",
            )


# ---------------------------------------------------------------------------
# TestPercentileRange
# ---------------------------------------------------------------------------

class TestPercentileRange:

    @pytest.mark.schema
    def test_valid_percentile_range(self):
        """Create ordered percentiles, passes."""
        pr = _make_percentile_range()
        assert pr.p10 < pr.p25 < pr.p50 < pr.p75 < pr.p90

    @pytest.mark.schema
    def test_unordered_percentile_range_fails(self):
        """p10 > p50 should fail validation."""
        with pytest.raises(ValueError, match="Percentiles not in order"):
            PercentileRange(
                p10=20.0,
                p25=15.0,
                p50=10.0,
                p75=18.0,
                p90=26.0,
                p10_dollar=1_800_000.0,
                p50_dollar=1_650_000.0,
                p90_dollar=1_890_000.0,
            )


# ---------------------------------------------------------------------------
# TestRiskMetrics
# ---------------------------------------------------------------------------

class TestRiskMetrics:

    @pytest.mark.schema
    def test_valid_risk_metrics(self):
        """Create valid RiskMetrics, passes."""
        rm = _make_risk_metrics()
        assert rm.max_drawdown_with_stops <= rm.max_drawdown_without_stops
        assert rm.projected_annual_volatility > 0

    @pytest.mark.schema
    def test_drawdown_ordering(self):
        """max_drawdown_with_stops > max_drawdown_without_stops should fail."""
        with pytest.raises(ValueError, match="exceeds max_drawdown_without_stops"):
            _make_risk_metrics(
                max_drawdown_with_stops=40.0,
                max_drawdown_without_stops=35.0,
            )


# ---------------------------------------------------------------------------
# TestReturnsProjectionOutput
# ---------------------------------------------------------------------------

class TestReturnsProjectionOutput:

    @pytest.mark.schema
    def test_valid_full_output(self):
        """Construct complete valid output, verify isinstance."""
        output = _make_full_output()
        assert isinstance(output, ReturnsProjectionOutput)
        assert output.portfolio_value == DEFAULT_PORTFOLIO_VALUE
        assert output.market_regime == "neutral"

    @pytest.mark.schema
    def test_exactly_3_scenarios(self):
        """Only 2 scenarios should fail."""
        with pytest.raises(ValueError):
            _make_full_output(
                scenarios=[
                    _make_scenario("bull", 0.50),
                    _make_scenario("base", 0.50),
                ],
            )

    @pytest.mark.schema
    def test_scenario_weights_sum_to_1(self):
        """Scenario probabilities that don't sum to ~1.0 should fail."""
        with pytest.raises(ValueError, match="probabilities sum to"):
            _make_full_output(
                scenarios=[
                    _make_scenario("bull", 0.50),
                    _make_scenario("base", 0.50),
                    _make_scenario("bear", 0.50),
                ],
            )

    @pytest.mark.schema
    def test_exactly_3_benchmarks(self):
        """Only 2 benchmarks should fail."""
        with pytest.raises(ValueError):
            _make_full_output(
                benchmark_comparisons=[
                    _make_benchmark("SPY", "S&P 500"),
                    _make_benchmark("DIA", "Dow Jones"),
                ],
            )

    @pytest.mark.schema
    def test_exactly_3_tier_mixes(self):
        """Only 2 tier mixes should fail."""
        with pytest.raises(ValueError):
            _make_full_output(
                tier_mix_comparison=[
                    _make_tier_mix("Aggressive"),
                    _make_tier_mix("Balanced"),
                ],
            )

    @pytest.mark.schema
    def test_caveats_non_empty(self):
        """Empty caveats list should fail."""
        with pytest.raises(ValueError):
            _make_full_output(caveats=[])

    @pytest.mark.schema
    def test_summary_min_length(self):
        """Summary shorter than 50 chars should fail."""
        with pytest.raises(ValueError):
            _make_full_output(summary="Too short.")

    @pytest.mark.schema
    def test_ending_value_math(self):
        """Verify start * (1 + return/100) approximately equals ending_value."""
        output = _make_full_output()
        for sp in output.scenarios:
            expected_ending = output.portfolio_value * (1 + sp.portfolio_return_12m / 100.0)
            assert abs(sp.ending_value_12m - expected_ending) < 1.0, (
                f"{sp.scenario}: ending_value_12m={sp.ending_value_12m} "
                f"!= expected {expected_ending:.2f}"
            )
