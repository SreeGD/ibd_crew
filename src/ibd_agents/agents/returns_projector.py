"""
Agent 07: Returns Projector
Portfolio Return Projection Specialist — IBD Momentum Framework v4.0

Receives PortfolioOutput, RotationDetectionOutput, RiskAssessment, AnalystOutput.
Produces ReturnsProjectionOutput with:
- Bull/base/bear scenario projections
- Benchmark comparisons (SPY/DIA/QQQ)
- Alpha decomposition
- Risk metrics (drawdowns, Sharpe, stop triggers)
- Tier mix alternatives
- Confidence intervals (p10/p25/p50/p75/p90)

This agent projects returns. It never picks stocks or makes buy/sell recommendations.
"""

from __future__ import annotations

import logging
import math
from datetime import date

try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    Agent = None  # type: ignore
    Task = None  # type: ignore

from ibd_agents.schemas.analyst_output import AnalystOutput
from ibd_agents.schemas.portfolio_output import PortfolioOutput
from ibd_agents.schemas.returns_projection_output import (
    BENCHMARK_NAMES,
    BENCHMARK_RETURNS,
    BENCHMARKS,
    DEFAULT_PORTFOLIO_VALUE,
    RISK_FREE_RATE,
    SCENARIOS,
    STANDARD_CAVEATS,
    TIER_HISTORICAL_RETURNS,
    AlphaAnalysis,
    AlphaSource,
    AssetTypeDecomposition,
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
from ibd_agents.schemas.risk_output import RiskAssessment
from ibd_agents.schemas.rotation_output import RotationDetectionOutput
from ibd_agents.tools.alpha_calculator import (
    compute_alpha,
    compute_expected_alpha,
    decompose_alpha,
    estimate_outperform_probability,
)
from ibd_agents.tools.benchmark_fetcher import (
    compute_benchmark_expected,
    get_benchmark_stats,
)
from ibd_agents.tools.confidence_builder import (
    build_confidence_intervals,
    build_percentile_range,
)
from ibd_agents.tools.drawdown_estimator import (
    compute_drawdown_with_stops,
    compute_drawdown_without_stops,
    estimate_stop_triggers,
)
from ibd_agents.tools.scenario_weighter import (
    compute_expected,
    get_scenario_weights,
)
from ibd_agents.tools.tier_return_calculator import (
    compute_blended_return,
    compute_tier_contribution,
    compute_tier_return,
    scale_to_horizon,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI Agent Builder
# ---------------------------------------------------------------------------

def build_returns_projector_agent() -> "Agent":
    """Create the Returns Projector Agent. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")

    from ibd_agents.tools.alpha_calculator import AlphaCalculatorTool
    from ibd_agents.tools.benchmark_fetcher import BenchmarkFetcherTool
    from ibd_agents.tools.confidence_builder import ConfidenceIntervalBuilderTool
    from ibd_agents.tools.drawdown_estimator import DrawdownEstimatorTool
    from ibd_agents.tools.scenario_weighter import ScenarioWeighterTool
    from ibd_agents.tools.tier_return_calculator import TierReturnCalculatorTool

    return Agent(
        role="Portfolio Return Projection Specialist",
        goal=(
            "Project portfolio returns across bull/base/bear scenarios. "
            "Compare against SPY/DIA/QQQ benchmarks. Decompose alpha sources. "
            "Compute risk metrics (drawdowns, Sharpe ratios, stop triggers). "
            "Build confidence intervals at 3m/6m/12m horizons."
        ),
        backstory=(
            "You are a quantitative analyst who projects portfolio returns using "
            "historical tier characteristics and scenario analysis. You never "
            "make buy/sell recommendations — you only project and compare."
        ),
        tools=[
            TierReturnCalculatorTool(),
            BenchmarkFetcherTool(),
            ScenarioWeighterTool(),
            AlphaCalculatorTool(),
            DrawdownEstimatorTool(),
            ConfidenceIntervalBuilderTool(),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=10,
        temperature=0.3,
    )


def build_returns_projector_task(
    agent: "Agent",
    portfolio_json: str = "",
    rotation_json: str = "",
) -> "Task":
    """Create the Returns Projection task. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Task(
        description=f"""Project portfolio returns across scenarios.

STEPS:
1. Extract tier allocations from portfolio output
2. Determine market regime from rotation output
3. Compute per-tier returns for bull/base/bear scenarios
4. Compute blended portfolio returns at 3m/6m/12m
5. Compare against SPY/DIA/QQQ benchmarks
6. Decompose alpha sources
7. Compute risk metrics (drawdowns, Sharpe, stops)
8. Build 3 alternative tier mixes
9. Build confidence intervals
10. Add standard caveats

Portfolio data:
{portfolio_json}

Rotation data:
{rotation_json}
""",
        expected_output=(
            "JSON with tier_allocation, scenarios (3), expected_return, "
            "benchmark_comparisons (3), alpha_analysis, risk_metrics, "
            "tier_mix_comparison (3), confidence_intervals, summary, caveats."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Deterministic Pipeline
# ---------------------------------------------------------------------------

def run_returns_projector_pipeline(
    portfolio_output: PortfolioOutput,
    rotation_output: RotationDetectionOutput,
    risk_output: RiskAssessment,
    analyst_output: AnalystOutput,
) -> ReturnsProjectionOutput:
    """
    Run the deterministic returns projection pipeline without LLM.

    Args:
        portfolio_output: Validated PortfolioOutput from Agent 05
        rotation_output: Validated RotationDetectionOutput from Agent 03
        risk_output: Validated RiskAssessment from Agent 06
        analyst_output: Validated AnalystOutput from Agent 02

    Returns:
        Validated ReturnsProjectionOutput
    """
    logger.info("[Agent 07] Running Returns Projection pipeline ...")

    portfolio_value = DEFAULT_PORTFOLIO_VALUE

    # --- Step 1: Extract tier allocations ---
    t1_pct = portfolio_output.tier_1.actual_pct
    t2_pct = portfolio_output.tier_2.actual_pct
    t3_pct = portfolio_output.tier_3.actual_pct
    cash_pct = portfolio_output.cash_pct
    tier_pcts = {1: t1_pct, 2: t2_pct, 3: t3_pct}

    tier_allocation = TierAllocation(
        tier_1_pct=t1_pct,
        tier_2_pct=t2_pct,
        tier_3_pct=t3_pct,
        cash_pct=cash_pct,
        tier_1_value=portfolio_value * t1_pct / 100.0,
        tier_2_value=portfolio_value * t2_pct / 100.0,
        tier_3_value=portfolio_value * t3_pct / 100.0,
        cash_value=portfolio_value * cash_pct / 100.0,
    )
    logger.info(
        f"[Agent 07] Tier allocation: T1={t1_pct}%, T2={t2_pct}%, "
        f"T3={t3_pct}%, Cash={cash_pct}%"
    )

    # --- Step 2: Get regime and scenario weights ---
    regime = rotation_output.market_regime.regime
    weights = get_scenario_weights(regime)
    logger.info(f"[Agent 07] Market regime: {regime}, weights: {weights}")

    # --- Step 3: Determine sector momentum ---
    sector_momentum = _determine_sector_momentum(rotation_output)
    logger.info(f"[Agent 07] Sector momentum: {sector_momentum}")

    # --- Step 4: Compute per-tier returns and build scenarios ---
    total_positions = portfolio_output.total_positions
    scenarios = []
    portfolio_returns_by_scenario = {}

    for scenario in SCENARIOS:
        # Per-tier returns with sector momentum
        t1_ret = compute_tier_return(1, scenario, sector_momentum)
        t2_ret = compute_tier_return(2, scenario, sector_momentum)
        t3_ret = compute_tier_return(3, scenario, sector_momentum)
        tier_returns = {1: t1_ret, 2: t2_ret, 3: t3_ret}

        # Blended annual return
        blended_12m = compute_blended_return(tier_pcts, tier_returns, cash_pct)
        portfolio_returns_by_scenario[scenario] = blended_12m

        # Scale to horizons
        # Compute blended std for the portfolio
        blended_std = _compute_blended_std(tier_pcts, scenario, cash_pct)
        ret_3m, _ = scale_to_horizon(blended_12m, blended_std, "3_month")
        ret_6m, _ = scale_to_horizon(blended_12m, blended_std, "6_month")

        # Tier contributions
        t1_contrib = compute_tier_contribution(t1_pct, t1_ret)
        t2_contrib = compute_tier_contribution(t2_pct, t2_ret)
        t3_contrib = compute_tier_contribution(t3_pct, t3_ret)

        # Dollar impact
        gain_12m = portfolio_value * blended_12m / 100.0
        ending_12m = portfolio_value + gain_12m

        scenarios.append(ScenarioProjection(
            scenario=scenario,
            probability=weights[scenario],
            tier_1_return_pct=round(t1_ret, 2),
            tier_2_return_pct=round(t2_ret, 2),
            tier_3_return_pct=round(t3_ret, 2),
            portfolio_return_3m=round(ret_3m, 2),
            portfolio_return_6m=round(ret_6m, 2),
            portfolio_return_12m=round(blended_12m, 2),
            portfolio_gain_12m=round(gain_12m, 2),
            ending_value_12m=round(ending_12m, 2),
            tier_1_contribution=round(t1_contrib, 2),
            tier_2_contribution=round(t2_contrib, 2),
            tier_3_contribution=round(t3_contrib, 2),
            reasoning=_scenario_reasoning(scenario, blended_12m, regime),
        ))

    logger.info(
        f"[Agent 07] Scenario returns (12m): "
        f"bull={portfolio_returns_by_scenario['bull']:.1f}%, "
        f"base={portfolio_returns_by_scenario['base']:.1f}%, "
        f"bear={portfolio_returns_by_scenario['bear']:.1f}%"
    )

    # --- Step 5: Expected return (probability-weighted) ---
    expected_3m = compute_expected(
        {s: sc.portfolio_return_3m for s, sc in zip(SCENARIOS, scenarios)},
        weights,
    )
    expected_6m = compute_expected(
        {s: sc.portfolio_return_6m for s, sc in zip(SCENARIOS, scenarios)},
        weights,
    )
    expected_12m = compute_expected(
        portfolio_returns_by_scenario,
        weights,
    )
    expected_return = ExpectedReturn(
        expected_3m=round(expected_3m, 2),
        expected_6m=round(expected_6m, 2),
        expected_12m=round(expected_12m, 2),
    )
    logger.info(
        f"[Agent 07] Expected returns: 3m={expected_3m:.2f}%, "
        f"6m={expected_6m:.2f}%, 12m={expected_12m:.2f}%"
    )

    # --- Step 6: Benchmark comparisons ---
    benchmark_comparisons = []
    for sym in BENCHMARKS:
        b_expected = compute_benchmark_expected(sym, weights)

        # Per-scenario benchmark returns at 12m
        b_bull = get_benchmark_stats(sym, "bull")["mean"]
        b_base = get_benchmark_stats(sym, "base")["mean"]
        b_bear = get_benchmark_stats(sym, "bear")["mean"]

        # Scale benchmarks to horizons
        b_std_base = get_benchmark_stats(sym, "base")["std"]
        b_3m, _ = scale_to_horizon(b_expected, b_std_base, "3_month")
        b_6m, _ = scale_to_horizon(b_expected, b_std_base, "6_month")

        # Alpha per scenario (12m)
        alpha_bull = compute_alpha(
            portfolio_returns_by_scenario["bull"], b_bull
        )
        alpha_base = compute_alpha(
            portfolio_returns_by_scenario["base"], b_base
        )
        alpha_bear = compute_alpha(
            portfolio_returns_by_scenario["bear"], b_bear
        )

        # Expected alpha
        exp_alpha = compute_expected_alpha(
            portfolio_returns_by_scenario,
            {"bull": b_bull, "base": b_base, "bear": b_bear},
            weights,
        )

        # Outperform probability
        p_std = _compute_blended_std(tier_pcts, "base", cash_pct)
        outperform_prob = estimate_outperform_probability(
            expected_12m, p_std, b_expected, b_std_base
        )

        benchmark_comparisons.append(BenchmarkComparison(
            symbol=sym,
            name=BENCHMARK_NAMES[sym],
            benchmark_return_3m=round(b_3m, 2),
            benchmark_return_6m=round(b_6m, 2),
            benchmark_return_12m=round(b_expected, 2),
            alpha_bull_12m=round(alpha_bull, 2),
            alpha_base_12m=round(alpha_base, 2),
            alpha_bear_12m=round(alpha_bear, 2),
            expected_alpha_12m=round(exp_alpha, 2),
            outperform_probability=round(outperform_prob, 4),
        ))

    logger.info(
        f"[Agent 07] Alpha vs SPY={benchmark_comparisons[0].expected_alpha_12m:.2f}%, "
        f"DIA={benchmark_comparisons[1].expected_alpha_12m:.2f}%, "
        f"QQQ={benchmark_comparisons[2].expected_alpha_12m:.2f}%"
    )

    # --- Step 7: Alpha analysis ---
    # Compute tier contributions for base scenario
    base_t1_ret = compute_tier_return(1, "base", sector_momentum)
    base_t2_ret = compute_tier_return(2, "base", sector_momentum)
    base_t3_ret = compute_tier_return(3, "base", sector_momentum)
    tier_contribs = {
        1: compute_tier_contribution(t1_pct, base_t1_ret),
        2: compute_tier_contribution(t2_pct, base_t2_ret),
        3: compute_tier_contribution(t3_pct, base_t3_ret),
    }

    spy_base_return = get_benchmark_stats("SPY", "base")["mean"]
    sector_boost = _compute_sector_boost(sector_momentum, portfolio_returns_by_scenario["base"])
    raw_sources = decompose_alpha(tier_contribs, spy_base_return, sector_boost)

    # Split into sources and drags
    alpha_sources = []
    alpha_drags = []
    for src in raw_sources:
        alpha_obj = AlphaSource(
            source=src["source"],
            contribution_pct=src["contribution_pct"],
            confidence=src["confidence"],
            reasoning=src["reasoning"],
        )
        if src["contribution_pct"] >= 0:
            alpha_sources.append(alpha_obj)
        else:
            alpha_drags.append(alpha_obj)

    alpha_analysis = AlphaAnalysis(
        primary_alpha_sources=alpha_sources,
        primary_alpha_drags=alpha_drags,
        net_expected_alpha_vs_spy=round(benchmark_comparisons[0].expected_alpha_12m, 2),
        net_expected_alpha_vs_dia=round(benchmark_comparisons[1].expected_alpha_12m, 2),
        net_expected_alpha_vs_qqq=round(benchmark_comparisons[2].expected_alpha_12m, 2),
        alpha_persistence_note=(
            "Alpha from IBD momentum selection historically persists for 6-12 months "
            "before mean reversion; periodic rebalancing required to maintain edge"
        ),
    )

    # --- Step 8: Risk metrics ---
    dd_with = compute_drawdown_with_stops(tier_pcts)
    dd_without = compute_drawdown_without_stops(tier_pcts)
    dd_dollar = portfolio_value * dd_with / 100.0

    p_vol = _compute_blended_std(tier_pcts, "base", cash_pct)
    spy_vol = get_benchmark_stats("SPY", "base")["std"]

    # Sharpe ratios: (return - risk_free) / volatility
    base_ret = portfolio_returns_by_scenario["base"]
    bull_ret = portfolio_returns_by_scenario["bull"]
    bear_ret = portfolio_returns_by_scenario["bear"]
    bull_vol = _compute_blended_std(tier_pcts, "bull", cash_pct)
    bear_vol = _compute_blended_std(tier_pcts, "bear", cash_pct)

    sharpe_bull = (bull_ret - RISK_FREE_RATE) / bull_vol if bull_vol > 0 else 0.0
    sharpe_base = (base_ret - RISK_FREE_RATE) / p_vol if p_vol > 0 else 0.0
    sharpe_bear = (bear_ret - RISK_FREE_RATE) / bear_vol if bear_vol > 0 else 0.0
    spy_sharpe = (spy_base_return - RISK_FREE_RATE) / spy_vol if spy_vol > 0 else 0.0
    return_per_risk = base_ret / p_vol if p_vol > 0 else 0.0

    expected_stops, stop_prob = estimate_stop_triggers(total_positions, "base")

    risk_metrics = RiskMetrics(
        max_drawdown_with_stops=dd_with,
        max_drawdown_without_stops=dd_without,
        max_drawdown_dollar=round(dd_dollar, 2),
        projected_annual_volatility=round(p_vol, 2),
        spy_annual_volatility=round(spy_vol, 2),
        portfolio_sharpe_bull=round(sharpe_bull, 2),
        portfolio_sharpe_base=round(sharpe_base, 2),
        portfolio_sharpe_bear=round(sharpe_bear, 2),
        spy_sharpe_base=round(spy_sharpe, 2),
        return_per_unit_risk=round(return_per_risk, 2),
        probability_any_stop_triggers_12m=round(stop_prob, 2),
        expected_stops_triggered_12m=expected_stops,
        whipsaw_risk_note=(
            "Trailing stops in volatile markets may trigger prematurely, "
            "locking in temporary losses during recovery periods"
        ),
    )
    logger.info(
        f"[Agent 07] Risk: DD(stops)={dd_with}%, DD(no stops)={dd_without}%, "
        f"Sharpe(base)={sharpe_base:.2f}"
    )

    # --- Step 9: Tier mix alternatives ---
    tier_mix_comparison = _build_tier_mix_alternatives(
        tier_pcts, cash_pct, weights, sector_momentum
    )

    # --- Step 10: Confidence intervals ---
    # Use expected (probability-weighted) return and blended std
    returns_by_horizon = {}
    stds_by_horizon = {}
    for horizon_key in ("3_month", "6_month", "12_month"):
        r, s = scale_to_horizon(expected_12m, p_vol, horizon_key)
        returns_by_horizon[horizon_key] = r
        stds_by_horizon[horizon_key] = s

    ci_raw = build_confidence_intervals(returns_by_horizon, stds_by_horizon, portfolio_value)
    confidence_intervals = ConfidenceIntervals(
        horizon_3m=PercentileRange(**ci_raw["horizon_3m"]),
        horizon_6m=PercentileRange(**ci_raw["horizon_6m"]),
        horizon_12m=PercentileRange(**ci_raw["horizon_12m"]),
    )

    # --- Step 11: Asset type decomposition ---
    asset_decomposition = _compute_asset_decomposition(portfolio_output)
    logger.info(
        f"[Agent 07] Asset decomposition: "
        + ", ".join(
            f"T{d.tier} stocks={d.stock_count}/ETFs={d.etf_count}"
            for d in asset_decomposition
        )
    )

    # --- Step 12: Summary and caveats ---
    summary = _build_summary(expected_return, benchmark_comparisons, risk_metrics, regime)

    output = ReturnsProjectionOutput(
        portfolio_value=portfolio_value,
        analysis_date=date.today().isoformat(),
        market_regime=regime,
        regime_source="Rotation Detector Agent 03",
        tier_allocation=tier_allocation,
        scenarios=scenarios,
        expected_return=expected_return,
        benchmark_comparisons=benchmark_comparisons,
        alpha_analysis=alpha_analysis,
        risk_metrics=risk_metrics,
        tier_mix_comparison=tier_mix_comparison,
        confidence_intervals=confidence_intervals,
        asset_type_decomposition=asset_decomposition,
        summary=summary,
        caveats=list(STANDARD_CAVEATS),
    )

    logger.info(
        f"[Agent 07] Done — expected 12m return: {expected_12m:.1f}%, "
        f"Sharpe: {sharpe_base:.2f}, DD(stops): {dd_with}%"
    )

    return output


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _compute_asset_decomposition(
    portfolio: PortfolioOutput,
) -> list[AssetTypeDecomposition]:
    """Compute ETF vs stock return decomposition per tier.

    For each tier portfolio:
    - Separate positions by asset_type ("stock" vs "etf")
    - Sum target_pct for each type
    - Count each type
    - Estimate volatility: base std for stocks, 70% of that for ETFs
    """
    result: list[AssetTypeDecomposition] = []

    for tier_portfolio in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]:
        tier_num = tier_portfolio.tier
        stock_pct = 0.0
        etf_pct = 0.0
        stock_count = 0
        etf_count = 0

        for p in tier_portfolio.positions:
            if p.asset_type == "etf":
                etf_pct += p.target_pct
                etf_count += 1
            else:
                stock_pct += p.target_pct
                stock_count += 1

        # Estimate volatility using base scenario std
        base_std = TIER_HISTORICAL_RETURNS[tier_num]["base"]["std"]
        stock_vol = round(base_std, 2) if stock_count > 0 else None
        etf_vol = round(base_std * 0.70, 2) if etf_count > 0 else None

        result.append(AssetTypeDecomposition(
            tier=tier_num,
            stock_contribution_pct=round(stock_pct, 2),
            etf_contribution_pct=round(etf_pct, 2),
            stock_count=stock_count,
            etf_count=etf_count,
            stock_avg_volatility=stock_vol,
            etf_avg_volatility=etf_vol,
        ))

    return result


def _determine_sector_momentum(rotation_output: RotationDetectionOutput) -> str:
    """Determine sector momentum from rotation output."""
    if rotation_output.destination_sectors:
        return "leading"
    elif rotation_output.source_sectors:
        return "declining"
    elif rotation_output.verdict.value == "NONE":
        return "neutral"
    else:
        return "improving"


def _compute_blended_std(
    tier_pcts: dict[int, float],
    scenario: str,
    cash_pct: float,
) -> float:
    """Compute blended portfolio standard deviation for a scenario."""
    total_var = sum(
        (tier_pcts[t] / 100.0) ** 2
        * TIER_HISTORICAL_RETURNS[t][scenario]["std"] ** 2
        for t in (1, 2, 3)
        if t in tier_pcts
    )
    return math.sqrt(total_var)


def _compute_sector_boost(sector_momentum: str, base_return: float) -> float:
    """Compute sector momentum boost vs neutral base."""
    from ibd_agents.schemas.returns_projection_output import SECTOR_MOMENTUM_MULTIPLIER
    multiplier = SECTOR_MOMENTUM_MULTIPLIER.get(sector_momentum, 1.0)
    if multiplier == 1.0:
        return 0.0
    # Approximate boost: (multiplier - 1) * base_return
    return round((multiplier - 1.0) * base_return, 2)


def _build_tier_mix_alternatives(
    current_pcts: dict[int, float],
    current_cash: float,
    weights: dict[str, float],
    sector_momentum: str,
) -> list[TierMixScenario]:
    """Build 3 alternative tier mix scenarios."""
    alternatives = [
        {"label": "Aggressive", "t1": 50.0, "t2": 35.0, "t3": 13.0, "cash": 2.0},
        {"label": "Current (Recommended)", "t1": current_pcts[1], "t2": current_pcts[2], "t3": current_pcts[3], "cash": current_cash},
        {"label": "Conservative", "t1": 25.0, "t2": 35.0, "t3": 35.0, "cash": 5.0},
    ]

    result = []
    for alt in alternatives:
        pcts = {1: alt["t1"], 2: alt["t2"], 3: alt["t3"]}

        # Compute returns for each scenario
        scenario_returns = {}
        for scenario in SCENARIOS:
            tier_rets = {t: compute_tier_return(t, scenario, sector_momentum) for t in (1, 2, 3)}
            scenario_returns[scenario] = compute_blended_return(pcts, tier_rets, alt["cash"])

        expected_ret = compute_expected(scenario_returns, weights)
        dd = compute_drawdown_with_stops(pcts)
        vol = _compute_blended_std(pcts, "base", alt["cash"])
        sharpe = (scenario_returns["base"] - RISK_FREE_RATE) / vol if vol > 0 else 0.0

        comparison = _tier_mix_comparison_note(alt["label"], expected_ret, dd, sharpe)

        result.append(TierMixScenario(
            label=alt["label"],
            tier_1_pct=alt["t1"],
            tier_2_pct=alt["t2"],
            tier_3_pct=alt["t3"],
            cash_pct=alt["cash"],
            expected_return_12m=round(expected_ret, 2),
            bull_return_12m=round(scenario_returns["bull"], 2),
            bear_return_12m=round(scenario_returns["bear"], 2),
            max_drawdown_with_stops=dd,
            sharpe_ratio_base=round(sharpe, 2),
            comparison_note=comparison,
        ))

    return result


def _tier_mix_comparison_note(label: str, ret: float, dd: float, sharpe: float) -> str:
    """Generate comparison note for a tier mix."""
    if "Aggressive" in label:
        return (
            f"Higher T1 allocation targets {ret:.1f}% return but increases "
            f"max drawdown to {dd:.1f}% and raises volatility"
        )
    elif "Conservative" in label:
        return (
            f"Higher T3 defensive allocation reduces drawdown to {dd:.1f}% "
            f"but lowers expected return to {ret:.1f}%"
        )
    else:
        return (
            f"Recommended allocation balances growth and protection with "
            f"{ret:.1f}% expected return and {dd:.1f}% max drawdown"
        )


def _scenario_reasoning(scenario: str, ret: float, regime: str) -> str:
    """Generate reasoning for a scenario projection."""
    if scenario == "bull":
        return (
            f"Bull scenario projects {ret:.1f}% portfolio return driven by T1 momentum "
            f"outperformance in a favorable market environment (current regime: {regime})"
        )
    elif scenario == "base":
        return (
            f"Base scenario projects {ret:.1f}% portfolio return under normal market conditions, "
            f"with IBD-rated stocks expected to outperform benchmarks modestly (regime: {regime})"
        )
    else:
        return (
            f"Bear scenario projects {ret:.1f}% portfolio return in adverse conditions, "
            f"with trailing stops limiting losses and T3 defensive positions providing buffer (regime: {regime})"
        )


def _build_summary(
    expected_return: ExpectedReturn,
    benchmarks: list[BenchmarkComparison],
    risk: RiskMetrics,
    regime: str,
) -> str:
    """Build summary string (>= 50 chars)."""
    spy_alpha = next((b.expected_alpha_12m for b in benchmarks if b.symbol == "SPY"), 0)
    parts = [
        f"Returns projection ({regime} regime): "
        f"expected 12-month return {expected_return.expected_12m:.1f}%.",
        f"Alpha vs SPY: {spy_alpha:+.1f}%.",
        f"Max drawdown with stops: {risk.max_drawdown_with_stops:.1f}%.",
        f"Sharpe ratio (base): {risk.portfolio_sharpe_base:.2f}.",
    ]
    summary = " ".join(parts)
    if len(summary) < 50:
        summary += " Projections based on IBD tier historical characteristics."
    return summary
