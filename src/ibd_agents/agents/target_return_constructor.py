"""
Agent 14: Target Return Portfolio Constructor
IBD Momentum Investment Framework v4.0

Reverse-engineers a portfolio targeting a specific annualized return
(default 30%) by optimizing tier mix, selecting/sizing positions,
running Monte Carlo simulations, and providing probability assessments
with alternative portfolios.

This agent constructs portfolios. It never guarantees returns —
it presents probability-weighted scenarios.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    Agent = None  # type: ignore
    Task = None  # type: ignore

from ibd_agents.schemas.analyst_output import AnalystOutput
from ibd_agents.schemas.returns_projection_output import (
    BENCHMARK_RETURNS,
    RISK_FREE_RATE,
    TIER_HISTORICAL_RETURNS,
)
from ibd_agents.schemas.risk_output import RiskAssessment
from ibd_agents.schemas.rotation_output import RotationDetectionOutput
from ibd_agents.schemas.strategy_output import SectorStrategyOutput
from ibd_agents.schemas.returns_projection_output import ReturnsProjectionOutput
from ibd_agents.schemas.target_return_output import (
    ACHIEVABILITY_RATINGS,
    DEFAULT_FRICTION,
    DEFAULT_HORIZON_MONTHS,
    DEFAULT_TARGET_RETURN,
    DEFAULT_TOTAL_CAPITAL,
    MAX_PROB_ACHIEVE,
    MAX_SECTOR_CONCENTRATION_PCT,
    MAX_SINGLE_POSITION_PCT,
    REGIME_GUIDANCE,
    STANDARD_DISCLAIMER,
    AlternativePortfolio,
    ProbabilityAssessment,
    RiskDisclosure,
    ScenarioAnalysis,
    SectorWeight,
    TargetPosition,
    TargetReturnOutput,
    TargetReturnScenario,
    TargetTierAllocation,
)
from ibd_agents.tools.candidate_ranker import rank_candidates, select_positions
from ibd_agents.tools.constraint_validator_target import validate_constraints
from ibd_agents.tools.drawdown_estimator import compute_drawdown_with_stops
from ibd_agents.tools.monte_carlo_engine import (
    compute_benchmark_beat_probability,
    run_monte_carlo,
)
from ibd_agents.tools.position_sizer import compute_trailing_stop
from ibd_agents.tools.scenario_weighter import get_scenario_weights
from ibd_agents.tools.tier_mix_optimizer import get_regime_guidance, solve_tier_mix
from ibd_agents.tools.tier_return_calculator import compute_blended_return, compute_tier_return

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI Agent Builder
# ---------------------------------------------------------------------------

def build_target_return_agent() -> "Agent":
    """Create the Target Return Constructor Agent. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")

    from ibd_agents.tools.candidate_ranker import CandidateRankerTool
    from ibd_agents.tools.constraint_validator_target import ConstraintValidatorTool
    from ibd_agents.tools.monte_carlo_engine import MonteCarloTool
    from ibd_agents.tools.tier_mix_optimizer import TierMixOptimizerTool
    from ibd_agents.tools.transition_planner import TransitionPlannerTool

    return Agent(
        role="Target Return Portfolio Architect",
        goal=(
            "Reverse-engineer a portfolio that maximizes the probability of "
            "achieving a target annualized return by determining the optimal "
            "tier mix, position count, sector weights, and individual stock "
            "allocations, while respecting all risk constraints."
        ),
        backstory=(
            "You are a quantitative portfolio architect with 15 years of "
            "experience in goal-based portfolio construction. You start with "
            "a return TARGET and engineer backwards to determine what portfolio "
            "composition gives the highest probability of getting there. "
            "You NEVER guarantee returns. You present probability-weighted "
            "scenarios and clearly communicate what must go right AND wrong."
        ),
        tools=[
            TierMixOptimizerTool(),
            CandidateRankerTool(),
            MonteCarloTool(),
            ConstraintValidatorTool(),
            TransitionPlannerTool(),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=10,
        temperature=0.5,
    )


def build_target_return_task(
    agent: "Agent",
    target_return_pct: float = 30.0,
    regime: str = "bull",
) -> "Task":
    """Create the Target Return Construction task. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Task(
        description=f"""Construct a portfolio targeting {target_return_pct}% annualized returns.

PROCESS:
1. Assess achievability given current {regime} regime
2. Determine optimal tier mix (T1/T2/T3)
3. Select sectors aligned with momentum leaders
4. Pick individual stocks from tier-rated universe
5. Size positions with appropriate concentration
6. Run Monte Carlo simulation for probability assessment
7. Validate against all risk constraints
8. Generate 2-3 alternative portfolios
9. Write construction rationale

CRITICAL: Be HONEST about probability. If {target_return_pct}% is a stretch,
say so clearly and present a more achievable alternative prominently.
""",
        expected_output=(
            "A complete TargetReturnOutput with positions, probability assessment, "
            "scenario analysis, alternatives, risk disclosure, and rationale."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Deterministic Pipeline
# ---------------------------------------------------------------------------

def run_target_return_pipeline(
    analyst_output: AnalystOutput,
    rotation_output: RotationDetectionOutput,
    strategy_output: SectorStrategyOutput,
    risk_output: RiskAssessment,
    returns_output: ReturnsProjectionOutput,
    target_return_pct: float = DEFAULT_TARGET_RETURN,
    total_capital: float = DEFAULT_TOTAL_CAPITAL,
    time_horizon_months: int = DEFAULT_HORIZON_MONTHS,
) -> TargetReturnOutput:
    """
    Run the deterministic target return portfolio construction pipeline.

    Args:
        analyst_output: Validated AnalystOutput from Agent 02.
        rotation_output: Validated RotationDetectionOutput from Agent 03.
        strategy_output: Validated SectorStrategyOutput from Agent 04.
        risk_output: Validated RiskAssessment from Agent 06.
        returns_output: Validated ReturnsProjectionOutput from Agent 07.
        target_return_pct: Target annualized return percentage.
        total_capital: Total investable capital in USD.
        time_horizon_months: Investment horizon in months.

    Returns:
        Validated TargetReturnOutput.
    """
    logger.info(
        f"[Agent 14] Running Target Return Constructor: "
        f"target={target_return_pct}%, capital=${total_capital:,.0f}, "
        f"horizon={time_horizon_months}mo"
    )

    # --- Step 1: Extract market regime ---
    regime = rotation_output.market_regime.regime
    logger.info(f"[Agent 14] Market regime: {regime}")

    # --- Step 2: Get regime guidance ---
    guidance = get_regime_guidance(regime)
    achievability = guidance["achievability"]
    pos_range = guidance["position_count_range"]
    cash_range = guidance["cash_reserve_range"]
    logger.info(
        f"[Agent 14] Regime guidance: achievability={achievability}, "
        f"positions={pos_range}, cash={cash_range}"
    )

    # --- Step 3: Determine sector momentum map ---
    sector_momentum_map = _build_sector_momentum_map(strategy_output)
    logger.info(f"[Agent 14] Sector momentum: {len(sector_momentum_map)} sectors mapped")

    # Determine overall sector momentum for tier mix optimizer
    if rotation_output.destination_sectors:
        sector_momentum = "leading"
    elif rotation_output.source_sectors:
        sector_momentum = "declining"
    elif rotation_output.verdict.value == "NONE":
        sector_momentum = "neutral"
    else:
        sector_momentum = "improving"

    # --- Step 4: Solve tier mix ---
    cash_pct = (cash_range[0] + cash_range[1]) / 2.0
    solutions = solve_tier_mix(
        target_return=target_return_pct,
        regime=regime,
        friction=DEFAULT_FRICTION,
        sector_momentum=sector_momentum,
        cash_pct=cash_pct,
    )

    # --- Step 5: Select best tier mix ---
    best = solutions[0]
    t1_pct = best["w1"]
    t2_pct = best["w2"]
    t3_pct = best["w3"]
    logger.info(
        f"[Agent 14] Best tier mix: T1={t1_pct:.0f}%, T2={t2_pct:.0f}%, "
        f"T3={t3_pct:.0f}%, expected={best['expected_return']:.1f}%, "
        f"prob={best['prob_achieve']:.2%}"
    )

    tier_allocation = TargetTierAllocation(
        t1_momentum_pct=t1_pct,
        t2_quality_growth_pct=t2_pct,
        t3_defensive_pct=t3_pct,
        rationale=(
            f"T1={t1_pct:.0f}%/T2={t2_pct:.0f}%/T3={t3_pct:.0f}% tier mix optimized for "
            f"{target_return_pct}% target in {regime} regime. Expected return "
            f"{best['expected_return']:.1f}% with {best['prob_achieve']:.0%} probability "
            f"of achieving target."
        ),
    )

    # --- Step 6: Determine position count per tier ---
    target_pos_count = (pos_range[0] + pos_range[1]) // 2
    tier_counts = _allocate_position_counts(t1_pct, t2_pct, t3_pct, target_pos_count)

    # --- Step 7: Rank candidates and select positions per tier ---
    stock_metrics = _build_stock_metrics(analyst_output)
    all_positions = []
    for tier in (1, 2, 3):
        count = tier_counts[tier]
        if count <= 0:
            continue

        tier_weight = {1: t1_pct, 2: t2_pct, 3: t3_pct}[tier]
        candidates = rank_candidates(
            analyst_output.rated_stocks,
            tier=tier,
            sector_momentum_map=sector_momentum_map,
            stock_metrics=stock_metrics,
        )

        if not candidates:
            logger.warning(f"[Agent 14] No candidates for tier {tier}")
            continue

        sized = select_positions(
            candidates,
            tier_weight_pct=tier_weight,
            position_count=count,
            total_capital=total_capital,
            total_positions=target_pos_count,
        )

        all_positions.extend(sized)

    logger.info(f"[Agent 14] Selected {len(all_positions)} positions across tiers")

    # --- Step 8: Compute stop losses (prefer Agent 06 LLM recommendations) ---
    for pos in all_positions:
        tier = pos["tier"]
        stop_pct = _get_stop_for_symbol(pos["ticker"], tier, risk_output)
        entry_price = pos["target_entry_price"]
        pos["stop_loss_pct"] = stop_pct
        pos["stop_loss_price"] = round(entry_price * (1 - stop_pct / 100.0), 2)

    # --- Step 9: Normalize allocations ---
    # Ensure positions + cash = 100%
    pos_total = sum(p["allocation_pct"] for p in all_positions)
    invested_target = 100.0 - cash_pct
    if pos_total > 0 and abs(pos_total - invested_target) > 1.0:
        scale = invested_target / pos_total
        for pos in all_positions:
            pos["allocation_pct"] = round(pos["allocation_pct"] * scale, 2)
            pos["dollar_amount"] = round(total_capital * pos["allocation_pct"] / 100.0, 2)
            est_price = pos["target_entry_price"]
            pos["shares"] = max(1, int(pos["dollar_amount"] / est_price))

    # Ensure minimum 1.0% allocation per position (schema constraint)
    for pos in all_positions:
        if pos["allocation_pct"] < 1.0:
            pos["allocation_pct"] = 1.0
            pos["dollar_amount"] = round(total_capital * 1.0 / 100.0, 2)
            est_price = pos["target_entry_price"]
            pos["shares"] = max(1, int(pos["dollar_amount"] / est_price))

    # --- Step 9a: Enforce sector concentration limits ---
    for _ in range(3):  # iterate to converge
        sector_allocs: dict[str, float] = {}
        for pos in all_positions:
            sector_allocs[pos["sector"]] = sector_allocs.get(pos["sector"], 0) + pos["allocation_pct"]

        excess_freed = 0.0
        for sector, total in sector_allocs.items():
            if total > MAX_SECTOR_CONCENTRATION_PCT:
                scale_down = MAX_SECTOR_CONCENTRATION_PCT / total
                for pos in all_positions:
                    if pos["sector"] == sector:
                        old_alloc = pos["allocation_pct"]
                        new_alloc = max(1.0, round(old_alloc * scale_down, 2))
                        excess_freed += old_alloc - new_alloc
                        pos["allocation_pct"] = new_alloc
                        pos["dollar_amount"] = round(total_capital * new_alloc / 100.0, 2)
                        est_price = pos["target_entry_price"]
                        pos["shares"] = max(1, int(pos["dollar_amount"] / est_price))
                logger.info(
                    f"[Agent 14] Scaled sector {sector} from {total:.1f}% to "
                    f"~{MAX_SECTOR_CONCENTRATION_PCT:.0f}%"
                )

        # Redistribute freed allocation to non-capped sectors (spread to top positions)
        if excess_freed > 0:
            non_capped = [
                p for p in all_positions
                if sector_allocs.get(p["sector"], 0) <= MAX_SECTOR_CONCENTRATION_PCT
                and p["allocation_pct"] < MAX_SINGLE_POSITION_PCT
            ]
            if non_capped:
                per_pos = excess_freed / len(non_capped)
                for pos in non_capped:
                    add = min(per_pos, MAX_SINGLE_POSITION_PCT - pos["allocation_pct"])
                    pos["allocation_pct"] = round(pos["allocation_pct"] + add, 2)
                    pos["dollar_amount"] = round(total_capital * pos["allocation_pct"] / 100.0, 2)
                    est_price = pos["target_entry_price"]
                    pos["shares"] = max(1, int(pos["dollar_amount"] / est_price))

    # --- Step 10: Compute expected return contributions ---
    weights = get_scenario_weights(regime)
    for pos in all_positions:
        tier = pos["tier"]
        alloc = pos["allocation_pct"]
        # Expected return contribution = allocation * expected tier return (weighted)
        tier_expected = sum(
            weights[s] * compute_tier_return(tier, s, sector_momentum)
            for s in ("bull", "base", "bear")
        )
        contribution = alloc / 100.0 * tier_expected
        pos["expected_return_contribution_pct"] = round(contribution, 2)

    # Scale contributions to match target
    contrib_total = sum(p["expected_return_contribution_pct"] for p in all_positions)
    if contrib_total > 0 and abs(contrib_total - target_return_pct) > 0.5:
        scale = target_return_pct / contrib_total
        for pos in all_positions:
            pos["expected_return_contribution_pct"] = round(
                pos["expected_return_contribution_pct"] * scale, 2,
            )

    # --- Step 11: Add selection rationale ---
    for pos in all_positions:
        pos["selection_rationale"] = _build_selection_rationale(pos, sector_momentum_map)

    # --- Step 11.5: LLM selection rationale enrichment ---
    selection_source = "template"
    try:
        from ibd_agents.tools.target_return_enrichment import enrich_selection_rationale_llm
        pos_inputs = [
            {
                "ticker": p["ticker"],
                "company_name": p["company_name"],
                "tier": p["tier"],
                "sector": p["sector"],
                "composite_score": p["composite_score"],
                "rs_rating": p["rs_rating"],
                "eps_rating": p["eps_rating"],
                "conviction_level": p.get("conviction_level", "MEDIUM"),
                "allocation_pct": p["allocation_pct"],
                "sector_momentum": sector_momentum_map.get(p["sector"], "neutral"),
                "multi_source_count": p.get("multi_source_count", 0),
            }
            for p in all_positions
        ]
        enhanced = enrich_selection_rationale_llm(pos_inputs)
        if enhanced:
            for pos in all_positions:
                if pos["ticker"] in enhanced:
                    pos["selection_rationale"] = enhanced[pos["ticker"]]
            selection_source = "llm"
            logger.info(
                f"[Agent 14] LLM selection rationale: "
                f"{len(enhanced)}/{len(all_positions)} enriched"
            )
    except Exception as e:
        logger.warning(f"[Agent 14] LLM selection rationale failed: {e}")

    # --- Step 12: Validate constraints ---
    validation = validate_constraints(
        all_positions, cash_pct,
        max_drawdown_pct=-compute_drawdown_with_stops({1: t1_pct, 2: t2_pct, 3: t3_pct}),
        prob_achieve_target=best["prob_achieve"],
        t1_allocation_pct=t1_pct,
    )
    if not validation["passed"]:
        logger.warning(
            f"[Agent 14] Constraint violations: {validation['violations']}"
        )
    for w in validation["warnings"]:
        logger.info(f"[Agent 14] Warning: {w}")

    # --- Step 13: Run Monte Carlo ---
    volatility_map = _build_volatility_map(analyst_output)
    mc_positions = [
        {"tier": p["tier"], "allocation_pct": p["allocation_pct"],
         "stop_loss_pct": p["stop_loss_pct"],
         "volatility_pct": volatility_map.get(p["ticker"])}
        for p in all_positions
    ]
    mc_result = run_monte_carlo(
        mc_positions,
        tier_pcts={1: t1_pct, 2: t2_pct, 3: t3_pct},
        regime=regime,
        target_return=target_return_pct,
    )

    # --- Step 14: Compute benchmark probabilities ---
    prob_beat_spy = compute_benchmark_beat_probability(
        mc_result["returns_distribution"], "SPY", regime,
    )
    prob_beat_qqq = compute_benchmark_beat_probability(
        mc_result["returns_distribution"], "QQQ", regime,
    )
    prob_beat_dow = compute_benchmark_beat_probability(
        mc_result["returns_distribution"], "DIA", regime,
    )

    probability_assessment = ProbabilityAssessment(
        prob_achieve_target=mc_result["prob_achieve_target"],
        prob_positive_return=mc_result["prob_positive_return"],
        prob_beat_sp500=prob_beat_spy,
        prob_beat_nasdaq=prob_beat_qqq,
        prob_beat_dow=prob_beat_dow,
        expected_return_pct=mc_result["expected_return_pct"],
        median_return_pct=mc_result["median_return_pct"],
        p10_return_pct=mc_result["p10"],
        p25_return_pct=mc_result["p25"],
        p50_return_pct=mc_result["p50"],
        p75_return_pct=mc_result["p75"],
        p90_return_pct=mc_result["p90"],
        key_assumptions=_build_key_assumptions(regime, target_return_pct),
        key_risks=_build_key_risks(regime),
    )

    # --- Step 15: Build scenario analysis ---
    scenarios = _build_scenario_analysis(
        all_positions, t1_pct, t2_pct, t3_pct, cash_pct,
        regime, sector_momentum,
    )

    # --- Step 16: Build sector weights ---
    sector_weights = _build_sector_weights(all_positions)

    # --- Step 17: Build alternatives ---
    alternatives = _build_alternatives(
        target_return_pct, regime, sector_momentum, best,
    )

    # --- Step 17.5: LLM alternative reasoning enrichment ---
    reasoning_source = "template"
    try:
        from ibd_agents.tools.target_return_enrichment import enrich_alternative_reasoning_llm
        alt_inputs = [
            {
                "name": alt.name,
                "target_return_pct": alt.target_return_pct,
                "prob_achieve_target": alt.prob_achieve_target,
                "t1_pct": alt.t1_pct,
                "t2_pct": alt.t2_pct,
                "t3_pct": alt.t3_pct,
                "max_drawdown_pct": alt.max_drawdown_pct,
            }
            for alt in alternatives
        ]
        alt_context = {
            "primary_target": target_return_pct,
            "primary_prob": f"{best['prob_achieve']:.0%}",
            "regime": regime,
        }
        enhanced_alts = enrich_alternative_reasoning_llm(alt_inputs, alt_context)
        if enhanced_alts:
            for idx, alt in enumerate(alternatives):
                if alt.name in enhanced_alts:
                    d = enhanced_alts[alt.name]
                    alternatives[idx] = AlternativePortfolio(
                        name=alt.name,
                        target_return_pct=alt.target_return_pct,
                        prob_achieve_target=alt.prob_achieve_target,
                        position_count=alt.position_count,
                        t1_pct=alt.t1_pct,
                        t2_pct=alt.t2_pct,
                        t3_pct=alt.t3_pct,
                        max_drawdown_pct=alt.max_drawdown_pct,
                        key_difference=d["key_difference"],
                        tradeoff=d["tradeoff"],
                        reasoning_source="llm",
                    )
            reasoning_source = "llm"
            logger.info(
                f"[Agent 14] LLM alternative reasoning: "
                f"{len(enhanced_alts)}/{len(alternatives)} enriched"
            )
    except Exception as e:
        logger.warning(f"[Agent 14] LLM alternative reasoning failed: {e}")

    # Set reasoning_source on non-enriched alternatives
    for idx, alt in enumerate(alternatives):
        if alt.reasoning_source is None:
            alternatives[idx] = AlternativePortfolio(
                name=alt.name,
                target_return_pct=alt.target_return_pct,
                prob_achieve_target=alt.prob_achieve_target,
                position_count=alt.position_count,
                t1_pct=alt.t1_pct,
                t2_pct=alt.t2_pct,
                t3_pct=alt.t3_pct,
                max_drawdown_pct=alt.max_drawdown_pct,
                key_difference=alt.key_difference,
                tradeoff=alt.tradeoff,
                reasoning_source="template",
            )

    # --- Step 18: Build risk disclosure ---
    dd = compute_drawdown_with_stops({1: t1_pct, 2: t2_pct, 3: t3_pct})
    risk_disclosure = RiskDisclosure(
        achievability_rating=achievability,
        achievability_rationale=_build_achievability_rationale(
            achievability, regime, target_return_pct, mc_result["prob_achieve_target"],
        ),
        max_expected_drawdown_pct=dd,
        recovery_time_months=_estimate_recovery_months(dd, regime),
        conditions_for_success=_build_conditions_for_success(regime),
        conditions_for_failure=_build_conditions_for_failure(regime),
        disclaimer=STANDARD_DISCLAIMER,
    )

    # --- Step 19: Build positions ---
    target_positions = [
        TargetPosition(
            ticker=p["ticker"],
            company_name=p["company_name"],
            tier=p["tier"],
            allocation_pct=p["allocation_pct"],
            dollar_amount=p["dollar_amount"],
            shares=p["shares"],
            entry_strategy=p["entry_strategy"],
            target_entry_price=p["target_entry_price"],
            stop_loss_price=p["stop_loss_price"],
            stop_loss_pct=p["stop_loss_pct"],
            expected_return_contribution_pct=p["expected_return_contribution_pct"],
            conviction_level=p["conviction_level"],
            selection_rationale=p["selection_rationale"],
            composite_score=p["composite_score"],
            eps_rating=p["eps_rating"],
            rs_rating=p["rs_rating"],
            sector=p["sector"],
            sector_rank=p["sector_rank"],
            multi_source_count=p["multi_source_count"],
            selection_source=selection_source,
        )
        for p in all_positions
    ]

    # --- Step 20: Build construction rationale ---
    construction_rationale = _build_construction_rationale(
        target_return_pct, regime, t1_pct, t2_pct, t3_pct,
        len(target_positions), mc_result["prob_achieve_target"],
        achievability, sector_momentum,
    )

    # --- Step 20.5: LLM construction narrative enrichment ---
    narrative_source = "template"
    try:
        from ibd_agents.tools.target_return_enrichment import enrich_construction_narrative_llm
        top_sectors_list = [
            sw.sector for sw in sorted(sector_weights, key=lambda x: -x.weight_pct)[:3]
        ]
        top_positions_list = [
            p.ticker for p in sorted(target_positions, key=lambda x: -x.allocation_pct)[:3]
        ]
        narrative_context = {
            "target_return_pct": target_return_pct,
            "regime": regime,
            "t1_pct": t1_pct,
            "t2_pct": t2_pct,
            "t3_pct": t3_pct,
            "n_positions": len(target_positions),
            "prob_achieve": f"{mc_result['prob_achieve_target']:.0%}",
            "achievability": achievability,
            "sector_momentum": sector_momentum,
            "top_sectors": top_sectors_list,
            "top_positions": top_positions_list,
        }
        narrative = enrich_construction_narrative_llm(narrative_context)
        if narrative:
            construction_rationale = narrative
            narrative_source = "llm"
            logger.info(f"[Agent 14] LLM construction narrative: {len(narrative)} chars")
    except Exception as e:
        logger.warning(f"[Agent 14] LLM construction narrative failed: {e}")

    # --- Step 21: Build summary ---
    summary = (
        f"Target return portfolio ({target_return_pct}% target, {regime} regime): "
        f"{len(target_positions)} positions, T1={t1_pct:.0f}%/T2={t2_pct:.0f}%/"
        f"T3={t3_pct:.0f}%, cash={cash_pct:.0f}%. "
        f"Probability of achieving target: {mc_result['prob_achieve_target']:.0%}. "
        f"Achievability: {achievability}. "
        f"Expected return: {mc_result['expected_return_pct']:.1f}%. "
        f"Max drawdown with stops: {dd:.1f}%."
    )

    # --- Step 22: Assemble output ---
    portfolio_name = (
        f"Growth-{target_return_pct:.0f} Portfolio — "
        f"{regime.title()} Regime {date.today().strftime('%b %Y')}"
    )

    output = TargetReturnOutput(
        portfolio_name=portfolio_name,
        target_return_pct=target_return_pct,
        time_horizon_months=time_horizon_months,
        total_capital=total_capital,
        market_regime=regime,
        generated_at=datetime.now().isoformat(),
        analysis_date=date.today().isoformat(),
        tier_allocation=tier_allocation,
        positions=target_positions,
        cash_reserve_pct=cash_pct,
        sector_weights=sector_weights,
        probability_assessment=probability_assessment,
        scenarios=scenarios,
        alternatives=alternatives,
        risk_disclosure=risk_disclosure,
        construction_rationale=construction_rationale,
        narrative_source=narrative_source,
        summary=summary,
    )

    logger.info(
        f"[Agent 14] Done — {len(target_positions)} positions, "
        f"prob={mc_result['prob_achieve_target']:.0%}, "
        f"achievability={achievability}"
    )

    return output


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _build_sector_momentum_map(
    strategy_output: SectorStrategyOutput,
) -> dict[str, str]:
    """Build sector → momentum status mapping from strategy allocations."""
    sector_map: dict[str, str] = {}
    alloc = strategy_output.sector_allocations.overall_allocation
    if not alloc:
        return sector_map

    sorted_sectors = sorted(alloc.items(), key=lambda x: -x[1])
    n = len(sorted_sectors)
    for i, (sector, pct) in enumerate(sorted_sectors):
        if i < n * 0.25:
            sector_map[sector] = "leading"
        elif i < n * 0.50:
            sector_map[sector] = "improving"
        elif i < n * 0.75:
            sector_map[sector] = "lagging"
        else:
            sector_map[sector] = "declining"

    return sector_map


def _allocate_position_counts(
    t1_pct: float, t2_pct: float, t3_pct: float, total: int,
) -> dict[int, int]:
    """Allocate position counts to tiers proportionally."""
    total_pct = t1_pct + t2_pct + t3_pct
    if total_pct <= 0:
        return {1: total, 2: 0, 3: 0}

    t1_count = max(1, round(total * t1_pct / total_pct)) if t1_pct > 0 else 0
    t2_count = max(1, round(total * t2_pct / total_pct)) if t2_pct > 0 else 0
    t3_count = total - t1_count - t2_count
    if t3_pct <= 0:
        t3_count = 0
        t2_count = total - t1_count

    return {1: max(0, t1_count), 2: max(0, t2_count), 3: max(0, t3_count)}


def _build_selection_rationale(pos: dict, sector_momentum_map: dict) -> str:
    """Build selection rationale for a position."""
    ticker = pos["ticker"]
    tier = pos["tier"]
    comp = pos["composite_score"]
    rs = pos["rs_rating"]
    conv = pos.get("conviction_level", "MEDIUM")
    sector = pos["sector"]
    momentum = sector_momentum_map.get(sector, "neutral")

    parts = [
        f"{ticker}: Tier {tier} with Composite {comp}, RS {rs}.",
        f"{conv} conviction in {sector} sector ({momentum} momentum).",
    ]
    if pos.get("multi_source_count", 0) > 0:
        parts.append(f"Multi-source validated (score={pos['multi_source_count']}).")
    return " ".join(parts)


def _build_key_assumptions(regime: str, target: float) -> list[str]:
    """Build key assumptions that must hold for target to be achievable."""
    assumptions = [
        f"Market regime remains {regime} for the investment horizon",
        "IBD momentum characteristics continue to predict outperformance",
        "No major macroeconomic shocks or black swan events",
        f"Leading sectors maintain relative strength supporting {target:.0f}% target",
    ]
    if regime == "bull":
        assumptions.append("Bull market momentum carries through at least 6 more months")
    elif regime == "bear":
        assumptions.append("A regime shift to neutral/bull occurs within 3-4 months")
    return assumptions


def _build_key_risks(regime: str) -> list[str]:
    """Build key risks that could cause significant underperformance."""
    risks = [
        "Sudden regime shift to bear could trigger multiple stop-losses",
        "Sector rotation away from concentrated positions",
        "High concentration amplifies individual stock risk (earnings miss, etc.)",
    ]
    if regime == "bull":
        risks.append("Bull market overextension followed by sharp correction")
    elif regime == "bear":
        risks.extend([
            "Continued deterioration making target completely unachievable",
            "Cascading stop-loss triggers in volatile markets",
        ])
    else:
        risks.append("Regime uncertainty: could shift to either bull or bear")
    return risks


def _build_scenario_analysis(
    positions: list[dict],
    t1_pct: float, t2_pct: float, t3_pct: float, cash_pct: float,
    regime: str, sector_momentum: str,
) -> ScenarioAnalysis:
    """Build bull/base/bear scenario analysis."""
    weights = get_scenario_weights(regime)
    tier_pcts = {1: t1_pct, 2: t2_pct, 3: t3_pct}

    # Group positions by tier for impact analysis
    tier_positions: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    for p in positions:
        tier_positions[p["tier"]].append(p)

    scenario_list = []
    for scenario in ("bull", "base", "bear"):
        # Compute portfolio return
        tier_rets = {t: compute_tier_return(t, scenario, sector_momentum) for t in (1, 2, 3)}
        port_return = compute_blended_return(tier_pcts, tier_rets, cash_pct)

        # Benchmark returns
        spy_ret = BENCHMARK_RETURNS["SPY"][scenario]["mean"]
        qqq_ret = BENCHMARK_RETURNS["QQQ"][scenario]["mean"]
        dow_ret = BENCHMARK_RETURNS["DIA"][scenario]["mean"]

        # Drawdown
        dd = compute_drawdown_with_stops(tier_pcts)

        # Top contributors and drags
        if scenario == "bull":
            top = [p["ticker"] for p in sorted(tier_positions.get(1, []),
                   key=lambda x: -x["allocation_pct"])[:3]]
            drags = [p["ticker"] for p in sorted(tier_positions.get(3, []),
                     key=lambda x: -x["allocation_pct"])[:2]]
            stops = 0
        elif scenario == "base":
            top = [p["ticker"] for p in sorted(positions,
                   key=lambda x: -x["allocation_pct"])[:3]]
            drags = [p["ticker"] for p in sorted(positions,
                     key=lambda x: x["allocation_pct"])[:2]]
            stops = max(0, len(positions) // 10)
        else:  # bear
            top = [p["ticker"] for p in sorted(tier_positions.get(3, []),
                   key=lambda x: -x["allocation_pct"])[:2]]
            drags = [p["ticker"] for p in sorted(tier_positions.get(1, []),
                     key=lambda x: -x["allocation_pct"])[:3]]
            stops = max(1, len(positions) // 3)

        if not top:
            top = [positions[0]["ticker"]] if positions else ["N/A"]
        if not drags:
            drags = [positions[-1]["ticker"]] if positions else ["N/A"]

        scenario_list.append(TargetReturnScenario(
            name=scenario.title(),
            probability_pct=round(weights[scenario] * 100, 1),
            portfolio_return_pct=round(port_return, 2),
            sp500_return_pct=round(spy_ret, 2),
            nasdaq_return_pct=round(qqq_ret, 2),
            dow_return_pct=round(dow_ret, 2),
            alpha_vs_sp500=round(port_return - spy_ret, 2),
            max_drawdown_pct=round(-dd, 2),
            description=_scenario_description(scenario, port_return, regime),
            top_contributors=top,
            biggest_drags=drags,
            stops_triggered=stops,
        ))

    return ScenarioAnalysis(
        bull_scenario=scenario_list[0],
        base_scenario=scenario_list[1],
        bear_scenario=scenario_list[2],
    )


def _scenario_description(scenario: str, ret: float, regime: str) -> str:
    """Generate a description for a scenario."""
    if scenario == "bull":
        return (
            f"Bull scenario: {ret:.1f}% return driven by T1 momentum leaders in "
            f"favorable market conditions. Concentrated positions in sector leaders "
            f"amplify upside. (Current regime: {regime})"
        )
    elif scenario == "base":
        return (
            f"Base scenario: {ret:.1f}% return under normal conditions. IBD-rated "
            f"stocks expected to modestly outperform benchmarks. Portfolio concentration "
            f"provides alpha. (Current regime: {regime})"
        )
    else:
        return (
            f"Bear scenario: {ret:.1f}% return in adverse conditions. Stop-losses "
            f"limit downside. High T1 concentration increases vulnerability. "
            f"Cash reserve provides buffer. (Current regime: {regime})"
        )


def _build_sector_weights(positions: list[dict]) -> list[SectorWeight]:
    """Build sector weight breakdown from positions."""
    sector_data: dict[str, dict] = {}
    for p in positions:
        sector = p["sector"]
        if sector not in sector_data:
            sector_data[sector] = {"weight": 0.0, "count": 0}
        sector_data[sector]["weight"] += p["allocation_pct"]
        sector_data[sector]["count"] += 1

    weights = [
        SectorWeight(
            sector=sector,
            weight_pct=round(data["weight"], 2),
            stock_count=data["count"],
        )
        for sector, data in sorted(sector_data.items(), key=lambda x: -x[1]["weight"])
    ]
    return weights


def _build_alternatives(
    target: float, regime: str, momentum: str, best_solution: dict,
) -> list[AlternativePortfolio]:
    """Build 2-3 alternative portfolios with different risk/return profiles."""
    alternatives = []

    # Alternative 1: Lower target (more conservative)
    lower_target = max(10.0, target - 10.0)
    lower_solutions = solve_tier_mix(
        target_return=lower_target, regime=regime,
        sector_momentum=momentum,
    )
    if lower_solutions:
        ls = lower_solutions[0]
        dd = compute_drawdown_with_stops({1: ls["w1"], 2: ls["w2"], 3: ls["w3"]})
        alternatives.append(AlternativePortfolio(
            name=f"Higher Probability ({lower_target:.0f}% target)",
            target_return_pct=lower_target,
            prob_achieve_target=min(ls["prob_achieve"] + 0.15, MAX_PROB_ACHIEVE),
            position_count=12,
            t1_pct=ls["w1"],
            t2_pct=ls["w2"],
            t3_pct=ls["w3"],
            max_drawdown_pct=dd,
            key_difference=(
                f"Lower target ({lower_target:.0f}% vs {target:.0f}%) allows more "
                f"diversification and lower concentration risk"
            ),
            tradeoff=(
                f"Gain: higher probability of success, lower drawdown ({dd:.0f}% vs "
                f"{compute_drawdown_with_stops({1: best_solution['w1'], 2: best_solution['w2'], 3: best_solution['w3']}):.0f}%). "
                f"Give up: {target - lower_target:.0f}% potential return"
            ),
        ))

    # Alternative 2: Higher target (more aggressive)
    higher_target = target + 10.0
    higher_solutions = solve_tier_mix(
        target_return=higher_target, regime=regime,
        sector_momentum=momentum,
    )
    if higher_solutions:
        hs = higher_solutions[0]
        dd = compute_drawdown_with_stops({1: hs["w1"], 2: hs["w2"], 3: hs["w3"]})
        alternatives.append(AlternativePortfolio(
            name=f"Aggressive ({higher_target:.0f}% target)",
            target_return_pct=higher_target,
            prob_achieve_target=max(hs["prob_achieve"] - 0.10, 0.05),
            position_count=8,
            t1_pct=hs["w1"],
            t2_pct=hs["w2"],
            t3_pct=hs["w3"],
            max_drawdown_pct=dd,
            key_difference=(
                f"Higher target ({higher_target:.0f}% vs {target:.0f}%) requires extreme "
                f"concentration in T1 momentum names"
            ),
            tradeoff=(
                f"Gain: {higher_target - target:.0f}% higher potential return. "
                f"Give up: probability drops significantly, drawdown risk increases to {dd:.0f}%"
            ),
        ))

    # Ensure we have at least 2 alternatives
    if len(alternatives) < 2:
        alternatives.append(AlternativePortfolio(
            name=f"Balanced ({target - 5:.0f}% target)",
            target_return_pct=target - 5.0,
            prob_achieve_target=min(best_solution["prob_achieve"] + 0.10, MAX_PROB_ACHIEVE),
            position_count=12,
            t1_pct=40.0,
            t2_pct=40.0,
            t3_pct=20.0,
            max_drawdown_pct=compute_drawdown_with_stops({1: 40.0, 2: 40.0, 3: 20.0}),
            key_difference=(
                f"Balanced approach with equal T1/T2 weighting and more positions"
            ),
            tradeoff=(
                f"Gain: more diversification and lower volatility. "
                f"Give up: 5% potential return vs primary portfolio"
            ),
        ))

    return alternatives[:3]


def _build_achievability_rationale(
    rating: str, regime: str, target: float, prob: float,
) -> str:
    """Build achievability rationale based on regime and probability."""
    if rating == "REALISTIC":
        return (
            f"In a {regime} market regime, a {target:.0f}% target is achievable with "
            f"proper tier mix and concentrated positions. Monte Carlo shows "
            f"{prob:.0%} probability. Historical T1 momentum stocks return "
            f"35-80% in bull markets."
        )
    elif rating == "STRETCH":
        return (
            f"In a {regime} regime, {target:.0f}% requires concentrated high-conviction "
            f"bets. Monte Carlo shows {prob:.0%} probability — achievable but requires "
            f"favorable sector momentum and stock selection alpha."
        )
    elif rating == "AGGRESSIVE":
        return (
            f"In a {regime} regime, {target:.0f}% has only {prob:.0%} probability. "
            f"Consider the lower-target alternative which has higher probability "
            f"of success. A regime shift would be needed for {target:.0f}% to be realistic."
        )
    else:  # IMPROBABLE
        return (
            f"In a confirmed {regime} market, {target:.0f}% annual returns are improbable "
            f"without leverage. Monte Carlo shows only {prob:.0%} probability. "
            f"The lower-target alternative is presented as the primary recommendation."
        )


def _estimate_recovery_months(max_drawdown: float, regime: str) -> float:
    """Estimate months to recover from max drawdown."""
    base = {
        "bull": 3.0,
        "neutral": 6.0,
        "bear": 12.0,
    }.get(regime, 6.0)

    # Larger drawdowns take proportionally longer
    dd_factor = max_drawdown / 15.0  # 15% is baseline
    return round(base * max(1.0, dd_factor), 1)


def _build_conditions_for_success(regime: str) -> list[str]:
    """Build conditions that must hold for target to be realistic."""
    conditions = [
        "Leading sectors maintain RS ranking and institutional accumulation",
        "No recession or credit tightening within the investment horizon",
        "Selected stocks continue to show strong EPS and revenue growth",
    ]
    if regime == "bull":
        conditions.append("Market maintains uptrend with healthy breadth")
    elif regime == "bear":
        conditions.append("Regime shifts to neutral/bull within 3-4 months")
    else:
        conditions.append("No deterioration into bear regime")
    return conditions


def _build_conditions_for_failure(regime: str) -> list[str]:
    """Build conditions that would make target unachievable."""
    conditions = [
        "Sudden regime shift to bear with cascading stop-loss triggers",
        "Sector rotation away from concentrated positions",
        "Multiple earnings misses or guidance cuts in top holdings",
    ]
    if regime == "bull":
        conditions.append("Bull market exhaustion and sharp correction exceeding 20%")
    elif regime == "bear":
        conditions.append("Continued bear market deepening with no recovery signal")
    else:
        conditions.append("Deterioration into confirmed bear with sector breadth collapse")
    return conditions


def _build_construction_rationale(
    target: float, regime: str,
    t1: float, t2: float, t3: float,
    n_positions: int, prob: float,
    achievability: str, momentum: str,
) -> str:
    """Build the narrative construction rationale (>=100 chars)."""
    return (
        f"This portfolio targets {target:.0f}% annualized returns in a {regime} market regime "
        f"through a concentrated {n_positions}-position construction. The tier allocation of "
        f"T1={t1:.0f}%/T2={t2:.0f}%/T3={t3:.0f}% was optimized via grid search to maximize the "
        f"probability of achieving the target ({prob:.0%}). "
        f"The achievability rating of {achievability} reflects the current market environment "
        f"with {momentum} sector momentum. Positions were selected from the top-ranked stocks "
        f"in each tier, weighted by composite score, relative strength, conviction, and "
        f"multi-source validation. Higher T1 allocation drives return potential while trailing "
        f"stops limit downside risk. Two alternative portfolios are provided with different "
        f"risk/return profiles for comparison."
    )


def _get_stop_for_symbol(
    ticker: str, tier: int, risk_output: RiskAssessment,
) -> float:
    """Get stop loss % for a symbol: prefer Agent 06 LLM recommendation, fall back to tier default."""
    if risk_output and risk_output.stop_loss_recommendations:
        for rec in risk_output.stop_loss_recommendations:
            if rec.symbol == ticker:
                return rec.recommended_stop_pct
    return compute_trailing_stop(tier)


def _build_volatility_map(analyst_output: AnalystOutput) -> dict[str, float]:
    """Build ticker → volatility % map from AnalystOutput. Prefers LLM data."""
    vol_map: dict[str, float] = {}
    for stock in analyst_output.rated_stocks:
        vol = stock.llm_volatility or stock.estimated_volatility_pct
        if vol is not None:
            vol_map[stock.symbol] = vol
    return vol_map


def _build_stock_metrics(analyst_output: AnalystOutput) -> dict[str, dict]:
    """Build ticker → {sharpe_ratio, estimated_volatility_pct, estimated_beta} map."""
    metrics: dict[str, dict] = {}
    for stock in analyst_output.rated_stocks:
        metrics[stock.symbol] = {
            "sharpe_ratio": stock.sharpe_ratio,
            "estimated_volatility_pct": stock.llm_volatility or stock.estimated_volatility_pct,
            "estimated_beta": stock.llm_beta or stock.estimated_beta,
        }
    return metrics
