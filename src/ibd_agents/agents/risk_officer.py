"""
Agent 06: Risk Officer
Chief Risk Officer — IBD Momentum Framework v4.0

Receives PortfolioOutput from PM.
Produces RiskAssessment with:
- 10-point risk check pipeline
- Stress test scenarios
- Sleep Well scores
- Vetoes and warnings
- Keep validation

This agent reviews portfolios. It never picks stocks or builds portfolios.
"""

from __future__ import annotations

import logging
from datetime import date

try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    Agent = None  # type: ignore
    Task = None  # type: ignore

from ibd_agents.schemas.portfolio_output import PortfolioOutput
from ibd_agents.schemas.risk_output import (
    RiskAssessment,
    RiskWarning,
    Veto,
)
from ibd_agents.schemas.strategy_output import SectorStrategyOutput
from ibd_agents.tools.risk_analyzer import (
    RiskAnalyzerTool,
    build_stress_test_report,
    check_correlation,
    check_keeps,
    check_max_loss,
    check_position_sizing,
    check_regime_alignment,
    check_sector_concentration,
    check_tier_allocation,
    check_trailing_stops,
    check_volume,
    compute_sleep_well_scores,
    run_stress_tests,
    validate_keeps,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI Agent Builder
# ---------------------------------------------------------------------------

def build_risk_agent() -> "Agent":
    """Create the Risk Officer Agent. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Agent(
        role="Chief Risk Officer",
        goal=(
            "Review all portfolio positions against Framework v4.0 risk limits. "
            "Validate trailing stops, detect concentration breaches, assess "
            "market regime, run stress tests, assign Sleep Well scores, and "
            "veto portfolios that violate hard constraints."
        ),
        backstory=(
            "You are a veteran risk officer who survived 2000, 2008, and 2020 "
            "crashes. You are cautious, thorough, and protective. You run a "
            "10-point risk check pipeline and issue vetoes for hard violations. "
            "You never pick stocks or build portfolios — you only review."
        ),
        tools=[RiskAnalyzerTool()],
        verbose=True,
        allow_delegation=False,
        max_iter=10,
        temperature=0.3,
    )


def build_risk_task(
    agent: "Agent",
    portfolio_json: str = "",
) -> "Task":
    """Create the Risk Assessment task. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Task(
        description=f"""Review portfolio against Framework v4.0 risk limits.

10-POINT CHECK PIPELINE:
1. Position sizing (stocks ≤ 5/4/3%, ETFs ≤ 8/8/6%)
2. Trailing stops (22/18/12% or tightened)
3. Sector concentration (max 40%, min 8 sectors)
4. Tier allocation (within target ranges)
5. Max loss (position × stop ≤ framework limit)
6. Correlation (detect same-sector clustering)
7. Regime alignment (portfolio matches market regime)
8. Volume confirmation (validate recent entries)
9. Keeps validation (all 21 present)
10. Stress tests (3 scenarios)

THEN:
- Compute Sleep Well scores (1-10 per tier + overall)
- Issue vetoes for hard violations
- Issue warnings for soft concerns
- Determine overall status (APPROVED/CONDITIONAL/REJECTED)

Portfolio data:
{portfolio_json}
""",
        expected_output=(
            "JSON with check_results, vetoes, warnings, stress_test_results, "
            "sleep_well_scores, keep_validation, overall_status, analysis_date, summary."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Deterministic Pipeline
# ---------------------------------------------------------------------------

def run_risk_pipeline(
    portfolio_output: PortfolioOutput,
    strategy_output: SectorStrategyOutput,
) -> RiskAssessment:
    """
    Run the deterministic risk assessment pipeline without LLM.

    Args:
        portfolio_output: Validated PortfolioOutput from Agent 05
        strategy_output: Validated SectorStrategyOutput for regime info

    Returns:
        Validated RiskAssessment
    """
    logger.info("[Agent 06] Running Risk Assessment pipeline ...")

    # Extract regime from strategy output
    regime = strategy_output.regime_adjustment.split()[0].lower()

    # Step 1: Run all 10 checks
    check_results = [
        check_position_sizing(portfolio_output),
        check_trailing_stops(portfolio_output),
        check_sector_concentration(portfolio_output),
        check_tier_allocation(portfolio_output),
        check_max_loss(portfolio_output),
        check_correlation(portfolio_output),
        check_regime_alignment(portfolio_output, regime),
        check_volume(portfolio_output),
        check_keeps(portfolio_output),
        run_stress_tests(portfolio_output),
    ]

    # --- Step 1.5: LLM stop-loss tuning ---
    stop_loss_recommendations = []
    stop_loss_source = None
    try:
        from ibd_agents.tools.stop_loss_tuning import (
            compute_stop_recommendation,
            enrich_stop_loss_llm,
        )
        # Build position inputs from all tiers
        all_positions = (
            portfolio_output.tier_1.positions
            + portfolio_output.tier_2.positions
            + portfolio_output.tier_3.positions
        )
        pos_inputs = [
            {
                "symbol": p.symbol,
                "sector": p.sector,
                "cap_size": p.cap_size,
                "tier": p.tier,
                "current_stop_pct": p.trailing_stop_pct,
                "conviction": p.conviction,
                "asset_type": p.asset_type,
            }
            for p in all_positions
        ]
        stop_data = enrich_stop_loss_llm(pos_inputs)
        if stop_data:
            from ibd_agents.schemas.risk_output import StopLossRecommendation
            for p in all_positions:
                if p.symbol in stop_data:
                    d = stop_data[p.symbol]
                    rec_stop = compute_stop_recommendation(
                        p.trailing_stop_pct, d["recommended_stop_pct"], p.tier
                    )
                    stop_loss_recommendations.append(StopLossRecommendation(
                        symbol=p.symbol,
                        current_stop_pct=p.trailing_stop_pct,
                        recommended_stop_pct=rec_stop,
                        reason=d.get("reason", "LLM recommendation"),
                        volatility_flag=d.get("volatility_flag"),
                    ))
            stop_loss_source = "llm"
            logger.info(f"LLM stop-loss: {len(stop_loss_recommendations)} recommendations")
        else:
            stop_loss_source = "deterministic"
    except Exception as e:
        logger.warning(f"LLM stop-loss tuning failed: {e}")
        stop_loss_source = "deterministic"

    # Step 2: Collect vetoes and warnings
    vetoes: list[Veto] = []
    warnings: list[RiskWarning] = []

    for check in check_results:
        if check.status == "VETO":
            vetoes.append(Veto(
                check_name=check.check_name,
                reason=check.findings,
                required_fix=(
                    f"Portfolio Manager must resolve {check.check_name} violation: "
                    f"{check.findings}. Review and adjust affected positions."
                ),
            ))
        elif check.status == "WARNING":
            warnings.append(RiskWarning(
                check_name=check.check_name,
                severity="MEDIUM",
                description=check.findings,
                suggestion=f"Consider reviewing {check.check_name} to improve risk profile",
            ))

    # Step 3: Build stress test report
    stress_report = build_stress_test_report(portfolio_output)

    # Step 4: Compute Sleep Well scores
    sleep_scores = compute_sleep_well_scores(check_results, portfolio_output)

    # Step 5: Validate keeps
    keep_val = validate_keeps(portfolio_output)

    # Step 6: Determine overall status
    if vetoes:
        overall_status = "REJECTED"
    elif warnings:
        overall_status = "CONDITIONAL"
    else:
        overall_status = "APPROVED"

    # Step 7: Build summary
    summary = _build_summary(
        check_results, vetoes, warnings,
        sleep_scores, overall_status,
    )

    output = RiskAssessment(
        check_results=check_results,
        vetoes=vetoes,
        warnings=warnings,
        stress_test_results=stress_report,
        sleep_well_scores=sleep_scores,
        keep_validation=keep_val,
        overall_status=overall_status,
        analysis_date=date.today().isoformat(),
        summary=summary,
        stop_loss_recommendations=stop_loss_recommendations,
        stop_loss_source=stop_loss_source,
    )

    logger.info(
        f"[Agent 06] Done — status={overall_status}, "
        f"{len(vetoes)} vetoes, {len(warnings)} warnings, "
        f"Sleep Well={sleep_scores.overall_score}/10"
    )

    return output


# ---------------------------------------------------------------------------
# String Builder
# ---------------------------------------------------------------------------

def _build_summary(
    check_results,
    vetoes,
    warnings,
    sleep_scores,
    overall_status: str,
) -> str:
    """Build summary string (>= 50 chars)."""
    pass_count = sum(1 for c in check_results if c.status == "PASS")
    parts = [
        f"Risk Assessment: {overall_status}.",
        f"{pass_count}/10 checks passed.",
    ]
    if vetoes:
        parts.append(f"{len(vetoes)} veto(s) require PM action.")
    if warnings:
        parts.append(f"{len(warnings)} warning(s) noted.")
    parts.append(f"Sleep Well: {sleep_scores.overall_score}/10.")

    summary = " ".join(parts)
    if len(summary) < 50:
        summary += " Portfolio reviewed against Framework v4.0 risk limits."
    return summary
