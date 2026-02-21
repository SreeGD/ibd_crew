"""
Agent 15: Exit Strategist
Exit Discipline Specialist — IBD Momentum Framework v4.0

Receives PortfolioOutput + RiskAssessment.
Produces ExitStrategyOutput with:
- 8 IBD sell rule evaluation per position
- Urgency-ranked signals (CRITICAL -> HEALTHY)
- Evidence chains per recommendation
- Portfolio impact summary
- Health score (1-10)

Two execution paths:
1. Deterministic pipeline: run_exit_strategist_pipeline() — no LLM
2. Agentic pipeline: build_exit_strategist_agent() + build_exit_strategist_task()

This agent evaluates exits. It never picks stocks or builds portfolios.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Optional

try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    Agent = None  # type: ignore[assignment, misc]
    Task = None  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    from ibd_agents.schemas.analyst_output import AnalystOutput
    from ibd_agents.schemas.rotation_output import RotationDetectionOutput

from ibd_agents.schemas.exit_strategy_output import (
    ExitMarketRegime,
    ExitStrategyOutput,
    Urgency,
    URGENCY_ORDER,
)
from ibd_agents.schemas.portfolio_output import PortfolioOutput
from ibd_agents.schemas.reconciliation_output import HoldingsSummary
from ibd_agents.schemas.risk_output import RiskAssessment
from ibd_agents.schemas.strategy_output import SectorStrategyOutput
from ibd_agents.tools.exit_analyzer import (
    ExitAnalyzerTool,
    compute_health_score,
    compute_portfolio_impact,
    evaluate_position,
)
from ibd_agents.tools.position_monitor import (
    PositionMonitorTool,
    build_mock_market_data,
    build_mock_market_health,
    build_real_market_data,
    map_regime_to_exit_regime,
)
from ibd_agents.tools.market_data_fetcher import is_available as has_real_market_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI Agent Builder
# ---------------------------------------------------------------------------

def build_exit_strategist_agent() -> "Agent":
    """Create the Exit Strategist Agent. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError(
            "crewai is required for agentic mode. pip install crewai[tools]"
        )
    return Agent(
        role="Exit Strategist",
        goal=(
            "Protect capital and lock in profits by evaluating every open position "
            "against IBD sell methodology. Produce specific, evidence-backed exit "
            "recommendations with urgency levels. The 7-8% stop loss is absolute "
            "and non-negotiable. No position should ever be allowed to erode "
            "beyond acceptable risk thresholds."
        ),
        backstory=(
            "You are a disciplined exit strategist with deep expertise in IBD sell "
            "methodology. You know that the sell decision is harder than the buy — "
            "emotions fight discipline at every turn. You are the dispassionate voice "
            "that says 'the setup is broken, get out' when every instinct says "
            "'it will come back.' You follow William O'Neil's sell rules precisely. "
            "Small losses are the cost of business. Cutting them quickly preserves "
            "capital for the next winner. You never let a reasonable gain turn into "
            "a loss. You never fall in love with a stock. Your principle: protect "
            "the downside, and the upside takes care of itself."
        ),
        tools=[ExitAnalyzerTool(), PositionMonitorTool()],
        verbose=True,
        allow_delegation=False,
        max_iter=15,
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# CrewAI Task Builder
# ---------------------------------------------------------------------------

def build_exit_strategist_task(
    agent: "Agent",
    portfolio_json: str = "",
) -> "Task":
    """Create the Exit Strategy task. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError(
            "crewai is required for agentic mode. pip install crewai[tools]"
        )
    return Task(
        description=f"""Evaluate all open portfolio positions for exit signals.

For EACH position:
1. Check gain/loss against stop loss thresholds (7-8% standard, tighter in corrections)
2. Evaluate technical health: 50-day MA, RS Rating trend, sector distribution
3. Check for climax top signals if stock has gained significantly
4. Check earnings proximity and profit cushion adequacy
5. Assign urgency: CRITICAL, WARNING, WATCH, or HEALTHY
6. Provide evidence chain for every recommendation

ABSOLUTE RULE: The 7-8% stop loss is non-negotiable. If a stock is down 7-8% from
its buy point, recommend SELL regardless of any other factor.

Order all signals by urgency (CRITICAL first). Include portfolio impact summary
and overall health score (1-10).

DO NOT:
- Never recommend adding to a losing position (averaging down)
- Never override the 7-8% stop loss for any reason
- Never provide specific tax advice

Portfolio data:
{portfolio_json}""",
        expected_output=(
            "JSON with analysis_date, market_regime, portfolio_health_score, "
            "signals (list of PositionSignal with urgency/action/evidence), "
            "portfolio_impact, and summary. Must match ExitStrategyOutput schema."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Deterministic Pipeline
# ---------------------------------------------------------------------------

def run_exit_strategist_pipeline(
    portfolio_output: PortfolioOutput,
    risk_output: RiskAssessment,
    strategy_output: Optional[SectorStrategyOutput] = None,
    analyst_output: Optional["AnalystOutput"] = None,
    rotation_output: Optional["RotationDetectionOutput"] = None,
    brokerage_holdings: Optional[HoldingsSummary] = None,
) -> ExitStrategyOutput:
    """
    Run the deterministic Exit Strategist pipeline without LLM.

    Args:
        portfolio_output: Validated PortfolioOutput from Agent 05.
        risk_output: Validated RiskAssessment from Agent 06.
        strategy_output: Optional SectorStrategyOutput from Agent 04.
        analyst_output: Optional AnalystOutput from Agent 02.
        rotation_output: Optional RotationDetectionOutput from Agent 03.
        brokerage_holdings: Optional HoldingsSummary from brokerage PDFs.
            If provided, positions not in model portfolio are merged in
            with conservative defaults for exit signal evaluation.

    Returns:
        Validated ExitStrategyOutput.
    """
    logger.info("[Agent 15] Running Exit Strategist pipeline ...")

    # Step 1: Map regime
    regime_str = "neutral"
    if strategy_output is not None:
        regime_str = getattr(strategy_output, "regime_adjustment", "neutral") or "neutral"
    regime = map_regime_to_exit_regime(regime_str)
    logger.info(f"[Agent 15] Regime: {regime_str} -> {regime.value}")

    # Step 2: Get market data (real if available, mock fallback)
    if has_real_market_data():
        market_data = build_real_market_data(portfolio_output, analyst_output, brokerage_holdings)
        logger.info(f"[Agent 15] Using real market data for {len(market_data)} positions")
    else:
        market_data = build_mock_market_data(portfolio_output, analyst_output, brokerage_holdings)
        logger.info(f"[Agent 15] Using mock market data for {len(market_data)} positions")

    # Step 3: Generate market health
    market_health = build_mock_market_health(regime)

    # Step 4: Evaluate each position
    signals = []
    for data in market_data:
        signal = evaluate_position(
            data=data,
            regime=regime,
            market_health=market_health,
            tier=data.tier,
            asset_type=data.asset_type,
        )
        signals.append(signal)

    # Step 5: Sort by urgency (CRITICAL first)
    signals.sort(key=lambda s: URGENCY_ORDER[s.urgency.value])

    # Step 6: Compute portfolio impact and health score
    cash_pct = getattr(portfolio_output, "cash_allocation_pct", 2.0)
    if cash_pct is None:
        cash_pct = 2.0
    impact = compute_portfolio_impact(signals, cash_pct)
    health = compute_health_score(signals, regime)

    # Step 7: Build summary
    n_critical = sum(1 for s in signals if s.urgency == Urgency.CRITICAL)
    n_warning = sum(1 for s in signals if s.urgency == Urgency.WARNING)
    n_watch = sum(1 for s in signals if s.urgency == Urgency.WATCH)
    n_healthy = sum(1 for s in signals if s.urgency == Urgency.HEALTHY)

    summary = (
        f"Exit Strategy: {len(signals)} positions evaluated in {regime.value} regime. "
        f"{n_critical} critical, {n_warning} warning, {n_watch} watch, {n_healthy} healthy. "
        f"Portfolio health score: {health}/10. "
        f"Projected cash: {impact.projected_cash_pct:.1f}% if all actions executed."
    )

    # Step 8: Build and return output
    output = ExitStrategyOutput(
        analysis_date=date.today().isoformat(),
        market_regime=regime,
        portfolio_health_score=health,
        signals=signals,
        portfolio_impact=impact,
        summary=summary,
        reasoning_source="deterministic",
    )

    logger.info(
        f"[Agent 15] Done — {len(signals)} signals, "
        f"health={health}/10, "
        f"{n_critical} critical, {n_warning} warning, "
        f"{n_watch} watch, {n_healthy} healthy"
    )

    return output
