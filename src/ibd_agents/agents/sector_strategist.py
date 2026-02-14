"""
Agent 04: Sector Strategist
Sector Rotation & Allocation Specialist — IBD Momentum Framework v4.0

Receives RotationDetectionOutput + AnalystOutput.
Produces SectorStrategyOutput with:
- Regime-adjusted tier targets
- Sector allocations across T1/T2/T3
- Schwab theme ETF recommendations
- Rotation action signals

This agent translates rotation detection into portfolio allocation.
It never makes buy/sell recommendations or suggests specific position sizes.
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

from ibd_agents.schemas.analyst_output import AnalystOutput
from ibd_agents.schemas.rotation_output import (
    RotationDetectionOutput,
    RotationStatus,
)
from ibd_agents.schemas.strategy_output import (
    REGIME_ACTIONS,
    SectorStrategyOutput,
)
from ibd_agents.tools.sector_allocator import (
    SectorAllocatorTool,
    compute_regime_adjusted_targets,
    compute_sector_allocations,
)
from ibd_agents.tools.theme_mapper import (
    ThemeMapperTool,
    generate_rotation_action_signals,
    map_themes_to_recommendations,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI Agent Builder
# ---------------------------------------------------------------------------

def build_strategist_agent() -> "Agent":
    """Create the Sector Strategist Agent with tools. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Agent(
        role="Sector Rotation & Allocation Specialist",
        goal=(
            "Translate rotation detection signals and sector rankings into "
            "optimal sector allocations across a 3-tier portfolio architecture. "
            "Map Schwab themes to ETF recommendations. Generate actionable "
            "rotation signals with confirmation/invalidation criteria."
        ),
        backstory=(
            "You are a systematic sector strategist who converts rotation "
            "signals into portfolio sector allocations. You respect regime "
            "adjustments, diversification limits (8+ sectors, max 40% per "
            "sector), and the 3-tier architecture (Momentum/Quality/Defensive). "
            "You never recommend specific stock trades, position sizes, or "
            "trailing stops — you allocate at the sector level only."
        ),
        tools=[SectorAllocatorTool(), ThemeMapperTool()],
        verbose=True,
        allow_delegation=False,
        max_iter=10,
        temperature=0.5,
    )


def build_strategist_task(
    agent: "Agent",
    rotation_json: str = "",
    analyst_json: str = "",
) -> "Task":
    """Create the Sector Strategy task. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Task(
        description=f"""Translate rotation + analyst data into sector allocations.

STEPS:
1. Compute regime-adjusted tier targets (T1/T2/T3/Cash)
2. Compute sector allocations with rotation bias
3. Map Schwab themes to ETF recommendations
4. Generate rotation action signals

CONSTRAINTS:
- No single sector > 40% of overall portfolio
- Minimum 8 sectors in allocation
- Tier targets must sum to ~100%
- Growth themes → T1/T2 tier fit only
- Defensive themes → T2/T3 tier fit only
- Cash: 2-15% based on regime + rotation

DO NOT:
- Recommend specific stock trades
- Suggest position sizes or dollar amounts
- Use buy/sell language
- Set trailing stops or price targets

Rotation data:
{rotation_json}

Analyst data:
{analyst_json}
""",
        expected_output=(
            "JSON with rotation_response, regime_adjustment, sector_allocations, "
            "theme_recommendations, rotation_signals, analysis_date, summary."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Deterministic Pipeline
# ---------------------------------------------------------------------------

def run_strategist_pipeline(
    rotation_output: RotationDetectionOutput,
    analyst_output: AnalystOutput,
) -> SectorStrategyOutput:
    """
    Run the deterministic sector strategy pipeline without LLM.

    Args:
        rotation_output: Validated RotationDetectionOutput from Agent 03
        analyst_output: Validated AnalystOutput from Agent 02

    Returns:
        Validated SectorStrategyOutput
    """
    logger.info("[Agent 04] Running Sector Strategy pipeline ...")

    regime = rotation_output.market_regime.regime
    verdict = rotation_output.verdict

    # Step 1: Compute regime-adjusted tier targets
    tier_targets = compute_regime_adjusted_targets(regime)
    logger.info(f"[Agent 04] Tier targets: {tier_targets}")

    # Step 2: Compute sector allocations
    sector_allocations = compute_sector_allocations(
        sector_rankings=analyst_output.sector_rankings,
        source_sectors=rotation_output.source_sectors,
        dest_sectors=rotation_output.destination_sectors,
        stable_sectors=rotation_output.stable_sectors,
        verdict=verdict,
        tier_targets=tier_targets,
        rated_stocks=analyst_output.rated_stocks,
        regime=regime,
    )
    logger.info(
        f"[Agent 04] Allocated {len(sector_allocations.overall_allocation)} sectors, "
        f"cash={sector_allocations.cash_recommendation}%"
    )

    # Step 3: Map themes to recommendations (with ETF rankings if available)
    theme_recommendations = map_themes_to_recommendations(
        sector_rankings=analyst_output.sector_rankings,
        rated_stocks=analyst_output.rated_stocks,
        dest_sectors=rotation_output.destination_sectors,
        regime=regime,
        verdict=verdict,
        rated_etfs=analyst_output.rated_etfs,
    )
    logger.info(f"[Agent 04] {len(theme_recommendations)} theme recommendations")

    # Step 4: Generate rotation action signals
    rotation_signals = generate_rotation_action_signals(
        verdict=verdict,
        rotation_type=rotation_output.rotation_type,
        stage=rotation_output.rotation_stage,
        source_sectors=rotation_output.source_sectors,
        dest_sectors=rotation_output.destination_sectors,
        regime=regime,
    )

    # Step 5: Build rotation response
    rotation_response = _build_rotation_response(
        verdict, rotation_output.rotation_type.value,
        rotation_output.source_sectors, rotation_output.destination_sectors,
    )

    # Step 6: Build regime adjustment
    regime_adjustment = _build_regime_adjustment(regime, tier_targets)

    # Step 7: Build summary
    summary = _build_summary(
        regime, verdict, tier_targets,
        len(sector_allocations.overall_allocation),
        len(theme_recommendations),
    )

    # Assemble output
    output = SectorStrategyOutput(
        rotation_response=rotation_response,
        regime_adjustment=regime_adjustment,
        sector_allocations=sector_allocations,
        theme_recommendations=theme_recommendations,
        rotation_signals=rotation_signals,
        analysis_date=date.today().isoformat(),
        summary=summary,
    )

    logger.info(
        f"[Agent 04] Done — {len(sector_allocations.overall_allocation)} sectors, "
        f"{len(theme_recommendations)} themes, {len(rotation_signals)} signals"
    )

    return output


# ---------------------------------------------------------------------------
# String builders
# ---------------------------------------------------------------------------

def _build_rotation_response(
    verdict: RotationStatus,
    rotation_type: str,
    source_sectors: list,
    dest_sectors: list,
) -> str:
    """Build rotation response string (>= 20 chars)."""
    if verdict == RotationStatus.ACTIVE:
        dest_names = ", ".join(sf.sector for sf in dest_sectors[:3]) or "multiple sectors"
        source_names = ", ".join(sf.sector for sf in source_sectors[:3]) or "multiple sectors"
        return (
            f"Active {rotation_type} rotation detected: capital flowing from "
            f"{source_names} to {dest_names}. Allocations adjusted with rotation bias."
        )
    elif verdict == RotationStatus.EMERGING:
        return (
            "Emerging rotation detected: 2 signals triggered. "
            "Contingency positions prepared. Monitoring for confirmation."
        )
    else:
        return (
            "No rotation detected: maintaining current sector outlook "
            "and optimizing within existing leadership."
        )


def _build_regime_adjustment(
    regime: str,
    tier_targets: dict,
) -> str:
    """Build regime adjustment string (>= 20 chars)."""
    actions = REGIME_ACTIONS.get(regime, REGIME_ACTIONS["neutral"])
    t1 = tier_targets.get("T1", 39.0)
    t3 = tier_targets.get("T3", 22.0)
    return (
        f"{regime.capitalize()} regime: T1 target {t1:.0f}%, T3 target {t3:.0f}%. "
        f"{actions['action']}"
    )


def _build_summary(
    regime: str,
    verdict: RotationStatus,
    tier_targets: dict,
    sector_count: int,
    theme_count: int,
) -> str:
    """Build summary string (>= 50 chars)."""
    t1 = tier_targets.get("T1", 39.0)
    t2 = tier_targets.get("T2", 37.0)
    t3 = tier_targets.get("T3", 22.0)

    parts = [
        f"Sector Strategy: {regime} regime, {verdict.value} rotation.",
        f"Tier allocation: {t1:.0f}/{t2:.0f}/{t3:.0f}.",
        f"{sector_count} sectors allocated.",
    ]
    if theme_count > 0:
        parts.append(f"{theme_count} theme ETF recommendations.")

    summary = " ".join(parts)
    if len(summary) < 50:
        summary += " Portfolio optimized for current market conditions."
    return summary
