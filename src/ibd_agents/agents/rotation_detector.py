"""
Agent 03: Rotation Detector 🔄
Sector Rotation Detection Specialist — IBD Momentum Framework v4.0

Detects sector rotation using a 5-signal framework. Receives
AnalystOutput + ResearchOutput. Produces RotationDetectionOutput
with verdict (ACTIVE/EMERGING/NONE), market regime, rotation
type/stage, sector flows, and strategist notes.

This agent is the "smoke detector" — it detects the fire, not fights it.
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
from ibd_agents.schemas.research_output import ResearchOutput
from ibd_agents.schemas.rotation_output import (
    RotationDetectionOutput,
    RotationStage,
    RotationStatus,
    RotationType,
)
from ibd_agents.tools.market_regime import (
    classify_rotation_stage,
    classify_rotation_type,
    classify_velocity,
    compute_confidence,
    compute_market_regime,
    detect_sector_flows,
    generate_strategist_notes,
    is_regime_aligned,
)
from ibd_agents.tools.rotation_signals import (
    RotationSignalTool,
    compute_all_signals,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI Agent Builder
# ---------------------------------------------------------------------------

def build_rotation_agent() -> "Agent":
    """Create the Rotation Detector Agent with tools. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Agent(
        role="Sector Rotation Detection Specialist",
        goal=(
            "Analyze sector momentum data to detect whether sector rotation "
            "is currently in effect. Apply 5-signal detection framework. "
            "Classify rotation type (Cyclical/Defensive/Offensive/Thematic/Broad). "
            "Provide objective signals for the Sector Strategist."
        ),
        backstory=(
            "You are an expert at detecting sector rotation patterns using "
            "quantitative signals. You analyze RS divergence, leadership "
            "changes, breadth shifts, elite concentration, and IBD Keep "
            "migration. You never make buy/sell recommendations — you "
            "detect and classify rotation for downstream agents."
        ),
        tools=[RotationSignalTool()],
        verbose=True,
        allow_delegation=False,
        max_iter=10,
        temperature=0.2,
    )


def build_rotation_task(
    agent: "Agent",
    analyst_json: str = "",
    research_json: str = "",
) -> "Task":
    """Create the Rotation Detection task. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Task(
        description=f"""Analyze sector momentum for rotation signals.

APPLY 5-SIGNAL FRAMEWORK:
1. RS Divergence: Avg RS of top-3 vs bottom-3 sectors (threshold: gap > 15)
2. Leadership Change: Rank-strength misalignment (threshold: 2+ sectors)
3. Breadth Shift: Per-sector breadth divergence across clusters
4. Elite Concentration Shift: Elite% divergence between clusters
5. IBD Keep Migration: Keep candidates moving away from growth

DETERMINE VERDICT:
- 3-5 signals → ACTIVE
- 2 signals → EMERGING
- 0-1 signals → NONE

CLASSIFY: rotation type, stage, velocity, market regime.

Analyst data:
{analyst_json}

Research data:
{research_json}
""",
        expected_output=(
            "JSON with verdict, confidence, rotation_type, rotation_stage, "
            "market_regime, signals, source/destination sectors, velocity, "
            "strategist_notes, analysis_date, summary."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Deterministic Pipeline
# ---------------------------------------------------------------------------

def run_rotation_pipeline(
    analyst_output: AnalystOutput,
    research_output: ResearchOutput,
) -> RotationDetectionOutput:
    """
    Run the deterministic rotation detection pipeline without LLM.

    Args:
        analyst_output: Validated AnalystOutput from Agent 02
        research_output: Validated ResearchOutput from Agent 01

    Returns:
        Validated RotationDetectionOutput
    """
    logger.info("[Agent 03] Running Rotation Detection pipeline ...")

    # Step 1: Compute all 5 signals (with ETF evidence if available)
    signals = compute_all_signals(
        analyst_output, research_output,
        rated_etfs=analyst_output.rated_etfs,
    )
    logger.info(
        f"[Agent 03] Signals: {signals.signals_active}/5 triggered"
    )

    # Step 2: Compute market regime
    regime = compute_market_regime(
        analyst_output.sector_rankings,
        analyst_output.rated_stocks,
        analyst_output.tier_distribution,
    )
    logger.info(f"[Agent 03] Market regime: {regime.regime}")

    # Step 3: Detect sector flows
    source_sectors, dest_sectors, stable_sectors = detect_sector_flows(
        analyst_output.sector_rankings,
        research_output.sector_patterns,
        analyst_output.rated_stocks,
    )
    logger.info(
        f"[Agent 03] Flows: {len(source_sectors)} source, "
        f"{len(dest_sectors)} dest, {len(stable_sectors)} stable"
    )

    # Step 4: Classify rotation type
    rotation_type = classify_rotation_type(
        signals, source_sectors, dest_sectors, regime
    )

    # Step 5: Determine verdict
    active_count = signals.signals_active
    if active_count >= 3:
        verdict = RotationStatus.ACTIVE
    elif active_count == 2:
        verdict = RotationStatus.EMERGING
    else:
        verdict = RotationStatus.NONE

    # Adjust type for NONE verdict
    if verdict == RotationStatus.NONE:
        rotation_type = RotationType.NONE

    # Step 6: Classify stage
    if verdict == RotationStatus.NONE:
        rotation_stage = None
    else:
        rotation_stage = classify_rotation_stage(
            signals, source_sectors, dest_sectors
        )

    # Step 7: Classify velocity
    rs_gap = float(signals.rs_divergence.value) if signals.rs_divergence.value.replace(".", "").replace("-", "").isdigit() else 0.0
    if verdict == RotationStatus.NONE:
        velocity = None
    else:
        velocity = classify_velocity(signals, rs_gap)

    # Step 8: Compute confidence
    aligned = is_regime_aligned(rotation_type, regime)
    confidence = compute_confidence(active_count, aligned, rs_gap)

    # Ensure ACTIVE has confidence >= 50
    if verdict == RotationStatus.ACTIVE and confidence < 50:
        confidence = 50

    # Step 9: Generate strategist notes
    strategist_notes = generate_strategist_notes(
        signals, rotation_type, rotation_stage,
        source_sectors, dest_sectors, regime,
    )

    # Step 10: Build summary
    summary = _build_summary(
        verdict, active_count, rotation_type, rotation_stage,
        source_sectors, dest_sectors, regime,
    )

    # For NONE verdict, clear sector flows
    if verdict == RotationStatus.NONE:
        source_sectors = []
        dest_sectors = []

    # Step 11: Build ETF flow summary from signal ETF confirmations
    etf_confirmations = []
    for sig in [signals.rs_divergence, signals.breadth_shift]:
        if sig.etf_confirmation:
            etf_confirmations.append(sig.etf_confirmation)
    etf_flow_summary = "; ".join(etf_confirmations) if etf_confirmations else None

    # --- Step 11.5: LLM rotation narrative ---
    rotation_narrative = None
    narrative_source = None
    if verdict != RotationStatus.NONE:
        try:
            from ibd_agents.tools.rotation_narrative import generate_rotation_narrative_llm
            source_cluster_names = sorted(set(sf.cluster for sf in source_sectors))
            dest_cluster_names = sorted(set(sf.cluster for sf in dest_sectors))
            rotation_narrative = generate_rotation_narrative_llm({
                "verdict": verdict.value,
                "rotation_type": rotation_type.value,
                "stage": rotation_stage.value if rotation_stage else None,
                "source_clusters": source_cluster_names,
                "dest_clusters": dest_cluster_names,
                "regime": regime.regime,
                "signals_active": active_count,
                "velocity": velocity,
            })
            narrative_source = "llm" if rotation_narrative else "template"
        except Exception as e:
            logger.warning(f"LLM rotation narrative failed: {e}")
            narrative_source = "template"

    # Step 12: Assemble output
    output = RotationDetectionOutput(
        verdict=verdict,
        confidence=confidence,
        rotation_type=rotation_type,
        rotation_stage=rotation_stage,
        market_regime=regime,
        signals=signals,
        source_sectors=source_sectors,
        destination_sectors=dest_sectors,
        stable_sectors=stable_sectors,
        velocity=velocity,
        etf_flow_summary=etf_flow_summary,
        strategist_notes=strategist_notes,
        analysis_date=date.today().isoformat(),
        summary=summary,
        rotation_narrative=rotation_narrative,
        narrative_source=narrative_source,
    )

    logger.info(
        f"[Agent 03] Done — verdict={verdict.value}, "
        f"confidence={confidence}, type={rotation_type.value}, "
        f"stage={rotation_stage.value if rotation_stage else 'none'}"
    )

    return output


def _build_summary(
    verdict: RotationStatus,
    active_count: int,
    rotation_type: RotationType,
    rotation_stage: RotationStage | None,
    source_sectors: list,
    dest_sectors: list,
    regime,
) -> str:
    """Build a human-readable summary string (>= 50 chars)."""
    if verdict == RotationStatus.NONE:
        return (
            f"No rotation detected. {active_count}/5 signals triggered. "
            f"Market leadership stable across all sector clusters. "
            f"Regime: {regime.regime}."
        )

    source_clusters = sorted(set(sf.cluster for sf in source_sectors))
    dest_clusters = sorted(set(sf.cluster for sf in dest_sectors))

    parts = [
        f"Rotation {verdict.value.upper()}: {active_count}/5 signals triggered.",
    ]

    if rotation_type != RotationType.NONE:
        parts.append(f"{rotation_type.value.capitalize()} rotation detected.")

    if rotation_stage:
        parts.append(f"Stage: {rotation_stage.value}.")

    if source_clusters and dest_clusters:
        parts.append(
            f"Flow from {', '.join(source_clusters)} to {', '.join(dest_clusters)}."
        )
    elif dest_clusters:
        parts.append(f"Capital flowing into {', '.join(dest_clusters)}.")

    parts.append(f"Regime: {regime.regime}.")

    summary = " ".join(parts)
    # Ensure minimum 50 chars
    if len(summary) < 50:
        summary += " Monitoring for further developments in sector leadership."
    return summary
