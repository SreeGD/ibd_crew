"""
Sector Strategist Tool: Theme Mapping & Rotation Signals
IBD Momentum Investment Framework v4.0

Pure functions for:
- Mapping Schwab themes to ETF recommendations based on sector alignment
- Generating rotation action signals with confirmation/invalidation criteria

No LLM, no file I/O, no buy/sell recommendations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

from ibd_agents.schemas.analyst_output import RatedStock, SectorRank
from ibd_agents.schemas.rotation_output import (
    RotationStage,
    RotationStatus,
    RotationType,
    SectorFlow,
)
from ibd_agents.schemas.strategy_output import (
    DEFENSIVE_THEMES,
    GROWTH_THEMES,
    THEME_ETFS,
    THEME_SECTOR_MAP,
    RotationActionSignal,
    ThemeRecommendation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Theme → Recommendation mapping
# ---------------------------------------------------------------------------

def _score_theme(
    theme: str,
    sector_rankings: List[SectorRank],
    dest_sectors: List[SectorFlow],
    regime: str,
    verdict: RotationStatus,
) -> float:
    """
    Score a theme based on alignment with sector rankings, rotation
    destination sectors, and regime.

    Higher score = more relevant theme.
    """
    related_sectors = THEME_SECTOR_MAP.get(theme, [])
    if not related_sectors:
        return 0.0

    sector_rank_lookup = {sr.sector: sr for sr in sector_rankings}
    dest_set = {sf.sector for sf in dest_sectors}

    score = 0.0

    # 1. Sector strength: avg sector_score of related sectors that appear in rankings
    matching_scores = []
    for sector in related_sectors:
        sr = sector_rank_lookup.get(sector)
        if sr:
            matching_scores.append(sr.sector_score)
            # Bonus if sector is a rotation destination
            if sector in dest_set:
                score += 10.0

    if matching_scores:
        score += sum(matching_scores) / len(matching_scores)

    # 2. Regime alignment
    if theme in GROWTH_THEMES and regime == "bull":
        score += 5.0
    elif theme in DEFENSIVE_THEMES and regime == "bear":
        score += 5.0
    elif theme in GROWTH_THEMES and regime == "bear":
        score -= 5.0
    elif theme in DEFENSIVE_THEMES and regime == "bull":
        score -= 3.0

    # 3. Rotation bonus for active rotation toward related sectors
    if verdict == RotationStatus.ACTIVE:
        overlap = set(related_sectors) & dest_set
        if overlap:
            score += 8.0 * len(overlap)

    return round(score, 2)


def _assign_conviction(score: float, regime: str, theme: str) -> str:
    """Assign conviction level based on theme score and context."""
    if score >= 30.0:
        return "HIGH"
    elif score >= 15.0:
        return "MEDIUM"
    else:
        return "LOW"


def _assign_tier_fit(theme: str) -> int:
    """
    Growth themes → T1 (default), Defensive themes → T3 (default).
    Constrained by schema: growth=1|2, defensive=2|3.
    """
    if theme in GROWTH_THEMES:
        return 1
    elif theme in DEFENSIVE_THEMES:
        return 3
    return 2


def _generate_allocation_suggestion(
    tier_fit: int,
    conviction: str,
) -> str:
    """Generate allocation suggestion string (e.g., '3-5% of T1')."""
    tier_label = f"T{tier_fit}"
    if conviction == "HIGH":
        return f"3-5% of {tier_label}"
    elif conviction == "MEDIUM":
        return f"2-3% of {tier_label}"
    else:
        return f"1-2% of {tier_label}"


def map_themes_to_recommendations(
    sector_rankings: List[SectorRank],
    rated_stocks: List[RatedStock],
    dest_sectors: List[SectorFlow],
    regime: str,
    verdict: RotationStatus,
    rated_etfs: list = None,
) -> List[ThemeRecommendation]:
    """
    Map Schwab themes to ETF recommendations based on sector alignment.

    Filters:
    - Skip themes with empty ETF lists (e.g., Aging Population)
    - Only include themes with score > 5
    - Sort by score descending, take top 5
    """
    theme_scores: list[tuple[str, float]] = []

    for theme, etfs in THEME_ETFS.items():
        if not etfs:
            continue  # Skip themes with no ETFs

        score = _score_theme(theme, sector_rankings, dest_sectors, regime, verdict)
        if score > 5.0:
            theme_scores.append((theme, score))

    # Sort by score descending, take top 5
    theme_scores.sort(key=lambda x: -x[1])
    top_themes = theme_scores[:5]

    # Build ETF score lookup from rated_etfs
    etf_score_lookup: dict[str, float] = {}
    if rated_etfs:
        for retf in rated_etfs:
            etf_score_lookup[getattr(retf, "symbol", "")] = getattr(retf, "etf_score", 0.0)

    recommendations: list[ThemeRecommendation] = []
    for theme, score in top_themes:
        etfs = list(THEME_ETFS[theme])
        tier_fit = _assign_tier_fit(theme)
        conviction = _assign_conviction(score, regime, theme)
        alloc_suggestion = _generate_allocation_suggestion(tier_fit, conviction)

        # Build rationale
        related = THEME_SECTOR_MAP.get(theme, [])
        dest_set = {sf.sector for sf in dest_sectors}
        overlap = set(related) & dest_set
        rationale_parts = [f"{theme} theme aligns with sectors: {', '.join(related[:3])}"]
        if overlap:
            rationale_parts.append(f"destination overlap: {', '.join(overlap)}")
        rationale_parts.append(f"score={score:.1f}, regime={regime}")
        rationale = ". ".join(rationale_parts)

        # Rank ETFs by etf_score if rated_etfs provided
        etf_rankings = None
        if rated_etfs and etf_score_lookup:
            # Sort ETFs that have scores; keep order for unscored
            scored_etfs = [(e, etf_score_lookup[e]) for e in etfs if e in etf_score_lookup]
            unscored_etfs = [e for e in etfs if e not in etf_score_lookup]

            if scored_etfs:
                scored_etfs.sort(key=lambda x: -x[1])
                etfs = [e for e, _ in scored_etfs] + unscored_etfs
                etf_rankings = [
                    {"symbol": e, "etf_score": s, "rank": i + 1}
                    for i, (e, s) in enumerate(scored_etfs)
                ]

        recommendations.append(ThemeRecommendation(
            theme=theme,
            recommended_etfs=etfs[:3],  # Top 3 ETFs (reordered by score)
            tier_fit=tier_fit,
            allocation_suggestion=alloc_suggestion,
            conviction=conviction,
            rationale=rationale,
            etf_rankings=etf_rankings,
        ))

    return recommendations


# ---------------------------------------------------------------------------
# Rotation Action Signals
# ---------------------------------------------------------------------------

def generate_rotation_action_signals(
    verdict: RotationStatus,
    rotation_type: RotationType,
    stage: Optional[RotationStage],
    source_sectors: List[SectorFlow],
    dest_sectors: List[SectorFlow],
    regime: str,
) -> List[RotationActionSignal]:
    """
    Generate rotation action signals with confirmation/invalidation criteria.

    ACTIVE: Overweight destination sectors
    EMERGING: Prepare contingency allocation
    NONE: Maintain current allocations
    """
    signals: list[RotationActionSignal] = []

    if verdict == RotationStatus.ACTIVE:
        dest_names = ", ".join(sf.sector for sf in dest_sectors[:3]) or "destination sectors"
        source_names = ", ".join(sf.sector for sf in source_sectors[:3]) or "source sectors"

        signals.append(RotationActionSignal(
            action=(
                f"Overweight {dest_names} in allocation — "
                f"{rotation_type.value} rotation in {stage.value if stage else 'active'} stage"
            ),
            trigger=f"3+ rotation signals confirmed with {regime} regime alignment",
            confirmation=[
                "RS divergence gap continues to widen",
                "Elite concentration shifts toward destination clusters",
                "IBD Keep candidates migrate toward destination sectors",
            ],
            invalidation=[
                "RS divergence gap narrows below threshold",
                "2+ signals reverse within single analysis cycle",
                f"Regime shifts away from {regime}",
            ],
        ))

        signals.append(RotationActionSignal(
            action=(
                f"Reduce allocation to {source_names} as rotation capital "
                f"shifts away from these sectors"
            ),
            trigger=f"Source sectors showing sustained outflow with {rotation_type.value} rotation",
            confirmation=[
                "Source sector elite_pct continues to decline",
                "Breadth shift confirms capital moving to destination clusters",
            ],
            invalidation=[
                "Source sectors recover leadership position",
                "Rotation signals drop below EMERGING threshold",
            ],
        ))

    elif verdict == RotationStatus.EMERGING:
        signals.append(RotationActionSignal(
            action=(
                "Prepare contingency allocation plan — "
                "emerging rotation detected with 2 signals triggered"
            ),
            trigger="Rotation verdict changes from EMERGING to ACTIVE (3+ signals)",
            confirmation=[
                "Third rotation signal triggers",
                "RS divergence gap exceeds threshold by 5+ points",
            ],
            invalidation=[
                "Signals drop to 0-1 (verdict returns to NONE)",
                "Both triggered signals reverse simultaneously",
            ],
        ))

    else:  # NONE
        signals.append(RotationActionSignal(
            action=(
                "Maintain current sector allocations within existing leadership — "
                "no significant rotation detected"
            ),
            trigger="Rotation verdict changes from NONE to EMERGING or ACTIVE",
            confirmation=[
                "2+ rotation signals trigger simultaneously",
                "RS divergence gap exceeds threshold",
            ],
            invalidation=[
                "All signals remain below thresholds",
                "Regime remains stable with no breadth shift",
            ],
        ))

    return signals


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class ThemeMapperInput(BaseModel):
    analyst_json: str = Field(..., description="JSON string of AnalystOutput")
    rotation_json: str = Field(..., description="JSON string of RotationDetectionOutput")


class ThemeMapperTool(BaseTool):
    """Map Schwab themes to ETF recommendations and generate rotation signals."""

    name: str = "theme_mapper"
    description: str = (
        "Map Schwab investment themes to ETF recommendations "
        "based on sector alignment, and generate rotation action signals"
    )
    args_schema: type[BaseModel] = ThemeMapperInput

    def _run(self, analyst_json: str, rotation_json: str) -> str:
        import json
        from ibd_agents.schemas.analyst_output import AnalystOutput
        from ibd_agents.schemas.rotation_output import RotationDetectionOutput

        analyst = AnalystOutput.model_validate_json(analyst_json)
        rotation = RotationDetectionOutput.model_validate_json(rotation_json)

        recommendations = map_themes_to_recommendations(
            sector_rankings=analyst.sector_rankings,
            rated_stocks=analyst.rated_stocks,
            dest_sectors=rotation.destination_sectors,
            regime=rotation.market_regime.regime,
            verdict=rotation.verdict,
        )

        signals = generate_rotation_action_signals(
            verdict=rotation.verdict,
            rotation_type=rotation.rotation_type,
            stage=rotation.rotation_stage,
            source_sectors=rotation.source_sectors,
            dest_sectors=rotation.destination_sectors,
            regime=rotation.market_regime.regime,
        )

        return json.dumps({
            "theme_recommendations": [r.model_dump() for r in recommendations],
            "rotation_signals": [s.model_dump() for s in signals],
        }, indent=2)
