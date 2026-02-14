"""
Sector Strategist Tool: Allocation Computation
IBD Momentum Investment Framework v4.0

Pure functions for computing sector allocations across the 3-tier
portfolio architecture. Translates sector rankings + rotation flows
into a SectorAllocationPlan with regime-adjusted targets.

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
    RotationStatus,
    SectorFlow,
)
from ibd_agents.schemas.strategy_output import (
    REGIME_ACTIONS,
    SECTOR_LIMITS,
    TIER_ALLOCATION_TARGETS,
    TIER_TARGET_RANGES,
    SectorAllocationPlan,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime-adjusted targets
# ---------------------------------------------------------------------------

def compute_regime_adjusted_targets(regime: str) -> Dict[str, float]:
    """
    Apply REGIME_ACTIONS bias to baseline TIER_ALLOCATION_TARGETS.

    T1 gets the tier_1_bias, T3 gets the tier_3_bias,
    T2 absorbs remainder, Cash stays at baseline.
    Results clamped to TIER_TARGET_RANGES.

    Returns: {"T1": float, "T2": float, "T3": float, "Cash": float}
    """
    actions = REGIME_ACTIONS.get(regime, REGIME_ACTIONS["neutral"])
    t1_bias = actions["tier_1_bias"]
    t3_bias = actions["tier_3_bias"]

    base_t1 = TIER_ALLOCATION_TARGETS["tier_1_momentum"]["target_pct"]
    base_t2 = TIER_ALLOCATION_TARGETS["tier_2_quality"]["target_pct"]
    base_t3 = TIER_ALLOCATION_TARGETS["tier_3_defensive"]["target_pct"]
    base_cash = TIER_ALLOCATION_TARGETS["cash"]["target_pct"]

    raw_t1 = base_t1 + t1_bias
    raw_t3 = base_t3 + t3_bias
    raw_cash = float(base_cash)

    # Clamp T1, T3, Cash to ranges
    lo_t1, hi_t1 = TIER_TARGET_RANGES["T1"]
    lo_t3, hi_t3 = TIER_TARGET_RANGES["T3"]
    lo_cash, hi_cash = TIER_TARGET_RANGES["Cash"]

    t1 = max(lo_t1, min(hi_t1, float(raw_t1)))
    t3 = max(lo_t3, min(hi_t3, float(raw_t3)))
    cash = max(lo_cash, min(hi_cash, raw_cash))

    # T2 absorbs remainder so total = 100
    t2 = 100.0 - t1 - t3 - cash

    # Clamp T2 to range
    lo_t2, hi_t2 = TIER_TARGET_RANGES["T2"]
    t2 = max(lo_t2, min(hi_t2, t2))

    return {"T1": round(t1, 1), "T2": round(t2, 1), "T3": round(t3, 1), "Cash": round(cash, 1)}


# ---------------------------------------------------------------------------
# Cash recommendation
# ---------------------------------------------------------------------------

def compute_cash_recommendation(regime: str, verdict: RotationStatus) -> float:
    """
    Cash recommendation based on regime + rotation verdict.

    Bull + no rotation: 2%
    Bull + rotation: 3%
    Neutral: 3-5%
    Bear: 5-10%
    Active rotation adds 1-2% to base.
    """
    if regime == "bull":
        base = 2.0
    elif regime == "bear":
        base = 7.0
    else:
        base = 4.0

    if verdict == RotationStatus.ACTIVE:
        base += 2.0
    elif verdict == RotationStatus.EMERGING:
        base += 1.0

    return max(2.0, min(15.0, round(base, 1)))


# ---------------------------------------------------------------------------
# Sector scoring and allocation
# ---------------------------------------------------------------------------

def _score_sectors(
    sector_rankings: List[SectorRank],
    source_sectors: List[SectorFlow],
    dest_sectors: List[SectorFlow],
    verdict: RotationStatus,
) -> Dict[str, float]:
    """
    Score each sector using sector_score from rankings,
    then apply rotation bias.

    ACTIVE: dest × 1.3, source × 0.7
    EMERGING: dest × 1.15, source × 0.85
    NONE: no adjustment
    """
    source_set = {sf.sector for sf in source_sectors}
    dest_set = {sf.sector for sf in dest_sectors}

    scores: dict[str, float] = {}
    for sr in sector_rankings:
        base_score = sr.sector_score
        if verdict == RotationStatus.ACTIVE:
            if sr.sector in dest_set:
                base_score *= 1.3
            elif sr.sector in source_set:
                base_score *= 0.7
        elif verdict == RotationStatus.EMERGING:
            if sr.sector in dest_set:
                base_score *= 1.15
            elif sr.sector in source_set:
                base_score *= 0.85
        scores[sr.sector] = round(base_score, 2)

    return scores


def _normalize_and_cap(
    scores: Dict[str, float],
    cap_pct: float = 40.0,
) -> Dict[str, float]:
    """
    Normalize scores to sum to 100%, cap at max_single_sector,
    redistribute excess proportionally.
    """
    if not scores:
        return {}

    total = sum(scores.values())
    if total <= 0:
        n = len(scores)
        return {s: round(100.0 / n, 1) for s in scores}

    # Normalize to 100%
    normalized = {s: (v / total) * 100.0 for s, v in scores.items()}

    # Cap and redistribute (iterate to convergence)
    for _ in range(10):
        excess = 0.0
        uncapped_total = 0.0
        capped = set()

        for s, pct in normalized.items():
            if pct > cap_pct:
                excess += pct - cap_pct
                normalized[s] = cap_pct
                capped.add(s)

        if excess <= 0.01:
            break

        for s in normalized:
            if s not in capped:
                uncapped_total += normalized[s]

        if uncapped_total > 0:
            for s in normalized:
                if s not in capped:
                    normalized[s] += excess * (normalized[s] / uncapped_total)

    # Round
    result = {s: round(v, 1) for s, v in normalized.items()}
    return result


def _compute_tier_allocation(
    tier: int,
    sector_scores: Dict[str, float],
    rated_stocks: List[RatedStock],
) -> Dict[str, float]:
    """
    Compute per-tier sector allocation by weighting sector scores
    by how many stocks in that tier belong to each sector.
    """
    # Count stocks per sector in this tier
    sector_tier_count: dict[str, int] = {}
    for stock in rated_stocks:
        if stock.tier == tier:
            sector_tier_count[stock.sector] = sector_tier_count.get(stock.sector, 0) + 1

    if not sector_tier_count:
        # Fallback: use overall sector scores
        return _normalize_and_cap(sector_scores)

    # Weight = sector_score * sqrt(stock_count_in_tier) to balance
    weighted: dict[str, float] = {}
    for sector, count in sector_tier_count.items():
        score = sector_scores.get(sector, 1.0)
        weighted[sector] = score * (count ** 0.5)

    # Per-tier allocations: normalize to 100% but don't apply the 40% cap.
    # The 40% cap is for overall portfolio concentration, not within-tier splits.
    # A tier with 1-2 sectors legitimately needs >40% per sector.
    return _normalize_and_cap(weighted, cap_pct=100.0)


def _ensure_min_sectors(
    allocation: Dict[str, float],
    sector_scores: Dict[str, float],
    min_sectors: int = 8,
) -> Dict[str, float]:
    """
    If allocation has fewer than min_sectors sectors, add the highest-scored
    missing sectors with small allocations.
    """
    if len(allocation) >= min_sectors:
        return allocation

    missing = {s: v for s, v in sector_scores.items() if s not in allocation}
    sorted_missing = sorted(missing.items(), key=lambda x: -x[1])

    result = dict(allocation)
    needed = min_sectors - len(result)

    for sector, _ in sorted_missing[:needed]:
        result[sector] = 2.0  # Minimum viable allocation

    # Re-normalize
    return _normalize_and_cap(result)


# ---------------------------------------------------------------------------
# Main allocation function
# ---------------------------------------------------------------------------

def compute_sector_allocations(
    sector_rankings: List[SectorRank],
    source_sectors: List[SectorFlow],
    dest_sectors: List[SectorFlow],
    stable_sectors: List[str],
    verdict: RotationStatus,
    tier_targets: Dict[str, float],
    rated_stocks: List[RatedStock],
    regime: str,
) -> SectorAllocationPlan:
    """
    Master allocation function.

    1. Score sectors with rotation bias
    2. Compute per-tier allocations (weighted by tier membership)
    3. Compute overall allocation as weighted blend of per-tier × tier_targets
    4. Ensure ≥8 sectors, no sector >40%
    5. Compute cash and generate rationale
    """
    # 1. Score sectors
    sector_scores = _score_sectors(sector_rankings, source_sectors, dest_sectors, verdict)

    if not sector_scores:
        # Degenerate case: no rankings at all
        sector_scores = {"CHIPS": 10.0, "SOFTWARE": 9.0, "MINING": 8.0,
                         "BANKS": 7.0, "MEDICAL": 6.0, "ENERGY": 5.0,
                         "AEROSPACE": 4.0, "BUILDING": 3.0}

    # 2. Per-tier allocations
    t1_alloc = _compute_tier_allocation(1, sector_scores, rated_stocks)
    t2_alloc = _compute_tier_allocation(2, sector_scores, rated_stocks)
    t3_alloc = _compute_tier_allocation(3, sector_scores, rated_stocks)

    # 3. Overall allocation: weighted blend
    t1_weight = tier_targets.get("T1", 39.0) / 100.0
    t2_weight = tier_targets.get("T2", 37.0) / 100.0
    t3_weight = tier_targets.get("T3", 22.0) / 100.0

    all_sectors = set(t1_alloc) | set(t2_alloc) | set(t3_alloc)
    overall: dict[str, float] = {}
    for sector in all_sectors:
        val = (
            t1_alloc.get(sector, 0.0) * t1_weight
            + t2_alloc.get(sector, 0.0) * t2_weight
            + t3_alloc.get(sector, 0.0) * t3_weight
        )
        if val > 0.1:  # Drop negligible sectors
            overall[sector] = round(val, 1)

    # 4. Ensure min sectors and cap
    overall = _ensure_min_sectors(overall, sector_scores, SECTOR_LIMITS["min_sectors"])
    overall = _normalize_and_cap(overall, SECTOR_LIMITS["max_single_sector"])

    # 5. Cash recommendation
    cash_rec = compute_cash_recommendation(regime, verdict)

    # 6. Rationale
    rationale = generate_allocation_rationale(
        regime, verdict, tier_targets, len(overall),
        source_sectors, dest_sectors,
    )

    return SectorAllocationPlan(
        tier_1_allocation=t1_alloc,
        tier_2_allocation=t2_alloc,
        tier_3_allocation=t3_alloc,
        overall_allocation=overall,
        tier_targets=tier_targets,
        cash_recommendation=cash_rec,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Rationale generation
# ---------------------------------------------------------------------------

def generate_allocation_rationale(
    regime: str,
    verdict: RotationStatus,
    tier_targets: Dict[str, float],
    sector_count: int,
    source_sectors: List[SectorFlow],
    dest_sectors: List[SectorFlow],
) -> str:
    """Generate a ≥50-character allocation rationale string."""
    parts = []

    # Regime context
    parts.append(f"{regime.capitalize()} regime")

    # Rotation context
    if verdict == RotationStatus.ACTIVE:
        dest_names = ", ".join(sf.sector for sf in dest_sectors[:3])
        source_names = ", ".join(sf.sector for sf in source_sectors[:3])
        parts.append(f"active rotation from {source_names or 'broad'} to {dest_names or 'broad'}")
    elif verdict == RotationStatus.EMERGING:
        parts.append("emerging rotation detected — prepared contingency positions")
    else:
        parts.append("no rotation — maintaining baseline allocation targets")

    # Tier targets
    t1 = tier_targets.get("T1", 39.0)
    t2 = tier_targets.get("T2", 37.0)
    t3 = tier_targets.get("T3", 22.0)
    parts.append(f"tier targets {t1:.0f}/{t2:.0f}/{t3:.0f}")

    # Diversification
    parts.append(f"{sector_count} sectors allocated for diversification")

    rationale = ". ".join(parts)
    # Pad if needed (should be well over 50 chars)
    if len(rationale) < 50:
        rationale += ". Allocation optimized across momentum, quality, and defensive tiers."
    return rationale


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class SectorAllocatorInput(BaseModel):
    analyst_json: str = Field(..., description="JSON string of AnalystOutput")
    rotation_json: str = Field(..., description="JSON string of RotationDetectionOutput")


class SectorAllocatorTool(BaseTool):
    """Compute sector allocations for 3-tier portfolio architecture."""

    name: str = "sector_allocator"
    description: str = (
        "Compute regime-adjusted sector allocations across T1/T2/T3 "
        "portfolio tiers with rotation bias and concentration limits"
    )
    args_schema: type[BaseModel] = SectorAllocatorInput

    def _run(self, analyst_json: str, rotation_json: str) -> str:
        from ibd_agents.schemas.analyst_output import AnalystOutput
        from ibd_agents.schemas.rotation_output import RotationDetectionOutput

        analyst = AnalystOutput.model_validate_json(analyst_json)
        rotation = RotationDetectionOutput.model_validate_json(rotation_json)

        tier_targets = compute_regime_adjusted_targets(rotation.market_regime.regime)
        plan = compute_sector_allocations(
            sector_rankings=analyst.sector_rankings,
            source_sectors=rotation.source_sectors,
            dest_sectors=rotation.destination_sectors,
            stable_sectors=rotation.stable_sectors,
            verdict=rotation.verdict,
            tier_targets=tier_targets,
            rated_stocks=analyst.rated_stocks,
            regime=rotation.market_regime.regime,
        )
        return plan.model_dump_json(indent=2)
