"""
Rotation Detector Tool: Market Regime & Rotation Classification
IBD Momentum Investment Framework v4.0

Pure functions for:
- Market regime detection (bull/bear/neutral)
- Rotation type classification
- Rotation stage classification
- Velocity classification
- Confidence scoring
- Sector flow detection
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from ibd_agents.schemas.analyst_output import (
    AnalystOutput,
    RatedStock,
    SectorRank,
    TierDistribution,
)
from ibd_agents.schemas.research_output import SectorPattern
from ibd_agents.schemas.rotation_output import (
    SECTOR_TO_CLUSTER,
    VALID_CLUSTER_NAMES,
    MarketRegime,
    RotationSignals,
    RotationStage,
    RotationType,
    SectorFlow,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Market Regime
# ---------------------------------------------------------------------------

def compute_market_regime(
    sector_rankings: List[SectorRank],
    rated_stocks: List[RatedStock],
    tier_distribution: TierDistribution,
) -> MarketRegime:
    """
    Derive bull/bear/neutral from cross-sectional data:
    - Sector breadth: % sectors with avg_rs > 75
    - Tier concentration: T1 > T3 = bull signal
    - Elite penetration: overall elite_pct
    - Composite breadth: avg composite across all stocks
    """
    bull_signals: list[str] = []
    bear_signals: list[str] = []

    # 1. Sector breadth (% of sectors with avg_rs > 75)
    if sector_rankings:
        strong_sectors = sum(1 for sr in sector_rankings if sr.avg_rs > 75)
        breadth_pct = round((strong_sectors / len(sector_rankings)) * 100, 1)
    else:
        breadth_pct = 50.0

    if breadth_pct > 70:
        bull_signals.append(f"Strong sector breadth ({breadth_pct}% above RS 75)")
    elif breadth_pct < 40:
        bear_signals.append(f"Weak sector breadth ({breadth_pct}% above RS 75)")

    # 2. Tier distribution
    t1 = tier_distribution.tier_1_count
    t3 = tier_distribution.tier_3_count
    if t1 > t3:
        bull_signals.append(f"T1 ({t1}) > T3 ({t3}) tier concentration")
    elif t3 > t1 * 2:
        bear_signals.append(f"T3 ({t3}) dominates T1 ({t1})")

    # 3. Overall elite penetration
    if sector_rankings:
        total_elite = sum(sr.elite_pct * sr.stock_count for sr in sector_rankings)
        total_stocks = sum(sr.stock_count for sr in sector_rankings)
        overall_elite = round(total_elite / total_stocks, 1) if total_stocks else 0.0
        if overall_elite > 40:
            bull_signals.append(f"High elite penetration ({overall_elite}%)")
        elif overall_elite < 15:
            bear_signals.append(f"Low elite penetration ({overall_elite}%)")

    # 4. Composite breadth
    if rated_stocks:
        avg_comp = sum(s.composite_rating for s in rated_stocks) / len(rated_stocks)
        if avg_comp > 85:
            bull_signals.append(f"Strong composite breadth (avg {avg_comp:.0f})")
        elif avg_comp < 75:
            bear_signals.append(f"Weak composite breadth (avg {avg_comp:.0f})")

    # Determine regime
    bull_count = len(bull_signals)
    bear_count = len(bear_signals)

    if bull_count >= 2 and bear_count == 0:
        regime = "bull"
    elif bear_count >= 2 and bull_count == 0:
        regime = "bear"
    elif bull_count > bear_count:
        regime = "bull"
    elif bear_count > bull_count:
        regime = "bear"
    else:
        regime = "neutral"

    note_parts = []
    if bull_signals:
        note_parts.append(f"Bull: {'; '.join(bull_signals)}")
    if bear_signals:
        note_parts.append(f"Bear: {'; '.join(bear_signals)}")
    if not note_parts:
        note_parts.append("Neutral regime — no strong signals in either direction")
    regime_note = ". ".join(note_parts)

    return MarketRegime(
        regime=regime,
        bull_signals_present=bull_signals,
        bear_signals_present=bear_signals,
        sector_breadth_pct=breadth_pct,
        regime_note=regime_note,
    )


# ---------------------------------------------------------------------------
# Sector Flow Detection
# ---------------------------------------------------------------------------

def detect_sector_flows(
    sector_rankings: List[SectorRank],
    sector_patterns: List[SectorPattern],
    rated_stocks: List[RatedStock],
) -> Tuple[List[SectorFlow], List[SectorFlow], List[str]]:
    """
    Classify each sector as inflow, outflow, or stable.

    Returns (source_sectors, destination_sectors, stable_sectors).

    Inflow: top-half rank + (leading/improving strength OR high breadth)
    Outflow: bottom-half rank + (lagging/declining OR low breadth)
    Stable: everything else
    """
    pattern_lookup = {sp.sector: sp for sp in sector_patterns}

    # Compute per-sector breadth
    sector_stock_count: dict[str, int] = {}
    sector_high_rs_count: dict[str, int] = {}
    for stock in rated_stocks:
        sec = stock.sector
        sector_stock_count[sec] = sector_stock_count.get(sec, 0) + 1
        if stock.rs_rating >= 80:
            sector_high_rs_count[sec] = sector_high_rs_count.get(sec, 0) + 1

    n = len(sector_rankings)
    midpoint = n // 2 if n > 0 else 0

    source_sectors: list[SectorFlow] = []
    dest_sectors: list[SectorFlow] = []
    stable_sectors: list[str] = []

    for sr in sector_rankings:
        cluster = SECTOR_TO_CLUSTER.get(sr.sector)
        if cluster is None:
            stable_sectors.append(sr.sector)
            continue

        sp = pattern_lookup.get(sr.sector)
        strength = sp.strength if sp else "lagging"
        trend = sp.trend_direction if sp else "flat"

        in_top_half = sr.rank <= midpoint
        total_in_sector = sector_stock_count.get(sr.sector, 0)
        high_rs_in_sector = sector_high_rs_count.get(sr.sector, 0)
        breadth = (high_rs_in_sector / total_in_sector * 100) if total_in_sector > 0 else 0.0

        is_strong = strength in ("leading", "improving") or breadth > 60
        is_weak = strength in ("lagging", "declining") or breadth < 40

        if in_top_half and is_strong:
            magnitude = "Strong" if sr.elite_pct > 50 else ("Moderate" if sr.elite_pct > 25 else "Weak")
            dest_sectors.append(SectorFlow(
                sector=sr.sector,
                cluster=cluster,
                direction="inflow",
                current_rank=sr.rank,
                avg_rs=sr.avg_rs,
                elite_pct=sr.elite_pct,
                stock_count=sr.stock_count,
                magnitude=magnitude,
                evidence=(
                    f"{sr.sector} rank {sr.rank} (top half), "
                    f"strength={strength}, trend={trend}, "
                    f"avg_rs={sr.avg_rs:.0f}, elite={sr.elite_pct:.0f}%, "
                    f"breadth={breadth:.0f}%"
                ),
            ))
        elif not in_top_half and is_weak:
            magnitude = "Strong" if sr.elite_pct < 10 else ("Moderate" if sr.elite_pct < 30 else "Weak")
            source_sectors.append(SectorFlow(
                sector=sr.sector,
                cluster=cluster,
                direction="outflow",
                current_rank=sr.rank,
                avg_rs=sr.avg_rs,
                elite_pct=sr.elite_pct,
                stock_count=sr.stock_count,
                magnitude=magnitude,
                evidence=(
                    f"{sr.sector} rank {sr.rank} (bottom half), "
                    f"strength={strength}, trend={trend}, "
                    f"avg_rs={sr.avg_rs:.0f}, elite={sr.elite_pct:.0f}%, "
                    f"breadth={breadth:.0f}%"
                ),
            ))
        else:
            stable_sectors.append(sr.sector)

    return source_sectors, dest_sectors, stable_sectors


# ---------------------------------------------------------------------------
# Rotation Type Classification
# ---------------------------------------------------------------------------

def classify_rotation_type(
    signals: RotationSignals,
    source_sectors: List[SectorFlow],
    dest_sectors: List[SectorFlow],
    regime: MarketRegime,
) -> RotationType:
    """
    Classify rotation based on source/destination clusters and regime.

    CYCLICAL: commodity gaining while growth declining
    DEFENSIVE: defensive cluster gaining + bear regime
    OFFENSIVE: growth gaining + bull regime
    BROAD: 3+ destination clusters involved
    REVERSAL: destination clusters were previously source (inferred from strength)
    THEMATIC: single non-traditional cluster dominant in destination
    NONE: no rotation signals
    """
    if signals.signals_active < 2:
        return RotationType.NONE

    dest_clusters = set(sf.cluster for sf in dest_sectors)
    source_clusters = set(sf.cluster for sf in source_sectors)

    # BROAD: 3+ destination clusters
    if len(dest_clusters) >= 3:
        return RotationType.BROAD

    # CYCLICAL: commodity gaining, growth declining
    if "commodity" in dest_clusters and "growth" in source_clusters:
        return RotationType.CYCLICAL
    if "growth" in dest_clusters and "commodity" in source_clusters:
        return RotationType.CYCLICAL

    # DEFENSIVE: defensive gaining + bear regime
    if "defensive" in dest_clusters and regime.regime == "bear":
        return RotationType.DEFENSIVE

    # OFFENSIVE: growth gaining + bull regime
    if "growth" in dest_clusters and regime.regime == "bull":
        return RotationType.OFFENSIVE

    # REVERSAL: source and dest overlap in cluster families
    if source_clusters & dest_clusters:
        return RotationType.REVERSAL

    # THEMATIC: single cluster dominant in destination
    if len(dest_clusters) == 1:
        return RotationType.THEMATIC

    # Default for active rotation
    if signals.signals_active >= 2:
        return RotationType.CYCLICAL

    return RotationType.NONE


# ---------------------------------------------------------------------------
# Rotation Stage Classification
# ---------------------------------------------------------------------------

def classify_rotation_stage(
    signals: RotationSignals,
    source_sectors: List[SectorFlow],
    dest_sectors: List[SectorFlow],
) -> Optional[RotationStage]:
    """
    EARLY: 2-3 signals, destination elite < source
    MID: 3-4 signals, destination elite > source
    LATE: 4-5 signals, destination dominant
    EXHAUSTING: 5 signals + destination breadth may be peaking
    """
    active = signals.signals_active
    if active < 2:
        return None

    dest_elite = (
        sum(sf.elite_pct for sf in dest_sectors) / len(dest_sectors)
        if dest_sectors else 0.0
    )
    source_elite = (
        sum(sf.elite_pct for sf in source_sectors) / len(source_sectors)
        if source_sectors else 0.0
    )

    if active == 5:
        return RotationStage.EXHAUSTING
    elif active >= 4:
        if dest_elite > 60:
            return RotationStage.LATE
        return RotationStage.MID
    elif active == 3:
        if dest_elite > source_elite:
            return RotationStage.MID
        return RotationStage.EARLY
    else:  # active == 2
        return RotationStage.EARLY


# ---------------------------------------------------------------------------
# Velocity Classification
# ---------------------------------------------------------------------------

def classify_velocity(
    signals: RotationSignals,
    rs_gap: float,
) -> Optional[str]:
    """
    Fast/Moderate/Slow based on signal count + RS gap.
    """
    active = signals.signals_active
    if active < 2:
        return None

    if active >= 4 and rs_gap > 20:
        return "Fast"
    elif active >= 3 or rs_gap > 15:
        return "Moderate"
    else:
        return "Slow"


# ---------------------------------------------------------------------------
# Confidence Scoring
# ---------------------------------------------------------------------------

def compute_confidence(
    signals_active: int,
    regime_alignment: bool,
    rs_gap: float,
) -> int:
    """
    Base: signals * 18
    +5 for regime alignment (rotation type matches regime direction)
    +5 for large RS gap (> 20)
    Capped at 95, min 5.
    """
    base = signals_active * 18
    if regime_alignment:
        base += 5
    if rs_gap > 20:
        base += 5
    return max(5, min(95, base))


# ---------------------------------------------------------------------------
# Regime Alignment Check
# ---------------------------------------------------------------------------

def is_regime_aligned(
    rotation_type: RotationType,
    regime: MarketRegime,
) -> bool:
    """Check if rotation type aligns with market regime."""
    if rotation_type == RotationType.DEFENSIVE and regime.regime == "bear":
        return True
    if rotation_type == RotationType.OFFENSIVE and regime.regime == "bull":
        return True
    if rotation_type == RotationType.CYCLICAL and regime.regime in ("bull", "neutral"):
        return True
    return False


# ---------------------------------------------------------------------------
# Strategist Notes Generation
# ---------------------------------------------------------------------------

def generate_strategist_notes(
    signals: RotationSignals,
    rotation_type: RotationType,
    stage: Optional[RotationStage],
    source_sectors: List[SectorFlow],
    dest_sectors: List[SectorFlow],
    regime: MarketRegime,
) -> List[str]:
    """
    Generate 2-5 objective strategist notes. No buy/sell language,
    no allocation percentages, no duration predictions.
    """
    notes: list[str] = []

    # Note 1: Signal summary
    triggered = []
    if signals.rs_divergence.triggered:
        triggered.append("RS Divergence")
    if signals.leadership_change.triggered:
        triggered.append("Leadership Change")
    if signals.breadth_shift.triggered:
        triggered.append("Breadth Shift")
    if signals.elite_concentration_shift.triggered:
        triggered.append("Elite Concentration")
    if signals.ibd_keep_migration.triggered:
        triggered.append("IBD Keep Migration")

    if triggered:
        notes.append(
            f"Signals triggered ({len(triggered)}/5): {', '.join(triggered)}"
        )

    # Note 2: Cluster flow
    if dest_sectors:
        dest_clusters = set(sf.cluster for sf in dest_sectors)
        dest_names = ", ".join(sorted(dest_clusters))
        notes.append(f"Capital flowing into {dest_names} cluster(s)")
    if source_sectors:
        source_clusters = set(sf.cluster for sf in source_sectors)
        source_names = ", ".join(sorted(source_clusters))
        notes.append(f"Capital flowing out of {source_names} cluster(s)")

    # Note 3: Regime context
    notes.append(f"Market regime: {regime.regime} ({regime.regime_note[:100]})")

    # Note 4: Stage context (if applicable)
    if stage:
        notes.append(f"Rotation stage: {stage.value} — monitoring for progression")

    # Ensure at least 2 notes
    if len(notes) < 2:
        notes.append("Monitoring sector leadership for further developments")

    return notes[:5]
