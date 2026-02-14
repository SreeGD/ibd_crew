"""
Rotation Detector Tool: 5-Signal Detection Framework
IBD Momentum Investment Framework v4.0

Five pure signal detection functions that analyse single-snapshot data
from AnalystOutput + ResearchOutput to determine rotation signals.

Each detector returns a SignalReading (triggered/not, value, threshold, evidence).
compute_all_signals() aggregates all five into a RotationSignals model.
"""

from __future__ import annotations

import logging
from typing import List, Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

from ibd_agents.schemas.analyst_output import (
    AnalystOutput,
    IBDKeep,
    RatedStock,
    SectorRank,
)
from ibd_agents.schemas.research_output import ResearchOutput, SectorPattern
from ibd_agents.schemas.rotation_output import (
    RS_DIVERGENCE_THRESHOLD,
    BREADTH_HIGH_THRESHOLD,
    BREADTH_LOW_THRESHOLD,
    ELITE_CLUSTER_HIGH,
    ELITE_CLUSTER_LOW,
    LEADERSHIP_MISALIGNMENT_MIN,
    SECTOR_CLUSTERS,
    SECTOR_TO_CLUSTER,
    RotationSignals,
    SignalReading,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal 1: RS Divergence
# ---------------------------------------------------------------------------

def detect_rs_divergence(sector_rankings: List[SectorRank]) -> SignalReading:
    """
    Avg RS of top-3 ranked sectors vs bottom-3 sectors.
    Triggered if gap > RS_DIVERGENCE_THRESHOLD (15).

    Adapted for single-snapshot: measures cross-sectional divergence
    rather than temporal change.
    """
    if len(sector_rankings) < 6:
        return SignalReading(
            signal_name="RS Divergence",
            triggered=False,
            value="0.0",
            threshold=str(RS_DIVERGENCE_THRESHOLD),
            evidence=f"Not enough sectors to measure RS divergence ({len(sector_rankings)} < 6 required)",
        )

    sorted_sectors = sorted(sector_rankings, key=lambda s: s.rank)
    top_3 = sorted_sectors[:3]
    bottom_3 = sorted_sectors[-3:]

    top_avg = sum(s.avg_rs for s in top_3) / 3
    bottom_avg = sum(s.avg_rs for s in bottom_3) / 3
    gap = round(top_avg - bottom_avg, 1)

    triggered = gap > RS_DIVERGENCE_THRESHOLD

    top_names = ", ".join(f"{s.sector}({s.avg_rs:.0f})" for s in top_3)
    bottom_names = ", ".join(f"{s.sector}({s.avg_rs:.0f})" for s in bottom_3)

    return SignalReading(
        signal_name="RS Divergence",
        triggered=triggered,
        value=str(gap),
        threshold=str(RS_DIVERGENCE_THRESHOLD),
        evidence=(
            f"Top-3 avg RS={top_avg:.1f} [{top_names}] vs "
            f"Bottom-3 avg RS={bottom_avg:.1f} [{bottom_names}]. "
            f"Gap={gap} {'>' if triggered else '<='} {RS_DIVERGENCE_THRESHOLD} threshold"
        ),
    )


# ---------------------------------------------------------------------------
# Signal 2: Leadership Change
# ---------------------------------------------------------------------------

def detect_leadership_change(
    sector_rankings: List[SectorRank],
    sector_patterns: List[SectorPattern],
) -> SignalReading:
    """
    Sectors whose rank misaligns with their SectorPattern strength/trend.

    A 'leading' or 'improving' sector ranked in the bottom half, or a
    'lagging'/'declining' sector ranked in the top half, counts as a misalignment.
    Also checks cluster-level rank dispersion.

    Triggered if 2+ sectors are misaligned.
    """
    if not sector_rankings or not sector_patterns:
        return SignalReading(
            signal_name="Leadership Change",
            triggered=False,
            value="0",
            threshold=str(LEADERSHIP_MISALIGNMENT_MIN),
            evidence="Insufficient data for leadership change detection (no rankings or patterns)",
        )

    pattern_lookup = {sp.sector: sp for sp in sector_patterns}
    n = len(sector_rankings)
    midpoint = n // 2
    misaligned = []

    for sr in sector_rankings:
        sp = pattern_lookup.get(sr.sector)
        if sp is None:
            continue

        in_top_half = sr.rank <= midpoint
        is_strong = sp.strength in ("leading", "improving")
        is_weak = sp.strength in ("lagging", "declining")

        if is_strong and not in_top_half:
            misaligned.append(
                f"{sr.sector} (rank {sr.rank}, strength={sp.strength})"
            )
        elif is_weak and in_top_half:
            misaligned.append(
                f"{sr.sector} (rank {sr.rank}, strength={sp.strength})"
            )

    count = len(misaligned)
    triggered = count >= LEADERSHIP_MISALIGNMENT_MIN

    evidence_detail = "; ".join(misaligned[:5]) if misaligned else "None"

    return SignalReading(
        signal_name="Leadership Change",
        triggered=triggered,
        value=str(count),
        threshold=str(LEADERSHIP_MISALIGNMENT_MIN),
        evidence=(
            f"{count} sectors have rank-strength misalignment "
            f"(threshold={LEADERSHIP_MISALIGNMENT_MIN}). "
            f"Misaligned: {evidence_detail}"
        ),
    )


# ---------------------------------------------------------------------------
# Signal 3: Breadth Shift
# ---------------------------------------------------------------------------

def detect_breadth_shift(
    sector_rankings: List[SectorRank],
    rated_stocks: List[RatedStock],
) -> SignalReading:
    """
    Per-sector breadth = % of stocks with RS >= 80.
    Triggered if any source cluster sector < 40% and any destination
    cluster sector > 60%, and they belong to different clusters.

    Adapted for single-snapshot: cross-sectional breadth comparison.
    """
    if not sector_rankings or not rated_stocks:
        return SignalReading(
            signal_name="Breadth Shift",
            triggered=False,
            value="0.0",
            threshold=f"<{BREADTH_LOW_THRESHOLD}% vs >{BREADTH_HIGH_THRESHOLD}%",
            evidence="Insufficient data for breadth shift detection (no rankings or stocks)",
        )

    # Compute per-sector breadth
    sector_stock_count: dict[str, int] = {}
    sector_high_rs_count: dict[str, int] = {}
    for stock in rated_stocks:
        sec = stock.sector
        sector_stock_count[sec] = sector_stock_count.get(sec, 0) + 1
        if stock.rs_rating >= 80:
            sector_high_rs_count[sec] = sector_high_rs_count.get(sec, 0) + 1

    sector_breadth: dict[str, float] = {}
    for sec, total in sector_stock_count.items():
        high_rs = sector_high_rs_count.get(sec, 0)
        sector_breadth[sec] = round((high_rs / total) * 100, 1) if total > 0 else 0.0

    # Find low-breadth and high-breadth sectors
    low_breadth = []
    high_breadth = []
    for sec, breadth in sector_breadth.items():
        cluster = SECTOR_TO_CLUSTER.get(sec)
        if cluster is None:
            continue
        if breadth < BREADTH_LOW_THRESHOLD:
            low_breadth.append((sec, cluster, breadth))
        elif breadth > BREADTH_HIGH_THRESHOLD:
            high_breadth.append((sec, cluster, breadth))

    # Check cross-cluster divergence
    triggered = False
    source_example = ""
    dest_example = ""
    for low_sec, low_cluster, low_b in low_breadth:
        for high_sec, high_cluster, high_b in high_breadth:
            if low_cluster != high_cluster:
                triggered = True
                source_example = f"{low_sec}({low_cluster}, {low_b}%)"
                dest_example = f"{high_sec}({high_cluster}, {high_b}%)"
                break
        if triggered:
            break

    low_count = len(low_breadth)
    high_count = len(high_breadth)
    value_str = f"{low_count} low / {high_count} high"

    if triggered:
        evidence = (
            f"Breadth divergence detected: {source_example} below {BREADTH_LOW_THRESHOLD}% "
            f"vs {dest_example} above {BREADTH_HIGH_THRESHOLD}%. "
            f"{low_count} low-breadth and {high_count} high-breadth sectors across different clusters"
        )
    else:
        evidence = (
            f"No cross-cluster breadth divergence. "
            f"{low_count} sectors below {BREADTH_LOW_THRESHOLD}%, "
            f"{high_count} sectors above {BREADTH_HIGH_THRESHOLD}%"
        )

    return SignalReading(
        signal_name="Breadth Shift",
        triggered=triggered,
        value=value_str,
        threshold=f"<{BREADTH_LOW_THRESHOLD}% vs >{BREADTH_HIGH_THRESHOLD}%",
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Signal 4: Elite Concentration Shift
# ---------------------------------------------------------------------------

def detect_elite_concentration_shift(
    sector_rankings: List[SectorRank],
) -> SignalReading:
    """
    Weighted avg elite_pct per cluster.
    Triggered if one cluster > 50% avg elite and another < 20%.

    Uses sector_rankings' elite_pct field aggregated to cluster level.
    """
    if not sector_rankings:
        return SignalReading(
            signal_name="Elite Concentration Shift",
            triggered=False,
            value="0.0",
            threshold=f">{ELITE_CLUSTER_HIGH}% vs <{ELITE_CLUSTER_LOW}%",
            evidence="No sector rankings available for elite concentration analysis",
        )

    # Aggregate elite_pct by cluster (weighted by stock_count)
    cluster_elite_total: dict[str, float] = {}
    cluster_stock_total: dict[str, int] = {}

    for sr in sector_rankings:
        cluster = SECTOR_TO_CLUSTER.get(sr.sector)
        if cluster is None:
            continue
        cluster_elite_total[cluster] = (
            cluster_elite_total.get(cluster, 0.0) + sr.elite_pct * sr.stock_count
        )
        cluster_stock_total[cluster] = (
            cluster_stock_total.get(cluster, 0) + sr.stock_count
        )

    cluster_elite_avg: dict[str, float] = {}
    for cluster in cluster_elite_total:
        total_stocks = cluster_stock_total[cluster]
        if total_stocks > 0:
            cluster_elite_avg[cluster] = round(
                cluster_elite_total[cluster] / total_stocks, 1
            )
        else:
            cluster_elite_avg[cluster] = 0.0

    # Find high and low clusters
    high_clusters = {c: v for c, v in cluster_elite_avg.items() if v > ELITE_CLUSTER_HIGH}
    low_clusters = {c: v for c, v in cluster_elite_avg.items() if v < ELITE_CLUSTER_LOW}

    triggered = bool(high_clusters) and bool(low_clusters)

    # Build value string
    if cluster_elite_avg:
        max_cluster = max(cluster_elite_avg, key=cluster_elite_avg.get)
        min_cluster = min(cluster_elite_avg, key=cluster_elite_avg.get)
        value_str = f"{max_cluster}={cluster_elite_avg[max_cluster]}% / {min_cluster}={cluster_elite_avg[min_cluster]}%"
    else:
        value_str = "No data"

    if triggered:
        high_str = ", ".join(f"{c}({v}%)" for c, v in high_clusters.items())
        low_str = ", ".join(f"{c}({v}%)" for c, v in low_clusters.items())
        evidence = (
            f"Elite concentration divergence: high [{high_str}] "
            f"vs low [{low_str}]. "
            f"Threshold: >{ELITE_CLUSTER_HIGH}% vs <{ELITE_CLUSTER_LOW}%"
        )
    else:
        all_str = ", ".join(f"{c}({v}%)" for c, v in sorted(cluster_elite_avg.items(), key=lambda x: -x[1]))
        evidence = (
            f"No elite concentration divergence. "
            f"Cluster elite averages: {all_str}"
        )

    return SignalReading(
        signal_name="Elite Concentration Shift",
        triggered=triggered,
        value=value_str,
        threshold=f">{ELITE_CLUSTER_HIGH}% vs <{ELITE_CLUSTER_LOW}%",
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Signal 5: IBD Keep Migration
# ---------------------------------------------------------------------------

def detect_ibd_keep_migration(
    ibd_keeps: List[IBDKeep],
    rated_stocks: List[RatedStock],
) -> SignalReading:
    """
    Map IBD keeps to clusters. Triggered if dominant cluster is NOT
    "growth" or no single cluster holds > 40%.

    Rationale: In a "normal" market, IBD Keeps concentrate in growth.
    Migration away from growth indicates rotation.
    """
    if not ibd_keeps:
        return SignalReading(
            signal_name="IBD Keep Migration",
            triggered=False,
            value="0 keeps",
            threshold="Dominant cluster != growth or no cluster > 40%",
            evidence="No IBD Keep candidates found; cannot assess keep migration",
        )

    # Map keeps to sectors via rated_stocks lookup
    stock_sector: dict[str, str] = {s.symbol: s.sector for s in rated_stocks}

    cluster_counts: dict[str, int] = {}
    total_keeps = len(ibd_keeps)

    for keep in ibd_keeps:
        sector = stock_sector.get(keep.symbol)
        if sector is None:
            continue
        cluster = SECTOR_TO_CLUSTER.get(sector)
        if cluster is None:
            continue
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

    if not cluster_counts:
        return SignalReading(
            signal_name="IBD Keep Migration",
            triggered=False,
            value="0 mapped",
            threshold="Dominant cluster != growth or no cluster > 40%",
            evidence="No IBD Keeps could be mapped to clusters via rated stocks",
        )

    # Find dominant cluster
    dominant_cluster = max(cluster_counts, key=cluster_counts.get)
    dominant_pct = round((cluster_counts[dominant_cluster] / total_keeps) * 100, 1)

    # Triggered if dominant is NOT growth, or no cluster > 40%
    no_clear_dominant = dominant_pct < 40.0
    not_growth = dominant_cluster != "growth"
    triggered = not_growth or no_clear_dominant

    dist_str = ", ".join(
        f"{c}={n}" for c, n in sorted(cluster_counts.items(), key=lambda x: -x[1])
    )

    if triggered:
        if no_clear_dominant:
            reason = f"No cluster holds >40% of keeps (max: {dominant_cluster} at {dominant_pct}%)"
        else:
            reason = f"Dominant cluster is {dominant_cluster} ({dominant_pct}%), not growth"
        evidence = (
            f"IBD Keep migration detected: {reason}. "
            f"Distribution ({total_keeps} keeps): {dist_str}"
        )
    else:
        evidence = (
            f"IBD Keeps concentrated in growth cluster ({dominant_pct}%). "
            f"Distribution ({total_keeps} keeps): {dist_str}"
        )

    return SignalReading(
        signal_name="IBD Keep Migration",
        triggered=triggered,
        value=f"{dominant_cluster}={dominant_pct}%",
        threshold="Dominant cluster != growth or no cluster > 40%",
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# ETF Sector Evidence
# ---------------------------------------------------------------------------

def _compute_etf_sector_evidence(
    rated_etfs: list,
    sector_rankings: List[SectorRank],
) -> dict[str, dict]:
    """
    Map each ETF's theme_tags to sectors to clusters and compute
    per-cluster: avg RS, avg volume%, ETF count.

    Returns dict[cluster_name, {"avg_rs": float, "avg_volume_pct": float, "etf_count": int}]
    """
    from ibd_agents.schemas.strategy_output import THEME_SECTOR_MAP

    cluster_rs: dict[str, list[float]] = {}
    cluster_vol: dict[str, list[float]] = {}

    for etf in rated_etfs:
        # Map ETF theme_tags -> sectors -> clusters
        etf_clusters: set[str] = set()
        for tag in getattr(etf, "theme_tags", []):
            related_sectors = THEME_SECTOR_MAP.get(tag, [])
            for sector in related_sectors:
                cluster = SECTOR_TO_CLUSTER.get(sector)
                if cluster:
                    etf_clusters.add(cluster)

        # If no themes mapped, skip this ETF
        if not etf_clusters:
            continue

        rs = getattr(etf, "rs_rating", None)
        vol_pct = getattr(etf, "volume_pct_change", None)

        for cluster in etf_clusters:
            if rs is not None:
                cluster_rs.setdefault(cluster, []).append(float(rs))
            if vol_pct is not None:
                cluster_vol.setdefault(cluster, []).append(float(vol_pct))

    result: dict[str, dict] = {}
    all_clusters = set(cluster_rs.keys()) | set(cluster_vol.keys())
    for cluster in all_clusters:
        rs_vals = cluster_rs.get(cluster, [])
        vol_vals = cluster_vol.get(cluster, [])
        result[cluster] = {
            "avg_rs": round(sum(rs_vals) / len(rs_vals), 1) if rs_vals else 0.0,
            "avg_volume_pct": round(sum(vol_vals) / len(vol_vals), 1) if vol_vals else 0.0,
            "etf_count": max(len(rs_vals), len(vol_vals)),
        }

    return result


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def compute_all_signals(
    analyst_output: AnalystOutput,
    research_output: ResearchOutput,
    rated_etfs: list = None,
) -> RotationSignals:
    """
    Run all 5 signal detectors and return aggregated RotationSignals.

    If rated_etfs is provided, annotates Signal 1 (rs_divergence) with
    ETF RS comparison across clusters and Signal 3 (breadth_shift) with
    ETF volume data across clusters.
    """
    sig1 = detect_rs_divergence(analyst_output.sector_rankings)
    sig2 = detect_leadership_change(
        analyst_output.sector_rankings, research_output.sector_patterns
    )
    sig3 = detect_breadth_shift(
        analyst_output.sector_rankings, analyst_output.rated_stocks
    )
    sig4 = detect_elite_concentration_shift(analyst_output.sector_rankings)
    sig5 = detect_ibd_keep_migration(
        analyst_output.ibd_keeps, analyst_output.rated_stocks
    )

    # Annotate signals with ETF evidence if available
    if rated_etfs:
        etf_evidence = _compute_etf_sector_evidence(
            rated_etfs, analyst_output.sector_rankings
        )
        if etf_evidence:
            # Signal 1: RS Divergence — annotate with ETF RS across clusters
            rs_parts = []
            for cluster, data in sorted(etf_evidence.items(), key=lambda x: -x[1]["avg_rs"]):
                if data["avg_rs"] > 0:
                    rs_parts.append(f"{cluster}: RS={data['avg_rs']:.0f} ({data['etf_count']} ETFs)")
            if rs_parts:
                sig1 = SignalReading(
                    signal_name=sig1.signal_name,
                    triggered=sig1.triggered,
                    value=sig1.value,
                    threshold=sig1.threshold,
                    evidence=sig1.evidence,
                    etf_confirmation=f"ETF RS by cluster: {'; '.join(rs_parts)}",
                )

            # Signal 3: Breadth Shift — annotate with ETF volume data
            vol_parts = []
            for cluster, data in sorted(etf_evidence.items(), key=lambda x: -x[1]["avg_volume_pct"]):
                if data["etf_count"] > 0:
                    vol_parts.append(
                        f"{cluster}: vol={data['avg_volume_pct']:+.1f}% ({data['etf_count']} ETFs)"
                    )
            if vol_parts:
                sig3 = SignalReading(
                    signal_name=sig3.signal_name,
                    triggered=sig3.triggered,
                    value=sig3.value,
                    threshold=sig3.threshold,
                    evidence=sig3.evidence,
                    etf_confirmation=f"ETF volume by cluster: {'; '.join(vol_parts)}",
                )

    active = sum(1 for s in [sig1, sig2, sig3, sig4, sig5] if s.triggered)

    return RotationSignals(
        rs_divergence=sig1,
        leadership_change=sig2,
        breadth_shift=sig3,
        elite_concentration_shift=sig4,
        ibd_keep_migration=sig5,
        signals_active=active,
    )


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class RotationSignalInput(BaseModel):
    analyst_json: str = Field(..., description="JSON string of AnalystOutput")
    research_json: str = Field(..., description="JSON string of ResearchOutput")


class RotationSignalTool(BaseTool):
    """Compute all 5 rotation detection signals."""

    name: str = "rotation_signal_detector"
    description: str = (
        "Compute 5-signal rotation framework: RS Divergence, "
        "Leadership Change, Breadth Shift, Elite Concentration Shift, "
        "IBD Keep Migration"
    )
    args_schema: type[BaseModel] = RotationSignalInput

    def _run(self, analyst_json: str, research_json: str) -> str:
        analyst = AnalystOutput.model_validate_json(analyst_json)
        research = ResearchOutput.model_validate_json(research_json)
        signals = compute_all_signals(analyst, research)
        return signals.model_dump_json(indent=2)
