"""
Agent 12: PatternAlpha — Pattern Analyzer Tool
Per-Stock Pattern Scoring Engine + Portfolio-Level Pattern Analysis

Deterministic functions for 100-point base scoring, enhanced rating
classification, per-stock pattern alerts, and tier fit assessment.
LLM function for 5-pattern scoring (Platform Economics, Self-Cannibalization,
Capital Allocation, Category Creation, Inflection Timing).
Also retains portfolio-level analysis functions (sector concentration,
momentum clustering, regime fit, etc.) for secondary analysis.

Never makes buy/sell recommendations — only discovers, scores, and organizes.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

from ibd_agents.schemas.rotation_output import SECTOR_CLUSTERS, SECTOR_TO_CLUSTER
from ibd_agents.schemas.strategy_output import TIER_ALLOCATION_TARGETS
from ibd_agents.schemas.pattern_output import (
    PATTERN_MAX_SCORES,
    ENHANCED_RATING_BANDS,
    PATTERN_ALERT_TYPES,
    TIER_PATTERN_REQUIREMENTS,
)
from ibd_agents.tools.token_tracker import track as track_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Sector Concentrations
# ---------------------------------------------------------------------------

def compute_sector_concentrations(positions: list[dict]) -> list[dict]:
    """
    Group portfolio positions by sector and compute concentration metrics.

    Args:
        positions: List of dicts with keys: symbol, sector, asset_type,
            target_pct, conviction, rs_rating.

    Returns:
        List of sector concentration dicts sorted by weight_pct descending.
        Each dict has: sector, weight_pct, stock_count, etf_count,
        avg_rs, avg_conviction, is_overconcentrated.
    """
    if not positions:
        return []

    sectors: dict[str, dict] = {}
    for pos in positions:
        sector = pos.get("sector", "UNKNOWN")
        if sector not in sectors:
            sectors[sector] = {
                "sector": sector,
                "weight_pct": 0.0,
                "stock_count": 0,
                "etf_count": 0,
                "rs_values": [],
                "conviction_values": [],
            }
        entry = sectors[sector]
        entry["weight_pct"] += pos.get("target_pct", 0.0)
        asset_type = str(pos.get("asset_type", "stock")).lower()
        if asset_type == "etf":
            entry["etf_count"] += 1
        else:
            entry["stock_count"] += 1
        rs = pos.get("rs_rating")
        if rs is not None:
            entry["rs_values"].append(float(rs))
        conv = pos.get("conviction")
        if conv is not None:
            entry["conviction_values"].append(float(conv))

    result = []
    for entry in sectors.values():
        rs_vals = entry.pop("rs_values")
        conv_vals = entry.pop("conviction_values")
        entry["avg_rs"] = round(sum(rs_vals) / len(rs_vals), 1) if rs_vals else None
        entry["avg_conviction"] = (
            round(sum(conv_vals) / len(conv_vals), 2) if conv_vals else None
        )
        entry["weight_pct"] = round(entry["weight_pct"], 2)
        entry["is_overconcentrated"] = entry["weight_pct"] > 25.0
        result.append(entry)

    result.sort(key=lambda x: x["weight_pct"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# 2. Momentum Clusters
# ---------------------------------------------------------------------------

def detect_momentum_clusters(positions: list[dict]) -> list[dict]:
    """
    Map positions into SECTOR_CLUSTERS and detect active clusters.

    A cluster is "active" if it has >= 3 positions.

    Args:
        positions: List of dicts with keys: symbol, sector, asset_type,
            target_pct, conviction, rs_rating, eps_rating.

    Returns:
        List of active cluster dicts sorted by total_weight descending.
        Each dict has: cluster_name, sectors, symbols, position_count,
        total_weight_pct, avg_rs, avg_eps, correlation_risk, regime_fit.
    """
    if not positions:
        return []

    clusters: dict[str, dict] = {}
    for pos in positions:
        sector = pos.get("sector", "UNKNOWN")
        cluster_name = SECTOR_TO_CLUSTER.get(sector)
        if cluster_name is None:
            continue

        if cluster_name not in clusters:
            clusters[cluster_name] = {
                "cluster_name": cluster_name,
                "sectors_seen": set(),
                "symbols": [],
                "position_count": 0,
                "total_weight_pct": 0.0,
                "rs_values": [],
                "eps_values": [],
            }
        cl = clusters[cluster_name]
        cl["sectors_seen"].add(sector)
        cl["symbols"].append(pos.get("symbol", "???"))
        cl["position_count"] += 1
        cl["total_weight_pct"] += pos.get("target_pct", 0.0)
        rs = pos.get("rs_rating")
        if rs is not None:
            cl["rs_values"].append(float(rs))
        eps = pos.get("eps_rating")
        if eps is not None:
            cl["eps_values"].append(float(eps))

    result = []
    for cl in clusters.values():
        if cl["position_count"] < 3:
            continue

        total_weight = round(cl["total_weight_pct"], 2)
        count = cl["position_count"]
        rs_vals = cl["rs_values"]
        eps_vals = cl["eps_values"]

        # Correlation risk assessment
        if total_weight >= 20.0 and count >= 4:
            correlation_risk = "High"
        elif total_weight >= 12.0 and count >= 3:
            correlation_risk = "Moderate"
        else:
            correlation_risk = "Low"

        # Regime fit (simplified — actual regime integration in analyze_regime_fit)
        cluster_name = cl["cluster_name"]
        if cluster_name in ("growth", "commodity"):
            regime_fit = "Strong Fit"  # bullish tilt
        elif cluster_name in ("defensive", "financial"):
            regime_fit = "Moderate Fit"
        else:
            regime_fit = "Moderate Fit"

        result.append({
            "cluster_name": cluster_name,
            "sectors": sorted(cl["sectors_seen"]),
            "symbols": cl["symbols"],
            "position_count": count,
            "total_weight_pct": total_weight,
            "avg_rs": round(sum(rs_vals) / len(rs_vals), 1) if rs_vals else 0.0,
            "avg_eps": round(sum(eps_vals) / len(eps_vals), 1) if eps_vals else None,
            "correlation_risk": correlation_risk,
            "regime_fit": regime_fit,
        })

    result.sort(key=lambda x: x["total_weight_pct"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# 3. Value-Momentum Divergences
# ---------------------------------------------------------------------------

def analyze_value_momentum_divergences(
    positions: list[dict],
    value_data: dict[str, dict],
) -> list[dict]:
    """
    Find portfolio positions where momentum and value signals diverge.

    Args:
        positions: Portfolio positions with symbol, target_pct, rs_rating.
        value_data: {symbol: {value_score, value_category,
            momentum_value_alignment}} from ValueInvestorOutput.

    Returns:
        List of divergence dicts sorted by value_score descending, up to 20.
        Each dict has: symbol, rs_rating, value_score, value_category,
        momentum_value_alignment, in_portfolio, position_pct.
    """
    if not positions or not value_data:
        return []

    divergences = []
    for pos in positions:
        symbol = pos.get("symbol", "")
        vd = value_data.get(symbol)
        if vd is None:
            continue

        alignment = vd.get("momentum_value_alignment", "")
        if alignment not in ("Strong Mismatch", "Mild Mismatch"):
            continue

        divergences.append({
            "symbol": symbol,
            "rs_rating": pos.get("rs_rating", 0),
            "value_score": vd.get("value_score", 0.0),
            "value_category": vd.get("value_category", "Not Value"),
            "momentum_value_alignment": alignment,
            "in_portfolio": True,
            "position_pct": pos.get("target_pct", 0.0),
        })

    divergences.sort(key=lambda x: x["value_score"], reverse=True)
    return divergences[:20]


# ---------------------------------------------------------------------------
# 4. Tier Efficiency
# ---------------------------------------------------------------------------

def compute_tier_efficiency(tier_actual_pcts: dict[int, float]) -> list[dict]:
    """
    Compare actual tier allocations against TIER_ALLOCATION_TARGETS.

    Args:
        tier_actual_pcts: {1: actual_pct, 2: actual_pct, 3: actual_pct}

    Returns:
        List of 3 dicts (one per tier), each with: tier, target_pct,
        actual_pct, delta_pct, status.
    """
    # Map tier number to TIER_ALLOCATION_TARGETS key
    tier_key_map = {
        1: "tier_1_momentum",
        2: "tier_2_quality",
        3: "tier_3_defensive",
    }

    result = []
    for tier_num in (1, 2, 3):
        key = tier_key_map[tier_num]
        target = TIER_ALLOCATION_TARGETS[key]["target_pct"]
        actual = tier_actual_pcts.get(tier_num, 0.0)
        delta = round(actual - target, 2)

        if abs(delta) <= 3:
            status = "On Target"
        elif delta > 3:
            status = "Overweight"
        else:
            status = "Underweight"

        result.append({
            "tier": tier_num,
            "target_pct": float(target),
            "actual_pct": round(actual, 2),
            "delta_pct": delta,
            "status": status,
        })

    return result


# ---------------------------------------------------------------------------
# 5. Regime Fit Analysis
# ---------------------------------------------------------------------------

def analyze_regime_fit(
    regime: str,
    tier_actual_pcts: dict[int, float],
    avg_portfolio_rs: float,
    top_cluster: str,
) -> dict:
    """
    Assess how well the portfolio fits the current market regime.

    Args:
        regime: "bull", "bear", or "neutral" from rotation detector.
        tier_actual_pcts: {1: actual_pct, 2: actual_pct, 3: actual_pct}
        avg_portfolio_rs: Average RS rating across portfolio.
        top_cluster: Name of the highest-weight cluster.

    Returns:
        Dict with: detected_regime, portfolio_tilt, regime_fit_label,
        regime_fit_score, misalignment_details.
    """
    t1 = tier_actual_pcts.get(1, 0.0)
    t3 = tier_actual_pcts.get(3, 0.0)
    misalignments = []
    score = 50.0  # start at neutral

    # Determine portfolio tilt
    if t1 >= t3 + 10:
        portfolio_tilt = "Offensive"
    elif t3 >= t1 + 5:
        portfolio_tilt = "Defensive"
    else:
        portfolio_tilt = "Balanced"

    regime = regime.lower() if regime else "neutral"

    if regime == "bull":
        # In bull: good if T1 >= 35% and avg_rs >= 80
        if t1 >= 35:
            score += 20
        else:
            misalignments.append(
                f"T1 allocation {t1:.1f}% below 35% target for bull regime"
            )
            score -= 15

        if avg_portfolio_rs >= 80:
            score += 15
        else:
            misalignments.append(
                f"Avg RS {avg_portfolio_rs:.0f} below 80 for bull regime"
            )
            score -= 10

        if top_cluster in ("growth", "commodity"):
            score += 15
        else:
            misalignments.append(
                f"Top cluster '{top_cluster}' is not growth/commodity for bull regime"
            )

    elif regime == "bear":
        # In bear: good if T3 >= 25% and defensive cluster is top
        if t3 >= 25:
            score += 20
        else:
            misalignments.append(
                f"T3 allocation {t3:.1f}% below 25% target for bear regime"
            )
            score -= 15

        if top_cluster == "defensive":
            score += 15
        else:
            misalignments.append(
                f"Top cluster '{top_cluster}' is not defensive for bear regime"
            )
            score -= 10

        if avg_portfolio_rs < 80:
            score += 10  # lower RS is expected in bear
        else:
            misalignments.append(
                f"Avg RS {avg_portfolio_rs:.0f} unusually high for bear regime"
            )
            score -= 5

    else:  # neutral
        # In neutral: balanced is good
        if abs(t1 - 39) <= 5:
            score += 15
        if abs(t3 - 22) <= 5:
            score += 10
        if 70 <= avg_portfolio_rs <= 90:
            score += 10
        if not misalignments:
            score += 5

    # Clamp score 0-100
    score = max(0.0, min(100.0, score))

    # Determine label from score
    if score >= 75:
        label = "Strong Fit"
    elif score >= 50:
        label = "Moderate Fit"
    elif score >= 30:
        label = "Weak Fit"
    else:
        label = "Misaligned"

    return {
        "detected_regime": regime,
        "portfolio_tilt": portfolio_tilt,
        "regime_fit_label": label,
        "regime_fit_score": round(score, 1),
        "misalignment_details": misalignments,
    }


# ---------------------------------------------------------------------------
# 6. Portfolio Profile
# ---------------------------------------------------------------------------

def build_portfolio_profile(
    concentrations: list[dict],
    clusters: list[dict],
    tier_eff: list[dict],
    regime_fit: dict,
    positions: list[dict],
    value_data: dict[str, dict],
) -> dict:
    """
    Build a categorical portfolio profile from all pattern analyses.

    Args:
        concentrations: Output of compute_sector_concentrations.
        clusters: Output of detect_momentum_clusters.
        tier_eff: Output of compute_tier_efficiency.
        regime_fit: Output of analyze_regime_fit.
        positions: Original portfolio positions.
        value_data: {symbol: {value_score, value_category, ...}}

    Returns:
        Dict with: sector_profile, risk_profile, momentum_concentration_pct,
        value_coverage_pct, morningstar_coverage_pct, avg_conviction,
        conviction_sizing_correlation, regime_alignment.
    """
    # --- sector_profile ---
    top_cluster_name = clusters[0]["cluster_name"] if clusters else ""
    top_cluster_weight = clusters[0]["total_weight_pct"] if clusters else 0.0

    if top_cluster_name == "growth" and top_cluster_weight >= 30:
        sector_profile = "Growth-Concentrated"
    elif top_cluster_name == "defensive" and top_cluster_weight >= 25:
        sector_profile = "Defensive-Tilted"
    elif not clusters or all(c["total_weight_pct"] <= 20 for c in clusters):
        sector_profile = "Balanced"
    else:
        sector_profile = "Mixed"

    # --- risk_profile ---
    rs_values = [
        float(p["rs_rating"]) for p in positions if p.get("rs_rating") is not None
    ]
    avg_rs = sum(rs_values) / len(rs_values) if rs_values else 0.0

    # Get T1 and T3 actual percentages from tier efficiency
    t1_actual = 0.0
    t3_actual = 0.0
    for te in tier_eff:
        if te["tier"] == 1:
            t1_actual = te["actual_pct"]
        elif te["tier"] == 3:
            t3_actual = te["actual_pct"]

    if avg_rs >= 85 and t1_actual >= 40:
        risk_profile = "Aggressive"
    elif avg_rs < 75 and t3_actual >= 25:
        risk_profile = "Conservative"
    else:
        risk_profile = "Moderate"

    # --- momentum_concentration_pct ---
    total_positions = len(positions) if positions else 1
    high_rs_count = sum(1 for p in positions if p.get("rs_rating", 0) >= 85)
    momentum_concentration_pct = round(
        (high_rs_count / total_positions) * 100, 1
    )

    # --- value_coverage_pct ---
    value_count = 0
    for p in positions:
        sym = p.get("symbol", "")
        vd = value_data.get(sym)
        if vd and vd.get("value_category", "Not Value") != "Not Value":
            value_count += 1
    value_coverage_pct = round((value_count / total_positions) * 100, 1)

    # --- morningstar_coverage_pct ---
    ms_count = 0
    for vd in value_data.values():
        if vd.get("value_score", 0) > 0:
            ms_count += 1
    total_value_entries = len(value_data) if value_data else 1
    morningstar_coverage_pct = round(
        (ms_count / total_value_entries) * 100, 1
    )

    # --- avg_conviction ---
    conv_values = [
        float(p["conviction"]) for p in positions if p.get("conviction") is not None
    ]
    avg_conviction = (
        round(sum(conv_values) / len(conv_values), 2) if conv_values else 5.0
    )

    # --- conviction_sizing_correlation (Pearson r) ---
    pairs = [
        (float(p["conviction"]), float(p["target_pct"]))
        for p in positions
        if p.get("conviction") is not None and p.get("target_pct") is not None
    ]
    conviction_sizing_correlation = _pearson_r(pairs)

    # --- regime_alignment ---
    regime_alignment = regime_fit.get("regime_fit_label", "Moderate Fit")

    return {
        "sector_profile": sector_profile,
        "risk_profile": risk_profile,
        "momentum_concentration_pct": momentum_concentration_pct,
        "value_coverage_pct": value_coverage_pct,
        "morningstar_coverage_pct": morningstar_coverage_pct,
        "avg_conviction": avg_conviction,
        "conviction_sizing_correlation": conviction_sizing_correlation,
        "regime_alignment": regime_alignment,
    }


def _pearson_r(pairs: list[tuple[float, float]]) -> float:
    """
    Compute Pearson correlation coefficient for (x, y) pairs.

    Returns 0.0 if fewer than 3 pairs or zero variance.
    """
    n = len(pairs)
    if n < 3:
        return 0.0

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / n
    var_x = sum((x - mean_x) ** 2 for x in xs) / n
    var_y = sum((y - mean_y) ** 2 for y in ys) / n

    if var_x == 0 or var_y == 0:
        return 0.0

    r = cov / (math.sqrt(var_x) * math.sqrt(var_y))
    return round(max(-1.0, min(1.0, r)), 3)


# ---------------------------------------------------------------------------
# 7. Pattern Alerts
# ---------------------------------------------------------------------------

def generate_pattern_alerts(
    concentrations: list[dict],
    clusters: list[dict],
    divergences: list[dict],
    tier_eff: list[dict],
    regime_fit: dict,
    risk_warnings: list[str],
) -> list[dict]:
    """
    Generate actionable alerts from detected portfolio patterns.

    Alert severities: High > Medium > Low.

    Args:
        concentrations: Output of compute_sector_concentrations.
        clusters: Output of detect_momentum_clusters.
        divergences: Output of analyze_value_momentum_divergences.
        tier_eff: Output of compute_tier_efficiency.
        regime_fit: Output of analyze_regime_fit.
        risk_warnings: Additional risk warning strings.

    Returns:
        List of alert dicts sorted by severity (High first), each with:
        pattern_name, severity, description, affected_symbols, recommendation.
    """
    alerts: list[dict] = []

    # --- HIGH severity alerts ---

    # Any sector > 30% weight
    for conc in concentrations:
        if conc["weight_pct"] > 30:
            alerts.append({
                "pattern_name": "Sector Overconcentration",
                "severity": "High",
                "description": (
                    f"Sector '{conc['sector']}' represents {conc['weight_pct']:.1f}% "
                    f"of the portfolio, exceeding the 30% critical threshold"
                ),
                "affected_symbols": [],
                "recommendation": (
                    "Reduce exposure by trimming positions or diversifying "
                    "into underweight sectors"
                ),
            })

    # Any cluster with correlation_risk == "High"
    for cl in clusters:
        if cl.get("correlation_risk") == "High":
            alerts.append({
                "pattern_name": "Momentum Cluster Risk",
                "severity": "High",
                "description": (
                    f"Cluster '{cl['cluster_name']}' has {cl['position_count']} "
                    f"positions totaling {cl['total_weight_pct']:.1f}% with high "
                    f"correlation risk — correlated drawdowns likely"
                ),
                "affected_symbols": cl.get("symbols", []),
                "recommendation": (
                    "Diversify across clusters or add hedging positions "
                    "to reduce correlated exposure"
                ),
            })

    # Regime misalignment (score < 30)
    if regime_fit.get("regime_fit_score", 50) < 30:
        alerts.append({
            "pattern_name": "Regime Misalignment",
            "severity": "High",
            "description": (
                f"Portfolio is misaligned with the detected '{regime_fit.get('detected_regime', 'unknown')}' "
                f"regime (fit score {regime_fit.get('regime_fit_score', 0):.0f}/100). "
                f"Misalignments: {'; '.join(regime_fit.get('misalignment_details', []))}"
            ),
            "affected_symbols": [],
            "recommendation": (
                "Rebalance tier allocations to align with current market regime"
            ),
        })

    # --- MEDIUM severity alerts ---

    # Any sector > 25% weight (not already flagged as > 30%)
    for conc in concentrations:
        if 25 < conc["weight_pct"] <= 30:
            alerts.append({
                "pattern_name": "Elevated Sector Weight",
                "severity": "Medium",
                "description": (
                    f"Sector '{conc['sector']}' at {conc['weight_pct']:.1f}% "
                    f"is above the 25% concentration warning level"
                ),
                "affected_symbols": [],
                "recommendation": (
                    "Monitor closely and consider trimming if sector momentum weakens"
                ),
            })

    # 3+ value trap stocks in portfolio
    value_trap_symbols = [
        d["symbol"] for d in divergences
        if d.get("momentum_value_alignment") == "Strong Mismatch"
    ]
    if len(value_trap_symbols) >= 3:
        alerts.append({
            "pattern_name": "Value Trap Exposure",
            "severity": "Medium",
            "description": (
                f"Portfolio contains {len(value_trap_symbols)} positions with strong "
                f"value-momentum mismatches, indicating potential value trap exposure"
            ),
            "affected_symbols": value_trap_symbols[:10],
            "recommendation": (
                "Review positions with high value scores but weak momentum "
                "for fundamental deterioration"
            ),
        })

    # Tier allocation drift > 5% for any tier
    for te in tier_eff:
        if abs(te["delta_pct"]) > 5:
            alerts.append({
                "pattern_name": "Tier Allocation Drift",
                "severity": "Medium",
                "description": (
                    f"Tier {te['tier']} allocation at {te['actual_pct']:.1f}% "
                    f"is {abs(te['delta_pct']):.1f}pp away from target "
                    f"{te['target_pct']:.1f}% ({te['status']})"
                ),
                "affected_symbols": [],
                "recommendation": (
                    f"Rebalance Tier {te['tier']} toward the "
                    f"{te['target_pct']:.0f}% target allocation"
                ),
            })

    # --- LOW severity alerts ---

    # Weak conviction-sizing link (handled by caller via profile, but
    # we accept risk_warnings for this)
    for warning in risk_warnings:
        if "conviction" in warning.lower() and "sizing" in warning.lower():
            alerts.append({
                "pattern_name": "Weak Conviction-Sizing Link",
                "severity": "Low",
                "description": (
                    f"Position sizes do not correlate well with conviction "
                    f"ratings, suggesting allocation decisions may not reflect "
                    f"analyst confidence levels"
                ),
                "affected_symbols": [],
                "recommendation": (
                    "Align position sizes more closely with conviction ratings"
                ),
            })
        elif "value coverage" in warning.lower():
            alerts.append({
                "pattern_name": "Low Value Coverage",
                "severity": "Low",
                "description": (
                    f"Less than 20% of portfolio positions have value category "
                    f"classifications, limiting value-based risk assessment"
                ),
                "affected_symbols": [],
                "recommendation": (
                    "Expand Morningstar/value screening coverage for better "
                    "risk visibility"
                ),
            })

    # Sort: High first, then Medium, then Low
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    alerts.sort(key=lambda a: severity_order.get(a["severity"], 9))

    return alerts


# ---------------------------------------------------------------------------
# 8. Per-Stock Base Scoring (100 pts deterministic)
# ---------------------------------------------------------------------------

_SHARPE_PTS: dict[str, int] = {
    "Excellent": 10, "Good": 8, "Moderate": 5, "Below Average": 2,
}
_ALPHA_PTS: dict[str, int] = {
    "Strong Outperformer": 10, "Outperformer": 7,
    "Slight Underperformer": 3, "Underperformer": 0,
}
_RISK_PTS: dict[str, int] = {
    "Excellent": 10, "Good": 8, "Moderate": 5,
    "Below Average": 2, "Poor": 0,
}


def compute_base_score(stock: dict) -> dict:
    """
    Compute 100-point base score from existing analyst metrics.

    Args:
        stock: Dict with keys from RatedStock: composite_rating, rs_rating,
            eps_rating, validation_score, is_ibd_keep, is_multi_source_validated,
            sharpe_category, alpha_category, risk_rating.

    Returns:
        Dict with ibd_score, analyst_score, risk_score, base_total.
    """
    # --- IBD Score (40 pts) ---
    composite = float(stock.get("composite_rating", 50))
    rs = float(stock.get("rs_rating", 50))
    eps = float(stock.get("eps_rating", 50))

    composite_pts = (composite / 99.0) * 15.0  # 0-15
    rs_pts = (rs / 99.0) * 15.0                # 0-15
    eps_pts = (eps / 99.0) * 10.0              # 0-10
    ibd_score = round(composite_pts + rs_pts + eps_pts, 2)
    ibd_score = min(40.0, ibd_score)

    # --- Analyst Score (30 pts) ---
    validation_pts = min(float(stock.get("validation_score", 0)), 10.0)
    keep_pts = 10.0 if stock.get("is_ibd_keep", False) else 0.0
    multi_pts = 10.0 if stock.get("is_multi_source_validated", False) else 0.0
    analyst_score = round(validation_pts + keep_pts + multi_pts, 2)
    analyst_score = min(30.0, analyst_score)

    # --- Risk Score (30 pts) ---
    sharpe_pts = float(_SHARPE_PTS.get(
        stock.get("sharpe_category", ""), 3
    ))
    alpha_pts = float(_ALPHA_PTS.get(
        stock.get("alpha_category", ""), 3
    ))
    risk_pts = float(_RISK_PTS.get(
        stock.get("risk_rating", ""), 3
    ))
    risk_score = round(sharpe_pts + alpha_pts + risk_pts, 2)
    risk_score = min(30.0, risk_score)

    base_total = round(ibd_score + analyst_score + risk_score)
    base_total = min(100, base_total)

    return {
        "ibd_score": ibd_score,
        "analyst_score": analyst_score,
        "risk_score": risk_score,
        "base_total": base_total,
    }


def classify_enhanced_rating(score: int) -> tuple[str, str]:
    """
    Map enhanced score (0-150) to (star_rating, label).

    Returns:
        Tuple of (stars, label) e.g. ("★★★★★+", "Max Conviction").
    """
    for lo, hi, stars, label in ENHANCED_RATING_BANDS:
        if lo <= score <= hi:
            return stars, label
    return "★", "Review"


# ---------------------------------------------------------------------------
# 9. Per-Stock Pattern Alerts
# ---------------------------------------------------------------------------

def generate_stock_pattern_alerts(
    symbol: str,
    breakdown: dict,
) -> list[str]:
    """
    Generate pattern alert strings for a single stock based on its 5-pattern scores.

    Alert types from spec:
    - Category King: P4 >= 8
    - Inflection Alert: P5 >= 6
    - Disruption Risk: P2 = 0
    - Pattern Imbalance: any pattern = 0 while another >= 8

    Args:
        symbol: Stock ticker.
        breakdown: Dict with keys p1, p2, p3, p4, p5 (int scores).

    Returns:
        List of alert type strings for this stock.
    """
    alerts: list[str] = []

    p1 = breakdown.get("p1", 0)
    p2 = breakdown.get("p2", 0)
    p3 = breakdown.get("p3", 0)
    p4 = breakdown.get("p4", 0)
    p5 = breakdown.get("p5", 0)

    if p4 >= 8:
        alerts.append("Category King")
    if p5 >= 6:
        alerts.append("Inflection Alert")
    if p2 == 0:
        alerts.append("Disruption Risk")

    # Pattern Imbalance: any pattern = 0 while another >= 8
    scores = [p1, p2, p3, p4, p5]
    if any(s == 0 for s in scores) and any(s >= 8 for s in scores):
        if "Pattern Imbalance" not in alerts:
            alerts.append("Pattern Imbalance")

    return alerts


# ---------------------------------------------------------------------------
# 10. Tier Fit Assessment
# ---------------------------------------------------------------------------

def assess_tier_fit(
    enhanced_score: int,
    breakdown: dict | None,
    current_tier: int,
) -> str:
    """
    Assess whether a stock meets the pattern requirements for its current tier.

    Tier 1: enhanced >= 115 and >= 2 patterns scoring at 7+
    Tier 2: enhanced >= 95 and >= 2 patterns scoring at 5+
    Tier 3: enhanced >= 80 and P3 Capital Allocation preferably at 7+

    Args:
        enhanced_score: The 150-point enhanced score.
        breakdown: Dict with keys p1, p2, p3, p4, p5. None if no pattern scoring.
        current_tier: Stock's current tier (1, 2, or 3).

    Returns:
        String assessment of tier fit.
    """
    if breakdown is None:
        return f"Tier {current_tier}: Pattern data unavailable for full assessment"

    req = TIER_PATTERN_REQUIREMENTS.get(current_tier)
    if req is None:
        return f"Tier {current_tier}: No requirements defined"

    scores = [
        breakdown.get("p1", 0),
        breakdown.get("p2", 0),
        breakdown.get("p3", 0),
        breakdown.get("p4", 0),
        breakdown.get("p5", 0),
    ]

    if current_tier == 1:
        min_enh = req["min_enhanced"]
        min_at_7 = req["min_patterns_at_7"]
        count_at_7 = sum(1 for s in scores if s >= 7)
        meets_score = enhanced_score >= min_enh
        meets_patterns = count_at_7 >= min_at_7
        if meets_score and meets_patterns:
            return f"Tier 1: Strong fit — Enhanced {enhanced_score} >= {min_enh}, {count_at_7} patterns at 7+"
        parts = []
        if not meets_score:
            parts.append(f"Enhanced {enhanced_score} < {min_enh}")
        if not meets_patterns:
            parts.append(f"Only {count_at_7} patterns at 7+ (need {min_at_7})")
        return f"Tier 1: Gap — {'; '.join(parts)}"

    elif current_tier == 2:
        min_enh = req["min_enhanced"]
        min_at_5 = req["min_patterns_at_5"]
        count_at_5 = sum(1 for s in scores if s >= 5)
        meets_score = enhanced_score >= min_enh
        meets_patterns = count_at_5 >= min_at_5
        if meets_score and meets_patterns:
            return f"Tier 2: Strong fit — Enhanced {enhanced_score} >= {min_enh}, {count_at_5} patterns at 5+"
        parts = []
        if not meets_score:
            parts.append(f"Enhanced {enhanced_score} < {min_enh}")
        if not meets_patterns:
            parts.append(f"Only {count_at_5} patterns at 5+ (need {min_at_5})")
        return f"Tier 2: Gap — {'; '.join(parts)}"

    else:  # tier 3
        min_enh = req["min_enhanced"]
        preferred_p3 = req["preferred_p3"]
        p3_score = breakdown.get("p3", 0)
        meets_score = enhanced_score >= min_enh
        meets_p3 = p3_score >= preferred_p3
        if meets_score and meets_p3:
            return f"Tier 3: Strong fit — Enhanced {enhanced_score} >= {min_enh}, P3={p3_score} >= {preferred_p3}"
        parts = []
        if not meets_score:
            parts.append(f"Enhanced {enhanced_score} < {min_enh}")
        if not meets_p3:
            parts.append(f"P3 Capital Allocation {p3_score} < preferred {preferred_p3}")
        return f"Tier 3: {'Adequate' if meets_score else 'Gap'} — {'; '.join(parts)}"


# ---------------------------------------------------------------------------
# 11. LLM Pattern Scoring (5 patterns per stock)
# ---------------------------------------------------------------------------

_PATTERN_SCORING_PROMPT_TEMPLATE = """\
Score each stock against 5 wealth-creation patterns. Return ONLY valid JSON array.

P1: Platform Economics (0-12) = Network Effects(0-4) + Switching Costs(0-4) + Ecosystem Lock-in(0-4)
P2: Self-Cannibalization (0-10) = Active Cannibal(0-4) + Mgmt Track Record(0-3) + R&D Intensity(0-3)
P3: Capital Allocation (0-10) = ROIC vs WACC(0-4) + Deployment(0-3) + Insider Alignment(0-3)
P4: Category Creation (0-10) = Category Ownership(0-4) + TAM Expansion(0-3) + Moat Depth(0-3)
P5: Inflection Timing (0-8) = Inflection ID(0-3) + Rev Acceleration(0-3) + Consensus(0-2)

Per stock return: symbol, p1(0-12), p2(0-10), p3(0-10), p4(0-10), p5(0-8),
p1_justification(1 sentence), p2_justification, p3_justification,
p4_justification, p5_justification,
dominant_pattern(str), pattern_narrative(2-3 sentences)

Stocks:
{stock_list}
"""


def score_patterns_llm(
    stocks: list[dict],
    model: str = "claude-haiku-4-5-20251001",
    batch_size: int = 15,
) -> dict[str, dict]:
    """
    Score 5 wealth-creation patterns per stock using LLM.

    Processes stocks in batches. 60s total timeout.
    Graceful fallback: returns empty dict on any failure.

    Args:
        stocks: List of dicts with at least 'symbol', 'company_name', 'sector'.
        model: Anthropic model ID.
        batch_size: Number of stocks per LLM call.

    Returns:
        {symbol: {p1, p2, p3, p4, p5, p1_justification, ...,
                  dominant_pattern, pattern_narrative}}
    """
    if not stocks:
        return {}

    _t0 = time.monotonic()
    results: dict[str, dict] = {}

    try:
        import anthropic
        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0
        )
    except ImportError:
        logger.warning("anthropic SDK not installed — skipping LLM pattern scoring")
        return {}
    except Exception as e:
        logger.warning(f"Failed to initialize Anthropic client: {e}")
        return {}

    for i in range(0, len(stocks), batch_size):
        if time.monotonic() - _t0 > 60:
            logger.warning(
                f"LLM pattern scoring timed out after 60s, "
                f"returning {len(results)} partial results"
            )
            break

        batch = stocks[i:i + batch_size]
        stock_lines = []
        for s in batch:
            stock_lines.append(
                f"  - {s.get('symbol', '???')} ({s.get('company_name', '???')}) "
                f"[{s.get('sector', '???')}] Tier {s.get('tier', '?')}"
            )
        stock_list_str = "\n".join(stock_lines)

        prompt = _PATTERN_SCORING_PROMPT_TEMPLATE.format(
            stock_list=stock_list_str
        )

        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            track_tokens("score_patterns_llm", response)
            text = response.content[0].text if response.content else ""
            batch_results = _parse_pattern_scores(text, {s["symbol"] for s in batch})
            results.update(batch_results)
            logger.info(
                f"LLM pattern scoring batch {i // batch_size + 1}: "
                f"scored {len(batch_results)}/{len(batch)} stocks"
            )
        except Exception as e:
            logger.warning(
                f"LLM pattern scoring batch {i // batch_size + 1} failed: {e}"
            )
            continue

    return results


def _parse_pattern_scores(
    text: str, valid_symbols: set[str]
) -> dict[str, dict]:
    """Parse LLM JSON response into {symbol: pattern_scores_dict}."""
    result: dict[str, dict] = {}

    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.warning("LLM pattern scoring response did not contain a JSON array")
        return result

    json_str = text[start:end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM pattern scoring JSON: {e}")
        return result

    if not isinstance(data, list):
        return result

    max_scores = {"p1": 12, "p2": 10, "p3": 10, "p4": 10, "p5": 8}

    for item in data:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip().upper()
        if not symbol or symbol not in valid_symbols:
            continue

        # Validate and clamp pattern scores
        entry: dict = {}
        valid = True
        for key, max_s in max_scores.items():
            raw = item.get(key)
            if raw is None:
                valid = False
                break
            try:
                score = int(raw)
                entry[key] = max(0, min(max_s, score))
            except (ValueError, TypeError):
                valid = False
                break

        if not valid:
            continue

        # Copy justifications and narrative
        for key in ("p1_justification", "p2_justification", "p3_justification",
                     "p4_justification", "p5_justification"):
            entry[key] = str(item.get(key, ""))

        entry["dominant_pattern"] = str(
            item.get("dominant_pattern", "Platform Economics")
        )
        entry["pattern_narrative"] = str(
            item.get("pattern_narrative", "Pattern analysis completed.")
        )

        result[symbol] = entry

    return result


# ---------------------------------------------------------------------------
# 12. LLM Alert Narrative Enrichment (portfolio-level, legacy)
# ---------------------------------------------------------------------------

_ALERT_NARRATIVE_PROMPT_TEMPLATE = """\
You are a portfolio risk analyst providing context-aware alert narratives.

Portfolio profile:
- Sector Profile: {sector_profile}
- Risk Profile: {risk_profile}
- Avg Conviction: {avg_conviction}
- Momentum Concentration: {momentum_concentration_pct}%
- Value Coverage: {value_coverage_pct}%

For each alert below, provide an enriched description (2-3 sentences) that:
1. Names the specific stocks/sectors affected
2. Explains the real-world risk scenario
3. Suggests what could trigger the risk materializing

Return a JSON array with one object per alert. Use ONLY valid JSON — no markdown, no code fences, no extra text.

Fields per alert:
- pattern_name: str (exactly as provided)
- enriched_description: str (2-3 sentence context-aware narrative, minimum 20 characters)

Alerts:
{alert_list}
"""


def enrich_alert_narratives_llm(
    alerts: list[dict],
    profile: dict,
    model: str = "claude-haiku-4-5-20251001",
) -> list[dict]:
    """
    Enrich alert descriptions with LLM-generated context-aware narratives.

    Args:
        alerts: List of alert dicts from generate_pattern_alerts.
        profile: Portfolio profile dict from build_portfolio_profile.
        model: Anthropic model ID.

    Returns:
        Same alerts list with enriched descriptions where available.
        Falls back to returning original alerts unchanged on any failure.
    """
    if not alerts:
        return alerts

    _t0 = time.monotonic()

    try:
        import anthropic
        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0
        )

        alert_lines = []
        for a in alerts:
            symbols_str = ", ".join(a.get("affected_symbols", [])) or "N/A"
            alert_lines.append(
                f"  - {a['pattern_name']} [{a['severity']}]: "
                f"{a['description']} | Symbols: {symbols_str}"
            )
        alert_list_str = "\n".join(alert_lines)

        prompt = _ALERT_NARRATIVE_PROMPT_TEMPLATE.format(
            sector_profile=profile.get("sector_profile", "Unknown"),
            risk_profile=profile.get("risk_profile", "Unknown"),
            avg_conviction=profile.get("avg_conviction", 0),
            momentum_concentration_pct=profile.get("momentum_concentration_pct", 0),
            value_coverage_pct=profile.get("value_coverage_pct", 0),
            alert_list=alert_list_str,
        )

        if time.monotonic() - _t0 > 60:
            logger.warning("LLM alert narrative enrichment timed out before API call")
            return alerts

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        track_tokens("enrich_alert_narratives_llm", response)
        text = response.content[0].text if response.content else ""

        if time.monotonic() - _t0 > 60:
            logger.warning("LLM alert narrative enrichment timed out after API call")
            return alerts

        enriched_map = _parse_enriched_alerts(text, {a["pattern_name"] for a in alerts})

        for alert in alerts:
            enriched = enriched_map.get(alert["pattern_name"])
            if enriched and len(enriched) >= 20:
                alert["description"] = enriched

        logger.info(
            f"LLM alert narrative enrichment: enriched {len(enriched_map)}/{len(alerts)} alerts"
        )

    except ImportError:
        logger.warning("anthropic SDK not installed — skipping LLM alert enrichment")
    except Exception as e:
        logger.warning(f"LLM alert narrative enrichment error: {e}")

    return alerts


def _parse_enriched_alerts(text: str, valid_names: set[str]) -> dict[str, str]:
    """Parse LLM JSON response into {pattern_name: enriched_description}."""
    result: dict[str, str] = {}

    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.warning("LLM alert narrative response did not contain a JSON array")
        return result

    json_str = text[start:end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM alert narrative JSON: {e}")
        return result

    if not isinstance(data, list):
        return result

    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("pattern_name", "")).strip()
        if not name or name not in valid_names:
            continue
        desc = item.get("enriched_description")
        if desc and len(str(desc)) >= 20:
            result[name] = str(desc)

    return result


# ---------------------------------------------------------------------------
# 13. LLM Historical Pattern Matching (portfolio-level, legacy)
# ---------------------------------------------------------------------------

_HISTORICAL_PATTERN_PROMPT_TEMPLATE = """\
You are a financial market historian analyzing portfolio patterns.

Current portfolio profile:
- Sector Profile: {sector_profile}
- Risk Profile: {risk_profile}
- Momentum Concentration: {momentum_concentration_pct}%
- Value Coverage: {value_coverage_pct}%
- Avg Conviction: {avg_conviction}
- Conviction-Sizing Correlation: {conviction_sizing_correlation}

Current regime fit:
- Detected Regime: {detected_regime}
- Portfolio Tilt: {portfolio_tilt}
- Regime Fit Label: {regime_fit_label}
- Regime Fit Score: {regime_fit_score}/100

Identify ONE historical parallel to this portfolio pattern from the past 25 years of market history.

Return a JSON object (not array) with these fields:
- pattern_name: str (name for the historical pattern, e.g. "Late-Cycle Growth Concentration")
- historical_period: str (e.g. "Q4 2021", "2007-2008")
- similarity_score: int (0-100, how similar the current pattern is)
- description: str (2-3 sentences describing the historical parallel, minimum 20 characters)
- outcome_hint: str (1-2 sentences on what happened next historically, minimum 10 characters)

Return ONLY valid JSON — no markdown, no code fences, no extra text.
"""


def match_historical_patterns_llm(
    profile: dict,
    regime_fit: dict,
    model: str = "claude-haiku-4-5-20251001",
) -> Optional[dict]:
    """
    Find a historical parallel to the current portfolio pattern using LLM.

    Args:
        profile: Portfolio profile dict from build_portfolio_profile.
        regime_fit: Regime fit dict from analyze_regime_fit.
        model: Anthropic model ID.

    Returns:
        Dict with pattern_name, historical_period, similarity_score,
        description, outcome_hint.
        Returns None on failure or timeout.
    """
    _t0 = time.monotonic()

    try:
        import anthropic
        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0
        )

        prompt = _HISTORICAL_PATTERN_PROMPT_TEMPLATE.format(
            sector_profile=profile.get("sector_profile", "Unknown"),
            risk_profile=profile.get("risk_profile", "Unknown"),
            momentum_concentration_pct=profile.get("momentum_concentration_pct", 0),
            value_coverage_pct=profile.get("value_coverage_pct", 0),
            avg_conviction=profile.get("avg_conviction", 0),
            conviction_sizing_correlation=profile.get("conviction_sizing_correlation", 0),
            detected_regime=regime_fit.get("detected_regime", "unknown"),
            portfolio_tilt=regime_fit.get("portfolio_tilt", "unknown"),
            regime_fit_label=regime_fit.get("regime_fit_label", "unknown"),
            regime_fit_score=regime_fit.get("regime_fit_score", 0),
        )

        if time.monotonic() - _t0 > 60:
            logger.warning("LLM historical pattern matching timed out before API call")
            return None

        response = client.messages.create(
            model=model,
            max_tokens=768,
            messages=[{"role": "user", "content": prompt}],
        )
        track_tokens("find_historical_pattern_llm", response)
        text = response.content[0].text if response.content else ""

        if time.monotonic() - _t0 > 60:
            logger.warning("LLM historical pattern matching timed out after API call")
            return None

        parsed = _parse_historical_pattern(text)
        if parsed:
            logger.info(
                f"LLM historical pattern match: '{parsed.get('pattern_name', '?')}' "
                f"({parsed.get('historical_period', '?')})"
            )
        return parsed

    except ImportError:
        logger.warning("anthropic SDK not installed — skipping LLM historical pattern matching")
        return None
    except Exception as e:
        logger.warning(f"LLM historical pattern matching error: {e}")
        return None


def _parse_historical_pattern(text: str) -> Optional[dict]:
    """Parse LLM JSON response into a historical pattern dict."""
    text = text.strip()

    # Try to find JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.warning("LLM historical pattern response did not contain a JSON object")
        return None

    json_str = text[start:end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM historical pattern JSON: {e}")
        return None

    if not isinstance(data, dict):
        return None

    # Validate required fields
    required = ("pattern_name", "historical_period", "similarity_score",
                "description", "outcome_hint")
    for field in required:
        if field not in data or data[field] is None:
            logger.warning(f"LLM historical pattern missing field: {field}")
            return None

    # Validate types and ranges
    try:
        similarity = int(data["similarity_score"])
        if not (0 <= similarity <= 100):
            similarity = max(0, min(100, similarity))
    except (ValueError, TypeError):
        return None

    description = str(data["description"])
    if len(description) < 20:
        logger.warning(
            f"LLM historical pattern description too short: {len(description)} chars"
        )
        return None

    outcome_hint = str(data["outcome_hint"])
    if len(outcome_hint) < 10:
        logger.warning(
            f"LLM historical pattern outcome_hint too short: {len(outcome_hint)} chars"
        )
        return None

    return {
        "pattern_name": str(data["pattern_name"]),
        "historical_period": str(data["historical_period"]),
        "similarity_score": similarity,
        "description": description,
        "outcome_hint": outcome_hint,
    }


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class PatternAnalyzerInput(BaseModel):
    stocks_json: str = Field(
        ..., description="Rated stocks as JSON array with symbol, company_name, sector, tier, etc."
    )


class PatternAnalyzerTool(BaseTool):
    """Score stocks against 5 wealth-creation patterns to produce 150-point Enhanced Scores."""

    name: str = "pattern_alpha_scorer"
    description: str = (
        "Per-stock pattern scoring engine. Computes 100-point base scores from "
        "IBD/analyst/risk metrics, then scores 5 wealth-creation patterns "
        "(Platform Economics, Self-Cannibalization, Capital Allocation, "
        "Category Creation, Inflection Timing) via LLM for a 150-point "
        "Enhanced Score per stock. Never makes buy/sell recommendations."
    )
    args_schema: type[BaseModel] = PatternAnalyzerInput

    def _run(self, stocks_json: str) -> str:
        try:
            stocks = json.loads(stocks_json)
        except (json.JSONDecodeError, TypeError):
            return "ERROR: Could not parse stocks_json"

        # Compute base scores
        results = []
        for stock in stocks:
            base = compute_base_score(stock)
            stars, label = classify_enhanced_rating(base["base_total"])
            results.append(
                f"{stock.get('symbol', '???')}: Base={base['base_total']}/100 "
                f"(IBD={base['ibd_score']:.0f}, Analyst={base['analyst_score']:.0f}, "
                f"Risk={base['risk_score']:.0f}) → {stars} {label}"
            )

        lines = [
            f"PatternAlpha Base Scoring: {len(stocks)} stocks scored",
            "",
        ] + results

        return "\n".join(lines)
