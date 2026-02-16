"""
Value Investor Tool: Value Screener
Screen and score stocks through a value investing lens.

Computes composite Value Score (0-100), classifies into value
categories, detects value traps, and assesses momentum-value alignment.
"""

from __future__ import annotations

from typing import Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

from ibd_agents.schemas.value_investor_output import (
    MOAT_POINTS,
    PE_SCORE_MAP,
    STAR_POINTS,
    UNCERTAINTY_POINTS,
    VALUE_SCORE_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Morningstar Score (0-100)
# ---------------------------------------------------------------------------

def compute_morningstar_score(
    morningstar_rating: Optional[str],
    economic_moat: Optional[str],
    price_to_fair_value: Optional[float],
    morningstar_uncertainty: Optional[str],
) -> dict:
    """
    Compute Morningstar composite score (0-100).

    Components: star_pts(0-40) + moat_pts(0-30) + discount_pts(0-20) + certainty_pts(0-10).
    """
    star_pts = STAR_POINTS.get(morningstar_rating or "", 0)
    moat_pts = MOAT_POINTS.get(economic_moat or "", 0)
    pfv = price_to_fair_value
    discount_pts = max(0, min(20, round((1.0 - pfv) * 40))) if pfv is not None else 0
    cert_pts = UNCERTAINTY_POINTS.get(morningstar_uncertainty or "", 0)
    total = star_pts + moat_pts + discount_pts + cert_pts

    return {
        "morningstar_score": min(100, total),
        "star_points": star_pts,
        "moat_points": moat_pts,
        "discount_points": discount_pts,
        "certainty_points": cert_pts,
    }


# ---------------------------------------------------------------------------
# Component Score Functions (each 0-100)
# ---------------------------------------------------------------------------

def compute_pe_value_score(
    estimated_pe: Optional[float],
    pe_category: Optional[str],
    sector: str,
) -> float:
    """
    Score P/E attractiveness (0-100).
    Deep Value=100, Value=75, Reasonable=50, Growth Premium=25, Speculative=0.
    """
    return PE_SCORE_MAP.get(pe_category or "", 0.0)


def compute_peg_value_score(
    peg_ratio: Optional[float],
    peg_category: Optional[str],
) -> float:
    """
    Score PEG attractiveness (0-100).
    PEG < 0.5 -> 100, 0.5-1.0 -> 80, 1.0-1.5 -> 60, 1.5-2.0 -> 40, >2.0 -> 10.
    """
    if peg_ratio is None:
        return 0.0
    if peg_ratio < 0.5:
        return 100.0
    elif peg_ratio < 1.0:
        return 80.0
    elif peg_ratio < 1.5:
        return 60.0
    elif peg_ratio <= 2.0:
        return 40.0
    else:
        return 10.0


def compute_moat_quality_score(
    economic_moat: Optional[str],
    morningstar_rating: Optional[str],
    smr_rating: Optional[str],
) -> float:
    """
    Score moat/quality (0-100).
    Wide=80, Narrow=40, None/missing=0.
    +10 if 5-star, +5 if 4-star, +10 if SMR A/B.
    """
    base = {"Wide": 80.0, "Narrow": 40.0, "None": 0.0}.get(economic_moat or "", 0.0)
    if morningstar_rating == "5-star":
        base += 10.0
    elif morningstar_rating == "4-star":
        base += 5.0
    if smr_rating in ("A", "B"):
        base += 10.0
    return min(100.0, base)


def compute_discount_score(
    price_to_fair_value: Optional[float],
) -> float:
    """
    Score discount to fair value (0-100).
    Linear: (1.0 - P/FV) * 200, capped 0-100.
    P/FV >= 1.0 -> 0 (no discount).
    """
    if price_to_fair_value is None or price_to_fair_value >= 1.0:
        return 0.0
    pfv = max(0.3, price_to_fair_value)
    return round(min(100.0, (1.0 - pfv) * 200.0), 1)


# ---------------------------------------------------------------------------
# Composite Value Score (0-100)
# ---------------------------------------------------------------------------

def compute_value_score(
    morningstar_score: float,
    pe_value_score: float,
    peg_value_score: float,
    moat_quality_score: float,
    discount_score: float,
) -> float:
    """
    Weighted composite value score (0-100).

    Weights: M* 30%, P/E 20%, PEG 20%, Moat 15%, Discount 15%.
    If M* data is missing (score=0 and discount=0), redistribute
    45% weight to P/E and PEG equally.
    """
    w = dict(VALUE_SCORE_WEIGHTS)

    if morningstar_score == 0 and discount_score == 0:
        w["pe_value"] = 0.425
        w["peg_value"] = 0.425
        w["moat_quality"] = 0.15
        w["morningstar"] = 0.0
        w["discount"] = 0.0

    score = (
        morningstar_score * w["morningstar"]
        + pe_value_score * w["pe_value"]
        + peg_value_score * w["peg_value"]
        + moat_quality_score * w["moat_quality"]
        + discount_score * w["discount"]
    )
    return round(min(100.0, max(0.0, score)), 1)


# ---------------------------------------------------------------------------
# Value Category Classification
# ---------------------------------------------------------------------------

def classify_value_category(
    pe_category: Optional[str],
    peg_ratio: Optional[float],
    peg_category: Optional[str],
    rs_rating: int,
    eps_rating: int,
    economic_moat: Optional[str],
    morningstar_rating: Optional[str],
    price_to_fair_value: Optional[float],
    llm_dividend_yield: Optional[float],
) -> tuple:
    """
    Classify stock into value category. Returns (category, reasoning).

    Priority order (first match wins):
    1. Quality Value: Wide moat + (4/5-star M*) + P/FV <= 1.0
    2. Deep Value: P/E "Deep Value" OR P/FV < 0.80
    3. GARP: PEG <= 2.0 + P/E Reasonable/Value + RS >= 80 + EPS >= 75
    4. Dividend Value: Div yield >= 2.0% + P/E Deep Value/Value/Reasonable
    5. Not Value: default
    """
    # Quality Value
    if (economic_moat == "Wide"
            and morningstar_rating in ("5-star", "4-star")
            and price_to_fair_value is not None
            and price_to_fair_value <= 1.0):
        return ("Quality Value",
                f"Wide moat, {morningstar_rating}, P/FV {price_to_fair_value:.2f} "
                f"— premier quality at fair-or-better price")

    # Deep Value
    if pe_category == "Deep Value":
        pfv_str = f", P/FV {price_to_fair_value:.2f}" if price_to_fair_value is not None else ""
        return ("Deep Value",
                f"Deep Value P/E category{pfv_str} "
                f"— statistically cheap vs sector peers")
    if price_to_fair_value is not None and price_to_fair_value < 0.80:
        return ("Deep Value",
                f"P/FV {price_to_fair_value:.2f} (>20% below fair value) "
                f"— significant discount to intrinsic value")

    # GARP
    if (peg_ratio is not None and peg_ratio <= 2.0
            and pe_category in ("Reasonable", "Value")
            and rs_rating >= 80 and eps_rating >= 75):
        return ("GARP",
                f"PEG {peg_ratio:.2f}, {pe_category} PE, RS {rs_rating}, EPS {eps_rating} "
                f"— growth at reasonable price with momentum confirmation")

    # Dividend Value
    if (llm_dividend_yield is not None and llm_dividend_yield >= 2.0
            and pe_category in ("Deep Value", "Value", "Reasonable")):
        return ("Dividend Value",
                f"Yield {llm_dividend_yield:.1f}%, {pe_category} PE "
                f"— income-generating value opportunity")

    # Not Value
    return ("Not Value",
            f"PE category: {pe_category or 'N/A'}, PEG: {peg_ratio or 'N/A'}, "
            f"moat: {economic_moat or 'N/A'} — does not meet value criteria")


# ---------------------------------------------------------------------------
# Value Trap Detection
# ---------------------------------------------------------------------------

def detect_value_trap(
    pe_category: Optional[str],
    rs_rating: int,
    eps_rating: int,
    smr_rating: Optional[str],
    acc_dis_rating: Optional[str],
    price_to_fair_value: Optional[float],
    value_category: str,
) -> tuple:
    """
    Detect value trap risk. Returns (risk_level, signals).

    Only applies to stocks in value categories (not "Not Value").
    """
    if value_category == "Not Value":
        return ("None", [])

    signals = []

    if rs_rating < 50:
        signals.append(f"Weak RS {rs_rating} — price lagging market significantly")

    if eps_rating < 60:
        signals.append(f"Low EPS rating {eps_rating} — deteriorating earnings trajectory")

    if smr_rating in ("C", "D", "E"):
        signals.append(f"Poor SMR '{smr_rating}' — weak sales/margins/ROE")

    if acc_dis_rating in ("D", "E"):
        signals.append(f"Acc/Dis '{acc_dis_rating}' — institutional distribution")

    if len(signals) >= 3:
        risk = "High"
    elif len(signals) >= 2:
        risk = "Moderate"
    elif len(signals) >= 1:
        risk = "Low"
    else:
        risk = "None"

    return (risk, signals)


# ---------------------------------------------------------------------------
# Momentum-Value Alignment
# ---------------------------------------------------------------------------

def assess_momentum_value_alignment(
    rs_rating: int,
    value_score: float,
    value_category: str,
) -> tuple:
    """
    Assess alignment between momentum (RS) and value (value_score).
    Returns (alignment_label, detail).
    """
    high_rs = rs_rating >= 80
    high_value = value_score >= 50

    if high_rs and high_value:
        return ("Aligned",
                f"RS {rs_rating} + Value Score {value_score:.0f} — "
                f"momentum confirms value, ideal convergence")
    elif not high_rs and not high_value:
        return ("Aligned",
                f"RS {rs_rating} + Value Score {value_score:.0f} — "
                f"neither momentum nor value standout")
    elif high_rs and not high_value:
        return ("Mild Mismatch",
                f"RS {rs_rating} (strong momentum) but Value Score {value_score:.0f} — "
                f"momentum-driven, not a value opportunity")
    else:
        return ("Strong Mismatch",
                f"Value Score {value_score:.0f} (attractive) but RS {rs_rating} — "
                f"cheap for a reason, or early-stage value recovery")


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class ValueScreenerInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    sector: str = Field(..., description="IBD sector")
    rs_rating: int = Field(..., description="RS Rating")
    eps_rating: int = Field(..., description="EPS Rating")
    pe_category: Optional[str] = Field(None, description="PE category")
    peg_ratio: Optional[float] = Field(None, description="PEG ratio")
    morningstar_rating: Optional[str] = Field(None, description="M* rating")
    economic_moat: Optional[str] = Field(None, description="Economic moat")
    price_to_fair_value: Optional[float] = Field(None, description="P/FV ratio")
    morningstar_uncertainty: Optional[str] = Field(None, description="Uncertainty")
    smr_rating: Optional[str] = Field(None, description="SMR rating")
    llm_dividend_yield: Optional[float] = Field(None, description="Dividend yield %")


class ValueScreenerTool(BaseTool):
    """Screen a stock for value investing characteristics."""

    name: str = "value_screener"
    description: str = (
        "Screen a stock for value characteristics: compute Value Score (0-100), "
        "classify into value category (GARP/Deep Value/Quality Value/Dividend Value), "
        "detect value trap risk, and assess momentum-value alignment."
    )
    args_schema: type[BaseModel] = ValueScreenerInput

    def _run(
        self,
        symbol: str,
        sector: str,
        rs_rating: int,
        eps_rating: int,
        pe_category: Optional[str] = None,
        peg_ratio: Optional[float] = None,
        morningstar_rating: Optional[str] = None,
        economic_moat: Optional[str] = None,
        price_to_fair_value: Optional[float] = None,
        morningstar_uncertainty: Optional[str] = None,
        smr_rating: Optional[str] = None,
        llm_dividend_yield: Optional[float] = None,
    ) -> str:
        ms = compute_morningstar_score(
            morningstar_rating, economic_moat,
            price_to_fair_value, morningstar_uncertainty,
        )
        pe_vs = compute_pe_value_score(None, pe_category, sector)
        peg_vs = compute_peg_value_score(peg_ratio, None)
        moat_qs = compute_moat_quality_score(economic_moat, morningstar_rating, smr_rating)
        disc_s = compute_discount_score(price_to_fair_value)
        v_score = compute_value_score(
            ms["morningstar_score"], pe_vs, peg_vs, moat_qs, disc_s,
        )
        cat, reason = classify_value_category(
            pe_category, peg_ratio, None, rs_rating, eps_rating,
            economic_moat, morningstar_rating, price_to_fair_value,
            llm_dividend_yield,
        )
        trap_risk, trap_signals = detect_value_trap(
            pe_category, rs_rating, eps_rating, smr_rating, None,
            price_to_fair_value, cat,
        )
        align, align_detail = assess_momentum_value_alignment(rs_rating, v_score, cat)

        return (
            f"{symbol}: Value Score={v_score:.1f}, Category={cat}, "
            f"Trap={trap_risk}, Alignment={align}\n"
            f"Components: M*={ms['morningstar_score']}, PE={pe_vs:.0f}, "
            f"PEG={peg_vs:.0f}, Moat={moat_qs:.0f}, Discount={disc_s:.0f}\n"
            f"Reasoning: {reason}"
        )
