"""
Analyst Agent Tool: Elite Screener
Apply Framework v4.0 §2.3 Elite Screening, tier assignment,
conviction scoring, and per-stock analysis generation.
"""

from __future__ import annotations

from typing import Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

from ibd_agents.schemas.research_output import (
    ResearchStock,
    compute_preliminary_tier,
    is_ibd_keep_candidate,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIER_LABELS: dict[int, str] = {
    1: "Momentum",
    2: "Quality Growth",
    3: "Defensive",
}

ELITE_GOOD_LETTERS = {"A", "B", "B-"}


# ---------------------------------------------------------------------------
# Elite Filter Functions (spec §4.1)
# ---------------------------------------------------------------------------

def passes_elite_eps(eps_rating: Optional[int]) -> bool:
    """EPS >= 85."""
    return eps_rating is not None and eps_rating >= 85


def passes_elite_rs(rs_rating: Optional[int]) -> bool:
    """RS >= 75."""
    return rs_rating is not None and rs_rating >= 75


def passes_elite_smr(smr_rating: Optional[str]) -> Optional[bool]:
    """SMR in A/B/B-. Returns None if rating missing."""
    if smr_rating is None:
        return None
    return smr_rating in ELITE_GOOD_LETTERS


def passes_elite_acc_dis(acc_dis_rating: Optional[str]) -> Optional[bool]:
    """Acc/Dis in A/B/B-. Returns None if rating missing."""
    if acc_dis_rating is None:
        return None
    return acc_dis_rating in ELITE_GOOD_LETTERS


def passes_all_elite(
    eps_rating: Optional[int],
    rs_rating: Optional[int],
    smr_rating: Optional[str],
    acc_dis_rating: Optional[str],
) -> bool:
    """All 4 elite filters must pass. Missing letter rating = fail."""
    return (
        passes_elite_eps(eps_rating)
        and passes_elite_rs(rs_rating)
        and passes_elite_smr(smr_rating) is True
        and passes_elite_acc_dis(acc_dis_rating) is True
    )


# ---------------------------------------------------------------------------
# Conviction Scoring (deterministic, 1-10)
# ---------------------------------------------------------------------------

def calculate_conviction(
    tier: Optional[int],
    elite: bool,
    is_multi_source_validated: bool,
    ibd_lists: list[str],
    schwab_themes: list[str],
    morningstar_rating: Optional[str] = None,
    economic_moat: Optional[str] = None,
) -> int:
    """
    Deterministic conviction score.

    Base from tier: T1=7, T2=5, T3=3, None=1
    +1 if passes all elite
    +1 if multi-source validated
    +1 if on 2+ IBD lists
    +1 if has Schwab themes
    +1 if Morningstar 5-star
    +1 if Wide economic moat
    Cap at 10.
    """
    if tier == 1:
        score = 7
    elif tier == 2:
        score = 5
    elif tier == 3:
        score = 3
    else:
        score = 1

    if elite:
        score += 1
    if is_multi_source_validated:
        score += 1
    if len(ibd_lists) >= 2:
        score += 1
    if len(schwab_themes) >= 1:
        score += 1
    if morningstar_rating == "5-star":
        score += 1
    if economic_moat == "Wide":
        score += 1

    return min(score, 10)


# ---------------------------------------------------------------------------
# Strengths / Weaknesses / Catalyst (template-based)
# ---------------------------------------------------------------------------

def generate_strengths(
    stock: ResearchStock,
    p_eps: bool,
    p_rs: bool,
    p_smr: Optional[bool],
    p_acc_dis: Optional[bool],
) -> list[str]:
    """Generate 1-5 strengths based on ratings and sources."""
    strengths = []

    if stock.composite_rating is not None and stock.composite_rating >= 95:
        strengths.append(f"Elite Composite Rating ({stock.composite_rating})")
    elif stock.composite_rating is not None and stock.composite_rating >= 85:
        strengths.append(f"Strong Composite Rating ({stock.composite_rating})")

    if p_eps and stock.eps_rating is not None:
        strengths.append(f"Top earnings growth (EPS {stock.eps_rating})")

    if p_rs and stock.rs_rating is not None and stock.rs_rating >= 90:
        strengths.append(f"Strong relative strength (RS {stock.rs_rating})")

    if p_smr is True:
        strengths.append(f"Solid fundamentals (SMR {stock.smr_rating})")

    if p_acc_dis is True:
        strengths.append(f"Institutional accumulation (Acc/Dis {stock.acc_dis_rating})")

    if stock.is_multi_source_validated:
        strengths.append(f"Multi-source validated (score {stock.validation_score})")

    if len(stock.ibd_lists) >= 2:
        strengths.append(f"On {len(stock.ibd_lists)} IBD lists")

    if stock.schwab_themes:
        strengths.append(f"Thematic exposure: {', '.join(stock.schwab_themes[:2])}")

    if stock.economic_moat == "Wide":
        strengths.append("Wide economic moat (Morningstar)")
    if stock.price_to_fair_value is not None and stock.price_to_fair_value < 0.80:
        strengths.append(f"Trading below fair value (P/FV {stock.price_to_fair_value:.2f})")
    if stock.morningstar_rating == "5-star":
        strengths.append("Morningstar 5-star rating")

    # Cap at 5
    return strengths[:5] if strengths else ["Meets tier threshold requirements"]


def generate_weaknesses(
    stock: ResearchStock,
    p_eps: bool,
    p_rs: bool,
    p_smr: Optional[bool],
    p_acc_dis: Optional[bool],
) -> list[str]:
    """Generate 1-5 weaknesses based on filter failures and gaps."""
    weaknesses = []

    if not p_eps and stock.eps_rating is not None:
        weaknesses.append(f"EPS below elite threshold ({stock.eps_rating} < 85)")

    if not p_rs and stock.rs_rating is not None:
        weaknesses.append(f"RS below elite threshold ({stock.rs_rating} < 75)")

    if p_smr is False:
        weaknesses.append(f"Weak SMR rating ({stock.smr_rating})")

    if p_acc_dis is False:
        weaknesses.append(f"Weak accumulation/distribution ({stock.acc_dis_rating})")

    if p_smr is None:
        weaknesses.append("Missing SMR rating — cannot confirm fundamentals")

    if p_acc_dis is None:
        weaknesses.append("Missing Acc/Dis rating — cannot confirm institutional support")

    if not stock.is_multi_source_validated:
        weaknesses.append("Limited cross-source validation")

    if len(stock.ibd_lists) < 2:
        weaknesses.append("Appears on fewer than 2 IBD lists")

    if stock.economic_moat in ("Narrow", "None"):
        weaknesses.append(f"Limited economic moat ({stock.economic_moat})")
    if stock.price_to_fair_value is not None and stock.price_to_fair_value > 1.10:
        weaknesses.append(f"Trading above fair value (P/FV {stock.price_to_fair_value:.2f})")
    if stock.morningstar_uncertainty in ("High", "Very High"):
        weaknesses.append(f"High Morningstar uncertainty ({stock.morningstar_uncertainty})")

    # Cap at 5
    return weaknesses[:5] if weaknesses else ["No significant weaknesses identified"]


def generate_catalyst(stock: ResearchStock) -> str:
    """Generate catalyst based on stock characteristics."""
    parts = []

    if stock.schwab_themes:
        parts.append(f"thematic tailwinds ({', '.join(stock.schwab_themes[:2])})")

    if stock.eps_rating is not None and stock.eps_rating >= 90:
        parts.append("accelerating earnings growth")

    if stock.rs_rating is not None and stock.rs_rating >= 90:
        parts.append("strong price momentum")

    if stock.fool_status:
        parts.append(f"Motley Fool {stock.fool_status} endorsement")

    if len(stock.ibd_lists) >= 2:
        parts.append("multi-list IBD recognition")

    if parts:
        return f"Key catalysts: {'; '.join(parts)}"
    return f"Sector momentum in {stock.sector} with solid IBD ratings"


# ---------------------------------------------------------------------------
# ETF Screening Gates (spec §4.1 adapted for ETFs)
# ---------------------------------------------------------------------------

def passes_etf_rs_screen(rs_rating: Optional[int]) -> Optional[bool]:
    """RS >= 70 for ETFs (looser than stock elite threshold of 75)."""
    if rs_rating is None:
        return None
    return rs_rating >= 70


def passes_etf_acc_dis_screen(acc_dis_rating: Optional[str]) -> Optional[bool]:
    """Acc/Dis in A/B/B- for ETFs."""
    if acc_dis_rating is None:
        return None
    return acc_dis_rating in ELITE_GOOD_LETTERS


def passes_etf_screen(
    rs_rating: Optional[int],
    acc_dis_rating: Optional[str],
) -> bool:
    """Both ETF screening gates must pass. Missing = fail."""
    return (
        passes_etf_rs_screen(rs_rating) is True
        and passes_etf_acc_dis_screen(acc_dis_rating) is True
    )


# ---------------------------------------------------------------------------
# ETF Conviction Scoring (deterministic, 1-10)
# ---------------------------------------------------------------------------

def calculate_etf_conviction(
    tier: Optional[int],
    rs_rating: Optional[int],
    acc_dis_rating: Optional[str],
    ytd_change: Optional[float],
    schwab_themes: list[str],
    etf_score: Optional[float],
) -> int:
    """
    Deterministic ETF conviction score.

    Base from tier: T1=7, T2=5, T3=3
    +1 if RS >= 85
    +1 if Acc/Dis A
    +1 if YTD > 10%
    +1 if has Schwab themes
    Cap at 10.
    """
    if tier == 1:
        score = 7
    elif tier == 2:
        score = 5
    elif tier == 3:
        score = 3
    else:
        score = 3

    if rs_rating is not None and rs_rating >= 85:
        score += 1
    if acc_dis_rating == "A":
        score += 1
    if ytd_change is not None and ytd_change > 10.0:
        score += 1
    if len(schwab_themes) >= 1:
        score += 1

    return min(score, 10)


# ---------------------------------------------------------------------------
# Sector-level stock ranking (spec §4.5)
# ---------------------------------------------------------------------------

def rank_stocks_in_sector(stocks: list[ResearchStock]) -> list[ResearchStock]:
    """
    Rank stocks within a sector.
    Primary: Composite desc, Secondary: RS desc, Tertiary: EPS desc.
    """
    return sorted(
        stocks,
        key=lambda s: (
            s.composite_rating or 0,
            s.rs_rating or 0,
            s.eps_rating or 0,
        ),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class EliteScreenerInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    composite_rating: int = Field(..., description="IBD Composite Rating")
    rs_rating: int = Field(..., description="IBD RS Rating")
    eps_rating: int = Field(..., description="IBD EPS Rating")
    smr_rating: Optional[str] = Field(None, description="SMR Rating")
    acc_dis_rating: Optional[str] = Field(None, description="Acc/Dis Rating")


class EliteScreenerTool(BaseTool):
    """Apply Framework v4.0 elite screening and tier assignment."""

    name: str = "elite_screener"
    description: str = (
        "Apply the 4-gate elite screening (EPS>=85, RS>=75, SMR A/B/B-, "
        "Acc/Dis A/B/B-) and assign tier (T1/T2/T3) to a stock."
    )
    args_schema: type[BaseModel] = EliteScreenerInput

    def _run(
        self,
        symbol: str,
        composite_rating: int,
        rs_rating: int,
        eps_rating: int,
        smr_rating: Optional[str] = None,
        acc_dis_rating: Optional[str] = None,
    ) -> str:
        tier = compute_preliminary_tier(composite_rating, rs_rating, eps_rating)
        elite = passes_all_elite(eps_rating, rs_rating, smr_rating, acc_dis_rating)
        keep = is_ibd_keep_candidate(composite_rating, rs_rating)
        tier_label = TIER_LABELS.get(tier, "Below Threshold") if tier else "Below Threshold"

        return (
            f"{symbol}: Tier={tier_label}, Elite={'PASS' if elite else 'FAIL'}, "
            f"Keep={'YES' if keep else 'NO'}\n"
            f"Filters: EPS({'P' if passes_elite_eps(eps_rating) else 'F'}) "
            f"RS({'P' if passes_elite_rs(rs_rating) else 'F'}) "
            f"SMR({passes_elite_smr(smr_rating)}) "
            f"Acc/Dis({passes_elite_acc_dis(acc_dis_rating)})"
        )
