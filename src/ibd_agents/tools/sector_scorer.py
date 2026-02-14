"""
Analyst Agent Tool: Sector Scorer
Calculate sector scores and rank sectors per Framework v4.0 §4.4.

Formula: (avg_comp * 0.25) + (avg_rs * 0.25) + (elite_pct * 0.30) + (multi_list_pct * 0.20)
"""

from __future__ import annotations

from typing import Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

from ibd_agents.schemas.analyst_output import SectorRank
from ibd_agents.schemas.research_output import ResearchStock, is_ibd_keep_candidate
from ibd_agents.tools.elite_screener import passes_all_elite


# ---------------------------------------------------------------------------
# Sector Score Formula (spec §4.4)
# ---------------------------------------------------------------------------

def sector_score(
    avg_composite: float,
    avg_rs: float,
    elite_pct: float,
    multi_list_pct: float,
) -> float:
    """
    Framework v4.0 sector scoring.
    elite_pct and multi_list_pct are 0-100 scale.
    """
    return (
        (avg_composite * 0.25)
        + (avg_rs * 0.25)
        + (elite_pct * 0.30)
        + (multi_list_pct * 0.20)
    )


# ---------------------------------------------------------------------------
# Sector Ranking
# ---------------------------------------------------------------------------

def rank_sectors(stocks: list[ResearchStock]) -> list[SectorRank]:
    """
    Group stocks by sector, compute sector scores, and rank.

    Only includes sectors with at least 1 ratable stock (has Comp+RS+EPS).
    Returns list sorted by sector_score descending with rank assigned.
    """
    # Group by sector
    sector_stocks: dict[str, list[ResearchStock]] = {}
    for s in stocks:
        sec = s.sector
        if not sec or sec == "UNKNOWN":
            continue
        if s.composite_rating is None or s.rs_rating is None or s.eps_rating is None:
            continue
        if sec not in sector_stocks:
            sector_stocks[sec] = []
        sector_stocks[sec].append(s)

    # Score each sector
    scored: list[dict] = []
    for sec, sec_stocks in sector_stocks.items():
        comps = [s.composite_rating for s in sec_stocks if s.composite_rating]
        rs_vals = [s.rs_rating for s in sec_stocks if s.rs_rating]
        avg_comp = sum(comps) / len(comps) if comps else 0.0
        avg_rs = sum(rs_vals) / len(rs_vals) if rs_vals else 0.0

        # Elite count (all 4 filters pass)
        elite_count = sum(
            1 for s in sec_stocks
            if passes_all_elite(s.eps_rating, s.rs_rating, s.smr_rating, s.acc_dis_rating)
        )
        elite_pct = (elite_count / len(sec_stocks) * 100) if sec_stocks else 0.0

        # Multi-list count (on 2+ IBD lists)
        multi_list_count = sum(1 for s in sec_stocks if len(s.ibd_lists) >= 2)
        multi_list_pct = (multi_list_count / len(sec_stocks) * 100) if sec_stocks else 0.0

        # IBD keep count
        keep_count = sum(
            1 for s in sec_stocks
            if is_ibd_keep_candidate(s.composite_rating, s.rs_rating)
        )

        # Top stocks by composite desc
        sorted_stocks = sorted(
            sec_stocks,
            key=lambda s: (s.composite_rating or 0, s.rs_rating or 0),
            reverse=True,
        )
        top_symbols = [s.symbol for s in sorted_stocks[:5]]

        score = sector_score(avg_comp, avg_rs, elite_pct, multi_list_pct)

        scored.append({
            "sector": sec,
            "stock_count": len(sec_stocks),
            "avg_composite": round(avg_comp, 2),
            "avg_rs": round(avg_rs, 2),
            "elite_pct": round(elite_pct, 1),
            "multi_list_pct": round(multi_list_pct, 1),
            "sector_score": round(score, 2),
            "top_stocks": top_symbols,
            "ibd_keep_count": keep_count,
        })

    # Sort by sector_score descending
    scored.sort(key=lambda x: x["sector_score"], reverse=True)

    # Assign ranks
    results: list[SectorRank] = []
    for i, s in enumerate(scored, 1):
        results.append(SectorRank(rank=i, **s))

    return results


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class SectorScorerInput(BaseModel):
    avg_composite: float = Field(..., description="Average composite rating")
    avg_rs: float = Field(..., description="Average RS rating")
    elite_pct: float = Field(..., description="Elite pass percentage (0-100)")
    multi_list_pct: float = Field(..., description="Multi-list percentage (0-100)")


class SectorScorerTool(BaseTool):
    """Calculate sector score using Framework v4.0 formula."""

    name: str = "sector_scorer"
    description: str = (
        "Calculate sector score: (avg_comp * 0.25) + (avg_rs * 0.25) + "
        "(elite_pct * 0.30) + (multi_list_pct * 0.20)"
    )
    args_schema: type[BaseModel] = SectorScorerInput

    def _run(
        self,
        avg_composite: float,
        avg_rs: float,
        elite_pct: float,
        multi_list_pct: float,
    ) -> str:
        score = sector_score(avg_composite, avg_rs, elite_pct, multi_list_pct)
        return (
            f"Sector Score: {score:.2f}\n"
            f"Components: Comp({avg_composite:.1f}*0.25={avg_composite*0.25:.2f}) + "
            f"RS({avg_rs:.1f}*0.25={avg_rs*0.25:.2f}) + "
            f"Elite({elite_pct:.1f}*0.30={elite_pct*0.30:.2f}) + "
            f"Multi({multi_list_pct:.1f}*0.20={multi_list_pct*0.20:.2f})"
        )
