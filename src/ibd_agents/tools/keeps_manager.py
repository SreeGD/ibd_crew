"""
Portfolio Manager Tool: Keeps Management
IBD Momentum Investment Framework v4.0

Pure functions for:
- Placing 14 pre-committed keep positions into portfolio tiers
- Building PortfolioPosition objects for each keep

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

from ibd_agents.schemas.analyst_output import RatedStock, UnratedStock
from ibd_agents.schemas.portfolio_output import (
    ALL_KEEPS,
    FUNDAMENTAL_KEEPS,
    IBD_KEEPS,
    KEEP_METADATA,
    MAX_LOSS,
    TRAILING_STOPS,
    KeepDetail,
    KeepsPlacement,
    PortfolioPosition,
)
from ibd_agents.tools.position_sizer import compute_max_loss, size_position

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------

def _find_rated_stock(symbol: str, rated_stocks: List[RatedStock]) -> Optional[RatedStock]:
    """Find a RatedStock by symbol."""
    for rs in rated_stocks:
        if rs.symbol == symbol:
            return rs
    return None


def _determine_keep_tier(
    symbol: str,
    rated_stocks: List[RatedStock],
) -> int:
    """
    Determine which tier a keep should be placed in.

    Logic:
    1. If the keep appears in rated_stocks, use its analyst-assigned tier
    2. Otherwise, use the default tier from KEEP_METADATA
    """
    rs = _find_rated_stock(symbol, rated_stocks)
    if rs is not None:
        return rs.tier
    return KEEP_METADATA[symbol]["default_tier"]


def _get_keep_sector(
    symbol: str,
    rated_stocks: List[RatedStock],
) -> str:
    """Get sector for a keep symbol. Falls back to KEEP_METADATA sector."""
    rs = _find_rated_stock(symbol, rated_stocks)
    if rs is not None:
        return rs.sector
    return KEEP_METADATA[symbol].get("sector", "OTHER")


def _get_keep_conviction(
    symbol: str,
    rated_stocks: List[RatedStock],
    category: str,
) -> int:
    """
    Get conviction for a keep.
    IBD keeps with ratings get their analyst conviction.
    Fundamental keeps get a default based on category.
    """
    rs = _find_rated_stock(symbol, rated_stocks)
    if rs is not None:
        return rs.conviction
    # Default convictions for keeps without analyst rating
    defaults = {"fundamental": 7, "ibd": 8}
    return defaults.get(category, 6)


def place_keeps(
    rated_stocks: List[RatedStock],
) -> KeepsPlacement:
    """
    Place all 14 keeps into tiers and build KeepsPlacement.

    Each keep is assigned:
    - Tier: from analyst rating or default
    - Target %: from position_sizer based on conviction
    """
    def _build_details(symbols: list[str], category: str) -> list[KeepDetail]:
        details = []
        for sym in symbols:
            tier = _determine_keep_tier(sym, rated_stocks)
            conviction = _get_keep_conviction(sym, rated_stocks, category)
            asset_type = KEEP_METADATA[sym]["asset_type"]
            target_pct = size_position(tier, asset_type, conviction, is_keep=True)

            details.append(KeepDetail(
                symbol=sym,
                category=category,
                tier_placed=tier,
                target_pct=target_pct,
                reason=KEEP_METADATA[sym]["reason"],
            ))
        return details

    fund = _build_details(FUNDAMENTAL_KEEPS, "fundamental")
    ibd = _build_details(IBD_KEEPS, "ibd")

    total_pct = sum(k.target_pct for k in fund + ibd)

    return KeepsPlacement(
        fundamental_keeps=fund,
        ibd_keeps=ibd,
        total_keeps=len(fund) + len(ibd),
        keeps_pct=round(total_pct, 2),
    )


def build_keep_positions(
    keeps_placement: KeepsPlacement,
    rated_stocks: List[RatedStock],
) -> List[PortfolioPosition]:
    """
    Build PortfolioPosition objects for all keeps.
    """
    positions: list[PortfolioPosition] = []

    all_details = (
        keeps_placement.fundamental_keeps
        + keeps_placement.ibd_keeps
    )

    for kd in all_details:
        meta = KEEP_METADATA[kd.symbol]
        asset_type = meta["asset_type"]
        sector = _get_keep_sector(kd.symbol, rated_stocks)
        conviction = _get_keep_conviction(kd.symbol, rated_stocks, kd.category)
        stop = float(TRAILING_STOPS[kd.tier_placed])
        max_loss = compute_max_loss(kd.tier_placed, asset_type, kd.target_pct)

        rs = _find_rated_stock(kd.symbol, rated_stocks)
        company = rs.company_name if rs else f"{kd.symbol}"

        positions.append(PortfolioPosition(
            symbol=kd.symbol,
            company_name=company,
            sector=sector,
            cap_size=getattr(rs, 'cap_size', None) if rs else None,
            tier=kd.tier_placed,
            asset_type=asset_type,
            target_pct=kd.target_pct,
            trailing_stop_pct=stop,
            max_loss_pct=max_loss,
            keep_category=kd.category,
            conviction=conviction,
            reasoning=f"{kd.category.capitalize()} keep: {kd.reason}",
        ))

    return positions


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class KeepsManagerInput(BaseModel):
    analyst_json: str = Field(..., description="JSON string of AnalystOutput")


class KeepsManagerTool(BaseTool):
    """Manage placement of 14 pre-committed keep positions."""

    name: str = "keeps_manager"
    description: str = (
        "Place 14 pre-committed keep positions (5 fundamental, "
        "9 IBD) into appropriate portfolio tiers based on ratings"
    )
    args_schema: type[BaseModel] = KeepsManagerInput

    def _run(self, analyst_json: str) -> str:
        import json
        from ibd_agents.schemas.analyst_output import AnalystOutput

        analyst = AnalystOutput.model_validate_json(analyst_json)
        placement = place_keeps(analyst.rated_stocks)
        return json.dumps(placement.model_dump(), indent=2)
