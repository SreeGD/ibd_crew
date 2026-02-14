"""
Portfolio Manager Tool: Position Sizing & Stop Calculation
IBD Momentum Investment Framework v4.0

Pure functions for:
- Sizing positions within tier-specific limits
- Computing trailing stops (initial + tightening)
- Computing max loss per position
- Assigning tier based on stock ratings

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

from ibd_agents.schemas.analyst_output import RatedStock
from ibd_agents.schemas.portfolio_output import (
    ETF_LIMITS,
    MAX_LOSS,
    STOCK_LIMITS,
    STOP_TIGHTENING,
    TRAILING_STOPS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------

def assign_tier(rated_stock: RatedStock) -> int:
    """
    Assign tier based on analyst's existing tier assignment.
    RatedStock already has a deterministic tier from the Analyst Agent.
    """
    return rated_stock.tier


def size_position(
    tier: int,
    asset_type: str,
    conviction: int,
    is_keep: bool = False,
) -> float:
    """
    Compute target position size as % of total portfolio.

    Sizing logic:
    - Base size from conviction (1-10) mapped to tier typical range
    - Keeps get a slight size boost (they're pre-committed)
    - Never exceeds tier-specific limit
    """
    limit = ETF_LIMITS[tier] if asset_type == "etf" else STOCK_LIMITS[tier]

    # Typical ranges by tier for stocks
    typical_ranges = {
        1: (2.0, 3.5),   # T1: 2-3.5% typical
        2: (1.5, 3.0),   # T2: 1.5-3% typical
        3: (1.0, 2.5),   # T3: 1-2.5% typical
    }

    # ETF ranges are wider
    if asset_type == "etf":
        typical_ranges = {
            1: (3.0, 5.0),
            2: (2.5, 4.5),
            3: (2.0, 4.0),
        }

    lo, hi = typical_ranges[tier]

    # Map conviction (1-10) to position within range
    # conviction 10 → top of range, conviction 1 → bottom
    frac = (conviction - 1) / 9.0  # 0.0 to 1.0
    base_size = lo + frac * (hi - lo)

    # Keep boost: +0.3% for pre-committed positions
    if is_keep:
        base_size += 0.3

    # Cap at tier limit
    return round(min(base_size, limit), 2)


def compute_trailing_stop(tier: int, gain_pct: float = 0.0) -> float:
    """
    Compute trailing stop percentage based on tier and unrealized gain.

    Tightening protocol:
    - Initial: T1=22%, T2=18%, T3=12%
    - After 10% gain: tighten to range
    - After 20% gain: tighten further
    """
    initial = TRAILING_STOPS[tier]

    if gain_pct >= 20.0:
        # Use midpoint of after_20pct range
        lo, hi = STOP_TIGHTENING[tier]["after_20pct_gain"]
        return round((lo + hi) / 2, 1)
    elif gain_pct >= 10.0:
        lo, hi = STOP_TIGHTENING[tier]["after_10pct_gain"]
        return round((lo + hi) / 2, 1)
    else:
        return float(initial)


def compute_max_loss(
    tier: int,
    asset_type: str,
    position_pct: float,
) -> float:
    """
    Compute max portfolio loss for a position.

    max_loss = position_pct × (trailing_stop_pct / 100)
    Capped at framework max per tier/asset_type.
    """
    stop_pct = TRAILING_STOPS[tier] / 100.0
    actual_loss = position_pct * stop_pct
    framework_max = MAX_LOSS[asset_type][tier]
    return round(min(actual_loss, framework_max), 4)


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class PositionSizerInput(BaseModel):
    tier: int = Field(..., description="Tier 1, 2, or 3")
    asset_type: str = Field(..., description="stock or etf")
    conviction: int = Field(..., description="Conviction 1-10")
    is_keep: bool = Field(False, description="Whether this is a keep position")


class PositionSizerTool(BaseTool):
    """Size portfolio positions within framework limits."""

    name: str = "position_sizer"
    description: str = (
        "Compute position size, trailing stop, and max loss "
        "for a given tier, asset type, and conviction level"
    )
    args_schema: type[BaseModel] = PositionSizerInput

    def _run(self, tier: int, asset_type: str, conviction: int, is_keep: bool = False) -> str:
        import json
        pct = size_position(tier, asset_type, conviction, is_keep)
        stop = compute_trailing_stop(tier)
        max_loss = compute_max_loss(tier, asset_type, pct)
        return json.dumps({
            "target_pct": pct,
            "trailing_stop_pct": stop,
            "max_loss_pct": max_loss,
        })
