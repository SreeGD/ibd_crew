"""
Returns Projector Tool: Tier Return Calculator
IBD Momentum Investment Framework v4.0

Pure functions for:
- Computing per-tier annualized returns by scenario
- Applying sector momentum multipliers
- Blending tier returns into portfolio return
- Scaling annual returns/volatility to sub-annual horizons
- Computing tier contribution to portfolio return

No LLM, no file I/O, no buy/sell recommendations.
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

from ibd_agents.schemas.returns_projection_output import (
    RISK_FREE_RATE,
    SECTOR_MOMENTUM_MULTIPLIER,
    TIER_HISTORICAL_RETURNS,
    TIME_HORIZONS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------

def compute_tier_return(
    tier: int,
    scenario: str,
    sector_momentum: str = "neutral",
) -> float:
    """
    Compute annualized return % for a given tier and scenario.

    Args:
        tier: 1, 2, or 3.
        scenario: "bull", "base", or "bear".
        sector_momentum: one of "leading", "improving", "lagging",
            "declining", or "neutral" (no adjustment).

    Returns:
        Annualized return percentage (e.g. 18.0 for 18%).
    """
    mean = TIER_HISTORICAL_RETURNS[tier][scenario]["mean"]
    multiplier = SECTOR_MOMENTUM_MULTIPLIER.get(sector_momentum, 1.0)
    return mean * multiplier


def compute_blended_return(
    tier_pcts: dict[int, float],
    tier_returns: dict[int, float],
    cash_pct: float,
    risk_free: float = RISK_FREE_RATE,
) -> float:
    """
    Compute blended portfolio return from tier allocations and cash.

    Args:
        tier_pcts: {1: 39.0, 2: 37.0, 3: 22.0} — allocation percentages.
        tier_returns: {1: 18.0, 2: 14.0, 3: 10.0} — annualized return %.
        cash_pct: percentage held in cash (e.g. 2.0 for 2%).
        risk_free: annualized risk-free rate % (default from schema).

    Returns:
        Blended portfolio return percentage.
    """
    total = sum(
        tier_pcts[t] / 100.0 * tier_returns[t]
        for t in (1, 2, 3)
        if t in tier_pcts and t in tier_returns
    )
    total += cash_pct / 100.0 * risk_free
    return total


def scale_to_horizon(
    annual_return: float,
    annual_std: float,
    horizon_key: str,
) -> tuple[float, float]:
    """
    Scale annual return and volatility to a sub-annual horizon.

    Args:
        annual_return: annualized return %.
        annual_std: annualized standard deviation %.
        horizon_key: one of "3_month", "6_month", "12_month".

    Returns:
        (scaled_return, scaled_std) as percentages.
    """
    horizon = TIME_HORIZONS[horizon_key]
    fraction = horizon["fraction"]
    vol_scale = horizon["volatility_scale"]
    scaled_return = annual_return * fraction
    scaled_std = annual_std * vol_scale
    return (scaled_return, scaled_std)


def compute_tier_contribution(
    tier_pct: float,
    tier_return: float,
) -> float:
    """
    Compute a single tier's contribution to portfolio return.

    Args:
        tier_pct: allocation percentage (e.g. 39.0 for 39%).
        tier_return: tier annualized return %.

    Returns:
        Contribution percentage (e.g. 7.02 for T1 at 39% allocation with 18% return).
    """
    return tier_pct / 100.0 * tier_return


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class TierReturnCalculatorInput(BaseModel):
    tier: int = Field(..., description="Tier 1, 2, or 3")
    scenario: str = Field(..., description="bull, base, or bear")
    sector_momentum: str = Field(
        "neutral",
        description="Sector momentum: leading, improving, lagging, declining, or neutral",
    )


class TierReturnCalculatorTool(BaseTool):
    """Compute tier-level return projections with sector momentum adjustments."""

    name: str = "tier_return_calculator"
    description: str = (
        "Compute the expected annualized return for a given tier (1/2/3) "
        "and scenario (bull/base/bear), optionally adjusted for sector "
        "momentum (leading/improving/lagging/declining/neutral)."
    )
    args_schema: type[BaseModel] = TierReturnCalculatorInput

    def _run(
        self,
        tier: int,
        scenario: str,
        sector_momentum: str = "neutral",
    ) -> str:
        import json

        tier_ret = compute_tier_return(tier, scenario, sector_momentum)
        std = TIER_HISTORICAL_RETURNS[tier][scenario]["std"]
        multiplier = SECTOR_MOMENTUM_MULTIPLIER.get(sector_momentum, 1.0)

        return json.dumps({
            "tier": tier,
            "scenario": scenario,
            "sector_momentum": sector_momentum,
            "annualized_return_pct": round(tier_ret, 2),
            "annualized_std_pct": round(std * multiplier, 2),
            "multiplier_applied": round(multiplier, 2),
        })
