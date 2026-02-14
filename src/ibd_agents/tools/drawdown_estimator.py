"""
Returns Projector Tool: Drawdown Estimator
IBD Momentum Investment Framework v4.0

Pure functions for:
- Computing maximum drawdown with trailing stops
- Computing maximum drawdown without stops (bear proxy)
- Estimating expected stop trigger counts by scenario

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
    STOP_LOSS_BY_TIER,
    TIER_HISTORICAL_RETURNS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------

def compute_drawdown_with_stops(
    tier_pcts: dict[int, float],
    stop_losses: Optional[dict[int, float]] = None,
) -> float:
    """
    Compute maximum portfolio drawdown assuming all trailing stops trigger.

    Args:
        tier_pcts: {1: 39.0, 2: 37.0, 3: 22.0} — allocation percentages.
        stop_losses: {1: 0.22, 2: 0.18, 3: 0.12} — stop loss fractions.
            Defaults to STOP_LOSS_BY_TIER from schema.

    Returns:
        Maximum portfolio loss percentage (e.g. 17.88 for ~17.88%).
    """
    if stop_losses is None:
        stop_losses = STOP_LOSS_BY_TIER

    max_loss = sum(
        tier_pcts[t] / 100.0 * stop_losses[t] * 100.0
        for t in (1, 2, 3)
        if t in tier_pcts and t in stop_losses
    )
    return round(max_loss, 2)


def compute_drawdown_without_stops(
    tier_pcts: dict[int, float],
    bear_scenario: bool = True,
) -> float:
    """
    Compute maximum portfolio drawdown without trailing stops.

    Uses bear-case standard deviation as a proxy for max drawdown,
    scaled by 1.5x (approximately 2 std + buffer for tail risk).

    Args:
        tier_pcts: {1: 39.0, 2: 37.0, 3: 22.0} — allocation percentages.
        bear_scenario: if True, use bear-case std; otherwise use base-case.

    Returns:
        Maximum portfolio loss percentage (should exceed drawdown_with_stops).
    """
    scenario = "bear" if bear_scenario else "base"

    max_loss = sum(
        tier_pcts[t] / 100.0 * TIER_HISTORICAL_RETURNS[t][scenario]["std"] * 1.5
        for t in (1, 2, 3)
        if t in tier_pcts
    )
    return round(max_loss, 2)


def estimate_stop_triggers(
    num_positions: int,
    scenario: str = "base",
    horizon_months: int = 12,
) -> tuple[int, float]:
    """
    Estimate the number of positions expected to hit trailing stops.

    Base probabilities by scenario (for 12-month horizon):
        bull=0.15, base=0.25, bear=0.45

    Args:
        num_positions: total number of positions in portfolio.
        scenario: "bull", "base", or "bear".
        horizon_months: holding period in months (currently unused,
            reserved for future sub-annual scaling).

    Returns:
        (expected_stops, probability) where expected_stops is the
        rounded count and probability is the per-position trigger rate.
    """
    probabilities = {
        "bull": 0.15,
        "base": 0.25,
        "bear": 0.45,
    }

    prob = probabilities.get(scenario, probabilities["base"])
    expected_stops = round(num_positions * prob)
    return (expected_stops, prob)


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class DrawdownEstimatorInput(BaseModel):
    tier_1_pct: float = Field(..., description="% allocated to Tier 1")
    tier_2_pct: float = Field(..., description="% allocated to Tier 2")
    tier_3_pct: float = Field(..., description="% allocated to Tier 3")
    num_positions: int = Field(30, description="Total number of portfolio positions")
    scenario: str = Field("base", description="Scenario: bull, base, or bear")


class DrawdownEstimatorTool(BaseTool):
    """Estimate portfolio drawdown with and without trailing stops."""

    name: str = "drawdown_estimator"
    description: str = (
        "Estimate maximum portfolio drawdown with trailing stops active, "
        "without stops, and the expected number of stop triggers for a "
        "given tier allocation and scenario."
    )
    args_schema: type[BaseModel] = DrawdownEstimatorInput

    def _run(
        self,
        tier_1_pct: float,
        tier_2_pct: float,
        tier_3_pct: float,
        num_positions: int = 30,
        scenario: str = "base",
    ) -> str:
        import json

        tier_pcts = {1: tier_1_pct, 2: tier_2_pct, 3: tier_3_pct}
        dd_with = compute_drawdown_with_stops(tier_pcts)
        dd_without = compute_drawdown_without_stops(tier_pcts)
        expected_stops, prob = estimate_stop_triggers(num_positions, scenario)

        return json.dumps({
            "max_drawdown_with_stops_pct": dd_with,
            "max_drawdown_without_stops_pct": dd_without,
            "expected_stops_triggered": expected_stops,
            "stop_trigger_probability": prob,
            "scenario": scenario,
        })
