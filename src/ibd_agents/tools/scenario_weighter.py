"""
Returns Projector Tool: Scenario Weighter
IBD Momentum Investment Framework v4.0

Pure functions for:
- Retrieving scenario probability weights by market regime
- Computing probability-weighted expected values across scenarios

No LLM, no file I/O, no buy/sell recommendations.
"""

from __future__ import annotations

import logging

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

from ibd_agents.schemas.returns_projection_output import (
    SCENARIO_WEIGHTS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------

def get_scenario_weights(regime: str) -> dict[str, float]:
    """
    Retrieve scenario probability weights for a market regime.

    Args:
        regime: "bull", "neutral", or "bear".
            Falls back to "neutral" if regime not found.

    Returns:
        {"bull": ..., "base": ..., "bear": ...} probability weights summing to 1.0.
    """
    return SCENARIO_WEIGHTS.get(regime, SCENARIO_WEIGHTS["neutral"])


def compute_expected(
    scenario_values: dict[str, float],
    weights: dict[str, float],
) -> float:
    """
    Compute probability-weighted expected value from scenario values.

    Args:
        scenario_values: {"bull": ..., "base": ..., "bear": ...} values.
        weights: {"bull": ..., "base": ..., "bear": ...} probabilities.

    Returns:
        Probability-weighted expected value.
    """
    return sum(weights[s] * scenario_values[s] for s in weights)


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class ScenarioWeighterInput(BaseModel):
    regime: str = Field(
        ..., description="Market regime: bull, neutral, or bear"
    )


class ScenarioWeighterTool(BaseTool):
    """Retrieve scenario probability weights for a given market regime."""

    name: str = "scenario_weighter"
    description: str = (
        "Retrieve the bull/base/bear scenario probability weights for "
        "a given market regime (bull/neutral/bear). Used to compute "
        "probability-weighted expected returns."
    )
    args_schema: type[BaseModel] = ScenarioWeighterInput

    def _run(self, regime: str) -> str:
        import json

        weights = get_scenario_weights(regime)
        return json.dumps({
            "regime": regime,
            "weights": weights,
        })
