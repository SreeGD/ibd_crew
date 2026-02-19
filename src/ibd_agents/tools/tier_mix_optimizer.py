"""
Target Return Constructor Tool: Tier Mix Optimizer
IBD Momentum Investment Framework v4.0

Pure functions for:
- Solving for tier weights (w1, w2, w3) that achieve a target return
- Providing regime-specific portfolio guidance
- Estimating probability of achieving target from tier mix

No LLM, no file I/O, no buy/sell recommendations.
"""

from __future__ import annotations

import logging
import math

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

from ibd_agents.schemas.returns_projection_output import (
    RISK_FREE_RATE,
    TIER_HISTORICAL_RETURNS,
)
from ibd_agents.schemas.target_return_output import (
    DEFAULT_FRICTION,
    REGIME_GUIDANCE,
)
from ibd_agents.tools.scenario_weighter import get_scenario_weights
from ibd_agents.tools.tier_return_calculator import compute_blended_return, compute_tier_return

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------

def _compute_blended_std(
    tier_pcts: dict[int, float],
    scenario: str,
    cash_pct: float,
) -> float:
    """Compute blended portfolio standard deviation for a scenario."""
    total_var = sum(
        (tier_pcts[t] / 100.0) ** 2
        * TIER_HISTORICAL_RETURNS[t][scenario]["std"] ** 2
        for t in (1, 2, 3)
        if t in tier_pcts
    )
    return math.sqrt(total_var)


def _estimate_prob_achieve(
    expected_return: float,
    std: float,
    target: float,
) -> float:
    """
    Estimate probability of achieving target return from normal distribution.

    Uses the standard normal CDF approximation.
    """
    if std <= 0:
        return 1.0 if expected_return >= target else 0.0

    z = (target - expected_return) / std
    # Approximate CDF using error function
    prob_below = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    prob_above = 1.0 - prob_below
    return max(0.0, min(0.85, prob_above))  # cap at 0.85


def solve_tier_mix(
    target_return: float = 30.0,
    regime: str = "bull",
    friction: float = DEFAULT_FRICTION,
    sector_momentum: str = "neutral",
    cash_pct: float = 8.0,
) -> list[dict]:
    """
    Solve for tier weights that achieve a target return.

    Grid searches over w1/w2/w3 combinations and ranks solutions by
    probability of achieving the target return.

    Args:
        target_return: Target annualized return percentage.
        regime: Market regime (bull/neutral/bear).
        friction: Friction drag percentage.
        sector_momentum: Sector momentum status.
        cash_pct: Cash reserve percentage.

    Returns:
        Top 5 solutions ranked by probability, each containing:
        {w1, w2, w3, expected_return, prob_achieve, sharpe, rank}
    """
    weights = get_scenario_weights(regime)
    adjusted_target = target_return + friction

    solutions = []
    step = 5.0  # 5% increments for efficiency

    w1 = 0.0
    while w1 <= 100.0:
        w2 = 0.0
        while w2 <= 100.0 - w1:
            w3 = 100.0 - w1 - w2
            if w3 < 0:
                w2 += step
                continue

            tier_pcts = {1: w1, 2: w2, 3: w3}

            # Compute expected return across scenarios
            scenario_returns = {}
            for scenario in ("bull", "base", "bear"):
                tier_rets = {
                    t: compute_tier_return(t, scenario, sector_momentum)
                    for t in (1, 2, 3)
                }
                scenario_returns[scenario] = compute_blended_return(
                    tier_pcts, tier_rets, cash_pct,
                )

            expected_ret = sum(
                weights[s] * scenario_returns[s] for s in weights
            )
            net_return = expected_ret - friction

            # Compute portfolio std (base scenario)
            p_std = _compute_blended_std(tier_pcts, "base", cash_pct)

            # Estimate probability of achieving target
            prob = _estimate_prob_achieve(net_return, p_std, target_return)

            # Sharpe ratio
            base_ret = scenario_returns["base"] - friction
            sharpe = (base_ret - RISK_FREE_RATE) / p_std if p_std > 0 else 0.0

            solutions.append({
                "w1": w1,
                "w2": w2,
                "w3": w3,
                "expected_return": round(net_return, 2),
                "gross_return": round(expected_ret, 2),
                "prob_achieve": round(prob, 4),
                "sharpe": round(sharpe, 2),
                "std": round(p_std, 2),
            })

            w2 += step
        w1 += step

    # Sort by probability (desc), then Sharpe (desc)
    solutions.sort(key=lambda s: (-s["prob_achieve"], -s["sharpe"]))

    # Assign ranks to top 5
    top = solutions[:5]
    for i, sol in enumerate(top):
        sol["rank"] = i + 1

    logger.info(
        f"[TierMixOptimizer] Found {len(solutions)} solutions, "
        f"top prob={top[0]['prob_achieve']:.2%} at "
        f"T1={top[0]['w1']:.0f}/T2={top[0]['w2']:.0f}/T3={top[0]['w3']:.0f}"
    )

    return top


def get_regime_guidance(regime: str) -> dict:
    """
    Get regime-specific portfolio guidance.

    Args:
        regime: Market regime (bull/neutral/bear).

    Returns:
        Dict with achievability, recommended ranges, position count, etc.
    """
    return REGIME_GUIDANCE.get(regime, REGIME_GUIDANCE["neutral"])


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class TierMixOptimizerInput(BaseModel):
    target_return: float = Field(
        default=30.0, description="Target annualized return %"
    )
    regime: str = Field(
        default="bull", description="Market regime: bull/neutral/bear"
    )


class TierMixOptimizerTool(BaseTool):
    """Solve for tier weights that maximize probability of achieving target return."""

    name: str = "tier_mix_optimizer"
    description: str = (
        "Solve for optimal tier mix (T1/T2/T3 weights) that maximizes "
        "the probability of achieving a target annualized return. "
        "Returns ranked solutions with expected return and probability."
    )
    args_schema: type[BaseModel] = TierMixOptimizerInput

    def _run(
        self,
        target_return: float = 30.0,
        regime: str = "bull",
    ) -> str:
        import json

        solutions = solve_tier_mix(target_return=target_return, regime=regime)
        return json.dumps(solutions, indent=2)
