"""
Returns Projector Tool: Alpha Calculator
IBD Momentum Investment Framework v4.0

Pure functions for:
- Computing portfolio alpha vs benchmarks
- Computing probability-weighted expected alpha
- Decomposing alpha into tier-level sources
- Estimating outperformance probability

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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------

def compute_alpha(
    portfolio_return: float,
    benchmark_return: float,
) -> float:
    """
    Compute alpha as simple excess return.

    Args:
        portfolio_return: portfolio return %.
        benchmark_return: benchmark return %.

    Returns:
        Alpha percentage (portfolio - benchmark).
    """
    return portfolio_return - benchmark_return


def compute_expected_alpha(
    portfolio_scenarios: dict[str, float],
    benchmark_scenarios: dict[str, float],
    weights: dict[str, float],
) -> float:
    """
    Compute probability-weighted expected alpha across scenarios.

    Args:
        portfolio_scenarios: {"bull": ..., "base": ..., "bear": ...} portfolio returns %.
        benchmark_scenarios: {"bull": ..., "base": ..., "bear": ...} benchmark returns %.
        weights: {"bull": ..., "base": ..., "bear": ...} probability weights.

    Returns:
        Expected alpha percentage.
    """
    return sum(
        weights[s] * (portfolio_scenarios[s] - benchmark_scenarios[s])
        for s in weights
    )


def decompose_alpha(
    tier_contributions: dict[int, float],
    benchmark_return: float,
    sector_boost_pct: float = 0.0,
) -> list[dict]:
    """
    Decompose portfolio alpha into tier-level sources.

    Uses default tier weight fractions:
        T1 = 0.39, T2 = 0.37, T3 = 0.22

    Args:
        tier_contributions: {1: ..., 2: ..., 3: ...} each tier's
            contribution to portfolio return (pct).
        benchmark_return: benchmark annualized return %.
        sector_boost_pct: additional alpha from sector momentum tilt (pct).

    Returns:
        List of alpha source dicts with keys:
            source, contribution_pct, confidence, reasoning.
    """
    tier_weight_fractions = {1: 0.39, 2: 0.37, 3: 0.22}
    sources: list[dict] = []

    # T1 Momentum outperformance
    t1_alpha = tier_contributions.get(1, 0.0) - benchmark_return * tier_weight_fractions[1]
    t1_conf = _classify_confidence(t1_alpha)
    sources.append({
        "source": "T1 Momentum outperformance",
        "contribution_pct": round(t1_alpha, 2),
        "confidence": t1_conf,
        "reasoning": (
            f"Tier 1 momentum stocks contribute {tier_contributions.get(1, 0.0):.2f}% "
            f"vs benchmark-equivalent allocation of {benchmark_return * tier_weight_fractions[1]:.2f}%"
        ),
    })

    # T2 Quality growth
    t2_alpha = tier_contributions.get(2, 0.0) - benchmark_return * tier_weight_fractions[2]
    t2_conf = _classify_confidence(t2_alpha)
    sources.append({
        "source": "T2 Quality growth",
        "contribution_pct": round(t2_alpha, 2),
        "confidence": t2_conf,
        "reasoning": (
            f"Tier 2 quality growth stocks contribute {tier_contributions.get(2, 0.0):.2f}% "
            f"vs benchmark-equivalent allocation of {benchmark_return * tier_weight_fractions[2]:.2f}%"
        ),
    })

    # T3 Defensive drag
    t3_alpha = tier_contributions.get(3, 0.0) - benchmark_return * tier_weight_fractions[3]
    t3_conf = _classify_confidence(t3_alpha)
    sources.append({
        "source": "T3 Defensive drag",
        "contribution_pct": round(t3_alpha, 2),
        "confidence": t3_conf,
        "reasoning": (
            f"Tier 3 defensive stocks contribute {tier_contributions.get(3, 0.0):.2f}% "
            f"vs benchmark-equivalent allocation of {benchmark_return * tier_weight_fractions[3]:.2f}%"
        ),
    })

    # Sector momentum tilt
    sec_conf = _classify_confidence(sector_boost_pct)
    sources.append({
        "source": "Sector momentum tilt",
        "contribution_pct": round(sector_boost_pct, 2),
        "confidence": sec_conf,
        "reasoning": (
            f"Sector momentum adjustments add {sector_boost_pct:.2f}% "
            f"based on current sector leadership positioning"
        ),
    })

    return sources


def estimate_outperform_probability(
    p_mean: float,
    p_std: float,
    b_mean: float,
    b_std: float,
) -> float:
    """
    Estimate probability that portfolio outperforms benchmark.

    Uses difference distribution (portfolio - benchmark) and
    computes P(diff > 0) via sigmoid approximation to normal CDF.

    Args:
        p_mean: portfolio expected return %.
        p_std: portfolio standard deviation %.
        b_mean: benchmark expected return %.
        b_std: benchmark standard deviation %.

    Returns:
        Probability of outperformance, clamped to [0.01, 0.99].
    """
    mean_diff = p_mean - b_mean
    std_diff = math.sqrt(p_std ** 2 + b_std ** 2)

    if std_diff > 0:
        z = mean_diff / std_diff
    else:
        z = 0.0

    # Sigmoid approximation to normal CDF: Phi(z) ~ 1 / (1 + exp(-1.7*z))
    prob = 1.0 / (1.0 + math.exp(-1.7 * z))

    # Clamp to [0.01, 0.99]
    return max(0.01, min(0.99, prob))


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _classify_confidence(contribution: float) -> str:
    """Classify confidence based on absolute contribution magnitude."""
    abs_val = abs(contribution)
    if abs_val > 2.0:
        return "high"
    elif abs_val > 0.5:
        return "medium"
    else:
        return "low"


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class AlphaCalculatorInput(BaseModel):
    portfolio_return: float = Field(..., description="Portfolio annualized return %")
    benchmark_return: float = Field(..., description="Benchmark annualized return %")


class AlphaCalculatorTool(BaseTool):
    """Compute portfolio alpha and outperformance probability vs benchmarks."""

    name: str = "alpha_calculator"
    description: str = (
        "Compute portfolio alpha (excess return vs benchmark), "
        "decompose alpha into tier-level sources, and estimate "
        "the probability of outperforming a given benchmark."
    )
    args_schema: type[BaseModel] = AlphaCalculatorInput

    def _run(
        self,
        portfolio_return: float,
        benchmark_return: float,
    ) -> str:
        import json

        alpha = compute_alpha(portfolio_return, benchmark_return)
        return json.dumps({
            "portfolio_return_pct": portfolio_return,
            "benchmark_return_pct": benchmark_return,
            "alpha_pct": round(alpha, 2),
        })
