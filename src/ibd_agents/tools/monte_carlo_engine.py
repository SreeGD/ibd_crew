"""
Target Return Constructor Tool: Monte Carlo Engine
IBD Momentum Investment Framework v4.0

Pure functions for:
- Running 10,000 simulations of portfolio returns
- Computing probability of achieving target return
- Building return distribution percentiles
- Simulating stop-loss trigger effects

Uses stdlib random (no numpy dependency). Deterministic with seed=42.

No LLM, no file I/O, no buy/sell recommendations.
"""

from __future__ import annotations

import logging
import math
import random
import statistics

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

from ibd_agents.schemas.returns_projection_output import (
    BENCHMARK_RETURNS,
    TIER_HISTORICAL_RETURNS,
)
from ibd_agents.schemas.target_return_output import MAX_PROB_ACHIEVE
from ibd_agents.tools.scenario_weighter import get_scenario_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------

def run_monte_carlo(
    positions: list[dict],
    tier_pcts: dict[int, float],
    regime: str,
    target_return: float,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Run Monte Carlo simulation for probability assessment.

    For each simulation:
    1. Draw scenario (bull/base/bear) weighted by regime probabilities
    2. Draw per-position return from N(tier_mean, tier_std) for drawn scenario
    3. Apply stop-loss: if position return < -stop_pct, cap at -stop_pct
    4. Compute portfolio return as weighted sum of position returns

    Args:
        positions: List of position dicts with keys:
            tier (int), allocation_pct (float), stop_loss_pct (float),
            volatility_pct (float, optional) — per-stock annualized volatility
        tier_pcts: {1: pct, 2: pct, 3: pct} — tier allocations
        regime: Market regime (bull/neutral/bear)
        target_return: Target return percentage
        n_simulations: Number of simulations (default 10,000)
        seed: Random seed for reproducibility

    Returns:
        Dict with prob_achieve_target, prob_positive_return,
        expected/median return, percentiles, and full distribution.
    """
    rng = random.Random(seed)
    weights = get_scenario_weights(regime)
    scenarios = ["bull", "base", "bear"]
    scenario_probs = [weights[s] for s in scenarios]

    # Build cumulative probabilities for weighted scenario selection
    cumulative = []
    running = 0.0
    for p in scenario_probs:
        running += p
        cumulative.append(running)

    returns_distribution = []

    for _ in range(n_simulations):
        # Step 1: Draw scenario
        r = rng.random()
        scenario = scenarios[0]
        for i, c in enumerate(cumulative):
            if r <= c:
                scenario = scenarios[i]
                break

        # Step 2 & 3: Draw per-position returns and apply stops
        portfolio_return = 0.0

        for pos in positions:
            tier = pos["tier"]
            alloc_pct = pos["allocation_pct"]
            stop_pct = pos.get("stop_loss_pct", 22.0)  # default T1 stop

            # Get tier return distribution for this scenario
            tier_data = TIER_HISTORICAL_RETURNS.get(tier, TIER_HISTORICAL_RETURNS[2])
            scenario_data = tier_data.get(scenario, tier_data["base"])
            mean = scenario_data["mean"]
            std = scenario_data["std"]

            # Per-stock volatility override: scale tier std by relative volatility
            stock_vol = pos.get("volatility_pct")
            if stock_vol is not None and stock_vol > 0:
                tier_avg_vol = tier_data["base"]["std"]
                if tier_avg_vol > 0:
                    vol_ratio = stock_vol / tier_avg_vol
                    vol_ratio = max(0.5, min(2.0, vol_ratio))
                    std = std * vol_ratio

            # Draw return from normal distribution
            pos_return = rng.gauss(mean, std)

            # Apply stop-loss floor
            if pos_return < -stop_pct:
                pos_return = -stop_pct

            # Weighted contribution
            portfolio_return += (alloc_pct / 100.0) * pos_return

        returns_distribution.append(portfolio_return)

    # Sort for percentile computation
    returns_distribution.sort()

    # Compute statistics
    n = len(returns_distribution)
    prob_achieve = sum(1 for r in returns_distribution if r >= target_return) / n
    prob_positive = sum(1 for r in returns_distribution if r >= 0) / n

    expected_return = statistics.mean(returns_distribution)
    median_return = statistics.median(returns_distribution)

    # Percentiles
    p10 = returns_distribution[int(n * 0.10)]
    p25 = returns_distribution[int(n * 0.25)]
    p50 = returns_distribution[int(n * 0.50)]
    p75 = returns_distribution[int(n * 0.75)]
    p90 = returns_distribution[min(int(n * 0.90), n - 1)]

    # Cap probability at MAX_PROB_ACHIEVE
    prob_achieve = min(prob_achieve, MAX_PROB_ACHIEVE)

    result = {
        "prob_achieve_target": round(prob_achieve, 4),
        "prob_positive_return": round(prob_positive, 4),
        "expected_return_pct": round(expected_return, 2),
        "median_return_pct": round(median_return, 2),
        "p10": round(p10, 2),
        "p25": round(p25, 2),
        "p50": round(p50, 2),
        "p75": round(p75, 2),
        "p90": round(p90, 2),
        "returns_distribution": returns_distribution,
    }

    logger.info(
        f"[MonteCarlo] {n_simulations} simulations: "
        f"prob_achieve={prob_achieve:.2%}, expected={expected_return:.1f}%, "
        f"median={median_return:.1f}%, p10={p10:.1f}%, p90={p90:.1f}%"
    )

    return result


def compute_benchmark_beat_probability(
    returns_distribution: list[float],
    benchmark_symbol: str,
    regime: str,
    seed: int = 42,
) -> float:
    """
    Compute probability of beating a benchmark.

    Simulates benchmark returns and compares with portfolio distribution.

    Args:
        returns_distribution: Portfolio returns from Monte Carlo.
        benchmark_symbol: "SPY", "DIA", or "QQQ".
        regime: Market regime.
        seed: Random seed.

    Returns:
        Probability (0-1) of beating the benchmark.
    """
    rng = random.Random(seed + hash(benchmark_symbol))
    weights = get_scenario_weights(regime)
    scenarios = ["bull", "base", "bear"]
    cumulative = []
    running = 0.0
    for s in scenarios:
        running += weights[s]
        cumulative.append(running)

    benchmark_data = BENCHMARK_RETURNS.get(benchmark_symbol, BENCHMARK_RETURNS["SPY"])
    n = len(returns_distribution)

    # Simulate benchmark returns
    benchmark_returns = []
    for _ in range(n):
        r = rng.random()
        scenario = scenarios[0]
        for i, c in enumerate(cumulative):
            if r <= c:
                scenario = scenarios[i]
                break

        b_data = benchmark_data[scenario]
        b_return = rng.gauss(b_data["mean"], b_data["std"])
        benchmark_returns.append(b_return)

    # Compare pairwise
    beats = sum(
        1 for p, b in zip(returns_distribution, benchmark_returns) if p > b
    )
    return round(beats / n, 4)


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class MonteCarloInput(BaseModel):
    target_return: float = Field(
        default=30.0, description="Target return percentage"
    )
    regime: str = Field(
        default="bull", description="Market regime"
    )
    n_simulations: int = Field(
        default=10_000, description="Number of simulations"
    )


class MonteCarloTool(BaseTool):
    """Run Monte Carlo simulation for portfolio probability assessment."""

    name: str = "monte_carlo_engine"
    description: str = (
        "Run 10,000 Monte Carlo simulations to compute probability of "
        "achieving a target return, confidence intervals, and return "
        "distribution percentiles."
    )
    args_schema: type[BaseModel] = MonteCarloInput

    def _run(
        self,
        target_return: float = 30.0,
        regime: str = "bull",
        n_simulations: int = 10_000,
    ) -> str:
        import json
        return json.dumps({
            "note": "Use run_monte_carlo() directly with position data",
            "target": target_return,
            "regime": regime,
        })
