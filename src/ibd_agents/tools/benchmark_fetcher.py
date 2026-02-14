"""
Returns Projector Tool: Benchmark Fetcher
IBD Momentum Investment Framework v4.0

Pure functions for:
- Retrieving benchmark return distributions (SPY, DIA, QQQ)
- Computing probability-weighted expected benchmark returns

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
    BENCHMARK_RETURNS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------

def get_benchmark_stats(
    symbol: str,
    scenario: str,
) -> dict[str, float]:
    """
    Retrieve benchmark return statistics for a symbol and scenario.

    Args:
        symbol: one of "SPY", "DIA", "QQQ".
        scenario: "bull", "base", or "bear".

    Returns:
        {"mean": ..., "std": ...} annualized percentages.
    """
    return {
        "mean": BENCHMARK_RETURNS[symbol][scenario]["mean"],
        "std": BENCHMARK_RETURNS[symbol][scenario]["std"],
    }


def compute_benchmark_expected(
    symbol: str,
    weights: dict[str, float],
) -> float:
    """
    Compute probability-weighted expected return for a benchmark.

    Args:
        symbol: one of "SPY", "DIA", "QQQ".
        weights: scenario probability weights, e.g.
            {"bull": 0.25, "base": 0.50, "bear": 0.25}.

    Returns:
        Expected annualized return percentage.
    """
    return sum(
        weights[s] * BENCHMARK_RETURNS[symbol][s]["mean"]
        for s in weights
    )


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class BenchmarkFetcherInput(BaseModel):
    symbol: str = Field(..., description="Benchmark symbol: SPY, DIA, or QQQ")
    scenario: str = Field(
        "base",
        description="Scenario: bull, base, or bear (for single-scenario lookup)",
    )


class BenchmarkFetcherTool(BaseTool):
    """Retrieve benchmark return distributions for SPY/DIA/QQQ."""

    name: str = "benchmark_fetcher"
    description: str = (
        "Retrieve the mean return and standard deviation for a benchmark "
        "(SPY, DIA, QQQ) under a given scenario (bull/base/bear), or "
        "compute the probability-weighted expected return."
    )
    args_schema: type[BaseModel] = BenchmarkFetcherInput

    def _run(self, symbol: str, scenario: str = "base") -> str:
        import json

        symbol = symbol.upper()
        if symbol not in BENCHMARK_RETURNS:
            return f"Unknown benchmark symbol: {symbol}. Valid: SPY, DIA, QQQ"

        stats = get_benchmark_stats(symbol, scenario)
        return json.dumps({
            "symbol": symbol,
            "scenario": scenario,
            "mean_return_pct": stats["mean"],
            "std_pct": stats["std"],
        })
