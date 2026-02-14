"""
Returns Projector Tool: Confidence Interval Builder
IBD Momentum Investment Framework v4.0

Pure functions for:
- Building percentile ranges (p10/p25/p50/p75/p90) from mean/std
- Converting return percentiles to dollar values
- Building confidence intervals across all time horizons

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
    DEFAULT_PORTFOLIO_VALUE,
    PERCENTILE_Z,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------

def build_percentile_range(
    mean: float,
    std: float,
    portfolio_value: float,
) -> dict:
    """
    Build percentile range from return distribution parameters.

    Computes p10, p25, p50, p75, p90 return percentiles using
    z-scores from PERCENTILE_Z, then converts p10/p50/p90 to
    dollar ending values.

    Args:
        mean: expected return percentage for the horizon.
        std: standard deviation percentage for the horizon.
        portfolio_value: starting portfolio value in dollars.

    Returns:
        Dict with keys: p10, p25, p50, p75, p90 (return %),
        p10_dollar, p50_dollar, p90_dollar (ending value $).
    """
    p10 = mean + PERCENTILE_Z["p10"] * std
    p25 = mean + PERCENTILE_Z["p25"] * std
    p50 = mean + PERCENTILE_Z["p50"] * std
    p75 = mean + PERCENTILE_Z["p75"] * std
    p90 = mean + PERCENTILE_Z["p90"] * std

    p10_dollar = portfolio_value * (1.0 + p10 / 100.0)
    p50_dollar = portfolio_value * (1.0 + p50 / 100.0)
    p90_dollar = portfolio_value * (1.0 + p90 / 100.0)

    return {
        "p10": round(p10, 2),
        "p25": round(p25, 2),
        "p50": round(p50, 2),
        "p75": round(p75, 2),
        "p90": round(p90, 2),
        "p10_dollar": round(p10_dollar, 2),
        "p50_dollar": round(p50_dollar, 2),
        "p90_dollar": round(p90_dollar, 2),
    }


def build_confidence_intervals(
    returns_by_horizon: dict[str, float],
    stds_by_horizon: dict[str, float],
    portfolio_value: float,
) -> dict:
    """
    Build confidence intervals across all time horizons.

    Args:
        returns_by_horizon: {"3_month": ..., "6_month": ..., "12_month": ...}
            expected return % for each horizon.
        stds_by_horizon: {"3_month": ..., "6_month": ..., "12_month": ...}
            standard deviation % for each horizon.
        portfolio_value: starting portfolio value in dollars.

    Returns:
        {"horizon_3m": {...}, "horizon_6m": {...}, "horizon_12m": {...}}
        where each value is a percentile range dict.
    """
    horizon_key_map = {
        "3_month": "horizon_3m",
        "6_month": "horizon_6m",
        "12_month": "horizon_12m",
    }

    result = {}
    for src_key, dst_key in horizon_key_map.items():
        mean = returns_by_horizon.get(src_key, 0.0)
        std = stds_by_horizon.get(src_key, 0.0)
        result[dst_key] = build_percentile_range(mean, std, portfolio_value)

    return result


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class ConfidenceIntervalBuilderInput(BaseModel):
    mean_return_pct: float = Field(
        ..., description="Expected return % for the horizon"
    )
    std_pct: float = Field(
        ..., description="Standard deviation % for the horizon"
    )
    portfolio_value: float = Field(
        DEFAULT_PORTFOLIO_VALUE,
        description="Starting portfolio value in dollars",
    )


class ConfidenceIntervalBuilderTool(BaseTool):
    """Build percentile-based confidence intervals for portfolio returns."""

    name: str = "confidence_interval_builder"
    description: str = (
        "Build percentile-based confidence intervals (p10/p25/p50/p75/p90) "
        "for portfolio returns given an expected return, standard deviation, "
        "and portfolio value. Converts return percentiles to dollar values."
    )
    args_schema: type[BaseModel] = ConfidenceIntervalBuilderInput

    def _run(
        self,
        mean_return_pct: float,
        std_pct: float,
        portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
    ) -> str:
        import json

        percentiles = build_percentile_range(mean_return_pct, std_pct, portfolio_value)
        return json.dumps({
            "mean_return_pct": mean_return_pct,
            "std_pct": std_pct,
            "portfolio_value": portfolio_value,
            **percentiles,
        })
