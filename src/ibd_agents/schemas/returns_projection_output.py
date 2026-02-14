"""
Agent 07: Returns Projector — Output Schema
IBD Momentum Investment Framework v4.0

Output contract for the Returns Projector.
Projects portfolio returns across bull/base/bear scenarios,
benchmarks against SPY/DIA/QQQ, and presents confidence intervals.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Historical return distributions by tier (annualized %)
TIER_HISTORICAL_RETURNS: dict[int, dict[str, dict[str, float]]] = {
    1: {  # Momentum: Comp>=95, RS>=90, EPS>=80
        "bull":  {"mean": 35.0, "std": 18.0},
        "base":  {"mean": 18.0, "std": 15.0},
        "bear":  {"mean": -8.0, "std": 22.0},
    },
    2: {  # Quality Growth: Comp>=85, RS>=80, EPS>=75
        "bull":  {"mean": 25.0, "std": 14.0},
        "base":  {"mean": 14.0, "std": 12.0},
        "bear":  {"mean": -5.0, "std": 16.0},
    },
    3: {  # Defensive: Comp>=80, RS>=75, EPS>=70
        "bull":  {"mean": 15.0, "std": 10.0},
        "base":  {"mean": 10.0, "std": 8.0},
        "bear":  {"mean": 2.0, "std": 12.0},
    },
}

# Benchmark return distributions (annualized %)
BENCHMARK_RETURNS: dict[str, dict[str, dict[str, float]]] = {
    "SPY": {
        "bull":  {"mean": 22.0, "std": 12.0},
        "base":  {"mean": 10.5, "std": 14.0},
        "bear":  {"mean": -15.0, "std": 18.0},
    },
    "DIA": {
        "bull":  {"mean": 18.0, "std": 11.0},
        "base":  {"mean": 9.0, "std": 13.0},
        "bear":  {"mean": -12.0, "std": 16.0},
    },
    "QQQ": {
        "bull":  {"mean": 30.0, "std": 16.0},
        "base":  {"mean": 14.0, "std": 18.0},
        "bear":  {"mean": -22.0, "std": 24.0},
    },
}

# Scenario probability weights by market regime
SCENARIO_WEIGHTS: dict[str, dict[str, float]] = {
    "bull":    {"bull": 0.55, "base": 0.35, "bear": 0.10},
    "neutral": {"bull": 0.25, "base": 0.50, "bear": 0.25},
    "bear":    {"bull": 0.10, "base": 0.35, "bear": 0.55},
}

# Trailing stop loss percentages by tier
STOP_LOSS_BY_TIER: dict[int, float] = {1: 0.22, 2: 0.18, 3: 0.12}

# Maximum portfolio loss with all stops triggered
MAX_PORTFOLIO_LOSS_WITH_STOPS: float = 17.88

# Sector momentum multiplier for return adjustments
SECTOR_MOMENTUM_MULTIPLIER: dict[str, float] = {
    "leading":   1.15,
    "improving": 1.05,
    "lagging":   0.95,
    "declining": 0.85,
}

# Time horizon definitions
TIME_HORIZONS: dict[str, dict[str, float]] = {
    "3_month":  {"fraction": 0.25, "volatility_scale": 0.50},
    "6_month":  {"fraction": 0.50, "volatility_scale": 0.71},
    "12_month": {"fraction": 1.00, "volatility_scale": 1.00},
}

# Risk-free rate and market return assumptions (%)
RISK_FREE_RATE: float = 4.5
MARKET_RETURN: float = 12.0

# Default portfolio value ($)
DEFAULT_PORTFOLIO_VALUE: float = 1_500_000.0

# Scenario names
SCENARIOS: tuple[str, ...] = ("bull", "base", "bear")

# Benchmark symbols and display names
BENCHMARKS: tuple[str, ...] = ("SPY", "DIA", "QQQ")
BENCHMARK_NAMES: dict[str, str] = {
    "SPY": "S&P 500",
    "DIA": "Dow Jones",
    "QQQ": "NASDAQ-100",
}

# Percentile z-scores for normal distribution
PERCENTILE_Z: dict[str, float] = {
    "p10": -1.28,
    "p25": -0.67,
    "p50": 0.00,
    "p75": 0.67,
    "p90": 1.28,
}

# Standard caveats (required disclaimers)
STANDARD_CAVEATS: list[str] = [
    "Projections based on historical IBD tier characteristics, not guarantees",
    "Past performance does not predict future results",
    "Trailing stops may trigger during volatile periods causing realized losses",
    "Actual returns depend on entry timing, market conditions, and execution",
    "These projections assume no portfolio changes during the holding period",
]


# ---------------------------------------------------------------------------
# Supporting Models
# ---------------------------------------------------------------------------

class AssetTypeDecomposition(BaseModel):
    """ETF vs stock return decomposition within a tier."""

    tier: int = Field(..., ge=1, le=3)
    stock_contribution_pct: float = Field(
        ..., description="Stocks' allocation % in tier"
    )
    etf_contribution_pct: float = Field(
        ..., description="ETFs' allocation % in tier"
    )
    stock_count: int = Field(0, ge=0)
    etf_count: int = Field(0, ge=0)
    stock_avg_volatility: Optional[float] = Field(None)
    etf_avg_volatility: Optional[float] = Field(
        None, description="ETFs typically ~30% less volatile"
    )


class TierAllocation(BaseModel):
    """Portfolio allocation across tiers and cash."""

    tier_1_pct: float = Field(..., description="% allocated to Tier 1 Momentum")
    tier_2_pct: float = Field(..., description="% allocated to Tier 2 Quality Growth")
    tier_3_pct: float = Field(..., description="% allocated to Tier 3 Defensive")
    cash_pct: float = Field(..., description="% held in cash")
    tier_1_value: float = Field(..., description="$ allocated to Tier 1")
    tier_2_value: float = Field(..., description="$ allocated to Tier 2")
    tier_3_value: float = Field(..., description="$ allocated to Tier 3")
    cash_value: float = Field(..., description="$ held in cash")

    @model_validator(mode="after")
    def validate_pcts_sum_to_100(self) -> "TierAllocation":
        """All pcts must sum to approximately 100% (within 2%)."""
        total = self.tier_1_pct + self.tier_2_pct + self.tier_3_pct + self.cash_pct
        if not (98.0 <= total <= 102.0):
            raise ValueError(
                f"Tier allocation pcts sum to {total:.2f}%, expected 98-102%"
            )
        return self


class ScenarioProjection(BaseModel):
    """Return projection for a single scenario (bull/base/bear)."""

    scenario: Literal["bull", "base", "bear"] = Field(...)
    probability: float = Field(..., ge=0.0, le=1.0, description="Scenario probability")
    tier_1_return_pct: float = Field(..., description="Expected T1 return %")
    tier_2_return_pct: float = Field(..., description="Expected T2 return %")
    tier_3_return_pct: float = Field(..., description="Expected T3 return %")
    portfolio_return_3m: float = Field(..., description="Portfolio return % over 3 months")
    portfolio_return_6m: float = Field(..., description="Portfolio return % over 6 months")
    portfolio_return_12m: float = Field(..., description="Portfolio return % over 12 months")
    portfolio_gain_12m: float = Field(..., description="Dollar gain/loss over 12 months")
    ending_value_12m: float = Field(..., description="Portfolio ending value at 12 months")
    tier_1_contribution: float = Field(
        ..., description="T1 contribution to portfolio return %"
    )
    tier_2_contribution: float = Field(
        ..., description="T2 contribution to portfolio return %"
    )
    tier_3_contribution: float = Field(
        ..., description="T3 contribution to portfolio return %"
    )
    reasoning: str = Field(..., min_length=30, description="Scenario rationale")


class ExpectedReturn(BaseModel):
    """Probability-weighted expected returns across all scenarios."""

    expected_3m: float = Field(
        ..., description="Expected 3-month return % (probability-weighted)"
    )
    expected_6m: float = Field(
        ..., description="Expected 6-month return % (probability-weighted)"
    )
    expected_12m: float = Field(
        ..., description="Expected 12-month return % (probability-weighted)"
    )


class BenchmarkComparison(BaseModel):
    """Comparison of portfolio returns against a benchmark."""

    symbol: Literal["SPY", "DIA", "QQQ"] = Field(...)
    name: str = Field(..., description="Benchmark display name")
    benchmark_return_3m: float = Field(..., description="Benchmark 3-month return %")
    benchmark_return_6m: float = Field(..., description="Benchmark 6-month return %")
    benchmark_return_12m: float = Field(..., description="Benchmark 12-month return %")
    alpha_bull_12m: float = Field(
        ..., description="Alpha vs benchmark in bull scenario (12m)"
    )
    alpha_base_12m: float = Field(
        ..., description="Alpha vs benchmark in base scenario (12m)"
    )
    alpha_bear_12m: float = Field(
        ..., description="Alpha vs benchmark in bear scenario (12m)"
    )
    expected_alpha_12m: float = Field(
        ..., description="Probability-weighted expected alpha (12m)"
    )
    outperform_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of outperforming benchmark"
    )


class AlphaSource(BaseModel):
    """A single source of alpha or alpha drag."""

    source: str = Field(..., description="Name of the alpha source/drag")
    contribution_pct: float = Field(..., description="Contribution to alpha %")
    confidence: Literal["high", "medium", "low"] = Field(...)
    reasoning: str = Field(..., min_length=20, description="Explanation of alpha source")


class AlphaAnalysis(BaseModel):
    """Analysis of portfolio alpha sources and drags."""

    primary_alpha_sources: List[AlphaSource] = Field(
        ..., description="Sources contributing positive alpha"
    )
    primary_alpha_drags: List[AlphaSource] = Field(
        ..., description="Sources contributing negative alpha"
    )
    net_expected_alpha_vs_spy: float = Field(
        ..., description="Net expected alpha vs SPY %"
    )
    net_expected_alpha_vs_dia: float = Field(
        ..., description="Net expected alpha vs DIA %"
    )
    net_expected_alpha_vs_qqq: float = Field(
        ..., description="Net expected alpha vs QQQ %"
    )
    alpha_persistence_note: str = Field(
        ..., min_length=30,
        description="Note on alpha persistence and sustainability"
    )


class RiskMetrics(BaseModel):
    """Portfolio risk metrics and Sharpe ratios."""

    max_drawdown_with_stops: float = Field(
        ..., description="Max drawdown % with trailing stops"
    )
    max_drawdown_without_stops: float = Field(
        ..., description="Max drawdown % without trailing stops"
    )
    max_drawdown_dollar: float = Field(
        ..., description="Max drawdown in dollar terms"
    )
    projected_annual_volatility: float = Field(
        ..., description="Projected annual portfolio volatility %"
    )
    spy_annual_volatility: float = Field(
        ..., description="SPY annual volatility % for comparison"
    )
    portfolio_sharpe_bull: float = Field(
        ..., description="Portfolio Sharpe ratio in bull scenario"
    )
    portfolio_sharpe_base: float = Field(
        ..., description="Portfolio Sharpe ratio in base scenario"
    )
    portfolio_sharpe_bear: float = Field(
        ..., description="Portfolio Sharpe ratio in bear scenario"
    )
    spy_sharpe_base: float = Field(
        ..., description="SPY Sharpe ratio in base scenario"
    )
    return_per_unit_risk: float = Field(
        ..., description="Portfolio return per unit of risk"
    )
    probability_any_stop_triggers_12m: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability any trailing stop triggers in 12 months"
    )
    expected_stops_triggered_12m: int = Field(
        ..., ge=0,
        description="Expected number of stops triggered in 12 months"
    )
    whipsaw_risk_note: str = Field(
        ..., min_length=20,
        description="Note on whipsaw risk from trailing stops"
    )

    @model_validator(mode="after")
    def validate_drawdown_ordering(self) -> "RiskMetrics":
        """Max drawdown with stops must be <= max drawdown without stops."""
        if self.max_drawdown_with_stops > self.max_drawdown_without_stops + 0.01:
            raise ValueError(
                f"max_drawdown_with_stops ({self.max_drawdown_with_stops:.2f}%) "
                f"exceeds max_drawdown_without_stops ({self.max_drawdown_without_stops:.2f}%)"
            )
        return self


class TierMixScenario(BaseModel):
    """Alternative tier mix scenario for comparison."""

    label: str = Field(..., description="Scenario label (e.g. 'Aggressive')")
    tier_1_pct: float = Field(..., description="% allocated to Tier 1")
    tier_2_pct: float = Field(..., description="% allocated to Tier 2")
    tier_3_pct: float = Field(..., description="% allocated to Tier 3")
    cash_pct: float = Field(..., description="% held in cash")
    expected_return_12m: float = Field(
        ..., description="Expected 12-month return %"
    )
    bull_return_12m: float = Field(
        ..., description="Bull scenario 12-month return %"
    )
    bear_return_12m: float = Field(
        ..., description="Bear scenario 12-month return %"
    )
    max_drawdown_with_stops: float = Field(
        ..., description="Max drawdown % with trailing stops"
    )
    sharpe_ratio_base: float = Field(
        ..., description="Sharpe ratio in base scenario"
    )
    comparison_note: str = Field(
        ..., min_length=20,
        description="Note comparing this mix to the recommended allocation"
    )


class PercentileRange(BaseModel):
    """Return distribution percentiles for a time horizon."""

    p10: float = Field(..., description="10th percentile return %")
    p25: float = Field(..., description="25th percentile return %")
    p50: float = Field(..., description="50th percentile (median) return %")
    p75: float = Field(..., description="75th percentile return %")
    p90: float = Field(..., description="90th percentile return %")
    p10_dollar: float = Field(..., description="10th percentile ending value $")
    p50_dollar: float = Field(..., description="50th percentile ending value $")
    p90_dollar: float = Field(..., description="90th percentile ending value $")

    @model_validator(mode="after")
    def validate_percentile_ordering(self) -> "PercentileRange":
        """Percentiles must be in non-decreasing order: p10 <= p25 <= p50 <= p75 <= p90."""
        if not (
            self.p10 <= self.p25 + 0.01
            and self.p25 <= self.p50 + 0.01
            and self.p50 <= self.p75 + 0.01
            and self.p75 <= self.p90 + 0.01
        ):
            raise ValueError(
                f"Percentiles not in order: "
                f"p10={self.p10}, p25={self.p25}, p50={self.p50}, "
                f"p75={self.p75}, p90={self.p90}"
            )
        return self


class ConfidenceIntervals(BaseModel):
    """Confidence intervals across all time horizons."""

    horizon_3m: PercentileRange = Field(..., description="3-month confidence intervals")
    horizon_6m: PercentileRange = Field(..., description="6-month confidence intervals")
    horizon_12m: PercentileRange = Field(..., description="12-month confidence intervals")


# ---------------------------------------------------------------------------
# Top-level Output
# ---------------------------------------------------------------------------

class ReturnsProjectionOutput(BaseModel):
    """
    Top-level output contract for the Returns Projector.

    Projects portfolio returns across bull/base/bear scenarios,
    benchmarks against SPY/DIA/QQQ, and presents confidence intervals.
    """

    portfolio_value: float = Field(
        ..., gt=0.0, description="Starting portfolio value ($)"
    )
    analysis_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD"
    )
    market_regime: str = Field(
        ..., description="Current market regime (bull/neutral/bear)"
    )
    regime_source: str = Field(
        default="Rotation Detector Agent 03",
        description="Source of market regime determination"
    )
    tier_allocation: TierAllocation = Field(...)
    scenarios: List[ScenarioProjection] = Field(
        ..., min_length=3, max_length=3,
        description="Exactly 3 scenario projections (bull/base/bear)"
    )
    expected_return: ExpectedReturn = Field(...)
    benchmark_comparisons: List[BenchmarkComparison] = Field(
        ..., min_length=3, max_length=3,
        description="Exactly 3 benchmark comparisons (SPY/DIA/QQQ)"
    )
    alpha_analysis: AlphaAnalysis = Field(...)
    risk_metrics: RiskMetrics = Field(...)
    tier_mix_comparison: List[TierMixScenario] = Field(
        ..., min_length=3, max_length=3,
        description="Exactly 3 tier mix scenarios for comparison"
    )
    confidence_intervals: ConfidenceIntervals = Field(...)
    asset_type_decomposition: Optional[List[AssetTypeDecomposition]] = Field(
        None, description="ETF vs stock return decomposition per tier"
    )
    summary: str = Field(..., min_length=50, description="Executive summary")
    caveats: List[str] = Field(
        ..., min_length=1, description="Required disclaimers"
    )

    # --- Validators ---

    @model_validator(mode="after")
    def validate_scenario_count(self) -> "ReturnsProjectionOutput":
        """Scenarios must have exactly 3 items."""
        if len(self.scenarios) != 3:
            raise ValueError(
                f"Expected exactly 3 scenarios, got {len(self.scenarios)}"
            )
        return self

    @model_validator(mode="after")
    def validate_scenario_probabilities_sum(self) -> "ReturnsProjectionOutput":
        """Scenario probabilities must sum to approximately 1.0 (within 0.01)."""
        total = sum(s.probability for s in self.scenarios)
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Scenario probabilities sum to {total:.4f}, expected ~1.0 (within 0.01)"
            )
        return self

    @model_validator(mode="after")
    def validate_benchmark_count(self) -> "ReturnsProjectionOutput":
        """Benchmark comparisons must have exactly 3 items."""
        if len(self.benchmark_comparisons) != 3:
            raise ValueError(
                f"Expected exactly 3 benchmark comparisons, "
                f"got {len(self.benchmark_comparisons)}"
            )
        return self

    @model_validator(mode="after")
    def validate_tier_mix_count(self) -> "ReturnsProjectionOutput":
        """Tier mix comparison must have exactly 3 items."""
        if len(self.tier_mix_comparison) != 3:
            raise ValueError(
                f"Expected exactly 3 tier mix scenarios, "
                f"got {len(self.tier_mix_comparison)}"
            )
        return self

    @model_validator(mode="after")
    def validate_caveats_non_empty(self) -> "ReturnsProjectionOutput":
        """Caveats must be non-empty."""
        if not self.caveats:
            raise ValueError("caveats list must not be empty")
        return self
