"""
Agent 14: Target Return Portfolio Constructor — Output Schema
IBD Momentum Investment Framework v4.0

Output contract for the Target Return Constructor.
Reverse-engineers a portfolio targeting a specific annualized return
by optimizing tier mix, selecting/sizing positions, and providing
probability assessments with alternatives.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TARGET_RETURN: float = 30.0
DEFAULT_TOTAL_CAPITAL: float = 250_000.0
DEFAULT_HORIZON_MONTHS: int = 12

ACHIEVABILITY_RATINGS: tuple[str, ...] = (
    "REALISTIC", "STRETCH", "AGGRESSIVE", "IMPROBABLE",
)

CONVICTION_LEVELS: tuple[str, ...] = ("HIGH", "MEDIUM", "MODERATE")

ENTRY_STRATEGIES: tuple[str, ...] = ("market", "limit_at_pivot", "pullback_buy")

# Never output probability > 0.85 (overconfidence cap)
MAX_PROB_ACHIEVE: float = 0.85

MIN_POSITIONS: int = 5
MAX_POSITIONS: int = 20

MAX_SINGLE_POSITION_PCT: float = 15.0
MAX_SECTOR_CONCENTRATION_PCT: float = 40.0

# Friction drag % (stops, slippage, taxes)
DEFAULT_FRICTION: float = 4.0

# Regime-specific guidance
REGIME_GUIDANCE: dict[str, dict] = {
    "bull": {
        "achievability": "REALISTIC",
        "recommended_t1_range": (50, 60),
        "recommended_t2_range": (30, 40),
        "recommended_t3_range": (0, 10),
        "position_count_range": (8, 12),
        "cash_reserve_range": (5, 10),
        "top_concentration_pct": (12, 15),
    },
    "neutral": {
        "achievability": "STRETCH",
        "recommended_t1_range": (55, 65),
        "recommended_t2_range": (25, 35),
        "recommended_t3_range": (5, 10),
        "position_count_range": (6, 10),
        "cash_reserve_range": (10, 15),
        "top_concentration_pct": (12, 15),
    },
    "bear": {
        "achievability": "IMPROBABLE",
        "recommended_t1_range": (30, 40),
        "recommended_t2_range": (20, 30),
        "recommended_t3_range": (20, 30),
        "position_count_range": (6, 8),
        "cash_reserve_range": (20, 30),
        "top_concentration_pct": (10, 12),
    },
}

# Standard risk disclaimer
STANDARD_DISCLAIMER: str = (
    "Projected returns are probability-weighted estimates based on "
    "historical IBD tier performance and current market regime assessment. "
    "Actual returns may differ materially. Past performance of IBD-rated "
    "stocks does not guarantee future results. This is not investment advice."
)


# ---------------------------------------------------------------------------
# Supporting Models
# ---------------------------------------------------------------------------

class TargetTierAllocation(BaseModel):
    """Tier weight allocation optimized for target return."""

    t1_momentum_pct: float = Field(..., ge=0, le=100)
    t2_quality_growth_pct: float = Field(..., ge=0, le=100)
    t3_defensive_pct: float = Field(..., ge=0, le=100)
    rationale: str = Field(
        ..., min_length=20,
        description="Why this mix maximizes probability of hitting target",
    )

    @model_validator(mode="after")
    def tiers_sum_to_100(self) -> "TargetTierAllocation":
        total = self.t1_momentum_pct + self.t2_quality_growth_pct + self.t3_defensive_pct
        if not (98.0 <= total <= 102.0):
            raise ValueError(
                f"Tier allocations must sum to ~100%, got {total:.1f}%"
            )
        return self


class SectorWeight(BaseModel):
    """Sector allocation within the target portfolio."""

    sector: str = Field(..., min_length=1)
    weight_pct: float = Field(..., ge=0, le=100)
    stock_count: int = Field(..., ge=0)


class TargetPosition(BaseModel):
    """A position in the target return portfolio."""

    ticker: str = Field(..., min_length=1, max_length=10)
    company_name: str = Field(..., min_length=1)
    tier: int = Field(..., ge=1, le=3)
    allocation_pct: float = Field(..., ge=1.0, le=20.0)
    dollar_amount: float = Field(..., ge=0)
    shares: int = Field(..., ge=1)
    entry_strategy: str = Field(..., description="market/limit_at_pivot/pullback_buy")
    target_entry_price: float = Field(..., gt=0)
    stop_loss_price: float = Field(..., gt=0)
    stop_loss_pct: float = Field(..., gt=0, le=30)

    # Return contribution
    expected_return_contribution_pct: float = Field(
        ..., description="This position's weighted contribution to total portfolio return",
    )
    conviction_level: str = Field(..., description="HIGH/MEDIUM/MODERATE")
    selection_rationale: str = Field(
        ..., min_length=10,
        description="Why this stock over alternatives in same tier/sector",
    )

    # Key metrics
    composite_score: float = Field(..., ge=0, le=99)
    eps_rating: int = Field(..., ge=1, le=99)
    rs_rating: int = Field(..., ge=1, le=99)
    sector: str = Field(..., min_length=1)
    sector_rank: int = Field(..., ge=1)
    multi_source_count: int = Field(..., ge=0)

    # LLM enrichment tracking
    selection_source: Optional[Literal["llm", "template"]] = None

    @field_validator("entry_strategy")
    @classmethod
    def valid_entry_strategy(cls, v: str) -> str:
        if v not in ENTRY_STRATEGIES:
            raise ValueError(f"entry_strategy must be one of {ENTRY_STRATEGIES}, got '{v}'")
        return v

    @field_validator("conviction_level")
    @classmethod
    def valid_conviction(cls, v: str) -> str:
        if v not in CONVICTION_LEVELS:
            raise ValueError(f"conviction_level must be one of {CONVICTION_LEVELS}, got '{v}'")
        return v

    @model_validator(mode="after")
    def stop_below_entry(self) -> "TargetPosition":
        if self.stop_loss_price >= self.target_entry_price:
            raise ValueError(
                f"stop_loss_price ({self.stop_loss_price}) must be below "
                f"target_entry_price ({self.target_entry_price})"
            )
        return self


class ProbabilityAssessment(BaseModel):
    """Monte Carlo results for target achievement."""

    prob_achieve_target: float = Field(
        ..., ge=0, le=1.0,
        description="Probability of achieving target return",
    )
    prob_positive_return: float = Field(..., ge=0, le=1.0)
    prob_beat_sp500: float = Field(..., ge=0, le=1.0)
    prob_beat_nasdaq: float = Field(..., ge=0, le=1.0)
    prob_beat_dow: float = Field(..., ge=0, le=1.0)

    expected_return_pct: float = Field(
        ..., description="Probability-weighted expected return",
    )
    median_return_pct: float

    # Confidence intervals
    p10_return_pct: float
    p25_return_pct: float
    p50_return_pct: float
    p75_return_pct: float
    p90_return_pct: float

    # What must go right / wrong
    key_assumptions: List[str] = Field(..., min_length=3)
    key_risks: List[str] = Field(..., min_length=3)

    @field_validator("prob_achieve_target")
    @classmethod
    def cap_overconfidence(cls, v: float) -> float:
        if v > MAX_PROB_ACHIEVE:
            return MAX_PROB_ACHIEVE
        return v

    @model_validator(mode="after")
    def percentiles_ordered(self) -> "ProbabilityAssessment":
        vals = [
            self.p10_return_pct,
            self.p25_return_pct,
            self.p50_return_pct,
            self.p75_return_pct,
            self.p90_return_pct,
        ]
        for i in range(len(vals) - 1):
            if vals[i] > vals[i + 1] + 0.01:  # small tolerance
                raise ValueError(
                    f"Percentiles must be ordered: p10={vals[0]}, p25={vals[1]}, "
                    f"p50={vals[2]}, p75={vals[3]}, p90={vals[4]}"
                )
        return self


class TargetReturnScenario(BaseModel):
    """A bull/base/bear scenario for the target portfolio."""

    name: str = Field(..., description="Bull/Base/Bear")
    probability_pct: float = Field(..., ge=0, le=100)
    portfolio_return_pct: float
    sp500_return_pct: float
    nasdaq_return_pct: float
    dow_return_pct: float
    alpha_vs_sp500: float
    max_drawdown_pct: float = Field(..., le=0.01)  # should be negative or zero
    description: str = Field(
        ..., min_length=20,
        description="What this scenario looks like",
    )
    top_contributors: List[str] = Field(..., min_length=1)
    biggest_drags: List[str] = Field(..., min_length=1)
    stops_triggered: int = Field(..., ge=0)


class ScenarioAnalysis(BaseModel):
    """Three-scenario analysis for target portfolio."""

    bull_scenario: TargetReturnScenario
    base_scenario: TargetReturnScenario
    bear_scenario: TargetReturnScenario

    @model_validator(mode="after")
    def probabilities_sum(self) -> "ScenarioAnalysis":
        total = (
            self.bull_scenario.probability_pct
            + self.base_scenario.probability_pct
            + self.bear_scenario.probability_pct
        )
        if not (98.0 <= total <= 102.0):
            raise ValueError(
                f"Scenario probabilities must sum to ~100%, got {total:.1f}%"
            )
        return self


class AlternativePortfolio(BaseModel):
    """Alternative construction with different risk/return tradeoff."""

    name: str = Field(..., min_length=3)
    target_return_pct: float
    prob_achieve_target: float = Field(..., ge=0, le=1.0)
    position_count: int = Field(..., ge=MIN_POSITIONS, le=MAX_POSITIONS)
    t1_pct: float = Field(..., ge=0, le=100)
    t2_pct: float = Field(..., ge=0, le=100)
    t3_pct: float = Field(..., ge=0, le=100)
    max_drawdown_pct: float
    key_difference: str = Field(
        ..., min_length=10,
        description="How this differs from primary portfolio",
    )
    tradeoff: str = Field(
        ..., min_length=10,
        description="What you gain and what you give up vs primary",
    )

    # LLM enrichment tracking
    reasoning_source: Optional[Literal["llm", "template"]] = None


class TransitionAction(BaseModel):
    """A single transition action (sell/buy/resize)."""

    ticker: str = Field(..., min_length=1, max_length=10)
    action: str = Field(..., description="SELL_FULL/SELL_PARTIAL/BUY_NEW/RESIZE_UP/RESIZE_DOWN")
    current_allocation_pct: Optional[float] = None
    target_allocation_pct: Optional[float] = None
    dollar_amount: float
    priority: int = Field(..., ge=1)
    rationale: str = Field(..., min_length=5)


class TransitionPlan(BaseModel):
    """How to get from current portfolio to target portfolio."""

    positions_to_sell: List[TransitionAction] = Field(default_factory=list)
    positions_to_buy: List[TransitionAction] = Field(default_factory=list)
    positions_to_resize: List[TransitionAction] = Field(default_factory=list)
    positions_to_keep: List[str] = Field(default_factory=list)
    transition_urgency: str = Field(
        ..., description="immediate/phase_over_1_week/phase_over_2_weeks",
    )
    transition_sequence: List[str] = Field(
        ..., min_length=1,
        description="Ordered steps: which trades to execute first and why",
    )


class RiskDisclosure(BaseModel):
    """Mandatory risk context."""

    achievability_rating: str = Field(
        ..., description="REALISTIC/STRETCH/AGGRESSIVE/IMPROBABLE",
    )
    achievability_rationale: str = Field(..., min_length=20)
    max_expected_drawdown_pct: float
    recovery_time_months: float = Field(
        ..., gt=0, description="Expected months to recover from max drawdown",
    )
    conditions_for_success: List[str] = Field(..., min_length=3)
    conditions_for_failure: List[str] = Field(..., min_length=3)
    disclaimer: str = Field(default=STANDARD_DISCLAIMER)

    @field_validator("achievability_rating")
    @classmethod
    def valid_rating(cls, v: str) -> str:
        if v not in ACHIEVABILITY_RATINGS:
            raise ValueError(
                f"achievability_rating must be one of {ACHIEVABILITY_RATINGS}, got '{v}'"
            )
        return v


# ---------------------------------------------------------------------------
# Main Output Model
# ---------------------------------------------------------------------------

class TargetReturnOutput(BaseModel):
    """Complete output from Target Return Portfolio Constructor (Agent 14)."""

    # Portfolio identity
    portfolio_name: str = Field(
        ..., min_length=5,
        description="e.g., 'Growth-30 Portfolio — Bull Regime Feb 2026'",
    )
    target_return_pct: float = Field(..., ge=5.0, le=100.0)
    time_horizon_months: int = Field(..., ge=3, le=24)
    total_capital: float = Field(..., gt=0)
    market_regime: str = Field(..., description="bull/neutral/bear")
    generated_at: str = Field(..., description="ISO timestamp")
    analysis_date: str = Field(..., description="YYYY-MM-DD")

    # Tier composition
    tier_allocation: TargetTierAllocation

    # Positions
    positions: List[TargetPosition] = Field(
        ..., min_length=MIN_POSITIONS, max_length=MAX_POSITIONS,
    )
    cash_reserve_pct: float = Field(..., ge=0, le=30)

    # Sector breakdown
    sector_weights: List[SectorWeight]

    # Probability assessment (Monte Carlo)
    probability_assessment: ProbabilityAssessment

    # Scenario analysis
    scenarios: ScenarioAnalysis

    # Alternative portfolios
    alternatives: List[AlternativePortfolio] = Field(
        ..., min_length=2, max_length=3,
    )

    # Transition plan (optional — only if current holdings available)
    transition: Optional[TransitionPlan] = None

    # Risk disclosure (mandatory)
    risk_disclosure: RiskDisclosure

    # Rationale
    construction_rationale: str = Field(
        ..., min_length=100,
        description="Narrative explaining WHY this specific construction",
    )

    # LLM enrichment tracking
    narrative_source: Optional[Literal["llm", "template"]] = None

    # Summary
    summary: str = Field(..., min_length=50)

    @model_validator(mode="after")
    def allocations_sum_to_100(self) -> "TargetReturnOutput":
        pos_sum = sum(p.allocation_pct for p in self.positions)
        total = pos_sum + self.cash_reserve_pct
        if not (97.0 <= total <= 103.0):
            raise ValueError(
                f"Position allocations ({pos_sum:.1f}%) + cash ({self.cash_reserve_pct:.1f}%) "
                f"must sum to ~100%, got {total:.1f}%"
            )
        return self

    @model_validator(mode="after")
    def return_contributions_near_target(self) -> "TargetReturnOutput":
        contrib_sum = sum(p.expected_return_contribution_pct for p in self.positions)
        # Allow ±5% tolerance from target
        if abs(contrib_sum - self.target_return_pct) > 5.0:
            raise ValueError(
                f"Sum of expected_return_contribution_pct ({contrib_sum:.1f}%) "
                f"must be within ±5% of target ({self.target_return_pct:.1f}%)"
            )
        return self

    @model_validator(mode="after")
    def no_position_exceeds_max(self) -> "TargetReturnOutput":
        for p in self.positions:
            if p.allocation_pct > MAX_SINGLE_POSITION_PCT + 0.1:
                raise ValueError(
                    f"Position {p.ticker} allocation ({p.allocation_pct:.1f}%) "
                    f"exceeds max {MAX_SINGLE_POSITION_PCT}%"
                )
        return self

    @model_validator(mode="after")
    def no_sector_exceeds_max(self) -> "TargetReturnOutput":
        for sw in self.sector_weights:
            if sw.weight_pct > MAX_SECTOR_CONCENTRATION_PCT + 0.1:
                raise ValueError(
                    f"Sector {sw.sector} weight ({sw.weight_pct:.1f}%) "
                    f"exceeds max {MAX_SECTOR_CONCENTRATION_PCT}%"
                )
        return self
