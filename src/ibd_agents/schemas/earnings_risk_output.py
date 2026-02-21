"""
Agent 17: Earnings Risk Analyst — Output Schema
IBD Momentum Investment Framework v4.0

Output contract for the Earnings Risk Analyst.
Analyzes portfolio positions approaching earnings,
produces cushion-ratio-based risk classifications,
strategy options with scenario tables, and
portfolio-level concentration assessment.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EarningsRisk(str, Enum):
    """Position-level earnings risk classification."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CushionCategory(str, Enum):
    """Cushion ratio classification (MUST-M3)."""
    COMFORTABLE = "COMFORTABLE"       # Ratio > 2.0
    MANAGEABLE = "MANAGEABLE"         # Ratio 1.0-2.0
    THIN = "THIN"                     # Ratio 0.5-1.0
    INSUFFICIENT = "INSUFFICIENT"     # Ratio < 0.5


class EstimateRevision(str, Enum):
    """Analyst estimate revision direction (MUST-M5)."""
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


class ImpliedVolSignal(str, Enum):
    """Options implied volatility signal (MUST-M6)."""
    NORMAL = "NORMAL"
    ELEVATED_EXPECTATIONS = "ELEVATED_EXPECTATIONS"


class StrategyType(str, Enum):
    """Pre-earnings strategy recommendation."""
    HOLD_FULL = "HOLD_FULL"
    HOLD_AND_ADD = "HOLD_AND_ADD"
    TRIM_TO_HALF = "TRIM_TO_HALF"
    TRIM_TO_QUARTER = "TRIM_TO_QUARTER"
    EXIT_BEFORE_EARNINGS = "EXIT_BEFORE_EARNINGS"
    HEDGE_WITH_PUT = "HEDGE_WITH_PUT"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Lookforward window (MUST-M1)
LOOKFORWARD_DAYS: int = 21

# Cushion ratio thresholds (MUST-M3)
CUSHION_COMFORTABLE: float = 2.0
CUSHION_MANAGEABLE_LOW: float = 1.0
CUSHION_THIN_LOW: float = 0.5

# Implied vol threshold (MUST-M6)
IV_ELEVATED_RATIO: float = 1.5

# Concentration thresholds (MUST-M7)
CONCENTRATION_MODERATE_PCT: float = 30.0
CONCENTRATION_CRITICAL_PCT: float = 50.0

# Position size amplification threshold (MUST_NOT-S3)
LARGE_POSITION_PCT: float = 10.0

# MUST_NOT-S2 hold-and-add conditions
ADD_MIN_BEAT_RATE: float = 87.5     # 7/8 = 87.5%
ADD_MIN_CUSHION_RATIO: float = 2.0
ADD_MAX_POSITION_PCT: float = 5.0

# Minimum quarters for confident analysis (MUST-Q4)
MIN_CONFIDENT_QUARTERS: int = 4

# Strategy aggressiveness ladder (EXIT is most conservative, ADD is most aggressive)
STRATEGY_LADDER: list[str] = [
    "EXIT_BEFORE_EARNINGS",
    "TRIM_TO_QUARTER",
    "TRIM_TO_HALF",
    "HEDGE_WITH_PUT",
    "HOLD_FULL",
    "HOLD_AND_ADD",
]

# Risk level ordering
RISK_ORDER: dict[str, int] = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MODERATE": 2,
    "LOW": 3,
}

# Weak regimes where MUST_NOT-S1 applies
WEAK_REGIMES: set[str] = {
    "UPTREND_UNDER_PRESSURE",
    "CORRECTION",
    "RALLY_ATTEMPT",
}

# Prediction phrases forbidden by MUST_NOT-S6
FORBIDDEN_PREDICTIONS: list[str] = [
    "will beat",
    "likely to beat",
    "expect a beat",
    "will miss",
    "likely to miss",
    "should report strong",
    "going to beat",
    "going to miss",
]


# ---------------------------------------------------------------------------
# Supporting Models
# ---------------------------------------------------------------------------

class ScenarioOutcome(BaseModel):
    """Outcome of one scenario (BEST/BASE/WORST) for a strategy."""

    scenario: str = Field(
        ..., description="'BEST', 'BASE', or 'WORST'",
    )
    expected_move_pct: float = Field(
        ..., description="Expected stock price move %",
    )
    resulting_gain_loss_pct: float = Field(
        ..., description="Position gain/loss after move",
    )
    dollar_impact: float = Field(
        ..., description="Dollar impact on portfolio",
    )
    math: str = Field(
        ..., min_length=5,
        description="Arithmetic: 'shares x $move = $impact'",
    )

    @field_validator("scenario")
    @classmethod
    def validate_scenario_name(cls, v: str) -> str:
        if v not in ("BEST", "BASE", "WORST"):
            raise ValueError(f"scenario must be BEST/BASE/WORST, got '{v}'")
        return v


class StrategyOption(BaseModel):
    """One strategy option with scenario outcomes."""

    strategy: StrategyType = Field(...)
    description: str = Field(
        ..., min_length=10,
        description="1-2 sentence description of what to do",
    )
    shares_to_sell: Optional[int] = Field(
        default=None,
        description="Shares to sell before earnings. None for HOLD strategies.",
    )
    estimated_hedge_cost: Optional[float] = Field(
        default=None,
        description="Cost of protective put if HEDGE strategy. None otherwise.",
    )
    scenarios: list[ScenarioOutcome] = Field(
        ..., min_length=3, max_length=3,
        description="BEST, BASE, and WORST case outcomes",
    )
    risk_reward_summary: str = Field(
        ..., min_length=10,
        description="'Risk $X to make $Y' or 'Max loss $X, max gain $Y'",
    )

    @model_validator(mode="after")
    def validate_scenario_names(self) -> "StrategyOption":
        """Ensure all 3 scenario names are present."""
        names = {s.scenario for s in self.scenarios}
        expected = {"BEST", "BASE", "WORST"}
        if names != expected:
            raise ValueError(
                f"Scenarios must include BEST/BASE/WORST, got {names}"
            )
        return self


class HistoricalEarnings(BaseModel):
    """Historical post-earnings reaction data (MUST-M2)."""

    quarters_analyzed: int = Field(
        ..., ge=0,
        description="Number of quarters with data (target 8)",
    )
    beat_count: int = Field(..., ge=0)
    miss_count: int = Field(..., ge=0)
    beat_rate_pct: float = Field(
        ..., ge=0, le=100,
        description="Beat percentage",
    )
    avg_move_pct: float = Field(
        ..., ge=0,
        description="Average absolute post-earnings move %",
    )
    avg_gap_up_pct: float = Field(
        ..., ge=0,
        description="Average move on beats",
    )
    avg_gap_down_pct: float = Field(
        ..., le=0,
        description="Average move on misses (negative)",
    )
    max_adverse_move_pct: float = Field(
        ..., le=0,
        description="Worst single-quarter move (negative)",
    )
    move_std_dev: float = Field(
        ..., ge=0,
        description="Standard deviation of moves",
    )
    recent_trend: str = Field(
        ..., description="'improving', 'declining', or 'consistent'",
    )

    @field_validator("recent_trend")
    @classmethod
    def validate_trend(cls, v: str) -> str:
        valid = ("improving", "declining", "consistent")
        if v not in valid:
            raise ValueError(f"recent_trend must be one of {valid}, got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_counts(self) -> "HistoricalEarnings":
        """Beat + miss counts must equal quarters analyzed."""
        if self.beat_count + self.miss_count != self.quarters_analyzed:
            raise ValueError(
                f"beat_count ({self.beat_count}) + miss_count ({self.miss_count}) "
                f"must equal quarters_analyzed ({self.quarters_analyzed})"
            )
        return self


class PositionEarningsAnalysis(BaseModel):
    """Complete earnings analysis for one position (MUST-M1 through MUST-M8)."""

    ticker: str = Field(..., min_length=1, max_length=10)
    account: str = Field(..., min_length=1)
    earnings_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="YYYY-MM-DD",
    )
    days_until_earnings: int = Field(..., ge=0)
    reporting_time: str = Field(
        ...,
        description="'BEFORE_OPEN', 'AFTER_CLOSE', or 'UNKNOWN'",
    )

    # Current position context
    shares: int = Field(..., ge=1)
    current_price: float = Field(..., gt=0)
    buy_price: float = Field(..., gt=0)
    gain_loss_pct: float = Field(...)
    position_value: float = Field(..., gt=0)
    portfolio_pct: float = Field(..., gt=0)

    # Historical analysis
    historical: HistoricalEarnings = Field(...)

    # Cushion analysis
    cushion_ratio: float = Field(
        ..., description="gain_loss_pct / avg_move_pct",
    )
    cushion_category: CushionCategory = Field(...)

    # Forward signals
    estimate_revision: EstimateRevision = Field(...)
    estimate_revision_detail: str = Field(
        ..., min_length=5,
        description="Specific detail about estimate changes",
    )
    implied_vol_signal: ImpliedVolSignal = Field(...)
    implied_move_pct: Optional[float] = Field(
        default=None,
        description="Options-implied expected move %. None if unavailable.",
    )

    # Risk classification
    risk_level: EarningsRisk = Field(...)
    risk_factors: list[str] = Field(
        ..., min_length=1,
        description="Specific risk factors explaining classification",
    )

    # Strategies
    strategies: list[StrategyOption] = Field(
        ..., min_length=2,
        description="Minimum 2 strategy options with scenario tables",
    )
    recommended_strategy: StrategyType = Field(...)
    recommendation_rationale: str = Field(
        ..., min_length=20,
        description="2-3 sentences explaining the recommendation",
    )

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("reporting_time")
    @classmethod
    def validate_reporting_time(cls, v: str) -> str:
        valid = ("BEFORE_OPEN", "AFTER_CLOSE", "UNKNOWN")
        if v not in valid:
            raise ValueError(f"reporting_time must be one of {valid}, got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_recommended_in_strategies(self) -> "PositionEarningsAnalysis":
        """Recommended strategy must be one of the presented strategies."""
        strategy_types = {s.strategy for s in self.strategies}
        if self.recommended_strategy not in strategy_types:
            raise ValueError(
                f"recommended_strategy {self.recommended_strategy.value} "
                f"not in presented strategies {strategy_types}"
            )
        return self


class EarningsWeek(BaseModel):
    """One week in the earnings calendar (MUST-Q5)."""

    week_start: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Monday of the week, YYYY-MM-DD",
    )
    week_label: str = Field(
        ..., description="'Week of Feb 24'",
    )
    positions_reporting: list[str] = Field(
        ..., min_length=1,
        description="Tickers reporting this week",
    )
    aggregate_portfolio_pct: float = Field(
        ..., ge=0,
    )
    concentration_flag: Optional[str] = Field(
        default=None,
        description="'CONCENTRATED' if >30%, 'CRITICAL' if >50%. None otherwise.",
    )

    @field_validator("concentration_flag")
    @classmethod
    def validate_flag(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("CONCENTRATED", "CRITICAL"):
            raise ValueError(
                f"concentration_flag must be CONCENTRATED/CRITICAL/None, got '{v}'"
            )
        return v


class PortfolioEarningsConcentration(BaseModel):
    """Portfolio-level earnings concentration assessment (MUST-M7)."""

    total_positions_approaching: int = Field(..., ge=0)
    total_portfolio_pct_exposed: float = Field(..., ge=0)
    earnings_calendar: list[EarningsWeek] = Field(
        ..., description="Weeks with earnings, ordered chronologically",
    )
    concentration_risk: str = Field(
        ..., description="'LOW', 'MODERATE', 'HIGH', or 'CRITICAL'",
    )
    concentration_recommendation: Optional[str] = Field(
        default=None,
        description="Portfolio-level recommendation if concentration is HIGH+",
    )

    @field_validator("concentration_risk")
    @classmethod
    def validate_risk(cls, v: str) -> str:
        valid = ("LOW", "MODERATE", "HIGH", "CRITICAL")
        if v not in valid:
            raise ValueError(
                f"concentration_risk must be one of {valid}, got '{v}'"
            )
        return v


# ---------------------------------------------------------------------------
# Top-level Output
# ---------------------------------------------------------------------------

class EarningsRiskOutput(BaseModel):
    """Top-level output contract for the Earnings Risk Analyst."""

    analysis_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="YYYY-MM-DD",
    )
    market_regime: str = Field(
        ..., description="Market regime from Regime Detector",
    )
    lookforward_days: int = Field(
        ..., ge=1,
        description="Lookforward window in calendar days",
    )

    analyses: list[PositionEarningsAnalysis] = Field(
        default_factory=list,
        description="One analysis per position with upcoming earnings, ordered by date",
    )
    positions_clear: list[str] = Field(
        default_factory=list,
        description="Tickers with no earnings in the lookforward window",
    )
    concentration: PortfolioEarningsConcentration = Field(...)

    executive_summary: str = Field(
        ..., min_length=50,
        description="3-4 sentence summary of earnings risk exposure",
    )
    data_source: Optional[str] = Field(
        None, description="'real' or 'mock'",
    )

    @model_validator(mode="after")
    def validate_analyses_ordered_by_date(self) -> "EarningsRiskOutput":
        """Analyses must be ordered by earnings_date ascending."""
        if len(self.analyses) > 1:
            dates = [a.earnings_date for a in self.analyses]
            if dates != sorted(dates):
                raise ValueError(
                    "Analyses must be ordered by earnings_date ascending"
                )
        return self

    @model_validator(mode="after")
    def validate_concentration_count(self) -> "EarningsRiskOutput":
        """Concentration count must match analyses."""
        if self.concentration.total_positions_approaching != len(self.analyses):
            raise ValueError(
                f"concentration.total_positions_approaching "
                f"({self.concentration.total_positions_approaching}) "
                f"must equal len(analyses) ({len(self.analyses)})"
            )
        return self

    @model_validator(mode="after")
    def validate_must_not_s1(self) -> "EarningsRiskOutput":
        """MUST_NOT-S1: No HOLD_FULL/HOLD_AND_ADD with thin cushion + weak market."""
        if self.market_regime in WEAK_REGIMES:
            for analysis in self.analyses:
                if analysis.cushion_ratio < CUSHION_THIN_LOW:
                    forbidden = {StrategyType.HOLD_FULL, StrategyType.HOLD_AND_ADD}
                    for strat in analysis.strategies:
                        if strat.strategy in forbidden:
                            raise ValueError(
                                f"MUST_NOT-S1: {analysis.ticker} has cushion ratio "
                                f"{analysis.cushion_ratio:.2f} (<{CUSHION_THIN_LOW}) "
                                f"in {self.market_regime} regime — "
                                f"{strat.strategy.value} is forbidden"
                            )
        return self

    @model_validator(mode="after")
    def validate_must_not_s2(self) -> "EarningsRiskOutput":
        """MUST_NOT-S2: HOLD_AND_ADD only when all 4 conditions met."""
        for analysis in self.analyses:
            has_add = any(
                s.strategy == StrategyType.HOLD_AND_ADD
                for s in analysis.strategies
            )
            if not has_add:
                continue
            # Check all 4 conditions
            conditions_met = (
                analysis.historical.beat_rate_pct >= ADD_MIN_BEAT_RATE
                and analysis.cushion_ratio >= ADD_MIN_CUSHION_RATIO
                and self.market_regime == "CONFIRMED_UPTREND"
                and analysis.portfolio_pct < ADD_MAX_POSITION_PCT
            )
            if not conditions_met:
                raise ValueError(
                    f"MUST_NOT-S2: {analysis.ticker} has HOLD_AND_ADD but "
                    f"conditions not met — beat_rate={analysis.historical.beat_rate_pct}% "
                    f"(need >={ADD_MIN_BEAT_RATE}%), cushion={analysis.cushion_ratio:.1f} "
                    f"(need >={ADD_MIN_CUSHION_RATIO}), regime={self.market_regime} "
                    f"(need CONFIRMED_UPTREND), portfolio_pct={analysis.portfolio_pct}% "
                    f"(need <{ADD_MAX_POSITION_PCT}%)"
                )
        return self

    @model_validator(mode="after")
    def validate_no_predictions(self) -> "EarningsRiskOutput":
        """MUST_NOT-S6: No beat/miss predictions in rationales."""
        for analysis in self.analyses:
            text = analysis.recommendation_rationale.lower()
            for phrase in FORBIDDEN_PREDICTIONS:
                if phrase in text:
                    raise ValueError(
                        f"MUST_NOT-S6: {analysis.ticker} rationale contains "
                        f"prediction language: '{phrase}'"
                    )
        return self

    @field_validator("data_source")
    @classmethod
    def validate_data_source(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("real", "mock"):
            raise ValueError(
                f"data_source must be 'real' or 'mock', got '{v}'"
            )
        return v
