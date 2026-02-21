"""
Agent 16: Regime Detector — Output Schema
IBD Momentum Investment Framework v4.0

Output contract for the Regime Detector.
Classifies market into 5 regimes using distribution days, breadth,
leading stock health, sector rotation, and follow-through day analysis.

Five regimes:
  CONFIRMED_UPTREND — Full offense (80-100% exposure)
  UPTREND_UNDER_PRESSURE — Reduce new buys (40-60%)
  FOLLOW_THROUGH_DAY — Transitional, test buys (20-40%)
  RALLY_ATTEMPT — No new buys (0-20%)
  CORRECTION — Full defense (0%)
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Distribution day thresholds (MUST-M2)
DIST_DAY_DECLINE_PCT: float = 0.2         # Index must decline >= 0.2%
DIST_DAY_WINDOW: int = 25                 # Trailing 25 trading sessions
DIST_DAY_UPTREND_CAP: int = 5             # 5+ = cannot be CONFIRMED_UPTREND (MUST_NOT-S3)
DIST_DAY_CORRECTION_THRESHOLD: int = 6    # 6+ = CORRECTION hard rule
POWER_UP_GAIN_PCT: float = 1.0            # 1%+ up on higher vol removes oldest dist day

# Follow-Through Day thresholds (MUST-M3)
FTD_MIN_GAIN_PCT: float = 1.25            # Minimum gain for valid FTD
FTD_STRONG_GAIN_PCT: float = 1.7          # Strong FTD gain threshold
FTD_STRONG_VOLUME_RATIO: float = 1.5      # Strong FTD volume requirement
FTD_MIN_RALLY_DAY: int = 4                # Day 4 minimum
FTD_STRONG_MAX_DAY: int = 7               # Strong: Day 4-7
FTD_MODERATE_MAX_DAY: int = 10            # Moderate: Day 4-10; Weak: after Day 10

# Leader health thresholds (MUST-M5)
LEADER_HEALTHY_PCT: float = 60.0          # > 60% = HEALTHY
LEADER_MIXED_LOW_PCT: float = 40.0        # 40-60% = MIXED
LEADER_DETERIORATING_PCT: float = 40.0    # < 40% = DETERIORATING

# Sector defensiveness (MUST-M6)
DEFENSIVE_SECTORS: set[str] = {
    "Utilities", "Consumer Staples", "Healthcare", "REITs", "Bonds/Treasuries",
}
DEFENSIVE_ROTATION_THRESHOLD: int = 3     # 3+ defensive in top 5

# Exposure ranges per regime (MUST-M7, MUST_NOT-S4)
EXPOSURE_RANGES: dict[str, tuple[int, int]] = {
    "CONFIRMED_UPTREND": (80, 100),
    "UPTREND_UNDER_PRESSURE": (40, 60),
    "FOLLOW_THROUGH_DAY": (20, 40),
    "RALLY_ATTEMPT": (0, 20),
    "CORRECTION": (0, 0),
}

# Health score ranges per regime (MUST-Q6)
HEALTH_SCORE_RANGES: dict[str, tuple[int, int]] = {
    "CONFIRMED_UPTREND": (7, 10),
    "UPTREND_UNDER_PRESSURE": (4, 6),
    "FOLLOW_THROUGH_DAY": (4, 6),
    "RALLY_ATTEMPT": (2, 4),
    "CORRECTION": (1, 3),
}

# Backward-compatibility: 5-state -> 3-state mapping
REGIME_TO_LEGACY: dict[str, str] = {
    "CONFIRMED_UPTREND": "bull",
    "UPTREND_UNDER_PRESSURE": "neutral",
    "FOLLOW_THROUGH_DAY": "neutral",
    "RALLY_ATTEMPT": "bear",
    "CORRECTION": "bear",
}

# Number of indicator groups (dist, breadth, leaders, sectors, ftd)
INDICATOR_COUNT: int = 5


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MarketRegime5(str, Enum):
    """5-state market regime classification."""
    CONFIRMED_UPTREND = "CONFIRMED_UPTREND"
    UPTREND_UNDER_PRESSURE = "UPTREND_UNDER_PRESSURE"
    CORRECTION = "CORRECTION"
    RALLY_ATTEMPT = "RALLY_ATTEMPT"
    FOLLOW_THROUGH_DAY = "FOLLOW_THROUGH_DAY"


class PreviousRegime(str, Enum):
    """Previous regime — includes UNKNOWN for first run."""
    CONFIRMED_UPTREND = "CONFIRMED_UPTREND"
    UPTREND_UNDER_PRESSURE = "UPTREND_UNDER_PRESSURE"
    CORRECTION = "CORRECTION"
    RALLY_ATTEMPT = "RALLY_ATTEMPT"
    FOLLOW_THROUGH_DAY = "FOLLOW_THROUGH_DAY"
    UNKNOWN = "UNKNOWN"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SignalDirection(str, Enum):
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"


class Trend(str, Enum):
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DETERIORATING = "DETERIORATING"


class LeaderHealth(str, Enum):
    HEALTHY = "HEALTHY"
    MIXED = "MIXED"
    DETERIORATING = "DETERIORATING"


class SectorCharacter(str, Enum):
    GROWTH_LEADING = "GROWTH_LEADING"
    BALANCED = "BALANCED"
    DEFENSIVE_ROTATION = "DEFENSIVE_ROTATION"


class FTDQuality(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NONE = "NONE"


# ---------------------------------------------------------------------------
# Assessment Models
# ---------------------------------------------------------------------------

class DistributionDayAssessment(BaseModel):
    """Distribution day count and trend for S&P 500 and Nasdaq."""

    sp500_count: int = Field(..., ge=0, description="Distribution days on S&P 500 in trailing 25 sessions")
    nasdaq_count: int = Field(..., ge=0, description="Distribution days on Nasdaq in trailing 25 sessions")
    sp500_direction: Trend = Field(..., description="Are distribution days accumulating or expiring?")
    nasdaq_direction: Trend = Field(..., description="Are distribution days accumulating or expiring?")
    signal: SignalDirection
    detail: str = Field(..., min_length=10, description="Brief explanation")


class BreadthAssessment(BaseModel):
    """Market breadth indicators."""

    pct_above_200ma: float = Field(..., ge=0.0, le=100.0, description="% of S&P 500 stocks above 200-day MA")
    pct_above_50ma: float = Field(..., ge=0.0, le=100.0, description="% of S&P 500 stocks above 50-day MA")
    new_highs: int = Field(..., ge=0, description="New 52-week highs (NYSE + Nasdaq)")
    new_lows: int = Field(..., ge=0, description="New 52-week lows (NYSE + Nasdaq)")
    advance_decline_direction: Trend
    breadth_divergence: bool = Field(
        ..., description="True if index at/near highs but breadth deteriorating"
    )
    signal: SignalDirection
    detail: str = Field(..., min_length=10)


class LeaderAssessment(BaseModel):
    """Leading stock health assessment."""

    rs90_above_50ma_pct: float = Field(..., ge=0.0, le=100.0, description="% of RS >= 90 stocks above their 50-day MA")
    rs90_new_highs: int = Field(..., ge=0, description="RS >= 90 stocks making new 52-week highs")
    rs90_new_lows: int = Field(..., ge=0, description="RS >= 90 stocks making new 52-week lows")
    health: LeaderHealth
    signal: SignalDirection
    detail: str = Field(..., min_length=10)


class SectorAssessment(BaseModel):
    """Sector rotation assessment."""

    top_5_sectors: List[str] = Field(..., min_length=1, max_length=5, description="Current top 5 sectors by ranking")
    defensive_in_top_5: int = Field(..., ge=0, le=5, description="Count of defensive sectors in top 5")
    character: SectorCharacter
    signal: SignalDirection
    detail: str = Field(..., min_length=10)


class FollowThroughDayAssessment(BaseModel):
    """Follow-through day detection result."""

    detected: bool = Field(..., description="Was a follow-through day detected?")
    quality: FTDQuality
    index: Optional[str] = Field(None, description="Which index: 'S&P 500' or 'Nasdaq'")
    gain_pct: Optional[float] = Field(None, description="Index gain on the FTD")
    volume_vs_prior: Optional[float] = Field(None, description="Volume ratio vs prior day")
    rally_day_number: Optional[int] = Field(None, ge=1, description="Which day of the rally attempt")
    detail: str = Field(..., min_length=10)

    @model_validator(mode="after")
    def validate_ftd_fields(self) -> "FollowThroughDayAssessment":
        """When detected=True, quality must not be NONE and index must be set."""
        if self.detected:
            if self.quality == FTDQuality.NONE:
                raise ValueError("FTD detected but quality is NONE")
            if self.index is None:
                raise ValueError("FTD detected but index is not specified")
        else:
            if self.quality != FTDQuality.NONE:
                raise ValueError(f"FTD not detected but quality is {self.quality}")
        return self


class TransitionCondition(BaseModel):
    """What would change the regime classification."""

    direction: Literal["UPGRADE", "DOWNGRADE"] = Field(..., description="'UPGRADE' or 'DOWNGRADE'")
    target_regime: MarketRegime5
    condition: str = Field(
        ..., min_length=20,
        description="Specific, measurable condition"
    )
    likelihood: Literal["LIKELY", "POSSIBLE", "UNLIKELY"] = Field(
        ..., description="Based on current trend"
    )


class RegimeChange(BaseModel):
    """Regime change documentation."""

    changed: bool = Field(..., description="Did the regime change from the previous run?")
    previous: Optional[PreviousRegime] = None
    current: Optional[MarketRegime5] = None
    trigger: Optional[str] = Field(
        None, description="What specific evidence triggered the change"
    )

    @model_validator(mode="after")
    def validate_change_fields(self) -> "RegimeChange":
        """When changed=True, previous, current, and trigger must be set."""
        if self.changed:
            if self.previous is None:
                raise ValueError("Regime changed but previous is not set")
            if self.current is None:
                raise ValueError("Regime changed but current is not set")
            if self.trigger is None or len(self.trigger) < 10:
                raise ValueError("Regime changed but trigger is missing or too short")
        return self


# ---------------------------------------------------------------------------
# Input Schema
# ---------------------------------------------------------------------------

class RegimeDetectorInput(BaseModel):
    """Input for the Regime Detector agent."""

    analysis_date: date = Field(..., description="Date of analysis (end-of-day data)")
    previous_regime: PreviousRegime = Field(
        default=PreviousRegime.UNKNOWN,
        description="Regime classification from the previous run",
    )
    previous_exposure_recommendation: Optional[int] = Field(
        None, ge=0, le=100,
        description="Exposure % recommended in the previous run",
    )
    correction_low_date: Optional[date] = Field(
        None, description="Date of the most recent correction low (for FTD counting)"
    )
    correction_low_sp500: Optional[float] = Field(
        None, description="S&P 500 closing price at the most recent correction low"
    )
    correction_low_nasdaq: Optional[float] = Field(
        None, description="Nasdaq closing price at the most recent correction low"
    )


# ---------------------------------------------------------------------------
# Top-level Output
# ---------------------------------------------------------------------------

class RegimeDetectorOutput(BaseModel):
    """Top-level output contract for the Regime Detector."""

    analysis_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD"
    )
    regime: MarketRegime5
    confidence: Confidence
    market_health_score: int = Field(
        ..., ge=1, le=10,
        description="1-10 market health. 10=strong uptrend, 1=severe correction",
    )
    exposure_recommendation: int = Field(
        ..., ge=0, le=100,
        description="Recommended portfolio exposure as percentage",
    )

    # Evidence — one assessment per indicator
    distribution_days: DistributionDayAssessment
    breadth: BreadthAssessment
    leaders: LeaderAssessment
    sectors: SectorAssessment
    follow_through_day: FollowThroughDayAssessment

    # Signal summary
    bullish_signals: int = Field(..., ge=0, le=INDICATOR_COUNT, description="Count of BULLISH indicators")
    neutral_signals: int = Field(..., ge=0, le=INDICATOR_COUNT, description="Count of NEUTRAL indicators")
    bearish_signals: int = Field(..., ge=0, le=INDICATOR_COUNT, description="Count of BEARISH indicators")

    # Transitions
    regime_change: RegimeChange
    transition_conditions: List[TransitionCondition] = Field(
        ..., min_length=1,
        description="What would cause the regime to change",
    )

    executive_summary: str = Field(
        ..., min_length=100,
        description="3-4 sentence summary for the Portfolio Manager",
    )

    reasoning_source: Literal["deterministic", "llm"] = Field(
        default="deterministic",
        description="How the classification was produced",
    )

    # -------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_signal_counts(self) -> "RegimeDetectorOutput":
        """bullish + neutral + bearish must equal INDICATOR_COUNT (5)."""
        total = self.bullish_signals + self.neutral_signals + self.bearish_signals
        if total != INDICATOR_COUNT:
            raise ValueError(
                f"Signal counts must sum to {INDICATOR_COUNT}, got {total} "
                f"(bullish={self.bullish_signals}, neutral={self.neutral_signals}, "
                f"bearish={self.bearish_signals})"
            )
        return self

    @model_validator(mode="after")
    def validate_exposure_range(self) -> "RegimeDetectorOutput":
        """Exposure must fall within EXPOSURE_RANGES for the regime (MUST_NOT-S4)."""
        lo, hi = EXPOSURE_RANGES[self.regime.value]
        if not (lo <= self.exposure_recommendation <= hi):
            raise ValueError(
                f"Exposure {self.exposure_recommendation}% outside range "
                f"[{lo}, {hi}] for regime {self.regime.value}"
            )
        return self

    @model_validator(mode="after")
    def validate_health_score_range(self) -> "RegimeDetectorOutput":
        """Health score must fall within HEALTH_SCORE_RANGES for the regime (MUST-Q6)."""
        lo, hi = HEALTH_SCORE_RANGES[self.regime.value]
        if not (lo <= self.market_health_score <= hi):
            raise ValueError(
                f"Health score {self.market_health_score} outside range "
                f"[{lo}, {hi}] for regime {self.regime.value}"
            )
        return self

    @model_validator(mode="after")
    def validate_ftd_regime_consistency(self) -> "RegimeDetectorOutput":
        """FOLLOW_THROUGH_DAY regime requires ftd.detected == True."""
        if self.regime == MarketRegime5.FOLLOW_THROUGH_DAY:
            if not self.follow_through_day.detected:
                raise ValueError(
                    "Regime is FOLLOW_THROUGH_DAY but follow_through_day.detected is False"
                )
        return self

    @model_validator(mode="after")
    def validate_dist_day_cap(self) -> "RegimeDetectorOutput":
        """5+ distribution days on either index cannot be CONFIRMED_UPTREND (MUST_NOT-S3)."""
        max_dist = max(
            self.distribution_days.sp500_count,
            self.distribution_days.nasdaq_count,
        )
        if max_dist >= DIST_DAY_UPTREND_CAP and self.regime == MarketRegime5.CONFIRMED_UPTREND:
            raise ValueError(
                f"Cannot be CONFIRMED_UPTREND with {max_dist} distribution days "
                f"(threshold: {DIST_DAY_UPTREND_CAP}). MUST_NOT-S3 violated."
            )
        return self

    # -------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------

    def to_legacy_regime(self) -> str:
        """Map 5-state regime to 3-state (bull/bear/neutral) for backward compatibility."""
        return REGIME_TO_LEGACY[self.regime.value]
