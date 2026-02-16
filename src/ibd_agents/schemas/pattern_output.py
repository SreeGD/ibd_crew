"""
Agent 12: PatternAlpha — Per-Stock Pattern Scoring Engine
IBD Momentum Investment Framework v4.0

Evaluates every security against 5 wealth-creation patterns
(Platform Economics, Self-Cannibalization, Capital Allocation,
Category Creation, Inflection Timing) to produce a 150-point
Enhanced Score per stock.

Never makes buy/sell recommendations — only discovers, scores, and organizes.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATTERN_NAMES = (
    "Platform Economics",
    "Self-Cannibalization",
    "Capital Allocation",
    "Category Creation",
    "Inflection Timing",
)

PATTERN_MAX_SCORES: dict[str, int] = {
    "Platform Economics": 12,
    "Self-Cannibalization": 10,
    "Capital Allocation": 10,
    "Category Creation": 10,
    "Inflection Timing": 8,
}

# Enhanced rating bands (score_min, score_max, stars, label)
ENHANCED_RATING_BANDS: list[tuple[int, int, str, str]] = [
    (130, 150, "★★★★★+", "Max Conviction"),
    (115, 129, "★★★★★",  "Strong"),
    (100, 114, "★★★★",   "Favorable"),
    (85,  99,  "★★★",    "Notable"),
    (70,  84,  "★★",     "Monitor"),
    (0,   69,  "★",      "Review"),
]

ENHANCED_RATING_LABELS = (
    "Max Conviction", "Strong", "Favorable", "Notable", "Monitor", "Review",
)

PATTERN_ALERT_TYPES = (
    "Category King",
    "Inflection Alert",
    "Disruption Risk",
    "Pattern Imbalance",
)

# Tier pattern requirements from spec §2
TIER_PATTERN_REQUIREMENTS: dict[int, dict] = {
    1: {"min_enhanced": 115, "min_patterns_at_7": 2},
    2: {"min_enhanced": 95,  "min_patterns_at_5": 2},
    3: {"min_enhanced": 80,  "preferred_p3": 7},
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PatternSubScore(BaseModel):
    """Score for a single wealth-creation pattern."""

    pattern_name: str = Field(..., min_length=1)
    score: int = Field(..., ge=0)
    max_score: int = Field(..., ge=1)
    justification: str = Field(default="")

    @model_validator(mode="after")
    def validate_score_within_max(self) -> "PatternSubScore":
        if self.score > self.max_score:
            raise ValueError(
                f"score {self.score} exceeds max_score {self.max_score} "
                f"for pattern '{self.pattern_name}'"
            )
        return self


class PatternScoreBreakdown(BaseModel):
    """Breakdown of all 5 pattern scores for a stock."""

    p1_platform: PatternSubScore
    p2_cannibalization: PatternSubScore
    p3_capital_allocation: PatternSubScore
    p4_category_creation: PatternSubScore
    p5_inflection_timing: PatternSubScore
    pattern_total: int = Field(..., ge=0, le=50)
    dominant_pattern: str = Field(..., min_length=1)
    pattern_narrative: str = Field(
        ..., min_length=20,
        description="2-3 sentence thesis from pattern perspective",
    )

    @model_validator(mode="after")
    def validate_pattern_total(self) -> "PatternScoreBreakdown":
        """pattern_total must equal sum of 5 pattern sub-scores."""
        computed = (
            self.p1_platform.score
            + self.p2_cannibalization.score
            + self.p3_capital_allocation.score
            + self.p4_category_creation.score
            + self.p5_inflection_timing.score
        )
        if self.pattern_total != computed:
            raise ValueError(
                f"pattern_total={self.pattern_total} but sum of "
                f"sub-scores={computed}"
            )
        return self

    @model_validator(mode="after")
    def validate_max_scores(self) -> "PatternScoreBreakdown":
        """Verify each sub-score has the correct max_score per PATTERN_MAX_SCORES."""
        expected = [
            (self.p1_platform, "Platform Economics", 12),
            (self.p2_cannibalization, "Self-Cannibalization", 10),
            (self.p3_capital_allocation, "Capital Allocation", 10),
            (self.p4_category_creation, "Category Creation", 10),
            (self.p5_inflection_timing, "Inflection Timing", 8),
        ]
        for sub, name, max_s in expected:
            if sub.max_score != max_s:
                raise ValueError(
                    f"{name} max_score should be {max_s}, got {sub.max_score}"
                )
        return self


class BaseScoreBreakdown(BaseModel):
    """Breakdown of the 100-point base score from existing metrics."""

    ibd_score: float = Field(..., ge=0, le=40)
    analyst_score: float = Field(..., ge=0, le=30)
    risk_score: float = Field(..., ge=0, le=30)
    base_total: int = Field(..., ge=0, le=100)

    @model_validator(mode="after")
    def validate_base_total(self) -> "BaseScoreBreakdown":
        """base_total must equal round(ibd + analyst + risk)."""
        computed = round(self.ibd_score + self.analyst_score + self.risk_score)
        if self.base_total != computed:
            raise ValueError(
                f"base_total={self.base_total} but round(ibd+analyst+risk)={computed}"
            )
        return self


class EnhancedStockAnalysis(BaseModel):
    """Per-stock enhanced analysis with base + pattern scores."""

    symbol: str = Field(..., min_length=1, max_length=10)
    company_name: str = Field(..., min_length=1)
    sector: str = Field(..., min_length=1)
    tier: int = Field(..., ge=1, le=3)
    base_score: BaseScoreBreakdown
    pattern_score: Optional[PatternScoreBreakdown] = None
    enhanced_score: int = Field(..., ge=0, le=150)
    enhanced_rating: str = Field(..., min_length=1)
    enhanced_rating_label: str
    pattern_alerts: List[str] = Field(default_factory=list)
    tier_recommendation: str = Field(
        default="",
        description="Tier fit assessment based on pattern requirements",
    )
    scoring_source: str = Field(
        ..., description="'llm' or 'deterministic'"
    )

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("scoring_source")
    @classmethod
    def validate_scoring_source(cls, v: str) -> str:
        if v not in ("llm", "deterministic"):
            raise ValueError(f"scoring_source must be 'llm' or 'deterministic', got '{v}'")
        return v

    @field_validator("enhanced_rating_label")
    @classmethod
    def validate_rating_label(cls, v: str) -> str:
        if v not in ENHANCED_RATING_LABELS:
            raise ValueError(f"enhanced_rating_label '{v}' not in {ENHANCED_RATING_LABELS}")
        return v

    @model_validator(mode="after")
    def validate_enhanced_score(self) -> "EnhancedStockAnalysis":
        """enhanced_score = base_total + pattern_total (or base_total if no pattern)."""
        base = self.base_score.base_total
        pattern = self.pattern_score.pattern_total if self.pattern_score else 0
        expected = base + pattern
        if self.enhanced_score != expected:
            raise ValueError(
                f"enhanced_score={self.enhanced_score} but "
                f"base({base}) + pattern({pattern}) = {expected}"
            )
        return self


class PatternAlert(BaseModel):
    """Spec-defined pattern alert (Category King, Inflection, etc.)."""

    alert_type: str
    symbol: str = Field(..., min_length=1, max_length=10)
    description: str = Field(..., min_length=20)
    pattern_name: str = Field(..., min_length=1)
    pattern_score: int = Field(..., ge=0)

    @field_validator("alert_type")
    @classmethod
    def validate_alert_type(cls, v: str) -> str:
        if v not in PATTERN_ALERT_TYPES:
            raise ValueError(f"alert_type '{v}' not in {PATTERN_ALERT_TYPES}")
        return v

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()


class PortfolioPatternSummary(BaseModel):
    """Aggregate summary of pattern scoring across all stocks."""

    total_stocks_scored: int = Field(..., ge=0)
    stocks_with_patterns: int = Field(..., ge=0)
    avg_enhanced_score: float = Field(..., ge=0)
    tier_1_candidates: int = Field(
        ..., ge=0, description="Count meeting T1 pattern requirements"
    )
    category_kings: int = Field(
        ..., ge=0, description="Count with P4 Category Creation >= 8"
    )
    inflection_alerts: int = Field(
        ..., ge=0, description="Count with P5 Inflection Timing >= 6"
    )
    disruption_risks: int = Field(
        ..., ge=0, description="Count with P2 Self-Cannibalization = 0"
    )


# ---------------------------------------------------------------------------
# Top-level Output
# ---------------------------------------------------------------------------


class PortfolioPatternOutput(BaseModel):
    """Top-level output for the PatternAlpha per-stock scoring engine."""

    stock_analyses: List[EnhancedStockAnalysis] = Field(
        ..., min_length=1,
        description="Per-stock enhanced analyses with base + pattern scores",
    )
    pattern_alerts: List[PatternAlert] = Field(
        default_factory=list,
        description="Pattern-specific alerts (Category Kings, Inflection, etc.)",
    )
    portfolio_summary: PortfolioPatternSummary
    scoring_source: str = Field(
        ..., description="'llm' or 'deterministic'"
    )
    methodology_notes: str = Field(..., min_length=1)
    analysis_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD"
    )
    summary: str = Field(..., min_length=50)

    @field_validator("scoring_source")
    @classmethod
    def validate_scoring_source(cls, v: str) -> str:
        if v not in ("llm", "deterministic"):
            raise ValueError(f"scoring_source must be 'llm' or 'deterministic', got '{v}'")
        return v
