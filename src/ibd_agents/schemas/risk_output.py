"""
Agent 06: Risk Officer — Output Schema
IBD Momentum Investment Framework v4.0

Output contract for the Risk Officer.
Reviews portfolio against framework limits, runs stress tests,
assigns Sleep Well scores, and issues vetoes for violations.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECK_NAMES: list[str] = [
    "position_sizing",
    "trailing_stops",
    "sector_concentration",
    "tier_allocation",
    "max_loss",
    "correlation",
    "regime_alignment",
    "volume_confirmation",
    "keeps_validation",
    "stress_test",
]

VALID_CHECK_STATUS = ("PASS", "WARNING", "VETO")
VALID_SEVERITY = ("LOW", "MEDIUM", "HIGH")
VALID_OVERALL_STATUS = ("APPROVED", "CONDITIONAL", "REJECTED")

SLEEP_WELL_RANGE: tuple[int, int] = (1, 10)

STRESS_SCENARIOS: list[dict] = [
    {
        "name": "Market Crash",
        "description": "Broad market decline of 20%",
        "market_impact_pct": -20.0,
    },
    {
        "name": "Sector Correction",
        "description": "Leading sector corrects 30%",
        "market_impact_pct": -30.0,
    },
    {
        "name": "Rate Hike",
        "description": "Unexpected 50bp rate increase",
        "market_impact_pct": -8.0,
    },
]


# ---------------------------------------------------------------------------
# Supporting Models
# ---------------------------------------------------------------------------

class RiskCheck(BaseModel):
    """Result of a single risk check."""

    check_name: str = Field(...)
    status: str = Field(...)
    findings: str = Field(..., min_length=20)
    details: List[str] = Field(default_factory=list)

    @field_validator("check_name")
    @classmethod
    def validate_check_name(cls, v: str) -> str:
        if v not in CHECK_NAMES:
            raise ValueError(f"check_name '{v}' not in {CHECK_NAMES}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in VALID_CHECK_STATUS:
            raise ValueError(f"status must be one of {VALID_CHECK_STATUS}, got '{v}'")
        return v


class Veto(BaseModel):
    """A hard veto — PM must fix before proceeding."""

    check_name: str = Field(...)
    reason: str = Field(..., min_length=10)
    required_fix: str = Field(..., min_length=50)

    @field_validator("check_name")
    @classmethod
    def validate_check_name(cls, v: str) -> str:
        if v not in CHECK_NAMES:
            raise ValueError(f"check_name '{v}' not in {CHECK_NAMES}")
        return v


class RiskWarning(BaseModel):
    """A non-blocking warning — proceed with caution."""

    check_name: str = Field(...)
    severity: str = Field(...)
    description: str = Field(..., min_length=10)
    suggestion: str = Field(..., min_length=10)

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        if v not in VALID_SEVERITY:
            raise ValueError(f"severity must be one of {VALID_SEVERITY}, got '{v}'")
        return v


class StressScenario(BaseModel):
    """One stress test scenario result."""

    scenario_name: str = Field(..., min_length=1)
    impact_description: str = Field(..., min_length=20)
    estimated_drawdown_pct: float = Field(...)
    positions_most_affected: List[str] = Field(default_factory=list)


class StressTestReport(BaseModel):
    """Aggregate stress test results."""

    scenarios: List[StressScenario] = Field(..., min_length=3)
    overall_resilience: str = Field(..., min_length=10)


class SleepWellScores(BaseModel):
    """Sleep Well score per tier + overall (1-10)."""

    tier_1_score: int = Field(..., ge=1, le=10)
    tier_2_score: int = Field(..., ge=1, le=10)
    tier_3_score: int = Field(..., ge=1, le=10)
    overall_score: int = Field(..., ge=1, le=10)
    factors: List[str] = Field(default_factory=list, description="Factors affecting scores")


class KeepValidation(BaseModel):
    """Validation of 21 keep positions."""

    total_keeps_found: int = Field(...)
    missing_keeps: List[str] = Field(default_factory=list)
    status: str = Field(...)


class StopLossRecommendation(BaseModel):
    """LLM-recommended stop-loss adjustment for a position."""

    symbol: str = Field(..., min_length=1, max_length=10)
    current_stop_pct: float = Field(..., ge=0)
    recommended_stop_pct: float = Field(..., ge=5.0, le=35.0)
    reason: str = Field(..., min_length=10)
    volatility_flag: Optional[str] = Field(None, description="high/normal/low")

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("volatility_flag")
    @classmethod
    def validate_volatility_flag(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("high", "normal", "low"):
            raise ValueError(f"volatility_flag must be high/normal/low, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Top-level Output
# ---------------------------------------------------------------------------

class RiskAssessment(BaseModel):
    """Top-level output contract for the Risk Officer."""

    check_results: List[RiskCheck] = Field(
        ..., min_length=10, max_length=10,
        description="All 10 risk checks"
    )
    vetoes: List[Veto] = Field(default_factory=list)
    warnings: List[RiskWarning] = Field(default_factory=list)
    stress_test_results: StressTestReport = Field(...)
    sleep_well_scores: SleepWellScores = Field(...)
    keep_validation: KeepValidation = Field(...)
    overall_status: str = Field(...)
    analysis_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD"
    )
    summary: str = Field(..., min_length=50)
    stop_loss_recommendations: List[StopLossRecommendation] = Field(default_factory=list)
    stop_loss_source: Optional[str] = Field(None, description="'llm' or 'deterministic'")

    @field_validator("overall_status")
    @classmethod
    def validate_overall_status(cls, v: str) -> str:
        if v not in VALID_OVERALL_STATUS:
            raise ValueError(
                f"overall_status must be one of {VALID_OVERALL_STATUS}, got '{v}'"
            )
        return v

    @field_validator("stop_loss_source")
    @classmethod
    def validate_stop_loss_source(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("llm", "deterministic"):
            raise ValueError(f"stop_loss_source must be 'llm' or 'deterministic', got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_check_names_complete(self) -> "RiskAssessment":
        """All 10 check names present."""
        names = {c.check_name for c in self.check_results}
        expected = set(CHECK_NAMES)
        missing = expected - names
        if missing:
            raise ValueError(f"Missing risk checks: {missing}")
        return self

    @model_validator(mode="after")
    def validate_status_consistency(self) -> "RiskAssessment":
        """Overall status matches vetoes/warnings."""
        has_vetoes = len(self.vetoes) > 0
        has_warnings = len(self.warnings) > 0

        if has_vetoes and self.overall_status != "REJECTED":
            raise ValueError(
                f"Has {len(self.vetoes)} vetoes but status is "
                f"'{self.overall_status}', expected 'REJECTED'"
            )
        if not has_vetoes and has_warnings and self.overall_status == "REJECTED":
            raise ValueError(
                "No vetoes but status is REJECTED — should be CONDITIONAL"
            )
        if not has_vetoes and not has_warnings and self.overall_status != "APPROVED":
            raise ValueError(
                "No vetoes or warnings but status is not APPROVED"
            )
        return self
