"""
Agent 03: Rotation Detector — Output Schema
IBD Momentum Investment Framework v4.0

Output contract for the Rotation Detector.
Detects sector rotation using a 5-signal framework,
classifies rotation type/stage, and assesses market regime.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ibd_agents.schemas.research_output import IBD_SECTORS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sector clusters for rotation analysis (spec §4.3)
SECTOR_CLUSTERS: dict[str, list[str]] = {
    "growth": ["CHIPS", "COMPUTER", "SOFTWARE", "INTERNET", "ELECTRONICS"],
    "commodity": ["MINING", "ENERGY", "CHEMICALS"],
    "defensive": ["HEALTHCARE", "MEDICAL", "UTILITIES", "FOOD/BEVERAGE", "CONSUMER"],
    "financial": ["BANKS", "FINANCE", "INSURANCE", "REAL ESTATE"],
    "industrial": ["AEROSPACE", "BUILDING", "MACHINERY", "TRANSPORTATION", "DEFENSE"],
    "consumer_cyclical": ["RETAIL", "LEISURE", "MEDIA", "AUTO"],
    "services": ["BUSINESS SERVICES", "TELECOM"],
}

# Reverse lookup: sector -> cluster name
SECTOR_TO_CLUSTER: dict[str, str] = {
    sector: cluster
    for cluster, sectors in SECTOR_CLUSTERS.items()
    for sector in sectors
}

VALID_CLUSTER_NAMES: set[str] = set(SECTOR_CLUSTERS.keys())

# Breadth thresholds (spec §4.2)
BREADTH_THRESHOLDS: dict[str, int] = {
    "bullish": 70,
    "neutral_high": 60,
    "neutral_low": 40,
    "bearish": 40,
}

# Signal thresholds (adapted for single-snapshot analysis)
RS_DIVERGENCE_THRESHOLD: float = 15.0
LEADERSHIP_MISALIGNMENT_MIN: int = 2
BREADTH_LOW_THRESHOLD: float = 40.0
BREADTH_HIGH_THRESHOLD: float = 60.0
ELITE_CLUSTER_HIGH: float = 50.0
ELITE_CLUSTER_LOW: float = 20.0

VALID_VELOCITY_VALUES = ("Slow", "Moderate", "Fast")
VALID_MAGNITUDE_VALUES = ("Strong", "Moderate", "Weak")
VALID_REGIME_VALUES = ("bull", "bear", "neutral")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RotationStatus(str, Enum):
    ACTIVE = "active"
    EMERGING = "emerging"
    NONE = "none"


class RotationType(str, Enum):
    CYCLICAL = "cyclical"
    DEFENSIVE = "defensive"
    OFFENSIVE = "offensive"
    THEMATIC = "thematic"
    BROAD = "broad"
    REVERSAL = "reversal"
    NONE = "none"


class RotationStage(str, Enum):
    EARLY = "early"
    MID = "mid"
    LATE = "late"
    EXHAUSTING = "exhausting"


# ---------------------------------------------------------------------------
# Supporting Models
# ---------------------------------------------------------------------------

class MarketRegime(BaseModel):
    """Market regime assessment derived from available data."""

    regime: Literal["bull", "bear", "neutral"]
    bull_signals_present: List[str] = Field(default_factory=list)
    bear_signals_present: List[str] = Field(default_factory=list)
    sector_breadth_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    distribution_day_count: Optional[int] = Field(None, ge=0)
    regime_note: str = Field(..., min_length=10)


class SignalReading(BaseModel):
    """Result of a single rotation signal detector."""

    signal_name: str = Field(..., min_length=1)
    triggered: bool
    value: str = Field(..., min_length=1)
    threshold: str = Field(..., min_length=1)
    evidence: str = Field(..., min_length=10)
    etf_confirmation: Optional[str] = Field(None, description="ETF-based evidence supporting or contradicting this signal")


class RotationSignals(BaseModel):
    """All 5 rotation signal readings."""

    rs_divergence: SignalReading
    leadership_change: SignalReading
    breadth_shift: SignalReading
    elite_concentration_shift: SignalReading
    ibd_keep_migration: SignalReading
    signals_active: int = Field(..., ge=0, le=5)

    @model_validator(mode="after")
    def validate_signals_active_count(self) -> "RotationSignals":
        count = sum(1 for s in [
            self.rs_divergence,
            self.leadership_change,
            self.breadth_shift,
            self.elite_concentration_shift,
            self.ibd_keep_migration,
        ] if s.triggered)
        if self.signals_active != count:
            raise ValueError(
                f"signals_active={self.signals_active} but {count} signals are triggered"
            )
        return self


class SectorFlow(BaseModel):
    """A sector gaining or losing leadership."""

    sector: str = Field(...)
    cluster: str = Field(...)
    direction: Literal["outflow", "inflow"]
    current_rank: int = Field(..., ge=1)
    avg_rs: float = Field(..., ge=1.0, le=99.0)
    elite_pct: float = Field(..., ge=0.0, le=100.0)
    stock_count: int = Field(..., ge=0)
    magnitude: str = Field(...)
    evidence: str = Field(..., min_length=10)

    @field_validator("cluster")
    @classmethod
    def validate_cluster(cls, v: str) -> str:
        if v not in VALID_CLUSTER_NAMES:
            raise ValueError(f"cluster '{v}' not in {VALID_CLUSTER_NAMES}")
        return v

    @field_validator("magnitude")
    @classmethod
    def validate_magnitude(cls, v: str) -> str:
        if v not in VALID_MAGNITUDE_VALUES:
            raise ValueError(f"magnitude must be one of {VALID_MAGNITUDE_VALUES}")
        return v


# ---------------------------------------------------------------------------
# Top-level Output
# ---------------------------------------------------------------------------

class RotationDetectionOutput(BaseModel):
    """Top-level output contract for the Rotation Detector."""

    verdict: RotationStatus
    confidence: int = Field(..., ge=0, le=100)
    rotation_type: RotationType
    rotation_stage: Optional[RotationStage] = None
    market_regime: MarketRegime
    signals: RotationSignals
    source_sectors: List[SectorFlow] = Field(default_factory=list)
    destination_sectors: List[SectorFlow] = Field(default_factory=list)
    stable_sectors: List[str] = Field(default_factory=list)
    velocity: Optional[str] = Field(None)
    etf_flow_summary: Optional[str] = Field(None, description="Summary of ETF RS/volume trends supporting rotation signals")
    strategist_notes: List[str] = Field(..., min_length=1)
    analysis_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD"
    )
    summary: str = Field(..., min_length=50)
    rotation_narrative: Optional[str] = Field(None, min_length=100, description="LLM historical context for rotation pattern")
    narrative_source: Optional[str] = Field(None, description="'llm' or 'template'")

    @field_validator("narrative_source")
    @classmethod
    def validate_narrative_source(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("llm", "template"):
            raise ValueError(f"narrative_source must be 'llm' or 'template', got '{v}'")
        return v

    @field_validator("velocity")
    @classmethod
    def validate_velocity(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_VELOCITY_VALUES:
            raise ValueError(f"velocity must be one of {VALID_VELOCITY_VALUES}")
        return v

    @model_validator(mode="after")
    def validate_verdict_signals(self) -> "RotationDetectionOutput":
        """ACTIVE requires >=3 signals + confidence>=50, EMERGING==2, NONE<=1."""
        sa = self.signals.signals_active
        if self.verdict == RotationStatus.ACTIVE:
            if sa < 3:
                raise ValueError(f"ACTIVE requires signals_active >= 3, got {sa}")
            if self.confidence < 50:
                raise ValueError(
                    f"ACTIVE requires confidence >= 50, got {self.confidence}"
                )
        elif self.verdict == RotationStatus.EMERGING:
            if sa != 2:
                raise ValueError(f"EMERGING requires signals_active == 2, got {sa}")
        elif self.verdict == RotationStatus.NONE:
            if sa > 1:
                raise ValueError(f"NONE requires signals_active <= 1, got {sa}")
        return self

    @model_validator(mode="after")
    def validate_source_direction(self) -> "RotationDetectionOutput":
        for sf in self.source_sectors:
            if sf.direction != "outflow":
                raise ValueError(
                    f"source_sectors must have direction='outflow', got '{sf.direction}'"
                )
        return self

    @model_validator(mode="after")
    def validate_dest_direction(self) -> "RotationDetectionOutput":
        for sf in self.destination_sectors:
            if sf.direction != "inflow":
                raise ValueError(
                    f"destination_sectors must have direction='inflow', got '{sf.direction}'"
                )
        return self

    @model_validator(mode="after")
    def validate_rotation_type_with_verdict(self) -> "RotationDetectionOutput":
        """NONE verdict requires NONE type; non-NONE requires non-NONE type."""
        if self.verdict == RotationStatus.NONE and self.rotation_type != RotationType.NONE:
            raise ValueError(
                f"NONE verdict requires NONE rotation_type, got {self.rotation_type}"
            )
        if self.verdict != RotationStatus.NONE and self.rotation_type == RotationType.NONE:
            raise ValueError(
                f"Non-NONE verdict ({self.verdict}) requires a rotation_type other than NONE"
            )
        return self

    @model_validator(mode="after")
    def validate_stage_with_verdict(self) -> "RotationDetectionOutput":
        """ACTIVE/EMERGING must have a stage; NONE must not."""
        if self.verdict != RotationStatus.NONE and self.rotation_stage is None:
            raise ValueError(f"Verdict {self.verdict} requires a rotation_stage")
        if self.verdict == RotationStatus.NONE and self.rotation_stage is not None:
            raise ValueError("NONE verdict should not have rotation_stage")
        return self
