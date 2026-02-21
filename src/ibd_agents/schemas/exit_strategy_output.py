"""
Agent 15: Exit Strategist — Output Schema
IBD Momentum Investment Framework v4.0

Output contract for the Exit Strategist.
Evaluates every open position against 8 IBD sell rules,
produces urgency-ranked exit signals with evidence chains,
portfolio impact summary, and health score.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ExitMarketRegime(str, Enum):
    """IBD market regime for exit analysis."""
    CONFIRMED_UPTREND = "CONFIRMED_UPTREND"
    UPTREND_UNDER_PRESSURE = "UPTREND_UNDER_PRESSURE"
    CORRECTION = "CORRECTION"
    RALLY_ATTEMPT = "RALLY_ATTEMPT"


class Urgency(str, Enum):
    """Signal urgency level."""
    CRITICAL = "CRITICAL"    # Act today
    WARNING = "WARNING"      # Act within 2-3 days
    WATCH = "WATCH"          # Monitor closely
    HEALTHY = "HEALTHY"      # No action needed


class ExitActionType(str, Enum):
    """Recommended action for a position."""
    SELL_ALL = "SELL_ALL"
    TRIM = "TRIM"
    TIGHTEN_STOP = "TIGHTEN_STOP"
    HOLD = "HOLD"
    HOLD_8_WEEK = "HOLD_8_WEEK"


class SellType(str, Enum):
    """Whether the sell is offensive (taking profit) or defensive (cutting loss)."""
    OFFENSIVE = "OFFENSIVE"
    DEFENSIVE = "DEFENSIVE"
    NOT_APPLICABLE = "N/A"


class SellRule(str, Enum):
    """IBD sell rules."""
    STOP_LOSS_7_8 = "STOP_LOSS_7_8"
    CLIMAX_TOP = "CLIMAX_TOP"
    BELOW_50_DAY_MA = "BELOW_50_DAY_MA"
    RS_DETERIORATION = "RS_DETERIORATION"
    PROFIT_TARGET_20_25 = "PROFIT_TARGET_20_25"
    SECTOR_DISTRIBUTION = "SECTOR_DISTRIBUTION"
    REGIME_TIGHTENED_STOP = "REGIME_TIGHTENED_STOP"
    EARNINGS_RISK = "EARNINGS_RISK"
    NONE = "NONE"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SELL_RULE_NAMES: list[str] = [r.value for r in SellRule if r != SellRule.NONE]

# Stop loss thresholds by regime (low%, high%)
REGIME_STOP_THRESHOLDS: dict[str, tuple[float, float]] = {
    "CONFIRMED_UPTREND": (7.0, 8.0),
    "UPTREND_UNDER_PRESSURE": (3.0, 5.0),
    "CORRECTION": (2.0, 3.0),
    "RALLY_ATTEMPT": (5.0, 7.0),
}

# Profit target range
PROFIT_TARGET_RANGE: tuple[float, float] = (20.0, 25.0)

# 8-week hold rule: 20%+ gain in under 3 weeks = hold 8 weeks from breakout
EIGHT_WEEK_HOLD_DAYS: int = 56
FAST_GAIN_MAX_DAYS: int = 21

# Climax top detection
CLIMAX_SURGE_MIN_PCT: float = 25.0
CLIMAX_SURGE_MAX_PCT: float = 50.0
CLIMAX_SURGE_MAX_WEEKS: int = 3

# RS deterioration
RS_MIN_THRESHOLD: int = 70
RS_DROP_THRESHOLD: int = 15

# Sector distribution
SECTOR_DIST_DAYS_THRESHOLD: int = 4
SECTOR_DIST_WEEKS: int = 3

# Earnings risk
EARNINGS_PROXIMITY_DAYS: int = 21
EARNINGS_MIN_CUSHION_PCT: float = 15.0

# Hard backstop (MUST_NOT-S1)
HARD_BACKSTOP_LOSS_PCT: float = 10.0

# Health score
HEALTH_SCORE_RANGE: tuple[int, int] = (1, 10)

# Urgency sort order
URGENCY_ORDER: dict[str, int] = {
    "CRITICAL": 0,
    "WARNING": 1,
    "WATCH": 2,
    "HEALTHY": 3,
}

# Rule ID mapping
RULE_ID_MAP: dict[str, str] = {
    "STOP_LOSS_7_8": "MUST-M2",
    "CLIMAX_TOP": "MUST-M3",
    "BELOW_50_DAY_MA": "MUST-M4",
    "RS_DETERIORATION": "MUST-M5",
    "PROFIT_TARGET_20_25": "MUST-M6",
    "REGIME_TIGHTENED_STOP": "MUST-M7",
    "SECTOR_DISTRIBUTION": "MUST-M8",
    "EARNINGS_RISK": "MUST_NOT-S2",
    "NONE": "HEALTHY",
}


# ---------------------------------------------------------------------------
# Supporting Models
# ---------------------------------------------------------------------------

class EvidenceLink(BaseModel):
    """Evidence chain: data_point -> rule_triggered -> recommendation."""

    data_point: str = Field(..., min_length=10, description="Specific data observation")
    rule_triggered: SellRule = Field(...)
    rule_id: str = Field(..., min_length=4, description="Rule reference e.g. 'MUST-M2'")


class SellRuleResult(BaseModel):
    """Result of evaluating one sell rule against one position."""

    rule: SellRule = Field(...)
    triggered: bool = Field(...)
    value: Optional[float] = Field(
        None, description="Measured value (e.g., loss %, RS drop)",
    )
    threshold: Optional[float] = Field(
        None, description="Threshold that was compared against",
    )
    detail: str = Field(..., min_length=10)


class PositionSignal(BaseModel):
    """Exit signal for a single position."""

    symbol: str = Field(..., min_length=1, max_length=10)
    tier: int = Field(..., ge=1, le=3)
    asset_type: Literal["stock", "etf"] = Field(...)
    current_price: float = Field(..., gt=0)
    buy_price: float = Field(..., gt=0)
    gain_loss_pct: float = Field(...)
    days_held: int = Field(..., ge=0)
    urgency: Urgency = Field(...)
    action: ExitActionType = Field(...)
    sell_type: SellType = Field(...)
    sell_pct: Optional[float] = Field(
        None, ge=0, le=100,
        description="% of position to sell. None for HOLD actions.",
    )
    stop_price: float = Field(..., ge=0, description="Current mental stop price")
    rules_triggered: list[SellRule] = Field(default_factory=list)
    evidence: list[EvidenceLink] = Field(..., min_length=1)
    reasoning: str = Field(..., min_length=20)
    what_would_change: str = Field(..., min_length=20)

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()

    @model_validator(mode="after")
    def validate_sell_type_consistency(self) -> "PositionSignal":
        """SELL/TRIM must have OFFENSIVE or DEFENSIVE; HOLD variants must have N/A."""
        if self.action in (ExitActionType.SELL_ALL, ExitActionType.TRIM):
            if self.sell_type == SellType.NOT_APPLICABLE:
                raise ValueError(
                    f"{self.symbol}: {self.action.value} requires OFFENSIVE or DEFENSIVE sell_type"
                )
        if self.action in (
            ExitActionType.HOLD,
            ExitActionType.HOLD_8_WEEK,
            ExitActionType.TIGHTEN_STOP,
        ):
            if self.sell_type != SellType.NOT_APPLICABLE:
                raise ValueError(
                    f"{self.symbol}: {self.action.value} requires N/A sell_type"
                )
        return self

    @model_validator(mode="after")
    def validate_evidence_per_triggered_rule(self) -> "PositionSignal":
        """At least one evidence link per triggered rule."""
        triggered = {r for r in self.rules_triggered if r != SellRule.NONE}
        evidenced = {e.rule_triggered for e in self.evidence if e.rule_triggered != SellRule.NONE}
        missing = triggered - evidenced
        if missing:
            raise ValueError(
                f"{self.symbol}: rules {missing} triggered but no evidence provided"
            )
        return self

    @model_validator(mode="after")
    def validate_sell_pct_for_actions(self) -> "PositionSignal":
        """SELL_ALL=100%, TRIM=1-99%, others=None."""
        if self.action == ExitActionType.SELL_ALL:
            if self.sell_pct is not None and self.sell_pct != 100.0:
                raise ValueError(
                    f"{self.symbol}: SELL_ALL must have sell_pct=100.0 or None"
                )
        if self.action == ExitActionType.TRIM:
            if self.sell_pct is None or not (1.0 <= self.sell_pct < 100.0):
                raise ValueError(
                    f"{self.symbol}: TRIM must have sell_pct between 1 and 99"
                )
        return self


class PortfolioImpact(BaseModel):
    """Portfolio-level impact if all recommendations are executed."""

    current_cash_pct: float = Field(..., ge=0, le=100)
    projected_cash_pct: float = Field(..., ge=0, le=100)
    current_top_holding_pct: float = Field(..., ge=0, le=100)
    projected_top_holding_pct: float = Field(..., ge=0, le=100)
    positions_healthy: int = Field(..., ge=0)
    positions_watch: int = Field(..., ge=0)
    positions_warning: int = Field(..., ge=0)
    positions_critical: int = Field(..., ge=0)
    sector_concentration_risk: str = Field(...)

    @field_validator("sector_concentration_risk")
    @classmethod
    def validate_concentration_risk(cls, v: str) -> str:
        if v not in ("HIGH", "MEDIUM", "LOW"):
            raise ValueError(
                f"sector_concentration_risk must be HIGH/MEDIUM/LOW, got '{v}'"
            )
        return v


# ---------------------------------------------------------------------------
# Top-level Output
# ---------------------------------------------------------------------------

class ExitStrategyOutput(BaseModel):
    """Top-level output contract for the Exit Strategist."""

    analysis_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD",
    )
    market_regime: ExitMarketRegime = Field(...)
    portfolio_health_score: int = Field(
        ..., ge=1, le=10,
        description="1-10, where 10=all healthy, 1=multiple critical signals",
    )
    signals: list[PositionSignal] = Field(
        ..., min_length=1,
        description="One signal per position, ordered by urgency (CRITICAL first)",
    )
    portfolio_impact: PortfolioImpact = Field(...)
    summary: str = Field(..., min_length=50)
    reasoning_source: Optional[str] = Field(
        None, description="'llm' or 'deterministic'",
    )

    @model_validator(mode="after")
    def validate_signals_ordered_by_urgency(self) -> "ExitStrategyOutput":
        """Signals must be ordered CRITICAL -> WARNING -> WATCH -> HEALTHY."""
        urgencies = [URGENCY_ORDER[s.urgency.value] for s in self.signals]
        if urgencies != sorted(urgencies):
            raise ValueError(
                "Signals must be ordered by urgency (CRITICAL first)"
            )
        return self

    @model_validator(mode="after")
    def validate_impact_counts_match(self) -> "ExitStrategyOutput":
        """Impact urgency counts must match signals."""
        actual = {
            "CRITICAL": sum(1 for s in self.signals if s.urgency == Urgency.CRITICAL),
            "WARNING": sum(1 for s in self.signals if s.urgency == Urgency.WARNING),
            "WATCH": sum(1 for s in self.signals if s.urgency == Urgency.WATCH),
            "HEALTHY": sum(1 for s in self.signals if s.urgency == Urgency.HEALTHY),
        }
        impact = self.portfolio_impact
        if impact.positions_critical != actual["CRITICAL"]:
            raise ValueError(
                f"Impact critical={impact.positions_critical} vs signals {actual['CRITICAL']}"
            )
        if impact.positions_warning != actual["WARNING"]:
            raise ValueError(
                f"Impact warning={impact.positions_warning} vs signals {actual['WARNING']}"
            )
        if impact.positions_watch != actual["WATCH"]:
            raise ValueError(
                f"Impact watch={impact.positions_watch} vs signals {actual['WATCH']}"
            )
        if impact.positions_healthy != actual["HEALTHY"]:
            raise ValueError(
                f"Impact healthy={impact.positions_healthy} vs signals {actual['HEALTHY']}"
            )
        return self

    @field_validator("reasoning_source")
    @classmethod
    def validate_reasoning_source(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("llm", "deterministic"):
            raise ValueError(
                f"reasoning_source must be 'llm' or 'deterministic', got '{v}'"
            )
        return v
