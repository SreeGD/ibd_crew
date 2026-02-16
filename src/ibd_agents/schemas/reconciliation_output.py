"""
Agent 08: Portfolio Reconciler — Output Schema
IBD Momentum Investment Framework v4.0

Output contract for the Portfolio Reconciler.
Compares current holdings vs recommended portfolio,
generates action plan with money flow and 4-week implementation.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ibd_agents.schemas.portfolio_output import ALL_KEEPS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMPLEMENTATION_PHASES: dict[int, str] = {
    1: "LIQUIDATION",
    2: "T1 MOMENTUM",
    3: "T2 QUALITY GROWTH",
    4: "T3 DEFENSIVE",
}

ACTION_TYPES = ("KEEP", "SELL", "BUY", "ADD", "TRIM")
ACTION_PRIORITIES = ("HIGH", "MEDIUM", "LOW")


# ---------------------------------------------------------------------------
# Supporting Models
# ---------------------------------------------------------------------------

class CurrentHolding(BaseModel):
    """A position in the current portfolio."""

    symbol: str = Field(..., min_length=1, max_length=10)
    shares: float = Field(..., ge=0)
    market_value: float = Field(..., ge=0)
    account: str = Field(..., min_length=1)
    sector: str = Field(default="UNKNOWN")

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()


class HoldingsSummary(BaseModel):
    """Summary of current holdings."""

    holdings: List[CurrentHolding] = Field(default_factory=list)
    total_value: float = Field(..., ge=0)
    account_count: int = Field(..., ge=1)


class PositionAction(BaseModel):
    """An action for one position (KEEP/SELL/BUY/ADD/TRIM)."""

    symbol: str = Field(..., min_length=1, max_length=10)
    action_type: str = Field(...)
    cap_size: Optional[str] = Field(None, description="Market cap: large/mid/small")
    current_pct: float = Field(..., ge=0.0)
    target_pct: float = Field(..., ge=0.0)
    dollar_change: float = Field(..., description="Positive=inflow, negative=outflow")
    priority: str = Field(...)
    week: int = Field(..., ge=1, le=4)
    rationale: str = Field(..., min_length=10)
    sharpe_ratio: Optional[float] = Field(None, description="Estimated Sharpe ratio for this position")
    alpha_pct: Optional[float] = Field(None, description="Estimated CAPM alpha %")

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        if v not in ACTION_TYPES:
            raise ValueError(f"action_type must be one of {ACTION_TYPES}, got '{v}'")
        return v

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        if v not in ACTION_PRIORITIES:
            raise ValueError(f"priority must be one of {ACTION_PRIORITIES}, got '{v}'")
        return v


class MoneyFlow(BaseModel):
    """Sources and uses of capital in the transition."""

    sell_proceeds: float = Field(..., ge=0)
    trim_proceeds: float = Field(..., ge=0)
    buy_cost: float = Field(..., ge=0)
    add_cost: float = Field(..., ge=0)
    net_cash_change: float = Field(...)
    cash_reserve_pct: float = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_balanced(self) -> "MoneyFlow":
        """Sources should cover uses (allowing small tolerance)."""
        sources = self.sell_proceeds + self.trim_proceeds
        uses = self.buy_cost + self.add_cost
        # Net cash change = sources - uses
        expected_net = sources - uses
        if abs(self.net_cash_change - expected_net) > 1.0:
            raise ValueError(
                f"net_cash_change={self.net_cash_change:.2f} doesn't match "
                f"sources({sources:.2f}) - uses({uses:.2f}) = {expected_net:.2f}"
            )
        return self


class ImplementationWeek(BaseModel):
    """One week of the 4-week implementation plan."""

    week: int = Field(..., ge=1, le=4)
    phase_name: str = Field(..., min_length=1)
    actions: List[PositionAction] = Field(default_factory=list)


class TransformationMetrics(BaseModel):
    """Before/after portfolio metrics."""

    before_sector_count: int = Field(..., ge=0)
    after_sector_count: int = Field(..., ge=0)
    before_position_count: int = Field(..., ge=0)
    after_position_count: int = Field(..., ge=0)
    turnover_pct: float = Field(..., ge=0.0, le=100.0)


class KeepVerification(BaseModel):
    """Verification of 21 keep positions."""

    keeps_in_current: List[str] = Field(default_factory=list)
    keeps_missing: List[str] = Field(default_factory=list)
    keeps_to_buy: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level Output
# ---------------------------------------------------------------------------

class ReconciliationOutput(BaseModel):
    """Top-level output contract for the Portfolio Reconciler."""

    current_holdings: HoldingsSummary = Field(...)
    actions: List[PositionAction] = Field(default_factory=list)
    money_flow: MoneyFlow = Field(...)
    implementation_plan: List[ImplementationWeek] = Field(
        ..., min_length=4, max_length=4
    )
    keep_verification: KeepVerification = Field(...)
    transformation_metrics: TransformationMetrics = Field(...)
    etf_implementation_notes: List[str] = Field(
        default_factory=list,
        description="ETF-specific trading sequence and execution guidance"
    )
    analysis_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD"
    )
    summary: str = Field(..., min_length=50)
    rationale_source: Optional[str] = Field(None, description="'llm' or 'deterministic'")

    @model_validator(mode="after")
    def validate_4_weeks(self) -> "ReconciliationOutput":
        """Implementation plan has weeks 1-4."""
        weeks = sorted(w.week for w in self.implementation_plan)
        if weeks != [1, 2, 3, 4]:
            raise ValueError(f"Implementation weeks must be [1,2,3,4], got {weeks}")
        return self

    @model_validator(mode="after")
    def validate_week_1_liquidation(self) -> "ReconciliationOutput":
        """Week 1 should focus on liquidation."""
        w1 = next((w for w in self.implementation_plan if w.week == 1), None)
        if w1 and "LIQUIDATION" not in w1.phase_name.upper():
            raise ValueError(
                f"Week 1 phase should be LIQUIDATION, got '{w1.phase_name}'"
            )
        return self

    @field_validator("rationale_source")
    @classmethod
    def validate_rationale_source(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("llm", "deterministic"):
            raise ValueError(f"rationale_source must be 'llm' or 'deterministic', got '{v}'")
        return v
