"""
Agent 10: Executive Summary Synthesizer — Output Schema
IBD Momentum Investment Framework v4.0

Output contract for the Executive Summary Synthesizer.
Reads all 9 agent outputs and produces a unified investment thesis
with cross-agent connections, contradictions, and key metrics dashboard.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Supporting Models
# ---------------------------------------------------------------------------

class KeyNumbersDashboard(BaseModel):
    """Key numbers extracted from all 9 agent outputs."""

    # Research (Agent 01)
    total_stocks_scanned: int = Field(..., ge=0)
    stocks_in_universe: int = Field(..., ge=0)
    etfs_in_universe: int = Field(..., ge=0)

    # Analyst (Agent 02)
    stocks_rated: int = Field(..., ge=0)
    etfs_rated: int = Field(..., ge=0)
    tier_1_count: int = Field(..., ge=0)
    tier_2_count: int = Field(..., ge=0)
    tier_3_count: int = Field(..., ge=0)

    # Rotation (Agent 03)
    rotation_verdict: str = Field(...)
    rotation_confidence: int = Field(..., ge=0, le=100)
    market_regime: str = Field(...)

    # Portfolio (Agent 05)
    portfolio_positions: int = Field(..., ge=0)
    stock_count: int = Field(..., ge=0)
    etf_count: int = Field(..., ge=0)
    large_cap_count: int = Field(0, ge=0)
    mid_cap_count: int = Field(0, ge=0)
    small_cap_count: int = Field(0, ge=0)
    cash_pct: float = Field(..., ge=0.0)

    # Risk (Agent 06)
    risk_status: str = Field(...)
    sleep_well_score: int = Field(..., ge=1, le=10)
    risk_warnings_count: int = Field(..., ge=0)

    # Returns (Agent 07)
    base_case_return_12m: float = Field(...)
    bull_case_return_12m: float = Field(...)
    bear_case_return_12m: float = Field(...)

    # Reconciler (Agent 08)
    turnover_pct: float = Field(..., ge=0.0)
    actions_count: int = Field(..., ge=0)


class CrossAgentConnection(BaseModel):
    """A link between findings from two different agents."""

    from_agent: str = Field(..., min_length=1)
    to_agent: str = Field(..., min_length=1)
    connection: str = Field(..., min_length=10)
    implication: str = Field(..., min_length=10)


class Contradiction(BaseModel):
    """A contradiction or tension between two agent findings."""

    agent_a: str = Field(..., min_length=1)
    agent_b: str = Field(..., min_length=1)
    finding_a: str = Field(..., min_length=10)
    finding_b: str = Field(..., min_length=10)
    resolution: str = Field(..., min_length=10)


# ---------------------------------------------------------------------------
# Top-level Output
# ---------------------------------------------------------------------------

class SynthesisOutput(BaseModel):
    """Top-level output contract for the Executive Summary Synthesizer."""

    # Narrative sections (LLM-generated or template-filled)
    investment_thesis: str = Field(
        ..., min_length=200,
        description="Why your portfolio looks the way it does"
    )
    portfolio_narrative: str = Field(
        ..., min_length=100,
        description="Connects rotation → strategy → portfolio construction"
    )
    risk_reward_assessment: str = Field(
        ..., min_length=100,
        description="Risk/reward tradeoff summary"
    )
    market_context: str = Field(
        ..., min_length=100,
        description="Current market environment and regime"
    )

    # Structured data (always deterministic)
    key_numbers: KeyNumbersDashboard = Field(...)
    cross_agent_connections: List[CrossAgentConnection] = Field(
        ..., min_length=3,
        description="Links between agent findings"
    )
    contradictions: List[Contradiction] = Field(
        default_factory=list,
        description="Tensions between agent findings"
    )
    action_items: List[str] = Field(
        ..., min_length=3, max_length=5,
        description="3-5 actionable next steps"
    )

    # Metadata
    synthesis_source: Literal["llm", "template"] = Field(
        ..., description="Whether narrative was LLM-generated or template-filled"
    )
    analysis_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD"
    )
    summary: str = Field(..., min_length=50)

    @model_validator(mode="after")
    def validate_action_items_non_empty(self) -> "SynthesisOutput":
        """Each action item must have content."""
        for i, item in enumerate(self.action_items):
            if len(item.strip()) < 10:
                raise ValueError(
                    f"action_items[{i}] too short: '{item}'. Must be >= 10 chars."
                )
        return self
