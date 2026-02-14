"""
Agent 09: Educator — Output Schema
IBD Momentum Investment Framework v4.0

Output contract for the Investment Educator.
Explains every decision made by the 8-agent crew using
clear language, analogies, and zero unexplained jargon.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_GLOSSARY_TERMS = [
    "Composite Rating",
    "RS Rating",
    "EPS Rating",
    "SMR Rating",
    "Acc/Dis Rating",
    "CAN SLIM",
    "Tier 1 Momentum",
    "Tier 2 Quality Growth",
    "Tier 3 Defensive",
    "Trailing Stop",
    "Stop Tightening",
    "Sector Rotation",
    "Multi-Source Validated",
    "IBD Keep Threshold",
    "Sleep Well Score",
]

IBD_CONCEPTS_TO_TEACH = [
    "Composite Rating",
    "Relative Strength (RS) Rating",
    "EPS Rating",
    "SMR Rating",
    "Accumulation/Distribution Rating",
    "CAN SLIM Methodology",
    "Three-Tier Portfolio System",
    "Trailing Stop Protocol",
    "Sector Rotation",
    "Multi-Source Validation",
    "ETF Selection and Ranking",
    "Theme ETFs vs Broad Market ETFs",
]

MIN_STOCK_EXPLANATIONS = 15
MIN_CONCEPT_LESSONS = 5
MIN_GLOSSARY_ENTRIES = 15
MIN_ACTION_ITEMS = 3
MAX_ACTION_ITEMS = 7


# ---------------------------------------------------------------------------
# Supporting Models
# ---------------------------------------------------------------------------

class TierGuide(BaseModel):
    """Explains the 3-tier portfolio system."""

    overview: str = Field(..., min_length=50)
    tier_1_description: str = Field(..., min_length=30)
    tier_2_description: str = Field(..., min_length=30)
    tier_3_description: str = Field(..., min_length=30)
    choosing_advice: str = Field(..., min_length=30)
    current_allocation: str = Field(..., min_length=10)
    analogy: str = Field(..., min_length=20)


class KeepExplanations(BaseModel):
    """Explains why 14 positions are pre-committed."""

    overview: str = Field(..., min_length=50)
    fundamental_keeps: str = Field(..., min_length=30)
    ibd_keeps: str = Field(..., min_length=30)
    total_keeps: int = Field(...)

    @field_validator("total_keeps")
    @classmethod
    def validate_total_keeps(cls, v: int) -> int:
        if v != 14:
            raise ValueError(f"total_keeps must be 14, got {v}")
        return v


class StockExplanation(BaseModel):
    """Educational explanation for one stock."""

    symbol: str = Field(..., min_length=1, max_length=10)
    company_name: str = Field(..., min_length=1)
    tier: int = Field(..., ge=1, le=3)
    cap_size: Optional[str] = Field(None, description="Market cap: large/mid/small")
    keep_category: Optional[str] = Field(default=None)
    one_liner: str = Field(..., min_length=10)
    why_selected: str = Field(..., min_length=50)
    ibd_ratings_explained: str = Field(..., min_length=20)
    key_strength: str = Field(..., min_length=10)
    key_risk: str = Field(..., min_length=10)
    position_context: str = Field(..., min_length=10)
    analogy: Optional[str] = Field(default=None)

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("keep_category")
    @classmethod
    def validate_keep_category(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("fundamental", "ibd"):
            raise ValueError(f"keep_category must be fundamental/ibd/None, got '{v}'")
        return v


class ETFExplanation(BaseModel):
    """Educational explanation for one ETF position."""

    symbol: str = Field(..., min_length=1, max_length=10)
    name: str = Field(..., min_length=1)
    tier: int = Field(..., ge=1, le=3)
    one_liner: str = Field(..., min_length=10)
    why_selected: str = Field(..., min_length=30)
    theme_context: Optional[str] = Field(None)
    conviction_explained: Optional[str] = Field(None)
    key_strength: str = Field(..., min_length=10)
    key_risk: str = Field(..., min_length=10)

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()


class TransitionGuide(BaseModel):
    """Explains the portfolio transition plan from Agent 08."""

    overview: str = Field(..., min_length=100)
    what_to_sell: str = Field(..., min_length=30)
    what_to_buy: str = Field(..., min_length=30)
    money_flow_explained: str = Field(..., min_length=100)
    timeline_explained: str = Field(..., min_length=30)
    before_after_summary: str = Field(..., min_length=30)
    key_numbers: str = Field(..., min_length=20)


class ConceptLesson(BaseModel):
    """One educational lesson about an IBD concept."""

    concept: str = Field(..., min_length=3)
    simple_explanation: str = Field(..., min_length=30)
    analogy: str = Field(..., min_length=15)
    why_it_matters: str = Field(..., min_length=20)
    example_from_analysis: str = Field(..., min_length=20)
    framework_reference: str = Field(..., min_length=10)


# ---------------------------------------------------------------------------
# Top-level Output
# ---------------------------------------------------------------------------

class EducatorOutput(BaseModel):
    """Top-level output contract for the Investment Educator."""

    executive_summary: str = Field(..., min_length=200)
    tier_guide: TierGuide = Field(...)
    keep_explanations: KeepExplanations = Field(...)
    stock_explanations: List[StockExplanation] = Field(
        ..., min_length=MIN_STOCK_EXPLANATIONS
    )
    rotation_explanation: str = Field(..., min_length=100)
    risk_explainer: str = Field(..., min_length=100)
    returns_explanation: str = Field(
        default="",
        description="Plain-language explanation of returns projections (from Agent 07)"
    )
    transition_guide: TransitionGuide = Field(...)
    concept_lessons: List[ConceptLesson] = Field(
        ..., min_length=MIN_CONCEPT_LESSONS
    )
    action_items: List[str] = Field(
        ..., min_length=MIN_ACTION_ITEMS, max_length=MAX_ACTION_ITEMS
    )
    etf_explanations: List[ETFExplanation] = Field(
        default_factory=list,
        description="Educational explanations for ETF positions"
    )
    glossary: Dict[str, str] = Field(...)
    analysis_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD"
    )
    summary: str = Field(..., min_length=50)

    @model_validator(mode="after")
    def validate_glossary_coverage(self) -> "EducatorOutput":
        """Glossary must have at least MIN_GLOSSARY_ENTRIES entries."""
        if len(self.glossary) < MIN_GLOSSARY_ENTRIES:
            raise ValueError(
                f"Glossary needs >= {MIN_GLOSSARY_ENTRIES} entries, got {len(self.glossary)}"
            )
        return self

    @model_validator(mode="after")
    def validate_stock_tiers(self) -> "EducatorOutput":
        """Every stock explanation must have a valid tier."""
        for s in self.stock_explanations:
            if s.tier not in (1, 2, 3):
                raise ValueError(f"Stock {s.symbol} has invalid tier {s.tier}")
        return self
