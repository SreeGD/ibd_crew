"""
Agent 11: Value Investor Output Schema
IBD Momentum Investment Framework v4.0

Pydantic models for value-oriented stock analysis:
Value Score (0-100), value categories (GARP, Deep Value,
Quality Value, Dividend Value), value trap detection,
and momentum-value alignment assessment.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALUE_CATEGORIES = ("GARP", "Deep Value", "Quality Value", "Dividend Value", "Not Value")

VALUE_TRAP_RISK_LEVELS = ("High", "Moderate", "Low", "None")

MV_ALIGNMENT_LABELS = ("Aligned", "Mild Mismatch", "Strong Mismatch")

# Morningstar scoring maps
STAR_POINTS: Dict[str, int] = {"5-star": 40, "4-star": 25, "3-star": 10, "2-star": 0, "1-star": 0}
MOAT_POINTS: Dict[str, int] = {"Wide": 30, "Narrow": 15, "None": 0}
UNCERTAINTY_POINTS: Dict[str, int] = {"Low": 10, "Medium": 7, "High": 3, "Very High": 0}

# Value Score component weights (sum to 1.0)
VALUE_SCORE_WEIGHTS: Dict[str, float] = {
    "morningstar": 0.30,
    "pe_value": 0.20,
    "peg_value": 0.20,
    "moat_quality": 0.15,
    "discount": 0.15,
}

# P/E category to value score mapping
PE_SCORE_MAP: Dict[str, float] = {
    "Deep Value": 100.0,
    "Value": 75.0,
    "Reasonable": 50.0,
    "Growth Premium": 25.0,
    "Speculative": 0.0,
}

# Value category descriptions
VALUE_CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "Quality Value": "Wide moat companies with strong Morningstar ratings trading at or below fair value — premier quality at a reasonable price.",
    "Deep Value": "Stocks trading at significant discounts to intrinsic value or sector peers — statistically cheap with potential for mean reversion.",
    "GARP": "Growth at a Reasonable Price — stocks with strong earnings growth and momentum but reasonable P/E and PEG valuations.",
    "Dividend Value": "Income-generating stocks with attractive dividend yields and reasonable valuations — value through cash returns.",
    "Not Value": "Stocks that do not meet value investing criteria — may be momentum or speculative plays.",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ValueStock(BaseModel):
    """A stock with complete value investor assessment."""

    symbol: str = Field(..., min_length=1, max_length=10)
    company_name: str = Field(..., min_length=1)
    sector: str
    tier: int = Field(..., ge=1, le=3)
    tier_label: str

    # IBD Ratings (carried forward)
    composite_rating: int = Field(..., ge=1, le=99)
    rs_rating: int = Field(..., ge=1, le=99)
    eps_rating: int = Field(..., ge=1, le=99)
    smr_rating: Optional[str] = None
    acc_dis_rating: Optional[str] = None

    # Morningstar Data
    morningstar_rating: Optional[str] = None
    economic_moat: Optional[str] = None
    fair_value: Optional[float] = None
    morningstar_price: Optional[float] = None
    price_to_fair_value: Optional[float] = None
    morningstar_uncertainty: Optional[str] = None

    # Valuation Metrics (from Agent 02)
    estimated_pe: Optional[float] = None
    pe_category: Optional[str] = None
    peg_ratio: Optional[float] = None
    peg_category: Optional[str] = None
    llm_pe_ratio: Optional[float] = None
    llm_forward_pe: Optional[float] = None
    llm_dividend_yield: Optional[float] = None
    llm_market_cap_b: Optional[float] = None
    llm_valuation_grade: Optional[str] = None

    # === Value Investor Computed Fields ===

    # Morningstar Score (0-100)
    morningstar_score: float = Field(0, ge=0, le=100)
    star_points: int = Field(0, ge=0, le=40)
    moat_points: int = Field(0, ge=0, le=30)
    discount_points: int = Field(0, ge=0, le=20)
    certainty_points: int = Field(0, ge=0, le=10)

    # Component scores (0-100 each)
    pe_value_score: float = Field(0, ge=0, le=100)
    peg_value_score: float = Field(0, ge=0, le=100)
    moat_quality_score: float = Field(0, ge=0, le=100)
    discount_score: float = Field(0, ge=0, le=100)

    # Composite Value Score (0-100, weighted average)
    value_score: float = Field(..., ge=0, le=100)

    # Value Category
    value_category: str
    value_category_reasoning: str = Field(..., min_length=20)

    # Value Rank (1 = most attractive)
    value_rank: int = Field(..., ge=1)

    # Value Trap Assessment
    is_value_trap_risk: bool = Field(False)
    value_trap_risk_level: str = Field("None")
    value_trap_signals: List[str] = Field(default_factory=list)

    # Momentum-Value Alignment
    momentum_value_alignment: str
    alignment_detail: str = Field(..., min_length=10)

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("value_category")
    @classmethod
    def validate_value_category(cls, v: str) -> str:
        if v not in VALUE_CATEGORIES:
            raise ValueError(f"value_category '{v}' not in {VALUE_CATEGORIES}")
        return v

    @field_validator("value_trap_risk_level")
    @classmethod
    def validate_trap_risk(cls, v: str) -> str:
        if v not in VALUE_TRAP_RISK_LEVELS:
            raise ValueError(f"value_trap_risk_level '{v}' not in {VALUE_TRAP_RISK_LEVELS}")
        return v

    @field_validator("momentum_value_alignment")
    @classmethod
    def validate_alignment(cls, v: str) -> str:
        if v not in MV_ALIGNMENT_LABELS:
            raise ValueError(f"momentum_value_alignment '{v}' not in {MV_ALIGNMENT_LABELS}")
        return v


class SectorValueAnalysis(BaseModel):
    """Value characteristics at the sector level."""

    sector: str
    stock_count: int = Field(..., ge=0)
    avg_value_score: float = Field(..., ge=0, le=100)
    avg_pe: Optional[float] = None
    avg_peg: Optional[float] = None
    avg_pfv: Optional[float] = None
    garp_count: int = Field(0, ge=0)
    deep_value_count: int = Field(0, ge=0)
    quality_value_count: int = Field(0, ge=0)
    dividend_value_count: int = Field(0, ge=0)
    value_trap_count: int = Field(0, ge=0)
    moat_wide_count: int = Field(0, ge=0)
    moat_narrow_count: int = Field(0, ge=0)
    top_value_stocks: List[str] = Field(default_factory=list)
    sector_value_rank: int = Field(..., ge=1)


class ValueCategorySummary(BaseModel):
    """Aggregate statistics for a value category."""

    category: str
    stock_count: int = Field(..., ge=0)
    avg_value_score: float = Field(..., ge=0, le=100)
    avg_pe: Optional[float] = None
    avg_peg: Optional[float] = None
    avg_pfv: Optional[float] = None
    top_stocks: List[str] = Field(default_factory=list)
    description: str = Field(..., min_length=20)


class MoatAnalysisSummary(BaseModel):
    """Overall moat analysis."""

    wide_moat_count: int = Field(0, ge=0)
    narrow_moat_count: int = Field(0, ge=0)
    no_moat_count: int = Field(0, ge=0)
    no_data_count: int = Field(0, ge=0)
    wide_moat_stocks: List[str] = Field(default_factory=list)
    avg_value_score_wide: Optional[float] = None
    avg_value_score_narrow: Optional[float] = None
    avg_pfv_wide: Optional[float] = None
    avg_pfv_narrow: Optional[float] = None


class ValueInvestorOutput(BaseModel):
    """Top-level output contract for the Value Investor Agent."""

    value_stocks: List[ValueStock] = Field(
        ..., min_length=1,
        description="All rated stocks with value assessment, ranked by value_score"
    )
    sector_value_analysis: List[SectorValueAnalysis] = Field(
        ..., min_length=1,
        description="Value analysis per sector"
    )
    value_category_summaries: List[ValueCategorySummary] = Field(
        ..., min_length=1,
        description="Summary per value category"
    )
    moat_analysis: MoatAnalysisSummary
    value_traps: List[str] = Field(
        default_factory=list,
        description="Symbols flagged as value trap risks"
    )
    momentum_value_mismatches: List[str] = Field(
        default_factory=list,
        description="Symbols with strong momentum-value divergence"
    )
    top_value_picks: List[str] = Field(
        ..., min_length=1,
        description="Top value opportunities by value_score"
    )
    methodology_notes: str = Field(..., min_length=1)
    analysis_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    summary: str = Field(..., min_length=50)

    @model_validator(mode="after")
    def validate_value_ranks_unique(self) -> "ValueInvestorOutput":
        ranks = [vs.value_rank for vs in self.value_stocks]
        if len(ranks) != len(set(ranks)):
            raise ValueError("Value ranks must be unique")
        return self

    @model_validator(mode="after")
    def validate_traps_in_stocks(self) -> "ValueInvestorOutput":
        stock_symbols = {vs.symbol for vs in self.value_stocks}
        for trap in self.value_traps:
            if trap not in stock_symbols:
                raise ValueError(f"Value trap '{trap}' not found in value_stocks")
        return self

    @model_validator(mode="after")
    def validate_top_picks_in_stocks(self) -> "ValueInvestorOutput":
        stock_symbols = {vs.symbol for vs in self.value_stocks}
        for pick in self.top_value_picks:
            if pick not in stock_symbols:
                raise ValueError(f"Top pick '{pick}' not found in value_stocks")
        return self
