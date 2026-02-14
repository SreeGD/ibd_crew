"""
Agent 02: Analyst Agent — Output Schema
IBD Momentum Investment Framework v4.0

Output contract for the Analyst Agent.
Applies Elite Screening, Tier Assignment, Sector Ranking,
and per-stock analysis to Research Agent's stocks.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ibd_agents.schemas.research_output import (
    IBD_SECTORS,
    VALID_ACC_DIS_RATINGS,
    VALID_SMR_RATINGS,
    is_ibd_keep_candidate,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIER_LABELS: dict[int, str] = {
    1: "Momentum",
    2: "Quality Growth",
    3: "Defensive",
}

PE_CATEGORIES = ("Deep Value", "Value", "Reasonable", "Growth Premium", "Speculative")
PEG_CATEGORIES = ("Undervalued", "Fair Value", "Expensive")
RETURN_CATEGORIES = ("Strong", "Good", "Moderate", "Weak")
SHARPE_CATEGORIES = ("Excellent", "Good", "Moderate", "Below Average")
ALPHA_CATEGORIES = ("Strong Outperformer", "Outperformer", "Slight Underperformer", "Underperformer")
RISK_RATING_CATEGORIES = ("Excellent", "Good", "Moderate", "Below Average", "Poor")
CATALYST_TYPES = ("earnings", "fda", "product_launch", "conference", "guidance", "dividend", "split", "other")


# ---------------------------------------------------------------------------
# Supporting Models
# ---------------------------------------------------------------------------

class EliteFilterSummary(BaseModel):
    """Summary of 4-gate elite screening results."""

    total_screened: int = Field(..., ge=0)
    passed_all_four: int = Field(..., ge=0)
    failed_eps: int = Field(..., ge=0)
    failed_rs: int = Field(..., ge=0)
    failed_smr: int = Field(..., ge=0)
    failed_acc_dis: int = Field(..., ge=0)
    missing_ratings: int = Field(..., ge=0, description="Couldn't apply filters")


class TierDistribution(BaseModel):
    """Distribution of stocks across tiers."""

    tier_1_count: int = Field(..., ge=0)
    tier_2_count: int = Field(..., ge=0)
    tier_3_count: int = Field(..., ge=0)
    below_threshold_count: int = Field(..., ge=0)
    unrated_count: int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Stock-level Models
# ---------------------------------------------------------------------------

class RatedStock(BaseModel):
    """A stock with full tier assignment and analyst assessment."""

    symbol: str = Field(..., min_length=1, max_length=10)
    company_name: str = Field(..., min_length=1)
    sector: str = Field(..., description="IBD sector")
    security_type: Literal["stock"] = "stock"

    # Tier Assignment (deterministic)
    tier: int = Field(..., description="1, 2, or 3")
    tier_label: str = Field(..., description="Momentum / Quality Growth / Defensive")

    # IBD Ratings (required for rated stocks)
    composite_rating: int = Field(..., ge=1, le=99)
    rs_rating: int = Field(..., ge=1, le=99)
    eps_rating: int = Field(..., ge=1, le=99)
    smr_rating: Optional[str] = Field(None, description="A/B/B-/C/D/E")
    acc_dis_rating: Optional[str] = Field(None, description="A/B/B-/C/D/E")

    # Elite Filter Results
    passes_eps_filter: bool = Field(..., description="EPS >= 85")
    passes_rs_filter: bool = Field(..., description="RS >= 75")
    passes_smr_filter: Optional[bool] = Field(None, description="SMR = A/B/B-")
    passes_acc_dis_filter: Optional[bool] = Field(None, description="Acc/Dis = A/B/B-")
    passes_all_elite: bool = Field(..., description="All applicable filters pass")

    # Flags
    is_ibd_keep: bool = Field(..., description="Comp>=93 AND RS>=90")
    is_multi_source_validated: bool = Field(False)

    # From Research
    ibd_lists: List[str] = Field(default_factory=list)
    schwab_themes: List[str] = Field(default_factory=list)
    validation_score: int = Field(0, ge=0)
    cap_size: Optional[Literal["large", "mid", "small"]] = Field(
        None, description="Market cap category from Research Agent"
    )

    # Morningstar metadata (from Research Agent)
    morningstar_rating: Optional[str] = Field(None, description="e.g. '5-star', '4-star'")
    economic_moat: Optional[str] = Field(None, description="Wide/Narrow/None")
    fair_value: Optional[float] = Field(None, description="Morningstar fair value estimate")
    morningstar_price: Optional[float] = Field(None, description="Price at Morningstar report")
    price_to_fair_value: Optional[float] = Field(None, description="Price/Fair Value ratio")
    morningstar_uncertainty: Optional[str] = Field(None, description="Low/Medium/High/Very High")

    # Analyst Assessment
    conviction: int = Field(..., ge=1, le=10)
    strengths: List[str] = Field(..., min_length=1, max_length=5)
    weaknesses: List[str] = Field(..., min_length=1, max_length=5)
    catalyst: str = Field(..., min_length=1)
    reasoning: str = Field(..., min_length=50)
    sector_rank_in_sector: int = Field(..., ge=1, description="Rank within sector")

    # Valuation & Risk Metrics (estimated, no external API)
    estimated_pe: Optional[float] = Field(None, ge=0.0, description="Estimated P/E ratio")
    pe_category: Optional[str] = Field(None, description="Deep Value/Value/Reasonable/Growth Premium/Speculative")
    peg_ratio: Optional[float] = Field(None, ge=0.0, description="P/E / (EPS Rating * 0.5)")
    peg_category: Optional[str] = Field(None, description="Undervalued/Fair Value/Expensive")
    estimated_beta: Optional[float] = Field(None, ge=0.0, description="Estimated beta vs S&P 500")
    estimated_return_pct: Optional[float] = Field(None, description="Estimated 12-month return %")
    return_category: Optional[str] = Field(None, description="Strong/Good/Moderate/Weak")
    estimated_volatility_pct: Optional[float] = Field(None, ge=0.0, description="Estimated annualized volatility %")
    sharpe_ratio: Optional[float] = Field(None, description="(Return - RiskFree) / Volatility")
    sharpe_category: Optional[str] = Field(None, description="Excellent/Good/Moderate/Below Average")
    alpha_pct: Optional[float] = Field(None, description="CAPM excess return %")
    alpha_category: Optional[str] = Field(None, description="Strong Outperformer to Underperformer")
    risk_rating: Optional[str] = Field(None, description="Excellent/Good/Moderate/Below Average/Poor")

    # LLM-Enriched Metrics (from Claude knowledge base)
    llm_pe_ratio: Optional[float] = Field(None, description="Real trailing P/E from LLM")
    llm_forward_pe: Optional[float] = Field(None, description="Real forward P/E from LLM")
    llm_beta: Optional[float] = Field(None, description="Real 5-year beta from LLM")
    llm_annual_return_1y: Optional[float] = Field(None, description="Real 1-year return % from LLM")
    llm_volatility: Optional[float] = Field(None, description="Real annualized volatility from LLM")
    llm_dividend_yield: Optional[float] = Field(None, description="Dividend yield % from LLM")
    llm_market_cap_b: Optional[float] = Field(None, description="Market cap in billions from LLM")
    llm_valuation_grade: Optional[str] = Field(None, description="LLM valuation assessment")
    llm_risk_grade: Optional[str] = Field(None, description="LLM risk assessment")
    llm_guidance: Optional[str] = Field(None, description="2-3 sentence investment guidance from LLM")
    valuation_source: Optional[str] = Field(None, description="'llm' or 'estimated'")

    # LLM-Enriched Catalyst Fields
    catalyst_date: Optional[str] = Field(None, description="Next catalyst date YYYY-MM-DD")
    catalyst_type: Optional[str] = Field(None, description="earnings/fda/product_launch/conference/guidance/other")
    catalyst_conviction_adjustment: int = Field(0, ge=-1, le=2, description="Conviction adjustment from catalyst timing")
    catalyst_source: Optional[str] = Field(None, description="'llm' or 'template'")

    # --- Validators ---

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("tier")
    @classmethod
    def validate_tier(cls, v: int) -> int:
        if v not in (1, 2, 3):
            raise ValueError(f"Tier must be 1, 2, or 3. Got {v}")
        return v

    @field_validator("tier_label")
    @classmethod
    def validate_tier_label(cls, v: str) -> str:
        if v not in TIER_LABELS.values():
            raise ValueError(f"tier_label must be one of {list(TIER_LABELS.values())}. Got '{v}'")
        return v

    @field_validator("smr_rating")
    @classmethod
    def validate_smr(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_SMR_RATINGS:
            raise ValueError(f"SMR rating '{v}' not in {VALID_SMR_RATINGS}")
        return v

    @field_validator("acc_dis_rating")
    @classmethod
    def validate_acc_dis(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_ACC_DIS_RATINGS:
            raise ValueError(f"Acc/Dis rating '{v}' not in {VALID_ACC_DIS_RATINGS}")
        return v

    @field_validator("pe_category")
    @classmethod
    def validate_pe_category(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in PE_CATEGORIES:
            raise ValueError(f"pe_category '{v}' not in {PE_CATEGORIES}")
        return v

    @field_validator("peg_category")
    @classmethod
    def validate_peg_category(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in PEG_CATEGORIES:
            raise ValueError(f"peg_category '{v}' not in {PEG_CATEGORIES}")
        return v

    @field_validator("return_category")
    @classmethod
    def validate_return_category(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in RETURN_CATEGORIES:
            raise ValueError(f"return_category '{v}' not in {RETURN_CATEGORIES}")
        return v

    @field_validator("sharpe_category")
    @classmethod
    def validate_sharpe_category(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in SHARPE_CATEGORIES:
            raise ValueError(f"sharpe_category '{v}' not in {SHARPE_CATEGORIES}")
        return v

    @field_validator("alpha_category")
    @classmethod
    def validate_alpha_category(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ALPHA_CATEGORIES:
            raise ValueError(f"alpha_category '{v}' not in {ALPHA_CATEGORIES}")
        return v

    @field_validator("risk_rating")
    @classmethod
    def validate_risk_rating(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in RISK_RATING_CATEGORIES:
            raise ValueError(f"risk_rating '{v}' not in {RISK_RATING_CATEGORIES}")
        return v

    @field_validator("llm_valuation_grade")
    @classmethod
    def validate_llm_valuation_grade(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in PE_CATEGORIES:
            raise ValueError(f"llm_valuation_grade '{v}' not in {PE_CATEGORIES}")
        return v

    @field_validator("llm_risk_grade")
    @classmethod
    def validate_llm_risk_grade(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in RISK_RATING_CATEGORIES:
            raise ValueError(f"llm_risk_grade '{v}' not in {RISK_RATING_CATEGORIES}")
        return v

    @field_validator("valuation_source")
    @classmethod
    def validate_valuation_source(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("llm", "estimated"):
            raise ValueError(f"valuation_source must be 'llm' or 'estimated', got '{v}'")
        return v

    @field_validator("catalyst_date")
    @classmethod
    def validate_catalyst_date(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            import re
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
                raise ValueError(f"catalyst_date must be YYYY-MM-DD, got '{v}'")
        return v

    @field_validator("catalyst_type")
    @classmethod
    def validate_catalyst_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in CATALYST_TYPES:
            raise ValueError(f"catalyst_type '{v}' not in {CATALYST_TYPES}")
        return v

    @field_validator("catalyst_source")
    @classmethod
    def validate_catalyst_source(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("llm", "template"):
            raise ValueError(f"catalyst_source must be 'llm' or 'template', got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_tier_label_matches(self) -> "RatedStock":
        expected = TIER_LABELS.get(self.tier)
        if self.tier_label != expected:
            raise ValueError(
                f"tier_label '{self.tier_label}' doesn't match tier {self.tier} "
                f"(expected '{expected}')"
            )
        return self

    @model_validator(mode="after")
    def validate_keep_logic(self) -> "RatedStock":
        expected = is_ibd_keep_candidate(self.composite_rating, self.rs_rating)
        if self.is_ibd_keep != expected:
            raise ValueError(
                f"is_ibd_keep={self.is_ibd_keep} but Comp={self.composite_rating}, "
                f"RS={self.rs_rating} -> expected {expected}"
            )
        return self


class UnratedStock(BaseModel):
    """A stock that cannot be tier-assigned due to missing ratings."""

    symbol: str = Field(..., min_length=1, max_length=10)
    company_name: str = Field(..., min_length=1)
    sector: Optional[str] = None
    reason_unrated: str = Field(..., min_length=1)
    schwab_themes: List[str] = Field(default_factory=list)
    validation_score: int = Field(0, ge=0)
    sources: List[str] = Field(default_factory=list)
    note: str = Field(..., min_length=1, description="What PM should know")

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()


class SectorRank(BaseModel):
    """Sector scored and ranked by Framework v4.0 formula."""

    rank: int = Field(..., ge=1, description="1 = strongest")
    sector: str = Field(...)
    stock_count: int = Field(..., ge=0)
    avg_composite: float = Field(..., ge=1.0, le=99.0)
    avg_rs: float = Field(..., ge=1.0, le=99.0)
    elite_pct: float = Field(..., ge=0.0, le=100.0)
    multi_list_pct: float = Field(..., ge=0.0, le=100.0)
    sector_score: float = Field(..., ge=0.0, description="Formula result")
    top_stocks: List[str] = Field(..., min_length=1, max_length=5)
    ibd_keep_count: int = Field(0, ge=0)
    avg_pe: Optional[float] = Field(None, ge=0.0, description="Avg estimated P/E for sector")
    avg_beta: Optional[float] = Field(None, ge=0.0, description="Avg estimated beta for sector")
    avg_volatility: Optional[float] = Field(None, ge=0.0, description="Avg estimated volatility %")


class IBDKeep(BaseModel):
    """A stock meeting IBD Keep threshold: Comp>=93 AND RS>=90."""

    symbol: str = Field(..., min_length=1, max_length=10)
    composite_rating: int = Field(..., ge=93)
    rs_rating: int = Field(..., ge=90)
    eps_rating: int = Field(..., ge=1, le=99)
    ibd_lists: List[str] = Field(default_factory=list)
    tier: int = Field(..., description="1, 2, or 3")
    keep_rationale: str = Field(..., min_length=1)
    override_risk: Optional[str] = Field(None, description="Any reason to question")

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("tier")
    @classmethod
    def validate_tier(cls, v: int) -> int:
        if v not in (1, 2, 3):
            raise ValueError(f"Tier must be 1, 2, or 3. Got {v}")
        return v


# ---------------------------------------------------------------------------
# Valuation Summary Models
# ---------------------------------------------------------------------------

class MarketContext(BaseModel):
    """Market valuation context."""
    sp500_forward_pe: float
    sp500_10y_avg_pe: float
    current_premium_pct: float
    risk_free_rate: float
    market_return_assumption: float


class PEDistribution(BaseModel):
    """P/E distribution across valuation categories."""
    deep_value_count: int = Field(0, ge=0)
    value_count: int = Field(0, ge=0)
    reasonable_count: int = Field(0, ge=0)
    growth_premium_count: int = Field(0, ge=0)
    speculative_count: int = Field(0, ge=0)
    total: int = Field(0, ge=0)


class TierPEStats(BaseModel):
    """P/E statistics for a single tier."""
    tier: int = Field(..., ge=1, le=3)
    avg_pe: float = Field(..., ge=0.0)
    median_pe: float = Field(..., ge=0.0)
    min_pe: float = Field(..., ge=0.0)
    max_pe: float = Field(..., ge=0.0)
    stock_count: int = Field(..., ge=0)


class SectorPEEntry(BaseModel):
    """Sector P/E summary entry."""
    sector: str
    avg_pe: float = Field(..., ge=0.0)
    stock_count: int = Field(..., ge=0)


class PEGAnalysis(BaseModel):
    """PEG ratio analysis summary."""
    avg_peg: Optional[float] = None
    median_peg: Optional[float] = None
    undervalued_count: int = Field(0, ge=0)
    fair_value_count: int = Field(0, ge=0)
    expensive_count: int = Field(0, ge=0)
    pct_undervalued: float = Field(0.0, ge=0.0, le=100.0)


class ValuationSummary(BaseModel):
    """Complete valuation analysis summary."""
    market_context: MarketContext
    pe_distribution: PEDistribution
    tier_pe_stats: List[TierPEStats] = Field(default_factory=list)
    top_sectors_by_pe: List[SectorPEEntry] = Field(default_factory=list)
    peg_analysis: PEGAnalysis
    avg_sharpe: Optional[float] = None
    avg_beta: Optional[float] = None
    avg_alpha: Optional[float] = None


# ---------------------------------------------------------------------------
# ETF-level Models
# ---------------------------------------------------------------------------

class ETFTierDistribution(BaseModel):
    """Distribution of ETFs across tiers."""

    tier_1_count: int = Field(0, ge=0)
    tier_2_count: int = Field(0, ge=0)
    tier_3_count: int = Field(0, ge=0)
    total_rated: int = Field(0, ge=0)

class RatedETF(BaseModel):
    """An ETF with tier assignment and analyst assessment."""

    symbol: str = Field(..., min_length=1, max_length=10)
    name: str = Field(..., min_length=1)
    tier: int = Field(..., ge=1, le=3)
    rs_rating: Optional[int] = Field(None, ge=1, le=99)
    acc_dis_rating: Optional[str] = Field(None)
    ytd_change: Optional[float] = Field(None)
    volume_pct_change: Optional[float] = Field(None)
    price_change: Optional[float] = Field(None)
    close_price: Optional[float] = Field(None)
    div_yield: Optional[float] = Field(None)
    etf_score: float = Field(..., description="Composite ETF ranking score")
    etf_rank: int = Field(..., ge=1, description="Rank among all rated ETFs")
    theme_tags: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)

    # Conviction and screening (ETF parity with stocks)
    conviction: int = Field(1, ge=1, le=10, description="ETF conviction score 1-10")
    focus: Optional[str] = Field(None, description="ETF focus area, e.g. 'Semiconductors'")
    focus_group: Optional[str] = Field(None, description="Grouping key for ranking")
    focus_rank: Optional[int] = Field(None, ge=1, description="Rank within focus group")
    passes_rs_screen: Optional[bool] = Field(None, description="RS >= 70")
    passes_acc_dis_screen: Optional[bool] = Field(None, description="Acc/Dis A/B/B-")
    passes_etf_screen: bool = Field(False, description="Both screening gates pass")

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("acc_dis_rating")
    @classmethod
    def validate_acc_dis(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_ACC_DIS_RATINGS:
            raise ValueError(f"acc_dis_rating must be A/B/B-/C/D/E, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Top-level Output
# ---------------------------------------------------------------------------

class AnalystOutput(BaseModel):
    """Top-level output contract for the Analyst Agent."""

    rated_stocks: List[RatedStock] = Field(
        ..., min_length=1,
        description="All tiered stocks with full analysis"
    )
    unrated_stocks: List[UnratedStock] = Field(
        default_factory=list, description="Stocks missing ratings"
    )
    sector_rankings: List[SectorRank] = Field(
        ..., min_length=1, description="All sectors scored and ranked"
    )
    ibd_keeps: List[IBDKeep] = Field(
        default_factory=list, description="Comp>=93 AND RS>=90"
    )
    elite_filter_summary: EliteFilterSummary = Field(...)
    tier_distribution: TierDistribution = Field(...)
    methodology_notes: str = Field(..., min_length=1)
    analysis_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD"
    )
    summary: str = Field(..., min_length=50)
    rated_etfs: List[RatedETF] = Field(
        default_factory=list, description="ETFs with tier assignment and scoring"
    )
    etf_tier_distribution: Optional[ETFTierDistribution] = Field(
        None, description="ETF distribution across tiers"
    )
    valuation_summary: Optional[ValuationSummary] = Field(
        None, description="Valuation and risk metrics summary"
    )

    @model_validator(mode="after")
    def validate_sector_ranks_unique(self) -> "AnalystOutput":
        ranks = [sr.rank for sr in self.sector_rankings]
        if len(ranks) != len(set(ranks)):
            raise ValueError(f"Sector ranks must be unique. Got duplicates in {ranks}")
        return self
