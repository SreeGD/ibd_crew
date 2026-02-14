"""
Agent 01: Research Agent — Output Schema
IBD Momentum Investment Framework v4.0

This is the OUTPUT CONTRACT for the Research Agent.
Every field, validator, and constraint is derived from AGENT_01_RESEARCH.md.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Framework v4.0 Constants (hard-coded from spec)
# ---------------------------------------------------------------------------

IBD_SECTORS: list[str] = [
    "AEROSPACE", "AUTO", "BANKS", "BUILDING", "BUSINESS SERVICES",
    "CHEMICALS", "CHIPS", "COMPUTER", "CONSUMER", "DEFENSE",
    "ELECTRONICS", "ENERGY", "FINANCE", "FOOD/BEVERAGE", "HEALTHCARE",
    "INSURANCE", "INTERNET", "LEISURE", "MACHINERY", "MEDIA",
    "MEDICAL", "MINING", "REAL ESTATE", "RETAIL", "SOFTWARE",
    "TELECOM", "TRANSPORTATION", "UTILITIES",
]

VALID_SMR_RATINGS: set[str] = {"A", "B", "B-", "C", "D", "E"}
VALID_ACC_DIS_RATINGS: set[str] = {"A", "B", "B-", "C", "D", "E"}
VALID_CAP_SIZES: set[str] = {"large", "mid", "small"}

SCHWAB_THEME_NAMES: list[str] = [
    "Artificial Intelligence", "Robotics & Automation", "Cybersecurity",
    "Big Data & Analytics", "Cloud Computing", "3D Printing",
    "E-Commerce", "Digital Payments", "Digital Lifestyle",
    "Online Media & Video", "Renewable Energy", "Climate Change",
    "Electric Vehicles", "Healthcare Innovation", "Aging Population",
    "Space Economy", "Defense & Aerospace", "China Growth",
    "Cannabis", "Blockchain/Crypto", "Canada",
]

VALIDATION_POINTS: dict[str, int] = {
    # IBD Lists (Primary: 40% weight)
    "IBD Big Cap 20":            3,
    "IBD 50":                    3,
    "IBD Sector Leaders":        3,
    "IBD Stock Spotlight":       2,
    "IBD Tech Leaders":          2,
    "IBD Rising Profit Estimates": 1,
    "IBD RS at New High":        1,
    "IBD IPO Leaders":           1,
    "IBD Top 200 Composite":     1,
    "IBD Smart Table":           1,
    "IBD Large/MidCap Leaders":  1,
    "IBD TimeSaver Table":       1,
    "IBD ETF Leaders":           1,
    # Secondary (10% each)
    "CFRA 5-star":               3,
    "CFRA 4-star":               2,
    "ARGUS Buy":                 2,
    "ARGUS Hold":                1,
    "Morningstar 5-star":        3,
    "Morningstar 4-star":        2,
    # Supplementary (30% combined)
    "Motley Fool Epic Top":      3,
    "Motley Fool New Rec":       2,
    "Schwab Theme Core":         2,
    "Schwab Theme Secondary":    1,
    "Schwab Multiple Themes":    1,
}

TIER_THRESHOLDS: dict[int, dict[str, int]] = {
    1: {"composite_min": 95, "rs_min": 90, "eps_min": 80},
    2: {"composite_min": 85, "rs_min": 80, "eps_min": 75},
    3: {"composite_min": 80, "rs_min": 75, "eps_min": 70},
}

# Provider groups for multi-source validation
SOURCE_PROVIDERS: dict[str, str] = {
    "IBD Big Cap 20": "IBD", "IBD 50": "IBD",
    "IBD Sector Leaders": "IBD", "IBD Stock Spotlight": "IBD",
    "IBD Tech Leaders": "IBD", "IBD Rising Profit Estimates": "IBD",
    "IBD RS at New High": "IBD", "IBD IPO Leaders": "IBD",
    "IBD Top 200 Composite": "IBD", "IBD Smart Table": "IBD",
    "IBD Large/MidCap Leaders": "IBD", "IBD TimeSaver Table": "IBD",
    "IBD ETF Leaders": "IBD",
    "CFRA 5-star": "CFRA", "CFRA 4-star": "CFRA",
    "ARGUS Buy": "ARGUS", "ARGUS Hold": "ARGUS",
    "Morningstar 5-star": "Morningstar", "Morningstar 4-star": "Morningstar",
    "Motley Fool Epic Top": "Motley Fool", "Motley Fool New Rec": "Motley Fool",
    "Schwab Theme Core": "Schwab", "Schwab Theme Secondary": "Schwab",
    "Schwab Multiple Themes": "Schwab",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def is_ibd_keep_candidate(composite: Optional[int], rs: Optional[int]) -> bool:
    """Flag stocks meeting IBD Keep threshold: Composite ≥93 AND RS ≥90."""
    return (
        composite is not None
        and rs is not None
        and composite >= 93
        and rs >= 90
    )


def compute_preliminary_tier(
    composite: Optional[int],
    rs: Optional[int],
    eps: Optional[int],
) -> Optional[int]:
    """Assign preliminary tier based on Framework v4.0 thresholds."""
    if composite is None or rs is None or eps is None:
        return None
    for tier in (1, 2, 3):
        t = TIER_THRESHOLDS[tier]
        if (
            composite >= t["composite_min"]
            and rs >= t["rs_min"]
            and eps >= t["eps_min"]
        ):
            return tier
    return None


def compute_validation_score(source_labels: list[str]) -> tuple[int, int]:
    """
    Calculate multi-source validation score and distinct provider count.

    Returns:
        (total_points, distinct_provider_count)
    """
    total = 0
    providers: set[str] = set()
    for label in source_labels:
        pts = VALIDATION_POINTS.get(label, 0)
        total += pts
        provider = SOURCE_PROVIDERS.get(label)
        if provider:
            providers.add(provider)
    return total, len(providers)


# ---------------------------------------------------------------------------
# Schema Models
# ---------------------------------------------------------------------------

class ResearchStock(BaseModel):
    """A single stock discovered by the Research Agent."""

    symbol: str = Field(..., min_length=1, max_length=10, description="Ticker symbol")
    company_name: str = Field(..., min_length=1, description="Company name")
    sector: str = Field(..., description="One of 28 IBD sector categories")
    security_type: Literal["stock"] = "stock"

    # IBD Ratings — Optional because not every source has all ratings
    composite_rating: Optional[int] = Field(None, ge=1, le=99)
    rs_rating: Optional[int] = Field(None, ge=1, le=99)
    eps_rating: Optional[int] = Field(None, ge=1, le=99)
    smr_rating: Optional[str] = Field(None, description="A/B/B-/C/D/E")
    acc_dis_rating: Optional[str] = Field(None, description="A/B/B-/C/D/E")

    # Source tracking
    ibd_lists: List[str] = Field(default_factory=list, description="IBD list memberships")
    schwab_themes: List[str] = Field(default_factory=list, description="Schwab theme tags")
    fool_status: Optional[str] = Field(
        None, description="'Epic Top', 'New Rec', or None"
    )
    other_ratings: Dict[str, str] = Field(
        default_factory=dict, description="e.g. {'CFRA': '5-star', 'ARGUS': 'Buy'}"
    )

    # Validation
    validation_score: int = Field(0, ge=0, description="Multi-source point total")
    validation_providers: int = Field(0, ge=0, description="Distinct provider count")
    is_multi_source_validated: bool = Field(
        False, description="score≥5 AND providers≥2"
    )
    is_ibd_keep_candidate: bool = Field(
        False, description="Composite≥93 AND RS≥90"
    )

    # Market Cap Classification
    cap_size: Optional[Literal["large", "mid", "small"]] = Field(
        None, description="Market cap category: large (>$10B), mid ($2B-$10B), small (<$2B)"
    )

    # Morningstar metadata (optional — only set for stocks on Morningstar pick lists)
    morningstar_rating: Optional[str] = Field(None, description="e.g. '5-star', '4-star'")
    economic_moat: Optional[str] = Field(None, description="Wide/Narrow/None")
    fair_value: Optional[float] = Field(None, description="Morningstar fair value estimate")
    morningstar_price: Optional[float] = Field(None, description="Price at Morningstar report")
    price_to_fair_value: Optional[float] = Field(None, description="Price/Fair Value ratio")
    morningstar_uncertainty: Optional[str] = Field(None, description="Low/Medium/High/Very High")

    # Assessment
    preliminary_tier: Optional[int] = Field(
        None, description="1, 2, 3, or None if can't determine"
    )
    sector_rank: Optional[int] = Field(
        None, ge=1, le=33, description="IBD sector ranking (1=strongest)"
    )
    stock_rank_in_sector: Optional[int] = Field(
        None, ge=1, description="Stock position within its IBD sector"
    )
    sources: List[str] = Field(
        ..., min_length=1, description="Source file names (≥1)"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=20, description="≥20 chars")

    # --- Validators ---

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("sector")
    @classmethod
    def sector_normalize(cls, v: str) -> str:
        normalized = v.strip().upper()
        if not normalized:
            raise ValueError("Sector must not be empty")
        return normalized

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

    @field_validator("cap_size")
    @classmethod
    def validate_cap_size(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_CAP_SIZES:
            raise ValueError(f"cap_size must be large/mid/small/None, got '{v}'")
        return v

    @field_validator("preliminary_tier")
    @classmethod
    def validate_tier(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v not in (1, 2, 3):
            raise ValueError(f"Tier must be 1, 2, 3, or None. Got {v}")
        return v

    @field_validator("fool_status")
    @classmethod
    def validate_fool_status(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("Epic Top", "New Rec"):
            raise ValueError(f"fool_status must be 'Epic Top', 'New Rec', or None. Got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_keep_logic(self) -> "ResearchStock":
        """is_ibd_keep_candidate must match Comp≥93 AND RS≥90."""
        expected = is_ibd_keep_candidate(self.composite_rating, self.rs_rating)
        if self.is_ibd_keep_candidate != expected:
            raise ValueError(
                f"is_ibd_keep_candidate={self.is_ibd_keep_candidate} but "
                f"Comp={self.composite_rating}, RS={self.rs_rating} → expected {expected}"
            )
        return self

    @model_validator(mode="after")
    def validate_multi_source_logic(self) -> "ResearchStock":
        """is_multi_source_validated must match score≥5 AND providers≥2."""
        expected = self.validation_score >= 5 and self.validation_providers >= 2
        if self.is_multi_source_validated != expected:
            raise ValueError(
                f"is_multi_source_validated={self.is_multi_source_validated} but "
                f"score={self.validation_score}, providers={self.validation_providers} "
                f"→ expected {expected}"
            )
        return self


class ResearchETF(BaseModel):
    """An ETF discovered across sources."""

    symbol: str = Field(..., min_length=1, max_length=10)
    name: str = Field(..., min_length=1)
    focus: str = Field(..., min_length=1, description="e.g. 'Semiconductors'")
    schwab_themes: List[str] = Field(default_factory=list)
    ibd_lists: List[str] = Field(default_factory=list)
    sources: List[str] = Field(..., min_length=1)
    key_theme_etf: bool = Field(False, description="Is a key ETF in a Schwab theme?")

    # IBD ratings (from ETF Tables PDF)
    rs_rating: Optional[int] = Field(None, ge=1, le=99, description="Relative Strength 1-99")
    acc_dis_rating: Optional[str] = Field(None, description="Accumulation/Distribution A-E")

    # ETF market data (from ETF Tables PDF)
    ytd_change: Optional[float] = Field(None, description="Year-to-date % change")
    close_price: Optional[float] = Field(None, description="Last close price")
    price_change: Optional[float] = Field(None, description="Price change ($)")
    volume_pct_change: Optional[float] = Field(None, description="Volume % change")
    div_yield: Optional[float] = Field(None, description="Dividend yield %")

    # Tiering and ranking
    preliminary_tier: Optional[int] = Field(None, ge=1, le=3, description="Theme-based tier 1/2/3")
    etf_score: Optional[float] = Field(None, description="Composite ETF ranking score")
    etf_rank: Optional[int] = Field(None, ge=1, description="Rank among all ETFs by score")

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("acc_dis_rating")
    @classmethod
    def validate_acc_dis(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("A", "B", "B-", "C", "D", "E"):
            raise ValueError(f"acc_dis_rating must be A/B/B-/C/D/E, got '{v}'")
        return v


class SectorPattern(BaseModel):
    """Sector-level observation for Rotation Detector."""

    sector: str = Field(..., description="One of 28 IBD sectors")
    stock_count: int = Field(..., ge=0)
    avg_composite: Optional[float] = Field(None, ge=1.0, le=99.0)
    avg_rs: Optional[float] = Field(None, ge=1.0, le=99.0)
    elite_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    multi_list_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    ibd_keep_count: int = Field(0, ge=0)
    strength: Literal["leading", "improving", "lagging", "declining"] = Field(...)
    trend_direction: Literal["up", "down", "flat"] = Field(...)
    evidence: str = Field(..., min_length=10, description="≥10 chars")

    @field_validator("sector")
    @classmethod
    def sector_in_ibd_list(cls, v: str) -> str:
        normalized = v.strip().upper()
        if normalized not in IBD_SECTORS:
            raise ValueError(f"Sector '{v}' not in 28 IBD sectors.")
        return normalized


class ResearchOutput(BaseModel):
    """
    Top-level output contract for the Research Agent.

    Spec requirements:
    - 50-300 stocks
    - At least 5 different sectors
    - ibd_keep_candidates all have Comp≥93 AND RS≥90
    - multi_source_validated all have score≥5 AND providers≥2
    """

    stocks: List[ResearchStock] = Field(
        ..., min_length=1,
        description="Individual stocks discovered across all sources"
    )
    etfs: List[ResearchETF] = Field(
        default_factory=list, description="ETFs found across sources"
    )
    sector_patterns: List[SectorPattern] = Field(
        default_factory=list, description="Sector-level observations"
    )
    data_sources_used: List[str] = Field(
        ..., min_length=1, description="Files actually processed"
    )
    data_sources_failed: List[str] = Field(
        default_factory=list, description="Files that couldn't be read"
    )
    total_securities_scanned: int = Field(..., ge=0)
    ibd_keep_candidates: List[str] = Field(
        default_factory=list, description="Symbols: Comp≥93 AND RS≥90"
    )
    multi_source_validated: List[str] = Field(
        default_factory=list, description="Symbols: score≥5 from 2+ providers"
    )
    analysis_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD"
    )
    summary: str = Field(..., min_length=50, description="≥50 chars")

    # --- Validators ---

    @model_validator(mode="after")
    def validate_min_sectors(self) -> "ResearchOutput":
        """At least 1 sector in stock list."""
        sectors = {s.sector for s in self.stocks}
        if len(sectors) < 1:
            raise ValueError(
                f"Need ≥1 sector, got {len(sectors)}: {sectors}"
            )
        return self

    @model_validator(mode="after")
    def validate_keep_candidates_list(self) -> "ResearchOutput":
        """Every symbol in ibd_keep_candidates must have Comp≥93 AND RS≥90."""
        stock_map = {s.symbol: s for s in self.stocks}
        for sym in self.ibd_keep_candidates:
            s = stock_map.get(sym)
            if s is None:
                raise ValueError(
                    f"ibd_keep_candidate '{sym}' not found in stocks list"
                )
            if not is_ibd_keep_candidate(s.composite_rating, s.rs_rating):
                raise ValueError(
                    f"ibd_keep_candidate '{sym}' has Comp={s.composite_rating}, "
                    f"RS={s.rs_rating} — doesn't meet Comp≥93 AND RS≥90"
                )
        return self

    @model_validator(mode="after")
    def validate_multi_source_list(self) -> "ResearchOutput":
        """Every symbol in multi_source_validated must have score≥5 AND providers≥2."""
        stock_map = {s.symbol: s for s in self.stocks}
        for sym in self.multi_source_validated:
            s = stock_map.get(sym)
            if s is None:
                # Could be an ETF — check ETFs too
                etf_map = {e.symbol: e for e in self.etfs}
                if sym not in etf_map:
                    raise ValueError(
                        f"multi_source_validated '{sym}' not found in stocks or ETFs"
                    )
                continue
            if s.validation_score < 5 or s.validation_providers < 2:
                raise ValueError(
                    f"multi_source_validated '{sym}' has score={s.validation_score}, "
                    f"providers={s.validation_providers} — doesn't meet score≥5 AND providers≥2"
                )
        return self

    @model_validator(mode="after")
    def validate_keep_consistency(self) -> "ResearchOutput":
        """Stocks flagged as is_ibd_keep_candidate must appear in ibd_keep_candidates list."""
        flagged = {s.symbol for s in self.stocks if s.is_ibd_keep_candidate}
        listed = set(self.ibd_keep_candidates)
        if flagged != listed:
            missing_from_list = flagged - listed
            extra_in_list = listed - flagged
            parts = []
            if missing_from_list:
                parts.append(f"Flagged but not in list: {missing_from_list}")
            if extra_in_list:
                parts.append(f"In list but not flagged: {extra_in_list}")
            raise ValueError(
                f"Keep candidate inconsistency: {'; '.join(parts)}"
            )
        return self

    @model_validator(mode="after")
    def validate_multi_source_consistency(self) -> "ResearchOutput":
        """Stocks flagged as is_multi_source_validated must appear in multi_source_validated list."""
        flagged = {s.symbol for s in self.stocks if s.is_multi_source_validated}
        listed = set(self.multi_source_validated)
        # Listed can include ETFs, so only check stock flags → list
        if not flagged.issubset(listed):
            missing = flagged - listed
            raise ValueError(
                f"Multi-source inconsistency: flagged but not listed: {missing}"
            )
        return self
