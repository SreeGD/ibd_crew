"""
Agent 05: Portfolio Manager — Output Schema
IBD Momentum Investment Framework v4.0

Output contract for the Portfolio Manager.
Constructs a 3-tier portfolio with position sizing, trailing stops,
keep placement, and implementation timeline.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Position sizing limits by tier (spec §4.2)
STOCK_LIMITS: dict[int, float] = {1: 5.0, 2: 4.0, 3: 3.0}
ETF_LIMITS: dict[int, float] = {1: 8.0, 2: 8.0, 3: 6.0}

# Initial trailing stop percentages by tier (spec §6.1)
TRAILING_STOPS: dict[int, int] = {1: 22, 2: 18, 3: 12}

# Trailing stop tightening protocol (spec §4.3)
STOP_TIGHTENING: dict[int, dict] = {
    1: {"initial": 22, "after_10pct_gain": (15, 18), "after_20pct_gain": (10, 12)},
    2: {"initial": 18, "after_10pct_gain": (12, 15), "after_20pct_gain": (8, 10)},
    3: {"initial": 12, "after_10pct_gain": (8, 10), "after_20pct_gain": (5, 8)},
}

# Maximum loss per position as % of total portfolio (spec §4.4)
MAX_LOSS: dict[str, dict[int, float]] = {
    "stock": {1: 1.10, 2: 0.72, 3: 0.36},
    "etf": {1: 1.76, 2: 1.44, 3: 0.72},
}

# Three keep categories (spec §4.6)
FUNDAMENTAL_KEEPS: list[str] = ["UNH", "MU", "AMZN", "MRK", "COP"]
IBD_KEEPS: list[str] = ["CLS", "AGI", "TFPM", "GMED", "SCCO", "HWM", "KLAC", "TPR", "GS"]
ALL_KEEPS: list[str] = FUNDAMENTAL_KEEPS + IBD_KEEPS  # 14 total

# Keep stock metadata: symbol → {type, default_tier, reason}
KEEP_METADATA: dict[str, dict] = {
    # Fundamental Review (5) — retained for value/forward earnings
    "UNH":  {"category": "fundamental", "asset_type": "stock", "default_tier": 3, "sector": "MEDICAL", "reason": "PE 17.9 vs 22 historical; defensive quality"},
    "MU":   {"category": "fundamental", "asset_type": "stock", "default_tier": 1, "sector": "CHIPS", "reason": "Forward PE 12 vs trailing 32; AI memory leader"},
    "AMZN": {"category": "fundamental", "asset_type": "stock", "default_tier": 1, "sector": "RETAIL", "reason": "Forward PE 29; AWS cloud leader"},
    "MRK":  {"category": "fundamental", "asset_type": "stock", "default_tier": 3, "sector": "MEDICAL", "reason": "PE 74% below 10-yr avg; Keytruda"},
    "COP":  {"category": "fundamental", "asset_type": "stock", "default_tier": 2, "sector": "ENERGY", "reason": "PE 13.5 vs sector 16.4; energy quality"},
    # IBD High-Rated (9) — Composite >= 93 AND RS >= 90
    "CLS":  {"category": "ibd", "asset_type": "stock", "default_tier": 1, "sector": "ELECTRONICS", "reason": "Comp 99, RS 97; elite momentum"},
    "AGI":  {"category": "ibd", "asset_type": "stock", "default_tier": 1, "sector": "MINING", "reason": "Comp 99, RS 92; gold momentum"},
    "TFPM": {"category": "ibd", "asset_type": "stock", "default_tier": 1, "sector": "MINING", "reason": "Comp 99, RS 92; precious metals"},
    "GMED": {"category": "ibd", "asset_type": "stock", "default_tier": 2, "sector": "MEDICAL", "reason": "Comp 98, RS 91; medical device"},
    "SCCO": {"category": "ibd", "asset_type": "stock", "default_tier": 2, "sector": "MINING", "reason": "Comp 98, RS 94; copper mining"},
    "HWM":  {"category": "ibd", "asset_type": "stock", "default_tier": 2, "sector": "AEROSPACE", "reason": "Comp 98, RS 90; aerospace"},
    "KLAC": {"category": "ibd", "asset_type": "stock", "default_tier": 1, "sector": "CHIPS", "reason": "Comp 97, RS 94; chip equipment"},
    "TPR":  {"category": "ibd", "asset_type": "stock", "default_tier": 2, "sector": "CONSUMER", "reason": "Comp 97, RS 93; consumer"},
    "GS":   {"category": "ibd", "asset_type": "stock", "default_tier": 2, "sector": "FINANCE", "reason": "Comp 93, RS 91; investment bank"},
}

# Diversification targets (spec §4.5)
STOCK_ETF_SPLIT: tuple[int, int] = (50, 50)  # target %, ±5% tolerance

# Tier allocation target ranges (same as strategy constants)
TIER_RANGES: dict[int, tuple[float, float]] = {
    1: (35.0, 45.0),
    2: (33.0, 40.0),
    3: (20.0, 30.0),
}
CASH_RANGE: tuple[float, float] = (2.0, 15.0)

# Order action types
VALID_ORDER_ACTIONS = ("BUY", "SELL", "ADD", "TRIM")

# Implementation week focuses
WEEK_FOCUSES: dict[int, str] = {
    1: "Liquidation",
    2: "T1 Momentum",
    3: "T2 Quality Growth",
    4: "T3 Defensive",
}


# ---------------------------------------------------------------------------
# Supporting Models
# ---------------------------------------------------------------------------

class PortfolioPosition(BaseModel):
    """A single position in the portfolio."""

    symbol: str = Field(..., min_length=1, max_length=10)
    company_name: str = Field(..., min_length=1)
    sector: str = Field(...)
    cap_size: Optional[str] = Field(None, description="Market cap: large/mid/small")
    tier: int = Field(..., ge=1, le=3)
    asset_type: Literal["stock", "etf"] = Field(...)
    target_pct: float = Field(..., ge=0.0, description="% of total portfolio")
    trailing_stop_pct: float = Field(..., ge=0.0, le=100.0)
    max_loss_pct: float = Field(..., ge=0.0, description="Max portfolio loss %")
    keep_category: Optional[str] = Field(
        None, description="fundamental / ibd / None"
    )
    conviction: int = Field(..., ge=1, le=10)
    reasoning: str = Field(..., min_length=10)
    volatility_adjustment: float = Field(0.0, ge=-2.0, le=2.0, description="LLM vol-based size adjustment %")
    sizing_source: Optional[str] = Field(None, description="'llm' or 'deterministic'")
    selection_source: Optional[str] = Field(None, description="'momentum'/'value'/'pattern'/'keep'")

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("keep_category")
    @classmethod
    def validate_keep_category(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("fundamental", "ibd"):
            raise ValueError(f"keep_category must be fundamental/ibd, got '{v}'")
        return v

    @field_validator("sizing_source")
    @classmethod
    def validate_sizing_source(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("llm", "deterministic"):
            raise ValueError(f"sizing_source must be 'llm' or 'deterministic', got '{v}'")
        return v

    @field_validator("selection_source")
    @classmethod
    def validate_selection_source(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("momentum", "value", "pattern", "keep"):
            raise ValueError(f"selection_source must be momentum/value/pattern/keep, got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_position_limit(self) -> "PortfolioPosition":
        """Position does not exceed tier-specific limit."""
        limit = (
            ETF_LIMITS[self.tier]
            if self.asset_type == "etf"
            else STOCK_LIMITS[self.tier]
        )
        if self.target_pct > limit + 0.01:  # tiny tolerance for float math
            raise ValueError(
                f"{self.symbol}: {self.asset_type} in T{self.tier} at "
                f"{self.target_pct:.2f}% exceeds max {limit}%"
            )
        return self

    @model_validator(mode="after")
    def validate_stop_matches_tier(self) -> "PortfolioPosition":
        """Trailing stop matches tier (initial or tightened)."""
        initial = TRAILING_STOPS[self.tier]
        # Allow initial stop or any tightened value down to the after_20pct minimum
        tight_min = STOP_TIGHTENING[self.tier]["after_20pct_gain"][0]
        if not (tight_min <= self.trailing_stop_pct <= initial + 0.01):
            raise ValueError(
                f"{self.symbol}: T{self.tier} stop {self.trailing_stop_pct}% "
                f"not in [{tight_min}, {initial}]"
            )
        return self


class TierPortfolio(BaseModel):
    """One tier of the 3-tier portfolio."""

    tier: int = Field(..., ge=1, le=3)
    label: str = Field(...)
    target_pct: float = Field(..., ge=0.0, le=100.0)
    actual_pct: float = Field(..., ge=0.0, le=100.0)
    positions: List[PortfolioPosition] = Field(default_factory=list)
    stock_count: int = Field(..., ge=0)
    etf_count: int = Field(..., ge=0)

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        valid = ("Momentum", "Quality Growth", "Defensive")
        if v not in valid:
            raise ValueError(f"label must be one of {valid}, got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_counts(self) -> "TierPortfolio":
        """Stock and ETF counts match positions."""
        stocks = sum(1 for p in self.positions if p.asset_type == "stock")
        etfs = sum(1 for p in self.positions if p.asset_type == "etf")
        if stocks != self.stock_count:
            raise ValueError(
                f"T{self.tier}: stock_count={self.stock_count} but "
                f"found {stocks} stocks in positions"
            )
        if etfs != self.etf_count:
            raise ValueError(
                f"T{self.tier}: etf_count={self.etf_count} but "
                f"found {etfs} ETFs in positions"
            )
        return self


class KeepDetail(BaseModel):
    """Detail for one keep position."""

    symbol: str = Field(..., min_length=1, max_length=10)
    category: Literal["fundamental", "ibd"] = Field(...)
    tier_placed: int = Field(..., ge=1, le=3)
    target_pct: float = Field(..., ge=0.0)
    reason: str = Field(..., min_length=10)

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()


class KeepsPlacement(BaseModel):
    """Placement of all 14 pre-committed keep positions."""

    fundamental_keeps: List[KeepDetail] = Field(...)
    ibd_keeps: List[KeepDetail] = Field(...)
    total_keeps: int = Field(...)
    keeps_pct: float = Field(..., ge=0.0, le=100.0, description="% of portfolio in keeps")

    @model_validator(mode="after")
    def validate_total(self) -> "KeepsPlacement":
        """Total keeps must equal sum of categories and must be 14."""
        actual = (
            len(self.fundamental_keeps)
            + len(self.ibd_keeps)
        )
        if actual != self.total_keeps:
            raise ValueError(
                f"total_keeps={self.total_keeps} but found {actual} keep details"
            )
        if self.total_keeps != len(ALL_KEEPS):
            raise ValueError(
                f"Expected {len(ALL_KEEPS)} keeps, got {self.total_keeps}"
            )
        return self

    @model_validator(mode="after")
    def validate_categories(self) -> "KeepsPlacement":
        """Correct count per category."""
        if len(self.fundamental_keeps) != len(FUNDAMENTAL_KEEPS):
            raise ValueError(
                f"Expected {len(FUNDAMENTAL_KEEPS)} fundamental keeps, "
                f"got {len(self.fundamental_keeps)}"
            )
        if len(self.ibd_keeps) != len(IBD_KEEPS):
            raise ValueError(
                f"Expected {len(IBD_KEEPS)} IBD keeps, "
                f"got {len(self.ibd_keeps)}"
            )
        return self


class OrderAction(BaseModel):
    """A portfolio order (buy/sell/add/trim)."""

    symbol: str = Field(..., min_length=1, max_length=10)
    action: str = Field(...)
    tier: int = Field(..., ge=1, le=3)
    target_pct: float = Field(..., ge=0.0)
    rationale: str = Field(..., min_length=10)

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        if v not in VALID_ORDER_ACTIONS:
            raise ValueError(f"action must be one of {VALID_ORDER_ACTIONS}, got '{v}'")
        return v


class ImplementationWeek(BaseModel):
    """One week of the 4-week implementation plan."""

    week: int = Field(..., ge=1, le=4)
    focus: str = Field(..., min_length=1)
    actions: List[str] = Field(default_factory=list, description="Action descriptions")


# ---------------------------------------------------------------------------
# Top-level Output
# ---------------------------------------------------------------------------

class PortfolioOutput(BaseModel):
    """Top-level output contract for the Portfolio Manager."""

    tier_1: TierPortfolio = Field(...)
    tier_2: TierPortfolio = Field(...)
    tier_3: TierPortfolio = Field(...)
    cash_pct: float = Field(..., ge=2.0, le=15.0)
    keeps_placement: KeepsPlacement = Field(...)
    orders: List[OrderAction] = Field(default_factory=list)
    implementation_plan: List[ImplementationWeek] = Field(
        ..., min_length=4, max_length=4
    )
    sector_exposure: Dict[str, float] = Field(
        ..., description="Sector → % of total portfolio"
    )
    total_positions: int = Field(..., ge=1)
    stock_count: int = Field(..., ge=0)
    etf_count: int = Field(..., ge=0)
    construction_methodology: str = Field(..., min_length=20)
    deviation_notes: List[str] = Field(default_factory=list)
    analysis_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD"
    )
    summary: str = Field(..., min_length=50)

    @model_validator(mode="after")
    def validate_tier_allocation_sum(self) -> "PortfolioOutput":
        """T1 + T2 + T3 + Cash should sum to 95-105%."""
        total = (
            self.tier_1.actual_pct
            + self.tier_2.actual_pct
            + self.tier_3.actual_pct
            + self.cash_pct
        )
        if not (95.0 <= total <= 105.0):
            raise ValueError(
                f"Tier allocations sum to {total:.1f}%, expected 95-105%"
            )
        return self

    @model_validator(mode="after")
    def validate_position_counts(self) -> "PortfolioOutput":
        """Position counts match actual positions."""
        actual_total = (
            len(self.tier_1.positions)
            + len(self.tier_2.positions)
            + len(self.tier_3.positions)
        )
        if actual_total != self.total_positions:
            raise ValueError(
                f"total_positions={self.total_positions} but "
                f"found {actual_total} positions across tiers"
            )
        actual_stocks = self.tier_1.stock_count + self.tier_2.stock_count + self.tier_3.stock_count
        actual_etfs = self.tier_1.etf_count + self.tier_2.etf_count + self.tier_3.etf_count
        if actual_stocks != self.stock_count:
            raise ValueError(
                f"stock_count={self.stock_count} but found {actual_stocks} across tiers"
            )
        if actual_etfs != self.etf_count:
            raise ValueError(
                f"etf_count={self.etf_count} but found {actual_etfs} across tiers"
            )
        return self

    @model_validator(mode="after")
    def validate_min_sectors(self) -> "PortfolioOutput":
        """At least 8 sectors in sector exposure."""
        if len(self.sector_exposure) < 8:
            raise ValueError(
                f"Only {len(self.sector_exposure)} sectors, minimum is 8"
            )
        return self

    @model_validator(mode="after")
    def validate_no_sector_exceeds_max(self) -> "PortfolioOutput":
        """No single sector > 40%."""
        for sector, pct in self.sector_exposure.items():
            if pct > 40.0 + 0.01:
                raise ValueError(
                    f"Sector '{sector}' at {pct:.1f}% exceeds max 40%"
                )
        return self

    @model_validator(mode="after")
    def validate_stock_etf_ratio(self) -> "PortfolioOutput":
        """Stock/ETF split within 35-65% when both types present."""
        total = self.stock_count + self.etf_count
        if total == 0:
            return self
        # Allow all-stock or all-ETF portfolios when the other type is unavailable
        if self.stock_count == 0 or self.etf_count == 0:
            return self
        stock_pct = (self.stock_count / total) * 100
        if not (35.0 <= stock_pct <= 65.0):
            raise ValueError(
                f"Stock ratio {stock_pct:.1f}% not within 35-65% "
                f"(stocks={self.stock_count}, etfs={self.etf_count})"
            )
        return self

    @model_validator(mode="after")
    def validate_implementation_weeks(self) -> "PortfolioOutput":
        """Implementation plan has weeks 1-4."""
        weeks = sorted(w.week for w in self.implementation_plan)
        if weeks != [1, 2, 3, 4]:
            raise ValueError(f"Implementation plan weeks must be [1,2,3,4], got {weeks}")
        return self
