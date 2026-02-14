"""
Agent 04: Sector Strategist — Output Schema
IBD Momentum Investment Framework v4.0

Output contract for the Sector Strategist.
Translates Rotation Detector verdict + Analyst sector rankings
into sector allocation recommendations for 3-tier portfolio architecture.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ibd_agents.schemas.research_output import IBD_SECTORS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Portfolio architecture targets (spec §4.1, Moderate Profile)
TIER_ALLOCATION_TARGETS: dict[str, dict] = {
    "tier_1_momentum":  {"target_pct": 39, "range": (35, 40)},
    "tier_2_quality":   {"target_pct": 37, "range": (33, 40)},
    "tier_3_defensive": {"target_pct": 22, "range": (20, 30)},
    "cash":             {"target_pct": 2,  "range": (2, 5)},
}

# Sector concentration limits (spec §4.4)
SECTOR_LIMITS: dict[str, int] = {
    "max_single_sector": 40,
    "min_sectors": 8,
}

# Regime-adjusted strategy (spec §4.6)
REGIME_ACTIONS: dict[str, dict] = {
    "bull": {
        "tier_1_bias": +5,
        "tier_3_bias": -5,
        "equity_target": "90-100%",
        "action": "Scale into T1 from IBD 50/Sector Leaders, add momentum ETFs",
    },
    "bear": {
        "tier_1_bias": -10,
        "tier_3_bias": +10,
        "equity_target": "20-40%",
        "action": "Rotate to T3 focus, raise cash, tighten T1 stops to 15-18%",
    },
    "neutral": {
        "tier_1_bias": 0,
        "tier_3_bias": 0,
        "equity_target": "70-90%",
        "action": "Maintain target allocations (39/37/22)",
    },
}

# Schwab theme → ETF mapping (spec §4.5)
THEME_ETFS: dict[str, list[str]] = {
    "Artificial Intelligence":   ["BOTZ", "ROBO", "AIQ"],
    "Robotics & Automation":     ["ROBO", "BOTZ"],
    "Cybersecurity":             ["CIBR", "HACK", "BUG"],
    "Big Data & Analytics":      ["CLOU", "WCLD"],
    "E-Commerce":                ["IBUY", "ONLN"],
    "Digital Payments":          ["IPAY", "FINX"],
    "Space Economy":             ["UFO", "ARKX"],
    "Defense & Aerospace":       ["ITA", "PPA", "XAR"],
    "Electric Vehicles":         ["DRIV", "IDRV", "LIT"],
    "China Growth":              ["MCHI", "FXI", "KWEB"],
    "Healthcare Innovation":     ["XBI", "IBB", "ARKG"],
    "Aging Population":          [],
    "Renewable Energy":          ["ICLN", "TAN", "QCLN"],
    "Climate Change":            [],
    "Cannabis":                  ["MJ", "MSOS"],
    "Blockchain/Crypto":         ["BLOK", "BITO"],
}

# Theme classification for tier-fit validation
GROWTH_THEMES: set[str] = {
    "Artificial Intelligence", "Robotics & Automation", "Cybersecurity",
    "Big Data & Analytics", "E-Commerce", "Digital Payments",
    "Space Economy", "Defense & Aerospace", "Electric Vehicles", "China Growth",
}

DEFENSIVE_THEMES: set[str] = {
    "Healthcare Innovation", "Aging Population", "Renewable Energy",
    "Climate Change", "Cannabis", "Blockchain/Crypto",
}

# Theme → related IBD sectors
THEME_SECTOR_MAP: dict[str, list[str]] = {
    "Artificial Intelligence":   ["CHIPS", "SOFTWARE", "COMPUTER", "ELECTRONICS"],
    "Robotics & Automation":     ["CHIPS", "MACHINERY", "ELECTRONICS"],
    "Cybersecurity":             ["SOFTWARE", "COMPUTER"],
    "Big Data & Analytics":      ["SOFTWARE", "INTERNET"],
    "E-Commerce":                ["INTERNET", "RETAIL"],
    "Digital Payments":          ["FINANCE", "INTERNET"],
    "Space Economy":             ["AEROSPACE", "DEFENSE"],
    "Defense & Aerospace":       ["AEROSPACE", "DEFENSE"],
    "Electric Vehicles":         ["AUTO", "ELECTRONICS", "MINING"],
    "China Growth":              ["INTERNET", "CHIPS", "CONSUMER"],
    "Healthcare Innovation":     ["MEDICAL", "HEALTHCARE"],
    "Aging Population":          ["MEDICAL", "HEALTHCARE", "INSURANCE"],
    "Renewable Energy":          ["ENERGY", "UTILITIES"],
    "Climate Change":            ["ENERGY", "UTILITIES", "CHEMICALS"],
    "Cannabis":                  ["CONSUMER", "MEDICAL"],
    "Blockchain/Crypto":         ["FINANCE", "SOFTWARE"],
}

VALID_CONVICTION_LEVELS = ("HIGH", "MEDIUM", "LOW")

# Validator ranges
TIER_TARGET_RANGES: dict[str, tuple[float, float]] = {
    "T1": (25.0, 45.0),
    "T2": (25.0, 45.0),
    "T3": (15.0, 35.0),
    "Cash": (2.0, 15.0),
}


# ---------------------------------------------------------------------------
# Supporting Models
# ---------------------------------------------------------------------------

class SectorAllocationPlan(BaseModel):
    """Sector allocation across 3-tier portfolio architecture."""

    tier_1_allocation: Dict[str, float] = Field(
        ..., description="Sector → % of T1 bucket"
    )
    tier_2_allocation: Dict[str, float] = Field(
        ..., description="Sector → % of T2 bucket"
    )
    tier_3_allocation: Dict[str, float] = Field(
        ..., description="Sector → % of T3 bucket"
    )
    overall_allocation: Dict[str, float] = Field(
        ..., description="Sector → % of total portfolio"
    )
    tier_targets: Dict[str, float] = Field(
        ..., description='{"T1": 39, "T2": 37, "T3": 22, "Cash": 2}'
    )
    cash_recommendation: float = Field(..., ge=2.0, le=15.0)
    rationale: str = Field(..., min_length=50)

    @model_validator(mode="after")
    def validate_tier_allocation_sums(self) -> "SectorAllocationPlan":
        """Each tier's sector allocations should sum to 95-105%."""
        for tier_name, alloc in [
            ("tier_1", self.tier_1_allocation),
            ("tier_2", self.tier_2_allocation),
            ("tier_3", self.tier_3_allocation),
        ]:
            if not alloc:
                continue
            total = sum(alloc.values())
            if not (95.0 <= total <= 105.0):
                raise ValueError(
                    f"{tier_name}_allocation sums to {total:.1f}%, expected 95-105%"
                )
        return self

    @model_validator(mode="after")
    def validate_no_sector_exceeds_max(self) -> "SectorAllocationPlan":
        """No single sector > 40% of total portfolio."""
        max_pct = SECTOR_LIMITS["max_single_sector"]
        for sector, pct in self.overall_allocation.items():
            if pct > max_pct:
                raise ValueError(
                    f"Sector '{sector}' at {pct:.1f}% exceeds max {max_pct}%"
                )
        return self

    @model_validator(mode="after")
    def validate_min_sectors(self) -> "SectorAllocationPlan":
        """At least 8 sectors in overall allocation."""
        min_sectors = SECTOR_LIMITS["min_sectors"]
        count = len(self.overall_allocation)
        if count < min_sectors:
            raise ValueError(
                f"Only {count} sectors in allocation, minimum is {min_sectors}"
            )
        return self

    @model_validator(mode="after")
    def validate_tier_target_ranges(self) -> "SectorAllocationPlan":
        """Tier targets within allowed ranges."""
        for key, (lo, hi) in TIER_TARGET_RANGES.items():
            val = self.tier_targets.get(key, 0.0)
            if not (lo <= val <= hi):
                raise ValueError(
                    f"tier_targets['{key}']={val} not in [{lo}, {hi}]"
                )
        return self

    @model_validator(mode="after")
    def validate_tier_targets_sum(self) -> "SectorAllocationPlan":
        """T1 + T2 + T3 + Cash should sum to 95-105%."""
        total = sum(self.tier_targets.values())
        if not (95.0 <= total <= 105.0):
            raise ValueError(
                f"tier_targets sum to {total:.1f}%, expected 95-105%"
            )
        return self


class ThemeRecommendation(BaseModel):
    """Schwab theme-to-ETF recommendation."""

    theme: str = Field(...)
    recommended_etfs: List[str] = Field(default_factory=list)
    tier_fit: int = Field(..., ge=1, le=3)
    allocation_suggestion: str = Field(..., min_length=1)
    conviction: str = Field(...)
    rationale: str = Field(..., min_length=10)
    etf_rankings: Optional[List[Dict[str, Any]]] = Field(None, description="Ranked ETFs for this theme")

    @field_validator("theme")
    @classmethod
    def validate_theme(cls, v: str) -> str:
        if v not in THEME_ETFS:
            raise ValueError(f"Theme '{v}' not in THEME_ETFS mapping")
        return v

    @field_validator("conviction")
    @classmethod
    def validate_conviction(cls, v: str) -> str:
        if v not in VALID_CONVICTION_LEVELS:
            raise ValueError(
                f"conviction must be one of {VALID_CONVICTION_LEVELS}, got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def validate_etfs_in_theme(self) -> "ThemeRecommendation":
        """All recommended ETFs must exist in THEME_ETFS[theme]."""
        valid_etfs = set(THEME_ETFS.get(self.theme, []))
        for etf in self.recommended_etfs:
            if etf not in valid_etfs:
                raise ValueError(
                    f"ETF '{etf}' not in THEME_ETFS['{self.theme}']: {valid_etfs}"
                )
        return self

    @model_validator(mode="after")
    def validate_theme_tier_fit(self) -> "ThemeRecommendation":
        """Growth themes → T1/T2, Defensive themes → T2/T3."""
        if self.theme in GROWTH_THEMES and self.tier_fit == 3:
            raise ValueError(
                f"Growth theme '{self.theme}' should have tier_fit 1 or 2, got 3"
            )
        if self.theme in DEFENSIVE_THEMES and self.tier_fit == 1:
            raise ValueError(
                f"Defensive theme '{self.theme}' should have tier_fit 2 or 3, got 1"
            )
        return self


class RotationActionSignal(BaseModel):
    """Action signal with confirmation/invalidation criteria."""

    action: str = Field(..., min_length=10)
    trigger: str = Field(..., min_length=10)
    confirmation: List[str] = Field(..., min_length=1)
    invalidation: List[str] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Top-level Output
# ---------------------------------------------------------------------------

class SectorStrategyOutput(BaseModel):
    """Top-level output contract for the Sector Strategist."""

    rotation_response: str = Field(..., min_length=20)
    regime_adjustment: str = Field(..., min_length=20)
    sector_allocations: SectorAllocationPlan
    theme_recommendations: List[ThemeRecommendation] = Field(default_factory=list)
    rotation_signals: List[RotationActionSignal] = Field(default_factory=list)
    analysis_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD"
    )
    summary: str = Field(..., min_length=50)
