"""
Agent 13: Historical Analyst — Output Schema
IBD Momentum Investment Framework v4.0

Historical context analysis using RAG over past IBD weekly snapshots.
Never makes buy/sell recommendations — only discovers, scores, and organizes.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOMENTUM_DIRECTIONS = ("improving", "stable", "deteriorating")
SIGNAL_TYPES = ("ENTRY", "EXIT")
COUNT_TRENDS = ("growing", "stable", "shrinking")


# ---------------------------------------------------------------------------
# Supporting Models
# ---------------------------------------------------------------------------


class RatingTrend(BaseModel):
    """Weekly rating time series for a single metric."""

    metric: str = Field(
        ..., description="composite/rs/eps"
    )
    values: List[dict] = Field(
        ..., min_length=1,
        description="List of {date, value} dicts",
    )
    direction: Literal["improving", "stable", "deteriorating"] = Field(...)
    change_4w: Optional[float] = Field(
        None, description="Change over last 4 weeks"
    )


class StockHistoricalContext(BaseModel):
    """Historical context for a single stock."""

    symbol: str = Field(..., min_length=1, max_length=10)
    weeks_tracked: int = Field(
        ..., ge=0, description="Number of weekly snapshots"
    )
    rating_trends: List[RatingTrend] = Field(default_factory=list)
    list_entries: List[dict] = Field(
        default_factory=list, description="ENTRY events"
    )
    list_exits: List[dict] = Field(
        default_factory=list, description="EXIT events"
    )
    momentum_direction: Literal[
        "improving", "stable", "deteriorating"
    ] = Field(...)
    momentum_score: float = Field(
        ..., ge=-100, le=100,
        description="Composite momentum score",
    )
    first_seen_date: Optional[str] = Field(
        None, pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    notes: str = Field(
        default="", description="Summary note about historical context"
    )

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()


class HistoricalAnalog(BaseModel):
    """A past stock that had a similar ratings profile."""

    symbol: str = Field(..., min_length=1, max_length=10)
    analog_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    sector: str = Field(...)
    composite_rating: Optional[int] = Field(None, ge=0, le=99)
    rs_rating: Optional[int] = Field(None, ge=0, le=99)
    eps_rating: Optional[int] = Field(None, ge=0, le=99)
    similarity_score: float = Field(..., ge=0, le=1)
    current_symbol: str = Field(
        ..., description="The stock this is an analog for"
    )
    context: str = Field(
        default="",
        description="What happened after this setup",
    )

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.strip().upper()


class SectorHistoricalTrend(BaseModel):
    """Historical trend for a sector."""

    sector: str = Field(...)
    snapshots: List[dict] = Field(
        ..., min_length=1, description="Weekly sector metrics"
    )
    avg_rs_trend: Literal[
        "improving", "stable", "deteriorating"
    ] = Field(...)
    stock_count_trend: Literal[
        "growing", "stable", "shrinking"
    ] = Field(...)
    elite_pct_trend: Literal[
        "improving", "stable", "deteriorating"
    ] = Field(...)
    notes: str = Field(default="")


class HistoricalStoreMeta(BaseModel):
    """Metadata about the historical store status."""

    store_available: bool = Field(...)
    total_snapshots: int = Field(..., ge=0)
    date_range: List[str] = Field(default_factory=list)
    total_records: int = Field(..., ge=0)
    unique_symbols: int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Top-Level Output
# ---------------------------------------------------------------------------


class HistoricalAnalysisOutput(BaseModel):
    """
    Output contract for the Historical Analyst (Agent 13).

    Provides historical context for the current pipeline's stocks
    using RAG over past weekly IBD snapshots.
    """

    store_meta: HistoricalStoreMeta = Field(...)

    stock_analyses: List[StockHistoricalContext] = Field(
        default_factory=list,
        description="Historical context per stock from current pipeline",
    )

    improving_stocks: List[str] = Field(
        default_factory=list,
        description="Symbols with improving momentum direction",
    )
    deteriorating_stocks: List[str] = Field(
        default_factory=list,
        description="Symbols with deteriorating momentum direction",
    )

    historical_analogs: List[HistoricalAnalog] = Field(
        default_factory=list,
        description="Past stocks with similar rating profiles",
    )

    sector_trends: List[SectorHistoricalTrend] = Field(
        default_factory=list,
        description="Historical sector-level trends",
    )

    new_to_ibd_lists: List[str] = Field(
        default_factory=list,
        description="Symbols that first appeared on IBD lists recently",
    )
    dropped_from_ibd_lists: List[str] = Field(
        default_factory=list,
        description="Symbols that recently fell off IBD lists",
    )

    analysis_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    historical_source: Literal["chromadb", "empty"] = Field(
        ...,
        description="'chromadb' if data was available, 'empty' if store was empty",
    )
    summary: str = Field(..., min_length=50)

    @model_validator(mode="after")
    def validate_momentum_consistency(self) -> "HistoricalAnalysisOutput":
        """improving/deteriorating lists must match stock_analyses directions."""
        analyzed = {s.symbol: s for s in self.stock_analyses}
        for sym in self.improving_stocks:
            match = analyzed.get(sym)
            if match and match.momentum_direction != "improving":
                raise ValueError(
                    f"{sym} in improving_stocks but "
                    f"direction={match.momentum_direction}"
                )
        for sym in self.deteriorating_stocks:
            match = analyzed.get(sym)
            if match and match.momentum_direction != "deteriorating":
                raise ValueError(
                    f"{sym} in deteriorating_stocks but "
                    f"direction={match.momentum_direction}"
                )
        return self
