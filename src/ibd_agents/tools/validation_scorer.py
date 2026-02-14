"""
Research Agent Tool: Validation Scorer
Calculate multi-source validation scores per Framework v4.0 §2.7.

Multi-Source Validated = total score ≥ 5 from 2+ different providers.
"""

from __future__ import annotations

from typing import Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

from ibd_agents.schemas.research_output import (
    SOURCE_PROVIDERS,
    VALIDATION_POINTS,
    compute_preliminary_tier,
    compute_validation_score,
    is_ibd_keep_candidate,
)


class StockAggregation:
    """
    Aggregate data for a single stock across all sources.
    Collects IBD lists, themes, ratings, and computes validation score.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.company_name: str = symbol
        self.sector: Optional[str] = None

        # IBD Ratings — keep best available
        self.composite_rating: Optional[int] = None
        self.rs_rating: Optional[int] = None
        self.eps_rating: Optional[int] = None
        self.smr_rating: Optional[str] = None
        self.acc_dis_rating: Optional[str] = None

        # Sources
        self.ibd_lists: list[str] = []
        self.schwab_themes: list[str] = []
        self.fool_status: Optional[str] = None
        self.other_ratings: dict[str, str] = {}
        self.source_files: list[str] = []

        # IBD Smart Table rankings
        self.sector_rank: Optional[int] = None
        self.stock_rank_in_sector: Optional[int] = None

        # ETF-specific metrics (from IBD ETF Tables PDF)
        self.ytd_change: Optional[float] = None
        self.close_price: Optional[float] = None
        self.price_change: Optional[float] = None
        self.volume_pct_change: Optional[int] = None
        self.div_yield: Optional[float] = None

        # Classification
        self.is_etf: bool = False
        self.cap_size: Optional[str] = None

        # Morningstar metadata
        self.morningstar_rating: Optional[str] = None
        self.economic_moat: Optional[str] = None
        self.fair_value: Optional[float] = None
        self.morningstar_price: Optional[float] = None
        self.price_to_fair_value: Optional[float] = None
        self.morningstar_uncertainty: Optional[str] = None

        # Validation labels for scoring
        self._validation_labels: list[str] = []

    def merge_ibd(self, data: dict) -> None:
        """Merge IBD XLS/PDF data into this stock."""
        # Update ratings (prefer non-None, keep highest for numeric)
        if data.get("composite_rating") is not None:
            if self.composite_rating is None or data["composite_rating"] > self.composite_rating:
                self.composite_rating = data["composite_rating"]
        if data.get("rs_rating") is not None:
            if self.rs_rating is None or data["rs_rating"] > self.rs_rating:
                self.rs_rating = data["rs_rating"]
        if data.get("eps_rating") is not None:
            if self.eps_rating is None or data["eps_rating"] > self.eps_rating:
                self.eps_rating = data["eps_rating"]
        if data.get("smr_rating") is not None:
            if self.smr_rating is None:
                self.smr_rating = data["smr_rating"]
        if data.get("acc_dis_rating") is not None:
            if self.acc_dis_rating is None:
                self.acc_dis_rating = data["acc_dis_rating"]

        # Track list and source
        if data.get("ibd_list") and data["ibd_list"] not in self.ibd_lists:
            self.ibd_lists.append(data["ibd_list"])
            self._validation_labels.append(data["ibd_list"])

        # Classify as ETF if record explicitly flagged
        if data.get("is_etf"):
            self.is_etf = True
        if data.get("source_file") and data["source_file"] not in self.source_files:
            self.source_files.append(data["source_file"])

        # Company name and sector
        if data.get("company_name") and data["company_name"] != self.symbol:
            self.company_name = data["company_name"]
        if data.get("sector") and data["sector"] != "UNKNOWN":
            if self.sector is None or self.sector == "UNKNOWN":
                self.sector = data["sector"]

        # IBD Smart Table rankings (keep first seen)
        if data.get("sector_rank") is not None and self.sector_rank is None:
            self.sector_rank = data["sector_rank"]
        if data.get("stock_rank_in_sector") is not None and self.stock_rank_in_sector is None:
            self.stock_rank_in_sector = data["stock_rank_in_sector"]

        # ETF-specific metrics (keep first seen)
        if data.get("ytd_change") is not None and self.ytd_change is None:
            self.ytd_change = data["ytd_change"]
        if data.get("close_price") is not None and self.close_price is None:
            self.close_price = data["close_price"]
        if data.get("price_change") is not None and self.price_change is None:
            self.price_change = data["price_change"]
        if data.get("volume_pct_change") is not None and self.volume_pct_change is None:
            self.volume_pct_change = data["volume_pct_change"]
        if data.get("div_yield") is not None and self.div_yield is None:
            self.div_yield = data["div_yield"]

    def merge_fool(self, data: dict) -> None:
        """Merge Motley Fool data."""
        status = data.get("fool_status")
        if status:
            self.fool_status = status
            label = f"Motley Fool {status}"
            if label not in self._validation_labels:
                self._validation_labels.append(label)

        if data.get("source_file") and data["source_file"] not in self.source_files:
            self.source_files.append(data["source_file"])

        if data.get("company_name") and data["company_name"] != self.symbol:
            self.company_name = data["company_name"]

    def merge_theme(self, data: dict) -> None:
        """Merge Schwab theme data."""
        if data.get("is_etf"):
            self.is_etf = True
        theme = data.get("theme")
        if theme and theme not in self.schwab_themes:
            self.schwab_themes.append(theme)

            # First theme adds "Core", subsequent add bonus
            if len(self.schwab_themes) == 1:
                self._validation_labels.append("Schwab Theme Core")
            elif len(self.schwab_themes) == 2:
                self._validation_labels.append("Schwab Theme Secondary")
                self._validation_labels.append("Schwab Multiple Themes")

        if data.get("source_file") and data["source_file"] not in self.source_files:
            self.source_files.append(data["source_file"])

    def merge_other_rating(self, provider: str, rating: str) -> None:
        """Merge CFRA, ARGUS, Morningstar ratings."""
        self.other_ratings[provider] = rating
        label = f"{provider} {rating}"
        if label not in self._validation_labels:
            self._validation_labels.append(label)

    def compute_scores(self) -> dict:
        """Compute final validation score and derived flags."""
        score, providers = compute_validation_score(self._validation_labels)
        tier = compute_preliminary_tier(
            self.composite_rating, self.rs_rating, self.eps_rating
        )
        keep = is_ibd_keep_candidate(self.composite_rating, self.rs_rating)

        return {
            "symbol": self.symbol,
            "company_name": self.company_name,
            "sector": self.sector or "UNKNOWN",
            "composite_rating": self.composite_rating,
            "rs_rating": self.rs_rating,
            "eps_rating": self.eps_rating,
            "smr_rating": self.smr_rating,
            "acc_dis_rating": self.acc_dis_rating,
            "ibd_lists": self.ibd_lists,
            "schwab_themes": self.schwab_themes,
            "fool_status": self.fool_status,
            "other_ratings": self.other_ratings,
            "validation_score": score,
            "validation_providers": providers,
            "is_multi_source_validated": score >= 5 and providers >= 2,
            "is_ibd_keep_candidate": keep,
            "preliminary_tier": tier,
            "is_etf": self.is_etf,
            "cap_size": self.cap_size,
            "sector_rank": self.sector_rank,
            "stock_rank_in_sector": self.stock_rank_in_sector,
            "sources": self.source_files,
            "validation_labels": self._validation_labels,
            "morningstar_rating": self.morningstar_rating,
            "economic_moat": self.economic_moat,
            "fair_value": self.fair_value,
            "morningstar_price": self.morningstar_price,
            "price_to_fair_value": self.price_to_fair_value,
            "morningstar_uncertainty": self.morningstar_uncertainty,
            "ytd_change": self.ytd_change,
            "close_price": self.close_price,
            "price_change": self.price_change,
            "volume_pct_change": self.volume_pct_change,
            "div_yield": self.div_yield,
        }


class StockUniverse:
    """
    Aggregates data across all sources into a unified stock universe.
    """

    def __init__(self):
        self._stocks: dict[str, StockAggregation] = {}

    def _get_or_create(self, symbol: str) -> StockAggregation:
        sym = symbol.strip().upper()
        if sym not in self._stocks:
            self._stocks[sym] = StockAggregation(sym)
        return self._stocks[sym]

    def add_ibd_data(self, records: list[dict]) -> int:
        """Add IBD XLS/PDF records. Returns count added."""
        for rec in records:
            sym = rec.get("symbol", "").strip().upper()
            if not sym:
                continue
            agg = self._get_or_create(sym)
            agg.merge_ibd(rec)
        return len(records)

    def add_fool_data(self, records: list[dict]) -> int:
        """Add Motley Fool records. Returns count added."""
        for rec in records:
            sym = rec.get("symbol", "").strip().upper()
            if not sym:
                continue
            agg = self._get_or_create(sym)
            agg.merge_fool(rec)
        return len(records)

    def add_theme_data(self, records: list[dict]) -> int:
        """Add Schwab theme records. Returns count added."""
        for rec in records:
            sym = rec.get("symbol", "").strip().upper()
            if not sym:
                continue
            agg = self._get_or_create(sym)
            agg.merge_theme(rec)
        return len(records)

    def add_other_rating(self, symbol: str, provider: str, rating: str) -> None:
        """Add CFRA/ARGUS/Morningstar rating."""
        agg = self._get_or_create(symbol)
        agg.merge_other_rating(provider, rating)

    def add_morningstar_data(self, records: list[dict]) -> int:
        """Add Morningstar pick list records. Returns count added."""
        added = 0
        for rec in records:
            sym = rec.get("symbol", "").strip().upper()
            if not sym or rec.get("is_deletion"):
                continue
            star_count = rec.get("star_count", 0)
            if star_count >= 5:
                rating = "5-star"
            elif star_count >= 4:
                rating = "4-star"
            else:
                continue  # Only 4+ star picks get validation points
            agg = self._get_or_create(sym)
            agg.merge_other_rating("Morningstar", rating)
            if rec.get("company_name") and not agg.company_name:
                agg.company_name = rec["company_name"]
            # Use friendly source name instead of GUID filename
            cap = rec.get("cap_list", "unknown")
            friendly = "Morningstar Large Cap" if cap == "large" else "Morningstar Mid Cap" if cap == "mid" else "Morningstar"
            if friendly not in agg.source_files:
                agg.source_files.append(friendly)
            # Preserve Morningstar metadata
            agg.morningstar_rating = rating
            agg.economic_moat = rec.get("economic_moat")
            agg.fair_value = rec.get("fair_value")
            agg.morningstar_price = rec.get("price")
            agg.price_to_fair_value = rec.get("price_to_fair_value")
            agg.morningstar_uncertainty = rec.get("uncertainty")
            added += 1
        return added

    def get_all_scored(self) -> list[dict]:
        """Get all stocks with computed scores, sorted by validation score desc."""
        results = [agg.compute_scores() for agg in self._stocks.values()]
        results.sort(key=lambda x: (x["validation_score"], x["composite_rating"] or 0), reverse=True)
        return results

    def get_keep_candidates(self) -> list[str]:
        """Get symbols that meet IBD keep threshold."""
        return [
            sym for sym, agg in self._stocks.items()
            if is_ibd_keep_candidate(agg.composite_rating, agg.rs_rating)
        ]

    def get_multi_source_validated(self) -> list[str]:
        """Get symbols that are multi-source validated."""
        validated = []
        for sym, agg in self._stocks.items():
            scores = agg.compute_scores()
            if scores["is_multi_source_validated"]:
                validated.append(sym)
        return validated

    def get_sectorless_symbols(self) -> list[str]:
        """Return symbols that still have no sector assigned."""
        return [
            sym for sym, agg in self._stocks.items()
            if agg.sector is None or agg.sector == "UNKNOWN"
        ]

    def set_sector(self, symbol: str, sector: str) -> None:
        """Set sector for a stock (used by sector classifier)."""
        sym = symbol.strip().upper()
        if sym in self._stocks:
            self._stocks[sym].sector = sector

    def set_cap_size(self, symbol: str, cap_size: str) -> None:
        """Set cap_size for a stock (used by cap classifier)."""
        sym = symbol.strip().upper()
        if sym in self._stocks:
            self._stocks[sym].cap_size = cap_size

    def get_unsized_symbols(self) -> list[str]:
        """Return symbols that still have no cap_size assigned."""
        return [
            sym for sym, agg in self._stocks.items()
            if agg.cap_size is None and not agg.is_etf
        ]

    @property
    def count(self) -> int:
        return len(self._stocks)


# --- CrewAI Tool wrapper ---

class ValidationScoreInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    source_labels: list[str] = Field(..., description="List of source labels for scoring")


class CalculateValidationScoreTool(BaseTool):
    """Calculate multi-source validation score for a stock."""
    name: str = "calculate_validation_score"
    description: str = (
        "Calculate the multi-source validation score for a stock based on "
        "its presence across IBD lists, Schwab themes, Motley Fool, and "
        "analyst ratings. Returns total score and provider count."
    )
    args_schema: type[BaseModel] = ValidationScoreInput

    def _run(self, symbol: str, source_labels: list[str]) -> str:
        score, providers = compute_validation_score(source_labels)
        validated = score >= 5 and providers >= 2
        return (
            f"{symbol}: Score={score}, Providers={providers}, "
            f"Multi-Source Validated={'✅ YES' if validated else '❌ NO'}\n"
            f"Labels: {', '.join(source_labels)}"
        )
