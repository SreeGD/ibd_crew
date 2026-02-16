"""
Agent 13: Historical Analyst — Schema Validation Tests
Level 1: Pure Pydantic validation, no LLM, no ChromaDB.

Tests historical_output.py models and validators.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ibd_agents.schemas.historical_output import (
    MOMENTUM_DIRECTIONS,
    HistoricalAnalog,
    HistoricalAnalysisOutput,
    HistoricalStoreMeta,
    RatingTrend,
    SectorHistoricalTrend,
    StockHistoricalContext,
)


# ---------------------------------------------------------------------------
# RatingTrend
# ---------------------------------------------------------------------------


@pytest.mark.schema
class TestRatingTrend:
    """Test RatingTrend model validation."""

    def test_valid_rating_trend(self):
        """Valid RatingTrend instantiates."""
        rt = RatingTrend(
            metric="composite",
            values=[{"date": "2026-02-01", "value": 95}],
            direction="improving",
            change_4w=3.0,
        )
        assert rt.metric == "composite"
        assert rt.direction == "improving"

    def test_invalid_direction(self):
        """Invalid direction raises ValidationError."""
        with pytest.raises(ValidationError):
            RatingTrend(
                metric="rs",
                values=[{"date": "2026-02-01", "value": 90}],
                direction="unknown",
                change_4w=0.0,
            )

    def test_empty_values_rejected(self):
        """Empty values list is rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            RatingTrend(
                metric="eps",
                values=[],
                direction="stable",
            )


# ---------------------------------------------------------------------------
# StockHistoricalContext
# ---------------------------------------------------------------------------


@pytest.mark.schema
class TestStockHistoricalContext:
    """Test StockHistoricalContext model validation."""

    def test_valid_context(self):
        """Valid StockHistoricalContext instantiates."""
        ctx = StockHistoricalContext(
            symbol="NVDA",
            weeks_tracked=4,
            momentum_direction="improving",
            momentum_score=25.0,
            notes="Strong momentum trend",
        )
        assert ctx.symbol == "NVDA"
        assert ctx.weeks_tracked == 4

    def test_symbol_uppercased(self):
        """Symbol is auto-uppercased."""
        ctx = StockHistoricalContext(
            symbol="nvda",
            weeks_tracked=1,
            momentum_direction="stable",
            momentum_score=0.0,
        )
        assert ctx.symbol == "NVDA"

    def test_momentum_score_range(self):
        """Momentum score outside -100..100 is rejected."""
        with pytest.raises(ValidationError):
            StockHistoricalContext(
                symbol="TEST",
                weeks_tracked=1,
                momentum_direction="improving",
                momentum_score=150.0,
            )

    def test_invalid_date_format(self):
        """Invalid first_seen_date format is rejected."""
        with pytest.raises(ValidationError):
            StockHistoricalContext(
                symbol="TEST",
                weeks_tracked=1,
                momentum_direction="stable",
                momentum_score=0.0,
                first_seen_date="02-01-2026",  # Wrong format
            )

    def test_valid_date_format(self):
        """Valid YYYY-MM-DD first_seen_date accepted."""
        ctx = StockHistoricalContext(
            symbol="TEST",
            weeks_tracked=1,
            momentum_direction="stable",
            momentum_score=0.0,
            first_seen_date="2026-02-01",
        )
        assert ctx.first_seen_date == "2026-02-01"


# ---------------------------------------------------------------------------
# HistoricalAnalog
# ---------------------------------------------------------------------------


@pytest.mark.schema
class TestHistoricalAnalog:
    """Test HistoricalAnalog model validation."""

    def test_valid_analog(self):
        """Valid HistoricalAnalog instantiates."""
        analog = HistoricalAnalog(
            symbol="AAPL",
            analog_date="2025-06-15",
            sector="CHIPS",
            composite_rating=95,
            rs_rating=90,
            eps_rating=88,
            similarity_score=0.85,
            current_symbol="NVDA",
            context="Historical analog for NVDA",
        )
        assert analog.symbol == "AAPL"
        assert analog.similarity_score == 0.85

    def test_similarity_out_of_range(self):
        """Similarity > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            HistoricalAnalog(
                symbol="AAPL",
                analog_date="2025-06-15",
                sector="CHIPS",
                similarity_score=1.5,
                current_symbol="NVDA",
            )

    def test_symbol_uppercased(self):
        """Symbol is auto-uppercased."""
        analog = HistoricalAnalog(
            symbol="aapl",
            analog_date="2025-06-15",
            sector="CHIPS",
            similarity_score=0.5,
            current_symbol="NVDA",
        )
        assert analog.symbol == "AAPL"


# ---------------------------------------------------------------------------
# SectorHistoricalTrend
# ---------------------------------------------------------------------------


@pytest.mark.schema
class TestSectorHistoricalTrend:
    """Test SectorHistoricalTrend model validation."""

    def test_valid_trend(self):
        """Valid SectorHistoricalTrend instantiates."""
        trend = SectorHistoricalTrend(
            sector="CHIPS",
            snapshots=[{"date": "2026-02-01", "avg_rs": 90}],
            avg_rs_trend="improving",
            stock_count_trend="growing",
            elite_pct_trend="stable",
            notes="4 weeks of data",
        )
        assert trend.sector == "CHIPS"
        assert trend.avg_rs_trend == "improving"

    def test_invalid_stock_count_trend(self):
        """Invalid stock_count_trend value is rejected."""
        with pytest.raises(ValidationError):
            SectorHistoricalTrend(
                sector="CHIPS",
                snapshots=[{"date": "2026-02-01"}],
                avg_rs_trend="stable",
                stock_count_trend="improving",  # Should be growing/stable/shrinking
                elite_pct_trend="stable",
            )


# ---------------------------------------------------------------------------
# HistoricalStoreMeta
# ---------------------------------------------------------------------------


@pytest.mark.schema
class TestHistoricalStoreMeta:
    """Test HistoricalStoreMeta model validation."""

    def test_valid_meta(self):
        """Valid HistoricalStoreMeta instantiates."""
        meta = HistoricalStoreMeta(
            store_available=True,
            total_snapshots=4,
            date_range=["2026-01-15", "2026-02-08"],
            total_records=120,
            unique_symbols=30,
        )
        assert meta.store_available is True
        assert meta.total_snapshots == 4

    def test_unavailable_meta(self):
        """Unavailable store with zeros is valid."""
        meta = HistoricalStoreMeta(
            store_available=False,
            total_snapshots=0,
            total_records=0,
            unique_symbols=0,
        )
        assert meta.store_available is False

    def test_negative_records_rejected(self):
        """Negative total_records is rejected (ge=0)."""
        with pytest.raises(ValidationError):
            HistoricalStoreMeta(
                store_available=True,
                total_snapshots=1,
                total_records=-1,
                unique_symbols=0,
            )


# ---------------------------------------------------------------------------
# HistoricalAnalysisOutput
# ---------------------------------------------------------------------------


@pytest.mark.schema
class TestHistoricalAnalysisOutput:
    """Test HistoricalAnalysisOutput model and validators."""

    def _make_empty_output(self, **overrides) -> dict:
        """Build minimal valid output dict."""
        base = {
            "store_meta": HistoricalStoreMeta(
                store_available=False,
                total_snapshots=0,
                total_records=0,
                unique_symbols=0,
            ),
            "analysis_date": "2026-02-15",
            "historical_source": "empty",
            "summary": (
                "Historical analysis unavailable — ChromaDB not installed. "
                "Install chromadb to enable historical context analysis."
            ),
        }
        base.update(overrides)
        return base

    def test_valid_empty_output(self):
        """Empty output (no ChromaDB) is valid."""
        output = HistoricalAnalysisOutput(**self._make_empty_output())
        assert output.historical_source == "empty"

    def test_valid_chromadb_output(self):
        """Output with chromadb source is valid."""
        output = HistoricalAnalysisOutput(
            store_meta=HistoricalStoreMeta(
                store_available=True,
                total_snapshots=4,
                total_records=120,
                unique_symbols=30,
            ),
            stock_analyses=[
                StockHistoricalContext(
                    symbol="NVDA",
                    weeks_tracked=4,
                    momentum_direction="improving",
                    momentum_score=25.0,
                ),
            ],
            improving_stocks=["NVDA"],
            analysis_date="2026-02-15",
            historical_source="chromadb",
            summary=(
                "Historical analysis across 4 weekly snapshots. "
                "Analyzed 1 stock: 1 improving, 0 deteriorating."
            ),
        )
        assert output.historical_source == "chromadb"
        assert len(output.stock_analyses) == 1

    def test_invalid_source(self):
        """Invalid historical_source is rejected."""
        with pytest.raises(ValidationError):
            HistoricalAnalysisOutput(
                **self._make_empty_output(historical_source="unknown")
            )

    def test_summary_too_short(self):
        """Summary shorter than 50 chars is rejected."""
        with pytest.raises(ValidationError):
            HistoricalAnalysisOutput(
                **self._make_empty_output(summary="Too short")
            )

    def test_momentum_consistency_improving(self):
        """Stock in improving_stocks must have direction=improving."""
        with pytest.raises(ValidationError, match="improving_stocks"):
            HistoricalAnalysisOutput(
                store_meta=HistoricalStoreMeta(
                    store_available=True,
                    total_snapshots=1,
                    total_records=10,
                    unique_symbols=5,
                ),
                stock_analyses=[
                    StockHistoricalContext(
                        symbol="NVDA",
                        weeks_tracked=4,
                        momentum_direction="stable",  # Not improving!
                        momentum_score=0.0,
                    ),
                ],
                improving_stocks=["NVDA"],  # Contradiction
                analysis_date="2026-02-15",
                historical_source="chromadb",
                summary=(
                    "Historical analysis test with momentum consistency "
                    "validation — this should fail due to mismatch."
                ),
            )

    def test_momentum_consistency_deteriorating(self):
        """Stock in deteriorating_stocks must have direction=deteriorating."""
        with pytest.raises(ValidationError, match="deteriorating_stocks"):
            HistoricalAnalysisOutput(
                store_meta=HistoricalStoreMeta(
                    store_available=True,
                    total_snapshots=1,
                    total_records=10,
                    unique_symbols=5,
                ),
                stock_analyses=[
                    StockHistoricalContext(
                        symbol="AAPL",
                        weeks_tracked=4,
                        momentum_direction="improving",
                        momentum_score=20.0,
                    ),
                ],
                deteriorating_stocks=["AAPL"],  # Contradiction
                analysis_date="2026-02-15",
                historical_source="chromadb",
                summary=(
                    "Historical analysis test with deteriorating consistency "
                    "validation — this should fail due to mismatch."
                ),
            )

    def test_invalid_date_format(self):
        """Invalid analysis_date format is rejected."""
        with pytest.raises(ValidationError):
            HistoricalAnalysisOutput(
                **self._make_empty_output(analysis_date="Feb 15 2026")
            )
