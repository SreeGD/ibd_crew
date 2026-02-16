"""
Agent 13: Historical Analyst — Store Unit Tests
Level 1: Pure function tests with temporary ChromaDB instances.

Tests historical_store.py functions: ingestion, rating history,
list signals, similar setups, sector trends, and utilities.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ibd_agents.tools.historical_store import (
    HAS_CHROMADB,
    extract_date_from_filename,
    find_similar_setups,
    get_sector_trends,
    get_snapshot_dates,
    get_store_stats,
    ingest_snapshot,
    search_list_signals,
    search_rating_history,
)

# Skip marker for tests that require chromadb
requires_chromadb = pytest.mark.skipif(
    not HAS_CHROMADB, reason="chromadb not installed"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_STOCKS_W1 = [
    {
        "symbol": "NVDA",
        "company_name": "NVIDIA Corp",
        "sector": "CHIPS",
        "composite_rating": 98,
        "rs_rating": 95,
        "eps_rating": 99,
        "smr_rating": "A",
        "acc_dis_rating": "B",
        "ibd_lists": ["IBD 50", "Tech Leaders"],
    },
    {
        "symbol": "MSFT",
        "company_name": "Microsoft Corp",
        "sector": "SOFTWARE",
        "composite_rating": 96,
        "rs_rating": 91,
        "eps_rating": 90,
        "smr_rating": "A",
        "acc_dis_rating": "A",
        "ibd_lists": ["IBD 50"],
    },
    {
        "symbol": "AAPL",
        "company_name": "Apple Inc",
        "sector": "CHIPS",
        "composite_rating": 92,
        "rs_rating": 80,
        "eps_rating": 85,
        "smr_rating": "B",
        "acc_dis_rating": "B-",
        "ibd_lists": [],
    },
]

SAMPLE_STOCKS_W2 = [
    {
        "symbol": "NVDA",
        "company_name": "NVIDIA Corp",
        "sector": "CHIPS",
        "composite_rating": 99,
        "rs_rating": 97,
        "eps_rating": 99,
        "smr_rating": "A",
        "acc_dis_rating": "A",
        "ibd_lists": ["IBD 50", "Tech Leaders", "Top 200"],
    },
    {
        "symbol": "MSFT",
        "company_name": "Microsoft Corp",
        "sector": "SOFTWARE",
        "composite_rating": 95,
        "rs_rating": 89,
        "eps_rating": 88,
        "smr_rating": "A",
        "acc_dis_rating": "B",
        "ibd_lists": [],  # Dropped from IBD 50
    },
    {
        "symbol": "GOOGL",
        "company_name": "Alphabet Inc",
        "sector": "INTERNET",
        "composite_rating": 94,
        "rs_rating": 88,
        "eps_rating": 92,
        "smr_rating": "A",
        "acc_dis_rating": "B",
        "ibd_lists": ["IBD 50"],
    },
]


@pytest.fixture
def db_path():
    """Create a temporary directory for ChromaDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test_chroma")


@pytest.fixture
def populated_db(db_path):
    """Ingest two weeks of data and return the db_path."""
    ingest_snapshot(SAMPLE_STOCKS_W1, "2026-02-01", "week1.pdf", db_path)
    ingest_snapshot(SAMPLE_STOCKS_W2, "2026-02-08", "week2.pdf", db_path)
    return db_path


# ---------------------------------------------------------------------------
# extract_date_from_filename tests
# ---------------------------------------------------------------------------


@pytest.mark.schema
class TestExtractDateFromFilename:
    """Test date extraction from IBD filenames."""

    def test_month_day_year(self):
        """'Feb 13, 2026' format."""
        result = extract_date_from_filename(
            "IBD Smart Tables -- Feb 13, 2026.pdf"
        )
        assert result == "2026-02-13"

    def test_iso_format(self):
        """'2026-02-13' format."""
        result = extract_date_from_filename("ibd_data_2026-02-13.xls")
        assert result == "2026-02-13"

    def test_underscore_format(self):
        """'02_13_2026' format."""
        result = extract_date_from_filename("ibd_02_13_2026.csv")
        assert result == "2026-02-13"

    def test_no_date(self):
        """No recognizable date returns None."""
        result = extract_date_from_filename("random_file.pdf")
        assert result is None

    def test_month_no_comma(self):
        """'Feb 13 2026' without comma."""
        result = extract_date_from_filename(
            "IBD Tables Feb 13 2026 data.pdf"
        )
        assert result == "2026-02-13"


# ---------------------------------------------------------------------------
# ingest_snapshot tests
# ---------------------------------------------------------------------------


@requires_chromadb
@pytest.mark.schema
class TestIngestSnapshot:
    """Test snapshot ingestion into ChromaDB."""

    def test_ingest_returns_counts(self, db_path):
        """ingest_snapshot returns stocks_added and sectors_updated."""
        result = ingest_snapshot(
            SAMPLE_STOCKS_W1, "2026-02-01", "test.pdf", db_path
        )
        assert result["stocks_added"] == 3
        assert result["sectors_updated"] == 2  # CHIPS + SOFTWARE
        assert result["snapshot_date"] == "2026-02-01"

    def test_ingest_deduplication(self, db_path):
        """Re-ingesting same date/symbols doesn't duplicate."""
        ingest_snapshot(SAMPLE_STOCKS_W1, "2026-02-01", "test.pdf", db_path)
        ingest_snapshot(SAMPLE_STOCKS_W1, "2026-02-01", "test.pdf", db_path)
        stats = get_store_stats(db_path)
        assert stats["total_records"] == 3  # Not 6

    def test_ingest_empty_list(self, db_path):
        """Empty stock list ingests without error."""
        result = ingest_snapshot([], "2026-02-01", "empty.pdf", db_path)
        assert result["stocks_added"] == 0

    def test_ingest_missing_symbol(self, db_path):
        """Stock without symbol is skipped."""
        stocks = [{"company_name": "NoSymbol Corp", "sector": "CHIPS"}]
        result = ingest_snapshot(stocks, "2026-02-01", "test.pdf", db_path)
        assert result["stocks_added"] == 0


# ---------------------------------------------------------------------------
# search_rating_history tests
# ---------------------------------------------------------------------------


@requires_chromadb
@pytest.mark.schema
class TestSearchRatingHistory:
    """Test rating history queries."""

    def test_history_returns_sorted_by_date(self, populated_db):
        """Rating history for NVDA returns 2 records sorted by date."""
        history = search_rating_history("NVDA", db_path=populated_db)
        assert len(history) == 2
        assert history[0]["snapshot_date"] < history[1]["snapshot_date"]

    def test_history_has_all_ratings(self, populated_db):
        """Each record has composite, rs, eps ratings."""
        history = search_rating_history("NVDA", db_path=populated_db)
        for record in history:
            assert "composite_rating" in record
            assert "rs_rating" in record
            assert "eps_rating" in record

    def test_history_metric_filter(self, populated_db):
        """Filtering by metric returns only that metric."""
        history = search_rating_history(
            "NVDA", metric="rs", db_path=populated_db
        )
        assert len(history) == 2
        assert "rs_rating" in history[0]
        assert "composite_rating" not in history[0]

    def test_history_unknown_symbol(self, populated_db):
        """Unknown symbol returns empty list."""
        history = search_rating_history("ZZZZZ", db_path=populated_db)
        assert history == []

    def test_history_date_filter(self, populated_db):
        """date_from filters to matching records only."""
        history = search_rating_history(
            "NVDA", date_from="2026-02-05", db_path=populated_db
        )
        assert len(history) == 1
        assert history[0]["snapshot_date"] == "2026-02-08"


# ---------------------------------------------------------------------------
# search_list_signals tests
# ---------------------------------------------------------------------------


@requires_chromadb
@pytest.mark.schema
class TestSearchListSignals:
    """Test list entry/exit signal detection."""

    def test_nvda_entries(self, populated_db):
        """NVDA has ENTRY events for its lists."""
        signals = search_list_signals(symbol="NVDA", db_path=populated_db)
        entries = [s for s in signals if s["event_type"] == "ENTRY"]
        assert len(entries) >= 2  # At least IBD 50, Tech Leaders in week 1

    def test_msft_exit(self, populated_db):
        """MSFT dropped from IBD 50 in week 2 -> EXIT event."""
        signals = search_list_signals(symbol="MSFT", db_path=populated_db)
        exits = [s for s in signals if s["event_type"] == "EXIT"]
        assert len(exits) >= 1
        assert any("IBD 50" in e.get("list_name", "") for e in exits)

    def test_nvda_new_entry_week2(self, populated_db):
        """NVDA gained Top 200 in week 2 -> new ENTRY event."""
        signals = search_list_signals(symbol="NVDA", db_path=populated_db)
        w2_entries = [
            s for s in signals
            if s["event_type"] == "ENTRY" and s["date"] == "2026-02-08"
        ]
        assert any("Top 200" in e.get("list_name", "") for e in w2_entries)

    def test_all_signals_sorted(self, populated_db):
        """All signals across all symbols are date-sorted."""
        signals = search_list_signals(db_path=populated_db)
        dates = [s["date"] for s in signals]
        assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# find_similar_setups tests
# ---------------------------------------------------------------------------


@requires_chromadb
@pytest.mark.schema
class TestFindSimilarSetups:
    """Test semantic similarity search."""

    def test_similar_returns_results(self, populated_db):
        """find_similar_setups returns at least 1 result for a known profile."""
        profile = {
            "symbol": "TEST",
            "sector": "CHIPS",
            "composite_rating": 97,
            "rs_rating": 94,
            "eps_rating": 98,
        }
        results = find_similar_setups(
            profile, top_n=5, db_path=populated_db
        )
        assert len(results) >= 1

    def test_similar_excludes_self(self, populated_db):
        """Self-matches (same symbol) are excluded."""
        profile = {
            "symbol": "NVDA",
            "sector": "CHIPS",
            "composite_rating": 98,
            "rs_rating": 95,
            "eps_rating": 99,
        }
        results = find_similar_setups(
            profile, top_n=10, db_path=populated_db
        )
        assert all(r["symbol"] != "NVDA" for r in results)

    def test_similar_has_scores(self, populated_db):
        """Each result has similarity_score between 0 and 1."""
        profile = {
            "symbol": "TEST",
            "sector": "SOFTWARE",
            "composite_rating": 95,
            "rs_rating": 90,
            "eps_rating": 88,
        }
        results = find_similar_setups(
            profile, top_n=3, db_path=populated_db
        )
        for r in results:
            assert 0 <= r["similarity_score"] <= 1


# ---------------------------------------------------------------------------
# get_sector_trends tests
# ---------------------------------------------------------------------------


@requires_chromadb
@pytest.mark.schema
class TestGetSectorTrends:
    """Test sector trend queries."""

    def test_chips_has_two_weeks(self, populated_db):
        """CHIPS sector has 2 weekly snapshots."""
        trends = get_sector_trends("CHIPS", db_path=populated_db)
        assert len(trends) == 2

    def test_sector_metrics_present(self, populated_db):
        """Each sector snapshot has avg_rs, stock_count, elite_pct."""
        trends = get_sector_trends("CHIPS", db_path=populated_db)
        for t in trends:
            assert "avg_rs" in t
            assert "stock_count" in t
            assert "elite_pct" in t

    def test_unknown_sector(self, populated_db):
        """Unknown sector returns empty list."""
        trends = get_sector_trends("NONEXISTENT", db_path=populated_db)
        assert trends == []


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


@requires_chromadb
@pytest.mark.schema
class TestStoreUtilities:
    """Test store utility functions."""

    def test_stats_available(self, populated_db):
        """get_store_stats returns available=True with data."""
        stats = get_store_stats(populated_db)
        assert stats["available"] is True
        assert stats["total_records"] == 6  # 3 w1 + 3 w2 (AAPL only w1, GOOGL only w2)
        assert stats["snapshot_count"] == 2
        assert stats["unique_symbols"] == 4  # NVDA, MSFT, AAPL, GOOGL

    def test_stats_empty_store(self, db_path):
        """Empty store returns available=True, records=0."""
        stats = get_store_stats(db_path)
        assert stats["available"] is True
        assert stats["total_records"] == 0

    def test_snapshot_dates(self, populated_db):
        """get_snapshot_dates returns sorted dates."""
        dates = get_snapshot_dates(populated_db)
        assert dates == ["2026-02-01", "2026-02-08"]

    def test_date_range_in_stats(self, populated_db):
        """date_range in stats covers first and last dates."""
        stats = get_store_stats(populated_db)
        assert stats["date_range"] == ["2026-02-01", "2026-02-08"]
