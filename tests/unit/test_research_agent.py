"""
Agent 01: Research Agent — Pipeline & Behavioral Tests

Level 2: Tests the deterministic pipeline (run_research_pipeline).
Level 3: Behavioral boundary tests (no buy/sell, no sizing, etc.)

These tests use mock XLS data, NOT the LLM agent path.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from ibd_agents.agents.research_agent import run_research_pipeline
from ibd_agents.schemas.research_output import IBD_SECTORS, ResearchOutput
from tests.fixtures.conftest import PADDING_STOCKS, SAMPLE_IBD_STOCKS, create_mock_csv, create_mock_xls


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def research_data_dir(tmp_path: Path) -> Path:
    """Create a comprehensive mock data directory."""
    xls_dir = tmp_path / "ibd_xls"
    xls_dir.mkdir()

    # Big Cap 20
    create_mock_xls(str(xls_dir / "BIG_CAP_20_1.xls"), SAMPLE_IBD_STOCKS[:20])
    # Sector Leaders
    create_mock_xls(
        str(xls_dir / "SECTOR_LEADERS_1.xls"),
        [s for s in SAMPLE_IBD_STOCKS if s["composite_rating"] >= 99],
    )
    # IBD 50
    create_mock_xls(str(xls_dir / "IBD_50_3.xls"), SAMPLE_IBD_STOCKS + PADDING_STOCKS[:30])
    # Rising Profit Estimates
    create_mock_xls(
        str(xls_dir / "RISING_PROFIT_ESTIMATES.xls"),
        [s for s in SAMPLE_IBD_STOCKS if s.get("eps_rating", 0) >= 90],
    )
    # RS at New High
    create_mock_xls(
        str(xls_dir / "RELATIVE_STRENGTH_AT_NEW_HIGH.xls"),
        [s for s in SAMPLE_IBD_STOCKS if s.get("rs_rating", 0) >= 93],
    )

    return tmp_path


# ---------------------------------------------------------------------------
# Level 2: Pipeline Output Tests
# ---------------------------------------------------------------------------

class TestPipelineOutput:
    """Test that the deterministic pipeline produces valid output."""

    def test_pipeline_produces_valid_output(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        assert isinstance(output, ResearchOutput)

    def test_pipeline_finds_stocks(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        assert len(output.stocks) >= 50

    def test_pipeline_separates_etfs(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        # All items in stocks list are type "stock"
        for s in output.stocks:
            assert s.security_type == "stock"

    def test_pipeline_finds_keep_candidates(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        # CLS (99/97), PLTR (99/93), AGI (99/92), etc. should be flagged
        assert len(output.ibd_keep_candidates) > 0
        # Verify all candidates actually meet threshold
        stock_map = {s.symbol: s for s in output.stocks}
        for sym in output.ibd_keep_candidates:
            s = stock_map[sym]
            assert s.composite_rating >= 93
            assert s.rs_rating >= 90

    def test_pipeline_has_multi_source_validated(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        # Stocks appearing on multiple XLS lists should have score≥5
        if output.multi_source_validated:
            stock_map = {s.symbol: s for s in output.stocks}
            for sym in output.multi_source_validated:
                s = stock_map[sym]
                assert s.validation_score >= 5
                assert s.validation_providers >= 2

    def test_pipeline_has_sector_patterns(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        assert len(output.sector_patterns) >= 5
        for sp in output.sector_patterns:
            assert sp.sector in IBD_SECTORS

    def test_pipeline_tracks_sources(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        assert len(output.data_sources_used) >= 1
        # Should include XLS files we created
        source_names = output.data_sources_used
        assert any("BIG_CAP_20" in s for s in source_names)

    def test_pipeline_assigns_preliminary_tiers(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        tiers = [s.preliminary_tier for s in output.stocks if s.preliminary_tier is not None]
        assert len(tiers) > 0
        assert all(t in (1, 2, 3) for t in tiers)

    def test_pipeline_real_symbols_present(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        symbols = {s.symbol for s in output.stocks}
        # Key stocks from SAMPLE_IBD_STOCKS should be present
        for expected in ["NVDA", "CLS", "GOOGL", "MU", "PLTR"]:
            assert expected in symbols, f"{expected} missing from output"

    def test_pipeline_ibd_ratings_preserved(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        stock_map = {s.symbol: s for s in output.stocks}

        nvda = stock_map.get("NVDA")
        assert nvda is not None
        assert nvda.composite_rating == 98
        assert nvda.eps_rating == 99

    def test_pipeline_validation_scores_computed(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        # Stocks on multiple lists should have higher scores
        stock_map = {s.symbol: s for s in output.stocks}
        # CLS is on Big Cap 20 + Sector Leaders = 3+3=6 pts minimum
        cls = stock_map.get("CLS")
        if cls:
            assert cls.validation_score >= 3  # At least on one list

    def test_output_serializable(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        json_str = output.model_dump_json()
        parsed = json.loads(json_str)
        assert "stocks" in parsed
        assert "sector_patterns" in parsed

    def test_output_round_trips(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        json_str = output.model_dump_json()
        restored = ResearchOutput.model_validate_json(json_str)
        assert len(restored.stocks) == len(output.stocks)

    def test_pipeline_assigns_cap_sizes(self, research_data_dir: Path):
        """Pipeline assigns cap_size from static map for known tickers."""
        output = run_research_pipeline(str(research_data_dir))
        stock_map = {s.symbol: s for s in output.stocks}
        # NVDA should be large
        nvda = stock_map.get("NVDA")
        assert nvda is not None
        assert nvda.cap_size == "large"
        # KGC should be mid
        kgc = stock_map.get("KGC")
        if kgc:
            assert kgc.cap_size == "mid"
        # Padding stocks won't be in static map
        pad = stock_map.get("PAD00")
        if pad:
            assert pad.cap_size is None


# ---------------------------------------------------------------------------
# Level 3: Behavioral Tests
# ---------------------------------------------------------------------------

class TestBehavioralBoundaries:
    """Verify the Research Agent stays in its lane."""

    def test_no_buy_sell_language(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        text = output.model_dump_json().lower()
        forbidden = ["buy", "sell", "hold", "recommend", "should purchase"]
        for word in forbidden:
            # Allow in field names like "acc_dis_rating" or source names
            occurrences = text.count(word)
            # Check it's not in reasoning or summary
            for stock in output.stocks:
                assert word not in stock.reasoning.lower(), \
                    f"'{word}' found in reasoning for {stock.symbol}"
            assert word not in output.summary.lower(), \
                f"'{word}' found in summary"

    def test_no_position_sizing(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        dump = output.model_dump_json().lower()
        sizing_terms = ["position size", "allocation", "% of portfolio",
                        "trailing stop", "stop loss"]
        for term in sizing_terms:
            assert term not in dump, f"Position sizing term '{term}' found in output"

    def test_no_sector_allocation_recs(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        # Sector patterns should describe, not prescribe
        for sp in output.sector_patterns:
            evidence_lower = sp.evidence.lower()
            assert "recommend" not in evidence_lower
            assert "should" not in evidence_lower
            assert "allocate" not in evidence_lower

    def test_no_trailing_stops(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        dump = output.model_dump_json().lower()
        assert "trailing stop" not in dump
        assert "stop_loss" not in dump

    def test_all_stocks_have_sources(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        for stock in output.stocks:
            assert len(stock.sources) >= 1, f"{stock.symbol} has no sources"

    def test_all_sectors_are_valid(self, research_data_dir: Path):
        output = run_research_pipeline(str(research_data_dir))
        for stock in output.stocks:
            assert stock.sector in IBD_SECTORS, \
                f"{stock.symbol} has invalid sector: {stock.sector}"

    def test_handles_empty_data_dir(self, tmp_path: Path):
        """Pipeline should handle empty/missing data gracefully."""
        empty_dir = tmp_path / "empty_data"
        empty_dir.mkdir()
        # This should either produce minimal output or raise gracefully
        try:
            output = run_research_pipeline(str(empty_dir))
            # If it succeeds, stocks might be empty — that's fine for graceful handling
        except Exception as e:
            # ValidationError from min stocks or sectors is expected
            assert "too_short" in str(e) or "min_length" in str(e) or "sector" in str(e)


# ---------------------------------------------------------------------------
# Level 2b: CSV Pipeline Tests
# ---------------------------------------------------------------------------

class TestCSVPipeline:
    """Test that the pipeline reads CSV files from data/ibd_xls/."""

    @pytest.fixture
    def csv_data_dir(self, tmp_path: Path) -> Path:
        """Create a data directory with CSV files only."""
        xls_dir = tmp_path / "ibd_xls"
        xls_dir.mkdir()

        # Big Cap 20 as CSV
        create_mock_csv(str(xls_dir / "BIG_CAP_20_1.csv"), SAMPLE_IBD_STOCKS[:20])
        # IBD 50 as CSV
        create_mock_csv(str(xls_dir / "IBD_50_3.csv"), SAMPLE_IBD_STOCKS + PADDING_STOCKS[:30])
        # Sector Leaders as CSV
        create_mock_csv(
            str(xls_dir / "SECTOR_LEADERS_1.csv"),
            [s for s in SAMPLE_IBD_STOCKS if s["composite_rating"] >= 99],
        )
        # Rising Profit Estimates as CSV
        create_mock_csv(
            str(xls_dir / "RISING_PROFIT_ESTIMATES.csv"),
            [s for s in SAMPLE_IBD_STOCKS if s.get("eps_rating", 0) >= 90],
        )
        # RS at New High as CSV
        create_mock_csv(
            str(xls_dir / "RELATIVE_STRENGTH_AT_NEW_HIGH.csv"),
            [s for s in SAMPLE_IBD_STOCKS if s.get("rs_rating", 0) >= 93],
        )

        return tmp_path

    def test_csv_pipeline_produces_valid_output(self, csv_data_dir: Path):
        output = run_research_pipeline(str(csv_data_dir))
        assert isinstance(output, ResearchOutput)

    def test_csv_pipeline_finds_stocks(self, csv_data_dir: Path):
        output = run_research_pipeline(str(csv_data_dir))
        assert len(output.stocks) >= 50

    def test_csv_pipeline_real_symbols_present(self, csv_data_dir: Path):
        output = run_research_pipeline(str(csv_data_dir))
        symbols = {s.symbol for s in output.stocks}
        for expected in ["NVDA", "CLS", "GOOGL", "MU", "PLTR"]:
            assert expected in symbols, f"{expected} missing from CSV output"

    def test_csv_pipeline_ratings_preserved(self, csv_data_dir: Path):
        output = run_research_pipeline(str(csv_data_dir))
        stock_map = {s.symbol: s for s in output.stocks}
        nvda = stock_map.get("NVDA")
        assert nvda is not None
        assert nvda.composite_rating == 98
        assert nvda.eps_rating == 99

    def test_csv_pipeline_tracks_sources(self, csv_data_dir: Path):
        output = run_research_pipeline(str(csv_data_dir))
        assert len(output.data_sources_used) >= 1
        assert any(s.endswith(".csv") for s in output.data_sources_used)

    def test_mixed_xls_and_csv(self, tmp_path: Path):
        """XLS and CSV files in the same directory are both processed and deduplicated."""
        xls_dir = tmp_path / "ibd_xls"
        xls_dir.mkdir()

        # XLS files
        create_mock_xls(str(xls_dir / "BIG_CAP_20_1.xls"), SAMPLE_IBD_STOCKS[:20])
        create_mock_xls(str(xls_dir / "IBD_50_3.xls"), SAMPLE_IBD_STOCKS + PADDING_STOCKS[:30])
        create_mock_xls(
            str(xls_dir / "SECTOR_LEADERS_1.xls"),
            [s for s in SAMPLE_IBD_STOCKS if s["composite_rating"] >= 99],
        )
        create_mock_xls(
            str(xls_dir / "RISING_PROFIT_ESTIMATES.xls"),
            [s for s in SAMPLE_IBD_STOCKS if s.get("eps_rating", 0) >= 90],
        )
        create_mock_xls(
            str(xls_dir / "RELATIVE_STRENGTH_AT_NEW_HIGH.xls"),
            [s for s in SAMPLE_IBD_STOCKS if s.get("rs_rating", 0) >= 93],
        )

        # CSV file with overlapping data — should deduplicate
        create_mock_csv(str(xls_dir / "STOCK_SPOTLIGHT.csv"), SAMPLE_IBD_STOCKS[:10])

        output = run_research_pipeline(str(tmp_path))
        assert isinstance(output, ResearchOutput)

        # Both XLS and CSV sources should appear
        assert any("BIG_CAP_20" in s for s in output.data_sources_used)
        assert any(s.endswith(".csv") for s in output.data_sources_used)

        # Symbols should not be duplicated in the output
        symbols = [s.symbol for s in output.stocks]
        assert len(symbols) == len(set(symbols))


# ---------------------------------------------------------------------------
# Motley Fool Name Resolution Tests
# ---------------------------------------------------------------------------

from ibd_agents.tools.validation_scorer import StockUniverse, _normalize_company


class TestFoolNameResolution:
    """Tests for resolving Motley Fool company names to ticker symbols."""

    @pytest.mark.schema
    def test_normalize_company(self):
        """Common suffixes are stripped and names lowercased."""
        assert _normalize_company("TransMedics Group Inc") == "transmedics"
        assert _normalize_company("Monday.com Ltd") == "mondaycom"
        assert _normalize_company("The Trade Desk") == "the trade desk"
        assert _normalize_company("Camping World Holdings") == "camping world"

    @pytest.mark.schema
    def test_exact_name_match(self):
        """Fool company name matches universe entry exactly after normalization."""
        universe = StockUniverse()
        universe.add_ibd_data([
            {"symbol": "TMDX", "company_name": "TransMedics Group Inc",
             "composite_rating": 90, "rs_rating": 85},
        ])
        records = [{"symbol": "", "company_name": "TransMedics Group",
                     "fool_status": "New Rec", "source_file": "fool.pdf"}]
        resolved = universe.resolve_fool_names(records)
        assert len(resolved) == 1
        assert resolved[0]["symbol"] == "TMDX"

    @pytest.mark.schema
    def test_substring_match(self):
        """Fool company name matches via substring when exact match fails."""
        universe = StockUniverse()
        universe.add_ibd_data([
            {"symbol": "CWH", "company_name": "Camping World Holdings Inc",
             "composite_rating": 85, "rs_rating": 80},
        ])
        records = [{"symbol": "", "company_name": "Camping World",
                     "fool_status": "New Rec", "source_file": "fool.pdf"}]
        resolved = universe.resolve_fool_names(records)
        assert len(resolved) == 1
        assert resolved[0]["symbol"] == "CWH"

    @pytest.mark.schema
    def test_no_match_skipped(self):
        """Unknown company names are skipped (not in universe)."""
        universe = StockUniverse()
        universe.add_ibd_data([
            {"symbol": "AAPL", "company_name": "Apple Inc",
             "composite_rating": 90, "rs_rating": 85},
        ])
        records = [{"symbol": "", "company_name": "Nonexistent Corp",
                     "fool_status": "New Rec", "source_file": "fool.pdf"}]
        resolved = universe.resolve_fool_names(records)
        assert len(resolved) == 0

    @pytest.mark.schema
    def test_existing_symbol_passed_through(self):
        """Records that already have a symbol are passed through unchanged."""
        universe = StockUniverse()
        records = [{"symbol": "AAPL", "company_name": "Apple Inc",
                     "fool_status": "Epic Top", "source_file": "fool.pdf"}]
        resolved = universe.resolve_fool_names(records)
        assert len(resolved) == 1
        assert resolved[0]["symbol"] == "AAPL"

    @pytest.mark.schema
    def test_fool_validation_label_applied(self):
        """After resolution and add_fool_data, Fool validation label is present."""
        universe = StockUniverse()
        universe.add_ibd_data([
            {"symbol": "INTC", "company_name": "Intel Corp",
             "composite_rating": 80, "rs_rating": 70},
        ])
        records = [{"symbol": "", "company_name": "Intel",
                     "fool_status": "New Rec", "source_file": "fool.pdf"}]
        resolved = universe.resolve_fool_names(records)
        universe.add_fool_data(resolved)

        agg = universe._stocks["INTC"]
        assert "Motley Fool New Rec" in agg._validation_labels
