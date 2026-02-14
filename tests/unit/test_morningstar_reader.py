"""
Unit tests for Morningstar Pick List PDF Reader.

Tests the pure parsing functions without requiring actual PDF files.
"""

from __future__ import annotations

import pytest

from ibd_agents.tools.morningstar_reader import (
    COLUMN_RANGES,
    _assign_column,
    _clean_company_name,
    _detect_list_type,
    _group_by_rows,
    _is_sector_header,
    _parse_q_rating,
    _safe_float,
)


# ---------------------------------------------------------------------------
# Helpers for creating mock word dicts
# ---------------------------------------------------------------------------

def _make_word(text: str, x0: float, top: float) -> dict:
    """Create a minimal pdfplumber-style word dict."""
    return {"text": text, "x0": x0, "top": top, "x1": x0 + len(text) * 5}


# ---------------------------------------------------------------------------
# Tests: _parse_q_rating
# ---------------------------------------------------------------------------

class TestParseQRating:
    @pytest.mark.schema
    @pytest.mark.parametrize("text,expected", [
        ("QQQQQ", 5),
        ("QQQQ", 4),
        ("QQQ", 3),
        ("QQ", 2),
        ("Q", 1),
    ])
    def test_valid_q_ratings(self, text, expected):
        assert _parse_q_rating(text) == expected

    @pytest.mark.schema
    @pytest.mark.parametrize("text", ["", "Wide", "Narrow", "QQQQQQQ", "abc", "4"])
    def test_invalid_q_ratings_return_zero(self, text):
        assert _parse_q_rating(text) == 0


# ---------------------------------------------------------------------------
# Tests: _assign_column
# ---------------------------------------------------------------------------

class TestAssignColumn:
    @pytest.mark.schema
    def test_company_column(self):
        assert _assign_column(36.0) == "company"

    @pytest.mark.schema
    def test_ticker_column(self):
        assert _assign_column(141.0) == "ticker"

    @pytest.mark.schema
    def test_rating_column(self):
        assert _assign_column(356.0) == "rating"

    @pytest.mark.schema
    def test_moat_column(self):
        assert _assign_column(404.0) == "moat"

    @pytest.mark.schema
    def test_fair_value_column(self):
        assert _assign_column(501.0) == "fair_value"

    @pytest.mark.schema
    def test_price_column(self):
        assert _assign_column(536.0) == "price"

    @pytest.mark.schema
    def test_pfv_column(self):
        assert _assign_column(580.0) == "pfv"

    @pytest.mark.schema
    def test_out_of_range_returns_none(self):
        assert _assign_column(0.0) is None
        assert _assign_column(999.0) is None

    @pytest.mark.schema
    def test_no_overlap_between_market_cap_and_rating(self):
        """Column at 350 should be rating, not market_cap."""
        assert _assign_column(350.0) == "rating"
        assert _assign_column(344.0) == "market_cap"


# ---------------------------------------------------------------------------
# Tests: _is_sector_header
# ---------------------------------------------------------------------------

class TestIsSectorHeader:
    @pytest.mark.schema
    @pytest.mark.parametrize("sector", [
        "Basic Materials", "Communication Services", "Consumer Cyclical",
        "Consumer Defensive", "Energy", "Financial Services", "Healthcare",
        "Industrials", "Real Estate", "Technology", "Utilities",
    ])
    def test_valid_sectors(self, sector):
        words = [_make_word(w, 36.0 + i * 30, 100.0) for i, w in enumerate(sector.split())]
        assert _is_sector_header(words) == sector

    @pytest.mark.schema
    def test_non_sector_returns_none(self):
        words = [_make_word("Equifax", 36.0, 100.0), _make_word("EFX", 141.0, 100.0)]
        assert _is_sector_header(words) is None


# ---------------------------------------------------------------------------
# Tests: _clean_company_name
# ---------------------------------------------------------------------------

class TestCleanCompanyName:
    @pytest.mark.schema
    def test_no_marker(self):
        words = [_make_word("Equifax", 36.0, 100.0)]
        company, marker = _clean_company_name(words)
        assert company == "Equifax"
        assert marker is None

    @pytest.mark.schema
    def test_e_addition_marker(self):
        words = [_make_word("EAirbnb", 28.0, 100.0)]
        company, marker = _clean_company_name(words)
        assert company == "Airbnb"
        assert marker == "E"

    @pytest.mark.schema
    def test_e_marker_multi_word(self):
        words = [_make_word("ECrown", 28.0, 100.0), _make_word("Castle", 53.0, 100.0)]
        company, marker = _clean_company_name(words)
        assert company == "Crown Castle"
        assert marker == "E"

    @pytest.mark.schema
    def test_bracket_close_marker_separate(self):
        words = [_make_word("]", 29.0, 100.0), _make_word("Tesla", 37.0, 100.0)]
        company, marker = _clean_company_name(words)
        assert company == "Tesla"
        assert marker == "]"

    @pytest.mark.schema
    def test_bracket_open_marker_separate(self):
        words = [_make_word("[", 29.0, 100.0), _make_word("Comcast", 37.0, 100.0)]
        company, marker = _clean_company_name(words)
        assert company == "Comcast"
        assert marker == "["

    @pytest.mark.schema
    def test_bracket_embedded_in_word(self):
        words = [_make_word("]Amcor", 36.0, 100.0), _make_word("PLC", 55.0, 100.0)]
        company, marker = _clean_company_name(words)
        assert company == "Amcor PLC"
        assert marker == "]"

    @pytest.mark.schema
    def test_bracket_open_embedded_in_word(self):
        words = [_make_word("[United", 29.0, 100.0), _make_word("Rentals", 53.0, 100.0)]
        company, marker = _clean_company_name(words)
        assert company == "United Rentals"
        assert marker == "["

    @pytest.mark.schema
    def test_eog_no_marker(self):
        """EOG at x=36 should NOT have E marker stripped."""
        words = [_make_word("EOG", 36.0, 100.0), _make_word("Resources", 55.0, 100.0)]
        company, marker = _clean_company_name(words)
        assert company == "EOG Resources"
        assert marker is None

    @pytest.mark.schema
    def test_empty_words(self):
        company, marker = _clean_company_name([])
        assert company == ""
        assert marker is None


# ---------------------------------------------------------------------------
# Tests: _group_by_rows
# ---------------------------------------------------------------------------

class TestGroupByRows:
    @pytest.mark.schema
    def test_groups_words_by_y_position(self):
        words = [
            _make_word("A", 36.0, 100.0),
            _make_word("B", 141.0, 100.5),  # Same row (within tolerance)
            _make_word("C", 36.0, 120.0),  # Different row
        ]
        rows = _group_by_rows(words)
        assert len(rows) == 2
        assert len(rows[0]) == 2
        assert len(rows[1]) == 1

    @pytest.mark.schema
    def test_empty_input(self):
        assert _group_by_rows([]) == []

    @pytest.mark.schema
    def test_rows_sorted_by_x(self):
        words = [
            _make_word("B", 141.0, 100.0),
            _make_word("A", 36.0, 100.0),
        ]
        rows = _group_by_rows(words)
        assert rows[0][0]["text"] == "A"
        assert rows[0][1]["text"] == "B"


# ---------------------------------------------------------------------------
# Tests: _safe_float
# ---------------------------------------------------------------------------

class TestSafeFloat:
    @pytest.mark.schema
    def test_normal_float(self):
        assert _safe_float("265.00") == 265.0

    @pytest.mark.schema
    def test_comma_separated(self):
        assert _safe_float("5,300.00") == 5300.0

    @pytest.mark.schema
    def test_invalid_returns_none(self):
        assert _safe_float("N/A") is None
        assert _safe_float("") is None


# ---------------------------------------------------------------------------
# Tests: _detect_list_type
# ---------------------------------------------------------------------------

class TestDetectListType:
    @pytest.mark.schema
    def test_large_cap(self):
        class MockPage:
            def extract_text(self):
                return "US Large Cap Pick List\nFebruary 2026"
        assert _detect_list_type([MockPage()]) == "large"

    @pytest.mark.schema
    def test_mid_cap(self):
        class MockPage:
            def extract_text(self):
                return "US Mid Cap Pick List\nFebruary 2026"
        assert _detect_list_type([MockPage()]) == "mid"

    @pytest.mark.schema
    def test_unknown(self):
        class MockPage:
            def extract_text(self):
                return "Some other document"
        assert _detect_list_type([MockPage()]) == "unknown"

    @pytest.mark.schema
    def test_empty_pages(self):
        assert _detect_list_type([]) == "unknown"


# ---------------------------------------------------------------------------
# Tests: Validation Scorer Integration
# ---------------------------------------------------------------------------

class TestValidationScorerIntegration:
    @pytest.mark.schema
    def test_add_morningstar_5star(self):
        from ibd_agents.tools.validation_scorer import StockUniverse
        universe = StockUniverse()
        records = [
            {"symbol": "ADBE", "company_name": "Adobe", "star_count": 5,
             "economic_moat": "Wide", "fair_value": 590.0, "price": 420.0,
             "price_to_fair_value": 0.71, "uncertainty": "Medium",
             "cap_list": "large",
             "is_deletion": False, "source_file": "morningstar.pdf"},
        ]
        added = universe.add_morningstar_data(records)
        assert added == 1
        scored = universe.get_all_scored()
        assert len(scored) == 1
        assert scored[0]["symbol"] == "ADBE"
        assert scored[0]["other_ratings"]["Morningstar"] == "5-star"
        assert scored[0]["validation_score"] == 3  # "Morningstar 5-star" = 3 pts
        # Verify metadata preserved
        assert scored[0]["morningstar_rating"] == "5-star"
        assert scored[0]["economic_moat"] == "Wide"
        assert scored[0]["fair_value"] == 590.0
        assert scored[0]["morningstar_price"] == 420.0
        assert scored[0]["price_to_fair_value"] == 0.71
        assert scored[0]["morningstar_uncertainty"] == "Medium"
        # Verify friendly source name
        assert "Morningstar Large Cap" in scored[0]["sources"]

    @pytest.mark.schema
    def test_add_morningstar_4star(self):
        from ibd_agents.tools.validation_scorer import StockUniverse
        universe = StockUniverse()
        records = [
            {"symbol": "MSFT", "company_name": "Microsoft", "star_count": 4,
             "economic_moat": "Wide", "fair_value": 490.0, "price": 410.0,
             "price_to_fair_value": 0.84, "uncertainty": "Medium",
             "cap_list": "large",
             "is_deletion": False, "source_file": "morningstar.pdf"},
        ]
        added = universe.add_morningstar_data(records)
        assert added == 1
        scored = universe.get_all_scored()
        assert scored[0]["other_ratings"]["Morningstar"] == "4-star"
        assert scored[0]["validation_score"] == 2  # "Morningstar 4-star" = 2 pts
        assert scored[0]["morningstar_rating"] == "4-star"
        assert scored[0]["economic_moat"] == "Wide"

    @pytest.mark.schema
    def test_mid_cap_friendly_source_name(self):
        from ibd_agents.tools.validation_scorer import StockUniverse
        universe = StockUniverse()
        records = [
            {"symbol": "EFX", "company_name": "Equifax", "star_count": 4,
             "cap_list": "mid",
             "is_deletion": False, "source_file": "guid.pdf"},
        ]
        universe.add_morningstar_data(records)
        scored = universe.get_all_scored()
        assert "Morningstar Mid Cap" in scored[0]["sources"]
        assert "guid.pdf" not in scored[0]["sources"]

    @pytest.mark.schema
    def test_skip_3star(self):
        from ibd_agents.tools.validation_scorer import StockUniverse
        universe = StockUniverse()
        records = [
            {"symbol": "TSLA", "company_name": "Tesla", "star_count": 3,
             "is_deletion": False, "source_file": "morningstar.pdf"},
        ]
        added = universe.add_morningstar_data(records)
        assert added == 0  # 3-star not added

    @pytest.mark.schema
    def test_skip_deletions(self):
        from ibd_agents.tools.validation_scorer import StockUniverse
        universe = StockUniverse()
        records = [
            {"symbol": "NVDA", "company_name": "NVIDIA", "star_count": 4,
             "is_deletion": True, "source_file": "morningstar.pdf"},
        ]
        added = universe.add_morningstar_data(records)
        assert added == 0

    @pytest.mark.schema
    def test_morningstar_contributes_to_multi_source(self):
        """Morningstar + IBD should trigger multi-source validation."""
        from ibd_agents.tools.validation_scorer import StockUniverse
        universe = StockUniverse()
        # Add IBD data (3+ points from IBD)
        universe.add_ibd_data([{
            "symbol": "ADBE",
            "company_name": "Adobe",
            "composite_rating": 95,
            "rs_rating": 92,
            "eps_rating": 85,
            "ibd_list": "IBD 50",
            "source_file": "ibd50.xls",
        }])
        # Add Morningstar 5-star (3 points)
        universe.add_morningstar_data([
            {"symbol": "ADBE", "star_count": 5, "is_deletion": False,
             "source_file": "morningstar.pdf"},
        ])
        scored = universe.get_all_scored()
        adbe = scored[0]
        assert adbe["symbol"] == "ADBE"
        assert adbe["is_multi_source_validated"] is True
        assert adbe["validation_providers"] >= 2


# ---------------------------------------------------------------------------
# Tests: File Lister Integration
# ---------------------------------------------------------------------------

class TestFileListerIntegration:
    @pytest.mark.schema
    def test_classify_morningstar_file(self):
        from ibd_agents.tools.file_lister import classify_file
        assert classify_file("Morningstar_US_Large_Cap_Pick_List.pdf") == "morningstar_pdf"
        assert classify_file("morningstar_picks.pdf") == "morningstar_pdf"

    @pytest.mark.schema
    def test_expected_files_includes_morningstar(self):
        from ibd_agents.tools.file_lister import EXPECTED_FILES
        assert "morningstar_pdf" in EXPECTED_FILES
        assert len(EXPECTED_FILES["morningstar_pdf"]) == 2


# ---------------------------------------------------------------------------
# Tests: Column Ranges Non-Overlapping
# ---------------------------------------------------------------------------

class TestColumnRanges:
    @pytest.mark.schema
    def test_no_overlapping_ranges(self):
        """Ensure no two column ranges overlap."""
        ranges = sorted(COLUMN_RANGES.values())
        for i in range(len(ranges) - 1):
            _, hi = ranges[i]
            lo_next, _ = ranges[i + 1]
            assert hi <= lo_next, (
                f"Overlap between range ending at {hi} and starting at {lo_next}"
            )
