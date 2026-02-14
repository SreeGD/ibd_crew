"""
Cap Size Classifier — Static Map & Tool Tests
Level 1: Pure function tests, no LLM calls, no file I/O.

Tests the static cap-size mapping, the classify_caps_static helper,
and the CapClassifierTool wrapper.
"""

from __future__ import annotations

import pytest

from ibd_agents.tools.cap_classifier import (
    KNOWN_CAPS,
    VALID_CAP_SIZES,
    CapClassifierTool,
    classify_caps_static,
)
from tests.fixtures.conftest import SAMPLE_IBD_STOCKS


# ---------------------------------------------------------------------------
# TestStaticCapMap
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestStaticCapMap:
    """Validate the KNOWN_CAPS static mapping."""

    def test_known_large_caps(self):
        """Well-known mega/large-cap tickers map to 'large'."""
        for sym in ("AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"):
            assert KNOWN_CAPS[sym] == "large", f"{sym} should be large"

    def test_known_mid_caps(self):
        """Mid-cap tickers map to 'mid'."""
        for sym in ("KGC", "SMCI", "CRDO"):
            assert KNOWN_CAPS[sym] == "mid", f"{sym} should be mid"

    def test_known_small_caps(self):
        """Small-cap tickers map to 'small'."""
        for sym in ("LCID", "HROW", "ANAB"):
            assert KNOWN_CAPS[sym] == "small", f"{sym} should be small"

    def test_all_values_valid(self):
        """Every value in the static map is one of the valid cap sizes."""
        for sym, cap in KNOWN_CAPS.items():
            assert cap in VALID_CAP_SIZES, f"{sym} has invalid cap '{cap}'"

    def test_sample_ibd_stocks_covered(self):
        """Every ticker from SAMPLE_IBD_STOCKS appears in KNOWN_CAPS."""
        for stock in SAMPLE_IBD_STOCKS:
            sym = stock["symbol"]
            assert sym in KNOWN_CAPS, f"{sym} from SAMPLE_IBD_STOCKS missing in KNOWN_CAPS"


# ---------------------------------------------------------------------------
# TestClassifyCapsStatic
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestClassifyCapsStatic:
    """Test the classify_caps_static pure function."""

    def test_known_symbols(self):
        """Known symbols return correct cap sizes."""
        result = classify_caps_static(["AAPL", "NVDA", "KGC"])
        assert result["AAPL"] == "large"
        assert result["NVDA"] == "large"
        assert result["KGC"] == "mid"

    def test_unknown_symbols_omitted(self):
        """Unknown symbols are not included in the result."""
        result = classify_caps_static(["AAPL", "XYZZY"])
        assert "AAPL" in result
        assert "XYZZY" not in result

    def test_case_insensitive(self):
        """Lowercase symbols are resolved against the uppercase static map."""
        result = classify_caps_static(["aapl", "nvda"])
        assert "aapl" in result
        assert "nvda" in result

    def test_empty_list(self):
        """Empty input returns empty dict."""
        result = classify_caps_static([])
        assert result == {}

    def test_ibd_list_hint_fallback(self):
        """Unknown symbol falls back to IBD list hint when available."""
        result = classify_caps_static(
            ["UNKNOWN1"],
            ibd_lists_map={"UNKNOWN1": ["IBD Big Cap 20"]},
        )
        assert result["UNKNOWN1"] == "large"

    def test_static_map_takes_priority_over_hint(self):
        """Static map value wins over IBD list hint."""
        result = classify_caps_static(
            ["KGC"],
            ibd_lists_map={"KGC": ["IBD Big Cap 20"]},
        )
        assert result["KGC"] == "mid"


# ---------------------------------------------------------------------------
# TestCapClassifierTool
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestCapClassifierTool:
    """Test the CrewAI CapClassifierTool wrapper."""

    def test_tool_has_correct_name(self):
        """Tool name matches expected identifier."""
        tool = CapClassifierTool()
        assert tool.name == "classify_cap_sizes"

    def test_tool_run_returns_string(self):
        """_run returns a human-readable string containing results."""
        tool = CapClassifierTool()
        output = tool._run(["AAPL", "NVDA"])
        assert isinstance(output, str)
        assert "AAPL" in output
        assert "large" in output
