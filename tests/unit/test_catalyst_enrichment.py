"""
Catalyst Enrichment — Unit Tests
Tests for compute_catalyst_adjustment() and _parse_catalyst_response().
"""

from __future__ import annotations

import json

import pytest

from ibd_agents.tools.catalyst_enrichment import (
    CATALYST_TYPES,
    _parse_catalyst_response,
    compute_catalyst_adjustment,
)


# ---------------------------------------------------------------------------
# Conviction Adjustment Tests
# ---------------------------------------------------------------------------

class TestComputeCatalystAdjustment:

    @pytest.mark.schema
    def test_imminent_catalyst_14_days(self):
        assert compute_catalyst_adjustment(14) == 2

    @pytest.mark.schema
    def test_imminent_catalyst_7_days(self):
        assert compute_catalyst_adjustment(7) == 2

    @pytest.mark.schema
    def test_imminent_catalyst_0_days(self):
        assert compute_catalyst_adjustment(0) == 2

    @pytest.mark.schema
    def test_near_term_catalyst_30_days(self):
        assert compute_catalyst_adjustment(30) == 1

    @pytest.mark.schema
    def test_near_term_catalyst_15_days(self):
        assert compute_catalyst_adjustment(15) == 1

    @pytest.mark.schema
    def test_far_catalyst_60_days(self):
        assert compute_catalyst_adjustment(60) == 0

    @pytest.mark.schema
    def test_far_catalyst_31_days(self):
        assert compute_catalyst_adjustment(31) == 0

    @pytest.mark.schema
    def test_none_returns_zero(self):
        assert compute_catalyst_adjustment(None) == 0


# ---------------------------------------------------------------------------
# Parse Response Tests
# ---------------------------------------------------------------------------

class TestParseCatalystResponse:

    @pytest.mark.schema
    def test_valid_response(self):
        data = [
            {
                "symbol": "AAPL",
                "catalyst_date": "2026-04-25",
                "catalyst_type": "earnings",
                "catalyst_description": "Q2 FY2026 earnings report.",
                "days_until": 74,
            }
        ]
        text = json.dumps(data)
        result = _parse_catalyst_response(text, {"AAPL"})
        assert "AAPL" in result
        assert result["AAPL"]["catalyst_date"] == "2026-04-25"
        assert result["AAPL"]["catalyst_type"] == "earnings"
        assert result["AAPL"]["days_until"] == 74

    @pytest.mark.schema
    def test_invalid_catalyst_type_defaults_to_other(self):
        data = [
            {
                "symbol": "NVDA",
                "catalyst_date": "2026-05-20",
                "catalyst_type": "ipo",
                "catalyst_description": "Not a valid type.",
                "days_until": 99,
            }
        ]
        text = json.dumps(data)
        result = _parse_catalyst_response(text, {"NVDA"})
        assert result["NVDA"]["catalyst_type"] == "other"

    @pytest.mark.schema
    def test_invalid_date_format_nullified(self):
        data = [
            {
                "symbol": "TSLA",
                "catalyst_date": "04/25/2026",
                "catalyst_type": "earnings",
                "catalyst_description": "Earnings date in wrong format.",
                "days_until": 30,
            }
        ]
        text = json.dumps(data)
        result = _parse_catalyst_response(text, {"TSLA"})
        assert result["TSLA"]["catalyst_date"] is None

    @pytest.mark.schema
    def test_impossible_date_nullified(self):
        data = [
            {
                "symbol": "META",
                "catalyst_date": "2026-02-30",
                "catalyst_type": "earnings",
                "catalyst_description": "Feb 30 does not exist.",
                "days_until": 20,
            }
        ]
        text = json.dumps(data)
        result = _parse_catalyst_response(text, {"META"})
        assert result["META"]["catalyst_date"] is None

    @pytest.mark.schema
    def test_unknown_symbol_filtered(self):
        data = [
            {
                "symbol": "UNKNOWN",
                "catalyst_date": "2026-03-15",
                "catalyst_type": "earnings",
                "catalyst_description": "Should be filtered.",
                "days_until": 33,
            }
        ]
        text = json.dumps(data)
        result = _parse_catalyst_response(text, {"AAPL", "NVDA"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_negative_days_clipped_to_zero(self):
        data = [
            {
                "symbol": "GOOG",
                "catalyst_date": "2026-01-15",
                "catalyst_type": "earnings",
                "catalyst_description": "Past date.",
                "days_until": -5,
            }
        ]
        text = json.dumps(data)
        result = _parse_catalyst_response(text, {"GOOG"})
        assert result["GOOG"]["days_until"] == 0

    @pytest.mark.schema
    def test_days_over_365_clipped(self):
        data = [
            {
                "symbol": "MSFT",
                "catalyst_date": "2027-06-01",
                "catalyst_type": "product_launch",
                "catalyst_description": "Far future product.",
                "days_until": 500,
            }
        ]
        text = json.dumps(data)
        result = _parse_catalyst_response(text, {"MSFT"})
        assert result["MSFT"]["days_until"] == 365

    @pytest.mark.schema
    def test_null_fields_handled(self):
        data = [
            {
                "symbol": "AMZN",
                "catalyst_date": None,
                "catalyst_type": "other",
                "catalyst_description": "No known catalyst.",
                "days_until": None,
            }
        ]
        text = json.dumps(data)
        result = _parse_catalyst_response(text, {"AMZN"})
        assert result["AMZN"]["catalyst_date"] is None
        assert result["AMZN"]["days_until"] is None

    @pytest.mark.schema
    def test_empty_response(self):
        result = _parse_catalyst_response("", {"AAPL"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_non_json_response(self):
        result = _parse_catalyst_response("Sorry, I can't help with that.", {"AAPL"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_multiple_stocks(self):
        data = [
            {
                "symbol": "AAPL",
                "catalyst_date": "2026-04-25",
                "catalyst_type": "earnings",
                "catalyst_description": "Q2 earnings.",
                "days_until": 74,
            },
            {
                "symbol": "NVDA",
                "catalyst_date": "2026-02-26",
                "catalyst_type": "earnings",
                "catalyst_description": "Q4 FY2026 earnings.",
                "days_until": 16,
            },
        ]
        text = json.dumps(data)
        result = _parse_catalyst_response(text, {"AAPL", "NVDA"})
        assert len(result) == 2
        assert result["AAPL"]["days_until"] == 74
        assert result["NVDA"]["days_until"] == 16

    @pytest.mark.schema
    def test_all_catalyst_types_accepted(self):
        for ct in CATALYST_TYPES:
            data = [
                {
                    "symbol": "TEST",
                    "catalyst_date": "2026-06-01",
                    "catalyst_type": ct,
                    "catalyst_description": f"Test {ct} catalyst.",
                    "days_until": 50,
                }
            ]
            text = json.dumps(data)
            result = _parse_catalyst_response(text, {"TEST"})
            assert result["TEST"]["catalyst_type"] == ct
