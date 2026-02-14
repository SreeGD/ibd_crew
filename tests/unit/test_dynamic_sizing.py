"""
Dynamic Position Sizing — Unit Tests
Tests for compute_size_adjustment(), _parse_sizing_response(),
and constants.
"""

from __future__ import annotations

import json

import pytest

from ibd_agents.tools.dynamic_sizing import (
    MAX_ADJUSTMENT,
    _parse_sizing_response,
    compute_size_adjustment,
)


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    @pytest.mark.schema
    def test_max_adjustment_3_tiers(self):
        """Max adjustment defined for tiers 1-3."""
        assert set(MAX_ADJUSTMENT.keys()) == {1, 2, 3}

    @pytest.mark.schema
    def test_t1_allows_largest_adjustment(self):
        """T1 momentum allows largest adjustment."""
        assert MAX_ADJUSTMENT[1] > MAX_ADJUSTMENT[2] > MAX_ADJUSTMENT[3]

    @pytest.mark.schema
    def test_all_adjustments_positive(self):
        """All max adjustments are positive."""
        for v in MAX_ADJUSTMENT.values():
            assert v > 0


# ---------------------------------------------------------------------------
# compute_size_adjustment Tests
# ---------------------------------------------------------------------------

class TestComputeSizeAdjustment:

    @pytest.mark.schema
    def test_clamp_positive_t1(self):
        """T1 positive adjustment clamped to 1.5."""
        assert compute_size_adjustment(3.0, 1) == 1.5

    @pytest.mark.schema
    def test_clamp_negative_t1(self):
        """T1 negative adjustment clamped to -1.5."""
        assert compute_size_adjustment(-3.0, 1) == -1.5

    @pytest.mark.schema
    def test_within_range_passes_through(self):
        """Adjustment within range passes through."""
        assert compute_size_adjustment(0.5, 1) == 0.5

    @pytest.mark.schema
    def test_t2_clamp(self):
        """T2 clamped to ±1.0."""
        assert compute_size_adjustment(2.0, 2) == 1.0
        assert compute_size_adjustment(-2.0, 2) == -1.0

    @pytest.mark.schema
    def test_t3_clamp(self):
        """T3 clamped to ±0.5."""
        assert compute_size_adjustment(1.0, 3) == 0.5
        assert compute_size_adjustment(-1.0, 3) == -0.5

    @pytest.mark.schema
    def test_unknown_tier_default(self):
        """Unknown tier uses 1.0 default max."""
        assert compute_size_adjustment(2.0, 99) == 1.0

    @pytest.mark.schema
    def test_zero_passes_through(self):
        """Zero adjustment passes through all tiers."""
        for tier in (1, 2, 3):
            assert compute_size_adjustment(0.0, tier) == 0.0

    @pytest.mark.schema
    def test_result_rounded(self):
        """Result is rounded to 2 decimal places."""
        result = compute_size_adjustment(0.777, 1)
        assert result == 0.78


# ---------------------------------------------------------------------------
# _parse_sizing_response Tests
# ---------------------------------------------------------------------------

class TestParseSizingResponse:

    @pytest.mark.schema
    def test_valid_response(self):
        data = [
            {
                "symbol": "NVDA",
                "volatility_score": 8,
                "size_adjustment_pct": -0.5,
                "reason": "High-beta semiconductor with imminent earnings catalyst.",
            }
        ]
        text = json.dumps(data)
        result = _parse_sizing_response(text, {"NVDA"})
        assert "NVDA" in result
        assert result["NVDA"]["volatility_score"] == 8
        assert result["NVDA"]["size_adjustment_pct"] == -0.5

    @pytest.mark.schema
    def test_unknown_symbol_filtered(self):
        data = [
            {
                "symbol": "UNKNOWN",
                "volatility_score": 5,
                "size_adjustment_pct": 0.0,
                "reason": "Should be filtered out from the results.",
            }
        ]
        text = json.dumps(data)
        result = _parse_sizing_response(text, {"NVDA"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_volatility_score_clamped_low(self):
        """Volatility score below 1 clamped to 1."""
        data = [
            {
                "symbol": "AAPL",
                "volatility_score": -5,
                "size_adjustment_pct": 0.3,
                "reason": "Test with negative volatility score value.",
            }
        ]
        text = json.dumps(data)
        result = _parse_sizing_response(text, {"AAPL"})
        assert result["AAPL"]["volatility_score"] == 1

    @pytest.mark.schema
    def test_volatility_score_clamped_high(self):
        """Volatility score above 10 clamped to 10."""
        data = [
            {
                "symbol": "TSLA",
                "volatility_score": 15,
                "size_adjustment_pct": -1.0,
                "reason": "Test with very high volatility score value.",
            }
        ]
        text = json.dumps(data)
        result = _parse_sizing_response(text, {"TSLA"})
        assert result["TSLA"]["volatility_score"] == 10

    @pytest.mark.schema
    def test_adjustment_clamped_range(self):
        """size_adjustment_pct clamped to [-2.0, 2.0]."""
        data = [
            {
                "symbol": "GOOG",
                "volatility_score": 5,
                "size_adjustment_pct": 5.0,
                "reason": "Test with out-of-range positive adjustment.",
            }
        ]
        text = json.dumps(data)
        result = _parse_sizing_response(text, {"GOOG"})
        assert result["GOOG"]["size_adjustment_pct"] == 2.0

    @pytest.mark.schema
    def test_missing_volatility_score_defaults(self):
        """Missing volatility_score defaults to 5."""
        data = [
            {
                "symbol": "META",
                "size_adjustment_pct": 0.3,
                "reason": "Test with missing volatility score field.",
            }
        ]
        text = json.dumps(data)
        result = _parse_sizing_response(text, {"META"})
        assert result["META"]["volatility_score"] == 5

    @pytest.mark.schema
    def test_missing_adjustment_defaults_to_zero(self):
        """Missing size_adjustment_pct defaults to 0.0."""
        data = [
            {
                "symbol": "AMZN",
                "volatility_score": 4,
                "reason": "Test with missing adjustment percentage.",
            }
        ]
        text = json.dumps(data)
        result = _parse_sizing_response(text, {"AMZN"})
        assert result["AMZN"]["size_adjustment_pct"] == 0.0

    @pytest.mark.schema
    def test_short_reason_gets_default(self):
        """Short reason replaced with default."""
        data = [
            {
                "symbol": "MSFT",
                "volatility_score": 3,
                "size_adjustment_pct": 0.5,
                "reason": "Short",
            }
        ]
        text = json.dumps(data)
        result = _parse_sizing_response(text, {"MSFT"})
        assert len(result["MSFT"]["reason"]) >= 10

    @pytest.mark.schema
    def test_empty_response(self):
        result = _parse_sizing_response("", {"AAPL"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_non_json_response(self):
        result = _parse_sizing_response("I cannot provide that.", {"AAPL"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_multiple_positions(self):
        data = [
            {
                "symbol": "AAPL",
                "volatility_score": 3,
                "size_adjustment_pct": 0.5,
                "reason": "Low-volatility large-cap suitable for overweight.",
            },
            {
                "symbol": "NVDA",
                "volatility_score": 8,
                "size_adjustment_pct": -0.8,
                "reason": "High-beta semiconductor with elevated volatility.",
            },
        ]
        text = json.dumps(data)
        result = _parse_sizing_response(text, {"AAPL", "NVDA"})
        assert len(result) == 2
        assert result["AAPL"]["size_adjustment_pct"] == 0.5
        assert result["NVDA"]["size_adjustment_pct"] == -0.8
