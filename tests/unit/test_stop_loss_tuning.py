"""
Stop-Loss Tuning — Unit Tests
Tests for compute_stop_recommendation(), _parse_stop_loss_response(),
and constants.
"""

from __future__ import annotations

import json

import pytest

from ibd_agents.tools.stop_loss_tuning import (
    STOP_RANGE,
    VALID_VOLATILITY_FLAGS,
    _parse_stop_loss_response,
    compute_stop_recommendation,
)


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    @pytest.mark.schema
    def test_stop_range_3_tiers(self):
        """Stop range defined for tiers 1-3."""
        assert set(STOP_RANGE.keys()) == {1, 2, 3}

    @pytest.mark.schema
    def test_stop_range_t1_wider_than_t3(self):
        """T1 has wider stop range than T3."""
        assert STOP_RANGE[1][1] > STOP_RANGE[3][1]

    @pytest.mark.schema
    def test_volatility_flags(self):
        """Three volatility flags: high, normal, low."""
        assert set(VALID_VOLATILITY_FLAGS) == {"high", "normal", "low"}


# ---------------------------------------------------------------------------
# compute_stop_recommendation Tests
# ---------------------------------------------------------------------------

class TestComputeStopRecommendation:

    @pytest.mark.schema
    def test_clamp_below_tier_min(self):
        """LLM stop below tier min is clamped up."""
        result = compute_stop_recommendation(22.0, 5.0, 1)
        assert result == STOP_RANGE[1][0]  # 10.0

    @pytest.mark.schema
    def test_clamp_above_tier_max(self):
        """LLM stop above tier max is clamped down."""
        result = compute_stop_recommendation(22.0, 30.0, 1)
        assert result == STOP_RANGE[1][1]  # 25.0

    @pytest.mark.schema
    def test_within_range_passes_through(self):
        """LLM stop within range passes through."""
        result = compute_stop_recommendation(22.0, 18.0, 1)
        assert result == 18.0

    @pytest.mark.schema
    def test_t2_range(self):
        """T2 clamping uses (8.0, 20.0) range."""
        assert compute_stop_recommendation(18.0, 3.0, 2) == 8.0
        assert compute_stop_recommendation(18.0, 25.0, 2) == 20.0

    @pytest.mark.schema
    def test_t3_range(self):
        """T3 clamping uses (5.0, 15.0) range."""
        assert compute_stop_recommendation(12.0, 2.0, 3) == 5.0
        assert compute_stop_recommendation(12.0, 20.0, 3) == 15.0

    @pytest.mark.schema
    def test_unknown_tier_default_range(self):
        """Unknown tier uses (5.0, 25.0) default range."""
        result = compute_stop_recommendation(15.0, 3.0, 99)
        assert result == 5.0

    @pytest.mark.schema
    def test_result_rounded(self):
        """Result is rounded to 1 decimal place."""
        result = compute_stop_recommendation(22.0, 17.777, 1)
        assert result == 17.8


# ---------------------------------------------------------------------------
# _parse_stop_loss_response Tests
# ---------------------------------------------------------------------------

class TestParseStopLossResponse:

    @pytest.mark.schema
    def test_valid_response(self):
        data = [
            {
                "symbol": "NVDA",
                "recommended_stop_pct": 18.5,
                "reason": "High-beta semiconductor with elevated volatility profile.",
                "volatility_flag": "high",
            }
        ]
        text = json.dumps(data)
        result = _parse_stop_loss_response(text, {"NVDA"})
        assert "NVDA" in result
        assert result["NVDA"]["recommended_stop_pct"] == 18.5
        assert result["NVDA"]["volatility_flag"] == "high"

    @pytest.mark.schema
    def test_unknown_symbol_filtered(self):
        data = [
            {
                "symbol": "UNKNOWN",
                "recommended_stop_pct": 15.0,
                "reason": "Should be filtered out from results.",
                "volatility_flag": "normal",
            }
        ]
        text = json.dumps(data)
        result = _parse_stop_loss_response(text, {"NVDA", "AAPL"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_stop_pct_clamped_low(self):
        """Stop pct below 5.0 is clamped to 5.0."""
        data = [
            {
                "symbol": "AAPL",
                "recommended_stop_pct": 2.0,
                "reason": "Very tight stop recommendation for testing.",
                "volatility_flag": "low",
            }
        ]
        text = json.dumps(data)
        result = _parse_stop_loss_response(text, {"AAPL"})
        assert result["AAPL"]["recommended_stop_pct"] == 5.0

    @pytest.mark.schema
    def test_stop_pct_clamped_high(self):
        """Stop pct above 35.0 is clamped to 35.0."""
        data = [
            {
                "symbol": "TSLA",
                "recommended_stop_pct": 50.0,
                "reason": "Very wide stop recommendation for testing.",
                "volatility_flag": "high",
            }
        ]
        text = json.dumps(data)
        result = _parse_stop_loss_response(text, {"TSLA"})
        assert result["TSLA"]["recommended_stop_pct"] == 35.0

    @pytest.mark.schema
    def test_missing_stop_pct_skipped(self):
        """Missing recommended_stop_pct skips entry."""
        data = [
            {
                "symbol": "GOOG",
                "reason": "No stop pct provided in this response.",
                "volatility_flag": "normal",
            }
        ]
        text = json.dumps(data)
        result = _parse_stop_loss_response(text, {"GOOG"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_invalid_volatility_defaults_to_normal(self):
        """Invalid volatility flag defaults to 'normal'."""
        data = [
            {
                "symbol": "META",
                "recommended_stop_pct": 16.0,
                "reason": "Test with invalid volatility flag value.",
                "volatility_flag": "extreme",
            }
        ]
        text = json.dumps(data)
        result = _parse_stop_loss_response(text, {"META"})
        assert result["META"]["volatility_flag"] == "normal"

    @pytest.mark.schema
    def test_short_reason_gets_default(self):
        """Short reason is replaced with default."""
        data = [
            {
                "symbol": "AMZN",
                "recommended_stop_pct": 20.0,
                "reason": "Short",
                "volatility_flag": "normal",
            }
        ]
        text = json.dumps(data)
        result = _parse_stop_loss_response(text, {"AMZN"})
        assert len(result["AMZN"]["reason"]) >= 10

    @pytest.mark.schema
    def test_empty_response(self):
        result = _parse_stop_loss_response("", {"AAPL"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_non_json_response(self):
        result = _parse_stop_loss_response("Sorry, I can't help.", {"AAPL"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_multiple_positions(self):
        data = [
            {
                "symbol": "AAPL",
                "recommended_stop_pct": 18.0,
                "reason": "Moderate volatility large-cap technology stock.",
                "volatility_flag": "normal",
            },
            {
                "symbol": "NVDA",
                "recommended_stop_pct": 22.0,
                "reason": "High-beta semiconductor with catalyst exposure.",
                "volatility_flag": "high",
            },
        ]
        text = json.dumps(data)
        result = _parse_stop_loss_response(text, {"AAPL", "NVDA"})
        assert len(result) == 2
        assert result["AAPL"]["recommended_stop_pct"] == 18.0
        assert result["NVDA"]["recommended_stop_pct"] == 22.0
