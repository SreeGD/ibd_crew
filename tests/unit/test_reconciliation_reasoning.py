"""
Reconciliation Reasoning — Unit Tests
Tests for enrich_rationale_llm() and _parse_reasoning_response().
"""

from __future__ import annotations

import json

import pytest

from ibd_agents.tools.reconciliation_reasoning import (
    _parse_reasoning_response,
    enrich_rationale_llm,
)


# ---------------------------------------------------------------------------
# _parse_reasoning_response Tests
# ---------------------------------------------------------------------------

class TestParseReasoningResponse:

    @pytest.mark.schema
    def test_valid_response(self):
        data = [
            {
                "symbol": "NVDA",
                "rationale": "SELL: Rotation away from growth cluster — risk officer flagged high sector concentration.",
            }
        ]
        text = json.dumps(data)
        result = _parse_reasoning_response(text, {"NVDA"})
        assert "NVDA" in result
        assert "Rotation" in result["NVDA"]

    @pytest.mark.schema
    def test_unknown_symbol_filtered(self):
        data = [
            {
                "symbol": "UNKNOWN",
                "rationale": "Should be filtered out from the results entirely.",
            }
        ]
        text = json.dumps(data)
        result = _parse_reasoning_response(text, {"NVDA", "AAPL"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_short_rationale_filtered(self):
        """Rationale < 10 chars is filtered out."""
        data = [
            {
                "symbol": "AAPL",
                "rationale": "Sell it",
            }
        ]
        text = json.dumps(data)
        result = _parse_reasoning_response(text, {"AAPL"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_empty_response(self):
        result = _parse_reasoning_response("", {"AAPL"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_non_json_response(self):
        result = _parse_reasoning_response("Sorry, I can't help with that.", {"AAPL"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_multiple_actions(self):
        data = [
            {
                "symbol": "NVDA",
                "rationale": "KEEP: Strong momentum position aligned with growth rotation destination cluster.",
            },
            {
                "symbol": "AAPL",
                "rationale": "BUY: Near-term catalyst (earnings in 10 days) with defensive characteristics.",
            },
            {
                "symbol": "CAT",
                "rationale": "SELL: Risk officer flagged regime misalignment — bear signals in industrial cluster.",
            },
        ]
        text = json.dumps(data)
        result = _parse_reasoning_response(text, {"NVDA", "AAPL", "CAT"})
        assert len(result) == 3

    @pytest.mark.schema
    def test_json_with_markdown_fences(self):
        """Parser handles JSON wrapped in markdown code fences."""
        data = [
            {
                "symbol": "GOOG",
                "rationale": "ADD: Below target allocation — regime supports growth overweight.",
            }
        ]
        text = f"```json\n{json.dumps(data)}\n```"
        result = _parse_reasoning_response(text, {"GOOG"})
        assert "GOOG" in result

    @pytest.mark.schema
    def test_missing_rationale_field(self):
        """Entry without rationale field is skipped."""
        data = [
            {
                "symbol": "META",
            }
        ]
        text = json.dumps(data)
        result = _parse_reasoning_response(text, {"META"})
        assert len(result) == 0


# ---------------------------------------------------------------------------
# enrich_rationale_llm Tests
# ---------------------------------------------------------------------------

class TestEnrichRationaleLlm:

    @pytest.mark.schema
    def test_empty_actions_returns_empty(self):
        """Empty actions list returns empty dict."""
        result = enrich_rationale_llm([], {})
        assert result == {}

    @pytest.mark.schema
    def test_graceful_without_sdk(self):
        """Without anthropic SDK or key, returns empty dict."""
        actions = [
            {"symbol": "NVDA", "action_type": "BUY", "current_pct": 0, "target_pct": 3.0, "priority": "HIGH"},
        ]
        context = {"rotation_verdict": "active", "rotation_type": "cyclical",
                    "risk_status": "APPROVED", "regime": "bull"}
        result = enrich_rationale_llm(actions, context)
        assert isinstance(result, dict)
