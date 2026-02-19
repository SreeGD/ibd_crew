"""
Target Return Enrichment — Unit Tests
Tests for enrich_selection_rationale_llm(), enrich_construction_narrative_llm(),
enrich_alternative_reasoning_llm(), and their parsers.
"""

from __future__ import annotations

import json

import pytest

from ibd_agents.tools.target_return_enrichment import (
    _parse_alternative_reasoning_response,
    _parse_selection_rationale_response,
    enrich_alternative_reasoning_llm,
    enrich_construction_narrative_llm,
    enrich_selection_rationale_llm,
)


# ---------------------------------------------------------------------------
# _parse_selection_rationale_response Tests
# ---------------------------------------------------------------------------

class TestParseSelectionRationaleResponse:

    @pytest.mark.schema
    def test_valid_response(self):
        """Valid JSON array parsed correctly."""
        data = [
            {
                "symbol": "NVDA",
                "rationale": "RS breakout above 95, EPS momentum in leading Chips sector with institutional accumulation.",
            }
        ]
        text = json.dumps(data)
        result = _parse_selection_rationale_response(text, {"NVDA"})
        assert "NVDA" in result
        assert "RS breakout" in result["NVDA"]

    @pytest.mark.schema
    def test_unknown_symbol_filtered(self):
        """Symbol not in valid_symbols is filtered out."""
        data = [
            {
                "symbol": "UNKNOWN",
                "rationale": "This should be filtered because symbol not in valid set.",
            }
        ]
        text = json.dumps(data)
        result = _parse_selection_rationale_response(text, {"NVDA", "AAPL"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_short_rationale_filtered(self):
        """Rationale < 20 chars is filtered out."""
        data = [
            {
                "symbol": "NVDA",
                "rationale": "Short text.",
            }
        ]
        text = json.dumps(data)
        result = _parse_selection_rationale_response(text, {"NVDA"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_empty_response(self):
        """Empty string returns empty dict."""
        result = _parse_selection_rationale_response("", {"NVDA"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_non_json_response(self):
        """Non-JSON text returns empty dict."""
        result = _parse_selection_rationale_response(
            "Sorry, I cannot help with that.", {"NVDA"},
        )
        assert len(result) == 0

    @pytest.mark.schema
    def test_json_with_markdown_fences(self):
        """JSON wrapped in markdown code fences is parsed correctly."""
        data = [
            {
                "symbol": "AAPL",
                "rationale": "Strong RS momentum with sector leadership in technology and consistent EPS growth.",
            }
        ]
        text = f"```json\n{json.dumps(data)}\n```"
        result = _parse_selection_rationale_response(text, {"AAPL"})
        assert "AAPL" in result

    @pytest.mark.schema
    def test_multiple_positions(self):
        """Multiple valid positions parsed."""
        data = [
            {
                "symbol": "NVDA",
                "rationale": "High-conviction T1 momentum leader in chips sector with RS breakout.",
            },
            {
                "symbol": "AAPL",
                "rationale": "Quality growth position with broad institutional support and earnings catalyst.",
            },
        ]
        text = json.dumps(data)
        result = _parse_selection_rationale_response(text, {"NVDA", "AAPL"})
        assert len(result) == 2
        assert "NVDA" in result
        assert "AAPL" in result

    @pytest.mark.schema
    def test_missing_rationale_field(self):
        """Entry without rationale field is skipped."""
        data = [
            {
                "symbol": "NVDA",
            }
        ]
        text = json.dumps(data)
        result = _parse_selection_rationale_response(text, {"NVDA"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_lowercase_symbol_uppercased(self):
        """Lowercase symbol in response is uppercased for matching."""
        data = [
            {
                "symbol": "nvda",
                "rationale": "Leading semiconductor stock with strong relative strength and momentum.",
            }
        ]
        text = json.dumps(data)
        result = _parse_selection_rationale_response(text, {"NVDA"})
        assert "NVDA" in result


# ---------------------------------------------------------------------------
# _parse_alternative_reasoning_response Tests
# ---------------------------------------------------------------------------

class TestParseAlternativeReasoningResponse:

    @pytest.mark.schema
    def test_valid_response(self):
        """Valid JSON with name, key_difference, tradeoff parsed."""
        data = [
            {
                "name": "Higher Probability (20% target)",
                "key_difference": "Lower target allows more diversification and reduces concentration risk significantly.",
                "tradeoff": "Gain higher probability of success and lower drawdown. Give up 10% potential return.",
            }
        ]
        text = json.dumps(data)
        result = _parse_alternative_reasoning_response(
            text, {"Higher Probability (20% target)"},
        )
        assert "Higher Probability (20% target)" in result
        assert "key_difference" in result["Higher Probability (20% target)"]
        assert "tradeoff" in result["Higher Probability (20% target)"]

    @pytest.mark.schema
    def test_unknown_name_filtered(self):
        """Alternative name not in valid_names is filtered."""
        data = [
            {
                "name": "Unknown Portfolio",
                "key_difference": "This should be filtered because name is not valid.",
                "tradeoff": "No tradeoff since this alternative does not exist.",
            }
        ]
        text = json.dumps(data)
        result = _parse_alternative_reasoning_response(text, {"Conservative"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_short_key_difference_filtered(self):
        """key_difference < 10 chars causes entry to be skipped."""
        data = [
            {
                "name": "Conservative",
                "key_difference": "Less risk",
                "tradeoff": "Gain safety and stability. Give up aggressive return potential.",
            }
        ]
        text = json.dumps(data)
        result = _parse_alternative_reasoning_response(text, {"Conservative"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_short_tradeoff_filtered(self):
        """tradeoff < 10 chars causes entry to be skipped."""
        data = [
            {
                "name": "Conservative",
                "key_difference": "Lower target with more defensive positioning and diversification.",
                "tradeoff": "Less risk",
            }
        ]
        text = json.dumps(data)
        result = _parse_alternative_reasoning_response(text, {"Conservative"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_empty_response(self):
        """Empty string returns empty dict."""
        result = _parse_alternative_reasoning_response("", {"Conservative"})
        assert len(result) == 0

    @pytest.mark.schema
    def test_non_json_response(self):
        """Non-JSON returns empty dict."""
        result = _parse_alternative_reasoning_response(
            "I don't know", {"Conservative"},
        )
        assert len(result) == 0

    @pytest.mark.schema
    def test_multiple_alternatives(self):
        """Two alternatives parsed correctly."""
        data = [
            {
                "name": "Conservative",
                "key_difference": "Lower target with heavier T3 defensive allocation for stability.",
                "tradeoff": "Gain 20% higher probability and lower drawdown. Give up 10% return.",
            },
            {
                "name": "Aggressive",
                "key_difference": "Higher target requiring extreme T1 concentration in momentum names.",
                "tradeoff": "Gain 10% more return potential. Give up probability and increase drawdown risk.",
            },
        ]
        text = json.dumps(data)
        result = _parse_alternative_reasoning_response(
            text, {"Conservative", "Aggressive"},
        )
        assert len(result) == 2


# ---------------------------------------------------------------------------
# enrich_selection_rationale_llm Tests
# ---------------------------------------------------------------------------

class TestEnrichSelectionRationaleLlm:

    @pytest.mark.schema
    def test_empty_positions_returns_empty(self):
        """Empty positions list returns empty dict."""
        result = enrich_selection_rationale_llm([])
        assert result == {}

    @pytest.mark.schema
    def test_graceful_without_sdk(self):
        """Without anthropic SDK or API key, returns empty dict (no crash)."""
        positions = [
            {
                "ticker": "NVDA",
                "company_name": "NVIDIA Corp",
                "tier": 1,
                "sector": "Chips/Semiconductors",
                "composite_score": 97,
                "rs_rating": 95,
                "eps_rating": 90,
                "conviction_level": "HIGH",
                "allocation_pct": 8.0,
                "sector_momentum": "leading",
                "multi_source_count": 3,
            }
        ]
        result = enrich_selection_rationale_llm(positions)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# enrich_construction_narrative_llm Tests
# ---------------------------------------------------------------------------

class TestEnrichConstructionNarrativeLlm:

    @pytest.mark.schema
    def test_empty_context_returns_none(self):
        """Empty context returns None."""
        result = enrich_construction_narrative_llm({})
        assert result is None

    @pytest.mark.schema
    def test_graceful_without_sdk(self):
        """Without SDK or API key, returns None (no crash)."""
        context = {
            "target_return_pct": 30.0,
            "regime": "bull",
            "t1_pct": 55,
            "t2_pct": 30,
            "t3_pct": 15,
            "n_positions": 12,
            "prob_achieve": "52%",
            "achievability": "REALISTIC",
            "sector_momentum": "leading",
            "top_sectors": ["Chips/Semiconductors", "Software", "Medical"],
            "top_positions": ["NVDA", "CLS", "GMED"],
        }
        result = enrich_construction_narrative_llm(context)
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# enrich_alternative_reasoning_llm Tests
# ---------------------------------------------------------------------------

class TestEnrichAlternativeReasoningLlm:

    @pytest.mark.schema
    def test_empty_alternatives_returns_empty(self):
        """Empty alternatives list returns empty dict."""
        result = enrich_alternative_reasoning_llm([], {})
        assert result == {}

    @pytest.mark.schema
    def test_graceful_without_sdk(self):
        """Without SDK or API key, returns empty dict (no crash)."""
        alternatives = [
            {
                "name": "Conservative (20% target)",
                "target_return_pct": 20.0,
                "prob_achieve_target": 0.65,
                "t1_pct": 40.0,
                "t2_pct": 40.0,
                "t3_pct": 20.0,
                "max_drawdown_pct": 15.0,
            }
        ]
        context = {
            "primary_target": 30.0,
            "primary_prob": "52%",
            "regime": "bull",
        }
        result = enrich_alternative_reasoning_llm(alternatives, context)
        assert isinstance(result, dict)
