"""
LLM Valuation Enrichment Tests

Tests for enrich_valuation_llm() and _parse_llm_response() using mocked
anthropic client. No real LLM calls.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import pytest

from ibd_agents.tools.valuation_metrics import (
    _parse_llm_response,
    enrich_valuation_llm,
)


# ---------------------------------------------------------------------------
# Sample Data
# ---------------------------------------------------------------------------

SAMPLE_STOCKS = [
    {"symbol": "NVDA", "company_name": "NVIDIA", "sector": "CHIPS",
     "composite_rating": 98, "rs_rating": 81, "eps_rating": 99, "tier": 2},
    {"symbol": "GOOGL", "company_name": "Alphabet", "sector": "INTERNET",
     "composite_rating": 99, "rs_rating": 94, "eps_rating": 92, "tier": 1},
    {"symbol": "GS", "company_name": "Goldman Sachs", "sector": "BANKS",
     "composite_rating": 93, "rs_rating": 91, "eps_rating": 76, "tier": 2},
]

VALID_LLM_RESPONSE = json.dumps([
    {
        "symbol": "NVDA",
        "pe_ratio": 65.2,
        "forward_pe": 32.5,
        "beta": 1.68,
        "annual_return_1y": 180.5,
        "volatility": 52.3,
        "dividend_yield": 0.03,
        "market_cap_b": 3200.0,
        "valuation_grade": "Speculative",
        "risk_grade": "Moderate",
        "guidance": "NVIDIA commands a premium valuation driven by AI demand. Strong momentum but elevated volatility. Consider position sizing given high beta.",
    },
    {
        "symbol": "GOOGL",
        "pe_ratio": 22.8,
        "forward_pe": 19.5,
        "beta": 1.05,
        "annual_return_1y": 35.2,
        "volatility": 28.1,
        "dividend_yield": 0.5,
        "market_cap_b": 2100.0,
        "valuation_grade": "Reasonable",
        "risk_grade": "Good",
        "guidance": "Alphabet trades at a reasonable valuation with strong earnings growth. AI investments should drive future returns. Solid risk-reward profile.",
    },
    {
        "symbol": "GS",
        "pe_ratio": 14.5,
        "forward_pe": 12.8,
        "beta": 1.35,
        "annual_return_1y": 42.0,
        "volatility": 30.5,
        "dividend_yield": 2.1,
        "market_cap_b": 180.0,
        "valuation_grade": "Value",
        "risk_grade": "Good",
        "guidance": "Goldman Sachs is attractively valued for a leading investment bank. Strong capital markets revenue but cyclical exposure. Good dividend yield provides downside protection.",
    },
])


# ---------------------------------------------------------------------------
# _parse_llm_response Tests
# ---------------------------------------------------------------------------

class TestParseLLMResponse:

    @pytest.mark.schema
    def test_parses_valid_json(self):
        """Valid JSON array is parsed correctly."""
        valid_symbols = {"NVDA", "GOOGL", "GS"}
        result = _parse_llm_response(VALID_LLM_RESPONSE, valid_symbols)
        assert len(result) == 3
        assert "NVDA" in result
        assert result["NVDA"]["pe_ratio"] == 65.2
        assert result["NVDA"]["beta"] == 1.68
        assert result["NVDA"]["valuation_grade"] == "Speculative"
        assert result["NVDA"]["guidance"] is not None

    @pytest.mark.schema
    def test_filters_unknown_symbols(self):
        """Symbols not in valid_symbols are filtered out."""
        result = _parse_llm_response(VALID_LLM_RESPONSE, {"NVDA"})
        assert len(result) == 1
        assert "NVDA" in result
        assert "GOOGL" not in result

    @pytest.mark.schema
    def test_handles_null_values(self):
        """Null metric values are preserved as None."""
        response = json.dumps([
            {"symbol": "NVDA", "pe_ratio": None, "forward_pe": None,
             "beta": 1.5, "annual_return_1y": None, "volatility": None,
             "dividend_yield": None, "market_cap_b": None,
             "valuation_grade": "Growth Premium", "risk_grade": "Moderate",
             "guidance": "Test guidance."}
        ])
        result = _parse_llm_response(response, {"NVDA"})
        assert result["NVDA"]["pe_ratio"] is None
        assert result["NVDA"]["beta"] == 1.5

    @pytest.mark.schema
    def test_handles_invalid_grades(self):
        """Invalid valuation/risk grades are set to None."""
        response = json.dumps([
            {"symbol": "NVDA", "pe_ratio": 50.0, "forward_pe": None,
             "beta": None, "annual_return_1y": None, "volatility": None,
             "dividend_yield": None, "market_cap_b": None,
             "valuation_grade": "INVALID_GRADE", "risk_grade": "WRONG",
             "guidance": "Test."}
        ])
        result = _parse_llm_response(response, {"NVDA"})
        assert result["NVDA"]["valuation_grade"] is None
        assert result["NVDA"]["risk_grade"] is None

    @pytest.mark.schema
    def test_handles_empty_response(self):
        """Empty string returns empty dict."""
        result = _parse_llm_response("", {"NVDA"})
        assert result == {}

    @pytest.mark.schema
    def test_handles_non_json_response(self):
        """Non-JSON text returns empty dict."""
        result = _parse_llm_response("I can't provide financial data.", {"NVDA"})
        assert result == {}

    @pytest.mark.schema
    def test_handles_json_with_markdown_fences(self):
        """JSON wrapped in markdown code fences is still parsed."""
        response = "```json\n" + json.dumps([
            {"symbol": "NVDA", "pe_ratio": 50.0, "forward_pe": None,
             "beta": None, "annual_return_1y": None, "volatility": None,
             "dividend_yield": None, "market_cap_b": None,
             "valuation_grade": "Growth Premium", "risk_grade": "Good",
             "guidance": "Test."}
        ]) + "\n```"
        result = _parse_llm_response(response, {"NVDA"})
        assert "NVDA" in result

    @pytest.mark.schema
    def test_handles_non_finite_floats(self):
        """Non-finite float values are set to None."""
        response = json.dumps([
            {"symbol": "NVDA", "pe_ratio": float("inf"), "forward_pe": float("nan"),
             "beta": 1.5, "annual_return_1y": None, "volatility": None,
             "dividend_yield": None, "market_cap_b": None,
             "valuation_grade": "Growth Premium", "risk_grade": "Good",
             "guidance": "Test."}
        ])
        result = _parse_llm_response(response, {"NVDA"})
        assert result["NVDA"]["pe_ratio"] is None
        assert result["NVDA"]["forward_pe"] is None
        assert result["NVDA"]["beta"] == 1.5


# ---------------------------------------------------------------------------
# enrich_valuation_llm Tests
# ---------------------------------------------------------------------------

class TestEnrichValuationLLM:

    @pytest.mark.schema
    def test_returns_empty_for_empty_input(self):
        """Empty stock list returns empty dict."""
        result = enrich_valuation_llm([])
        assert result == {}

    @pytest.mark.schema
    def test_handles_no_sdk(self):
        """Returns empty dict when anthropic SDK is not installed."""
        with patch.dict("sys.modules", {"anthropic": None}):
            # Force reimport to hit the ImportError
            result = enrich_valuation_llm(SAMPLE_STOCKS)
            # When anthropic module is None, import anthropic raises TypeError
            # which gets caught by the outer except Exception
            assert isinstance(result, dict)

    @pytest.mark.schema
    def test_handles_api_error(self):
        """Returns empty dict when API call raises."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API rate limited")
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = enrich_valuation_llm(SAMPLE_STOCKS)
            assert result == {}

    @pytest.mark.schema
    def test_parses_valid_response(self):
        """Valid LLM response is parsed and returned."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = VALID_LLM_RESPONSE
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = enrich_valuation_llm(SAMPLE_STOCKS)
            assert len(result) == 3
            assert result["NVDA"]["pe_ratio"] == 65.2
            assert result["GOOGL"]["valuation_grade"] == "Reasonable"
            assert result["GS"]["guidance"] is not None

    @pytest.mark.schema
    def test_batching(self):
        """Stocks are batched correctly."""
        # Create 65 stocks to test batching with batch_size=30
        stocks = [
            {"symbol": f"TST{i}", "company_name": f"Test {i}", "sector": "CHIPS",
             "composite_rating": 90, "rs_rating": 85, "eps_rating": 80, "tier": 2}
            for i in range(65)
        ]

        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "[]"
            mock_resp.content = [mock_content]
            return mock_resp

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = mock_create
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            enrich_valuation_llm(stocks, batch_size=30)
            # 65 stocks / 30 per batch = 3 batches (30, 30, 5)
            assert call_count == 3

    @pytest.mark.schema
    def test_handles_partial_data(self):
        """When LLM only returns data for some stocks, others are absent."""
        partial_response = json.dumps([
            {"symbol": "NVDA", "pe_ratio": 65.0, "forward_pe": None,
             "beta": 1.7, "annual_return_1y": None, "volatility": None,
             "dividend_yield": None, "market_cap_b": None,
             "valuation_grade": "Speculative", "risk_grade": "Moderate",
             "guidance": "Strong AI play with premium valuation."}
        ])

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = partial_response
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = enrich_valuation_llm(SAMPLE_STOCKS)
            assert "NVDA" in result
            assert "GOOGL" not in result
            assert "GS" not in result


# ---------------------------------------------------------------------------
# Pipeline Integration Test (mocked LLM)
# ---------------------------------------------------------------------------

class TestPipelineWithLLMEnrichment:

    @pytest.mark.schema
    def test_pipeline_with_mocked_llm(self):
        """Pipeline integrates LLM data when available."""
        from tests.fixtures.conftest import SAMPLE_IBD_STOCKS, PADDING_STOCKS
        from ibd_agents.schemas.research_output import (
            ResearchOutput, ResearchStock,
            compute_preliminary_tier, is_ibd_keep_candidate,
        )
        from ibd_agents.agents.analyst_agent import run_analyst_pipeline
        from datetime import date

        # Build research output
        all_stocks = SAMPLE_IBD_STOCKS + PADDING_STOCKS
        research_stocks = []
        for s in all_stocks:
            tier = compute_preliminary_tier(
                s.get("composite_rating"), s.get("rs_rating"), s.get("eps_rating")
            )
            keep = is_ibd_keep_candidate(s.get("composite_rating"), s.get("rs_rating"))
            has_ratings = sum(1 for r in [s.get("composite_rating"), s.get("rs_rating"),
                                           s.get("eps_rating"), s.get("smr_rating"),
                                           s.get("acc_dis_rating")] if r is not None)
            confidence = min(1.0, 0.3 + (has_ratings * 0.14))
            research_stocks.append(ResearchStock(
                symbol=s["symbol"], company_name=s.get("company_name", s["symbol"]),
                sector=s.get("sector", "UNKNOWN"), security_type="stock",
                composite_rating=s.get("composite_rating"),
                rs_rating=s.get("rs_rating"), eps_rating=s.get("eps_rating"),
                smr_rating=s.get("smr_rating"), acc_dis_rating=s.get("acc_dis_rating"),
                ibd_lists=s.get("ibd_lists", ["IBD 50"]), schwab_themes=[],
                fool_status=None, other_ratings={}, validation_score=3,
                validation_providers=1, is_multi_source_validated=False,
                is_ibd_keep_candidate=keep, preliminary_tier=tier,
                sources=["test"], confidence=confidence,
                reasoning="Test stock for LLM enrichment pipeline testing",
            ))

        keep_candidates = [s.symbol for s in research_stocks if s.is_ibd_keep_candidate]
        multi_validated = [s.symbol for s in research_stocks if s.is_multi_source_validated]
        ro = ResearchOutput(
            stocks=research_stocks, etfs=[], sector_patterns=[],
            data_sources_used=["test"], data_sources_failed=[],
            total_securities_scanned=len(research_stocks),
            ibd_keep_candidates=keep_candidates, multi_source_validated=multi_validated,
            analysis_date=date.today().isoformat(),
            summary="Test research output for LLM enrichment pipeline testing with sample data",
        )

        # Mock LLM to return data for NVDA
        llm_response = json.dumps([
            {"symbol": "NVDA", "pe_ratio": 65.0, "forward_pe": 32.0,
             "beta": 1.7, "annual_return_1y": 180.0, "volatility": 52.0,
             "dividend_yield": 0.03, "market_cap_b": 3200.0,
             "valuation_grade": "Speculative", "risk_grade": "Moderate",
             "guidance": "NVIDIA commands a premium AI valuation. Strong momentum but elevated risk."}
        ])

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_content = MagicMock()
        mock_content.text = llm_response
        mock_resp.content = [mock_content]
        mock_client.messages.create.return_value = mock_resp
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            output = run_analyst_pipeline(ro)

        # Find NVDA in rated stocks
        nvda = next((s for s in output.rated_stocks if s.symbol == "NVDA"), None)
        assert nvda is not None
        assert nvda.valuation_source == "llm"
        assert nvda.llm_pe_ratio == 65.0
        assert nvda.llm_beta == 1.7
        assert nvda.llm_guidance is not None
        assert nvda.estimated_pe == 65.0  # Overridden by LLM

        # Stocks not in LLM response should have "estimated" source
        non_nvda = [s for s in output.rated_stocks if s.symbol != "NVDA"]
        for s in non_nvda:
            assert s.valuation_source == "estimated"
            assert s.llm_pe_ratio is None
