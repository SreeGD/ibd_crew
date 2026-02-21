"""Tests for the token usage tracker module."""

import pytest

from ibd_agents.tools.token_tracker import (
    HAIKU_INPUT_COST_PER_MTOK,
    HAIKU_OUTPUT_COST_PER_MTOK,
    LLM_FUNCTION_REGISTRY,
    TokenTracker,
    _compute_cost,
)


class _MockUsage:
    """Minimal mock for anthropic response.usage."""

    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _MockResponse:
    """Minimal mock for an anthropic Messages response."""

    def __init__(self, input_tokens: int, output_tokens: int):
        self.usage = _MockUsage(input_tokens, output_tokens)


@pytest.fixture
def fresh_tracker():
    """Return a fresh TokenTracker for each test."""
    return TokenTracker()


# ------------------------------------------------------------------
# Core tracking
# ------------------------------------------------------------------


class TestTrackRecordsUsage:
    def test_single_track(self, fresh_tracker: TokenTracker):
        resp = _MockResponse(1000, 200)
        fresh_tracker.track("enrich_valuation_llm", resp)
        assert len(fresh_tracker._records) == 1
        rec = fresh_tracker._records[0]
        assert rec["function"] == "enrich_valuation_llm"
        assert rec["input_tokens"] == 1000
        assert rec["output_tokens"] == 200
        assert rec["agent_id"] == "Agent 02"
        assert rec["agent_name"] == "Analyst"
        assert rec["model"] == "claude-haiku"
        assert "timestamp" in rec

    def test_multiple_tracks(self, fresh_tracker: TokenTracker):
        fresh_tracker.track("enrich_valuation_llm", _MockResponse(500, 100))
        fresh_tracker.track("enrich_catalyst_llm", _MockResponse(300, 80))
        assert len(fresh_tracker._records) == 2


# ------------------------------------------------------------------
# Summaries
# ------------------------------------------------------------------


class TestGetSummaryTotals:
    def test_empty(self, fresh_tracker: TokenTracker):
        s = fresh_tracker.get_summary()
        assert s["total_input_tokens"] == 0
        assert s["total_output_tokens"] == 0
        assert s["total_tokens"] == 0
        assert s["estimated_cost_usd"] == 0
        assert s["num_calls"] == 0

    def test_totals(self, fresh_tracker: TokenTracker):
        fresh_tracker.track("enrich_valuation_llm", _MockResponse(1000, 200))
        fresh_tracker.track("enrich_catalyst_llm", _MockResponse(500, 100))
        s = fresh_tracker.get_summary()
        assert s["total_input_tokens"] == 1500
        assert s["total_output_tokens"] == 300
        assert s["total_tokens"] == 1800
        assert s["num_calls"] == 2


class TestGetByAgentGrouping:
    def test_groups_by_agent(self, fresh_tracker: TokenTracker):
        # Two calls from Agent 02 (Analyst)
        fresh_tracker.track("enrich_valuation_llm", _MockResponse(1000, 200))
        fresh_tracker.track("enrich_catalyst_llm", _MockResponse(500, 100))
        # One call from Agent 01 (Research)
        fresh_tracker.track("classify_sectors_llm", _MockResponse(300, 50))

        by_agent = fresh_tracker.get_by_agent()
        assert len(by_agent) == 2

        # Sorted by agent_id, so Agent 01 first
        assert by_agent[0]["agent_id"] == "Agent 01"
        assert by_agent[0]["input_tokens"] == 300
        assert by_agent[0]["output_tokens"] == 50
        assert by_agent[0]["total_tokens"] == 350
        assert by_agent[0]["calls"] == 1

        assert by_agent[1]["agent_id"] == "Agent 02"
        assert by_agent[1]["input_tokens"] == 1500
        assert by_agent[1]["output_tokens"] == 300
        assert by_agent[1]["total_tokens"] == 1800
        assert by_agent[1]["calls"] == 2


class TestGetByFunctionBreakdown:
    def test_per_function(self, fresh_tracker: TokenTracker):
        fresh_tracker.track("enrich_valuation_llm", _MockResponse(1000, 200))
        fresh_tracker.track("enrich_valuation_llm", _MockResponse(800, 150))
        fresh_tracker.track("enrich_catalyst_llm", _MockResponse(500, 100))

        by_func = fresh_tracker.get_by_function()
        assert len(by_func) == 2

        val_func = [f for f in by_func if f["function"] == "enrich_valuation_llm"][0]
        assert val_func["input_tokens"] == 1800
        assert val_func["output_tokens"] == 350
        assert val_func["calls"] == 2

        cat_func = [f for f in by_func if f["function"] == "enrich_catalyst_llm"][0]
        assert cat_func["input_tokens"] == 500
        assert cat_func["calls"] == 1


# ------------------------------------------------------------------
# Cost calculation
# ------------------------------------------------------------------


class TestCostCalculation:
    def test_haiku_pricing(self):
        # 1M input tokens + 1M output tokens
        cost = _compute_cost(1_000_000, 1_000_000, "claude-haiku")
        expected = HAIKU_INPUT_COST_PER_MTOK + HAIKU_OUTPUT_COST_PER_MTOK
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_small_amounts(self):
        # 1000 input, 200 output at Haiku rates
        cost = _compute_cost(1000, 200, "claude-haiku")
        expected = (1000 * 0.80 + 200 * 4.00) / 1_000_000
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_summary_cost_matches(self, fresh_tracker: TokenTracker):
        fresh_tracker.track("enrich_valuation_llm", _MockResponse(10000, 2000))
        s = fresh_tracker.get_summary()
        expected = _compute_cost(10000, 2000)
        assert s["estimated_cost_usd"] == pytest.approx(expected, abs=0.0001)


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------


class TestResetClears:
    def test_reset(self, fresh_tracker: TokenTracker):
        fresh_tracker.track("enrich_valuation_llm", _MockResponse(100, 50))
        assert fresh_tracker.has_records is True
        fresh_tracker.reset()
        assert fresh_tracker.has_records is False
        assert len(fresh_tracker._records) == 0


# ------------------------------------------------------------------
# Defensive behavior
# ------------------------------------------------------------------


class TestDefensiveNoUsage:
    def test_no_usage_attribute(self, fresh_tracker: TokenTracker):
        """Response without .usage should not crash or record."""

        class _BadResponse:
            pass

        fresh_tracker.track("enrich_valuation_llm", _BadResponse())
        assert len(fresh_tracker._records) == 0

    def test_none_response(self, fresh_tracker: TokenTracker):
        fresh_tracker.track("enrich_valuation_llm", None)
        assert len(fresh_tracker._records) == 0

    def test_usage_with_none_tokens(self, fresh_tracker: TokenTracker):
        class _BadUsage:
            input_tokens = None
            output_tokens = None

        class _BadResp:
            usage = _BadUsage()

        fresh_tracker.track("enrich_valuation_llm", _BadResp())
        assert len(fresh_tracker._records) == 0


class TestUnknownFunctionName:
    def test_unregistered_function(self, fresh_tracker: TokenTracker):
        fresh_tracker.track("some_unknown_function", _MockResponse(100, 50))
        assert len(fresh_tracker._records) == 1
        assert fresh_tracker._records[0]["agent_id"] == "Unknown"
        assert fresh_tracker._records[0]["agent_name"] == "Unknown"


# ------------------------------------------------------------------
# Registry completeness
# ------------------------------------------------------------------


class TestRegistryCompleteness:
    def test_all_15_functions_registered(self):
        assert len(LLM_FUNCTION_REGISTRY) == 15

    def test_all_entries_have_agent_id_and_name(self):
        for func, (agent_id, agent_name) in LLM_FUNCTION_REGISTRY.items():
            assert agent_id.startswith("Agent"), f"{func} has bad agent_id"
            assert len(agent_name) > 0, f"{func} has empty agent_name"
