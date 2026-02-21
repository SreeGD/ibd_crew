"""Centralized token usage tracker for all LLM calls in the pipeline.

Captures input/output token counts from anthropic SDK responses and provides
summaries by agent, by function, and overall totals with cost estimates.
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Pricing constants (USD per million tokens) — Claude Haiku 3.5
# ---------------------------------------------------------------------------
HAIKU_INPUT_COST_PER_MTOK = 0.80
HAIKU_OUTPUT_COST_PER_MTOK = 4.00

SONNET_INPUT_COST_PER_MTOK = 3.00
SONNET_OUTPUT_COST_PER_MTOK = 15.00

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku": (HAIKU_INPUT_COST_PER_MTOK, HAIKU_OUTPUT_COST_PER_MTOK),
    "claude-sonnet": (SONNET_INPUT_COST_PER_MTOK, SONNET_OUTPUT_COST_PER_MTOK),
}

# ---------------------------------------------------------------------------
# Registry: function name → (agent_id, agent_name)
# ---------------------------------------------------------------------------
LLM_FUNCTION_REGISTRY: dict[str, tuple[str, str]] = {
    "classify_sectors_llm": ("Agent 01", "Research"),
    "classify_cap_sizes_llm": ("Agent 01", "Research"),
    "enrich_valuation_llm": ("Agent 02", "Analyst"),
    "enrich_catalyst_llm": ("Agent 02", "Analyst"),
    "generate_rotation_narrative_llm": ("Agent 03", "Rotation"),
    "enrich_sizing_llm": ("Agent 05", "Portfolio"),
    "enrich_stop_loss_llm": ("Agent 06", "Risk"),
    "enrich_rationale_llm": ("Agent 08", "Reconciler"),
    "_generate_llm_synthesis": ("Agent 10", "Synthesizer"),
    "enrich_selection_rationale_llm": ("Agent 11", "Value Investor"),
    "enrich_construction_narrative_llm": ("Agent 11", "Value Investor"),
    "enrich_alternative_reasoning_llm": ("Agent 11", "Value Investor"),
    "score_patterns_llm": ("Agent 12", "Pattern Alpha"),
    "enrich_alert_narratives_llm": ("Agent 12", "Pattern Alpha"),
    "find_historical_pattern_llm": ("Agent 12", "Pattern Alpha"),
}


def _compute_cost(
    input_tokens: int, output_tokens: int, model: str = "claude-haiku"
) -> float:
    """Compute cost in USD for a given token count."""
    input_cost_per_mtok, output_cost_per_mtok = MODEL_PRICING.get(
        model, (HAIKU_INPUT_COST_PER_MTOK, HAIKU_OUTPUT_COST_PER_MTOK)
    )
    return (input_tokens * input_cost_per_mtok + output_tokens * output_cost_per_mtok) / 1_000_000


class TokenTracker:
    """Accumulates token usage records from LLM calls."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def track(
        self, function_name: str, response: Any, model: str = "claude-haiku"
    ) -> None:
        """Record token usage from an anthropic response.

        Defensively extracts response.usage.input_tokens / output_tokens.
        Never raises — logs nothing on failure so it cannot crash the pipeline.
        """
        try:
            usage = response.usage
            input_tokens = int(usage.input_tokens)
            output_tokens = int(usage.output_tokens)
        except Exception:
            return

        agent_id, agent_name = LLM_FUNCTION_REGISTRY.get(
            function_name, ("Unknown", "Unknown")
        )

        self._records.append(
            {
                "function": function_name,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": model,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Return overall totals and estimated cost."""
        total_input = sum(r["input_tokens"] for r in self._records)
        total_output = sum(r["output_tokens"] for r in self._records)
        total_tokens = total_input + total_output
        cost = _compute_cost(total_input, total_output)
        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(cost, 4),
            "num_calls": len(self._records),
        }

    def get_by_agent(self) -> list[dict[str, Any]]:
        """Return per-agent grouped totals, sorted by agent_id."""
        groups: dict[str, dict[str, Any]] = {}
        for r in self._records:
            key = r["agent_id"]
            if key not in groups:
                groups[key] = {
                    "agent_id": r["agent_id"],
                    "agent_name": r["agent_name"],
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "calls": 0,
                }
            groups[key]["input_tokens"] += r["input_tokens"]
            groups[key]["output_tokens"] += r["output_tokens"]
            groups[key]["calls"] += 1

        result = []
        for g in sorted(groups.values(), key=lambda x: x["agent_id"]):
            g["total_tokens"] = g["input_tokens"] + g["output_tokens"]
            g["cost_usd"] = round(
                _compute_cost(g["input_tokens"], g["output_tokens"]), 4
            )
            result.append(g)
        return result

    def get_by_function(self) -> list[dict[str, Any]]:
        """Return per-function breakdown, sorted by agent then function."""
        groups: dict[str, dict[str, Any]] = {}
        for r in self._records:
            key = r["function"]
            if key not in groups:
                groups[key] = {
                    "function": r["function"],
                    "agent_id": r["agent_id"],
                    "agent_name": r["agent_name"],
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "calls": 0,
                }
            groups[key]["input_tokens"] += r["input_tokens"]
            groups[key]["output_tokens"] += r["output_tokens"]
            groups[key]["calls"] += 1

        result = []
        for g in sorted(
            groups.values(), key=lambda x: (x["agent_id"], x["function"])
        ):
            g["total_tokens"] = g["input_tokens"] + g["output_tokens"]
            g["cost_usd"] = round(
                _compute_cost(g["input_tokens"], g["output_tokens"]), 4
            )
            result.append(g)
        return result

    @property
    def has_records(self) -> bool:
        return len(self._records) > 0

    def reset(self) -> None:
        """Clear all records (useful for testing)."""
        self._records.clear()


# ---------------------------------------------------------------------------
# Module-level singleton and convenience function
# ---------------------------------------------------------------------------
tracker = TokenTracker()


def track(function_name: str, response: Any, model: str = "claude-haiku") -> None:
    """Convenience wrapper around the global tracker."""
    tracker.track(function_name, response, model)
