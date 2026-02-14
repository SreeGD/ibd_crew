"""
Rotation Narrative Intelligence — Unit Tests
Tests for generate_rotation_narrative_llm() and prompt template.
"""

from __future__ import annotations

import pytest

from ibd_agents.tools.rotation_narrative import generate_rotation_narrative_llm


# ---------------------------------------------------------------------------
# generate_rotation_narrative_llm Tests
# ---------------------------------------------------------------------------

class TestGenerateRotationNarrative:

    @pytest.mark.schema
    def test_empty_context_returns_none(self):
        """Empty context dict returns None."""
        assert generate_rotation_narrative_llm({}) is None

    @pytest.mark.schema
    def test_none_context_returns_none(self):
        """None context returns None."""
        assert generate_rotation_narrative_llm(None) is None

    @pytest.mark.schema
    def test_none_verdict_returns_none(self):
        """Verdict='none' skips LLM call, returns None."""
        ctx = {
            "verdict": "none",
            "rotation_type": "none",
            "stage": None,
            "source_clusters": [],
            "dest_clusters": [],
            "regime": "bull",
            "signals_active": 0,
            "velocity": None,
        }
        result = generate_rotation_narrative_llm(ctx)
        assert result is None

    @pytest.mark.schema
    def test_active_verdict_graceful_without_sdk(self):
        """Active verdict without anthropic SDK returns None (graceful)."""
        ctx = {
            "verdict": "active",
            "rotation_type": "cyclical",
            "stage": "mid",
            "source_clusters": ["growth"],
            "dest_clusters": ["commodity"],
            "regime": "bull",
            "signals_active": 3,
            "velocity": "Moderate",
        }
        # Without ANTHROPIC_API_KEY or SDK, should return None gracefully
        result = generate_rotation_narrative_llm(ctx)
        # Result is None (no SDK/key in test env) — no error raised
        assert result is None or isinstance(result, str)

    @pytest.mark.schema
    def test_partial_context_handled(self):
        """Missing keys in context are handled with defaults."""
        ctx = {"verdict": "emerging"}
        # Should not raise — function uses .get() with defaults
        result = generate_rotation_narrative_llm(ctx)
        assert result is None or isinstance(result, str)
