"""
Agent 10: Executive Summary Synthesizer — Schema Tests
Level 1: Pure Pydantic validation, no LLM, no file I/O.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ibd_agents.schemas.synthesis_output import (
    Contradiction,
    CrossAgentConnection,
    KeyNumbersDashboard,
    SynthesisOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_key_numbers(**overrides) -> dict:
    """Build valid KeyNumbersDashboard kwargs."""
    defaults = {
        "total_stocks_scanned": 150,
        "stocks_in_universe": 71,
        "etfs_in_universe": 10,
        "stocks_rated": 60,
        "etfs_rated": 8,
        "tier_1_count": 15,
        "tier_2_count": 25,
        "tier_3_count": 20,
        "rotation_verdict": "active",
        "rotation_confidence": 75,
        "market_regime": "bull",
        "portfolio_positions": 35,
        "stock_count": 25,
        "etf_count": 10,
        "cash_pct": 3.0,
        "risk_status": "CONDITIONAL",
        "sleep_well_score": 7,
        "risk_warnings_count": 2,
        "base_case_return_12m": 12.5,
        "bull_case_return_12m": 25.0,
        "bear_case_return_12m": -5.0,
        "turnover_pct": 45.0,
        "actions_count": 30,
    }
    defaults.update(overrides)
    return defaults


def _valid_connection(**overrides) -> dict:
    defaults = {
        "from_agent": "Rotation Detector (Agent 03)",
        "to_agent": "Sector Strategist (Agent 04)",
        "connection": "Rotation verdict drove strategy tier targets and sector allocations",
        "implication": "Bull regime adjustment increased T1 allocation by 5% at expense of T3",
    }
    defaults.update(overrides)
    return defaults


def _valid_contradiction(**overrides) -> dict:
    defaults = {
        "agent_a": "Sector Strategist (Agent 04)",
        "agent_b": "Risk Officer (Agent 06)",
        "finding_a": "Strategist allocates 35% to CHIPS sector for momentum capture",
        "finding_b": "Risk Officer flags CHIPS concentration exceeds 30% threshold",
        "resolution": "High conviction in CHIPS reflects strong rotation signals; monitor closely",
    }
    defaults.update(overrides)
    return defaults


def _valid_synthesis_output(**overrides) -> dict:
    defaults = {
        "investment_thesis": (
            "This portfolio is constructed from a universe of 150 securities scanned, "
            "with 60 stocks and 8 ETFs rated through the IBD Momentum Framework. "
            "The Rotation Detector identifies active rotation at 75% confidence "
            "within a bull market regime, driving offensive tier allocations. "
            "Return projections range from -5% (bear) to 25% (bull) over 12 months."
        ),
        "portfolio_narrative": (
            "The bull regime with active rotation drove the Strategist to increase "
            "T1 allocation to 39%. The Portfolio Manager constructed 35 positions "
            "across 10 sectors with 3% cash reserve."
        ),
        "risk_reward_assessment": (
            "Risk status: CONDITIONAL with Sleep Well score 7/10. "
            "Max drawdown with stops: 17.9%. Base-case Sharpe: 1.2. "
            "Expected 12-month alpha vs SPY: +3.5%."
        ),
        "market_context": (
            "Market regime: BULL with 3/5 rotation signals active. "
            "Capital flowing from defensive sectors toward growth/commodity clusters. "
            "Rotation velocity: Moderate. Type: cyclical."
        ),
        "key_numbers": _valid_key_numbers(),
        "cross_agent_connections": [
            _valid_connection(),
            _valid_connection(
                from_agent="Sector Strategist (Agent 04)",
                to_agent="Portfolio Manager (Agent 05)",
                connection="Strategy tier targets shaped portfolio construction and position sizing",
                implication="Portfolio actual allocation closely tracks strategy targets within 2%",
            ),
            _valid_connection(
                from_agent="Portfolio Manager (Agent 05)",
                to_agent="Risk Officer (Agent 06)",
                connection="Portfolio sector exposure drives risk concentration analysis",
                implication="Risk assessment reflects portfolio's actual sector distribution",
            ),
        ],
        "contradictions": [],
        "action_items": [
            "Execute portfolio transition: 10 sells in Week 1, 15 buys across Weeks 2-4.",
            "Address 2 risk warnings before full deployment of new positions.",
            "Monitor for base-case +12.5% 12-month return with max drawdown of 17.9%.",
        ],
        "synthesis_source": "template",
        "analysis_date": "2026-02-10",
        "summary": (
            "Executive synthesis (template): active rotation in bull regime. "
            "35 positions, base return 12.5%, risk status CONDITIONAL."
        ),
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# KeyNumbersDashboard Tests
# ---------------------------------------------------------------------------

class TestKeyNumbersDashboard:

    @pytest.mark.schema
    def test_valid_construction(self):
        kn = KeyNumbersDashboard(**_valid_key_numbers())
        assert kn.total_stocks_scanned == 150
        assert kn.stocks_rated == 60
        assert kn.rotation_verdict == "active"
        assert kn.sleep_well_score == 7

    @pytest.mark.schema
    def test_negative_count_fails(self):
        with pytest.raises(ValidationError):
            KeyNumbersDashboard(**_valid_key_numbers(total_stocks_scanned=-1))

    @pytest.mark.schema
    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            KeyNumbersDashboard(**_valid_key_numbers(rotation_confidence=101))

    @pytest.mark.schema
    def test_sleep_well_range(self):
        with pytest.raises(ValidationError):
            KeyNumbersDashboard(**_valid_key_numbers(sleep_well_score=0))
        with pytest.raises(ValidationError):
            KeyNumbersDashboard(**_valid_key_numbers(sleep_well_score=11))


# ---------------------------------------------------------------------------
# CrossAgentConnection Tests
# ---------------------------------------------------------------------------

class TestCrossAgentConnection:

    @pytest.mark.schema
    def test_valid_construction(self):
        conn = CrossAgentConnection(**_valid_connection())
        assert "Rotation" in conn.from_agent
        assert "Strategist" in conn.to_agent

    @pytest.mark.schema
    def test_short_connection_fails(self):
        with pytest.raises(ValidationError):
            CrossAgentConnection(**_valid_connection(connection="short"))

    @pytest.mark.schema
    def test_short_implication_fails(self):
        with pytest.raises(ValidationError):
            CrossAgentConnection(**_valid_connection(implication="short"))


# ---------------------------------------------------------------------------
# Contradiction Tests
# ---------------------------------------------------------------------------

class TestContradiction:

    @pytest.mark.schema
    def test_valid_construction(self):
        c = Contradiction(**_valid_contradiction())
        assert "Strategist" in c.agent_a
        assert "Risk" in c.agent_b

    @pytest.mark.schema
    def test_short_finding_fails(self):
        with pytest.raises(ValidationError):
            Contradiction(**_valid_contradiction(finding_a="short"))

    @pytest.mark.schema
    def test_short_resolution_fails(self):
        with pytest.raises(ValidationError):
            Contradiction(**_valid_contradiction(resolution="short"))


# ---------------------------------------------------------------------------
# SynthesisOutput Tests
# ---------------------------------------------------------------------------

class TestSynthesisOutput:

    @pytest.mark.schema
    def test_valid_construction(self):
        output = SynthesisOutput(**_valid_synthesis_output())
        assert isinstance(output, SynthesisOutput)
        assert output.synthesis_source == "template"

    @pytest.mark.schema
    def test_thesis_too_short_fails(self):
        with pytest.raises(ValidationError):
            SynthesisOutput(**_valid_synthesis_output(
                investment_thesis="Too short thesis."
            ))

    @pytest.mark.schema
    def test_narrative_too_short_fails(self):
        with pytest.raises(ValidationError):
            SynthesisOutput(**_valid_synthesis_output(
                portfolio_narrative="Short."
            ))

    @pytest.mark.schema
    def test_risk_reward_too_short_fails(self):
        with pytest.raises(ValidationError):
            SynthesisOutput(**_valid_synthesis_output(
                risk_reward_assessment="Short."
            ))

    @pytest.mark.schema
    def test_market_context_too_short_fails(self):
        with pytest.raises(ValidationError):
            SynthesisOutput(**_valid_synthesis_output(
                market_context="Short."
            ))

    @pytest.mark.schema
    def test_fewer_than_3_connections_fails(self):
        with pytest.raises(ValidationError):
            SynthesisOutput(**_valid_synthesis_output(
                cross_agent_connections=[_valid_connection(), _valid_connection()]
            ))

    @pytest.mark.schema
    def test_fewer_than_3_action_items_fails(self):
        with pytest.raises(ValidationError):
            SynthesisOutput(**_valid_synthesis_output(
                action_items=[
                    "Execute portfolio transition with 10 sells and 15 buys.",
                    "Address 2 risk warnings before full deployment.",
                ]
            ))

    @pytest.mark.schema
    def test_more_than_5_action_items_fails(self):
        with pytest.raises(ValidationError):
            SynthesisOutput(**_valid_synthesis_output(
                action_items=[
                    f"Action item number {i} — monitor portfolio and adjust as needed."
                    for i in range(6)
                ]
            ))

    @pytest.mark.schema
    def test_invalid_analysis_date_fails(self):
        with pytest.raises(ValidationError):
            SynthesisOutput(**_valid_synthesis_output(analysis_date="02-10-2026"))

    @pytest.mark.schema
    def test_invalid_synthesis_source_fails(self):
        with pytest.raises(ValidationError):
            SynthesisOutput(**_valid_synthesis_output(synthesis_source="gpt"))

    @pytest.mark.schema
    def test_llm_synthesis_source_valid(self):
        output = SynthesisOutput(**_valid_synthesis_output(synthesis_source="llm"))
        assert output.synthesis_source == "llm"

    @pytest.mark.schema
    def test_empty_contradictions_valid(self):
        output = SynthesisOutput(**_valid_synthesis_output(contradictions=[]))
        assert len(output.contradictions) == 0

    @pytest.mark.schema
    def test_with_contradictions_valid(self):
        output = SynthesisOutput(**_valid_synthesis_output(
            contradictions=[_valid_contradiction()]
        ))
        assert len(output.contradictions) == 1

    @pytest.mark.schema
    def test_short_action_item_fails(self):
        with pytest.raises(ValidationError):
            SynthesisOutput(**_valid_synthesis_output(
                action_items=["short", "Execute full transition.", "Monitor returns."]
            ))

    @pytest.mark.schema
    def test_summary_too_short_fails(self):
        with pytest.raises(ValidationError):
            SynthesisOutput(**_valid_synthesis_output(summary="Short."))


# ---------------------------------------------------------------------------
# Cap Size Count Tests
# ---------------------------------------------------------------------------

class TestCapSizeCounts:

    @pytest.mark.schema
    def test_dashboard_cap_counts_default_zero(self):
        """large/mid/small_cap_count default to 0."""
        kn = KeyNumbersDashboard(**_valid_key_numbers())
        assert kn.large_cap_count == 0
        assert kn.mid_cap_count == 0
        assert kn.small_cap_count == 0

    @pytest.mark.schema
    def test_dashboard_cap_counts_set(self):
        """Cap counts can be set explicitly."""
        kn = KeyNumbersDashboard(**_valid_key_numbers(
            large_cap_count=10, mid_cap_count=5, small_cap_count=2,
        ))
        assert kn.large_cap_count == 10
        assert kn.mid_cap_count == 5
        assert kn.small_cap_count == 2
