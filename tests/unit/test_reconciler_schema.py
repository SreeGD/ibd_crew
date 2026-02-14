"""
Agent 08: Portfolio Reconciler — Schema Tests
Level 1: Pure Pydantic validation, no LLM, no file I/O.
"""

from __future__ import annotations

import pytest

from ibd_agents.schemas.reconciliation_output import (
    ACTION_PRIORITIES,
    ACTION_TYPES,
    IMPLEMENTATION_PHASES,
    CurrentHolding,
    HoldingsSummary,
    ImplementationWeek,
    KeepVerification,
    MoneyFlow,
    PositionAction,
    ReconciliationOutput,
    TransformationMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_action(
    symbol: str = "NVDA",
    action_type: str = "BUY",
    week: int = 2,
) -> PositionAction:
    return PositionAction(
        symbol=symbol,
        action_type=action_type,
        current_pct=0.0,
        target_pct=3.0,
        dollar_change=45600.0,
        priority="MEDIUM",
        week=week,
        rationale=f"New T1 position at 3.0% allocation for portfolio",
    )


def _make_money_flow() -> MoneyFlow:
    return MoneyFlow(
        sell_proceeds=200000.0,
        trim_proceeds=50000.0,
        buy_cost=180000.0,
        add_cost=40000.0,
        net_cash_change=30000.0,
        cash_reserve_pct=2.0,
    )


def _make_reconciliation(**overrides) -> ReconciliationOutput:
    defaults = dict(
        current_holdings=HoldingsSummary(
            holdings=[
                CurrentHolding(symbol="NVDA", shares=100, market_value=30000, account="Schwab", sector="CHIPS"),
                CurrentHolding(symbol="AAPL", shares=50, market_value=20000, account="Schwab", sector="COMPUTER"),
            ],
            total_value=50000.0,
            account_count=1,
        ),
        actions=[_make_action("NVDA", "KEEP", 1), _make_action("AAPL", "SELL", 1)],
        money_flow=_make_money_flow(),
        implementation_plan=[
            ImplementationWeek(week=1, phase_name="LIQUIDATION", actions=[]),
            ImplementationWeek(week=2, phase_name="T1 MOMENTUM", actions=[]),
            ImplementationWeek(week=3, phase_name="T2 QUALITY GROWTH", actions=[]),
            ImplementationWeek(week=4, phase_name="T3 DEFENSIVE", actions=[]),
        ],
        keep_verification=KeepVerification(
            keeps_in_current=["MU", "CLS"],
            keeps_missing=["AMZN"],
            keeps_to_buy=["AMZN"],
        ),
        transformation_metrics=TransformationMetrics(
            before_sector_count=8,
            after_sector_count=12,
            before_position_count=40,
            after_position_count=52,
            turnover_pct=45.0,
        ),
        analysis_date="2025-01-15",
        summary="Reconciliation: 50 actions across 4-week plan. 15 positions to liquidate, 20 new positions. Turnover: 45%.",
    )
    defaults.update(overrides)
    return ReconciliationOutput(**defaults)


# ---------------------------------------------------------------------------
# Constant Tests
# ---------------------------------------------------------------------------

class TestConstants:

    @pytest.mark.schema
    def test_4_implementation_phases(self):
        assert len(IMPLEMENTATION_PHASES) == 4
        assert IMPLEMENTATION_PHASES[1] == "LIQUIDATION"

    @pytest.mark.schema
    def test_5_action_types(self):
        assert set(ACTION_TYPES) == {"KEEP", "SELL", "BUY", "ADD", "TRIM"}

    @pytest.mark.schema
    def test_3_priorities(self):
        assert set(ACTION_PRIORITIES) == {"HIGH", "MEDIUM", "LOW"}


# ---------------------------------------------------------------------------
# PositionAction Tests
# ---------------------------------------------------------------------------

class TestPositionAction:

    @pytest.mark.schema
    def test_valid_action(self):
        a = _make_action()
        assert a.action_type == "BUY"

    @pytest.mark.schema
    def test_invalid_action_type_fails(self):
        with pytest.raises(ValueError, match="action_type"):
            PositionAction(
                symbol="NVDA", action_type="HOLD",
                current_pct=0, target_pct=3,
                dollar_change=1000, priority="MEDIUM", week=1,
                rationale="Invalid action type for testing purposes",
            )

    @pytest.mark.schema
    def test_invalid_priority_fails(self):
        with pytest.raises(ValueError, match="priority"):
            PositionAction(
                symbol="NVDA", action_type="BUY",
                current_pct=0, target_pct=3,
                dollar_change=1000, priority="URGENT", week=1,
                rationale="Invalid priority level for testing purposes",
            )

    @pytest.mark.schema
    def test_symbol_uppercased(self):
        a = _make_action("nvda")
        assert a.symbol == "NVDA"


# ---------------------------------------------------------------------------
# MoneyFlow Tests
# ---------------------------------------------------------------------------

class TestMoneyFlow:

    @pytest.mark.schema
    def test_valid_money_flow(self):
        mf = _make_money_flow()
        assert mf.net_cash_change == 30000.0

    @pytest.mark.schema
    def test_imbalanced_flow_fails(self):
        with pytest.raises(ValueError, match="net_cash_change"):
            MoneyFlow(
                sell_proceeds=100, trim_proceeds=0,
                buy_cost=50, add_cost=0,
                net_cash_change=999,  # Wrong: should be 50
                cash_reserve_pct=2.0,
            )


# ---------------------------------------------------------------------------
# ReconciliationOutput Tests
# ---------------------------------------------------------------------------

class TestReconciliationOutput:

    @pytest.mark.schema
    def test_valid_output(self):
        out = _make_reconciliation()
        assert len(out.implementation_plan) == 4

    @pytest.mark.schema
    def test_4_weeks_required(self):
        with pytest.raises(ValueError):
            _make_reconciliation(
                implementation_plan=[
                    ImplementationWeek(week=1, phase_name="LIQUIDATION", actions=[]),
                    ImplementationWeek(week=2, phase_name="T1", actions=[]),
                    ImplementationWeek(week=3, phase_name="T2", actions=[]),
                ],
            )

    @pytest.mark.schema
    def test_week_1_must_be_liquidation(self):
        with pytest.raises(ValueError, match="LIQUIDATION"):
            _make_reconciliation(
                implementation_plan=[
                    ImplementationWeek(week=1, phase_name="BUYING", actions=[]),
                    ImplementationWeek(week=2, phase_name="T1", actions=[]),
                    ImplementationWeek(week=3, phase_name="T2", actions=[]),
                    ImplementationWeek(week=4, phase_name="T3", actions=[]),
                ],
            )

    @pytest.mark.schema
    def test_duplicate_weeks_fails(self):
        with pytest.raises(ValueError):
            _make_reconciliation(
                implementation_plan=[
                    ImplementationWeek(week=1, phase_name="LIQUIDATION", actions=[]),
                    ImplementationWeek(week=2, phase_name="T1", actions=[]),
                    ImplementationWeek(week=2, phase_name="T2", actions=[]),
                    ImplementationWeek(week=4, phase_name="T3", actions=[]),
                ],
            )

    @pytest.mark.schema
    def test_summary_min_length(self):
        out = _make_reconciliation()
        assert len(out.summary) >= 50


# ---------------------------------------------------------------------------
# Rationale Source Field Tests
# ---------------------------------------------------------------------------

class TestRationaleSourceField:

    @pytest.mark.schema
    def test_rationale_source_defaults_to_none(self):
        """rationale_source defaults to None."""
        out = _make_reconciliation()
        assert out.rationale_source is None

    @pytest.mark.schema
    def test_rationale_source_llm_accepted(self):
        """rationale_source='llm' is valid."""
        out = _make_reconciliation(rationale_source="llm")
        assert out.rationale_source == "llm"

    @pytest.mark.schema
    def test_rationale_source_deterministic_accepted(self):
        """rationale_source='deterministic' is valid."""
        out = _make_reconciliation(rationale_source="deterministic")
        assert out.rationale_source == "deterministic"

    @pytest.mark.schema
    def test_rationale_source_invalid_rejected(self):
        """Invalid rationale_source is rejected."""
        with pytest.raises(ValueError, match="rationale_source"):
            _make_reconciliation(rationale_source="other")


# ---------------------------------------------------------------------------
# Cap Size Field Tests
# ---------------------------------------------------------------------------

class TestCapSizeField:

    @pytest.mark.schema
    def test_position_action_accepts_cap_size(self):
        """PositionAction accepts cap_size."""
        a = PositionAction(
            symbol="NVDA", action_type="BUY",
            current_pct=0.0, target_pct=3.0,
            dollar_change=45600.0, priority="MEDIUM", week=2,
            rationale="New T1 position at 3.0% allocation for portfolio",
            cap_size="large",
        )
        assert a.cap_size == "large"

    @pytest.mark.schema
    def test_position_action_cap_size_default_none(self):
        """cap_size defaults to None."""
        a = _make_action()
        assert a.cap_size is None
