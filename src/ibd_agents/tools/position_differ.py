"""
Portfolio Reconciler Tool: Position Differ
IBD Momentum Investment Framework v4.0

Pure functions for:
- Diffing current holdings vs recommended positions
- Computing money flow (sources/uses)
- Building 4-week implementation plan
- Verifying keep positions

No LLM, no file I/O.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

from ibd_agents.schemas.portfolio_output import ALL_KEEPS, PortfolioOutput
from ibd_agents.schemas.reconciliation_output import (
    ACTION_TYPES,
    IMPLEMENTATION_PHASES,
    CurrentHolding,
    HoldingsSummary,
    ImplementationWeek,
    KeepVerification,
    MoneyFlow,
    PositionAction,
    TransformationMetrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------

def diff_positions(
    current_holdings: HoldingsSummary,
    portfolio_output: PortfolioOutput,
) -> List[PositionAction]:
    """
    Diff current holdings vs recommended portfolio.

    Returns list of actions:
    - KEEP: in both, size is approximately right
    - SELL: in current but not recommended
    - BUY: in recommended but not current
    - ADD: in both but current < target
    - TRIM: in both but current > target
    """
    total_value = current_holdings.total_value
    if total_value == 0:
        total_value = 1_520_000.0  # Default

    # Build lookup: symbol → current holding
    current_map: dict[str, CurrentHolding] = {}
    for h in current_holdings.holdings:
        current_map[h.symbol] = h

    # Build lookup: symbol → recommended position info
    rec_map: dict[str, dict] = {}
    for tier in [portfolio_output.tier_1, portfolio_output.tier_2, portfolio_output.tier_3]:
        for p in tier.positions:
            rec_map[p.symbol] = {
                "target_pct": p.target_pct,
                "tier": p.tier,
                "sector": p.sector,
                "cap_size": p.cap_size,
            }

    actions: list[PositionAction] = []

    # 1. Process current holdings
    for sym, holding in current_map.items():
        current_pct = (holding.market_value / total_value) * 100

        if sym in rec_map:
            target_pct = rec_map[sym]["target_pct"]
            tier = rec_map[sym]["tier"]
            diff_pct = target_pct - current_pct
            dollar_change = diff_pct / 100.0 * total_value

            cap = rec_map[sym].get("cap_size")
            if abs(diff_pct) < 0.3:
                # Close enough — KEEP
                actions.append(PositionAction(
                    symbol=sym,
                    action_type="KEEP",
                    cap_size=cap,
                    current_pct=round(current_pct, 2),
                    target_pct=round(target_pct, 2),
                    dollar_change=0.0,
                    priority="LOW",
                    week=1,
                    rationale=f"Position approximately at target ({current_pct:.1f}% vs {target_pct:.1f}%)",
                ))
            elif diff_pct > 0:
                # Need more — ADD
                week = {1: 2, 2: 3, 3: 4}.get(tier, 3)
                actions.append(PositionAction(
                    symbol=sym,
                    action_type="ADD",
                    cap_size=cap,
                    current_pct=round(current_pct, 2),
                    target_pct=round(target_pct, 2),
                    dollar_change=round(dollar_change, 2),
                    priority="MEDIUM",
                    week=week,
                    rationale=f"Increase position from {current_pct:.1f}% to {target_pct:.1f}% (T{tier})",
                ))
            else:
                # Too much — TRIM
                actions.append(PositionAction(
                    symbol=sym,
                    action_type="TRIM",
                    cap_size=cap,
                    current_pct=round(current_pct, 2),
                    target_pct=round(target_pct, 2),
                    dollar_change=round(dollar_change, 2),
                    priority="MEDIUM",
                    week=1,
                    rationale=f"Reduce position from {current_pct:.1f}% to {target_pct:.1f}% (T{tier})",
                ))
        else:
            # Not recommended — SELL
            actions.append(PositionAction(
                symbol=sym,
                action_type="SELL",
                current_pct=round(current_pct, 2),
                target_pct=0.0,
                dollar_change=round(-holding.market_value, 2),
                priority="HIGH",
                week=1,
                rationale=f"Position not in recommended portfolio — liquidate {current_pct:.1f}%",
            ))

    # 2. Process recommended positions not in current
    for sym, info in rec_map.items():
        if sym not in current_map:
            target_pct = info["target_pct"]
            tier = info["tier"]
            cap = info.get("cap_size")
            dollar = target_pct / 100.0 * total_value
            week = {1: 2, 2: 3, 3: 4}.get(tier, 3)

            actions.append(PositionAction(
                symbol=sym,
                action_type="BUY",
                cap_size=cap,
                current_pct=0.0,
                target_pct=round(target_pct, 2),
                dollar_change=round(dollar, 2),
                priority="MEDIUM",
                week=week,
                rationale=f"New T{tier} position at {target_pct:.1f}% allocation",
            ))

    return actions


def compute_money_flow(
    actions: List[PositionAction],
    total_value: float,
) -> MoneyFlow:
    """Compute money flow from actions."""
    sell_proceeds = sum(abs(a.dollar_change) for a in actions if a.action_type == "SELL")
    trim_proceeds = sum(abs(a.dollar_change) for a in actions if a.action_type == "TRIM")
    buy_cost = sum(a.dollar_change for a in actions if a.action_type == "BUY")
    add_cost = sum(a.dollar_change for a in actions if a.action_type == "ADD")

    net = (sell_proceeds + trim_proceeds) - (buy_cost + add_cost)
    cash_pct = (net / total_value) * 100 if total_value > 0 else 0

    return MoneyFlow(
        sell_proceeds=round(sell_proceeds, 2),
        trim_proceeds=round(trim_proceeds, 2),
        buy_cost=round(buy_cost, 2),
        add_cost=round(add_cost, 2),
        net_cash_change=round(net, 2),
        cash_reserve_pct=round(max(0, cash_pct), 2),
    )


def build_implementation_plan(
    actions: List[PositionAction],
) -> List[ImplementationWeek]:
    """Build 4-week implementation plan from actions."""
    weeks: list[ImplementationWeek] = []

    for w in range(1, 5):
        phase = IMPLEMENTATION_PHASES[w]
        week_actions = [a for a in actions if a.week == w]

        weeks.append(ImplementationWeek(
            week=w,
            phase_name=phase,
            actions=week_actions,
        ))

    return weeks


def verify_keeps(
    current_holdings: HoldingsSummary,
) -> KeepVerification:
    """Verify which keeps are in current holdings."""
    current_symbols = {h.symbol for h in current_holdings.holdings}

    in_current = [k for k in ALL_KEEPS if k in current_symbols]
    missing = [k for k in ALL_KEEPS if k not in current_symbols]

    return KeepVerification(
        keeps_in_current=in_current,
        keeps_missing=missing,
        keeps_to_buy=missing,
    )


def compute_transformation_metrics(
    current_holdings: HoldingsSummary,
    portfolio_output: PortfolioOutput,
    actions: List[PositionAction],
) -> TransformationMetrics:
    """Compute before/after portfolio transformation metrics."""
    before_sectors = len(set(h.sector for h in current_holdings.holdings))
    after_sectors = len(portfolio_output.sector_exposure)
    before_count = len(current_holdings.holdings)
    after_count = portfolio_output.total_positions

    # Turnover = (buys + sells) / total actions
    changes = sum(1 for a in actions if a.action_type in ("BUY", "SELL"))
    total = len(actions) if actions else 1
    turnover = (changes / total) * 100

    return TransformationMetrics(
        before_sector_count=before_sectors,
        after_sector_count=after_sectors,
        before_position_count=before_count,
        after_position_count=after_count,
        turnover_pct=round(turnover, 1),
    )


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class PositionDifferInput(BaseModel):
    portfolio_json: str = Field(..., description="JSON string of PortfolioOutput")
    holdings_json: str = Field(..., description="JSON string of HoldingsSummary")


class PositionDifferTool(BaseTool):
    """Diff current holdings vs recommended portfolio."""

    name: str = "position_differ"
    description: str = (
        "Compare current holdings against recommended portfolio "
        "and generate KEEP/SELL/BUY/ADD/TRIM actions"
    )
    args_schema: type[BaseModel] = PositionDifferInput

    def _run(self, portfolio_json: str, holdings_json: str) -> str:
        import json
        portfolio = PortfolioOutput.model_validate_json(portfolio_json)
        holdings = HoldingsSummary.model_validate_json(holdings_json)

        actions = diff_positions(holdings, portfolio)
        return json.dumps([a.model_dump() for a in actions], indent=2)
