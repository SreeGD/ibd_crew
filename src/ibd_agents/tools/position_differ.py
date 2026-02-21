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
from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ibd_agents.schemas.value_investor_output import ValueInvestorOutput

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
# Priority Helpers
# ---------------------------------------------------------------------------

def _sell_priority(current_pct: float) -> str:
    """Prioritize sells by position size — largest frees the most capital first."""
    if current_pct >= 1.0:
        return "HIGH"
    if current_pct >= 0.4:
        return "MEDIUM"
    return "LOW"


def _buy_priority(selection_source: str | None, conviction: int) -> str:
    """Prioritize buys by selection source and conviction."""
    # Keeps and strategic picks (value/pattern) are highest priority
    if selection_source in ("keep", "value", "pattern"):
        return "HIGH"
    # High-conviction momentum
    if conviction >= 9:
        return "HIGH"
    if conviction >= 7:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Sell Quality Gate Constants
# ---------------------------------------------------------------------------

# Rule 1: High-conviction T1 shield
HIGH_CONVICTION_THRESHOLD = 8       # conviction >= this
HIGH_CONVICTION_TIER = 1            # must be T1

# Rule 2: Strong ETF protection
STRONG_ETF_RANK_THRESHOLD = 10      # etf_rank <= this
STRONG_ETF_SCORE_THRESHOLD = 70.0   # etf_score >= this

# Rule 3: Bear-hedge symbols preserved in bear regime
BEAR_HEDGE_SYMBOLS = {"GLD", "SLV", "TLT", "IAU", "SHY", "IEF", "SGOL", "GLDM"}

# Rule 4: Analyst-rated quality gate
QUALITY_COMPOSITE_THRESHOLD = 95    # composite >= this
QUALITY_RS_THRESHOLD = 90           # rs >= this
QUALITY_SHARPE_THRESHOLD = 1.0      # sharpe >= this

# Rule 5: Value shield — protect strong value stocks from liquidation
VALUE_SHIELD_MIN_SCORE = 60.0       # value_score >= this
VALUE_SHIELD_CATEGORIES = {"Quality Value", "GARP", "Deep Value"}
VALUE_SHIELD_MAX_TRAP_RISK = {"None", "Low"}  # value_trap_risk_level in these

# Rule 6: Turnover cap
MAX_TURNOVER_RATIO = 0.50           # max 50% of actions can be SELL+BUY

# Rule 7: Graduated sell target
GRADUATED_TRIM_TARGET_PCT = 0.4     # target_pct for converted positions
GRADUATED_TRIM_WEEK = 3             # week for converted trims


# ---------------------------------------------------------------------------
# Sell Quality Gate
# ---------------------------------------------------------------------------

@dataclass
class SellQualityContext:
    """Lookup data for sell quality gate decisions."""
    # symbol -> {tier, conviction, composite_rating, rs_rating, sharpe_ratio, sector}
    stock_map: dict[str, dict] = dc_field(default_factory=dict)
    # symbol -> {etf_rank, etf_score, conviction, tier}
    etf_map: dict[str, dict] = dc_field(default_factory=dict)
    # Market regime: "bull", "bear", "neutral", or None
    regime: str | None = None
    # symbol -> {value_score, value_category, value_trap_risk_level}
    value_map: dict[str, dict] = dc_field(default_factory=dict)


def build_sell_quality_context(
    analyst_output=None,
    rotation_output=None,
    value_output: Optional["ValueInvestorOutput"] = None,
) -> SellQualityContext:
    """Build lookup context from analyst, rotation, and value data.

    Returns empty context if no data is provided (no conversions will happen).
    """
    ctx = SellQualityContext()

    if analyst_output is not None:
        for rs in analyst_output.rated_stocks:
            ctx.stock_map[rs.symbol] = {
                "tier": rs.tier,
                "conviction": rs.conviction,
                "composite_rating": rs.composite_rating,
                "rs_rating": rs.rs_rating,
                "sharpe_ratio": getattr(rs, "sharpe_ratio", None),
                "sector": rs.sector,
            }
        for etf in analyst_output.rated_etfs:
            ctx.etf_map[etf.symbol] = {
                "tier": etf.tier,
                "etf_rank": etf.etf_rank,
                "etf_score": etf.etf_score,
                "conviction": etf.conviction,
            }

    if rotation_output is not None:
        ctx.regime = rotation_output.market_regime.regime

    if value_output is not None:
        for vs in value_output.value_stocks:
            ctx.value_map[vs.symbol] = {
                "value_score": vs.value_score,
                "value_category": vs.value_category,
                "value_trap_risk_level": vs.value_trap_risk_level,
            }

    return ctx


def _check_sell_shield(
    action: PositionAction,
    ctx: SellQualityContext,
) -> tuple[bool, str]:
    """Check if a SELL action should be shielded (converted to TRIM).

    Returns (should_shield, reason). Only one rule needs to match.
    Rules are checked in priority order.
    """
    sym = action.symbol

    # Rule 3: Bear-hedge preservation (check first — regime-specific)
    if ctx.regime == "bear" and sym in BEAR_HEDGE_SYMBOLS:
        return True, f"Bear-hedge preserved in bear regime ({sym})"

    # Rule 1: High-conviction T1 shield
    stock_info = ctx.stock_map.get(sym)
    if stock_info:
        if (stock_info["conviction"] >= HIGH_CONVICTION_THRESHOLD
                and stock_info["tier"] == HIGH_CONVICTION_TIER):
            return True, (
                f"High-conviction T1 shield (conviction={stock_info['conviction']}, "
                f"tier={stock_info['tier']})"
            )

        # Rule 4: Analyst-rated quality gate
        sharpe = stock_info.get("sharpe_ratio")
        if (stock_info["composite_rating"] >= QUALITY_COMPOSITE_THRESHOLD
                and stock_info["rs_rating"] >= QUALITY_RS_THRESHOLD
                and sharpe is not None
                and sharpe >= QUALITY_SHARPE_THRESHOLD):
            return True, (
                f"Quality gate: Comp={stock_info['composite_rating']}, "
                f"RS={stock_info['rs_rating']}, Sharpe={sharpe:.2f}"
            )

    # Rule 2: Strong ETF protection
    etf_info = ctx.etf_map.get(sym)
    if etf_info:
        if etf_info["etf_rank"] <= STRONG_ETF_RANK_THRESHOLD:
            return True, f"Strong ETF (rank #{etf_info['etf_rank']})"
        if etf_info["etf_score"] >= STRONG_ETF_SCORE_THRESHOLD:
            return True, f"Strong ETF (score={etf_info['etf_score']:.1f})"

    # Rule 5: Value shield — protect strong value stocks from liquidation
    value_info = ctx.value_map.get(sym)
    if value_info:
        if (value_info["value_score"] >= VALUE_SHIELD_MIN_SCORE
                and value_info["value_category"] in VALUE_SHIELD_CATEGORIES
                and value_info["value_trap_risk_level"] in VALUE_SHIELD_MAX_TRAP_RISK):
            return True, (
                f"Value shield: {value_info['value_category']} "
                f"(score={value_info['value_score']:.0f}, "
                f"trap_risk={value_info['value_trap_risk_level']})"
            )

    return False, ""


def apply_sell_quality_gate(
    actions: list[PositionAction],
    ctx: SellQualityContext,
    total_value: float = 1_520_000.0,
) -> list[PositionAction]:
    """Review SELL actions and convert eligible ones to graduated TRIMs.

    Applies per-position shields (rules 1-5), then turnover cap (rule 6).
    Converted TRIMs get a small target_pct and later-week scheduling (rule 7).

    Args:
        actions: List of PositionAction from diff_positions()
        ctx: SellQualityContext with analyst/rotation data
        total_value: Portfolio total value for dollar_change calculations

    Returns:
        Modified list of PositionAction with some SELLs converted to TRIMs
    """
    if not ctx.stock_map and not ctx.etf_map and ctx.regime is None and not ctx.value_map:
        # No analyst, regime, or value data — cannot evaluate quality, return unchanged
        return actions

    converted_indices: list[int] = []
    reasons: dict[str, str] = {}

    # --- Pass 1: Per-position shields (rules 1-5) ---
    for idx, action in enumerate(actions):
        if action.action_type != "SELL":
            continue

        should_shield, reason = _check_sell_shield(action, ctx)
        if should_shield:
            converted_indices.append(idx)
            reasons[action.symbol] = reason

    # --- Pass 2: Turnover cap (rule 6) ---
    remaining_sell_count = sum(
        1 for i, a in enumerate(actions)
        if a.action_type == "SELL" and i not in converted_indices
    )
    buy_count = sum(1 for a in actions if a.action_type == "BUY")
    total_actions = len(actions)

    if total_actions >= 10:
        churn_count = remaining_sell_count + buy_count
        churn_ratio = churn_count / total_actions

        if churn_ratio > MAX_TURNOVER_RATIO:
            target_churn = int(MAX_TURNOVER_RATIO * total_actions)
            excess = churn_count - target_churn

            _PRI_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            remaining_sells = [
                (idx, actions[idx])
                for idx in range(len(actions))
                if actions[idx].action_type == "SELL"
                and idx not in converted_indices
            ]
            remaining_sells.sort(
                key=lambda x: (_PRI_ORDER.get(x[1].priority, 1), abs(x[1].dollar_change))
            )

            for i, (idx, action) in enumerate(remaining_sells):
                if i >= excess:
                    break
                converted_indices.append(idx)
                reasons[action.symbol] = "Turnover cap — lowest-priority sell converted"

    # --- Pass 3: Apply conversions (rule 6: graduated trim) ---
    converted_set = set(converted_indices)
    result: list[PositionAction] = []

    for idx, action in enumerate(actions):
        if idx in converted_set:
            target = GRADUATED_TRIM_TARGET_PCT
            dollar_change = (target - action.current_pct) / 100.0 * total_value
            reason = reasons.get(action.symbol, "Quality gate")

            result.append(PositionAction(
                symbol=action.symbol,
                action_type="TRIM",
                cap_size=action.cap_size,
                current_pct=action.current_pct,
                target_pct=round(target, 2),
                dollar_change=round(dollar_change, 2),
                priority="LOW",
                week=GRADUATED_TRIM_WEEK,
                rationale=(
                    f"Reduced (not liquidated): {reason}. "
                    f"Trim from {action.current_pct:.1f}% to {target}%"
                ),
                sharpe_ratio=action.sharpe_ratio,
                alpha_pct=action.alpha_pct,
            ))
        else:
            result.append(action)

    if converted_indices:
        logger.info(
            f"[Sell Quality Gate] Converted {len(converted_indices)} SELLs to TRIMs: "
            f"{[actions[i].symbol for i in sorted(converted_indices)]}"
        )

    return result


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
                "selection_source": getattr(p, "selection_source", None),
                "conviction": getattr(p, "conviction", 5),
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
                source = rec_map[sym].get("selection_source")
                priority = _buy_priority(source, rec_map[sym].get("conviction", 5))
                actions.append(PositionAction(
                    symbol=sym,
                    action_type="ADD",
                    cap_size=cap,
                    current_pct=round(current_pct, 2),
                    target_pct=round(target_pct, 2),
                    dollar_change=round(dollar_change, 2),
                    priority=priority,
                    week=week,
                    rationale=f"Increase position from {current_pct:.1f}% to {target_pct:.1f}% (T{tier})",
                ))
            else:
                # Too much — TRIM
                priority = "HIGH" if abs(diff_pct) > 1.0 else "MEDIUM"
                actions.append(PositionAction(
                    symbol=sym,
                    action_type="TRIM",
                    cap_size=cap,
                    current_pct=round(current_pct, 2),
                    target_pct=round(target_pct, 2),
                    dollar_change=round(dollar_change, 2),
                    priority=priority,
                    week=1,
                    rationale=f"Reduce position from {current_pct:.1f}% to {target_pct:.1f}% (T{tier})",
                ))
        else:
            # Not recommended — SELL
            priority = _sell_priority(current_pct)
            actions.append(PositionAction(
                symbol=sym,
                action_type="SELL",
                current_pct=round(current_pct, 2),
                target_pct=0.0,
                dollar_change=round(-holding.market_value, 2),
                priority=priority,
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
            source = info.get("selection_source")
            priority = _buy_priority(source, info.get("conviction", 5))

            actions.append(PositionAction(
                symbol=sym,
                action_type="BUY",
                cap_size=cap,
                current_pct=0.0,
                target_pct=round(target_pct, 2),
                dollar_change=round(dollar, 2),
                priority=priority,
                week=week,
                rationale=f"New T{tier} position at {target_pct:.1f}% allocation",
            ))

    # 3. Sort actions within each week: HIGH first, then by absolute dollar impact
    _PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    actions.sort(key=lambda a: (a.week, _PRIORITY_ORDER.get(a.priority, 1), -abs(a.dollar_change)))

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
