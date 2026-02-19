"""
Target Return Constructor Tool: Transition Planner
IBD Momentum Investment Framework v4.0

Pure functions for:
- Diffing current holdings vs target positions
- Building ordered transition action plan
- Estimating transition urgency

No LLM, no file I/O, no buy/sell recommendations.
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------

def build_transition_plan(
    current_holdings: list[dict],
    target_positions: list[dict],
    total_capital: float,
) -> dict:
    """
    Build a transition plan from current holdings to target portfolio.

    Args:
        current_holdings: List of {ticker, allocation_pct} for current positions.
        target_positions: List of {ticker, allocation_pct} for target positions.
        total_capital: Total portfolio capital.

    Returns:
        Dict compatible with TransitionPlan schema:
        {
            positions_to_sell: [...],
            positions_to_buy: [...],
            positions_to_resize: [...],
            positions_to_keep: [...],
            transition_urgency: str,
            transition_sequence: [...],
        }
    """
    current_map = {h["ticker"]: h.get("allocation_pct", 0) for h in current_holdings}
    target_map = {p["ticker"]: p.get("allocation_pct", 0) for p in target_positions}

    all_tickers = set(current_map.keys()) | set(target_map.keys())

    sells = []
    buys = []
    resizes = []
    keeps = []

    priority = 1

    for ticker in sorted(all_tickers):
        current_pct = current_map.get(ticker, 0)
        target_pct = target_map.get(ticker, 0)
        diff = target_pct - current_pct

        if current_pct > 0 and target_pct == 0:
            # Sell entirely
            dollar_amount = total_capital * current_pct / 100.0
            sells.append({
                "ticker": ticker,
                "action": "SELL_FULL",
                "current_allocation_pct": current_pct,
                "target_allocation_pct": 0.0,
                "dollar_amount": round(dollar_amount, 2),
                "priority": priority,
                "rationale": f"Not in target portfolio — sell to free capital",
            })
            priority += 1

        elif current_pct == 0 and target_pct > 0:
            # Buy new
            dollar_amount = total_capital * target_pct / 100.0
            buys.append({
                "ticker": ticker,
                "action": "BUY_NEW",
                "current_allocation_pct": 0.0,
                "target_allocation_pct": target_pct,
                "dollar_amount": round(dollar_amount, 2),
                "priority": priority,
                "rationale": f"New position for target return portfolio at {target_pct:.1f}%",
            })
            priority += 1

        elif abs(diff) < 0.5:
            # Close enough — keep
            keeps.append(ticker)

        elif diff > 0:
            # Resize up
            dollar_amount = total_capital * diff / 100.0
            resizes.append({
                "ticker": ticker,
                "action": "RESIZE_UP",
                "current_allocation_pct": current_pct,
                "target_allocation_pct": target_pct,
                "dollar_amount": round(dollar_amount, 2),
                "priority": priority,
                "rationale": f"Increase from {current_pct:.1f}% to {target_pct:.1f}%",
            })
            priority += 1

        else:
            # Resize down
            dollar_amount = total_capital * abs(diff) / 100.0
            resizes.append({
                "ticker": ticker,
                "action": "RESIZE_DOWN",
                "current_allocation_pct": current_pct,
                "target_allocation_pct": target_pct,
                "dollar_amount": round(dollar_amount, 2),
                "priority": priority,
                "rationale": f"Decrease from {current_pct:.1f}% to {target_pct:.1f}%",
            })
            priority += 1

    # Determine urgency based on number of actions
    total_actions = len(sells) + len(buys) + len(resizes)
    if total_actions <= 5:
        urgency = "immediate"
    elif total_actions <= 10:
        urgency = "phase_over_1_week"
    else:
        urgency = "phase_over_2_weeks"

    # Build transition sequence
    sequence = []
    if sells:
        sequence.append(
            f"Step 1: Sell {len(sells)} positions to free capital "
            f"(${sum(s['dollar_amount'] for s in sells):,.0f})"
        )
    if resizes:
        resize_down = [r for r in resizes if r["action"] == "RESIZE_DOWN"]
        if resize_down:
            sequence.append(
                f"Step {len(sequence) + 1}: Trim {len(resize_down)} positions"
            )
    if buys:
        sequence.append(
            f"Step {len(sequence) + 1}: Buy {len(buys)} new positions "
            f"(${sum(b['dollar_amount'] for b in buys):,.0f})"
        )
    resize_up = [r for r in resizes if r["action"] == "RESIZE_UP"]
    if resize_up:
        sequence.append(
            f"Step {len(sequence) + 1}: Increase {len(resize_up)} existing positions"
        )
    if not sequence:
        sequence.append("No transitions needed — portfolio already matches target")

    result = {
        "positions_to_sell": sells,
        "positions_to_buy": buys,
        "positions_to_resize": resizes,
        "positions_to_keep": keeps,
        "transition_urgency": urgency,
        "transition_sequence": sequence,
    }

    logger.info(
        f"[TransitionPlanner] {len(sells)} sells, {len(buys)} buys, "
        f"{len(resizes)} resizes, {len(keeps)} keeps, urgency={urgency}"
    )

    return result


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class TransitionPlannerInput(BaseModel):
    total_capital: float = Field(
        ..., description="Total portfolio capital"
    )


class TransitionPlannerTool(BaseTool):
    """Plan transition from current holdings to target return portfolio."""

    name: str = "transition_planner"
    description: str = (
        "Build a transition plan from current portfolio holdings to "
        "a target return portfolio, with ordered actions and urgency."
    )
    args_schema: type[BaseModel] = TransitionPlannerInput

    def _run(self, total_capital: float) -> str:
        import json
        return json.dumps({
            "note": "Use build_transition_plan() directly with holdings data",
        })
