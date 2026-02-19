"""
Target Return Constructor Tool: Constraint Validator
IBD Momentum Investment Framework v4.0

Pure functions for:
- Validating target portfolio against risk constraints
- Checking position concentration limits
- Checking sector concentration limits
- Generating warnings for soft constraint violations

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

from ibd_agents.schemas.target_return_output import (
    MAX_POSITIONS,
    MAX_SECTOR_CONCENTRATION_PCT,
    MAX_SINGLE_POSITION_PCT,
    MIN_POSITIONS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------

def validate_constraints(
    positions: list[dict],
    cash_reserve_pct: float,
    max_drawdown_pct: float = 0.0,
    prob_achieve_target: float = 0.0,
    t1_allocation_pct: float = 0.0,
) -> dict:
    """
    Validate portfolio against risk constraints.

    Args:
        positions: List of position dicts with allocation_pct, sector,
            stop_loss_price, target_entry_price.
        cash_reserve_pct: Cash reserve as % of portfolio.
        max_drawdown_pct: Estimated max drawdown (for soft check).
        prob_achieve_target: Probability of achieving target (for soft check).
        t1_allocation_pct: T1 tier allocation (for soft check).

    Returns:
        {
            "passed": bool,
            "violations": list[str],  # hard constraint failures
            "warnings": list[str],    # soft constraint warnings
        }
    """
    violations: list[str] = []
    warnings: list[str] = []

    # --- Hard Checks ---

    # 1. Position count
    n_positions = len(positions)
    if n_positions < MIN_POSITIONS:
        violations.append(
            f"Too few positions: {n_positions} < minimum {MIN_POSITIONS}"
        )
    if n_positions > MAX_POSITIONS:
        violations.append(
            f"Too many positions: {n_positions} > maximum {MAX_POSITIONS}"
        )

    # 2. Single position concentration
    for pos in positions:
        alloc = pos.get("allocation_pct", 0)
        ticker = pos.get("ticker", "?")
        if alloc > MAX_SINGLE_POSITION_PCT + 0.1:
            violations.append(
                f"Position {ticker} allocation {alloc:.1f}% exceeds "
                f"max {MAX_SINGLE_POSITION_PCT}%"
            )

    # 3. Sector concentration
    sector_totals: dict[str, float] = {}
    for pos in positions:
        sector = pos.get("sector", "UNKNOWN")
        sector_totals[sector] = sector_totals.get(sector, 0) + pos.get("allocation_pct", 0)

    for sector, total in sector_totals.items():
        if total > MAX_SECTOR_CONCENTRATION_PCT + 0.1:
            violations.append(
                f"Sector {sector} weight {total:.1f}% exceeds "
                f"max {MAX_SECTOR_CONCENTRATION_PCT}%"
            )

    # 4. Allocations + cash sum to ~100%
    pos_sum = sum(pos.get("allocation_pct", 0) for pos in positions)
    total_alloc = pos_sum + cash_reserve_pct
    if not (97.0 <= total_alloc <= 103.0):
        violations.append(
            f"Total allocation {total_alloc:.1f}% not within 97-103% "
            f"(positions={pos_sum:.1f}%, cash={cash_reserve_pct:.1f}%)"
        )

    # 5. Stop loss below entry price
    for pos in positions:
        stop = pos.get("stop_loss_price", 0)
        entry = pos.get("target_entry_price", 0)
        if stop > 0 and entry > 0 and stop >= entry:
            violations.append(
                f"Position {pos.get('ticker', '?')}: stop_loss_price "
                f"({stop}) >= target_entry_price ({entry})"
            )

    # --- Soft Checks (warnings only) ---

    if prob_achieve_target > 0 and prob_achieve_target < 0.30:
        warnings.append(
            f"Low probability of achieving target: {prob_achieve_target:.0%} — "
            f"target may be unrealistic in current conditions"
        )

    if max_drawdown_pct < -25.0:
        warnings.append(
            f"High drawdown risk: max drawdown {max_drawdown_pct:.1f}% "
            f"exceeds -25% threshold"
        )

    if cash_reserve_pct < 5.0:
        warnings.append(
            f"Low cash reserve: {cash_reserve_pct:.1f}% — "
            f"no buffer for opportunities or margin calls"
        )

    if n_positions < 6:
        warnings.append(
            f"Very concentrated portfolio: only {n_positions} positions"
        )

    if t1_allocation_pct > 70.0:
        warnings.append(
            f"Extreme momentum exposure: T1 allocation {t1_allocation_pct:.0f}% > 70%"
        )

    passed = len(violations) == 0

    logger.info(
        f"[ConstraintValidator] {'PASSED' if passed else 'FAILED'}: "
        f"{len(violations)} violations, {len(warnings)} warnings"
    )

    return {
        "passed": passed,
        "violations": violations,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class ConstraintValidatorInput(BaseModel):
    position_count: int = Field(
        ..., description="Number of positions in portfolio"
    )
    cash_pct: float = Field(
        ..., description="Cash reserve percentage"
    )


class ConstraintValidatorTool(BaseTool):
    """Validate target return portfolio against risk constraints."""

    name: str = "constraint_validator_target"
    description: str = (
        "Validate a target return portfolio against position concentration, "
        "sector concentration, and other risk constraints. Returns pass/fail "
        "with violation details."
    )
    args_schema: type[BaseModel] = ConstraintValidatorInput

    def _run(self, position_count: int, cash_pct: float) -> str:
        import json
        return json.dumps({
            "note": "Use validate_constraints() directly with position data",
        })
