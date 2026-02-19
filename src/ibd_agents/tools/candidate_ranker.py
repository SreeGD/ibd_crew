"""
Target Return Constructor Tool: Candidate Ranker
IBD Momentum Investment Framework v4.0

Pure functions for:
- Ranking stocks within each tier for target return selection
- Scoring candidates based on composite, RS, conviction, momentum
- Selecting and sizing positions for concentrated portfolio

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

from ibd_agents.schemas.analyst_output import RatedStock
from ibd_agents.schemas.target_return_output import (
    CONVICTION_LEVELS,
    ENTRY_STRATEGIES,
    MAX_SINGLE_POSITION_PCT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Score weights for candidate ranking
COMPOSITE_WEIGHT: float = 0.25
RS_WEIGHT: float = 0.20
CONVICTION_WEIGHT: float = 0.15
VALIDATION_WEIGHT: float = 0.10
MOMENTUM_WEIGHT: float = 0.15
RISK_ADJUSTED_WEIGHT: float = 0.15

# Sector momentum bonuses
SECTOR_MOMENTUM_BONUS: dict[str, float] = {
    "leading": 15.0,
    "improving": 5.0,
    "neutral": 0.0,
    "lagging": -5.0,
    "declining": -10.0,
}

# Estimated prices for position sizing (by sector, rough averages)
SECTOR_PRICE_ESTIMATES: dict[str, float] = {
    "CHIPS": 180.0, "COMPUTER": 200.0, "SOFTWARE": 150.0,
    "INTERNET": 175.0, "ELECTRONICS": 120.0, "HEALTHCARE": 250.0,
    "MEDICAL": 200.0, "ENERGY": 100.0, "MINING": 80.0,
    "CHEMICALS": 130.0, "BANKS": 70.0, "FINANCE": 90.0,
    "INSURANCE": 120.0, "REAL ESTATE": 60.0, "AEROSPACE": 300.0,
    "BUILDING": 150.0, "MACHINERY": 200.0, "TRANSPORTATION": 130.0,
    "DEFENSE": 250.0, "RETAIL": 100.0, "LEISURE": 80.0,
    "MEDIA": 90.0, "AUTO": 100.0, "UTILITIES": 70.0,
    "FOOD/BEVERAGE": 60.0, "CONSUMER": 80.0,
    "BUSINESS SERVICES": 140.0, "TELECOM": 50.0,
}
DEFAULT_PRICE: float = 120.0


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------

def rank_candidates(
    rated_stocks: list[RatedStock],
    tier: int,
    sector_momentum_map: Optional[dict[str, str]] = None,
    required_stocks: Optional[list[str]] = None,
    excluded_stocks: Optional[list[str]] = None,
    max_count: int = 20,
    stock_metrics: Optional[dict[str, dict]] = None,
) -> list[dict]:
    """
    Rank stocks within a tier for target return portfolio selection.

    Score = composite * 0.25 + rs * 0.20 + conviction_norm * 0.15
          + validation_bonus * 0.10 + sector_momentum * 0.15
          + risk_adjusted * 0.15

    Args:
        rated_stocks: All rated stocks from analyst output.
        tier: Target tier (1, 2, or 3).
        sector_momentum_map: sector → momentum status mapping.
        required_stocks: Tickers that must be included.
        excluded_stocks: Tickers to exclude.
        max_count: Maximum candidates to return.
        stock_metrics: Optional ticker → {sharpe_ratio, estimated_volatility_pct, estimated_beta} map.

    Returns:
        Ranked list of candidate dicts with score, rank, and stock data.
    """
    if sector_momentum_map is None:
        sector_momentum_map = {}
    if required_stocks is None:
        required_stocks = []
    if excluded_stocks is None:
        excluded_stocks = []

    excluded_set = set(s.upper() for s in excluded_stocks)

    # Filter to target tier, excluding excluded stocks
    tier_stocks = [
        s for s in rated_stocks
        if s.tier == tier and s.symbol.upper() not in excluded_set
    ]

    # Score each candidate
    candidates = []
    for stock in tier_stocks:
        # Normalize conviction (1-10) to 0-100 scale
        conviction_norm = stock.conviction * 10.0

        # Multi-source validation bonus
        validation_bonus = 10.0 if stock.is_multi_source_validated else 0.0

        # Sector momentum bonus
        momentum = sector_momentum_map.get(stock.sector, "neutral")
        momentum_bonus = SECTOR_MOMENTUM_BONUS.get(momentum, 0.0)

        # Risk-adjusted score (Sharpe ratio normalized to 0-100 scale)
        risk_adj_bonus = 0.0
        if stock_metrics and stock.symbol in stock_metrics:
            sharpe = stock_metrics[stock.symbol].get("sharpe_ratio")
            if sharpe is not None:
                # Sharpe typically -1 to 3; normalize to 0-100
                risk_adj_bonus = max(0.0, min(100.0, (sharpe + 1.0) * 25.0))

        score = (
            stock.composite_rating * COMPOSITE_WEIGHT
            + stock.rs_rating * RS_WEIGHT
            + conviction_norm * CONVICTION_WEIGHT
            + validation_bonus * VALIDATION_WEIGHT
            + momentum_bonus * MOMENTUM_WEIGHT
            + risk_adj_bonus * RISK_ADJUSTED_WEIGHT
        )

        candidates.append({
            "ticker": stock.symbol,
            "company_name": stock.company_name,
            "tier": stock.tier,
            "sector": stock.sector,
            "composite_score": stock.composite_rating,
            "eps_rating": stock.eps_rating,
            "rs_rating": stock.rs_rating,
            "conviction": stock.conviction,
            "is_multi_source": stock.is_multi_source_validated,
            "sector_rank": stock.sector_rank_in_sector,
            "multi_source_count": stock.validation_score,
            "candidate_score": round(score, 2),
            "is_required": stock.symbol.upper() in [r.upper() for r in required_stocks],
            "cap_size": stock.cap_size,
            "estimated_return_pct": stock.estimated_return_pct,
        })

    # Sort: required first, then by score descending
    candidates.sort(
        key=lambda c: (not c["is_required"], -c["candidate_score"]),
    )

    # Assign ranks
    for i, c in enumerate(candidates[:max_count]):
        c["rank"] = i + 1

    result = candidates[:max_count]
    logger.info(
        f"[CandidateRanker] Tier {tier}: {len(tier_stocks)} stocks → "
        f"top {len(result)} candidates (top score={result[0]['candidate_score']:.1f})"
        if result else f"[CandidateRanker] Tier {tier}: no candidates"
    )

    return result


def select_positions(
    ranked_candidates: list[dict],
    tier_weight_pct: float,
    position_count: int,
    total_capital: float,
    total_positions: int = 10,
) -> list[dict]:
    """
    Size positions for a tier in a concentrated portfolio.

    Top 3 convictions: 10-15% each
    Next 3-4: 7-10% each
    Remaining: 3-7% each

    Args:
        ranked_candidates: Ranked candidates from rank_candidates().
        tier_weight_pct: Total weight for this tier (e.g. 55.0 for 55%).
        position_count: Number of positions to select for this tier.
        total_capital: Total portfolio capital.
        total_positions: Total positions across all tiers.

    Returns:
        List of position dicts with allocation_pct, dollar_amount, shares, etc.
    """
    if not ranked_candidates or position_count <= 0:
        return []

    # Select top N candidates
    selected = ranked_candidates[:position_count]

    # Size based on rank within tier
    positions = []
    remaining_pct = tier_weight_pct

    for i, cand in enumerate(selected):
        if i < 3:
            # Top 3: larger allocation
            base_pct = min(tier_weight_pct * 0.25, MAX_SINGLE_POSITION_PCT)
        elif i < 6:
            # Next 3: medium allocation
            base_pct = min(tier_weight_pct * 0.15, MAX_SINGLE_POSITION_PCT * 0.7)
        else:
            # Remaining: smaller allocation
            base_pct = min(tier_weight_pct * 0.10, MAX_SINGLE_POSITION_PCT * 0.5)

        # Ensure we don't exceed remaining weight
        allocation_pct = min(base_pct, remaining_pct)
        allocation_pct = max(allocation_pct, 1.0)  # minimum 1%
        remaining_pct -= allocation_pct

        # Dollar amount and shares
        dollar_amount = total_capital * allocation_pct / 100.0
        est_price = SECTOR_PRICE_ESTIMATES.get(cand["sector"], DEFAULT_PRICE)
        shares = max(1, int(dollar_amount / est_price))

        # Conviction level
        if i < 3:
            conviction_level = "HIGH"
        elif i < 6:
            conviction_level = "MEDIUM"
        else:
            conviction_level = "MODERATE"

        # Entry strategy
        if cand.get("conviction", 5) >= 8:
            entry_strategy = "market"
        elif cand.get("conviction", 5) >= 6:
            entry_strategy = "limit_at_pivot"
        else:
            entry_strategy = "pullback_buy"

        positions.append({
            "ticker": cand["ticker"],
            "company_name": cand["company_name"],
            "tier": cand["tier"],
            "allocation_pct": round(allocation_pct, 2),
            "dollar_amount": round(dollar_amount, 2),
            "shares": shares,
            "conviction_level": conviction_level,
            "entry_strategy": entry_strategy,
            "target_entry_price": est_price,
            "composite_score": cand["composite_score"],
            "eps_rating": cand["eps_rating"],
            "rs_rating": cand["rs_rating"],
            "sector": cand["sector"],
            "sector_rank": cand.get("sector_rank", 1),
            "multi_source_count": cand.get("multi_source_count", 0),
            "candidate_score": cand["candidate_score"],
            "estimated_return_pct": cand.get("estimated_return_pct"),
        })

    # Normalize: if remaining_pct > 0, redistribute to top positions
    if remaining_pct > 1.0 and positions:
        extra_per = remaining_pct / min(3, len(positions))
        for j in range(min(3, len(positions))):
            positions[j]["allocation_pct"] = round(
                positions[j]["allocation_pct"] + extra_per, 2,
            )
            positions[j]["dollar_amount"] = round(
                total_capital * positions[j]["allocation_pct"] / 100.0, 2,
            )
            est_price = positions[j]["target_entry_price"]
            positions[j]["shares"] = max(
                1, int(positions[j]["dollar_amount"] / est_price),
            )

    logger.info(
        f"[CandidateRanker] Selected {len(positions)} positions for tier, "
        f"total allocation={sum(p['allocation_pct'] for p in positions):.1f}%"
    )

    return positions


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class CandidateRankerInput(BaseModel):
    tier: int = Field(..., description="Target tier (1, 2, or 3)")
    max_count: int = Field(default=20, description="Max candidates to rank")


class CandidateRankerTool(BaseTool):
    """Rank stocks within a tier for target return portfolio selection."""

    name: str = "candidate_ranker"
    description: str = (
        "Rank and score stock candidates within a specific tier for "
        "target return portfolio construction. Considers composite score, "
        "RS rating, conviction, validation, and sector momentum."
    )
    args_schema: type[BaseModel] = CandidateRankerInput

    def _run(self, tier: int, max_count: int = 20) -> str:
        import json
        return json.dumps({"tier": tier, "max_count": max_count, "note": "Use rank_candidates() directly"})
