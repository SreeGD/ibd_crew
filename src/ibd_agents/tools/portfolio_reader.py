"""
Portfolio Reconciler Tool: Portfolio Reader
IBD Momentum Investment Framework v4.0

Pure functions for:
- Building mock current holdings for deterministic pipeline
- (Agentic mode would parse brokerage PDFs)

No LLM, no file I/O in deterministic mode.
"""

from __future__ import annotations

import logging
import random
from typing import List

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

from ibd_agents.schemas.analyst_output import RatedStock
from ibd_agents.schemas.portfolio_output import ALL_KEEPS, KEEP_METADATA
from ibd_agents.schemas.reconciliation_output import (
    CurrentHolding,
    HoldingsSummary,
)

logger = logging.getLogger(__name__)

# Use fixed seed for deterministic mock data
_RNG = random.Random(42)

# Total portfolio value for mock holdings
MOCK_PORTFOLIO_VALUE = 1_520_000.0

# Accounts
MOCK_ACCOUNTS = ["Schwab", "E*Trade-1157", "E*Trade-6270"]


def build_mock_current_holdings(
    rated_stocks: List[RatedStock],
) -> HoldingsSummary:
    """
    Build deterministic mock current holdings.

    Creates a realistic mock portfolio:
    - All 14 keeps are held (pre-committed)
    - ~15 non-recommended stocks (to be sold)
    - Random allocation across 3 accounts
    """
    holdings: list[CurrentHolding] = []

    # 1. Add all keeps (currently held)
    keep_value_total = 0.0
    for sym in ALL_KEEPS:
        meta = KEEP_METADATA[sym]
        # Assign to random account
        account = MOCK_ACCOUNTS[_RNG.randint(0, 2)]
        value = _RNG.uniform(10000, 45000)
        shares = round(value / _RNG.uniform(50, 500), 2)

        # Find sector from rated stocks
        sector = "UNKNOWN"
        for rs in rated_stocks:
            if rs.symbol == sym:
                sector = rs.sector
                break

        holdings.append(CurrentHolding(
            symbol=sym,
            shares=shares,
            market_value=round(value, 2),
            account=account,
            sector=sector,
        ))
        keep_value_total += value

    # 2. Add some rated stocks (overlap with recommended)
    overlap_count = min(15, len(rated_stocks))
    for rs in rated_stocks[:overlap_count]:
        if rs.symbol in ALL_KEEPS:
            continue
        account = MOCK_ACCOUNTS[_RNG.randint(0, 2)]
        value = _RNG.uniform(5000, 30000)
        shares = round(value / _RNG.uniform(30, 400), 2)

        holdings.append(CurrentHolding(
            symbol=rs.symbol,
            shares=shares,
            market_value=round(value, 2),
            account=account,
            sector=rs.sector,
        ))

    # 3. Add non-recommended stocks (to be sold)
    non_rec_stocks = [
        ("CAT", "MACHINERY"), ("IDXX", "MEDICAL"), ("AMD", "CHIPS"),
        ("VEA", "INTERNATIONAL"), ("ICLN", "ENERGY"), ("ACAAX", "MUTUAL_FUND"),
        ("FBGRX", "MUTUAL_FUND"), ("PFE", "MEDICAL"), ("DIS", "CONSUMER"),
        ("NFLX", "INTERNET"), ("BA", "AEROSPACE"), ("WMT", "RETAIL"),
        ("KO", "CONSUMER"), ("PG", "CONSUMER"), ("HD", "RETAIL"),
    ]
    for sym, sector in non_rec_stocks:
        account = MOCK_ACCOUNTS[_RNG.randint(0, 2)]
        value = _RNG.uniform(8000, 35000)
        shares = round(value / _RNG.uniform(30, 500), 2)

        holdings.append(CurrentHolding(
            symbol=sym,
            shares=shares,
            market_value=round(value, 2),
            account=account,
            sector=sector,
        ))

    total_value = sum(h.market_value for h in holdings)
    accounts_used = len(set(h.account for h in holdings))

    return HoldingsSummary(
        holdings=holdings,
        total_value=round(total_value, 2),
        account_count=accounts_used,
    )


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class PortfolioReaderInput(BaseModel):
    data_dir: str = Field("data/portfolios", description="Directory with brokerage files")


class PortfolioReaderTool(BaseTool):
    """Read current portfolio holdings from brokerage files."""

    name: str = "portfolio_reader"
    description: str = (
        "Read current portfolio holdings from brokerage PDF/CSV files "
        "and return a HoldingsSummary"
    )
    args_schema: type[BaseModel] = PortfolioReaderInput

    def _run(self, data_dir: str = "data/portfolios") -> str:
        import json
        # In agentic mode, this would parse actual PDFs
        # For now, return a note that actual file reading is needed
        return json.dumps({
            "note": "Agentic mode required for PDF parsing. Use deterministic mock for testing.",
        })
