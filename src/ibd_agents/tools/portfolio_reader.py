"""
Portfolio Reconciler Tool: Portfolio Reader
IBD Momentum Investment Framework v4.0

Pure functions for:
- Reading real brokerage PDF holdings (E*TRADE, Schwab)
- Building mock current holdings for deterministic pipeline (fallback)

Uses pdfplumber for PDF parsing when available.
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
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


# ---------------------------------------------------------------------------
# Real Brokerage PDF Parsing
# ---------------------------------------------------------------------------

def _clean_numeric(s: str) -> float:
    """Strip $, commas, +/- signs and convert to float."""
    cleaned = s.replace("$", "").replace(",", "").replace("+", "").strip()
    return float(cleaned)


def _extract_account_from_filename(filename: str) -> str:
    """Extract account identifier from brokerage PDF filename."""
    name = Path(filename).stem  # remove .pdf

    if "schwab" in name.lower() or "Schwab" in name:
        return "Schwab"

    # E*TRADE pattern: "E_TRADE Financial — XXXX"
    match = re.search(r'E_TRADE Financial\s*[—–-]\s*(.+)', name)
    if match:
        acct_id = match.group(1).strip()
        return f"E*Trade-{acct_id}"

    # Fallback
    return name[:20]


def _find_etrade_columns(table: list[list]) -> tuple[int, int, int | None, int | None] | None:
    """
    Find Qty, Value, Price Paid, and Total Gain % column indices from E*TRADE header.

    Returns (qty_idx, value_idx, price_paid_idx, gain_pct_idx) or None if header not found.
    """
    for row in table:
        if not row or not row[0]:
            continue
        cell0 = str(row[0]).strip()
        if cell0 != "Symbol":
            continue
        # Found header row — map column names to indices
        qty_idx = None
        val_idx = None
        price_paid_idx = None
        gain_pct_idx = None
        for i, cell in enumerate(row):
            if cell is None:
                continue
            cell_clean = cell.strip().replace("\n", " ")
            if "Qty" in cell_clean:
                qty_idx = i
            elif cell_clean == "Value $" or cell_clean.startswith("Value $"):
                val_idx = i
            elif "Price Paid" in cell_clean:
                price_paid_idx = i
            elif "Total Gain %" in cell_clean:
                gain_pct_idx = i
        if qty_idx is not None and val_idx is not None:
            return (qty_idx, val_idx, price_paid_idx, gain_pct_idx)
    return None


def read_etrade_pdf(pdf_path: Path) -> list[CurrentHolding]:
    """
    Parse an E*TRADE brokerage PDF and extract holdings.

    Uses header-based column mapping to handle varying layouts:
    - Page 1 may have extra None column (Qty at index 5, Value at 10)
    - Page 2+ has standard layout (Qty at index 4, Value at 9)

    Data rows have "SYMBOL Trade" in column 0.
    """
    import pdfplumber

    account = _extract_account_from_filename(pdf_path.name)
    holdings: list[CurrentHolding] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                # Find column mapping from header
                col_map = _find_etrade_columns(table)
                if col_map is None:
                    continue  # No valid header in this table

                qty_idx, val_idx, price_paid_idx, gain_pct_idx = col_map

                for row in table:
                    if not row or not row[0]:
                        continue

                    cell0 = row[0].strip()

                    # Skip non-data rows
                    if "Trade" not in cell0:
                        continue
                    if cell0.startswith("Cash") or cell0.startswith("Total"):
                        continue

                    # Extract symbol: strip "Trade" and any unicode icons
                    symbol = cell0.replace("Trade", "").strip()
                    # Remove unicode characters (alert/flag icons like \ue153)
                    symbol = re.sub(r'[^\w*.-]', '', symbol).strip()
                    if not symbol or not symbol[0].isalpha():
                        continue

                    # Need enough columns for both qty and value
                    if len(row) <= max(qty_idx, val_idx):
                        continue

                    try:
                        qty_str = row[qty_idx]
                        val_str = row[val_idx]
                        if not qty_str or not val_str:
                            continue

                        shares = float(qty_str.replace(",", "").strip())
                        market_value = _clean_numeric(val_str)

                        if shares <= 0 or market_value <= 0:
                            continue

                        # Extract cost basis from Price Paid (per-share)
                        cost_basis = None
                        if price_paid_idx is not None and len(row) > price_paid_idx:
                            try:
                                pp_str = row[price_paid_idx]
                                if pp_str:
                                    price_paid = float(pp_str.replace(",", "").strip())
                                    cost_basis = round(price_paid * shares, 2)
                            except (ValueError, TypeError):
                                pass

                        # Extract gain/loss %
                        gain_loss_pct = None
                        if gain_pct_idx is not None and len(row) > gain_pct_idx:
                            try:
                                gp_str = row[gain_pct_idx]
                                if gp_str:
                                    gain_loss_pct = float(gp_str.replace("%", "").replace("+", "").strip())
                            except (ValueError, TypeError):
                                pass

                        holdings.append(CurrentHolding(
                            symbol=symbol,
                            shares=round(shares, 4),
                            market_value=round(market_value, 2),
                            account=account,
                            sector="UNKNOWN",
                            cost_basis=cost_basis,
                            gain_loss_pct=gain_loss_pct,
                        ))
                    except (ValueError, TypeError, IndexError) as e:
                        logger.debug(f"Skipping row in {pdf_path.name}: {e}")
                        continue

    logger.info(f"[PortfolioReader] {pdf_path.name}: {len(holdings)} positions from {account}")
    return holdings


def read_schwab_pdf(pdf_path: Path) -> list[CurrentHolding]:
    """
    Parse a Charles Schwab brokerage PDF and extract holdings.

    Schwab PDFs use text extraction (table extraction is unreliable).
    Each data line has format:
    SYMBOL DESCRIPTION QTY $PRICE 3 +/-PCT +/-$CHG $COST $MKTVAL 3 ...

    Dollar amounts in order: Price, PriceChg, CostBasis, MktVal, DayChg, GainLoss
    Mkt Val is the 4th dollar amount (index 3).
    Qty is the number immediately before the first $ sign.
    """
    import pdfplumber

    account = _extract_account_from_filename(pdf_path.name)
    holdings: list[CurrentHolding] = []

    # Skip patterns
    skip_prefixes = (
        "Symbol", "Description", "Equities", "ETFs", "Total", "Account Total",
        "Cash & Cash", "Cash & Money", "Positions", "N/A", "Today",
        "Schwab", "New:", "Your positions", "Manage", "Group by",
        "Retirement", "View", "Updated", "(0423",
    )

    in_cash_section = False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Detect Cash & Money Market section — stop parsing
                if "Cash & Money Market" in line:
                    in_cash_section = True
                    continue

                # Reset cash section flag on new data sections
                if line.startswith("Equities") or line.startswith("ETFs"):
                    in_cash_section = False
                    continue

                if in_cash_section:
                    continue

                # Skip headers, totals, disclaimers
                if any(line.startswith(p) for p in skip_prefixes):
                    continue

                # Skip page number lines and disclaimer text
                if line.isdigit() or len(line) < 10:
                    continue

                # Data line must start with a ticker symbol (1-6 uppercase letters)
                sym_match = re.match(r'^([A-Z]{1,6})\s+', line)
                if not sym_match:
                    continue

                symbol = sym_match.group(1)

                # Extract qty: the number (possibly with commas) right before the first $ sign
                qty_match = re.search(r'([\d,]+(?:\.\d+)?)\s+\$', line)
                if not qty_match:
                    continue

                # Extract all dollar amounts from the line
                dollar_amounts = re.findall(r'[+-]?\$([\d,]+\.\d+)', line)
                if len(dollar_amounts) < 4:
                    # Not enough data columns — probably not a data row
                    continue

                try:
                    shares = float(qty_match.group(1).replace(",", ""))
                    # Dollar amounts: [Price, PriceChg, CostBasis, MktVal, ...]
                    cost_basis_total = float(dollar_amounts[2].replace(",", ""))
                    mkt_val = float(dollar_amounts[3].replace(",", ""))

                    if shares <= 0 or mkt_val <= 0:
                        continue

                    # Compute gain/loss % from cost basis and market value
                    gain_loss_pct = None
                    if cost_basis_total > 0:
                        gain_loss_pct = round(
                            (mkt_val - cost_basis_total) / cost_basis_total * 100, 2
                        )

                    holdings.append(CurrentHolding(
                        symbol=symbol,
                        shares=round(shares, 4),
                        market_value=round(mkt_val, 2),
                        account=account,
                        sector="UNKNOWN",
                        cost_basis=round(cost_basis_total, 2),
                        gain_loss_pct=gain_loss_pct,
                    ))
                except (ValueError, IndexError) as e:
                    logger.debug(f"Skipping line in {pdf_path.name}: {e}")
                    continue

    logger.info(f"[PortfolioReader] {pdf_path.name}: {len(holdings)} positions from {account}")
    return holdings


def read_brokerage_pdfs(data_dir: str = "data/portfolios") -> HoldingsSummary:
    """
    Read all brokerage PDF files from data_dir and return aggregated holdings.

    Routes each PDF to the appropriate parser based on filename:
    - Files containing "Schwab" → read_schwab_pdf()
    - Files containing "E_TRADE" → read_etrade_pdf()

    Returns HoldingsSummary with all holdings across all accounts.
    Falls back gracefully if pdfplumber is not installed.
    """
    dir_path = Path(data_dir)
    if not dir_path.exists():
        logger.warning(f"[PortfolioReader] Directory not found: {data_dir}")
        raise FileNotFoundError(f"Portfolio directory not found: {data_dir}")

    pdf_files = sorted(dir_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"[PortfolioReader] No PDF files found in {data_dir}")
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    all_holdings: list[CurrentHolding] = []

    for pdf_file in pdf_files:
        try:
            if "schwab" in pdf_file.name.lower() or "Schwab" in pdf_file.name:
                holdings = read_schwab_pdf(pdf_file)
            elif "e_trade" in pdf_file.name.lower() or "E_TRADE" in pdf_file.name:
                holdings = read_etrade_pdf(pdf_file)
            else:
                logger.warning(f"[PortfolioReader] Unknown brokerage format: {pdf_file.name}")
                continue

            all_holdings.extend(holdings)
        except Exception as e:
            logger.error(f"[PortfolioReader] Failed to parse {pdf_file.name}: {e}")
            continue

    if not all_holdings:
        raise ValueError("No holdings could be parsed from brokerage PDFs")

    total_value = sum(h.market_value for h in all_holdings)
    accounts_used = len(set(h.account for h in all_holdings))

    logger.info(
        f"[PortfolioReader] Total: {len(all_holdings)} positions, "
        f"${total_value:,.0f} across {accounts_used} accounts"
    )

    return HoldingsSummary(
        holdings=all_holdings,
        total_value=round(total_value, 2),
        account_count=accounts_used,
    )


# ---------------------------------------------------------------------------
# Mock Holdings (for tests and when no PDFs available)
# ---------------------------------------------------------------------------

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
        try:
            summary = read_brokerage_pdfs(data_dir)
            return json.dumps({
                "holdings_count": len(summary.holdings),
                "total_value": summary.total_value,
                "account_count": summary.account_count,
                "holdings": [
                    {
                        "symbol": h.symbol,
                        "shares": h.shares,
                        "market_value": h.market_value,
                        "account": h.account,
                    }
                    for h in summary.holdings
                ],
            })
        except Exception as e:
            return json.dumps({"error": str(e)})
