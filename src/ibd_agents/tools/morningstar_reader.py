"""
Research Agent Tool: Morningstar Pick List Reader
IBD Momentum Investment Framework v4.0

Reads Morningstar US Large Cap and Mid Cap Pick List PDFs
using pdfplumber word-level positioning for reliable column extraction.

Extracts: ticker, company name, star rating, economic moat,
fair value, price, price/fair value ratio, industry, and markers.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Column x-position ranges (non-overlapping, using midpoints between observed positions)
# Observed x0: Company≈36, Ticker≈141, Industry≈175, Exchange≈265,
#              MarketCap≈333, Rating≈356, Moat≈404, Uncertainty≈448,
#              FairValue≈501, Price≈536, P/FV≈580
COLUMN_RANGES = {
    "company":     (20, 89),
    "ticker":      (89, 158),
    "industry":    (158, 220),
    "exchange":    (220, 299),
    "market_cap":  (299, 345),
    "rating":      (345, 380),
    "moat":        (380, 426),
    "uncertainty": (426, 475),
    "fair_value":  (475, 519),
    "price":       (519, 558),
    "pfv":         (558, 620),
}

# Sector header names that appear as standalone lines
MORNINGSTAR_SECTORS = {
    "Basic Materials", "Communication Services", "Consumer Cyclical",
    "Consumer Defensive", "Energy", "Financial Services", "Healthcare",
    "Industrials", "Real Estate", "Technology", "Utilities",
}

# Words that are NOT tickers even if they appear in the ticker column position
NON_TICKER_WORDS = {
    "Ticker", "Cap", "Rating", "Moat", "Value", "Price", "Estimate",
    "Data", "as", "of", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
}

# Economic moat values
MOAT_VALUES = {"Wide", "Narrow", "None"}

# Uncertainty values
UNCERTAINTY_VALUES = {"Low", "Medium", "High", "Very"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_list_type(pages) -> str:
    """Detect if this is a Large Cap or Mid Cap pick list from first page text."""
    if not pages:
        return "unknown"
    text = pages[0].extract_text() or ""
    first_500 = text[:500].lower()
    if "large cap" in first_500:
        return "large"
    if "mid cap" in first_500:
        return "mid"
    return "unknown"


def _find_header_y(words: list[dict]) -> Optional[float]:
    """Find the y position of the 'Ticker' header word."""
    for w in words:
        if w["text"] == "Ticker" and w["x0"] > 130 and w["x0"] < 160:
            return w["top"]
    return None


def _group_by_rows(words: list[dict], y_tolerance: float = 4.0) -> list[list[dict]]:
    """Group words into rows by y position."""
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    rows: list[list[dict]] = []
    current_row: list[dict] = [sorted_words[0]]
    current_y = sorted_words[0]["top"]

    for w in sorted_words[1:]:
        if abs(w["top"] - current_y) <= y_tolerance:
            current_row.append(w)
        else:
            rows.append(sorted(current_row, key=lambda x: x["x0"]))
            current_row = [w]
            current_y = w["top"]
    if current_row:
        rows.append(sorted(current_row, key=lambda x: x["x0"]))
    return rows


def _assign_column(x0: float) -> Optional[str]:
    """Assign a word to a column based on its x position."""
    for col, (lo, hi) in COLUMN_RANGES.items():
        if lo <= x0 < hi:
            return col
    return None


def _is_sector_header(row_words: list[dict]) -> Optional[str]:
    """Check if a row is a sector header. Returns sector name or None."""
    text = " ".join(w["text"] for w in row_words).strip()
    if text in MORNINGSTAR_SECTORS:
        return text
    return None


def _parse_q_rating(text: str) -> int:
    """Convert Q-rating text to star count."""
    qs = text.strip()
    if re.match(r"^Q{1,5}$", qs):
        return len(qs)
    return 0


def _clean_company_name(words: list[dict]) -> tuple[str, Optional[str]]:
    """
    Extract company name and marker from company-column words.

    Returns (clean_company_name, marker) where marker is 'E', '[', ']', or None.
    """
    if not words:
        return ("", None)

    first = words[0]
    marker = None
    text_parts = [w["text"] for w in words]

    # Check for [/] markers — may be separate char or embedded in first word
    if first["text"] in ("[", "]") and first["x0"] < 35:
        marker = first["text"]
        text_parts = text_parts[1:]  # Drop the standalone marker
    elif first["text"][0] in ("[", "]") and len(first["text"]) > 1:
        marker = first["text"][0]
        text_parts[0] = first["text"][1:]  # Strip embedded marker
    # Check for E-addition marker (word starts with E at x < 33, E is prefixed)
    elif first["x0"] < 33 and first["text"].startswith("E") and len(first["text"]) > 1:
        char_after_e = first["text"][1]
        if char_after_e.isupper():
            marker = "E"
            text_parts[0] = first["text"][1:]  # Strip leading E

    company = " ".join(text_parts).strip()
    return (company, marker)


def _safe_float(text: str) -> Optional[float]:
    """Convert text to float, handling commas."""
    try:
        return float(text.replace(",", ""))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main Parser
# ---------------------------------------------------------------------------

def _parse_page_picks(page, header_y: float, cap_list: str, is_deletion: bool = False) -> list[dict]:
    """Parse stock picks from a single page using word positions."""
    words = page.extract_words()
    if not words:
        return []

    # Only consider words below the header row and above the footer
    data_words = [w for w in words if w["top"] > header_y + 8 and w["top"] < 740]
    if not data_words:
        return []

    rows = _group_by_rows(data_words)
    picks: list[dict] = []
    current_sector = None

    for row in rows:
        # Check for sector header
        sector = _is_sector_header(row)
        if sector is not None:
            current_sector = sector
            continue

        # Check for "Pick List Deletions" header
        row_text = " ".join(w["text"] for w in row)
        if "Pick List Deletions" in row_text:
            # Signal that subsequent rows are deletions
            return picks  # Return current picks; deletions parsed separately

        # Skip non-data rows (footer text, commentary, etc.)
        # A valid data row must have a ticker-column word
        ticker_words = [w for w in row if 135 <= w["x0"] < 175
                        and w["text"] not in NON_TICKER_WORDS
                        and re.match(r"^[A-Z][A-Z0-9.]{0,5}$", w["text"])]
        if not ticker_words:
            continue

        ticker = ticker_words[0]["text"]

        # Group remaining words by column
        columns: dict[str, list[dict]] = {}
        for w in row:
            col = _assign_column(w["x0"])
            if col:
                columns.setdefault(col, []).append(w)

        # Extract company name with marker detection
        company_words = columns.get("company", [])
        company_name, marker = _clean_company_name(company_words)

        # Extract industry
        industry_words = columns.get("industry", [])
        industry = " ".join(w["text"] for w in industry_words).strip() if industry_words else ""

        # Extract star rating (Q count)
        rating_words = columns.get("rating", [])
        star_count = 0
        if rating_words:
            star_count = _parse_q_rating(rating_words[0]["text"])

        # Extract economic moat
        moat_words = columns.get("moat", [])
        economic_moat = moat_words[0]["text"] if moat_words else ""

        # Extract uncertainty
        unc_words = columns.get("uncertainty", [])
        uncertainty = " ".join(w["text"] for w in unc_words).strip() if unc_words else ""

        # Extract fair value
        fv_words = columns.get("fair_value", [])
        fair_value = _safe_float(fv_words[0]["text"]) if fv_words else None

        # Extract price
        price_words = columns.get("price", [])
        price = _safe_float(price_words[0]["text"]) if price_words else None

        # Extract price/fair value
        pfv_words = columns.get("pfv", [])
        price_to_fair_value = _safe_float(pfv_words[0]["text"]) if pfv_words else None

        # Extract market cap
        mc_words = columns.get("market_cap", [])
        market_cap = None
        if mc_words:
            try:
                market_cap = int(mc_words[0]["text"].replace(",", ""))
            except (ValueError, TypeError):
                pass

        # Build star rating string
        morningstar_rating = f"{star_count}-star" if star_count > 0 else ""

        picks.append({
            "symbol": ticker,
            "company_name": company_name,
            "morningstar_rating": morningstar_rating,
            "star_count": star_count,
            "economic_moat": economic_moat,
            "fair_value": fair_value,
            "price": price,
            "price_to_fair_value": price_to_fair_value,
            "market_cap_mil": market_cap,
            "industry": industry,
            "uncertainty": uncertainty,
            "sector": current_sector,
            "cap_list": cap_list,
            "marker": marker,
            "is_deletion": is_deletion,
            "source_file": "",  # Set by caller
        })

    return picks


def read_morningstar_pdf(file_path: str) -> list[dict]:
    """
    Read a Morningstar US Large Cap or Mid Cap Pick List PDF.

    Returns list of dicts with keys:
        symbol, company_name, morningstar_rating, star_count,
        economic_moat, fair_value, price, price_to_fair_value,
        market_cap_mil, industry, uncertainty, sector, cap_list,
        marker, is_deletion, source_file

    Args:
        file_path: Path to the Morningstar Pick List PDF

    Returns:
        List of stock pick records
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber is required: pip install pdfplumber")
        return []

    filename = os.path.basename(file_path)
    logger.info(f"Reading Morningstar PDF: {filename}")

    try:
        pdf = pdfplumber.open(file_path)
    except Exception as e:
        logger.error(f"Failed to open {filename}: {e}")
        return []

    cap_list = _detect_list_type(pdf.pages)
    all_picks: list[dict] = []
    in_deletions = False

    for page_idx, page in enumerate(pdf.pages):
        words = page.extract_words()
        if not words:
            continue

        # Check if this page contains "Pick List Deletions"
        page_text = " ".join(w["text"] for w in words)
        if "Pick List Deletions" in page_text:
            in_deletions = True

        # Find header row on this page
        header_y = _find_header_y(words)
        if header_y is None:
            continue  # Skip pages without data tables (commentary, methodology)

        picks = _parse_page_picks(page, header_y, cap_list, is_deletion=in_deletions)
        for p in picks:
            p["source_file"] = filename
        all_picks.extend(picks)

    pdf.close()

    # Deduplicate by symbol (keep first occurrence)
    seen = set()
    unique_picks = []
    for p in all_picks:
        key = (p["symbol"], p["is_deletion"])
        if key not in seen:
            seen.add(key)
            unique_picks.append(p)

    logger.info(
        f"[Morningstar] {filename}: {len(unique_picks)} picks extracted "
        f"({cap_list} cap list)"
    )
    return unique_picks


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class ReadMorningstarPDFInput(BaseModel):
    """Input for ReadMorningstarPDFTool."""
    file_path: str = Field(..., description="Path to Morningstar Pick List PDF")


class ReadMorningstarPDFTool(BaseTool):
    """Read Morningstar US Large/Mid Cap Pick List PDFs."""

    name: str = "read_morningstar_pdf"
    description: str = (
        "Read a Morningstar US Large Cap or Mid Cap Pick List PDF. "
        "Extracts stock picks with star ratings, economic moat, "
        "fair value estimates, and price/fair value ratios."
    )
    args_schema: type[BaseModel] = ReadMorningstarPDFInput

    def _run(self, file_path: str) -> str:
        import json
        records = read_morningstar_pdf(file_path)
        if not records:
            return "No picks extracted from Morningstar PDF."
        # Return summary + first few records
        lines = [f"Extracted {len(records)} Morningstar picks:"]
        for r in records[:10]:
            lines.append(
                f"  {r['symbol']:6s} {r['star_count']}★ "
                f"Moat={r['economic_moat']:6s} "
                f"FV={r.get('fair_value', 'N/A')} "
                f"P/FV={r.get('price_to_fair_value', 'N/A')}"
            )
        if len(records) > 10:
            lines.append(f"  ... and {len(records) - 10} more")
        return "\n".join(lines)
