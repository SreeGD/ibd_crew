"""
Research Agent Tool: PDF Reader
Extract stocks and ratings from IBD PDFs and Motley Fool PDFs.
Uses pdfplumber for table and text extraction.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

from ibd_agents.schemas.research_output import IBD_SECTORS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Smart Table PDF parsing — sector-aware extraction
# ---------------------------------------------------------------------------

# Sector header regex: "1. MINING -2.4% Daily Change, +29.15% Since Jan. 1"
SECTOR_HEADER_RE = re.compile(
    r'^(\d+)\.\s+([A-Z][A-Z\s/&]+?)\s+([+-]?\d+\.?\d*%)'
)

# Map abbreviated Smart Table sector names → canonical IBD_SECTORS
SMART_TABLE_SECTOR_MAP: dict[str, str] = {
    "MINING": "MINING",
    "COMPUTER": "COMPUTER",
    "CHIPS": "CHIPS",
    "ELECTRNCS": "ELECTRONICS",
    "ELECTRONICS": "ELECTRONICS",
    "METALS": "MINING",
    "AEROSPACE": "AEROSPACE",
    "TELECOM": "TELECOM",
    "AUTO": "AUTO",
    "BUILDING": "BUILDING",
    "MACHINE": "MACHINERY",
    "MACHINERY": "MACHINERY",
    "INTERNET": "INTERNET",
    "ENERGY": "ENERGY",
    "REAL EST": "REAL ESTATE",
    "REAL ESTATE": "REAL ESTATE",
    "ALCOHL/TOB": "CONSUMER",
    "OFFICE": "BUSINESS SERVICES",
    "MEDICAL": "MEDICAL",
    "TRANSPRT": "TRANSPORTATION",
    "TRANSPORTATION": "TRANSPORTATION",
    "BANKS": "BANKS",
    "APPAREL": "CONSUMER",
    "AGRICULTRE": "FOOD/BEVERAGE",
    "AGRICULTURE": "FOOD/BEVERAGE",
    "BUSINS SVC": "BUSINESS SERVICES",
    "BUSINESS SVC": "BUSINESS SERVICES",
    "BUSINESS SERVICES": "BUSINESS SERVICES",
    "FINANCE": "FINANCE",
    "MEDIA": "MEDIA",
    "CHEMICAL": "CHEMICALS",
    "CHEMICALS": "CHEMICALS",
    "RETAIL": "RETAIL",
    "LEISURE": "LEISURE",
    "UTILITY": "UTILITIES",
    "UTILITIES": "UTILITIES",
    "FOOD/BEV": "FOOD/BEVERAGE",
    "FOOD/BEVERAGE": "FOOD/BEVERAGE",
    "INSURNCE": "INSURANCE",
    "INSURANCE": "INSURANCE",
    "MISC": "BUSINESS SERVICES",
    "CONSUMER": "CONSUMER",
    "SOFTWARE": "SOFTWARE",
    "DEFENSE": "DEFENSE",
    "HEALTHCARE": "HEALTHCARE",
}

# Stock data line: starts with 2-3 digit ratings then letter ratings
# e.g. "99 97 93 A C 45.2 Alamos Gold 0.2 r AGI 42.12 -2.31 ..."
STOCK_LINE_RE = re.compile(
    r'^(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+([A-E][\.\-\+]?)\s+([A-E][\.\-\+]?)\s+'
)

# Ticker inside a stock line: 1-5 uppercase letters followed by a price
TICKER_IN_LINE_RE = re.compile(r'\b([A-Z]{1,5})\s+(\d{1,4}\.\d)')

# Lines to skip
SKIP_PATTERNS = (
    "Key Data:", "Trend Lists", "SECTOR LEADER", "Cmp EPS Rel",
    "Rtg Rtg Str", "Market Stock Research", "Store",
)


def _normalize_sector(raw: str) -> str:
    """Map a raw Smart Table sector name to a canonical IBD_SECTORS value."""
    name = raw.strip().upper()
    if name in IBD_SECTORS:
        return name
    if name in SMART_TABLE_SECTOR_MAP:
        return SMART_TABLE_SECTOR_MAP[name]
    # Fuzzy fallback: first word match
    first_word = name.split()[0] if name else ""
    for sector in IBD_SECTORS:
        if sector.startswith(first_word) and len(first_word) >= 4:
            return sector
    return "UNKNOWN"


def _is_smart_table(filename: str) -> bool:
    """Detect if a PDF is an IBD Smart Table by filename."""
    lower = filename.lower()
    return "smart" in lower and ("nyse" in lower or "nasdaq" in lower or "table" in lower)


def _safe_rating_pdf(val: str) -> Optional[str]:
    """Convert a Smart Table letter rating to schema-valid value."""
    s = val.strip().upper()
    valid = {"A", "B", "B-", "C", "D", "E"}
    if s in valid:
        return s
    if len(s) == 2 and s[1] in ("+", "-"):
        base = s[0]
        if base in ("A", "B", "C", "D", "E"):
            return "B-" if s == "B-" else base
    return None


def _parse_smart_table_text(pages: list) -> list[dict]:
    """Parse IBD Smart Table PDF pages, extracting sector + ratings per stock.

    The Smart Table format organises stocks under numbered sector headers.
    Text extraction works; table extraction does not for this format.
    """
    results: list[dict] = []
    current_sector: str = "UNKNOWN"
    current_sector_rank: Optional[int] = None
    stock_position: int = 0  # counter within current sector
    seen: set[str] = set()

    for page in pages:
        text = page.extract_text()
        if not text:
            continue

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            # Skip noise lines
            if any(stripped.startswith(p) or p in stripped for p in SKIP_PATTERNS):
                # But don't skip if it also matches a stock line
                if not STOCK_LINE_RE.match(stripped):
                    continue

            # Check for sector header
            m_sector = SECTOR_HEADER_RE.match(stripped)
            if m_sector:
                current_sector_rank = int(m_sector.group(1))
                current_sector = _normalize_sector(m_sector.group(2))
                stock_position = 0
                continue

            # Try to parse as stock data line
            m_stock = STOCK_LINE_RE.match(stripped)
            if not m_stock:
                continue

            comp = int(m_stock.group(1))
            eps = int(m_stock.group(2))
            rs = int(m_stock.group(3))
            smr = _safe_rating_pdf(m_stock.group(4))
            acc_dis = _safe_rating_pdf(m_stock.group(5))

            # Extract ticker from the remainder of the line
            remainder = stripped[m_stock.end():]
            m_ticker = TICKER_IN_LINE_RE.search(remainder)
            if not m_ticker:
                continue

            ticker = m_ticker.group(1)
            # Skip common false positives
            if ticker in ("NH", "PE", "PM", "LQ", "RS", "NM", "DIS"):
                # Try the next ticker match
                for m2 in TICKER_IN_LINE_RE.finditer(remainder, m_ticker.end()):
                    if m2.group(1) not in ("NH", "PE", "PM", "LQ", "RS", "NM", "DIS"):
                        ticker = m2.group(1)
                        break
                else:
                    continue

            if ticker in seen:
                continue
            seen.add(ticker)

            # Extract company name: text between ratings and ticker
            pre_ticker = remainder[:remainder.find(ticker)].strip()
            # Remove numeric indicators (like "45.2", "NH", "r") to get company name
            name_parts = []
            for word in pre_ticker.split():
                if word in ("r", "NH", "NM"):
                    continue
                try:
                    float(word)
                    continue
                except ValueError:
                    name_parts.append(word)
            company = " ".join(name_parts) if name_parts else ticker

            stock_position += 1
            results.append({
                "symbol": ticker,
                "company_name": company,
                "composite_rating": comp if 1 <= comp <= 99 else None,
                "rs_rating": rs if 1 <= rs <= 99 else None,
                "eps_rating": eps if 1 <= eps <= 99 else None,
                "smr_rating": smr,
                "acc_dis_rating": acc_dis,
                "sector": current_sector,
                "sector_rank": current_sector_rank,
                "stock_rank_in_sector": stock_position,
            })

    logger.info(f"Smart Table: parsed {len(results)} stocks across {len({r['sector'] for r in results})} sectors")
    return results


# ---------------------------------------------------------------------------
# Generic PDF parsing (non-Smart-Table)
# ---------------------------------------------------------------------------

# Map PDF filename patterns → IBD list name
PDF_TO_LIST_NAME: dict[str, str] = {
    "Smart_NYSE": "IBD Smart Table",
    "Smart_Table": "IBD Smart Table",
    "Tech_Leaders": "IBD Tech Leaders",
    "ETF_Leaders": "IBD ETF Leaders",
    "Top_Sector_ETFs": "IBD ETF Leaders",
    "ETF_Tables": "IBD ETF Leaders",
    "Top_200_Composite": "IBD Top 200 Composite",
    "Large_MidCap": "IBD Large/MidCap Leaders",
    "TopRanked_LowPriced": "IBD Smart Table",
    "Most_Active": "IBD Smart Table",
    "TimeSaver": "IBD TimeSaver Table",
    "Rankings_Epic": "Motley Fool Epic Top",
    "New_Recs": "Motley Fool New Rec",
}

# Regex to detect stock symbols in text
# Matches 1-5 uppercase letters that look like tickers
TICKER_PATTERN = re.compile(r'\b([A-Z]{1,5})\b')

# Known non-ticker words to filter out
NON_TICKERS: set[str] = {
    "THE", "AND", "FOR", "NOT", "ARE", "BUT", "ALL", "ANY",
    "CAN", "HAS", "HER", "WAS", "ONE", "OUR", "OUT", "DAY",
    "HAD", "HOT", "OIL", "SIT", "TOP", "TWO", "WAR", "WHO",
    "BOY", "DID", "ITS", "LET", "PUT", "SAY", "SHE", "TOO",
    "USE", "NEW", "NOW", "OLD", "SEE", "WAY", "MAY", "IBD",
    "NYSE", "ETF", "IPO", "PDF", "USA", "CEO", "CFO", "GDP",
    "EPS", "SMR", "YTD", "QTD", "YOY", "MOM", "EST", "AVG",
    "HIGH", "LOW", "VOL", "COMP", "STOCK", "PRICE", "NAME",
    "DATE", "PAGE", "TABLE", "RATE", "FROM", "THAN", "WITH",
    "THIS", "THAT", "HAVE", "BEEN", "WILL", "EACH", "MAKE",
    "LIKE", "LONG", "LOOK", "MANY", "SOME", "TIME", "VERY",
    "WHEN", "COME", "MADE", "FIND", "BACK", "ONLY", "ALSO",
    "AFTER", "YEAR", "GIVE", "MOST", "JUST", "OVER", "SUCH",
    "TAKE", "INTO", "THAN", "THEM", "SAME", "DOWN", "SHOULD",
    "THESE", "THEIR", "ABOUT", "WOULD", "COULD", "OTHER",
    "WHICH", "THERE", "BEING", "STILL", "WHERE", "THOSE",
    "FIRST", "EVERY", "LARGE", "SMALL", "RIGHT", "MIGHT",
    "WHILE", "SINCE", "UNDER", "ALONG", "CLOSE", "ABOVE",
    "BELOW", "TOTAL", "DAILY", "INDEX",
    # Motley Fool service codes and action words
    "BUY", "SELL", "HOLD", "RB", "SA", "HG", "DI",
}


def _detect_list_name(filename: str) -> str:
    """Determine source list name from PDF filename."""
    fn = filename.lower().replace("_", " ")
    for pattern, list_name in PDF_TO_LIST_NAME.items():
        if pattern.lower().replace("_", " ") in fn:
            return list_name
    return f"IBD {filename}"


def _safe_int(val: Any) -> Optional[int]:
    """Convert to int rating (1-99), or None."""
    if val is None:
        return None
    try:
        v = int(float(str(val).strip()))
        return v if 1 <= v <= 99 else None
    except (ValueError, TypeError):
        return None


def _safe_rating(val: Any) -> Optional[str]:
    """Convert to letter rating (A/B/B-/C/D/E), or None."""
    if val is None:
        return None
    s = str(val).strip().upper()
    valid = {"A", "B", "B-", "C", "D", "E"}
    return s if s in valid else None


def _is_likely_ticker(word: str) -> bool:
    """Check if a word looks like a stock ticker."""
    if word in NON_TICKERS:
        return False
    if not word.isalpha():
        return False
    if len(word) < 1 or len(word) > 5:
        return False
    return word.isupper()


def _extract_from_tables(pages: list) -> list[dict]:
    """Extract structured data from PDF tables."""
    results = []

    for page in pages:
        tables = page.extract_tables()
        for table in tables:
            if not table or len(table) < 2:
                continue

            # First row is header
            header = [str(h).strip() if h else "" for h in table[0]]
            header_lower = [h.lower() for h in header]

            # Find column indices
            sym_idx = None
            name_idx = None
            comp_idx = None
            rs_idx = None
            eps_idx = None
            smr_idx = None
            ad_idx = None

            for i, h in enumerate(header_lower):
                if "sym" in h or "ticker" in h:
                    sym_idx = i
                elif "company" in h or "name" in h:
                    name_idx = i
                elif "comp" in h and "rating" in h or h == "comp":
                    comp_idx = i
                elif h in ("rs", "rel str", "relative strength", "rs rating"):
                    rs_idx = i
                elif h in ("eps", "eps rating", "eps rtg", "earnings"):
                    eps_idx = i
                elif h in ("smr", "smr rating"):
                    smr_idx = i
                elif "acc" in h or "a/d" in h:
                    ad_idx = i

            if sym_idx is None:
                # Try to detect symbol column by content
                for i in range(min(len(header), 5)):
                    if len(table) > 1 and table[1][i]:
                        val = str(table[1][i]).strip()
                        if _is_likely_ticker(val):
                            sym_idx = i
                            break

            if sym_idx is None:
                continue

            # Process data rows
            for row in table[1:]:
                if not row or len(row) <= sym_idx:
                    continue
                sym = str(row[sym_idx]).strip().upper() if row[sym_idx] else None
                if not sym or not _is_likely_ticker(sym):
                    continue

                stock = {
                    "symbol": sym,
                    "company_name": str(row[name_idx]).strip() if name_idx and name_idx < len(row) and row[name_idx] else sym,
                    "composite_rating": _safe_int(row[comp_idx]) if comp_idx and comp_idx < len(row) else None,
                    "rs_rating": _safe_int(row[rs_idx]) if rs_idx and rs_idx < len(row) else None,
                    "eps_rating": _safe_int(row[eps_idx]) if eps_idx and eps_idx < len(row) else None,
                    "smr_rating": _safe_rating(row[smr_idx]) if smr_idx and smr_idx < len(row) else None,
                    "acc_dis_rating": _safe_rating(row[ad_idx]) if ad_idx and ad_idx < len(row) else None,
                }
                results.append(stock)

    return results


def _extract_from_text(pages: list) -> list[str]:
    """
    Fallback: extract ticker symbols from raw text when tables fail.
    Returns list of unique ticker symbols found.
    """
    tickers: set[str] = set()
    for page in pages:
        text = page.extract_text()
        if not text:
            continue
        words = TICKER_PATTERN.findall(text)
        for w in words:
            if _is_likely_ticker(w):
                tickers.add(w)
    return sorted(tickers)


def _extract_fool_companies(pages: list) -> list[dict]:
    """
    Extract company names from Motley Fool tables where the Symbol column is empty.

    Handles the "New Recs / Epic" format where headers include "Company" and "Action"
    but ticker symbols are missing from the table cells.

    Returns list of dicts with keys: company_name, action, rec_date
    """
    results: list[dict] = []

    for page in pages:
        tables = page.extract_tables()
        for table in tables:
            if not table or len(table) < 2:
                continue

            header = [str(h).strip() if h else "" for h in table[0]]
            header_lower = [h.lower() for h in header]

            # Identify Fool-format tables: must have "company" column
            company_idx = None
            action_idx = None
            date_idx = None

            for i, h in enumerate(header_lower):
                if "company" in h or "name" in h:
                    company_idx = i
                elif h == "action":
                    action_idx = i
                elif "rec date" in h or "date" in h:
                    date_idx = i

            if company_idx is None:
                continue

            for row in table[1:]:
                if not row or len(row) <= company_idx:
                    continue

                name = str(row[company_idx]).strip() if row[company_idx] else ""
                if not name or len(name) < 2:
                    continue

                action = ""
                if action_idx is not None and action_idx < len(row) and row[action_idx]:
                    action = str(row[action_idx]).strip().upper()

                rec_date = ""
                if date_idx is not None and date_idx < len(row) and row[date_idx]:
                    rec_date = str(row[date_idx]).strip()

                results.append({
                    "company_name": name,
                    "action": action,
                    "rec_date": rec_date,
                })

    return results


# Regex for ETF Tables data lines:
# <ytd_chg> <rs_rating> <acc_dis_rating> <52wk_high> <fund_name...> <SYMBOL> <div_yld> <close> <chg> <vol_chg>
# Example: "13.0 81 A- 36.7 Dim US Small Cap Value DFSV 1.3 37.17 1.00 -19"
_ETF_LINE_RE = re.compile(
    r'^([+-]?\d+\.?\d*)\s+'     # 1: YTD change
    r'(\d{1,2})\s+'             # 2: RS rating (1-99)
    r'([A-E][+-]?)\s+'          # 3: Acc/Dis rating
    r'(\d+\.?\d*)\s+'           # 4: 52-week high %
    r'(.+?)\s+'                 # 5: Fund name (non-greedy)
    r'([A-Z]{2,5})\s+'          # 6: Symbol (2-5 uppercase letters)
    r'(\d+\.?\d*|\.\.)\s+'      # 7: Div yield (number or "..")
    r'(\d+\.?\d*)\s+'           # 8: Close price
    r'([+-]?\d+\.?\d*)\s+'      # 9: Price change
    r'([+-]?\d+)'               # 10: Volume % change
)


def _normalize_acc_dis(raw: str) -> str | None:
    """Normalize Acc/Dis rating to valid values: A, B, B-, C, D, E."""
    base = raw[0]  # A, B, C, D, E
    suffix = raw[1:] if len(raw) > 1 else ""
    # Only B- is a valid combo with suffix; all others map to base letter
    if base == "B" and suffix == "-":
        return "B-"
    return _safe_rating(base)


def _parse_etf_tables(pages: list) -> list[dict]:
    """Parse IBD ETF Tables PDF text into structured ETF records."""
    results: list[dict] = []
    seen: set[str] = set()

    for page in pages:
        text = page.extract_text()
        if not text:
            continue
        for line in text.split('\n'):
            line = line.strip()
            m = _ETF_LINE_RE.match(line)
            if not m:
                continue
            ytd_change = float(m.group(1))
            rs_rating = int(m.group(2))
            acc_dis = m.group(3)
            # group(4) = 52-week high %
            fund_name = m.group(5).strip()
            symbol = m.group(6)
            div_yield_raw = m.group(7)
            close_price = float(m.group(8))
            price_change = float(m.group(9))
            volume_pct_change = int(m.group(10))

            div_yield = float(div_yield_raw) if div_yield_raw != ".." else None

            if symbol in seen or symbol in NON_TICKERS:
                continue
            seen.add(symbol)

            results.append({
                "symbol": symbol,
                "company_name": fund_name,
                "composite_rating": None,
                "rs_rating": rs_rating if 1 <= rs_rating <= 99 else None,
                "eps_rating": None,
                "smr_rating": None,
                "acc_dis_rating": _normalize_acc_dis(acc_dis),
                "ytd_change": ytd_change,
                "close_price": close_price,
                "price_change": price_change,
                "volume_pct_change": volume_pct_change,
                "div_yield": div_yield,
            })

    return results


def read_ibd_pdf(file_path: str) -> list[dict]:
    """
    Read an IBD PDF file and extract stocks with available ratings.

    Tries table extraction first, falls back to text extraction for symbols only.

    Returns:
        List of dicts with keys: symbol, company_name, composite_rating,
        rs_rating, eps_rating, smr_rating, acc_dis_rating, ibd_list, source_file
    """
    import pdfplumber

    path = Path(file_path)
    if not path.exists():
        logger.warning(f"File not found: {file_path}")
        return []

    list_name = _detect_list_name(path.name)
    is_etf_list = "ETF" in list_name
    results: list[dict] = []

    try:
        with pdfplumber.open(file_path) as pdf:
            if _is_smart_table(path.name):
                # Smart Table: dedicated sector-aware parser
                smart_results = _parse_smart_table_text(pdf.pages)
                for stock in smart_results:
                    stock["ibd_list"] = list_name
                    stock["source_file"] = path.name
                    if is_etf_list:
                        stock["is_etf"] = True
                    results.append(stock)
            elif is_etf_list:
                # ETF Tables: dedicated line-by-line parser
                etf_results = _parse_etf_tables(pdf.pages)
                for etf in etf_results:
                    etf["ibd_list"] = list_name
                    etf["source_file"] = path.name
                    etf["is_etf"] = True
                    results.append(etf)
            else:
                # Other PDFs: try table extraction, then text fallback
                table_results = _extract_from_tables(pdf.pages)

                if table_results:
                    for stock in table_results:
                        stock["ibd_list"] = list_name
                        stock["source_file"] = path.name
                        results.append(stock)
                else:
                    tickers = _extract_from_text(pdf.pages)
                    for sym in tickers:
                        results.append({
                            "symbol": sym,
                            "company_name": sym,
                            "composite_rating": None,
                            "rs_rating": None,
                            "eps_rating": None,
                            "smr_rating": None,
                            "acc_dis_rating": None,
                            "ibd_list": list_name,
                            "source_file": path.name,
                        })

        logger.info(f"Read {len(results)} stocks from {path.name} ({list_name})")

    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")

    return results


def read_fool_pdf(file_path: str) -> list[dict]:
    """
    Read a Motley Fool PDF and extract stock rankings/recommendations.

    Returns:
        List of dicts with keys: symbol, company_name, fool_status, source_file
    """
    import pdfplumber

    path = Path(file_path)
    if not path.exists():
        logger.warning(f"File not found: {file_path}")
        return []

    list_name = _detect_list_name(path.name)
    # Determine status from filename
    name_lower = path.name.lower()
    if "new rec" in name_lower or "new_rec" in name_lower:
        fool_status = "New Rec"
    elif "ranking" in name_lower or "epic" in name_lower:
        fool_status = "Epic Top"
    else:
        fool_status = "Epic Top"

    results: list[dict] = []

    try:
        with pdfplumber.open(file_path) as pdf:
            # Try tables first
            table_results = _extract_from_tables(pdf.pages)

            if table_results:
                for stock in table_results:
                    stock["fool_status"] = fool_status
                    stock["source_file"] = path.name
                    results.append(stock)
            else:
                # Fool-specific: extract company names (symbol column is empty)
                fool_companies = _extract_fool_companies(pdf.pages)
                if fool_companies:
                    for rec in fool_companies:
                        results.append({
                            "symbol": "",  # resolved later via universe
                            "company_name": rec["company_name"],
                            "fool_status": fool_status,
                            "source_file": path.name,
                        })
                else:
                    # Last resort: text extraction
                    tickers = _extract_from_text(pdf.pages)
                    for sym in tickers:
                        results.append({
                            "symbol": sym,
                            "company_name": sym,
                            "fool_status": fool_status,
                            "source_file": path.name,
                        })

        logger.info(f"Read {len(results)} stocks from {path.name} ({list_name})")

    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")

    return results


# --- CrewAI Tool wrappers ---

class ReadIBDPDFInput(BaseModel):
    file_path: str = Field(..., description="Path to the IBD PDF file")


class ReadIBDPDFTool(BaseTool):
    """Extract stocks from IBD PDF files."""
    name: str = "read_ibd_pdf"
    description: str = (
        "Read an IBD PDF file (Smart Table, Tech Leaders, Top 200, etc.) "
        "and extract stock symbols with available ratings."
    )
    args_schema: type[BaseModel] = ReadIBDPDFInput

    def _run(self, file_path: str) -> str:
        stocks = read_ibd_pdf(file_path)
        if not stocks:
            return f"No stocks found in {file_path}"
        lines = [f"Found {len(stocks)} stocks in {Path(file_path).name}:"]
        for s in stocks[:5]:
            comp = s.get("composite_rating", "N/A")
            lines.append(f"  {s['symbol']}: Comp={comp}")
        if len(stocks) > 5:
            lines.append(f"  ... and {len(stocks) - 5} more")
        return "\n".join(lines)


class ReadFoolPDFInput(BaseModel):
    file_path: str = Field(..., description="Path to the Motley Fool PDF file")


class ReadFoolPDFTool(BaseTool):
    """Extract stocks from Motley Fool PDFs."""
    name: str = "read_fool_pdf"
    description: str = (
        "Read a Motley Fool PDF (Epic Rankings or New Recommendations) "
        "and extract stock symbols with recommendation status."
    )
    args_schema: type[BaseModel] = ReadFoolPDFInput

    def _run(self, file_path: str) -> str:
        stocks = read_fool_pdf(file_path)
        if not stocks:
            return f"No stocks found in {file_path}"
        lines = [f"Found {len(stocks)} stocks in {Path(file_path).name}:"]
        for s in stocks[:5]:
            lines.append(f"  {s['symbol']} — {s.get('fool_status', 'N/A')}")
        if len(stocks) > 5:
            lines.append(f"  ... and {len(stocks) - 5} more")
        return "\n".join(lines)
