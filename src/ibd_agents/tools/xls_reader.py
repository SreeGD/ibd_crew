"""
Research Agent Tool: XLS Reader
Read IBD Excel files (Big Cap 20, IBD 50, Sector Leaders, etc.)
and extract stock symbols with IBD ratings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Map XLS filename patterns → IBD list name for validation scoring
FILE_TO_LIST_NAME: dict[str, str] = {
    "BIG_CAP_20": "IBD Big Cap 20",
    "IBD_50": "IBD 50",
    "SECTOR_LEADERS": "IBD Sector Leaders",
    "STOCK_SPOTLIGHT": "IBD Stock Spotlight",
    "IPO_LEADERS": "IBD IPO Leaders",
    "RISING_PROFIT_ESTIMATES": "IBD Rising Profit Estimates",
    "RELATIVE_STRENGTH_AT_NEW_HIGH": "IBD RS at New High",
}

# Common column name mappings — IBD files have inconsistent headers
COLUMN_ALIASES: dict[str, list[str]] = {
    "symbol": ["Symbol", "Sym", "Ticker", "SYMBOL", "symbol"],
    "company_name": ["Company Name", "CompanyName", "Name", "Company", "COMPANY"],
    "composite": [
        "Composite Rating", "Comp Rating", "CompositeRating", "Comp",
        "IBD Composite", "Composite", "COMPOSITE",
    ],
    "rs": [
        "RS Rating", "Rel Str", "RelativeStrength", "RS",
        "Relative Strength", "Rel Strength", "REL STR",
    ],
    "eps": [
        "EPS Rating", "EPS", "EPS Rtg", "Earnings",
        "EPS Rank", "EPS RATING",
    ],
    "smr": [
        "SMR Rating", "SMR", "SMR Rtg", "Sales/Margins/ROE",
    ],
    "acc_dis": [
        "Acc/Dis Rating", "Acc/Dis", "A/D Rating", "Acc Dis",
        "Accumulation", "A/D", "ACC/DIS",
    ],
    "sector": [
        "Sector", "IBD Sector", "Industry Sector", "SECTOR",
    ],
    "industry_group": [
        "Industry Group", "Group", "Industry", "Ind Group",
    ],
}


def _find_column(df: pd.DataFrame, target: str) -> Optional[str]:
    """Find a column in the DataFrame matching known aliases."""
    aliases = COLUMN_ALIASES.get(target, [target])
    for alias in aliases:
        if alias in df.columns:
            return alias
        # Case-insensitive fallback
        for col in df.columns:
            if col.strip().lower() == alias.lower():
                return col
    return None


def _safe_int(val: Any) -> Optional[int]:
    """Convert value to int, returning None for non-numeric."""
    if val is None or pd.isna(val):
        return None
    try:
        v = int(float(val))
        return v if 1 <= v <= 99 else None
    except (ValueError, TypeError):
        return None


def _safe_rating(val: Any) -> Optional[str]:
    """Convert SMR/Acc-Dis value to standard rating string.

    Maps plus/minus variants to schema-valid values:
    B+→B, A+→A, A-→A, C+→C, D+→D (schema allows {A, B, B-, C, D, E}).
    B- is kept as-is since it's already valid.
    """
    if val is None or pd.isna(val):
        return None
    s = str(val).strip().upper()
    valid = {"A", "B", "B-", "C", "D", "E"}
    if s in valid:
        return s
    # Map plus variants to base letter, minus variants to base (except B-)
    if len(s) == 2 and s[1] in ("+", "-"):
        base = s[0]
        if base in ("A", "B", "C", "D", "E"):
            return "B-" if s == "B-" else base
    return None


def _find_header_row(df: pd.DataFrame, max_rows: int = 20) -> Optional[int]:
    """Scan the first *max_rows* rows of a header-less DataFrame for the header.

    Returns the row index containing "Symbol" (case-insensitive), or None.
    """
    limit = min(max_rows, len(df))
    for idx in range(limit):
        row_vals = [str(v).strip().lower() for v in df.iloc[idx] if v is not None and not (isinstance(v, float) and pd.isna(v))]
        if "symbol" in row_vals:
            return idx
    return None


def _detect_list_name(filename: str) -> str:
    """Determine IBD list name from filename."""
    upper = filename.upper()
    for pattern, list_name in FILE_TO_LIST_NAME.items():
        if pattern in upper:
            return list_name
    return f"IBD {filename}"


def read_ibd_xls(file_path: str) -> list[dict]:
    """
    Read an IBD XLS file and extract stocks with ratings.

    Returns:
        List of dicts with keys: symbol, company_name, composite_rating,
        rs_rating, eps_rating, smr_rating, acc_dis_rating, sector,
        ibd_list, source_file
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"File not found: {file_path}")
        return []

    list_name = _detect_list_name(path.stem)
    results: list[dict] = []

    try:
        # Try reading with different engines
        try:
            df_raw = pd.read_excel(file_path, header=None, engine="xlrd")
        except Exception:
            df_raw = pd.read_excel(file_path, header=None, engine="openpyxl")

        if df_raw.empty:
            logger.warning(f"Empty file: {file_path}")
            return []

        # Auto-detect header row (real IBD exports have metadata rows before headers)
        header_idx = _find_header_row(df_raw)
        if header_idx is not None:
            df = df_raw.iloc[header_idx + 1:].copy()
            df.columns = [str(v).strip() for v in df_raw.iloc[header_idx]]
            df.reset_index(drop=True, inplace=True)
        else:
            # Fallback: assume row 0 is header (e.g. test mock files)
            df = df_raw
            df.columns = [str(v).strip() for v in df.iloc[0]]
            df = df.iloc[1:].reset_index(drop=True)

        # Find columns
        sym_col = _find_column(df, "symbol")
        if sym_col is None:
            logger.warning(f"No symbol column found in {file_path}. Columns: {list(df.columns)}")
            return []

        name_col = _find_column(df, "company_name")
        comp_col = _find_column(df, "composite")
        rs_col = _find_column(df, "rs")
        eps_col = _find_column(df, "eps")
        smr_col = _find_column(df, "smr")
        ad_col = _find_column(df, "acc_dis")
        sector_col = _find_column(df, "sector")

        for _, row in df.iterrows():
            sym = row.get(sym_col) if sym_col else None
            if sym is None or pd.isna(sym):
                continue
            sym = str(sym).strip().upper()
            if not sym or len(sym) > 10:
                continue

            stock = {
                "symbol": sym,
                "company_name": str(row.get(name_col, sym)).strip() if name_col else sym,
                "composite_rating": _safe_int(row.get(comp_col)) if comp_col else None,
                "rs_rating": _safe_int(row.get(rs_col)) if rs_col else None,
                "eps_rating": _safe_int(row.get(eps_col)) if eps_col else None,
                "smr_rating": _safe_rating(row.get(smr_col)) if smr_col else None,
                "acc_dis_rating": _safe_rating(row.get(ad_col)) if ad_col else None,
                "sector": str(row.get(sector_col, "")).strip().upper() if sector_col else None,
                "ibd_list": list_name,
                "source_file": path.name,
            }
            results.append(stock)

        logger.info(f"Read {len(results)} stocks from {path.name} ({list_name})")

    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")

    return results


def read_ibd_csv(file_path: str) -> list[dict]:
    """
    Read an IBD CSV file and extract stocks with ratings.

    Expects the same column format as IBD XLS files (Symbol, Composite Rating,
    RS Rating, EPS Rating, SMR Rating, Acc/Dis Rating, Sector).

    Returns:
        List of dicts with keys: symbol, company_name, composite_rating,
        rs_rating, eps_rating, smr_rating, acc_dis_rating, sector,
        ibd_list, source_file
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"File not found: {file_path}")
        return []

    list_name = _detect_list_name(path.stem)
    results: list[dict] = []

    try:
        df_raw = pd.read_csv(file_path, header=None)

        if df_raw.empty:
            logger.warning(f"Empty file: {file_path}")
            return []

        # Auto-detect header row (real IBD exports have metadata rows before headers)
        header_idx = _find_header_row(df_raw)
        if header_idx is not None:
            df = df_raw.iloc[header_idx + 1:].copy()
            df.columns = [str(v).strip() for v in df_raw.iloc[header_idx]]
            df.reset_index(drop=True, inplace=True)
        else:
            # Fallback: assume row 0 is header (e.g. test mock files)
            df = df_raw
            df.columns = [str(v).strip() for v in df.iloc[0]]
            df = df.iloc[1:].reset_index(drop=True)

        # Find columns
        sym_col = _find_column(df, "symbol")
        if sym_col is None:
            logger.warning(f"No symbol column found in {file_path}. Columns: {list(df.columns)}")
            return []

        name_col = _find_column(df, "company_name")
        comp_col = _find_column(df, "composite")
        rs_col = _find_column(df, "rs")
        eps_col = _find_column(df, "eps")
        smr_col = _find_column(df, "smr")
        ad_col = _find_column(df, "acc_dis")
        sector_col = _find_column(df, "sector")

        for _, row in df.iterrows():
            sym = row.get(sym_col) if sym_col else None
            if sym is None or pd.isna(sym):
                continue
            sym = str(sym).strip().upper()
            if not sym or len(sym) > 10:
                continue

            stock = {
                "symbol": sym,
                "company_name": str(row.get(name_col, sym)).strip() if name_col else sym,
                "composite_rating": _safe_int(row.get(comp_col)) if comp_col else None,
                "rs_rating": _safe_int(row.get(rs_col)) if rs_col else None,
                "eps_rating": _safe_int(row.get(eps_col)) if eps_col else None,
                "smr_rating": _safe_rating(row.get(smr_col)) if smr_col else None,
                "acc_dis_rating": _safe_rating(row.get(ad_col)) if ad_col else None,
                "sector": str(row.get(sector_col, "")).strip().upper() if sector_col else None,
                "ibd_list": list_name,
                "source_file": path.name,
            }
            results.append(stock)

        logger.info(f"Read {len(results)} stocks from {path.name} ({list_name})")

    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")

    return results


class ReadIBDXLSInput(BaseModel):
    """Input for ReadIBDXLSTool."""
    file_path: str = Field(..., description="Path to the IBD XLS file")


class ReadIBDXLSTool(BaseTool):
    """Read IBD Excel files and extract stocks with ratings."""
    name: str = "read_ibd_xls"
    description: str = (
        "Read an IBD Excel file (Big Cap 20, IBD 50, Sector Leaders, etc.) "
        "and extract stock symbols with all available IBD ratings: "
        "Composite, RS, EPS, SMR, Acc/Dis."
    )
    args_schema: type[BaseModel] = ReadIBDXLSInput

    def _run(self, file_path: str) -> str:
        stocks = read_ibd_xls(file_path)
        if not stocks:
            return f"No stocks found in {file_path}"

        lines = [f"Found {len(stocks)} stocks in {Path(file_path).name}:"]
        for s in stocks[:5]:
            lines.append(
                f"  {s['symbol']}: Comp={s['composite_rating']} "
                f"RS={s['rs_rating']} EPS={s['eps_rating']} "
                f"SMR={s['smr_rating']} A/D={s['acc_dis_rating']}"
            )
        if len(stocks) > 5:
            lines.append(f"  ... and {len(stocks) - 5} more")
        return "\n".join(lines)


class ReadIBDCSVInput(BaseModel):
    """Input for ReadIBDCSVTool."""
    file_path: str = Field(..., description="Path to the IBD CSV file")


class ReadIBDCSVTool(BaseTool):
    """Read IBD CSV files and extract stocks with ratings."""
    name: str = "read_ibd_csv"
    description: str = (
        "Read an IBD CSV file (same column format as XLS: Symbol, Composite Rating, "
        "RS Rating, EPS Rating, SMR Rating, Acc/Dis Rating, Sector) "
        "and extract stock symbols with all available IBD ratings."
    )
    args_schema: type[BaseModel] = ReadIBDCSVInput

    def _run(self, file_path: str) -> str:
        stocks = read_ibd_csv(file_path)
        if not stocks:
            return f"No stocks found in {file_path}"

        lines = [f"Found {len(stocks)} stocks in {Path(file_path).name}:"]
        for s in stocks[:5]:
            lines.append(
                f"  {s['symbol']}: Comp={s['composite_rating']} "
                f"RS={s['rs_rating']} EPS={s['eps_rating']} "
                f"SMR={s['smr_rating']} A/D={s['acc_dis_rating']}"
            )
        if len(stocks) > 5:
            lines.append(f"  ... and {len(stocks) - 5} more")
        return "\n".join(lines)
