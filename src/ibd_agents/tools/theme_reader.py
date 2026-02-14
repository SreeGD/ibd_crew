"""
Research Agent Tool: Theme Reader
Extract stocks and ETFs from Schwab thematic investment PDFs.
Tags securities with theme memberships for multi-source validation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Map Schwab PDF filenames → theme names (from spec §4.3)
FILE_TO_THEME: dict[str, str] = {
    "Schwab_AI":                       "Artificial Intelligence",
    "Schwab_RR":                       "Robotics & Automation",
    "Schwab_AD":                       "Electric Vehicles",
    "Schwab_RenEn":                    "Renewable Energy",
    "Schwab_CC":                       "Climate Change",
    "SchwabSpaceEc":                   "Space Economy",
    "Schwab_CS":                       "Cybersecurity",
    "Schwab_BD":                       "Big Data & Analytics",
    "Schwab3d":                        "3D Printing",
    "Schwab_DLL":                      "Digital Lifestyle",
    "Schwab_Ecom":                     "E-Commerce",
    "Schwab_PM":                       "Digital Payments",
    "Schwab_CBT":                      "Cannabis",
    "SchwabAging":                     "Aging Population",
    "Schwab_HAE":                      "Healthcare Innovation",
    "Schwab_China":                    "China Growth",
    "Schwab_Can":                      "Canada",
    "SchwabDef":                       "Defense & Aerospace",
    "SchwabBC":                        "Blockchain/Crypto",
    "Schwab_OMV":                      "Online Media & Video",
    "Schwab_Investing_Themes":         "Overview",
}

# Key ETFs per theme (from spec §4.3)
THEME_KEY_ETFS: dict[str, list[str]] = {
    "Artificial Intelligence":    ["BOTZ", "ROBO", "AIQ"],
    "Robotics & Automation":      ["ROBO", "BOTZ"],
    "Cybersecurity":              ["CIBR", "HACK", "BUG"],
    "Big Data & Analytics":       ["CLOU", "WCLD"],
    "Cloud Computing":            ["SKYY", "CLOU"],
    "3D Printing":                ["PRNT"],
    "E-Commerce":                 ["IBUY", "ONLN"],
    "Digital Payments":           ["IPAY", "FINX"],
    "Digital Lifestyle":          [],
    "Online Media & Video":       [],
    "Renewable Energy":           ["ICLN", "TAN", "QCLN"],
    "Climate Change":             [],
    "Electric Vehicles":          ["DRIV", "IDRV", "LIT"],
    "Healthcare Innovation":      ["XBI", "IBB", "ARKG"],
    "Aging Population":           [],
    "Space Economy":              ["UFO", "ARKX"],
    "Defense & Aerospace":        ["ITA", "PPA", "XAR"],
    "China Growth":               ["MCHI", "FXI", "KWEB"],
    "Cannabis":                   ["MJ", "MSOS"],
    "Blockchain/Crypto":          ["BLOK", "BITO"],
    "Canada":                     [],
}

# Known ETF symbols (broader set for detection)
KNOWN_ETFS: set[str] = set()
for etfs in THEME_KEY_ETFS.values():
    KNOWN_ETFS.update(etfs)

# Ticker regex
TICKER_RE = re.compile(r'\b([A-Z]{1,5})\b')

# Common non-ticker words
NON_TICKERS: set[str] = {
    "THE", "AND", "FOR", "NOT", "ARE", "BUT", "ALL", "ANY",
    "CAN", "HAS", "HER", "WAS", "ONE", "OUR", "OUT", "DAY",
    "NEW", "NOW", "OLD", "SEE", "WAY", "MAY", "ETF", "PDF",
    "USA", "CEO", "CFO", "GDP", "EPS", "YTD", "TOP", "HIGH",
    "LOW", "VOL", "FUND", "STOCK", "INDEX", "FROM", "WITH",
    "THIS", "THAT", "HAVE", "BEEN", "WILL", "EACH", "MAKE",
    "SCHWAB", "CHARLES", "THEME", "INVEST",
}


def _detect_theme(filename: str) -> str:
    """Determine theme name from Schwab PDF filename."""
    stem = Path(filename).stem
    for pattern, theme in FILE_TO_THEME.items():
        if pattern.lower() in stem.lower():
            return theme
    return "Unknown Theme"


def _is_likely_ticker(word: str) -> bool:
    """Check if a word looks like a stock/ETF ticker."""
    if word in NON_TICKERS:
        return False
    if not word.isalpha() or not word.isupper():
        return False
    return 1 <= len(word) <= 5


def _classify_security(symbol: str) -> str:
    """Classify as stock or ETF."""
    if symbol in KNOWN_ETFS:
        return "etf"
    # Heuristic: 3-4 letter symbols ending in common ETF patterns
    if len(symbol) in (3, 4) and any(
        symbol.endswith(suffix) for suffix in ["X", "Q", "K"]
    ):
        return "etf"  # Possible ETF, but not definitive
    return "stock"


def read_schwab_theme(file_path: str) -> list[dict]:
    """
    Read a Schwab theme PDF and extract stocks/ETFs with theme tags.

    Returns:
        List of dicts with keys: symbol, company_name, theme, is_etf,
        is_key_theme_etf, source_file
    """
    import pdfplumber

    path = Path(file_path)
    if not path.exists():
        logger.warning(f"File not found: {file_path}")
        return []

    theme = _detect_theme(path.name)
    if theme == "Overview":
        logger.info(f"Skipping overview file: {path.name}")
        return []

    key_etfs = set(THEME_KEY_ETFS.get(theme, []))
    results: list[dict] = []
    seen_symbols: set[str] = set()

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # Try tables
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    for row in table:
                        for cell in row:
                            if not cell:
                                continue
                            words = TICKER_RE.findall(str(cell))
                            for w in words:
                                if _is_likely_ticker(w) and w not in seen_symbols:
                                    seen_symbols.add(w)
                                    sec_type = _classify_security(w)
                                    results.append({
                                        "symbol": w,
                                        "company_name": w,
                                        "theme": theme,
                                        "is_etf": sec_type == "etf",
                                        "is_key_theme_etf": w in key_etfs,
                                        "source_file": path.name,
                                    })

                # Also check raw text
                text = page.extract_text()
                if text:
                    words = TICKER_RE.findall(text)
                    for w in words:
                        if _is_likely_ticker(w) and w not in seen_symbols:
                            seen_symbols.add(w)
                            sec_type = _classify_security(w)
                            results.append({
                                "symbol": w,
                                "company_name": w,
                                "theme": theme,
                                "is_etf": sec_type == "etf",
                                "is_key_theme_etf": w in key_etfs,
                                "source_file": path.name,
                            })

        logger.info(f"Read {len(results)} securities from {path.name} ({theme})")

    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")

    return results


# --- CrewAI Tool wrapper ---

class ReadSchwabThemeInput(BaseModel):
    file_path: str = Field(..., description="Path to the Schwab theme PDF")


class ReadSchwabThemeTool(BaseTool):
    """Extract stocks and ETFs from Schwab thematic PDFs."""
    name: str = "read_schwab_theme"
    description: str = (
        "Read a Schwab investing theme PDF and extract stocks/ETFs "
        "with theme tags. Identifies key theme ETFs."
    )
    args_schema: type[BaseModel] = ReadSchwabThemeInput

    def _run(self, file_path: str) -> str:
        securities = read_schwab_theme(file_path)
        if not securities:
            return f"No securities found in {file_path}"

        theme = securities[0]["theme"] if securities else "Unknown"
        stocks = [s for s in securities if not s["is_etf"]]
        etfs = [s for s in securities if s["is_etf"]]
        key_etfs = [s for s in securities if s["is_key_theme_etf"]]

        lines = [
            f"Theme: {theme}",
            f"Found: {len(stocks)} stocks, {len(etfs)} ETFs ({len(key_etfs)} key theme ETFs)",
        ]
        if key_etfs:
            lines.append(f"Key ETFs: {', '.join(s['symbol'] for s in key_etfs)}")
        return "\n".join(lines)
