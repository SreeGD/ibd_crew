"""
Research Agent Tool: Cap Size Classifier
Assigns market cap categories to stocks using static mapping + optional LLM fallback.
Categories: large (>$10B), mid ($2B-$10B), small (<$2B).
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    from crewai.tools import BaseTool
    HAS_CREWAI = True
except ImportError:
    from pydantic import BaseModel as BaseTool
    HAS_CREWAI = False
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

VALID_CAP_SIZES: set[str] = {"large", "mid", "small"}

# Static cap-size mapping for well-known tickers.
# large = market cap > $10B
# mid   = $2B - $10B
# small = < $2B
KNOWN_CAPS: dict[str, str] = {
    # CHIPS — mostly large, some mid
    "NVDA": "large", "AMD": "large", "INTC": "large", "AVGO": "large",
    "TSM": "large", "QCOM": "large", "TXN": "large", "MU": "large",
    "AMAT": "large", "LRCX": "large", "KLAC": "large", "MRVL": "large",
    "ASML": "large", "ARM": "large", "ADI": "large", "NXPI": "large",
    "ON": "large", "SWKS": "mid", "MPWR": "large", "MCHP": "large",
    "SMCI": "mid", "CLS": "large", "ALAB": "mid", "CRDO": "mid",
    "SITM": "mid", "MTSI": "mid", "LITE": "mid", "NVMI": "mid",
    # SOFTWARE — mostly large, some mid
    "MSFT": "large", "ORCL": "large", "CRM": "large",
    "ADBE": "large", "NOW": "large", "PLTR": "large",
    "PANW": "large", "CRWD": "large", "DDOG": "large",
    "ZS": "large", "FTNT": "large", "WDAY": "large",
    "SNOW": "large", "NET": "large", "TEAM": "large",
    "HUBS": "large", "INTU": "large", "SNPS": "large",
    "CDNS": "large", "ANSS": "large", "TTD": "large",
    "APP": "large", "BILL": "mid", "SMAR": "mid",
    "SEI": "mid",
    # INTERNET — mostly large
    "GOOGL": "large", "GOOG": "large", "META": "large",
    "AMZN": "large", "NFLX": "large", "BKNG": "large",
    "ABNB": "large", "UBER": "large", "LYFT": "mid",
    "DASH": "large", "SNAP": "mid", "PINS": "large",
    "SHOP": "large", "ETSY": "mid", "EBAY": "large",
    "SPOT": "large",
    # COMPUTER
    "AAPL": "large", "DELL": "large", "HPQ": "large",
    "HPE": "large", "CSCO": "large", "ANET": "large",
    "NTAP": "large", "PSTG": "mid",
    # AUTO
    "TSLA": "large", "GM": "large", "F": "large", "TM": "large",
    "RIVN": "mid", "LCID": "small", "NIO": "mid",
    # AEROSPACE
    "BA": "large", "LMT": "large", "NOC": "large",
    "RTX": "large", "GD": "large", "HWM": "large",
    "TDG": "large", "HEI": "large", "KRMN": "mid",
    "WWD": "mid",
    # MEDICAL
    "LLY": "large", "UNH": "large", "JNJ": "large",
    "ABBV": "large", "MRK": "large", "PFE": "large",
    "TMO": "large", "ABT": "large", "AMGN": "large",
    "GILD": "large", "ISRG": "large", "REGN": "large",
    "VRTX": "large", "DXCM": "large", "HALO": "mid",
    "KRYS": "mid", "MIRM": "small", "HROW": "small",
    "ANAB": "small", "AXSM": "mid",
    # ENERGY
    "XOM": "large", "CVX": "large", "COP": "large",
    "SLB": "large", "EOG": "large", "PXD": "large",
    "MPC": "large", "VLO": "large", "PSX": "large",
    "OXY": "large", "HAL": "large", "BKV": "mid",
    # BANKS
    "JPM": "large", "BAC": "large", "WFC": "large",
    "C": "large", "GS": "large", "MS": "large",
    "USB": "large", "PNC": "large", "SCHW": "large",
    "NPB": "small",
    # FINANCE
    "BRK.B": "large", "V": "large", "MA": "large",
    "AXP": "large", "BLK": "large", "SPGI": "large",
    "ICE": "large", "CME": "large", "MCO": "large",
    "COIN": "large", "WT": "mid", "JCAP": "small",
    # RETAIL
    "WMT": "large", "COST": "large", "HD": "large",
    "LOW": "large", "TGT": "large", "TJX": "large",
    "ROST": "large", "DG": "large", "DLTR": "large",
    "FIVE": "mid",
    # MINING
    "NEM": "large", "GOLD": "large", "FNV": "large",
    "WPM": "large", "AEM": "large", "KGC": "mid",
    "AGI": "mid", "GFI": "mid", "AU": "mid",
    "HL": "mid", "AG": "mid", "RGLD": "mid",
    "SCCO": "large", "FCX": "large", "IAG": "mid",
    "ORLA": "small", "CDE": "mid", "EGO": "mid",
    "BVN": "mid", "ARMN": "small", "AUGO": "small",
    # BUILDING
    "FIX": "large", "STRL": "mid", "DY": "mid",
    "EME": "large", "MTZ": "mid", "PWR": "large",
    "VMC": "large", "MLM": "large", "CDNL": "small",
    # CONSUMER
    "PG": "large", "KO": "large", "PEP": "large",
    "PM": "large", "MO": "large", "CL": "large",
    "EL": "large", "NKE": "large", "TPR": "mid",
    # TRANSPORTATION
    "UNP": "large", "CSX": "large",
    "NSC": "large", "UPS": "large",
    "FDX": "large", "DAL": "large",
    "UAL": "large", "LUV": "large",
    "FTAI": "mid",
    # TELECOM
    "T": "large", "VZ": "large", "TMUS": "large",
    # MEDIA
    "DIS": "large", "CMCSA": "large", "WBD": "mid",
    "PARA": "mid", "NWSA": "mid",
    # INSURANCE
    "BRK.A": "large", "PGR": "large", "TRV": "large",
    "ALL": "large", "MET": "large", "AIG": "large",
    "HIG": "large", "HG": "mid",
    # ELECTRONICS
    "APH": "large", "TEL": "large", "GLW": "large",
    "KEYS": "large", "FTV": "large",
    "VRT": "large", "FN": "mid", "MOD": "mid",
    "ZEPP": "small", "AAOI": "small",
    # MACHINERY
    "CAT": "large", "DE": "large", "ETN": "large",
    "EMR": "large", "ROK": "large", "CMI": "large",
    "DOV": "large", "IR": "large", "WHR": "mid",
    # CHEMICALS
    "LIN": "large", "APD": "large", "SHW": "large",
    "DD": "large", "ECL": "large",
    # FOOD/BEVERAGE
    "MDLZ": "large", "GIS": "large",
    "K": "large", "HSY": "large",
    "SJM": "mid", "CAG": "mid",
    "COCO": "mid",
    # UTILITIES
    "NEE": "large", "DUK": "large", "SO": "large",
    "D": "large", "AEP": "large", "EXC": "large",
    "SRE": "large", "CEG": "large", "VST": "large",
    # LEISURE
    "MAR": "large", "HLT": "large", "WYNN": "mid",
    "LVS": "large", "MGM": "mid", "RCL": "large",
    "CCL": "large", "NCLH": "mid", "AS": "mid",
    # REAL ESTATE
    "AMT": "large", "PLD": "large", "CCI": "large",
    "EQIX": "large", "SPG": "large", "O": "large",
    # DEFENSE
    "HII": "mid", "LHX": "large",
    # HEALTHCARE
    "CI": "large", "ELV": "large", "HUM": "large",
    "CNC": "large", "MOH": "large", "ALHC": "mid",
    # BUSINESS SERVICES
    "ACN": "large", "FISV": "large",
    "ADP": "large", "PAYX": "large",
    "BR": "large", "FIS": "large",
    "GPN": "large", "WEX": "mid",
    "PACS": "mid", "BTSG": "mid",
    # Additional tickers from IBD lists
    "TAL": "mid", "LRN": "mid",
    "OR": "mid",
    "APLD": "mid",
    "TFPM": "mid", "GMED": "mid", "GE": "large",
}

# IBD list names that strongly imply cap size
_LIST_CAP_HINTS: dict[str, str] = {
    "IBD Big Cap 20": "large",
}


def classify_caps_static(
    symbols: list[str],
    ibd_lists_map: Optional[dict[str, list[str]]] = None,
) -> dict[str, str]:
    """Classify cap sizes using static mapping only (no LLM, no I/O).

    Args:
        symbols: List of ticker symbols to classify.
        ibd_lists_map: Optional dict mapping symbol -> list of IBD list names.
            If provided, list membership is used as a fallback hint.
    """
    result: dict[str, str] = {}
    for sym in symbols:
        cap = KNOWN_CAPS.get(sym.strip().upper())
        if cap:
            result[sym] = cap
        elif ibd_lists_map:
            lists = ibd_lists_map.get(sym, [])
            for lst in lists:
                hint = _LIST_CAP_HINTS.get(lst)
                if hint:
                    result[sym] = hint
                    break
    return result


def classify_caps_llm(
    symbols: list[str],
    model: str = "claude-haiku-4-5-20251001",
    batch_size: int = 100,
) -> dict[str, str]:
    """Classify cap sizes using Anthropic Claude for stocks not in static mapping.

    Applies static mapping first, then batches remaining symbols to Claude.
    Requires the anthropic SDK and ANTHROPIC_API_KEY env var.
    """
    result = classify_caps_static(symbols)
    remaining = [s for s in symbols if s not in result]

    if not remaining:
        return result

    try:
        import anthropic
        client = anthropic.Anthropic(timeout=60.0)

        for i in range(0, len(remaining), batch_size):
            batch = remaining[i:i + batch_size]
            prompt = (
                "Classify each US stock ticker by approximate current market capitalization "
                "into exactly one of these categories:\n"
                "- large: market cap > $10 billion\n"
                "- mid: market cap $2 billion to $10 billion\n"
                "- small: market cap < $2 billion\n\n"
                "Respond with ONLY ticker:category pairs, one per line. "
                "Use the EXACT category names: large, mid, small. "
                "No explanations, no extra text. If you don't recognize a ticker, skip it.\n\n"
                f"Tickers: {', '.join(batch)}"
            )

            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else ""

            classified_count = 0
            for line in text.strip().split("\n"):
                line = line.strip()
                if ":" not in line:
                    continue
                parts = line.split(":", 1)
                sym = parts[0].strip().upper()
                cap = parts[1].strip().lower()
                if sym in [s.strip().upper() for s in batch] and cap in VALID_CAP_SIZES:
                    result[sym] = cap
                    classified_count += 1

            logger.info(f"LLM cap classified batch {i // batch_size + 1}: "
                        f"{classified_count}/{len(batch)}")

    except ImportError:
        logger.warning("anthropic SDK not installed — skipping LLM cap classification")
    except Exception as e:
        logger.warning(f"LLM cap classification error: {e}")

    return result


# --- CrewAI Tool wrapper ---

class ClassifyCapInput(BaseModel):
    symbols: list[str] = Field(..., description="List of stock symbols to classify")


class CapClassifierTool(BaseTool):
    """Classify stocks by market capitalization size."""
    name: str = "classify_cap_sizes"
    description: str = (
        "Classify stock symbols into market cap categories (large, mid, small). "
        "Uses static mapping for well-known stocks and LLM for unknowns."
    )
    args_schema: type[BaseModel] = ClassifyCapInput

    def _run(self, symbols: list[str]) -> str:
        result = classify_caps_llm(symbols)
        lines = [f"Classified {len(result)}/{len(symbols)} symbols:"]
        for sym, cap in sorted(result.items()):
            lines.append(f"  {sym} -> {cap}")
        unclassified = [s for s in symbols if s not in result]
        if unclassified:
            lines.append(f"  Unclassified: {', '.join(unclassified[:20])}")
        return "\n".join(lines)
