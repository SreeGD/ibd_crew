"""
Research Agent Tool: Sector Classifier
Assigns IBD sectors to stocks using static mapping + optional LLM fallback.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

try:
    from crewai.tools import BaseTool
    HAS_CREWAI = True
except ImportError:
    from pydantic import BaseModel as BaseTool
    HAS_CREWAI = False
from pydantic import BaseModel, Field

from ibd_agents.schemas.research_output import IBD_SECTORS

logger = logging.getLogger(__name__)

# Static sector mapping for well-known tickers.
# Covers major stocks likely to appear across IBD lists.
KNOWN_SECTORS: dict[str, str] = {
    # CHIPS
    "NVDA": "CHIPS", "AMD": "CHIPS", "INTC": "CHIPS", "AVGO": "CHIPS",
    "TSM": "CHIPS", "QCOM": "CHIPS", "TXN": "CHIPS", "MU": "CHIPS",
    "AMAT": "CHIPS", "LRCX": "CHIPS", "KLAC": "CHIPS", "MRVL": "CHIPS",
    "ASML": "CHIPS", "ARM": "CHIPS", "ADI": "CHIPS", "NXPI": "CHIPS",
    "ON": "CHIPS", "SWKS": "CHIPS", "MPWR": "CHIPS", "MCHP": "CHIPS",
    "SMCI": "CHIPS", "CLS": "CHIPS", "ALAB": "CHIPS", "CRDO": "CHIPS",
    "SITM": "CHIPS", "MTSI": "CHIPS", "LITE": "CHIPS", "NVMI": "CHIPS",
    # SOFTWARE
    "MSFT": "SOFTWARE", "ORCL": "SOFTWARE", "CRM": "SOFTWARE",
    "ADBE": "SOFTWARE", "NOW": "SOFTWARE", "PLTR": "SOFTWARE",
    "PANW": "SOFTWARE", "CRWD": "SOFTWARE", "DDOG": "SOFTWARE",
    "ZS": "SOFTWARE", "FTNT": "SOFTWARE", "WDAY": "SOFTWARE",
    "SNOW": "SOFTWARE", "NET": "SOFTWARE", "TEAM": "SOFTWARE",
    "HUBS": "SOFTWARE", "INTU": "SOFTWARE", "SNPS": "SOFTWARE",
    "CDNS": "SOFTWARE", "ANSS": "SOFTWARE", "TTD": "SOFTWARE",
    "APP": "SOFTWARE", "BILL": "SOFTWARE", "SMAR": "SOFTWARE",
    # INTERNET
    "GOOGL": "INTERNET", "GOOG": "INTERNET", "META": "INTERNET",
    "AMZN": "INTERNET", "NFLX": "INTERNET", "BKNG": "INTERNET",
    "ABNB": "INTERNET", "UBER": "INTERNET", "LYFT": "INTERNET",
    "DASH": "INTERNET", "SNAP": "INTERNET", "PINS": "INTERNET",
    "SHOP": "INTERNET", "ETSY": "INTERNET", "EBAY": "INTERNET",
    "SPOT": "INTERNET",
    # COMPUTER
    "AAPL": "COMPUTER", "DELL": "COMPUTER", "HPQ": "COMPUTER",
    "HPE": "COMPUTER", "CSCO": "COMPUTER", "ANET": "COMPUTER",
    "NTAP": "COMPUTER", "PSTG": "COMPUTER",
    # AUTO
    "TSLA": "AUTO", "GM": "AUTO", "F": "AUTO", "TM": "AUTO",
    "RIVN": "AUTO", "LCID": "AUTO", "NIO": "AUTO",
    # AEROSPACE
    "BA": "AEROSPACE", "LMT": "AEROSPACE", "NOC": "AEROSPACE",
    "RTX": "AEROSPACE", "GD": "AEROSPACE", "HWM": "AEROSPACE",
    "TDG": "AEROSPACE", "HEI": "AEROSPACE", "KRMN": "AEROSPACE",
    # MEDICAL
    "LLY": "MEDICAL", "UNH": "MEDICAL", "JNJ": "MEDICAL",
    "ABBV": "MEDICAL", "MRK": "MEDICAL", "PFE": "MEDICAL",
    "TMO": "MEDICAL", "ABT": "MEDICAL", "AMGN": "MEDICAL",
    "GILD": "MEDICAL", "ISRG": "MEDICAL", "REGN": "MEDICAL",
    "VRTX": "MEDICAL", "DXCM": "MEDICAL", "HALO": "MEDICAL",
    "KRYS": "MEDICAL", "MIRM": "MEDICAL", "HROW": "MEDICAL",
    "ANAB": "MEDICAL",
    # ENERGY
    "XOM": "ENERGY", "CVX": "ENERGY", "COP": "ENERGY",
    "SLB": "ENERGY", "EOG": "ENERGY", "PXD": "ENERGY",
    "MPC": "ENERGY", "VLO": "ENERGY", "PSX": "ENERGY",
    "OXY": "ENERGY", "HAL": "ENERGY",
    # BANKS
    "JPM": "BANKS", "BAC": "BANKS", "WFC": "BANKS",
    "C": "BANKS", "GS": "BANKS", "MS": "BANKS",
    "USB": "BANKS", "PNC": "BANKS", "SCHW": "BANKS",
    # FINANCE
    "BRK.B": "FINANCE", "V": "FINANCE", "MA": "FINANCE",
    "AXP": "FINANCE", "BLK": "FINANCE", "SPGI": "FINANCE",
    "ICE": "FINANCE", "CME": "FINANCE", "MCO": "FINANCE",
    "COIN": "FINANCE",
    # RETAIL
    "WMT": "RETAIL", "COST": "RETAIL", "HD": "RETAIL",
    "LOW": "RETAIL", "TGT": "RETAIL", "TJX": "RETAIL",
    "ROST": "RETAIL", "DG": "RETAIL", "DLTR": "RETAIL",
    "FIVE": "RETAIL",
    # MINING
    "NEM": "MINING", "GOLD": "MINING", "FNV": "MINING",
    "WPM": "MINING", "AEM": "MINING", "KGC": "MINING",
    "AGI": "MINING", "GFI": "MINING", "AU": "MINING",
    "HL": "MINING", "AG": "MINING", "RGLD": "MINING",
    "SCCO": "MINING", "FCX": "MINING", "IAG": "MINING",
    "ORLA": "MINING", "CDE": "MINING", "EGO": "MINING",
    "BVN": "MINING",
    # BUILDING
    "FIX": "BUILDING", "STRL": "BUILDING", "DY": "BUILDING",
    "EME": "BUILDING", "MTZ": "BUILDING", "PWR": "BUILDING",
    "VMC": "BUILDING", "MLM": "BUILDING",
    # CONSUMER
    "PG": "CONSUMER", "KO": "CONSUMER", "PEP": "CONSUMER",
    "PM": "CONSUMER", "MO": "CONSUMER", "CL": "CONSUMER",
    "EL": "CONSUMER", "NKE": "CONSUMER",
    # TRANSPORTATION
    "UNP": "TRANSPORTATION", "CSX": "TRANSPORTATION",
    "NSC": "TRANSPORTATION", "UPS": "TRANSPORTATION",
    "FDX": "TRANSPORTATION", "DAL": "TRANSPORTATION",
    "UAL": "TRANSPORTATION", "LUV": "TRANSPORTATION",
    # TELECOM
    "T": "TELECOM", "VZ": "TELECOM", "TMUS": "TELECOM",
    # MEDIA
    "DIS": "MEDIA", "CMCSA": "MEDIA", "WBD": "MEDIA",
    "PARA": "MEDIA", "NWSA": "MEDIA",
    # INSURANCE
    "BRK.A": "INSURANCE", "PGR": "INSURANCE", "TRV": "INSURANCE",
    "ALL": "INSURANCE", "MET": "INSURANCE", "AIG": "INSURANCE",
    "HIG": "INSURANCE",
    # ELECTRONICS
    "APH": "ELECTRONICS", "TEL": "ELECTRONICS", "GLW": "ELECTRONICS",
    "KEYS": "ELECTRONICS", "FTV": "ELECTRONICS",
    # MACHINERY
    "CAT": "MACHINERY", "DE": "MACHINERY", "ETN": "MACHINERY",
    "EMR": "MACHINERY", "ROK": "MACHINERY", "CMI": "MACHINERY",
    "DOV": "MACHINERY", "IR": "MACHINERY",
    # CHEMICALS
    "LIN": "CHEMICALS", "APD": "CHEMICALS", "SHW": "CHEMICALS",
    "DD": "CHEMICALS", "ECL": "CHEMICALS",
    # FOOD/BEVERAGE
    "MDLZ": "FOOD/BEVERAGE", "GIS": "FOOD/BEVERAGE",
    "K": "FOOD/BEVERAGE", "HSY": "FOOD/BEVERAGE",
    "SJM": "FOOD/BEVERAGE", "CAG": "FOOD/BEVERAGE",
    # UTILITIES
    "NEE": "UTILITIES", "DUK": "UTILITIES", "SO": "UTILITIES",
    "D": "UTILITIES", "AEP": "UTILITIES", "EXC": "UTILITIES",
    "SRE": "UTILITIES", "CEG": "UTILITIES", "VST": "UTILITIES",
    # LEISURE
    "MAR": "LEISURE", "HLT": "LEISURE", "WYNN": "LEISURE",
    "LVS": "LEISURE", "MGM": "LEISURE", "RCL": "LEISURE",
    "CCL": "LEISURE", "NCLH": "LEISURE",
    # REAL ESTATE
    "AMT": "REAL ESTATE", "PLD": "REAL ESTATE", "CCI": "REAL ESTATE",
    "EQIX": "REAL ESTATE", "SPG": "REAL ESTATE", "O": "REAL ESTATE",
    # DEFENSE
    "LMT": "DEFENSE", "NOC": "DEFENSE", "GD": "DEFENSE",
    "HII": "DEFENSE", "LHX": "DEFENSE",
    # HEALTHCARE
    "CI": "HEALTHCARE", "ELV": "HEALTHCARE", "HUM": "HEALTHCARE",
    "CNC": "HEALTHCARE", "MOH": "HEALTHCARE",
    # BUSINESS SERVICES
    "ACN": "BUSINESS SERVICES", "FISV": "BUSINESS SERVICES",
    "ADP": "BUSINESS SERVICES", "PAYX": "BUSINESS SERVICES",
    "BR": "BUSINESS SERVICES", "FIS": "BUSINESS SERVICES",
    "GPN": "BUSINESS SERVICES", "WEX": "BUSINESS SERVICES",
    "PACS": "BUSINESS SERVICES", "BTSG": "BUSINESS SERVICES",
    # MISC
    "VRT": "ELECTRONICS", "FN": "ELECTRONICS",
    "MOD": "ELECTRONICS", "SEI": "SOFTWARE",
    "WWD": "AEROSPACE", "FTAI": "TRANSPORTATION",
    "COCO": "FOOD/BEVERAGE", "NPB": "BANKS", "HG": "INSURANCE",
    "ARMN": "MINING", "AUGO": "MINING",
    # Additional stocks from IBD 50 / IPO Leaders / Smart Table
    "TAL": "SOFTWARE",        # TAL Education Group — online education/tech
    "LRN": "SOFTWARE",        # Stride Inc — online education technology
    "ZEPP": "ELECTRONICS",    # Zepp Health — wearable fitness electronics
    "WHR": "MACHINERY",       # Whirlpool — home appliances / industrial
    "OR": "MINING",           # Osisko Gold Royalties — gold/mining royalties
    "WT": "FINANCE",          # WisdomTree — financial services / ETF provider
    "AS": "LEISURE",          # Amer Sports — sporting goods (Arc'teryx, Wilson)
    "AAOI": "ELECTRONICS",    # Applied Optoelectronics — fiber optic components
    "BKV": "ENERGY",          # BKV Corp — natural gas producer
    "JCAP": "FINANCE",        # Jefferson Capital Holdings — financial services
    "ALHC": "HEALTHCARE",     # Alignment Healthcare — healthcare services
    "AXSM": "MEDICAL",        # Axsome Therapeutics — biopharmaceutical
    "CDNL": "BUILDING",       # Cardinal Infrastructure — infrastructure development
    "APLD": "SOFTWARE",       # Applied Digital — cloud/AI infrastructure
}


def classify_sectors_static(symbols: list[str]) -> dict[str, str]:
    """Classify sectors using static mapping only (no LLM, no I/O)."""
    result = {}
    for sym in symbols:
        sector = KNOWN_SECTORS.get(sym.strip().upper())
        if sector:
            result[sym] = sector
    return result


def classify_sectors_llm(
    symbols: list[str],
    model: str = "claude-haiku-4-5-20251001",
    batch_size: int = 100,
) -> dict[str, str]:
    """Classify sectors using Anthropic Claude for stocks not in static mapping.

    Applies static mapping first, then batches remaining symbols to Claude.
    Requires the anthropic SDK and ANTHROPIC_API_KEY env var.
    """
    result = classify_sectors_static(symbols)
    remaining = [s for s in symbols if s not in result]

    if not remaining:
        return result

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0)

        sectors_str = ", ".join(IBD_SECTORS)

        for i in range(0, len(remaining), batch_size):
            batch = remaining[i:i + batch_size]
            prompt = (
                f"Classify each US stock ticker into exactly one of these IBD sectors:\n"
                f"{sectors_str}\n\n"
                f"Respond with ONLY ticker:SECTOR pairs, one per line. "
                f"Use the EXACT sector names from the list above. "
                f"No explanations, no extra text. If you don't recognize a ticker, skip it.\n\n"
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
                sector = parts[1].strip().upper()
                if sym in batch and sector in IBD_SECTORS:
                    result[sym] = sector
                    classified_count += 1

            logger.info(f"LLM classified batch {i//batch_size + 1}: "
                        f"{classified_count}/{len(batch)}")

    except ImportError:
        logger.warning("anthropic SDK not installed — skipping LLM sector classification")
    except Exception as e:
        logger.warning(f"LLM sector classification error: {e}")

    return result


# --- CrewAI Tool wrapper ---

class ClassifySectorInput(BaseModel):
    symbols: list[str] = Field(..., description="List of stock symbols to classify")


class ClassifySectorTool(BaseTool):
    """Classify stocks into IBD sectors."""
    name: str = "classify_sectors"
    description: str = (
        "Classify stock symbols into one of 28 IBD sector categories. "
        "Uses static mapping for well-known stocks and LLM for unknowns."
    )
    args_schema: type[BaseModel] = ClassifySectorInput

    def _run(self, symbols: list[str]) -> str:
        result = classify_sectors_llm(symbols)
        lines = [f"Classified {len(result)}/{len(symbols)} symbols:"]
        for sym, sector in sorted(result.items()):
            lines.append(f"  {sym} -> {sector}")
        unclassified = [s for s in symbols if s not in result]
        if unclassified:
            lines.append(f"  Unclassified: {', '.join(unclassified[:20])}")
        return "\n".join(lines)
