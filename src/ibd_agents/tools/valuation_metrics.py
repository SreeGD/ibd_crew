"""
Analyst Agent Tool: Valuation & Risk Metrics
Estimate P/E, PEG, Beta, Sharpe Ratio, Alpha, Volatility,
Estimated Return, and Risk Rating from sector baselines + IBD ratings.

Deterministic estimates are the default. Optional LLM enrichment
(`enrich_valuation_llm`) uses Claude Haiku to look up real financial
metrics and provide narrative investment guidance per stock.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Market Constants
# ---------------------------------------------------------------------------

RISK_FREE_RATE = 4.5       # Current T-bill rate %
MARKET_RETURN = 12.0       # S&P 500 annual return assumption %
SP500_FORWARD_PE = 23.3    # S&P 500 forward P/E
SP500_10Y_AVG_PE = 19.6    # 10-year average P/E


# ---------------------------------------------------------------------------
# Sector Baseline Constants (28 IBD Sectors + UNKNOWN fallback)
# ---------------------------------------------------------------------------

# P/E ranges: {"low": floor, "mid": midpoint, "high": ceiling}
SECTOR_PE_BASELINES: dict[str, dict[str, float]] = {
    "AEROSPACE":          {"low": 18.0, "mid": 24.0, "high": 32.0},
    "AUTO":               {"low": 8.0,  "mid": 12.0, "high": 18.0},
    "BANKS":              {"low": 8.0,  "mid": 11.0, "high": 16.0},
    "BUILDING":           {"low": 12.0, "mid": 18.0, "high": 26.0},
    "BUSINESS SERVICES":  {"low": 18.0, "mid": 25.0, "high": 35.0},
    "CHEMICALS":          {"low": 12.0, "mid": 18.0, "high": 25.0},
    "CHIPS":              {"low": 15.0, "mid": 25.0, "high": 40.0},
    "COMPUTER":           {"low": 15.0, "mid": 22.0, "high": 35.0},
    "CONSUMER":           {"low": 15.0, "mid": 22.0, "high": 30.0},
    "DEFENSE":            {"low": 16.0, "mid": 22.0, "high": 30.0},
    "ELECTRONICS":        {"low": 14.0, "mid": 22.0, "high": 32.0},
    "ENERGY":             {"low": 8.0,  "mid": 12.0, "high": 18.0},
    "FINANCE":            {"low": 10.0, "mid": 14.0, "high": 20.0},
    "FOOD/BEVERAGE":      {"low": 18.0, "mid": 24.0, "high": 30.0},
    "HEALTHCARE":         {"low": 18.0, "mid": 28.0, "high": 40.0},
    "INSURANCE":          {"low": 8.0,  "mid": 12.0, "high": 18.0},
    "INTERNET":           {"low": 20.0, "mid": 35.0, "high": 60.0},
    "LEISURE":            {"low": 14.0, "mid": 20.0, "high": 28.0},
    "MACHINERY":          {"low": 14.0, "mid": 20.0, "high": 28.0},
    "MEDIA":              {"low": 12.0, "mid": 18.0, "high": 28.0},
    "MEDICAL":            {"low": 22.0, "mid": 35.0, "high": 55.0},
    "MINING":             {"low": 10.0, "mid": 16.0, "high": 24.0},
    "REAL ESTATE":        {"low": 15.0, "mid": 22.0, "high": 30.0},
    "RETAIL":             {"low": 15.0, "mid": 22.0, "high": 35.0},
    "SOFTWARE":           {"low": 25.0, "mid": 40.0, "high": 70.0},
    "TELECOM":            {"low": 12.0, "mid": 18.0, "high": 25.0},
    "TRANSPORTATION":     {"low": 12.0, "mid": 18.0, "high": 26.0},
    "UTILITIES":          {"low": 14.0, "mid": 18.0, "high": 24.0},
}

# Fallback for UNKNOWN or unrecognized sectors
_DEFAULT_PE = {"low": 15.0, "mid": 22.0, "high": 35.0}

# Beta baselines per sector (vs S&P 500)
SECTOR_BETA_BASELINES: dict[str, float] = {
    "AEROSPACE": 1.15,  "AUTO": 1.20,  "BANKS": 1.10,  "BUILDING": 1.15,
    "BUSINESS SERVICES": 1.05,  "CHEMICALS": 1.10,  "CHIPS": 1.35,
    "COMPUTER": 1.20,  "CONSUMER": 0.95,  "DEFENSE": 0.85,
    "ELECTRONICS": 1.25,  "ENERGY": 1.30,  "FINANCE": 1.15,
    "FOOD/BEVERAGE": 0.65,  "HEALTHCARE": 0.90,  "INSURANCE": 0.80,
    "INTERNET": 1.30,  "LEISURE": 1.10,  "MACHINERY": 1.10,
    "MEDIA": 1.15,  "MEDICAL": 0.95,  "MINING": 1.40,
    "REAL ESTATE": 0.85,  "RETAIL": 1.05,  "SOFTWARE": 1.25,
    "TELECOM": 0.75,  "TRANSPORTATION": 1.00,  "UTILITIES": 0.55,
}

_DEFAULT_BETA = 1.0

# Annualized volatility % per sector
SECTOR_VOLATILITY_BASELINES: dict[str, float] = {
    "AEROSPACE": 28.0,  "AUTO": 32.0,  "BANKS": 25.0,  "BUILDING": 28.0,
    "BUSINESS SERVICES": 25.0,  "CHEMICALS": 27.0,  "CHIPS": 35.0,
    "COMPUTER": 30.0,  "CONSUMER": 22.0,  "DEFENSE": 20.0,
    "ELECTRONICS": 32.0,  "ENERGY": 35.0,  "FINANCE": 26.0,
    "FOOD/BEVERAGE": 16.0,  "HEALTHCARE": 22.0,  "INSURANCE": 20.0,
    "INTERNET": 33.0,  "LEISURE": 27.0,  "MACHINERY": 26.0,
    "MEDIA": 28.0,  "MEDICAL": 25.0,  "MINING": 38.0,
    "REAL ESTATE": 22.0,  "RETAIL": 25.0,  "SOFTWARE": 32.0,
    "TELECOM": 20.0,  "TRANSPORTATION": 24.0,  "UTILITIES": 18.0,
}

_DEFAULT_VOLATILITY = 28.0


# ---------------------------------------------------------------------------
# Category Constants
# ---------------------------------------------------------------------------

PE_CATEGORIES = ("Deep Value", "Value", "Reasonable", "Growth Premium", "Speculative")
PEG_CATEGORIES = ("Undervalued", "Fair Value", "Expensive")
RETURN_CATEGORIES = ("Strong", "Good", "Moderate", "Weak")
SHARPE_CATEGORIES = ("Excellent", "Good", "Moderate", "Below Average")
ALPHA_CATEGORIES = ("Strong Outperformer", "Outperformer", "Slight Underperformer", "Underperformer")
RISK_RATING_CATEGORIES = ("Excellent", "Good", "Moderate", "Below Average", "Poor")


# ---------------------------------------------------------------------------
# P/E Estimation
# ---------------------------------------------------------------------------

def estimate_pe(
    sector: str,
    composite_rating: int,
    eps_rating: int,
    rs_rating: int,
) -> float:
    """
    Estimate P/E from sector baseline and IBD ratings.

    Higher-growth stocks (high Composite + EPS + RS) command premium valuations.
    The growth factor places the stock along the sector's low-to-high P/E range.
    """
    pe_range = SECTOR_PE_BASELINES.get(sector, _DEFAULT_PE)
    low, mid, high = pe_range["low"], pe_range["mid"], pe_range["high"]

    # Growth factor 0.0-1.0 from weighted ratings
    growth_factor = (
        (composite_rating / 99.0) * 0.3
        + (eps_rating / 99.0) * 0.4
        + (rs_rating / 99.0) * 0.3
    )

    # Map to sector range with slight premium for top stocks
    sector_range = high - low
    estimated = low + (growth_factor * sector_range * 1.2)

    # Clamp
    return round(max(low * 0.8, min(high * 1.5, estimated)), 1)


def classify_pe_category(estimated_pe: float, sector: str) -> str:
    """
    Classify P/E into valuation category relative to sector.

    Deep Value: Below sector low
    Value: Below sector midpoint
    Reasonable: At/near sector midpoint (within 10%)
    Growth Premium: Above midpoint up to sector high
    Speculative: >1.5x sector high
    """
    pe_range = SECTOR_PE_BASELINES.get(sector, _DEFAULT_PE)
    low, mid, high = pe_range["low"], pe_range["mid"], pe_range["high"]

    if estimated_pe < low:
        return "Deep Value"
    elif estimated_pe < mid:
        return "Value"
    elif estimated_pe <= mid * 1.1:
        return "Reasonable"
    elif estimated_pe <= high * 1.5:
        return "Growth Premium"
    else:
        return "Speculative"


# ---------------------------------------------------------------------------
# PEG Ratio
# ---------------------------------------------------------------------------

def calculate_peg(estimated_pe: float, eps_rating: Optional[int]) -> Optional[float]:
    """
    PEG = P/E / (EPS Rating * 0.5).

    EPS Rating * 0.5 approximates earnings growth rate.
    Returns None if eps_rating is None or zero.
    """
    if eps_rating is None or eps_rating == 0:
        return None
    growth_rate = eps_rating * 0.5
    return round(estimated_pe / growth_rate, 2)


def classify_peg_category(peg: Optional[float]) -> Optional[str]:
    """
    Classify PEG ratio.

    Undervalued: PEG < 1.0
    Fair Value: 1.0 <= PEG <= 2.0
    Expensive: PEG > 2.0
    """
    if peg is None:
        return None
    if peg < 1.0:
        return "Undervalued"
    elif peg <= 2.0:
        return "Fair Value"
    else:
        return "Expensive"


# ---------------------------------------------------------------------------
# Beta Estimation
# ---------------------------------------------------------------------------

def estimate_beta(
    sector: str,
    composite_rating: int,
    rs_rating: int,
) -> float:
    """
    Estimate beta from sector baseline + momentum adjustment.

    High RS/Composite stocks move more aggressively -> positive beta adjustment.
    """
    baseline = SECTOR_BETA_BASELINES.get(sector, _DEFAULT_BETA)

    # Normalized deviation from moderate (75 = moderate, 99 = max)
    rs_dev = (rs_rating - 75) / 24.0
    comp_dev = (composite_rating - 75) / 24.0
    momentum_adj = (rs_dev * 0.6 + comp_dev * 0.4) * 0.3

    estimated = baseline + momentum_adj
    return round(max(0.2, min(2.5, estimated)), 2)


# ---------------------------------------------------------------------------
# Estimated Return
# ---------------------------------------------------------------------------

def estimate_return_pct(rs_rating: int) -> float:
    """
    Estimate 12-month return from RS rating (stepped mapping).

    RS 99 -> 60%, RS 95 -> 55%, RS 90 -> 50%, RS 85 -> 40%,
    RS 80 -> 30%, RS 70 -> 20%, RS 50 -> 10%, below -> 5%.
    """
    if rs_rating >= 99:
        return 60.0
    elif rs_rating >= 95:
        return 55.0
    elif rs_rating >= 90:
        return 50.0
    elif rs_rating >= 85:
        return 40.0
    elif rs_rating >= 80:
        return 30.0
    elif rs_rating >= 70:
        return 20.0
    elif rs_rating >= 50:
        return 10.0
    else:
        return 5.0


def classify_return_category(est_return: float) -> str:
    """Classify estimated return: Strong/Good/Moderate/Weak."""
    if est_return >= 40.0:
        return "Strong"
    elif est_return >= 20.0:
        return "Good"
    elif est_return >= 0.0:
        return "Moderate"
    else:
        return "Weak"


# ---------------------------------------------------------------------------
# Volatility Estimation
# ---------------------------------------------------------------------------

def estimate_volatility(
    sector: str,
    rs_rating: int,
    composite_rating: int,
) -> float:
    """
    Estimate annualized volatility from sector baseline + momentum.

    Higher momentum stocks tend to have higher volatility.
    """
    baseline = SECTOR_VOLATILITY_BASELINES.get(sector, _DEFAULT_VOLATILITY)

    # Momentum factor: RS 50-99 -> 0-1 range
    momentum_factor = max(0.0, (rs_rating - 50) / 49.0)
    vol_adjustment = momentum_factor * 8.0

    estimated = baseline + vol_adjustment
    return round(max(10.0, min(60.0, estimated)), 1)


# ---------------------------------------------------------------------------
# Sharpe Ratio
# ---------------------------------------------------------------------------

def calculate_sharpe(est_return: float, volatility: float) -> Optional[float]:
    """
    Sharpe Ratio = (Return - Risk-Free Rate) / Volatility.

    Returns None if volatility is zero.
    """
    if volatility == 0:
        return None
    return round((est_return - RISK_FREE_RATE) / volatility, 2)


def classify_sharpe_category(sharpe: Optional[float]) -> Optional[str]:
    """
    Classify Sharpe Ratio.

    Excellent: >= 1.5
    Good: >= 1.0
    Moderate: >= 0.7
    Below Average: < 0.7
    """
    if sharpe is None:
        return None
    if sharpe >= 1.5:
        return "Excellent"
    elif sharpe >= 1.0:
        return "Good"
    elif sharpe >= 0.7:
        return "Moderate"
    else:
        return "Below Average"


# ---------------------------------------------------------------------------
# Alpha (CAPM-based)
# ---------------------------------------------------------------------------

def calculate_alpha(est_return: float, beta: float) -> float:
    """
    Alpha = Actual Return - Expected Return (CAPM).

    Expected = RiskFree + Beta * (MarketReturn - RiskFree)
    """
    expected = RISK_FREE_RATE + beta * (MARKET_RETURN - RISK_FREE_RATE)
    return round(est_return - expected, 2)


def classify_alpha_category(alpha: float) -> str:
    """
    Classify Alpha.

    Strong Outperformer: >= +10%
    Outperformer: >= 0%
    Slight Underperformer: >= -5%
    Underperformer: < -5%
    """
    if alpha >= 10.0:
        return "Strong Outperformer"
    elif alpha >= 0.0:
        return "Outperformer"
    elif alpha >= -5.0:
        return "Slight Underperformer"
    else:
        return "Underperformer"


# ---------------------------------------------------------------------------
# Risk Rating (combined)
# ---------------------------------------------------------------------------

def calculate_risk_rating(sharpe: Optional[float], beta: float) -> str:
    """
    Overall risk rating combining Sharpe Ratio and Beta.

    Excellent: Sharpe >= 1.5 and Beta <= 1.2
    Good: Sharpe >= 1.0 and Beta <= 1.5
    Moderate: Sharpe >= 0.7
    Below Average: Sharpe >= 0.4
    Poor: Sharpe < 0.4
    """
    if sharpe is None:
        return "Moderate"
    if sharpe >= 1.5 and beta <= 1.2:
        return "Excellent"
    elif sharpe >= 1.0 and beta <= 1.5:
        return "Good"
    elif sharpe >= 0.7:
        return "Moderate"
    elif sharpe >= 0.4:
        return "Below Average"
    else:
        return "Poor"


# ---------------------------------------------------------------------------
# Convenience: Compute All Metrics for One Stock
# ---------------------------------------------------------------------------

def compute_all_valuation_metrics(
    sector: str,
    composite_rating: int,
    rs_rating: int,
    eps_rating: int,
) -> dict:
    """
    Compute all valuation and risk metrics for a single stock.

    Returns dict with 13 fields matching RatedStock valuation fields.
    """
    pe = estimate_pe(sector, composite_rating, eps_rating, rs_rating)
    pe_cat = classify_pe_category(pe, sector)
    peg = calculate_peg(pe, eps_rating)
    peg_cat = classify_peg_category(peg)
    beta = estimate_beta(sector, composite_rating, rs_rating)
    est_return = estimate_return_pct(rs_rating)
    ret_cat = classify_return_category(est_return)
    vol = estimate_volatility(sector, rs_rating, composite_rating)
    sharpe = calculate_sharpe(est_return, vol)
    sharpe_cat = classify_sharpe_category(sharpe)
    alpha = calculate_alpha(est_return, beta)
    alpha_cat = classify_alpha_category(alpha)
    risk = calculate_risk_rating(sharpe, beta)

    return {
        "estimated_pe": pe,
        "pe_category": pe_cat,
        "peg_ratio": peg,
        "peg_category": peg_cat,
        "estimated_beta": beta,
        "estimated_return_pct": est_return,
        "return_category": ret_cat,
        "estimated_volatility_pct": vol,
        "sharpe_ratio": sharpe,
        "sharpe_category": sharpe_cat,
        "alpha_pct": alpha,
        "alpha_category": alpha_cat,
        "risk_rating": risk,
    }


# ---------------------------------------------------------------------------
# Summary Aggregation
# ---------------------------------------------------------------------------

def build_market_context() -> dict:
    """Build market context dict."""
    return {
        "sp500_forward_pe": SP500_FORWARD_PE,
        "sp500_10y_avg_pe": SP500_10Y_AVG_PE,
        "current_premium_pct": round((SP500_FORWARD_PE / SP500_10Y_AVG_PE - 1) * 100, 1),
        "risk_free_rate": RISK_FREE_RATE,
        "market_return_assumption": MARKET_RETURN,
    }


def compute_valuation_summary(rated_stocks) -> dict:
    """
    Compute aggregate valuation summary from a list of RatedStock objects.

    Returns dict matching ValuationSummary fields.
    """
    # Collect PE values by tier
    pe_values: list[float] = []
    tier_pe: dict[int, list[float]] = {1: [], 2: [], 3: []}
    sector_pe: dict[str, list[float]] = {}
    peg_values: list[float] = []
    sharpe_values: list[float] = []
    beta_values: list[float] = []
    alpha_values: list[float] = []

    # PE category counts
    pe_cats = {"Deep Value": 0, "Value": 0, "Reasonable": 0, "Growth Premium": 0, "Speculative": 0}
    peg_cats = {"Undervalued": 0, "Fair Value": 0, "Expensive": 0}

    for s in rated_stocks:
        if s.estimated_pe is not None:
            pe_values.append(s.estimated_pe)
            if s.tier in tier_pe:
                tier_pe[s.tier].append(s.estimated_pe)
            if s.sector not in sector_pe:
                sector_pe[s.sector] = []
            sector_pe[s.sector].append(s.estimated_pe)

        if s.pe_category and s.pe_category in pe_cats:
            pe_cats[s.pe_category] += 1

        if s.peg_ratio is not None:
            peg_values.append(s.peg_ratio)
        if s.peg_category and s.peg_category in peg_cats:
            peg_cats[s.peg_category] += 1

        if s.sharpe_ratio is not None:
            sharpe_values.append(s.sharpe_ratio)
        if s.estimated_beta is not None:
            beta_values.append(s.estimated_beta)
        if s.alpha_pct is not None:
            alpha_values.append(s.alpha_pct)

    total = sum(pe_cats.values())

    # Tier PE stats
    tier_stats = []
    for t in [1, 2, 3]:
        vals = tier_pe[t]
        if vals:
            tier_stats.append({
                "tier": t,
                "avg_pe": round(statistics.mean(vals), 1),
                "median_pe": round(statistics.median(vals), 1),
                "min_pe": round(min(vals), 1),
                "max_pe": round(max(vals), 1),
                "stock_count": len(vals),
            })

    # Top 5 sectors by avg PE
    sector_avgs = []
    for sec, vals in sector_pe.items():
        if vals:
            sector_avgs.append({
                "sector": sec,
                "avg_pe": round(statistics.mean(vals), 1),
                "stock_count": len(vals),
            })
    sector_avgs.sort(key=lambda x: x["avg_pe"], reverse=True)
    top_sectors = sector_avgs[:5]

    # PEG analysis
    peg_undervalued = peg_cats.get("Undervalued", 0)
    peg_total = sum(peg_cats.values())

    peg_analysis = {
        "avg_peg": round(statistics.mean(peg_values), 2) if peg_values else None,
        "median_peg": round(statistics.median(peg_values), 2) if peg_values else None,
        "undervalued_count": peg_undervalued,
        "fair_value_count": peg_cats.get("Fair Value", 0),
        "expensive_count": peg_cats.get("Expensive", 0),
        "pct_undervalued": round(peg_undervalued / peg_total * 100, 1) if peg_total > 0 else 0.0,
    }

    return {
        "market_context": build_market_context(),
        "pe_distribution": {
            "deep_value_count": pe_cats["Deep Value"],
            "value_count": pe_cats["Value"],
            "reasonable_count": pe_cats["Reasonable"],
            "growth_premium_count": pe_cats["Growth Premium"],
            "speculative_count": pe_cats["Speculative"],
            "total": total,
        },
        "tier_pe_stats": tier_stats,
        "top_sectors_by_pe": top_sectors,
        "peg_analysis": peg_analysis,
        "avg_sharpe": round(statistics.mean(sharpe_values), 2) if sharpe_values else None,
        "avg_beta": round(statistics.mean(beta_values), 2) if beta_values else None,
        "avg_alpha": round(statistics.mean(alpha_values), 2) if alpha_values else None,
    }


# ---------------------------------------------------------------------------
# LLM Valuation Enrichment
# ---------------------------------------------------------------------------

_LLM_PROMPT_TEMPLATE = """\
For each stock below, provide real financial metrics from your knowledge.
Return a JSON array with one object per stock. Use ONLY valid JSON — no markdown, no code fences, no extra text.

Fields per stock:
- symbol: str (ticker, uppercase)
- pe_ratio: float or null (trailing P/E ratio)
- forward_pe: float or null (forward P/E ratio)
- beta: float or null (5-year beta vs S&P 500)
- annual_return_1y: float or null (approximate 1-year return %)
- volatility: float or null (annualized volatility %)
- dividend_yield: float or null (dividend yield %)
- market_cap_b: float or null (market cap in billions USD)
- valuation_grade: str (one of: "Deep Value", "Value", "Reasonable", "Growth Premium", "Speculative")
- risk_grade: str (one of: "Excellent", "Good", "Moderate", "Below Average", "Poor")
- guidance: str (2-3 sentence investment analysis covering valuation, momentum, and risk)

If you don't know a metric, use null. Use your best knowledge as of early 2025.

Stocks:
{stock_list}
"""


def enrich_valuation_llm(
    stocks: list[dict],
    model: str = "claude-haiku-4-5-20251001",
    batch_size: int = 30,
) -> dict[str, dict]:
    """
    Look up real P/E, Beta, etc. from LLM knowledge + generate guidance.

    Args:
        stocks: List of dicts with keys: symbol, sector, company_name,
                composite_rating, rs_rating, eps_rating, tier.
        model: Anthropic model ID.
        batch_size: Stocks per LLM call.

    Returns:
        {symbol: {pe_ratio, forward_pe, beta, annual_return_1y, volatility,
                  dividend_yield, market_cap_b, valuation_grade, risk_grade,
                  guidance}} for each stock the LLM provided data for.
        Returns empty dict on any failure (graceful fallback).
    """
    if not stocks:
        return {}

    result: dict[str, dict] = {}

    try:
        import anthropic
        client = anthropic.Anthropic(timeout=60.0)

        for i in range(0, len(stocks), batch_size):
            batch = stocks[i:i + batch_size]

            # Build readable stock list for prompt
            stock_lines = []
            for s in batch:
                stock_lines.append(
                    f"  {s['symbol']} | {s.get('company_name', s['symbol'])} | "
                    f"Sector: {s.get('sector', 'UNKNOWN')} | "
                    f"Comp: {s.get('composite_rating', '?')} | "
                    f"RS: {s.get('rs_rating', '?')} | "
                    f"EPS: {s.get('eps_rating', '?')} | "
                    f"Tier: {s.get('tier', '?')}"
                )
            stock_list_str = "\n".join(stock_lines)
            prompt = _LLM_PROMPT_TEMPLATE.format(stock_list=stock_list_str)

            response = client.messages.create(
                model=model,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else ""

            # Parse JSON array from response
            parsed = _parse_llm_response(text, {s["symbol"] for s in batch})
            result.update(parsed)

            logger.info(
                f"LLM valuation batch {i // batch_size + 1}: "
                f"enriched {len(parsed)}/{len(batch)} stocks"
            )

    except ImportError:
        logger.warning("anthropic SDK not installed — skipping LLM valuation enrichment")
    except Exception as e:
        logger.warning(f"LLM valuation enrichment error: {e}")

    return result


def _parse_llm_response(text: str, valid_symbols: set[str]) -> dict[str, dict]:
    """Parse and validate LLM JSON response into {symbol: metrics} dict."""
    result: dict[str, dict] = {}

    # Try to find JSON array in response (may have surrounding text)
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.warning("LLM response did not contain a JSON array")
        return result

    json_str = text[start:end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM JSON: {e}")
        return result

    if not isinstance(data, list):
        logger.warning("LLM JSON is not an array")
        return result

    valid_valuation_grades = {"Deep Value", "Value", "Reasonable", "Growth Premium", "Speculative"}
    valid_risk_grades = {"Excellent", "Good", "Moderate", "Below Average", "Poor"}

    for item in data:
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", "")).strip().upper()
        if not sym or sym not in valid_symbols:
            continue

        # Validate and clean each field
        cleaned: dict = {"symbol": sym}
        for float_key in ("pe_ratio", "forward_pe", "beta", "annual_return_1y",
                          "volatility", "dividend_yield", "market_cap_b"):
            val = item.get(float_key)
            if val is not None:
                try:
                    val = float(val)
                    if not math.isfinite(val):
                        val = None
                except (ValueError, TypeError):
                    val = None
            cleaned[float_key] = val

        # Validate grade fields
        vg = item.get("valuation_grade")
        cleaned["valuation_grade"] = vg if vg in valid_valuation_grades else None

        rg = item.get("risk_grade")
        cleaned["risk_grade"] = rg if rg in valid_risk_grades else None

        guidance = item.get("guidance")
        cleaned["guidance"] = str(guidance) if guidance else None

        result[sym] = cleaned

    return result


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class ValuationMetricsInput(BaseModel):
    sector: str = Field(..., description="IBD sector name")
    composite_rating: int = Field(..., description="IBD Composite Rating (1-99)")
    rs_rating: int = Field(..., description="IBD RS Rating (1-99)")
    eps_rating: int = Field(..., description="IBD EPS Rating (1-99)")


class ValuationMetricsTool(BaseTool):
    """Estimate valuation and risk metrics from sector baselines + IBD ratings."""

    name: str = "valuation_metrics"
    description: str = (
        "Estimate P/E, PEG, Beta, Sharpe Ratio, Alpha, Volatility, "
        "Return, and Risk Rating for a stock using sector baselines "
        "and IBD ratings. No external API required."
    )
    args_schema: type[BaseModel] = ValuationMetricsInput

    def _run(
        self,
        sector: str,
        composite_rating: int,
        rs_rating: int,
        eps_rating: int,
    ) -> str:
        m = compute_all_valuation_metrics(sector, composite_rating, rs_rating, eps_rating)
        return (
            f"P/E: {m['estimated_pe']:.1f} ({m['pe_category']}), "
            f"PEG: {m['peg_ratio']}, "
            f"Beta: {m['estimated_beta']:.2f}, "
            f"Est Return: {m['estimated_return_pct']:.0f}% ({m['return_category']}), "
            f"Vol: {m['estimated_volatility_pct']:.1f}%, "
            f"Sharpe: {m['sharpe_ratio']}, "
            f"Alpha: {m['alpha_pct']:.1f}% ({m['alpha_category']}), "
            f"Risk: {m['risk_rating']}"
        )
