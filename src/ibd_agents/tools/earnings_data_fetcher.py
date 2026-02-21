"""
Earnings Risk Analyst Tool: Earnings Data Fetcher
IBD Momentum Investment Framework v4.0

Pure functions for:
- Fetching earnings calendar (real via yfinance or mock)
- Fetching historical earnings reactions (real or mock)
- Computing cushion ratios and risk classification
- Building strategy options with scenario tables
- Assessing portfolio-level earnings concentration
- Optional LLM enrichment for estimate revisions

Follows the dual-path pattern: yfinance real data + deterministic mock.
"""

from __future__ import annotations

import logging
import math
import random
from datetime import date, datetime, timedelta
from typing import Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

from ibd_agents.schemas.earnings_risk_output import (
    ADD_MAX_POSITION_PCT,
    ADD_MIN_BEAT_RATE,
    ADD_MIN_CUSHION_RATIO,
    CONCENTRATION_CRITICAL_PCT,
    CONCENTRATION_MODERATE_PCT,
    CUSHION_COMFORTABLE,
    CUSHION_MANAGEABLE_LOW,
    CUSHION_THIN_LOW,
    IV_ELEVATED_RATIO,
    LARGE_POSITION_PCT,
    LOOKFORWARD_DAYS,
    MIN_CONFIDENT_QUARTERS,
    STRATEGY_LADDER,
    WEAK_REGIMES,
    CushionCategory,
    EarningsRisk,
    EarningsWeek,
    EstimateRevision,
    HistoricalEarnings,
    ImpliedVolSignal,
    PortfolioEarningsConcentration,
    ScenarioOutcome,
    StrategyOption,
    StrategyType,
)

logger = logging.getLogger(__name__)

# Deterministic RNG for mock data
_RNG = random.Random(42)

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    yf = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Real Data Fetching (yfinance)
# ---------------------------------------------------------------------------

def fetch_earnings_calendar_real(symbols: list[str]) -> dict[str, dict]:
    """
    Fetch next earnings date for each symbol via yfinance.

    Returns {symbol: {earnings_date, reporting_time, days_until_earnings}}.
    Symbols that fail are omitted.
    """
    if not HAS_YFINANCE or not symbols:
        return {}

    today = date.today()
    result: dict[str, dict] = {}

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            cal = ticker.calendar
            if cal is None:
                continue

            earnings_date_val = None

            # calendar can be DataFrame or dict
            if hasattr(cal, "empty") and not cal.empty:
                if "Earnings Date" in cal.columns:
                    for ed in cal["Earnings Date"]:
                        ed_date = ed.date() if hasattr(ed, "date") else ed
                        if ed_date >= today:
                            earnings_date_val = ed_date
                            break
            elif isinstance(cal, dict) and "Earnings Date" in cal:
                ed = cal["Earnings Date"]
                if isinstance(ed, list) and len(ed) > 0:
                    ed_val = ed[0]
                    ed_date = ed_val.date() if hasattr(ed_val, "date") else ed_val
                elif hasattr(ed, "date"):
                    ed_date = ed.date()
                else:
                    continue
                if ed_date >= today:
                    earnings_date_val = ed_date

            if earnings_date_val is not None:
                result[sym] = {
                    "earnings_date": earnings_date_val,
                    "reporting_time": "UNKNOWN",
                    "days_until_earnings": (earnings_date_val - today).days,
                }
        except Exception as e:
            logger.debug(f"[EarningsData] Calendar fetch failed for {sym}: {e}")
            continue

    logger.info(
        f"[EarningsData] Fetched earnings calendar for "
        f"{len(result)}/{len(symbols)} symbols"
    )
    return result


def fetch_historical_earnings_real(
    symbol: str, quarters: int = 8
) -> Optional[dict]:
    """
    Compute post-earnings price moves from yfinance data.

    Returns dict matching HistoricalEarnings fields, or None on failure.
    """
    if not HAS_YFINANCE:
        return None

    try:
        ticker = yf.Ticker(symbol)

        # Get earnings dates with EPS data
        earnings_dates = getattr(ticker, "earnings_dates", None)
        if earnings_dates is None or earnings_dates.empty:
            return None

        # Get 2 years of price history
        hist = ticker.history(period="2y")
        if hist is None or hist.empty:
            return None

        moves: list[float] = []
        beats: int = 0
        misses: int = 0

        # Process up to `quarters` most recent earnings
        processed = 0
        for idx in earnings_dates.index:
            if processed >= quarters:
                break

            ed = idx.date() if hasattr(idx, "date") else idx
            if ed >= date.today():
                continue  # Skip future earnings

            # Find close on earnings date and next trading day
            try:
                # Find the closest trading day on or before earnings
                close_dates = hist.index[hist.index.date <= ed]
                if len(close_dates) < 1:
                    continue
                earnings_day_close = float(hist.loc[close_dates[-1], "Close"])

                # Find next trading day
                after_dates = hist.index[hist.index.date > ed]
                if len(after_dates) < 1:
                    continue
                next_day_close = float(hist.loc[after_dates[0], "Close"])

                move_pct = (next_day_close - earnings_day_close) / earnings_day_close * 100
                moves.append(round(move_pct, 2))

                # Check surprise from earnings_dates columns
                row = earnings_dates.loc[idx]
                surprise = None
                if "Surprise(%)" in row.index:
                    surprise = row["Surprise(%)"]
                elif "EPS Estimate" in row.index and "Reported EPS" in row.index:
                    est = row.get("EPS Estimate")
                    actual = row.get("Reported EPS")
                    if est is not None and actual is not None and est != 0:
                        surprise = (actual - est) / abs(est) * 100

                if surprise is not None and not math.isnan(surprise):
                    if surprise >= 0:
                        beats += 1
                    else:
                        misses += 1
                else:
                    # If no surprise data, classify by price move
                    if move_pct >= 0:
                        beats += 1
                    else:
                        misses += 1

                processed += 1
            except Exception:
                continue

        if processed == 0:
            return None

        # Compute aggregates
        abs_moves = [abs(m) for m in moves]
        up_moves = [m for m in moves if m >= 0]
        down_moves = [m for m in moves if m < 0]

        avg_move = sum(abs_moves) / len(abs_moves) if abs_moves else 0.0
        avg_up = sum(up_moves) / len(up_moves) if up_moves else 0.0
        avg_down = sum(down_moves) / len(down_moves) if down_moves else 0.0
        max_adverse = min(moves) if moves else 0.0
        std_dev = _compute_std_dev(moves)
        total_q = beats + misses

        # Trend: compare last half vs first half
        half = len(moves) // 2
        if half > 0 and len(moves) > 1:
            recent_avg = sum(abs(m) for m in moves[:half]) / half
            older_avg = sum(abs(m) for m in moves[half:]) / (len(moves) - half)
            if recent_avg > older_avg * 1.1:
                trend = "improving"
            elif recent_avg < older_avg * 0.9:
                trend = "declining"
            else:
                trend = "consistent"
        else:
            trend = "consistent"

        return {
            "quarters_analyzed": total_q,
            "beat_count": beats,
            "miss_count": misses,
            "beat_rate_pct": round(beats / total_q * 100, 1) if total_q > 0 else 0.0,
            "avg_move_pct": round(avg_move, 1),
            "avg_gap_up_pct": round(avg_up, 1),
            "avg_gap_down_pct": round(avg_down, 1),
            "max_adverse_move_pct": round(max_adverse, 1),
            "move_std_dev": round(std_dev, 1),
            "recent_trend": trend,
        }

    except Exception as e:
        logger.debug(f"[EarningsData] Historical fetch failed for {symbol}: {e}")
        return None


def _compute_std_dev(values: list[float]) -> float:
    """Compute sample standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Mock Data (deterministic, seed=42)
# ---------------------------------------------------------------------------

# Known stocks with realistic mock earnings data
MOCK_EARNINGS_DATA: dict[str, dict] = {
    "NVDA": {
        "days_until": 14, "time": "AFTER_CLOSE",
        "beat_count": 7, "miss_count": 1, "avg_move": 8.0,
        "avg_up": 10.5, "avg_down": -12.0, "max_adverse": -15.0,
        "std_dev": 4.2, "trend": "consistent",
    },
    "CRM": {
        "days_until": 10, "time": "AFTER_CLOSE",
        "beat_count": 5, "miss_count": 3, "avg_move": 12.0,
        "avg_up": 9.0, "avg_down": -16.0, "max_adverse": -22.0,
        "std_dev": 5.5, "trend": "consistent",
    },
    "PANW": {
        "days_until": 18, "time": "AFTER_CLOSE",
        "beat_count": 6, "miss_count": 2, "avg_move": 9.0,
        "avg_up": 8.0, "avg_down": -11.0, "max_adverse": -16.0,
        "std_dev": 3.8, "trend": "improving",
    },
    "AXON": {
        "days_until": 12, "time": "AFTER_CLOSE",
        "beat_count": 8, "miss_count": 0, "avg_move": 10.0,
        "avg_up": 10.0, "avg_down": -5.0, "max_adverse": -5.0,
        "std_dev": 3.0, "trend": "improving",
    },
    "GOOGL": {
        "days_until": 18, "time": "AFTER_CLOSE",
        "beat_count": 7, "miss_count": 1, "avg_move": 6.0,
        "avg_up": 7.0, "avg_down": -8.0, "max_adverse": -10.0,
        "std_dev": 3.0, "trend": "improving",
    },
    "AAPL": {
        "days_until": 7, "time": "AFTER_CLOSE",
        "beat_count": 7, "miss_count": 1, "avg_move": 5.0,
        "avg_up": 4.0, "avg_down": -6.0, "max_adverse": -8.0,
        "std_dev": 2.5, "trend": "consistent",
    },
    "MSFT": {
        "days_until": 16, "time": "AFTER_CLOSE",
        "beat_count": 8, "miss_count": 0, "avg_move": 5.0,
        "avg_up": 5.5, "avg_down": -3.0, "max_adverse": -5.0,
        "std_dev": 2.0, "trend": "consistent",
    },
    "META": {
        "days_until": 15, "time": "AFTER_CLOSE",
        "beat_count": 6, "miss_count": 2, "avg_move": 10.0,
        "avg_up": 12.0, "avg_down": -14.0, "max_adverse": -25.0,
        "std_dev": 7.0, "trend": "consistent",
    },
    "AMZN": {
        "days_until": 19, "time": "AFTER_CLOSE",
        "beat_count": 7, "miss_count": 1, "avg_move": 7.0,
        "avg_up": 8.0, "avg_down": -10.0, "max_adverse": -12.0,
        "std_dev": 3.5, "trend": "improving",
    },
    "NOW": {
        "days_until": 12, "time": "AFTER_CLOSE",
        "beat_count": 7, "miss_count": 1, "avg_move": 10.0,
        "avg_up": 11.0, "avg_down": -14.0, "max_adverse": -16.0,
        "std_dev": 4.5, "trend": "consistent",
    },
}

# Additional mock entries for stocks that report but with longer timelines
MOCK_NO_EARNINGS_SYMBOLS: set[str] = {
    "GLD", "SCHD", "VEA", "ACWX", "XBI", "ITA", "PPA", "XLP",
    "URA", "TAN", "ILF", "COPX", "EWY", "IWN", "AVDV", "AVEM",
}


def fetch_earnings_calendar_mock(symbols: list[str]) -> dict[str, dict]:
    """Deterministic mock earnings calendar."""
    today = date.today()
    result: dict[str, dict] = {}
    rng = random.Random(42)

    for sym in symbols:
        if sym in MOCK_NO_EARNINGS_SYMBOLS:
            continue  # ETFs/funds don't report earnings

        if sym in MOCK_EARNINGS_DATA:
            data = MOCK_EARNINGS_DATA[sym]
            earnings_date = today + timedelta(days=data["days_until"])
            result[sym] = {
                "earnings_date": earnings_date,
                "reporting_time": data["time"],
                "days_until_earnings": data["days_until"],
            }
        else:
            # Generate mock: ~40% within 21 days, ~60% outside
            days = rng.randint(3, 60)
            if days <= LOOKFORWARD_DAYS:
                earnings_date = today + timedelta(days=days)
                result[sym] = {
                    "earnings_date": earnings_date,
                    "reporting_time": rng.choice(["BEFORE_OPEN", "AFTER_CLOSE"]),
                    "days_until_earnings": days,
                }
            # else: no earnings within window

    return result


def fetch_historical_earnings_mock(
    symbol: str, quarters: int = 8
) -> dict:
    """Deterministic mock historical earnings data."""
    if symbol in MOCK_EARNINGS_DATA:
        data = MOCK_EARNINGS_DATA[symbol]
        return {
            "quarters_analyzed": data["beat_count"] + data["miss_count"],
            "beat_count": data["beat_count"],
            "miss_count": data["miss_count"],
            "beat_rate_pct": round(
                data["beat_count"] / (data["beat_count"] + data["miss_count"]) * 100, 1
            ),
            "avg_move_pct": data["avg_move"],
            "avg_gap_up_pct": data["avg_up"],
            "avg_gap_down_pct": data["avg_down"],
            "max_adverse_move_pct": data["max_adverse"],
            "move_std_dev": data["std_dev"],
            "recent_trend": data["trend"],
        }

    # Generate from symbol hash for deterministic results
    rng = random.Random(hash(symbol) + 42)
    beat = rng.randint(3, 8)
    miss = 8 - beat
    avg_move = round(rng.uniform(5.0, 18.0), 1)
    avg_up = round(avg_move * rng.uniform(0.8, 1.3), 1)
    avg_down = round(-avg_move * rng.uniform(1.0, 1.8), 1)
    max_adverse = round(avg_down * rng.uniform(1.0, 1.5), 1)
    std_dev = round(avg_move * rng.uniform(0.3, 0.6), 1)
    trend = rng.choice(["improving", "declining", "consistent"])

    return {
        "quarters_analyzed": 8,
        "beat_count": beat,
        "miss_count": miss,
        "beat_rate_pct": round(beat / 8 * 100, 1),
        "avg_move_pct": avg_move,
        "avg_gap_up_pct": avg_up,
        "avg_gap_down_pct": avg_down,
        "max_adverse_move_pct": max_adverse,
        "move_std_dev": std_dev,
        "recent_trend": trend,
    }


# ---------------------------------------------------------------------------
# Pure Computation Functions
# ---------------------------------------------------------------------------

def compute_cushion_ratio(
    gain_loss_pct: float, avg_move_pct: float
) -> tuple[float, CushionCategory]:
    """
    MUST-M3: Compute cushion ratio and classify category.

    Cushion Ratio = Current Gain % / Average Earnings Move %
    """
    if avg_move_pct <= 0:
        # No meaningful earnings move data
        if gain_loss_pct > 0:
            return (999.0, CushionCategory.COMFORTABLE)
        return (0.0, CushionCategory.INSUFFICIENT)

    ratio = gain_loss_pct / avg_move_pct

    if ratio > CUSHION_COMFORTABLE:
        category = CushionCategory.COMFORTABLE
    elif ratio >= CUSHION_MANAGEABLE_LOW:
        category = CushionCategory.MANAGEABLE
    elif ratio >= CUSHION_THIN_LOW:
        category = CushionCategory.THIN
    else:
        category = CushionCategory.INSUFFICIENT

    return (round(ratio, 2), category)


def classify_risk_level(
    cushion_ratio: float,
    cushion_category: CushionCategory,
    beat_rate_pct: float,
    portfolio_pct: float,
    regime: str,
    estimate_revision: EstimateRevision,
    implied_vol_signal: ImpliedVolSignal,
    quarters_analyzed: int,
) -> tuple[EarningsRisk, list[str]]:
    """
    MUST-Q2 + MUST_NOT-S3: Classify earnings risk with risk factors.

    Base classification from cushion:
    - LOW: ratio > 2.0, beat_rate > 75%, healthy market
    - MODERATE: ratio 1.0-2.0, or mixed signals
    - HIGH: ratio 0.5-1.0, or large position with moderate cushion
    - CRITICAL: ratio < 0.5, or large position (>10%) with ratio < 1.0
    """
    risk_factors: list[str] = []

    # Start from cushion-based level
    if cushion_category == CushionCategory.COMFORTABLE:
        level = EarningsRisk.LOW
        risk_factors.append(f"Comfortable cushion (ratio {cushion_ratio:.1f})")
    elif cushion_category == CushionCategory.MANAGEABLE:
        level = EarningsRisk.MODERATE
        risk_factors.append(f"Manageable cushion (ratio {cushion_ratio:.1f})")
    elif cushion_category == CushionCategory.THIN:
        level = EarningsRisk.HIGH
        risk_factors.append(f"Thin cushion (ratio {cushion_ratio:.1f})")
    else:
        level = EarningsRisk.CRITICAL
        risk_factors.append(f"Insufficient cushion (ratio {cushion_ratio:.1f})")

    # Amplify by position size (MUST_NOT-S3)
    if portfolio_pct > LARGE_POSITION_PCT:
        risk_factors.append(
            f"Large position size ({portfolio_pct:.1f}% of portfolio)"
        )
        if level == EarningsRisk.LOW:
            level = EarningsRisk.MODERATE
        elif level == EarningsRisk.MODERATE:
            level = EarningsRisk.HIGH

    # Amplify by insufficient history (MUST-Q4)
    if quarters_analyzed < MIN_CONFIDENT_QUARTERS:
        risk_factors.append(
            f"Insufficient history ({quarters_analyzed} quarters, "
            f"need {MIN_CONFIDENT_QUARTERS}+)"
        )
        if level in (EarningsRisk.LOW, EarningsRisk.MODERATE):
            level = EarningsRisk.HIGH

    # Beat rate factor
    if beat_rate_pct < 50.0:
        risk_factors.append(f"Low beat rate ({beat_rate_pct:.0f}%)")
        if level == EarningsRisk.LOW:
            level = EarningsRisk.MODERATE
    elif beat_rate_pct >= 87.5:
        risk_factors.append(f"Strong beat rate ({beat_rate_pct:.0f}%)")

    # Estimate revision factor
    if estimate_revision == EstimateRevision.NEGATIVE:
        risk_factors.append("Estimates revised down")
        if level == EarningsRisk.LOW:
            level = EarningsRisk.MODERATE
        elif level == EarningsRisk.MODERATE:
            level = EarningsRisk.HIGH
    elif estimate_revision == EstimateRevision.POSITIVE:
        risk_factors.append("Estimates revised up")

    # Implied vol factor
    if implied_vol_signal == ImpliedVolSignal.ELEVATED_EXPECTATIONS:
        risk_factors.append("Elevated implied volatility")
        if level == EarningsRisk.LOW:
            level = EarningsRisk.MODERATE

    # Regime factor
    if regime in WEAK_REGIMES:
        risk_factors.append(f"Weak market regime ({regime})")
        if level == EarningsRisk.MODERATE:
            level = EarningsRisk.HIGH

    return (level, risk_factors)


# ---------------------------------------------------------------------------
# Strategy Decision Framework
# ---------------------------------------------------------------------------

# Base strategy matrix (regime x cushion) from spec Section 7
_BASE_STRATEGY_MATRIX: dict[tuple[str, str], StrategyType] = {
    ("CORRECTION", "INSUFFICIENT"): StrategyType.EXIT_BEFORE_EARNINGS,
    ("CORRECTION", "THIN"): StrategyType.EXIT_BEFORE_EARNINGS,
    ("CORRECTION", "MANAGEABLE"): StrategyType.TRIM_TO_HALF,
    ("CORRECTION", "COMFORTABLE"): StrategyType.TRIM_TO_HALF,

    ("UPTREND_UNDER_PRESSURE", "INSUFFICIENT"): StrategyType.EXIT_BEFORE_EARNINGS,
    ("UPTREND_UNDER_PRESSURE", "THIN"): StrategyType.TRIM_TO_HALF,
    ("UPTREND_UNDER_PRESSURE", "MANAGEABLE"): StrategyType.TRIM_TO_HALF,
    ("UPTREND_UNDER_PRESSURE", "COMFORTABLE"): StrategyType.HOLD_FULL,

    ("CONFIRMED_UPTREND", "INSUFFICIENT"): StrategyType.TRIM_TO_HALF,
    ("CONFIRMED_UPTREND", "THIN"): StrategyType.TRIM_TO_HALF,
    ("CONFIRMED_UPTREND", "MANAGEABLE"): StrategyType.HOLD_FULL,
    ("CONFIRMED_UPTREND", "COMFORTABLE"): StrategyType.HOLD_FULL,
}

# Map 5-state regimes to the 3 strategy matrix keys
_REGIME_MAP: dict[str, str] = {
    "CONFIRMED_UPTREND": "CONFIRMED_UPTREND",
    "UPTREND_UNDER_PRESSURE": "UPTREND_UNDER_PRESSURE",
    "CORRECTION": "CORRECTION",
    "RALLY_ATTEMPT": "CORRECTION",
    "FOLLOW_THROUGH_DAY": "UPTREND_UNDER_PRESSURE",
}


def determine_base_strategy(
    cushion_category: CushionCategory,
    regime: str,
) -> StrategyType:
    """Implement the Section 7 decision matrix."""
    mapped_regime = _REGIME_MAP.get(regime, "CONFIRMED_UPTREND")
    key = (mapped_regime, cushion_category.value)
    return _BASE_STRATEGY_MATRIX.get(key, StrategyType.HOLD_FULL)


def _shift_strategy(strategy: StrategyType, delta: int) -> StrategyType:
    """Shift strategy by delta on the aggressiveness ladder."""
    try:
        idx = STRATEGY_LADDER.index(strategy.value)
    except ValueError:
        return strategy
    new_idx = max(0, min(len(STRATEGY_LADDER) - 1, idx + delta))
    return StrategyType(STRATEGY_LADDER[new_idx])


def apply_strategy_modifiers(
    base_strategy: StrategyType,
    portfolio_pct: float,
    beat_rate_pct: float,
    estimate_revision: EstimateRevision,
    implied_vol_signal: ImpliedVolSignal,
    implied_move_pct: Optional[float],
    avg_move_pct: float,
    same_week_count: int,
) -> StrategyType:
    """Apply modifiers from spec Section 7 that shift ±1 level."""
    delta = 0

    # Conservative modifiers (-1 each)
    if portfolio_pct > LARGE_POSITION_PCT:
        delta -= 1
    if beat_rate_pct < 50.0:
        delta -= 1
    if estimate_revision == EstimateRevision.NEGATIVE:
        delta -= 1
    if implied_vol_signal == ImpliedVolSignal.ELEVATED_EXPECTATIONS:
        delta -= 1
    if same_week_count >= 3:
        delta -= 1

    # Aggressive modifiers (+1 each)
    if beat_rate_pct >= 87.5:
        delta += 1
    if estimate_revision == EstimateRevision.POSITIVE:
        delta += 1
    if (
        implied_move_pct is not None
        and avg_move_pct > 0
        and implied_move_pct < avg_move_pct
    ):
        delta += 1
    if portfolio_pct < 3.0:
        delta += 1

    return _shift_strategy(base_strategy, delta)


def build_scenario_outcomes(
    shares: int,
    current_price: float,
    buy_price: float,
    gain_loss_pct: float,
    move_pcts: tuple[float, float, float],
    position_fraction: float = 1.0,
) -> list[ScenarioOutcome]:
    """
    MUST-Q3, MUST_NOT-S4: Build 3 scenarios with math strings.

    move_pcts is (best_move, base_move, worst_move) as percentages.
    position_fraction: 1.0 for full, 0.5 for half, 0.25 for quarter.
    """
    labels = ("BEST", "BASE", "WORST")
    active_shares = max(1, int(shares * position_fraction))
    outcomes: list[ScenarioOutcome] = []

    for label, move_pct in zip(labels, move_pcts):
        price_move = current_price * move_pct / 100
        dollar_impact = round(active_shares * price_move, 2)
        new_price = current_price + price_move
        new_gain_pct = round((new_price - buy_price) / buy_price * 100, 1)

        # Build math string (MUST_NOT-S4)
        math_str = (
            f"{active_shares} shares x ${abs(price_move):.2f} "
            f"{'gain' if price_move >= 0 else 'loss'} = "
            f"{'$' if dollar_impact >= 0 else '-$'}{abs(dollar_impact):,.2f}"
        )

        outcomes.append(ScenarioOutcome(
            scenario=label,
            expected_move_pct=round(move_pct, 1),
            resulting_gain_loss_pct=new_gain_pct,
            dollar_impact=dollar_impact,
            math=math_str,
        ))

    return outcomes


def build_strategy_options(
    shares: int,
    current_price: float,
    buy_price: float,
    gain_loss_pct: float,
    portfolio_pct: float,
    historical: HistoricalEarnings,
    cushion_ratio: float,
    cushion_category: CushionCategory,
    regime: str,
    estimate_revision: EstimateRevision,
    implied_vol_signal: ImpliedVolSignal,
    implied_move_pct: Optional[float],
    risk_level: EarningsRisk,
    same_week_count: int,
) -> tuple[list[StrategyOption], StrategyType, str]:
    """
    Build >= 2 strategy options, recommend one, provide rationale.

    Returns (strategies, recommended_strategy, recommendation_rationale).
    Enforces MUST_NOT-S1 and MUST_NOT-S2.
    """
    strategies: list[StrategyOption] = []

    # Move percentages for scenarios
    best_move = historical.avg_gap_up_pct
    base_move = -historical.avg_move_pct  # Base = average move against
    worst_move = historical.max_adverse_move_pct

    is_weak_market = regime in WEAK_REGIMES
    insufficient = cushion_category == CushionCategory.INSUFFICIENT
    thin_or_less = cushion_ratio < CUSHION_THIN_LOW

    # --- Strategy A: Aggressive (Hold Full or Hold-and-Add) ---
    can_hold_full = not (thin_or_less and is_weak_market)  # MUST_NOT-S1

    # Check HOLD_AND_ADD conditions (MUST_NOT-S2)
    can_add = (
        historical.beat_rate_pct >= ADD_MIN_BEAT_RATE
        and cushion_ratio >= ADD_MIN_CUSHION_RATIO
        and regime == "CONFIRMED_UPTREND"
        and portfolio_pct < ADD_MAX_POSITION_PCT
    )

    if can_add:
        scenarios = build_scenario_outcomes(
            shares, current_price, buy_price, gain_loss_pct,
            (best_move, base_move, worst_move), 1.0,
        )
        best_dollar = scenarios[0].dollar_impact
        worst_dollar = scenarios[2].dollar_impact
        strategies.append(StrategyOption(
            strategy=StrategyType.HOLD_AND_ADD,
            description=(
                f"Hold all {shares} shares and consider adding before earnings. "
                f"Beat rate of {historical.beat_rate_pct:.0f}% and strong cushion support this."
            ),
            shares_to_sell=None,
            scenarios=scenarios,
            risk_reward_summary=(
                f"Risk ${abs(worst_dollar):,.0f} to make ${abs(best_dollar):,.0f}"
            ),
        ))

    if can_hold_full:
        scenarios = build_scenario_outcomes(
            shares, current_price, buy_price, gain_loss_pct,
            (best_move, base_move, worst_move), 1.0,
        )
        best_dollar = scenarios[0].dollar_impact
        worst_dollar = scenarios[2].dollar_impact
        strategies.append(StrategyOption(
            strategy=StrategyType.HOLD_FULL,
            description=(
                f"Hold all {shares} shares through earnings. "
                f"Full exposure to both upside and downside."
            ),
            shares_to_sell=None,
            scenarios=scenarios,
            risk_reward_summary=(
                f"Risk ${abs(worst_dollar):,.0f} to make ${abs(best_dollar):,.0f}"
            ),
        ))

    # --- Strategy B: Conservative (Trim or Exit) ---
    if insufficient or (thin_or_less and is_weak_market):
        # Exit or trim to quarter
        if risk_level == EarningsRisk.CRITICAL:
            sell_shares = shares
            frac = 0.0
            strat_type = StrategyType.EXIT_BEFORE_EARNINGS
            desc = (
                f"Sell all {shares} shares before earnings. "
                f"Cushion ratio of {cushion_ratio:.1f} is insufficient to absorb a miss."
            )
        else:
            sell_shares = shares - max(1, shares // 4)
            frac = 0.25
            strat_type = StrategyType.TRIM_TO_QUARTER
            hold_shares = max(1, shares // 4)
            desc = (
                f"Sell {sell_shares} shares, keep {hold_shares}. "
                f"Limits downside while maintaining small upside exposure."
            )
    else:
        # Trim to half
        sell_shares = max(1, shares // 2)
        hold_shares = shares - sell_shares
        frac = hold_shares / shares
        strat_type = StrategyType.TRIM_TO_HALF
        desc = (
            f"Sell {sell_shares} shares, hold {hold_shares}. "
            f"Locks partial profit and reduces worst-case impact."
        )

    trim_scenarios = build_scenario_outcomes(
        shares, current_price, buy_price, gain_loss_pct,
        (best_move, base_move, worst_move), frac,
    )
    # Add locked profit to trim scenario dollar impacts
    if sell_shares > 0 and frac > 0:
        locked_profit = sell_shares * (current_price - buy_price)
        for sc in trim_scenarios:
            sc.dollar_impact = round(sc.dollar_impact + locked_profit, 2)
            sc.math += f" + ${locked_profit:,.2f} locked profit"

    best_trim = trim_scenarios[0].dollar_impact
    worst_trim = trim_scenarios[2].dollar_impact
    strategies.append(StrategyOption(
        strategy=strat_type,
        description=desc,
        shares_to_sell=sell_shares,
        scenarios=trim_scenarios,
        risk_reward_summary=(
            f"Risk ${abs(worst_trim):,.0f} to make ${abs(best_trim):,.0f}"
        ),
    ))

    # --- Strategy C: Hedge (optional) ---
    if can_hold_full and risk_level in (EarningsRisk.MODERATE, EarningsRisk.HIGH):
        # Estimate hedge cost as ~3% of position value
        position_value = shares * current_price
        hedge_cost = round(position_value * 0.03, 2)

        hedge_scenarios = build_scenario_outcomes(
            shares, current_price, buy_price, gain_loss_pct,
            (best_move, base_move, worst_move), 1.0,
        )
        # Adjust for hedge cost
        for sc in hedge_scenarios:
            sc.dollar_impact = round(sc.dollar_impact - hedge_cost, 2)
            if sc.scenario == "WORST":
                # Put limits loss
                max_loss = position_value * 0.05 + hedge_cost  # ~5% loss + premium
                if abs(sc.dollar_impact) > max_loss:
                    sc.dollar_impact = round(-max_loss, 2)
                sc.math += f" (put limits loss, cost ${hedge_cost:,.2f})"

        best_hedge = hedge_scenarios[0].dollar_impact
        worst_hedge = hedge_scenarios[2].dollar_impact
        strategies.append(StrategyOption(
            strategy=StrategyType.HEDGE_WITH_PUT,
            description=(
                f"Hold all {shares} shares, buy protective put (~${hedge_cost:,.0f}). "
                f"Limits downside while keeping upside."
            ),
            shares_to_sell=None,
            estimated_hedge_cost=hedge_cost,
            scenarios=hedge_scenarios,
            risk_reward_summary=(
                f"Max risk ${abs(worst_hedge):,.0f} (capped), "
                f"max gain ${abs(best_hedge):,.0f}"
            ),
        ))

    # Ensure minimum 2 strategies
    if len(strategies) < 2:
        # Add exit as backup
        exit_scenarios = build_scenario_outcomes(
            shares, current_price, buy_price, gain_loss_pct,
            (0.0, 0.0, 0.0), 0.0,
        )
        locked = shares * (current_price - buy_price)
        for sc in exit_scenarios:
            sc.dollar_impact = round(locked, 2)
            sc.math = f"Lock in {'$' if locked >= 0 else '-$'}{abs(locked):,.2f} by selling all {shares} shares"

        strategies.append(StrategyOption(
            strategy=StrategyType.EXIT_BEFORE_EARNINGS,
            description=(
                f"Sell all {shares} shares before earnings. "
                f"Eliminates binary event risk entirely."
            ),
            shares_to_sell=shares,
            scenarios=exit_scenarios,
            risk_reward_summary=(
                f"Lock {'$' if locked >= 0 else '-$'}{abs(locked):,.2f}, no further risk"
            ),
        ))

    # --- Determine recommended strategy ---
    base = determine_base_strategy(cushion_category, regime)
    recommended = apply_strategy_modifiers(
        base, portfolio_pct, historical.beat_rate_pct,
        estimate_revision, implied_vol_signal, implied_move_pct,
        historical.avg_move_pct, same_week_count,
    )

    # Enforce MUST_NOT-S1
    if thin_or_less and is_weak_market:
        if recommended in (StrategyType.HOLD_FULL, StrategyType.HOLD_AND_ADD):
            recommended = StrategyType.TRIM_TO_HALF

    # Enforce MUST_NOT-S2
    if recommended == StrategyType.HOLD_AND_ADD and not can_add:
        recommended = StrategyType.HOLD_FULL

    # Ensure recommended is in strategies list
    available = {s.strategy for s in strategies}
    if recommended not in available:
        # Pick closest available from ladder
        rec_idx = STRATEGY_LADDER.index(recommended.value)
        best_dist = 999
        for s in strategies:
            s_idx = STRATEGY_LADDER.index(s.strategy.value)
            dist = abs(s_idx - rec_idx)
            if dist < best_dist:
                best_dist = dist
                recommended = s.strategy

    # Build rationale (MUST_NOT-S6: no predictions)
    rationale = _build_rationale(
        recommended, cushion_ratio, cushion_category,
        historical, regime, portfolio_pct, estimate_revision,
    )

    return (strategies, recommended, rationale)


def _build_rationale(
    strategy: StrategyType,
    cushion_ratio: float,
    cushion_category: CushionCategory,
    historical: HistoricalEarnings,
    regime: str,
    portfolio_pct: float,
    estimate_revision: EstimateRevision,
) -> str:
    """Build recommendation rationale without prediction language."""
    parts: list[str] = []

    # Cushion context
    if cushion_category == CushionCategory.COMFORTABLE:
        parts.append(
            f"Cushion ratio of {cushion_ratio:.1f} provides substantial buffer — "
            f"the position can absorb an average adverse move and remain profitable."
        )
    elif cushion_category == CushionCategory.MANAGEABLE:
        parts.append(
            f"Cushion ratio of {cushion_ratio:.1f} is adequate for an average move "
            f"but does not cover the worst historical case of "
            f"{historical.max_adverse_move_pct:.0f}%."
        )
    elif cushion_category == CushionCategory.THIN:
        parts.append(
            f"Cushion ratio of {cushion_ratio:.1f} means an average earnings move "
            f"would significantly erode the gain. Risk/reward of holding full is unfavorable."
        )
    else:
        parts.append(
            f"Cushion ratio of {cushion_ratio:.1f} is insufficient — even a small "
            f"adverse move would turn this position into a loss."
        )

    # Strategy-specific context
    if strategy in (StrategyType.TRIM_TO_HALF, StrategyType.TRIM_TO_QUARTER):
        parts.append(
            "Trimming locks partial profit and reduces worst-case portfolio impact."
        )
    elif strategy == StrategyType.EXIT_BEFORE_EARNINGS:
        parts.append(
            "Exiting eliminates binary event risk and preserves capital."
        )
    elif strategy == StrategyType.HOLD_FULL:
        beat_str = f"Beat rate of {historical.beat_rate_pct:.0f}% over {historical.quarters_analyzed} quarters"
        parts.append(f"{beat_str} and adequate cushion support holding through earnings.")
    elif strategy == StrategyType.HOLD_AND_ADD:
        parts.append(
            f"Exceptional beat rate ({historical.beat_rate_pct:.0f}%), strong cushion, "
            f"and small position size ({portfolio_pct:.1f}%) make this a candidate for adding."
        )

    # Regime context
    if regime in WEAK_REGIMES:
        parts.append(
            f"Market regime ({regime}) favors conservative positioning around earnings."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Concentration Assessment
# ---------------------------------------------------------------------------

def assess_concentration(
    analyses: list[dict],
    lookforward_days: int = LOOKFORWARD_DAYS,
) -> PortfolioEarningsConcentration:
    """
    MUST-M7, MUST-Q5: Group by week, compute aggregate %, flag concentration.

    analyses: list of {ticker, earnings_date (date), portfolio_pct}
    """
    if not analyses:
        return PortfolioEarningsConcentration(
            total_positions_approaching=0,
            total_portfolio_pct_exposed=0.0,
            earnings_calendar=[],
            concentration_risk="LOW",
        )

    # Group by week (Monday of the week)
    weeks: dict[date, list[dict]] = {}
    for a in analyses:
        ed = a["earnings_date"]
        if isinstance(ed, str):
            ed = date.fromisoformat(ed)
        # Get Monday of that week
        monday = ed - timedelta(days=ed.weekday())
        weeks.setdefault(monday, []).append(a)

    # Build EarningsWeek objects
    calendar: list[EarningsWeek] = []
    max_week_pct = 0.0

    for monday in sorted(weeks.keys()):
        positions = weeks[monday]
        tickers = [p["ticker"] for p in positions]
        agg_pct = round(sum(p["portfolio_pct"] for p in positions), 1)
        max_week_pct = max(max_week_pct, agg_pct)

        # Concentration flag
        flag = None
        if agg_pct >= CONCENTRATION_CRITICAL_PCT:
            flag = "CRITICAL"
        elif agg_pct >= CONCENTRATION_MODERATE_PCT:
            flag = "CONCENTRATED"

        # Week label
        month_name = monday.strftime("%b")
        week_label = f"Week of {month_name} {monday.day}"

        calendar.append(EarningsWeek(
            week_start=monday.isoformat(),
            week_label=week_label,
            positions_reporting=tickers,
            aggregate_portfolio_pct=agg_pct,
            concentration_flag=flag,
        ))

    total_pct = round(sum(a["portfolio_pct"] for a in analyses), 1)

    # Overall concentration risk
    if max_week_pct >= CONCENTRATION_CRITICAL_PCT:
        risk = "CRITICAL"
        rec = (
            f"Over {max_week_pct:.0f}% of portfolio has earnings in one week. "
            f"Consider reducing exposure in at least some positions to limit "
            f"correlated binary event risk."
        )
    elif max_week_pct >= CONCENTRATION_MODERATE_PCT:
        risk = "HIGH"
        rec = (
            f"{max_week_pct:.0f}% of portfolio has earnings in one week. "
            f"Monitor closely and consider trimming overlapping positions."
        )
    elif len(analyses) >= 5:
        risk = "MODERATE"
        rec = None
    else:
        risk = "LOW"
        rec = None

    return PortfolioEarningsConcentration(
        total_positions_approaching=len(analyses),
        total_portfolio_pct_exposed=total_pct,
        earnings_calendar=calendar,
        concentration_risk=risk,
        concentration_recommendation=rec,
    )


# ---------------------------------------------------------------------------
# LLM Enrichment (estimate revisions)
# ---------------------------------------------------------------------------

_ESTIMATE_REVISION_PROMPT = """\
For each stock below, assess whether analyst EPS estimates have been revised
in the last 30 days. Return a JSON array with one object per stock.
Use ONLY valid JSON — no markdown, no code fences, no extra text.

Fields per stock:
- symbol: str (ticker, uppercase)
- revision: str (one of: "POSITIVE", "NEUTRAL", "NEGATIVE")
- detail: str (1 sentence, e.g., "EPS estimate revised from $1.85 to $1.92 (+3.8%)")

If you don't have specific data, use "NEUTRAL" with detail "No recent revision data available."

Stocks:
{stock_list}
"""


def enrich_estimate_revisions_llm(
    symbols: list[str],
    model: str = "claude-haiku-4-5-20251001",
    batch_size: int = 30,
) -> dict[str, dict]:
    """
    Look up analyst estimate revisions via Claude Haiku.

    Returns {symbol: {revision: str, detail: str}}.
    Empty dict on any failure (graceful fallback).
    """
    if not symbols:
        return {}

    result: dict[str, dict] = {}

    try:
        import anthropic
        import json
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("[EarningsData] No ANTHROPIC_API_KEY — skipping LLM enrichment")
            return {}

        client = anthropic.Anthropic(api_key=api_key)

        from ibd_agents.tools.token_tracker import track as track_tokens

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            stock_list = "\n".join(f"- {sym}" for sym in batch)
            prompt = _ESTIMATE_REVISION_PROMPT.format(stock_list=stock_list)

            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                )
                track_tokens("estimate_revisions", response)

                text = response.content[0].text.strip()
                # Clean markdown fences if present
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

                data = json.loads(text)
                if isinstance(data, list):
                    for item in data:
                        sym = item.get("symbol", "").upper()
                        if sym in batch:
                            result[sym] = {
                                "revision": item.get("revision", "NEUTRAL"),
                                "detail": item.get("detail", "No data available"),
                            }
            except Exception as e:
                logger.warning(f"[EarningsData] LLM batch failed: {e}")
                continue

    except ImportError:
        logger.warning("[EarningsData] anthropic SDK not installed — skipping LLM enrichment")
    except Exception as e:
        logger.warning(f"[EarningsData] LLM enrichment error: {e}")

    return result


# ---------------------------------------------------------------------------
# CrewAI Tool Wrappers
# ---------------------------------------------------------------------------

class EarningsCalendarInput(BaseModel):
    symbols_json: str = Field(..., description="JSON list of ticker symbols")


class EarningsCalendarTool(BaseTool):
    """Get next earnings dates for portfolio positions."""

    name: str = "earnings_calendar"
    description: str = (
        "Get the next scheduled earnings report date for multiple stocks. "
        "Returns earnings date, reporting time, and days until earnings."
    )
    args_schema: type[BaseModel] = EarningsCalendarInput

    def _run(self, symbols_json: str) -> str:
        import json
        symbols = json.loads(symbols_json)
        if HAS_YFINANCE:
            data = fetch_earnings_calendar_real(symbols)
        else:
            data = fetch_earnings_calendar_mock(symbols)
        # Convert dates to strings for JSON
        for sym in data:
            if isinstance(data[sym].get("earnings_date"), date):
                data[sym]["earnings_date"] = data[sym]["earnings_date"].isoformat()
        return json.dumps(data)


class HistoricalEarningsInput(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol")
    quarters: int = Field(default=8, description="Number of quarters to analyze")


class HistoricalEarningsTool(BaseTool):
    """Get historical post-earnings price reactions."""

    name: str = "historical_earnings_reactions"
    description: str = (
        "Get historical post-earnings stock price reactions for the last N quarters. "
        "Returns beat/miss counts, average moves, worst case, and consistency."
    )
    args_schema: type[BaseModel] = HistoricalEarningsInput

    def _run(self, symbol: str, quarters: int = 8) -> str:
        import json
        if HAS_YFINANCE:
            data = fetch_historical_earnings_real(symbol, quarters)
        else:
            data = None
        if data is None:
            data = fetch_historical_earnings_mock(symbol, quarters)
        return json.dumps(data)
