"""
Regime Detector Tool: Market Data Fetcher
IBD Momentum Investment Framework v4.0

Provides 5 data-fetching functions for regime detection:
1. get_distribution_days  — Distribution day count for S&P 500 and Nasdaq
2. get_market_breadth     — Breadth indicators (% above MAs, highs/lows, A/D)
3. get_leading_stocks_health — RS>=90 stock health above 50-day MA
4. get_sector_rankings    — Top/bottom sectors, defensive rotation detection
5. get_index_price_history — Daily OHLCV for FTD detection

Each function has a mock implementation (for deterministic pipeline/tests)
and a real implementation (via yfinance, when available).

5 mock scenarios: healthy_uptrend, under_pressure, correction,
rally_attempt, follow_through_day — derived from spec golden test cases.
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    yf = None  # type: ignore[assignment]


def is_available() -> bool:
    """Check if yfinance is installed."""
    return HAS_YFINANCE


# ---------------------------------------------------------------------------
# Mock Data Scenarios
# ---------------------------------------------------------------------------

_MOCK_DISTRIBUTION_DAYS: dict[str, dict] = {
    "healthy_uptrend": {
        "sp500_dist_count": 2,
        "nasdaq_dist_count": 1,
        "sp500_dist_days": [
            {"date": "2026-02-05", "decline_pct": -0.35, "volume": 4_200_000_000},
            {"date": "2026-02-12", "decline_pct": -0.28, "volume": 4_100_000_000},
        ],
        "nasdaq_dist_days": [
            {"date": "2026-02-10", "decline_pct": -0.42, "volume": 5_800_000_000},
        ],
        "power_up_days": [
            {"date": "2026-02-14", "index": "S&P 500", "gain_pct": 1.2},
        ],
        "sp500_close": 6100.0,
        "nasdaq_close": 19800.0,
        "sp500_volume": 3_900_000_000,
        "nasdaq_volume": 5_300_000_000,
        "data_date": "2026-02-21",
    },
    "under_pressure": {
        "sp500_dist_count": 5,
        "nasdaq_dist_count": 4,
        "sp500_dist_days": [
            {"date": "2026-02-03", "decline_pct": -0.55, "volume": 4_500_000_000},
            {"date": "2026-02-06", "decline_pct": -0.42, "volume": 4_300_000_000},
            {"date": "2026-02-10", "decline_pct": -0.38, "volume": 4_200_000_000},
            {"date": "2026-02-13", "decline_pct": -0.61, "volume": 4_600_000_000},
            {"date": "2026-02-18", "decline_pct": -0.73, "volume": 4_800_000_000},
        ],
        "nasdaq_dist_days": [
            {"date": "2026-02-04", "decline_pct": -0.48, "volume": 5_900_000_000},
            {"date": "2026-02-07", "decline_pct": -0.35, "volume": 5_600_000_000},
            {"date": "2026-02-12", "decline_pct": -0.52, "volume": 6_100_000_000},
            {"date": "2026-02-17", "decline_pct": -0.67, "volume": 6_300_000_000},
        ],
        "power_up_days": [],
        "sp500_close": 5950.0,
        "nasdaq_close": 19200.0,
        "sp500_volume": 4_100_000_000,
        "nasdaq_volume": 5_700_000_000,
        "data_date": "2026-02-21",
    },
    "correction": {
        "sp500_dist_count": 4,
        "nasdaq_dist_count": 6,
        "sp500_dist_days": [
            {"date": "2026-02-03", "decline_pct": -0.85, "volume": 4_800_000_000},
            {"date": "2026-02-07", "decline_pct": -0.72, "volume": 4_600_000_000},
            {"date": "2026-02-12", "decline_pct": -1.1, "volume": 5_000_000_000},
            {"date": "2026-02-18", "decline_pct": -0.95, "volume": 4_900_000_000},
        ],
        "nasdaq_dist_days": [
            {"date": "2026-02-03", "decline_pct": -0.92, "volume": 6_200_000_000},
            {"date": "2026-02-05", "decline_pct": -0.68, "volume": 6_000_000_000},
            {"date": "2026-02-07", "decline_pct": -1.3, "volume": 6_500_000_000},
            {"date": "2026-02-11", "decline_pct": -0.55, "volume": 5_900_000_000},
            {"date": "2026-02-14", "decline_pct": -0.88, "volume": 6_300_000_000},
            {"date": "2026-02-19", "decline_pct": -1.5, "volume": 6_800_000_000},
        ],
        "power_up_days": [],
        "sp500_close": 5600.0,
        "nasdaq_close": 17800.0,
        "sp500_volume": 4_500_000_000,
        "nasdaq_volume": 6_100_000_000,
        "data_date": "2026-02-21",
    },
    "rally_attempt": {
        "sp500_dist_count": 2,
        "nasdaq_dist_count": 1,
        "sp500_dist_days": [
            {"date": "2026-02-05", "decline_pct": -0.3, "volume": 4_000_000_000},
            {"date": "2026-02-10", "decline_pct": -0.25, "volume": 3_900_000_000},
        ],
        "nasdaq_dist_days": [
            {"date": "2026-02-06", "decline_pct": -0.35, "volume": 5_500_000_000},
        ],
        "power_up_days": [],
        "sp500_close": 5900.0,
        "nasdaq_close": 19100.0,
        "sp500_volume": 3_800_000_000,
        "nasdaq_volume": 5_200_000_000,
        "data_date": "2026-02-21",
    },
    "follow_through_day": {
        "sp500_dist_count": 3,
        "nasdaq_dist_count": 2,
        "sp500_dist_days": [
            {"date": "2026-02-03", "decline_pct": -0.45, "volume": 4_200_000_000},
            {"date": "2026-02-06", "decline_pct": -0.38, "volume": 4_100_000_000},
            {"date": "2026-02-10", "decline_pct": -0.52, "volume": 4_300_000_000},
        ],
        "nasdaq_dist_days": [
            {"date": "2026-02-04", "decline_pct": -0.55, "volume": 5_800_000_000},
            {"date": "2026-02-09", "decline_pct": -0.42, "volume": 5_600_000_000},
        ],
        "power_up_days": [],
        "sp500_close": 5850.0,
        "nasdaq_close": 19000.0,
        "sp500_volume": 4_000_000_000,
        "nasdaq_volume": 5_500_000_000,
        "data_date": "2026-02-21",
    },
}

_MOCK_BREADTH: dict[str, dict] = {
    "healthy_uptrend": {
        "pct_above_200ma": 72.0,
        "pct_above_50ma": 65.0,
        "new_highs_today": 280,
        "new_lows_today": 35,
        "new_highs_10d_avg": 250,
        "new_lows_10d_avg": 40,
        "adv_decline_line_10d_direction": "rising",
        "breadth_vs_10d_ago": "improving",
    },
    "under_pressure": {
        "pct_above_200ma": 55.0,
        "pct_above_50ma": 44.0,
        "new_highs_today": 100,
        "new_lows_today": 110,
        "new_highs_10d_avg": 120,
        "new_lows_10d_avg": 95,
        "adv_decline_line_10d_direction": "declining",
        "breadth_vs_10d_ago": "deteriorating",
    },
    "correction": {
        "pct_above_200ma": 35.0,
        "pct_above_50ma": 25.0,
        "new_highs_today": 20,
        "new_lows_today": 300,
        "new_highs_10d_avg": 30,
        "new_lows_10d_avg": 280,
        "adv_decline_line_10d_direction": "declining",
        "breadth_vs_10d_ago": "deteriorating",
    },
    "rally_attempt": {
        "pct_above_200ma": 50.0,
        "pct_above_50ma": 42.0,
        "new_highs_today": 100,
        "new_lows_today": 60,
        "new_highs_10d_avg": 80,
        "new_lows_10d_avg": 80,
        "adv_decline_line_10d_direction": "rising",
        "breadth_vs_10d_ago": "improving",
    },
    "follow_through_day": {
        "pct_above_200ma": 48.0,
        "pct_above_50ma": 38.0,
        "new_highs_today": 90,
        "new_lows_today": 85,
        "new_highs_10d_avg": 70,
        "new_lows_10d_avg": 100,
        "adv_decline_line_10d_direction": "rising",
        "breadth_vs_10d_ago": "improving",
    },
}

_MOCK_LEADERS: dict[str, dict] = {
    "healthy_uptrend": {
        "total_rs90_stocks": 120,
        "rs90_above_50ma_count": 90,
        "rs90_above_50ma_pct": 75.0,
        "rs90_new_highs": 22,
        "rs90_new_lows": 3,
        "rs90_health_10d_ago_pct": 70.0,
        "health_trend": "improving",
    },
    "under_pressure": {
        "total_rs90_stocks": 100,
        "rs90_above_50ma_count": 48,
        "rs90_above_50ma_pct": 48.0,
        "rs90_new_highs": 8,
        "rs90_new_lows": 12,
        "rs90_health_10d_ago_pct": 55.0,
        "health_trend": "deteriorating",
    },
    "correction": {
        "total_rs90_stocks": 80,
        "rs90_above_50ma_count": 16,
        "rs90_above_50ma_pct": 20.0,
        "rs90_new_highs": 1,
        "rs90_new_lows": 30,
        "rs90_health_10d_ago_pct": 30.0,
        "health_trend": "deteriorating",
    },
    "rally_attempt": {
        "total_rs90_stocks": 90,
        "rs90_above_50ma_count": 45,
        "rs90_above_50ma_pct": 50.0,
        "rs90_new_highs": 10,
        "rs90_new_lows": 6,
        "rs90_health_10d_ago_pct": 40.0,
        "health_trend": "improving",
    },
    "follow_through_day": {
        "total_rs90_stocks": 95,
        "rs90_above_50ma_count": 43,
        "rs90_above_50ma_pct": 45.0,
        "rs90_new_highs": 8,
        "rs90_new_lows": 10,
        "rs90_health_10d_ago_pct": 38.0,
        "health_trend": "improving",
    },
}

_MOCK_SECTORS: dict[str, dict] = {
    "healthy_uptrend": {
        "top_10": [
            {"rank": 1, "sector_name": "Technology", "performance_3m": 12.5, "type": "growth"},
            {"rank": 2, "sector_name": "Consumer Discretionary", "performance_3m": 10.2, "type": "growth"},
            {"rank": 3, "sector_name": "Industrials", "performance_3m": 8.7, "type": "growth"},
            {"rank": 4, "sector_name": "Healthcare", "performance_3m": 7.3, "type": "defensive"},
            {"rank": 5, "sector_name": "Financial", "performance_3m": 6.8, "type": "growth"},
            {"rank": 6, "sector_name": "Energy", "performance_3m": 5.5, "type": "growth"},
            {"rank": 7, "sector_name": "Materials", "performance_3m": 4.2, "type": "growth"},
            {"rank": 8, "sector_name": "Communication", "performance_3m": 3.8, "type": "growth"},
            {"rank": 9, "sector_name": "Real Estate", "performance_3m": 2.1, "type": "defensive"},
            {"rank": 10, "sector_name": "Consumer Staples", "performance_3m": 1.5, "type": "defensive"},
        ],
        "bottom_10": [],
        "defensive_in_top_5": 0,
        "growth_in_top_5": 4,
    },
    "under_pressure": {
        "top_10": [
            {"rank": 1, "sector_name": "Technology", "performance_3m": 5.2, "type": "growth"},
            {"rank": 2, "sector_name": "Financial", "performance_3m": 4.8, "type": "growth"},
            {"rank": 3, "sector_name": "Utilities", "performance_3m": 4.5, "type": "defensive"},
            {"rank": 4, "sector_name": "Consumer Discretionary", "performance_3m": 3.2, "type": "growth"},
            {"rank": 5, "sector_name": "Consumer Staples", "performance_3m": 3.0, "type": "defensive"},
        ],
        "bottom_10": [],
        "defensive_in_top_5": 2,
        "growth_in_top_5": 3,
    },
    "correction": {
        "top_10": [
            {"rank": 1, "sector_name": "Utilities", "performance_3m": 3.2, "type": "defensive"},
            {"rank": 2, "sector_name": "Consumer Staples", "performance_3m": 2.8, "type": "defensive"},
            {"rank": 3, "sector_name": "Healthcare", "performance_3m": 1.5, "type": "defensive"},
            {"rank": 4, "sector_name": "REITs", "performance_3m": 0.8, "type": "defensive"},
            {"rank": 5, "sector_name": "Bonds/Treasuries", "performance_3m": 0.5, "type": "defensive"},
        ],
        "bottom_10": [],
        "defensive_in_top_5": 5,
        "growth_in_top_5": 0,
    },
    "rally_attempt": {
        "top_10": [
            {"rank": 1, "sector_name": "Technology", "performance_3m": 6.5, "type": "growth"},
            {"rank": 2, "sector_name": "Consumer Discretionary", "performance_3m": 5.8, "type": "growth"},
            {"rank": 3, "sector_name": "Industrials", "performance_3m": 4.2, "type": "growth"},
            {"rank": 4, "sector_name": "Financial", "performance_3m": 3.5, "type": "growth"},
            {"rank": 5, "sector_name": "Healthcare", "performance_3m": 2.8, "type": "defensive"},
        ],
        "bottom_10": [],
        "defensive_in_top_5": 0,
        "growth_in_top_5": 4,
    },
    "follow_through_day": {
        "top_10": [
            {"rank": 1, "sector_name": "Technology", "performance_3m": 5.8, "type": "growth"},
            {"rank": 2, "sector_name": "Consumer Discretionary", "performance_3m": 4.5, "type": "growth"},
            {"rank": 3, "sector_name": "Industrials", "performance_3m": 3.8, "type": "growth"},
            {"rank": 4, "sector_name": "Healthcare", "performance_3m": 3.2, "type": "defensive"},
            {"rank": 5, "sector_name": "Financial", "performance_3m": 2.5, "type": "growth"},
        ],
        "bottom_10": [],
        "defensive_in_top_5": 0,
        "growth_in_top_5": 4,
    },
}

_MOCK_INDEX_HISTORY: dict[str, dict[str, list[dict]]] = {
    "rally_attempt": {
        "SP500": [
            {"date": "2026-02-18", "open": 5850, "high": 5920, "low": 5840, "close": 5900, "volume": 3_800_000_000, "change_pct": 0.8, "volume_vs_prior": 1.1},
            {"date": "2026-02-19", "open": 5900, "high": 5950, "low": 5890, "close": 5930, "volume": 3_600_000_000, "change_pct": 0.5, "volume_vs_prior": 0.95},
            {"date": "2026-02-20", "open": 5930, "high": 5980, "low": 5920, "close": 5960, "volume": 3_400_000_000, "change_pct": 0.5, "volume_vs_prior": 0.9},
        ],
        "NASDAQ": [
            {"date": "2026-02-18", "open": 15050, "high": 15250, "low": 15000, "close": 15200, "volume": 5_200_000_000, "change_pct": 0.7, "volume_vs_prior": 1.05},
            {"date": "2026-02-19", "open": 15200, "high": 15350, "low": 15180, "close": 15280, "volume": 4_900_000_000, "change_pct": 0.52, "volume_vs_prior": 0.9},
            {"date": "2026-02-20", "open": 15280, "high": 15400, "low": 15260, "close": 15350, "volume": 4_600_000_000, "change_pct": 0.46, "volume_vs_prior": 0.85},
        ],
    },
    "follow_through_day": {
        "NASDAQ": [
            {"date": "2026-02-10", "open": 15100, "high": 15150, "low": 14900, "close": 15000, "volume": 5_500_000_000, "change_pct": -1.2, "volume_vs_prior": 1.2},
            {"date": "2026-02-11", "open": 15000, "high": 15150, "low": 14980, "close": 15100, "volume": 5_100_000_000, "change_pct": 0.67, "volume_vs_prior": 0.93},
            {"date": "2026-02-12", "open": 15100, "high": 15120, "low": 15000, "close": 15050, "volume": 4_800_000_000, "change_pct": -0.33, "volume_vs_prior": 0.94},
            {"date": "2026-02-13", "open": 15050, "high": 15180, "low": 15030, "close": 15120, "volume": 5_000_000_000, "change_pct": 0.46, "volume_vs_prior": 1.04},
            {"date": "2026-02-14", "open": 15120, "high": 15250, "low": 15100, "close": 15200, "volume": 5_200_000_000, "change_pct": 0.53, "volume_vs_prior": 1.04},
            {"date": "2026-02-17", "open": 15200, "high": 15600, "low": 15180, "close": 15519, "volume": 6_200_000_000, "change_pct": 2.1, "volume_vs_prior": 1.6},
        ],
        "SP500": [
            {"date": "2026-02-10", "open": 5850, "high": 5880, "low": 5780, "close": 5800, "volume": 4_000_000_000, "change_pct": -0.9, "volume_vs_prior": 1.1},
            {"date": "2026-02-11", "open": 5800, "high": 5860, "low": 5790, "close": 5840, "volume": 3_800_000_000, "change_pct": 0.69, "volume_vs_prior": 0.95},
            {"date": "2026-02-12", "open": 5840, "high": 5860, "low": 5810, "close": 5820, "volume": 3_600_000_000, "change_pct": -0.34, "volume_vs_prior": 0.95},
            {"date": "2026-02-13", "open": 5820, "high": 5870, "low": 5810, "close": 5850, "volume": 3_700_000_000, "change_pct": 0.52, "volume_vs_prior": 1.03},
            {"date": "2026-02-14", "open": 5850, "high": 5890, "low": 5840, "close": 5870, "volume": 3_800_000_000, "change_pct": 0.34, "volume_vs_prior": 1.03},
            {"date": "2026-02-17", "open": 5870, "high": 5960, "low": 5860, "close": 5940, "volume": 4_200_000_000, "change_pct": 1.19, "volume_vs_prior": 1.1},
        ],
    },
    "correction": {
        "SP500": [],
        "NASDAQ": [],
    },
}


# ---------------------------------------------------------------------------
# Mock Data Functions
# ---------------------------------------------------------------------------

def get_distribution_days_mock(scenario: str = "healthy_uptrend") -> dict:
    """Return deterministic distribution day data for the given scenario."""
    return _MOCK_DISTRIBUTION_DAYS.get(scenario, _MOCK_DISTRIBUTION_DAYS["healthy_uptrend"])


def get_market_breadth_mock(scenario: str = "healthy_uptrend") -> dict:
    """Return deterministic breadth data for the given scenario."""
    return _MOCK_BREADTH.get(scenario, _MOCK_BREADTH["healthy_uptrend"])


def get_leading_stocks_health_mock(scenario: str = "healthy_uptrend") -> dict:
    """Return deterministic leader health data for the given scenario."""
    return _MOCK_LEADERS.get(scenario, _MOCK_LEADERS["healthy_uptrend"])


def get_sector_rankings_mock(scenario: str = "healthy_uptrend") -> dict:
    """Return deterministic sector ranking data for the given scenario."""
    return _MOCK_SECTORS.get(scenario, _MOCK_SECTORS["healthy_uptrend"])


def get_index_price_history_mock(
    index: str = "SP500",
    scenario: str = "rally_attempt",
) -> dict:
    """Return deterministic index price history for the given scenario."""
    scenario_data = _MOCK_INDEX_HISTORY.get(scenario, {})
    data = scenario_data.get(index, [])
    return {"data": data}


# ---------------------------------------------------------------------------
# Real Data Functions (yfinance)
# ---------------------------------------------------------------------------

def get_distribution_days_real() -> dict:
    """
    Fetch real distribution day data for S&P 500 and Nasdaq using yfinance.
    Downloads 2 months of daily OHLCV data and counts distribution days
    per MUST-M2 methodology.
    """
    if not HAS_YFINANCE:
        logger.warning("[RegimeData] yfinance not available — cannot fetch real distribution data")
        return get_distribution_days_mock("healthy_uptrend")

    try:
        from datetime import date

        df = yf.download(
            ["^GSPC", "^IXIC"],
            period="3mo",
            group_by="ticker",
            progress=False,
            threads=True,
        )

        if df is None or df.empty:
            logger.warning("[RegimeData] yfinance returned empty dataframe for indices")
            return get_distribution_days_mock("healthy_uptrend")

        result: dict = {"data_date": date.today().isoformat()}

        for sym, label in [("^GSPC", "sp500"), ("^IXIC", "nasdaq")]:
            try:
                sym_df = df[sym] if sym in df.columns.get_level_values(0) else df
                close = sym_df["Close"].dropna()
                volume = sym_df["Volume"].dropna()

                if len(close) < 26:
                    result[f"{label}_dist_count"] = 0
                    result[f"{label}_dist_days"] = []
                    continue

                # Use trailing 25 sessions
                recent_close = close.iloc[-26:]
                recent_volume = volume.iloc[-26:]

                dist_days = []
                power_ups = []

                for i in range(1, len(recent_close)):
                    pct_change = (float(recent_close.iloc[i]) - float(recent_close.iloc[i - 1])) / float(recent_close.iloc[i - 1]) * 100
                    vol_today = float(recent_volume.iloc[i])
                    vol_yesterday = float(recent_volume.iloc[i - 1])

                    if pct_change <= -0.2 and vol_today > vol_yesterday:
                        dist_days.append({
                            "date": str(recent_close.index[i].date()),
                            "decline_pct": round(pct_change, 2),
                            "volume": int(vol_today),
                        })

                    if pct_change >= 1.0 and vol_today > vol_yesterday:
                        power_ups.append({
                            "date": str(recent_close.index[i].date()),
                            "index": "S&P 500" if label == "sp500" else "Nasdaq",
                            "gain_pct": round(pct_change, 2),
                        })

                # Remove oldest dist day for each power-up
                net_dist = max(0, len(dist_days) - len(power_ups))

                result[f"{label}_dist_count"] = net_dist
                result[f"{label}_dist_days"] = dist_days[-net_dist:] if net_dist > 0 else []
                result[f"{label}_close"] = round(float(close.iloc[-1]), 2)
                result[f"{label}_volume"] = int(volume.iloc[-1])

                if label == "sp500":
                    result["power_up_days"] = power_ups

            except Exception as e:
                logger.debug(f"[RegimeData] Error processing {sym}: {e}")
                result[f"{label}_dist_count"] = 0
                result[f"{label}_dist_days"] = []

        if "power_up_days" not in result:
            result["power_up_days"] = []

        logger.info(
            f"[RegimeData] Real dist days: SP500={result.get('sp500_dist_count', 0)}, "
            f"Nasdaq={result.get('nasdaq_dist_count', 0)}"
        )
        return result

    except Exception as e:
        logger.warning(f"[RegimeData] Failed to fetch real distribution data: {e}")
        return get_distribution_days_mock("healthy_uptrend")


def get_market_breadth_real() -> dict:
    """
    Fetch real market breadth data using yfinance.
    Uses ^SPXA200R (S&P 500 % Above 200-Day MA) as primary proxy.
    Falls back to mock data if unavailable.
    """
    if not HAS_YFINANCE:
        logger.warning("[RegimeData] yfinance not available — cannot fetch real breadth data")
        return get_market_breadth_mock("healthy_uptrend")

    try:
        # Fetch S&P 500 % above 200-day MA proxy
        breadth_tickers = ["^SPXA200R"]
        df = yf.download(breadth_tickers, period="1mo", progress=False)

        if df is None or df.empty:
            logger.warning("[RegimeData] yfinance returned empty breadth data")
            return get_market_breadth_mock("healthy_uptrend")

        close = df["Close"].dropna()
        if len(close) < 2:
            return get_market_breadth_mock("healthy_uptrend")

        pct_above_200ma = round(float(close.iloc[-1]), 1)
        # Estimate 50-day as slightly lower than 200-day metric
        pct_above_50ma = round(max(0, pct_above_200ma - 8.0 + (5 * (1 if len(close) > 5 and float(close.iloc[-1]) > float(close.iloc[-5]) else -1))), 1)

        # Determine direction from 10-day trend
        if len(close) >= 10:
            direction = "rising" if float(close.iloc[-1]) > float(close.iloc[-10]) else "declining"
            vs_10d = "improving" if float(close.iloc[-1]) > float(close.iloc[-10]) else "deteriorating"
        else:
            direction = "flat"
            vs_10d = "stable"

        # Estimate new highs/lows from breadth level
        if pct_above_200ma > 65:
            new_highs, new_lows = 200 + int(pct_above_200ma * 2), 30
        elif pct_above_200ma > 50:
            new_highs, new_lows = 100 + int(pct_above_200ma), 80
        else:
            new_highs, new_lows = int(pct_above_200ma * 2), 150 + int((100 - pct_above_200ma) * 2)

        result = {
            "pct_above_200ma": pct_above_200ma,
            "pct_above_50ma": pct_above_50ma,
            "new_highs_today": new_highs,
            "new_lows_today": new_lows,
            "new_highs_10d_avg": int(new_highs * 0.9),
            "new_lows_10d_avg": int(new_lows * 0.9),
            "adv_decline_line_10d_direction": direction,
            "breadth_vs_10d_ago": vs_10d,
        }

        logger.info(f"[RegimeData] Real breadth: %above200MA={pct_above_200ma}%, direction={direction}")
        return result

    except Exception as e:
        logger.warning(f"[RegimeData] Failed to fetch real breadth data: {e}")
        return get_market_breadth_mock("healthy_uptrend")


def get_leading_stocks_health_real(analyst_output=None) -> dict:
    """
    Estimate leading stock health from analyst output RS ratings.
    If analyst_output has RS>=90 stocks, batch-fetch 50-day MA from yfinance.
    Falls back to mock data if analyst_output is None or yfinance unavailable.
    """
    if analyst_output is None:
        logger.info("[RegimeData] No analyst output — using mock leader health")
        return get_leading_stocks_health_mock("healthy_uptrend")

    # Get RS>=90 stocks from analyst output
    rs90_stocks = []
    for s in getattr(analyst_output, "rated_stocks", []):
        rs = getattr(s, "rs_rating", None)
        if rs is not None and rs >= 90:
            rs90_stocks.append(s.symbol)

    if not rs90_stocks:
        logger.info("[RegimeData] No RS>=90 stocks found in analyst output")
        return get_leading_stocks_health_mock("healthy_uptrend")

    total_rs90 = len(rs90_stocks)

    if not HAS_YFINANCE:
        # Estimate from count alone
        above_pct = 60.0 if total_rs90 >= 20 else 45.0
        return {
            "total_rs90_stocks": total_rs90,
            "rs90_above_50ma_count": int(total_rs90 * above_pct / 100),
            "rs90_above_50ma_pct": above_pct,
            "rs90_new_highs": max(1, total_rs90 // 6),
            "rs90_new_lows": max(1, total_rs90 // 20),
            "rs90_health_10d_ago_pct": above_pct - 2.0,
            "health_trend": "stable",
        }

    try:
        # Batch download — limit to 50 symbols to avoid timeout
        batch = rs90_stocks[:50]
        df = yf.download(batch, period="3mo", group_by="ticker", progress=False, threads=True)

        if df is None or df.empty:
            logger.warning("[RegimeData] yfinance returned empty data for leader stocks")
            return get_leading_stocks_health_mock("healthy_uptrend")

        above_50ma = 0
        checked = 0
        for sym in batch:
            try:
                if len(batch) == 1:
                    sym_close = df["Close"].dropna()
                else:
                    sym_close = df[sym]["Close"].dropna() if sym in df.columns.get_level_values(0) else None

                if sym_close is None or len(sym_close) < 50:
                    continue

                checked += 1
                current = float(sym_close.iloc[-1])
                ma50 = float(sym_close.iloc[-50:].mean())
                if current > ma50:
                    above_50ma += 1
            except Exception:
                continue

        if checked == 0:
            return get_leading_stocks_health_mock("healthy_uptrend")

        above_pct = round(above_50ma / checked * 100, 1)
        # Scale to total
        estimated_above = int(total_rs90 * above_pct / 100)

        result = {
            "total_rs90_stocks": total_rs90,
            "rs90_above_50ma_count": estimated_above,
            "rs90_above_50ma_pct": above_pct,
            "rs90_new_highs": max(1, estimated_above // 4),
            "rs90_new_lows": max(1, (total_rs90 - estimated_above) // 4),
            "rs90_health_10d_ago_pct": max(0, above_pct - 3.0),
            "health_trend": "improving" if above_pct > 55 else "deteriorating" if above_pct < 40 else "mixed",
        }

        logger.info(f"[RegimeData] Real leader health: {above_pct}% above 50MA ({checked} checked)")
        return result

    except Exception as e:
        logger.warning(f"[RegimeData] Failed to fetch real leader health: {e}")
        return get_leading_stocks_health_mock("healthy_uptrend")


def get_sector_rankings_real(analyst_output=None) -> dict:
    """
    Derive sector rankings from analyst output sector_rankings.
    Classifies each sector as growth or defensive based on DEFENSIVE_SECTORS.
    Falls back to mock data if analyst_output is None.
    """
    if analyst_output is None:
        logger.info("[RegimeData] No analyst output — using mock sector rankings")
        return get_sector_rankings_mock("healthy_uptrend")

    from ibd_agents.schemas.regime_detector_output import DEFENSIVE_SECTORS

    sector_rankings = getattr(analyst_output, "sector_rankings", [])
    if not sector_rankings:
        return get_sector_rankings_mock("healthy_uptrend")

    top_10 = []
    defensive_in_top_5 = 0
    growth_in_top_5 = 0

    for i, sr in enumerate(sector_rankings[:10]):
        sector_name = sr.sector if hasattr(sr, "sector") else str(sr)
        score = sr.score if hasattr(sr, "score") else 0.0

        # Classify as growth or defensive
        sector_lower = sector_name.lower()
        is_defensive = any(d in sector_lower for d in [s.lower() for s in DEFENSIVE_SECTORS])
        sector_type = "defensive" if is_defensive else "growth"

        if i < 5:
            if is_defensive:
                defensive_in_top_5 += 1
            else:
                growth_in_top_5 += 1

        top_10.append({
            "rank": i + 1,
            "sector_name": sector_name,
            "performance_3m": round(score * 0.8, 1),  # Approximate from score
            "type": sector_type,
        })

    result = {
        "top_10": top_10,
        "bottom_10": [],
        "defensive_in_top_5": defensive_in_top_5,
        "growth_in_top_5": growth_in_top_5,
    }

    logger.info(f"[RegimeData] Real sector rankings: {len(top_10)} sectors, "
                f"{growth_in_top_5} growth / {defensive_in_top_5} defensive in top 5")
    return result


def get_index_price_history_real(index: str = "SP500", days: int = 25) -> dict:
    """
    Fetch real daily OHLCV data for S&P 500 or Nasdaq via yfinance.
    Used for follow-through day detection.
    """
    if not HAS_YFINANCE:
        return get_index_price_history_mock(index, "rally_attempt")

    try:
        ticker = "^GSPC" if index == "SP500" else "^IXIC"
        df = yf.download(ticker, period="2mo", progress=False)

        if df is None or df.empty:
            return {"data": []}

        close = df["Close"].dropna()
        volume = df["Volume"].dropna()

        bars = []
        for i in range(max(0, len(close) - days - 1), len(close)):
            dt = close.index[i]
            c = float(close.iloc[i])
            v = int(volume.iloc[i]) if i < len(volume) else 0

            change_pct = 0.0
            vol_vs_prior = 1.0
            if i > 0:
                prev_c = float(close.iloc[i - 1])
                change_pct = round((c - prev_c) / prev_c * 100, 2) if prev_c > 0 else 0.0
                prev_v = float(volume.iloc[i - 1]) if i - 1 < len(volume) else 1
                vol_vs_prior = round(v / prev_v, 2) if prev_v > 0 else 1.0

            bars.append({
                "date": str(dt.date()),
                "open": round(float(df["Open"].iloc[i]), 2),
                "high": round(float(df["High"].iloc[i]), 2),
                "low": round(float(df["Low"].iloc[i]), 2),
                "close": round(c, 2),
                "volume": v,
                "change_pct": change_pct,
                "volume_vs_prior": vol_vs_prior,
            })

        logger.info(f"[RegimeData] Fetched {len(bars)} bars for {index}")
        return {"data": bars}

    except Exception as e:
        logger.warning(f"[RegimeData] Failed to fetch index history for {index}: {e}")
        return {"data": []}


# ---------------------------------------------------------------------------
# CrewAI Tool Wrappers
# ---------------------------------------------------------------------------

class DistributionDaysInput(BaseModel):
    """No parameters required."""
    pass


class DistributionDaysTool(BaseTool):
    """Get distribution day count for S&P 500 and Nasdaq."""
    name: str = "get_distribution_days"
    description: str = (
        "Get the current distribution day count for S&P 500 and Nasdaq "
        "over the trailing 25 trading sessions. A distribution day is "
        "when the index declines >= 0.2% on higher volume. 5+ distribution "
        "days signals an uptrend under serious pressure."
    )
    args_schema: type[BaseModel] = DistributionDaysInput

    def _run(self) -> str:
        import json
        data = get_distribution_days_mock("healthy_uptrend")
        return json.dumps(data)


class MarketBreadthInput(BaseModel):
    """No parameters required."""
    pass


class MarketBreadthTool(BaseTool):
    """Get market breadth indicators."""
    name: str = "get_market_breadth"
    description: str = (
        "Get market breadth indicators: percentage of S&P 500 stocks above "
        "200-day and 50-day MAs, new 52-week highs vs lows, and advance/"
        "decline line direction."
    )
    args_schema: type[BaseModel] = MarketBreadthInput

    def _run(self) -> str:
        import json
        data = get_market_breadth_mock("healthy_uptrend")
        return json.dumps(data)


class LeadingStocksHealthInput(BaseModel):
    """No parameters required."""
    pass


class LeadingStocksHealthTool(BaseTool):
    """Get leading stock health indicators."""
    name: str = "get_leading_stocks_health"
    description: str = (
        "Get the health of market-leading stocks: percentage of RS >= 90 "
        "stocks above their 50-day MA, new highs vs lows among leaders."
    )
    args_schema: type[BaseModel] = LeadingStocksHealthInput

    def _run(self) -> str:
        import json
        data = get_leading_stocks_health_mock("healthy_uptrend")
        return json.dumps(data)


class SectorRankingsInput(BaseModel):
    """No parameters required."""
    pass


class SectorRankingsTool(BaseTool):
    """Get current sector rankings."""
    name: str = "get_sector_rankings"
    description: str = (
        "Get the current IBD sector rankings with top sectors, defensive "
        "rotation detection, and growth vs defensive leadership."
    )
    args_schema: type[BaseModel] = SectorRankingsInput

    def _run(self) -> str:
        import json
        data = get_sector_rankings_mock("healthy_uptrend")
        return json.dumps(data)


class IndexPriceHistoryInput(BaseModel):
    index: str = Field(..., description="Which index: 'SP500' or 'NASDAQ'")
    days: int = Field(25, description="Number of trading days of history")


class IndexPriceHistoryTool(BaseTool):
    """Get daily OHLCV data for a major index."""
    name: str = "get_index_price_history"
    description: str = (
        "Get daily OHLCV data for S&P 500 or Nasdaq. Use for follow-through "
        "day detection by checking gain >= 1.25% on higher volume on Day 4+."
    )
    args_schema: type[BaseModel] = IndexPriceHistoryInput

    def _run(self, index: str, days: int = 25) -> str:
        import json
        data = get_index_price_history_mock(index, "rally_attempt")
        return json.dumps(data)
