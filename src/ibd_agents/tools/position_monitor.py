"""
Exit Strategist Tool: Position Monitor
IBD Momentum Investment Framework v4.0

Generate mock market data for exit analysis in the deterministic pipeline.
In agentic mode, these would call real market data APIs.

Uses seed=42 for reproducible mock data.
"""

from __future__ import annotations

import logging
import random
from datetime import date, timedelta
from typing import Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool

from pydantic import BaseModel, Field

from ibd_agents.schemas.exit_strategy_output import ExitMarketRegime
from ibd_agents.schemas.portfolio_output import PortfolioOutput
from ibd_agents.schemas.reconciliation_output import HoldingsSummary

logger = logging.getLogger(__name__)

_RNG = random.Random(42)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class PositionMarketData(BaseModel):
    """Market data for one position."""

    symbol: str
    tier: int = 2
    asset_type: str = "stock"
    sector: str = "UNKNOWN"
    current_price: float
    buy_price: float
    gain_loss_pct: float
    days_held: int
    volume_ratio: float  # today volume / 50-day avg
    ma_50: float
    pct_from_50ma: float
    ma_200: float
    pct_from_200ma: float
    rs_rating_current: int
    rs_rating_peak_since_buy: int
    rs_rating_4w_ago: int
    sector_distribution_days_3w: int
    days_until_earnings: Optional[int] = None
    avg_earnings_move_pct: Optional[float] = None
    price_surge_pct_3w: float = 0.0
    price_surge_volume_ratio: float = 1.0


class MarketHealthData(BaseModel):
    """Broad market health indicators."""

    sp500_dist_days: int
    nasdaq_dist_days: int
    pct_above_200ma: float
    new_highs: int
    new_lows: int
    exposure_model: int  # 0-100


# ---------------------------------------------------------------------------
# Regime Mapping
# ---------------------------------------------------------------------------

def map_regime_to_exit_regime(strategy_regime: str) -> ExitMarketRegime:
    """
    Map existing codebase regime string to ExitMarketRegime.

    The strategy output uses strings like "bull regime", "neutral", "bear".
    """
    regime_lower = strategy_regime.lower().strip()
    if "bull" in regime_lower:
        return ExitMarketRegime.CONFIRMED_UPTREND
    if "bear" in regime_lower:
        return ExitMarketRegime.CORRECTION
    if "rally" in regime_lower:
        return ExitMarketRegime.RALLY_ATTEMPT
    # Default: neutral → UPTREND_UNDER_PRESSURE
    return ExitMarketRegime.UPTREND_UNDER_PRESSURE


def map_5state_regime_to_exit_regime(regime_5state: str) -> ExitMarketRegime:
    """
    Map Agent 16 five-state regime string to ExitMarketRegime.

    Direct 1:1 mapping since both use the same 4 regime names
    (Agent 16 has 5 states; ExitMarketRegime has 4 — FOLLOW_THROUGH_DAY
    maps to RALLY_ATTEMPT for exit purposes since it's still transitional).
    """
    _MAP = {
        "CONFIRMED_UPTREND": ExitMarketRegime.CONFIRMED_UPTREND,
        "UPTREND_UNDER_PRESSURE": ExitMarketRegime.UPTREND_UNDER_PRESSURE,
        "FOLLOW_THROUGH_DAY": ExitMarketRegime.RALLY_ATTEMPT,
        "RALLY_ATTEMPT": ExitMarketRegime.RALLY_ATTEMPT,
        "CORRECTION": ExitMarketRegime.CORRECTION,
    }
    return _MAP.get(regime_5state.upper().strip(), ExitMarketRegime.UPTREND_UNDER_PRESSURE)


# ---------------------------------------------------------------------------
# Mock Market Health
# ---------------------------------------------------------------------------

def build_mock_market_health(regime: ExitMarketRegime) -> MarketHealthData:
    """Generate market health data consistent with the regime."""
    if regime == ExitMarketRegime.CONFIRMED_UPTREND:
        return MarketHealthData(
            sp500_dist_days=2, nasdaq_dist_days=1,
            pct_above_200ma=72.0, new_highs=180, new_lows=30,
            exposure_model=80,
        )
    if regime == ExitMarketRegime.UPTREND_UNDER_PRESSURE:
        return MarketHealthData(
            sp500_dist_days=4, nasdaq_dist_days=3,
            pct_above_200ma=55.0, new_highs=90, new_lows=80,
            exposure_model=50,
        )
    if regime == ExitMarketRegime.CORRECTION:
        return MarketHealthData(
            sp500_dist_days=6, nasdaq_dist_days=5,
            pct_above_200ma=35.0, new_highs=40, new_lows=200,
            exposure_model=25,
        )
    # RALLY_ATTEMPT
    return MarketHealthData(
        sp500_dist_days=5, nasdaq_dist_days=4,
        pct_above_200ma=42.0, new_highs=60, new_lows=120,
        exposure_model=40,
    )


# ---------------------------------------------------------------------------
# Real Market Data (via yfinance)
# ---------------------------------------------------------------------------

def build_real_market_data(
    portfolio: PortfolioOutput,
    analyst_output=None,
    brokerage_holdings: Optional[HoldingsSummary] = None,
) -> list[PositionMarketData]:
    """
    Build market data using real prices from yfinance.

    Real data: current_price, ma_50, ma_200, volume_ratio,
    pct_from_50ma, pct_from_200ma, price_surge_pct_3w,
    price_surge_volume_ratio, days_until_earnings.

    From analyst/mock: rs_rating, sector_distribution_days,
    buy_price, days_held, avg_earnings_move_pct.

    If brokerage_holdings is provided, brokerage-only positions
    (not in model portfolio) are merged in with conservative defaults.
    """
    from ibd_agents.tools.market_data_fetcher import fetch_real_market_data

    all_positions = _collect_positions(portfolio)
    model_symbols = {p["symbol"] for p in all_positions}

    # Merge brokerage-only positions
    if brokerage_holdings is not None:
        brokerage_positions = _collect_brokerage_positions(
            brokerage_holdings, model_symbols, analyst_output,
        )
        if brokerage_positions:
            logger.info(f"[PositionMonitor] Merging {len(brokerage_positions)} brokerage-only positions")
            all_positions.extend(brokerage_positions)

    if not all_positions:
        return []

    symbols = [p["symbol"] for p in all_positions]
    real_data = fetch_real_market_data(symbols)

    if not real_data:
        logger.warning("[PositionMonitor] No real data returned — falling back to mock")
        return build_mock_market_data(portfolio, analyst_output, brokerage_holdings)

    # Build RS ratings lookup from analyst output
    rs_lookup: dict[str, int] = {}
    if analyst_output is not None:
        for s in getattr(analyst_output, "rated_stocks", []):
            if hasattr(s, "rs_rating") and s.rs_rating is not None:
                rs_lookup[s.symbol] = s.rs_rating

    rng = random.Random(42)
    results: list[PositionMarketData] = []

    for pos in all_positions:
        symbol = pos["symbol"]

        if symbol not in real_data:
            logger.debug(f"[PositionMonitor] {symbol} not in real data — using mock fallback")
            # Use real buy price if available, else generate mock
            real_buy = pos.get("buy_price")
            if real_buy is not None and real_buy > 0:
                base_price = real_buy
                gain = round(3.0 + rng.random() * 12.0, 1)
                current = round(base_price * (1 + gain / 100), 2)
            else:
                base_price = 50.0 + (hash(symbol) % 400)
                gain = round(3.0 + rng.random() * 12.0, 1)
                current = round(base_price * (1 + gain / 100), 2)
            ma50 = round(current * 0.97, 2)
            ma200 = round(ma50 * 0.95, 2)
            results.append(PositionMarketData(
                symbol=symbol, tier=pos["tier"], asset_type=pos["asset_type"],
                sector=pos["sector"], current_price=current, buy_price=base_price,
                gain_loss_pct=gain, days_held=30, volume_ratio=1.0,
                ma_50=ma50, pct_from_50ma=round((current - ma50) / ma50 * 100, 2),
                ma_200=ma200, pct_from_200ma=round((current - ma200) / ma200 * 100, 2),
                rs_rating_current=rs_lookup.get(symbol, 80),
                rs_rating_peak_since_buy=rs_lookup.get(symbol, 80),
                rs_rating_4w_ago=rs_lookup.get(symbol, 78),
                sector_distribution_days_3w=rng.randint(0, 2),
                days_until_earnings=rng.randint(30, 80),
                avg_earnings_move_pct=round(5.0 + rng.random() * 10.0, 1),
                price_surge_pct_3w=round(rng.random() * 8.0, 1),
                price_surge_volume_ratio=round(0.8 + rng.random() * 0.5, 2),
            ))
            continue

        rd = real_data[symbol]
        current_price = rd["current_price"]
        rs_base = rs_lookup.get(symbol, 80)

        # Use real buy price from brokerage cost basis if available
        real_buy = pos.get("buy_price")
        if real_buy is not None and real_buy > 0:
            buy_price = round(real_buy, 2)
            gain_loss_pct = round((current_price - buy_price) / buy_price * 100, 1)
        else:
            # Fallback: estimate buy price from mock gain
            mock_gain_pct = round(3.0 + rng.random() * 12.0, 1)
            buy_price = round(current_price / (1 + mock_gain_pct / 100), 2)
            gain_loss_pct = round((current_price - buy_price) / buy_price * 100, 1)

        results.append(PositionMarketData(
            symbol=symbol,
            tier=pos["tier"],
            asset_type=pos["asset_type"],
            sector=pos["sector"],
            current_price=current_price,
            buy_price=buy_price,
            gain_loss_pct=gain_loss_pct,
            days_held=30 + rng.randint(0, 60),
            volume_ratio=rd["volume_ratio"],
            ma_50=rd["ma_50"],
            pct_from_50ma=rd["pct_from_50ma"],
            ma_200=rd["ma_200"],
            pct_from_200ma=rd["pct_from_200ma"],
            rs_rating_current=rs_base,
            rs_rating_peak_since_buy=max(rs_base, rs_base + rng.randint(0, 5)),
            rs_rating_4w_ago=min(rs_base + rng.randint(0, 5), 99),
            sector_distribution_days_3w=rng.randint(0, 2),
            days_until_earnings=rd["days_until_earnings"],
            avg_earnings_move_pct=round(5.0 + rng.random() * 10.0, 1),
            price_surge_pct_3w=rd["price_surge_pct_3w"],
            price_surge_volume_ratio=rd["price_surge_volume_ratio"],
        ))

    logger.info(f"[PositionMonitor] Built real market data for {len(results)} positions "
                f"({len(real_data)} from yfinance)")
    return results


# ---------------------------------------------------------------------------
# Mock Position Market Data
# ---------------------------------------------------------------------------

def _collect_positions(portfolio: PortfolioOutput) -> list[dict]:
    """Extract all positions from portfolio tiers into a flat list of dicts."""
    positions = []
    for tier_obj in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]:
        for p in tier_obj.positions:
            positions.append({
                "symbol": p.symbol,
                "tier": p.tier,
                "asset_type": p.asset_type,
                "sector": p.sector,
                "target_pct": p.target_pct,
                "trailing_stop_pct": p.trailing_stop_pct,
            })
    # Sort by symbol for deterministic scenario assignment
    positions.sort(key=lambda x: x["symbol"])
    return positions


# Known ETF tickers for brokerage-only position classification
_KNOWN_ETFS = {
    "ACWX", "AVDV", "AVEM", "COPX", "DIA", "EWY", "GLD", "ILF", "ITA",
    "IWN", "PPA", "QQQ", "SCHD", "SPY", "TAN", "URA", "VEA", "XBI", "XLP",
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLU", "XLB", "XLY", "XLRE",
    "VTI", "VOO", "VWO", "VGK", "IVV", "IJR", "IJH", "EFA", "EEM",
    "AGG", "BND", "TLT", "HYG", "LQD", "IEFA", "IEMG", "SLV", "USO",
}


def _classify_asset_type(symbol: str, etf_symbols: set[str] | None = None) -> str:
    """Classify a brokerage symbol as stock or etf."""
    if symbol in _KNOWN_ETFS:
        return "etf"
    if etf_symbols and symbol in etf_symbols:
        return "etf"
    return "stock"


def _collect_brokerage_positions(
    holdings: HoldingsSummary,
    model_symbols: set[str],
    analyst_output=None,
) -> list[dict]:
    """
    Extract brokerage-only positions (not in model portfolio) with conservative defaults.

    Returns list of position dicts compatible with _collect_positions() output.
    """
    # Build sector lookup from analyst output
    sector_lookup: dict[str, str] = {}
    etf_symbols: set[str] = set()
    if analyst_output is not None:
        for s in getattr(analyst_output, "rated_stocks", []):
            if hasattr(s, "sector") and s.sector:
                sector_lookup[s.symbol] = s.sector
        for e in getattr(analyst_output, "rated_etfs", []):
            etf_symbols.add(e.symbol)
            if hasattr(e, "sector") and e.sector:
                sector_lookup[e.symbol] = e.sector

    # Build cost basis lookup: aggregate across accounts for same symbol
    cost_basis_lookup: dict[str, float] = {}
    mkt_val_lookup: dict[str, float] = {}
    for h in holdings.holdings:
        sym = h.symbol.strip().upper()
        if h.cost_basis is not None:
            cost_basis_lookup[sym] = cost_basis_lookup.get(sym, 0.0) + h.cost_basis
        mkt_val_lookup[sym] = mkt_val_lookup.get(sym, 0.0) + h.market_value

    positions = []
    seen = set()
    for h in holdings.holdings:
        sym = h.symbol.strip().upper()
        if sym in model_symbols or sym in seen:
            continue
        seen.add(sym)

        # Compute per-share buy price from aggregated cost basis
        total_shares = sum(
            hh.shares for hh in holdings.holdings if hh.symbol.strip().upper() == sym
        )
        cost_basis = cost_basis_lookup.get(sym)
        buy_price = None
        if cost_basis is not None and total_shares > 0:
            buy_price = round(cost_basis / total_shares, 4)

        positions.append({
            "symbol": sym,
            "tier": 3,
            "asset_type": _classify_asset_type(sym, etf_symbols),
            "sector": sector_lookup.get(sym, h.sector if h.sector != "UNKNOWN" else "UNKNOWN"),
            "target_pct": 0.0,
            "trailing_stop_pct": 12.0,
            "buy_price": buy_price,
        })

    positions.sort(key=lambda x: x["symbol"])
    return positions


def build_mock_market_data(
    portfolio: PortfolioOutput,
    analyst_output=None,
    brokerage_holdings: Optional[HoldingsSummary] = None,
) -> list[PositionMarketData]:
    """
    Generate deterministic mock market data for all portfolio positions.

    Assigns specific scenarios to sorted-index positions to ensure
    all 8 sell rules are exercised:
        0: Stop loss breach (-8.2%)
        1: Climax top (35% surge, 2.5x volume)
        2: Below 50-day MA (-2.1%, 2.0x volume)
        3: RS deterioration (90→72)
        4: Profit target (22%, 60 days)
        5: 8-week hold exception (22%, 12 days)
        6: Earnings risk (10 days, 8% gain)
        7: Sector distribution (5 dist days)
        8+: Healthy (3-15% gain, strong RS)

    If brokerage_holdings is provided, brokerage-only positions
    are merged in and assigned healthy mock scenarios.
    """
    all_positions = _collect_positions(portfolio)
    model_symbols = {p["symbol"] for p in all_positions}

    # Merge brokerage-only positions
    if brokerage_holdings is not None:
        brokerage_positions = _collect_brokerage_positions(
            brokerage_holdings, model_symbols, analyst_output,
        )
        if brokerage_positions:
            logger.info(f"[PositionMonitor] Merging {len(brokerage_positions)} brokerage-only positions (mock)")
            all_positions.extend(brokerage_positions)

    if not all_positions:
        return []

    # Build RS ratings lookup from analyst output
    rs_lookup: dict[str, int] = {}
    if analyst_output is not None:
        for s in getattr(analyst_output, "rated_stocks", []):
            if hasattr(s, "rs_rating") and s.rs_rating is not None:
                rs_lookup[s.symbol] = s.rs_rating

    results: list[PositionMarketData] = []
    today = date.today()
    rng = random.Random(42)  # Local RNG for deterministic output

    for idx, pos in enumerate(all_positions):
        symbol = pos["symbol"]
        tier = pos["tier"]
        asset_type = pos["asset_type"]
        sector = pos["sector"]

        # Base buy price: use real cost basis if available, else deterministic hash
        real_buy = pos.get("buy_price")
        if real_buy is not None and real_buy > 0:
            base_price = real_buy
        else:
            base_price = 50.0 + (hash(symbol) % 400)
        rs_base = rs_lookup.get(symbol, 80)

        if idx == 0:
            # Scenario: Stop loss breach
            gain_loss_pct = -8.2
            current_price = round(base_price * (1 + gain_loss_pct / 100), 2)
            days_held = 25
            volume_ratio = 1.8
            ma_50 = round(base_price * 1.02, 2)
            rs_current = rs_base
            rs_peak = rs_base
            sector_dist = 1
            days_until_earnings = None
            surge_pct = 0.0
            surge_vol = 1.0

        elif idx == 1:
            # Scenario: Climax top
            gain_loss_pct = 40.0
            current_price = round(base_price * (1 + gain_loss_pct / 100), 2)
            days_held = 90
            volume_ratio = 3.0
            ma_50 = round(current_price * 0.85, 2)
            rs_current = min(99, rs_base + 5)
            rs_peak = rs_current
            sector_dist = 1
            days_until_earnings = 45
            surge_pct = 35.0
            surge_vol = 2.5

        elif idx == 2:
            # Scenario: Below 50-day MA on heavy volume
            gain_loss_pct = -3.5
            current_price = round(base_price * (1 + gain_loss_pct / 100), 2)
            days_held = 40
            volume_ratio = 2.0
            ma_50 = round(current_price * 1.021, 2)  # 2.1% above current
            rs_current = rs_base
            rs_peak = rs_base
            sector_dist = 2
            days_until_earnings = 60
            surge_pct = 0.0
            surge_vol = 1.0

        elif idx == 3:
            # Scenario: RS deterioration
            gain_loss_pct = -1.5
            current_price = round(base_price * (1 + gain_loss_pct / 100), 2)
            days_held = 50
            volume_ratio = 1.2
            ma_50 = round(current_price * 0.99, 2)
            rs_current = 72
            rs_peak = 90
            sector_dist = 1
            days_until_earnings = 40
            surge_pct = 0.0
            surge_vol = 1.0

        elif idx == 4:
            # Scenario: Profit target (standard)
            gain_loss_pct = 22.0
            current_price = round(base_price * (1 + gain_loss_pct / 100), 2)
            days_held = 60
            volume_ratio = 1.1
            ma_50 = round(current_price * 0.95, 2)
            rs_current = min(99, rs_base + 3)
            rs_peak = rs_current
            sector_dist = 0
            days_until_earnings = 50
            surge_pct = 8.0
            surge_vol = 1.2

        elif idx == 5:
            # Scenario: 8-week hold exception (20%+ in < 3 weeks)
            gain_loss_pct = 22.0
            current_price = round(base_price * (1 + gain_loss_pct / 100), 2)
            days_held = 12
            volume_ratio = 2.2
            ma_50 = round(current_price * 0.92, 2)
            rs_current = min(99, rs_base + 8)
            rs_peak = rs_current
            sector_dist = 0
            days_until_earnings = 70
            surge_pct = 22.0
            surge_vol = 1.8

        elif idx == 6:
            # Scenario: Earnings risk (insufficient cushion)
            gain_loss_pct = 8.0
            current_price = round(base_price * (1 + gain_loss_pct / 100), 2)
            days_held = 35
            volume_ratio = 1.0
            ma_50 = round(current_price * 0.97, 2)
            rs_current = rs_base
            rs_peak = rs_base
            sector_dist = 1
            days_until_earnings = 10
            surge_pct = 3.0
            surge_vol = 1.0

        elif idx == 7:
            # Scenario: Sector distribution days
            gain_loss_pct = 5.0
            current_price = round(base_price * (1 + gain_loss_pct / 100), 2)
            days_held = 45
            volume_ratio = 1.3
            ma_50 = round(current_price * 0.98, 2)
            rs_current = rs_base
            rs_peak = rs_base
            sector_dist = 5
            days_until_earnings = 55
            surge_pct = 2.0
            surge_vol = 1.0

        else:
            # Scenario: Healthy position
            gain_loss_pct = round(3.0 + rng.random() * 12.0, 1)
            current_price = round(base_price * (1 + gain_loss_pct / 100), 2)
            days_held = 30 + rng.randint(0, 60)
            volume_ratio = round(0.8 + rng.random() * 0.4, 2)
            ma_50 = round(current_price * (0.96 + rng.random() * 0.02), 2)
            rs_current = min(99, rs_base + rng.randint(-3, 5))
            rs_peak = max(rs_current, rs_base)
            sector_dist = rng.randint(0, 2)
            days_until_earnings = rng.randint(30, 80)
            surge_pct = round(rng.random() * 8.0, 1)
            surge_vol = round(0.8 + rng.random() * 0.5, 2)

        # Compute derived fields
        pct_from_50ma = round((current_price - ma_50) / ma_50 * 100, 2) if ma_50 > 0 else 0.0
        ma_200 = round(ma_50 * 0.95, 2)
        pct_from_200ma = round((current_price - ma_200) / ma_200 * 100, 2) if ma_200 > 0 else 0.0
        rs_4w_ago = min(rs_peak, rs_current + rng.randint(0, 5))
        avg_earnings_move = round(5.0 + rng.random() * 10.0, 1) if days_until_earnings is not None else None

        results.append(PositionMarketData(
            symbol=symbol,
            tier=tier,
            asset_type=asset_type,
            sector=sector,
            current_price=current_price,
            buy_price=base_price,
            gain_loss_pct=gain_loss_pct,
            days_held=days_held,
            volume_ratio=volume_ratio,
            ma_50=ma_50,
            pct_from_50ma=pct_from_50ma,
            ma_200=ma_200,
            pct_from_200ma=pct_from_200ma,
            rs_rating_current=rs_current,
            rs_rating_peak_since_buy=rs_peak,
            rs_rating_4w_ago=rs_4w_ago,
            sector_distribution_days_3w=sector_dist,
            days_until_earnings=days_until_earnings,
            avg_earnings_move_pct=avg_earnings_move,
            price_surge_pct_3w=surge_pct,
            price_surge_volume_ratio=surge_vol,
        ))

    return results


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class PositionMonitorInput(BaseModel):
    portfolio_json: str = Field(
        ..., description="JSON string of PortfolioOutput",
    )


class PositionMonitorTool(BaseTool):
    """Generate market data for exit analysis of portfolio positions."""

    name: str = "position_monitor"
    description: str = (
        "Generate current market data for exit analysis including prices, "
        "moving averages, RS ratings, volume, earnings calendar, and "
        "sector distribution days for all portfolio positions."
    )
    args_schema: type[BaseModel] = PositionMonitorInput

    def _run(self, portfolio_json: str) -> str:
        import json
        portfolio = PortfolioOutput.model_validate_json(portfolio_json)
        data = build_mock_market_data(portfolio)
        return json.dumps([d.model_dump() for d in data], indent=2)
