"""
Market Data Fetcher — Real-time market data via yfinance.
IBD Momentum Investment Framework v4.0

Fetches real current prices, moving averages, volume ratios,
and earnings dates for portfolio positions. Graceful fallback
when yfinance is not installed or network is unavailable.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    yf = None  # type: ignore[assignment]


def is_available() -> bool:
    """Check if yfinance is installed and importable."""
    return HAS_YFINANCE


def fetch_real_market_data(symbols: list[str]) -> dict[str, dict]:
    """
    Batch-fetch real market data from yfinance for all symbols.

    Returns dict keyed by symbol with fields:
        current_price, ma_50, ma_200, volume_ratio,
        pct_from_50ma, pct_from_200ma,
        price_surge_pct_3w, price_surge_volume_ratio,
        days_until_earnings

    Symbols that fail to fetch are omitted from results.
    """
    if not HAS_YFINANCE:
        logger.warning("yfinance not installed — returning empty market data")
        return {}

    if not symbols:
        return {}

    result: dict[str, dict] = {}

    try:
        # Batch download 1 year of daily data for all symbols
        logger.info(f"[MarketData] Fetching data for {len(symbols)} symbols via yfinance ...")
        df = yf.download(
            symbols,
            period="1y",
            group_by="ticker",
            progress=False,
            threads=True,
        )

        if df is None or df.empty:
            logger.warning("[MarketData] yfinance returned empty dataframe")
            return {}

        is_single = len(symbols) == 1

        for sym in symbols:
            try:
                # Extract per-symbol data from multi-level columns
                if is_single:
                    sym_df = df
                else:
                    if sym not in df.columns.get_level_values(0):
                        logger.debug(f"[MarketData] {sym} not in download results")
                        continue
                    sym_df = df[sym]

                close = sym_df["Close"].dropna()
                volume = sym_df["Volume"].dropna()

                if len(close) < 5:
                    logger.debug(f"[MarketData] {sym} has insufficient price data ({len(close)} days)")
                    continue

                # Current price = last available close
                current_price = float(close.iloc[-1])

                # Moving averages
                ma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else current_price
                ma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else ma_50

                # Pct from MAs
                pct_from_50ma = round((current_price - ma_50) / ma_50 * 100, 2) if ma_50 > 0 else 0.0
                pct_from_200ma = round((current_price - ma_200) / ma_200 * 100, 2) if ma_200 > 0 else 0.0

                # Volume ratio: last day volume / 50-day average volume
                if len(volume) >= 50:
                    avg_vol_50 = float(volume.iloc[-50:].mean())
                    last_vol = float(volume.iloc[-1])
                    volume_ratio = round(last_vol / avg_vol_50, 2) if avg_vol_50 > 0 else 1.0
                else:
                    volume_ratio = 1.0

                # 3-week price surge: % change over last 15 trading days
                if len(close) >= 16:
                    price_15d_ago = float(close.iloc[-16])
                    price_surge_pct_3w = round((current_price - price_15d_ago) / price_15d_ago * 100, 1)
                else:
                    price_surge_pct_3w = 0.0

                # 3-week volume ratio during surge period
                if len(volume) >= 16:
                    surge_vol = float(volume.iloc[-15:].mean())
                    prior_vol = float(volume.iloc[-50:-15].mean()) if len(volume) >= 50 else surge_vol
                    price_surge_volume_ratio = round(surge_vol / prior_vol, 2) if prior_vol > 0 else 1.0
                else:
                    price_surge_volume_ratio = 1.0

                result[sym] = {
                    "current_price": round(current_price, 2),
                    "ma_50": round(ma_50, 2),
                    "ma_200": round(ma_200, 2),
                    "pct_from_50ma": pct_from_50ma,
                    "pct_from_200ma": pct_from_200ma,
                    "volume_ratio": volume_ratio,
                    "price_surge_pct_3w": price_surge_pct_3w,
                    "price_surge_volume_ratio": price_surge_volume_ratio,
                    "days_until_earnings": None,  # Filled below
                }

            except Exception as e:
                logger.debug(f"[MarketData] Error processing {sym}: {e}")
                continue

        # Fetch earnings dates (per-symbol, can't batch)
        _fetch_earnings_dates(symbols, result)

        logger.info(f"[MarketData] Successfully fetched data for {len(result)}/{len(symbols)} symbols")

    except Exception as e:
        logger.warning(f"[MarketData] Batch download failed: {e}")
        return {}

    return result


def _fetch_earnings_dates(symbols: list[str], result: dict[str, dict]) -> None:
    """Fetch next earnings date for symbols that have price data."""
    today = date.today()

    for sym in symbols:
        if sym not in result:
            continue
        try:
            ticker = yf.Ticker(sym)
            cal = ticker.calendar
            if cal is not None and not cal.empty:
                # calendar is a DataFrame with columns like 'Earnings Date'
                if "Earnings Date" in cal.columns:
                    earnings_dates = cal["Earnings Date"]
                    for ed in earnings_dates:
                        if hasattr(ed, "date"):
                            ed_date = ed.date()
                        else:
                            ed_date = ed
                        if ed_date >= today:
                            delta = (ed_date - today).days
                            result[sym]["days_until_earnings"] = delta
                            break
                elif isinstance(cal, dict) and "Earnings Date" in cal:
                    ed = cal["Earnings Date"]
                    if hasattr(ed, "date"):
                        ed_date = ed.date()
                    elif isinstance(ed, list) and len(ed) > 0:
                        ed_val = ed[0]
                        if hasattr(ed_val, "date"):
                            ed_date = ed_val.date()
                        else:
                            ed_date = ed_val
                    else:
                        continue
                    if ed_date >= today:
                        result[sym]["days_until_earnings"] = (ed_date - today).days
        except Exception:
            # Earnings calendar not available — leave as None
            pass
