"""
Agent 13: Historical Analyst
IBD Momentum Investment Framework v4.0

Uses RAG over historical IBD weekly snapshots to provide:
- Per-stock rating trends and list entry/exit signals
- Improving/deteriorating momentum classification
- Historical analog identification (similar past setups)
- Sector-level historical trends

Follows dual-path: ChromaDB when available, empty fallback otherwise.
Never makes buy/sell recommendations — only discovers, scores, and organizes.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    Agent = None  # type: ignore
    Task = None  # type: ignore

from ibd_agents.schemas.analyst_output import AnalystOutput
from ibd_agents.schemas.historical_output import (
    HistoricalAnalog,
    HistoricalAnalysisOutput,
    HistoricalStoreMeta,
    RatingTrend,
    SectorHistoricalTrend,
    StockHistoricalContext,
)
from ibd_agents.tools.historical_store import (
    HistoricalStoreTool,
    find_similar_setups,
    get_sector_trends,
    get_store_stats,
    ingest_snapshot,
    search_list_signals,
    search_rating_history,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI Builders
# ---------------------------------------------------------------------------

def build_historical_agent() -> "Agent":
    """Build the Historical Analyst CrewAI agent."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode")
    return Agent(
        role="Historical IBD Analyst",
        goal=(
            "Analyze historical IBD weekly snapshot data to provide context "
            "for the current pipeline: rating trends, list entry/exit signals, "
            "historical analogs, and sector trends. Never make buy/sell "
            "recommendations — only discover, score, and organize."
        ),
        backstory=(
            "You are a historical market data specialist who tracks IBD "
            "stock ratings over time. You use a ChromaDB-backed RAG store "
            "of weekly IBD snapshots to identify momentum shifts, list "
            "appearance patterns, and historically similar stock setups."
        ),
        tools=[HistoricalStoreTool()],
        verbose=True,
        allow_delegation=False,
        max_iter=15,
    )


def build_historical_task(
    agent: "Agent",
    analyst_json: str = "",
) -> "Task":
    """Build the Historical Analysis CrewAI task."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode")
    return Task(
        description=f"""Analyze historical IBD data for the current pipeline stocks.

6 PHASES:

1. CHECK STORE: Query store stats for availability
2. INGEST CURRENT: Add current week's stocks to the store
3. RATING HISTORY: For each stock, get weekly rating trends
4. LIST SIGNALS: Find recent ENTRY/EXIT events
5. SIMILAR SETUPS: For T1 stocks, find historical analogs
6. SECTOR TRENDS: Get sector-level metrics over time

Analyst data:
{analyst_json}
""",
        expected_output=(
            "JSON object with stock_analyses (per-stock historical context), "
            "improving_stocks, deteriorating_stocks, historical_analogs, "
            "sector_trends, new_to_ibd_lists, dropped_from_ibd_lists, "
            "historical_source, summary."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Momentum Classification
# ---------------------------------------------------------------------------

def _classify_momentum(history: list[dict]) -> tuple[str, float]:
    """
    Classify momentum direction from rating history.

    Returns (direction, momentum_score).
    direction: "improving" / "stable" / "deteriorating"
    momentum_score: -100 to +100
    """
    if len(history) < 2:
        return "stable", 0.0

    recent = history[-4:]

    rs_vals = [h.get("rs_rating") for h in recent if h.get("rs_rating")]
    comp_vals = [
        h.get("composite_rating") for h in recent
        if h.get("composite_rating")
    ]

    rs_delta = (rs_vals[-1] - rs_vals[0]) if len(rs_vals) >= 2 else 0
    comp_delta = (comp_vals[-1] - comp_vals[0]) if len(comp_vals) >= 2 else 0

    # Weighted: RS 60%, Composite 40%, scaled to -100..100
    raw = rs_delta * 0.6 + comp_delta * 0.4
    score = round(max(-100, min(100, raw * 5)), 1)

    if score > 5:
        return "improving", score
    elif score < -5:
        return "deteriorating", score
    else:
        return "stable", score


def _build_rating_trends(history: list[dict]) -> list[RatingTrend]:
    """Build RatingTrend objects for each tracked metric."""
    trends: list[RatingTrend] = []
    for metric in ("composite", "rs", "eps"):
        key = f"{metric}_rating"
        values = [
            {"date": h["snapshot_date"], "value": h.get(key)}
            for h in history
            if h.get(key) is not None
        ]
        if not values:
            continue

        if len(values) >= 2:
            first_val = values[0]["value"]
            last_val = values[-1]["value"]
            delta = last_val - first_val
            change_4w = float(delta)
            if delta > 2:
                direction = "improving"
            elif delta < -2:
                direction = "deteriorating"
            else:
                direction = "stable"
        else:
            direction = "stable"
            change_4w = 0.0

        trends.append(RatingTrend(
            metric=metric,
            values=values,
            direction=direction,
            change_4w=change_4w,
        ))

    return trends


def _classify_sector_trend(snapshots: list[dict], key: str) -> str:
    """Classify a sector metric trend."""
    if len(snapshots) < 2:
        return "stable" if key != "stock_count" else "stable"
    first = snapshots[0].get(key, 0)
    last = snapshots[-1].get(key, 0)
    delta = last - first
    if key == "stock_count":
        return "growing" if delta > 2 else (
            "shrinking" if delta < -2 else "stable"
        )
    return "improving" if delta > 2 else (
        "deteriorating" if delta < -2 else "stable"
    )


# ---------------------------------------------------------------------------
# Deterministic Pipeline
# ---------------------------------------------------------------------------

def run_historical_pipeline(
    analyst_output: AnalystOutput,
    db_path: str = "data/chroma_db",
    auto_ingest_current: bool = True,
) -> HistoricalAnalysisOutput:
    """
    Run the Historical Analyst deterministic pipeline.

    6 phases:
    1. Check store availability and stats
    2. Optionally ingest current week's data
    3. For each rated stock: query rating history, classify momentum
    4. Find list entry/exit signals
    5. Find historical analogs for top stocks
    6. Compute sector historical trends

    Falls back gracefully to empty output when ChromaDB unavailable.
    """
    today = date.today().isoformat()
    logger.info("[Agent 13] Running Historical Analyst pipeline ...")

    # --- Phase 1: Store stats ---
    stats = get_store_stats(db_path)
    store_meta = HistoricalStoreMeta(
        store_available=stats.get("available", False),
        total_snapshots=stats.get("snapshot_count", 0),
        date_range=stats.get("date_range", []),
        total_records=stats.get("total_records", 0),
        unique_symbols=stats.get("unique_symbols", 0),
    )

    logger.info(
        f"[Agent 13] Phase 1: Store available={store_meta.store_available}, "
        f"records={store_meta.total_records}, "
        f"snapshots={store_meta.total_snapshots}"
    )

    if not store_meta.store_available:
        return HistoricalAnalysisOutput(
            store_meta=store_meta,
            analysis_date=today,
            historical_source="empty",
            summary=(
                "Historical analysis unavailable — ChromaDB not installed. "
                "Install chromadb and ingest past IBD PDFs into "
                "data/ibd_history/ to enable historical context analysis."
            ),
        )

    # --- Phase 2: Ingest current week ---
    if auto_ingest_current:
        current_stocks = []
        for s in analyst_output.rated_stocks:
            stock_dict: dict = {
                "symbol": s.symbol,
                "company_name": s.company_name,
                "sector": s.sector,
                "composite_rating": s.composite_rating,
                "rs_rating": s.rs_rating,
                "eps_rating": s.eps_rating,
            }
            if hasattr(s, "smr_rating") and s.smr_rating:
                stock_dict["smr_rating"] = s.smr_rating
            if hasattr(s, "acc_dis_rating") and s.acc_dis_rating:
                stock_dict["acc_dis_rating"] = s.acc_dis_rating
            if hasattr(s, "ibd_lists") and s.ibd_lists:
                stock_dict["ibd_lists"] = s.ibd_lists
            current_stocks.append(stock_dict)

        ingest_result = ingest_snapshot(
            current_stocks, today, "pipeline_current", db_path
        )
        logger.info(
            f"[Agent 13] Phase 2: Ingested {ingest_result.get('stocks_added', 0)} "
            f"current stocks"
        )
        # Refresh stats after ingestion
        stats = get_store_stats(db_path)
        store_meta = HistoricalStoreMeta(
            store_available=stats.get("available", False),
            total_snapshots=stats.get("snapshot_count", 0),
            date_range=stats.get("date_range", []),
            total_records=stats.get("total_records", 0),
            unique_symbols=stats.get("unique_symbols", 0),
        )

    if store_meta.total_records == 0:
        return HistoricalAnalysisOutput(
            store_meta=store_meta,
            analysis_date=today,
            historical_source="empty",
            summary=(
                "Historical analysis unavailable — no historical data in "
                "ChromaDB store. Ingest past IBD PDFs into data/ibd_history/ "
                "and run ingestion to enable historical context analysis."
            ),
        )

    # --- Phase 3: Per-stock historical context ---
    stock_analyses: list[StockHistoricalContext] = []
    improving: list[str] = []
    deteriorating: list[str] = []

    stocks_to_analyze = analyst_output.rated_stocks[:50]
    for stock in stocks_to_analyze:
        history = search_rating_history(stock.symbol, db_path=db_path)

        if not history:
            stock_analyses.append(StockHistoricalContext(
                symbol=stock.symbol,
                weeks_tracked=0,
                momentum_direction="stable",
                momentum_score=0.0,
                notes="No historical data available",
            ))
            continue

        trends = _build_rating_trends(history)
        direction, score = _classify_momentum(history)

        signals = search_list_signals(symbol=stock.symbol, db_path=db_path)
        entries = [s for s in signals if s["event_type"] == "ENTRY"]
        exits = [s for s in signals if s["event_type"] == "EXIT"]

        first_seen = history[0]["snapshot_date"] if history else None

        notes = f"Tracked {len(history)} weeks. "
        if direction == "improving":
            notes += f"Momentum score +{score:.0f} — ratings trending up. "
            improving.append(stock.symbol)
        elif direction == "deteriorating":
            notes += f"Momentum score {score:.0f} — ratings trending down. "
            deteriorating.append(stock.symbol)
        else:
            notes += "Ratings stable over tracked period. "

        if entries:
            notes += f"{len(entries)} list entry event(s). "
        if exits:
            notes += f"{len(exits)} list exit event(s). "

        stock_analyses.append(StockHistoricalContext(
            symbol=stock.symbol,
            weeks_tracked=len(history),
            rating_trends=trends,
            list_entries=entries,
            list_exits=exits,
            momentum_direction=direction,
            momentum_score=score,
            first_seen_date=first_seen,
            notes=notes,
        ))

    logger.info(
        f"[Agent 13] Phase 3: Analyzed {len(stock_analyses)} stocks — "
        f"{len(improving)} improving, {len(deteriorating)} deteriorating"
    )

    # --- Phase 4: Historical analogs for T1 stocks ---
    analogs: list[HistoricalAnalog] = []
    t1_stocks = [s for s in analyst_output.rated_stocks if s.tier == 1][:10]
    for stock in t1_stocks:
        profile = {
            "symbol": stock.symbol,
            "company_name": stock.company_name,
            "sector": stock.sector,
            "composite_rating": stock.composite_rating,
            "rs_rating": stock.rs_rating,
            "eps_rating": stock.eps_rating,
        }
        if hasattr(stock, "smr_rating") and stock.smr_rating:
            profile["smr_rating"] = stock.smr_rating
        if hasattr(stock, "acc_dis_rating") and stock.acc_dis_rating:
            profile["acc_dis_rating"] = stock.acc_dis_rating

        similar = find_similar_setups(profile, top_n=3, db_path=db_path)
        for s in similar:
            analogs.append(HistoricalAnalog(
                symbol=s["symbol"],
                analog_date=s["snapshot_date"],
                sector=s.get("sector", "UNKNOWN"),
                composite_rating=s.get("composite_rating"),
                rs_rating=s.get("rs_rating"),
                eps_rating=s.get("eps_rating"),
                similarity_score=s.get("similarity_score", 0),
                current_symbol=stock.symbol,
                context=(
                    f"Historical analog for {stock.symbol} "
                    f"from {s['snapshot_date']}"
                ),
            ))

    logger.info(
        f"[Agent 13] Phase 4: Found {len(analogs)} historical analogs "
        f"for {len(t1_stocks)} T1 stocks"
    )

    # --- Phase 5: Sector trends ---
    sector_trends: list[SectorHistoricalTrend] = []
    active_sectors = set(s.sector for s in stocks_to_analyze)
    for sector in sorted(active_sectors):
        snapshots = get_sector_trends(sector, db_path=db_path)
        if not snapshots:
            continue

        sector_trends.append(SectorHistoricalTrend(
            sector=sector,
            snapshots=snapshots,
            avg_rs_trend=_classify_sector_trend(snapshots, "avg_rs"),
            stock_count_trend=_classify_sector_trend(
                snapshots, "stock_count"
            ),
            elite_pct_trend=_classify_sector_trend(
                snapshots, "elite_pct"
            ),
            notes=(
                f"{len(snapshots)} weeks of data for {sector} sector"
            ),
        ))

    logger.info(
        f"[Agent 13] Phase 5: Tracked {len(sector_trends)} sector trends"
    )

    # --- Phase 6: Recent list activity ---
    four_weeks_ago = (date.today() - timedelta(weeks=4)).isoformat()
    recent_signals = search_list_signals(
        date_from=four_weeks_ago, db_path=db_path
    )
    new_to_lists = sorted(set(
        s["symbol"] for s in recent_signals
        if s["event_type"] == "ENTRY"
    ))
    dropped = sorted(set(
        s["symbol"] for s in recent_signals
        if s["event_type"] == "EXIT"
    ))

    summary = (
        f"Historical analysis across {store_meta.total_snapshots} weekly "
        f"snapshots ({store_meta.total_records} records, "
        f"{store_meta.unique_symbols} unique symbols). "
        f"Analyzed {len(stock_analyses)} stocks: {len(improving)} improving, "
        f"{len(deteriorating)} deteriorating. "
        f"Found {len(analogs)} historical analogs for T1 stocks. "
        f"Tracked {len(sector_trends)} sector trends."
    )

    output = HistoricalAnalysisOutput(
        store_meta=store_meta,
        stock_analyses=stock_analyses,
        improving_stocks=improving,
        deteriorating_stocks=deteriorating,
        historical_analogs=analogs,
        sector_trends=sector_trends,
        new_to_ibd_lists=new_to_lists,
        dropped_from_ibd_lists=dropped,
        analysis_date=today,
        historical_source="chromadb",
        summary=summary,
    )

    logger.info(
        f"[Agent 13] Pipeline complete: {len(stock_analyses)} stocks, "
        f"{len(analogs)} analogs, {len(sector_trends)} sectors, "
        f"source=chromadb"
    )
    return output
