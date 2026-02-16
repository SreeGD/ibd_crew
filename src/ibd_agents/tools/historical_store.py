"""
Historical IBD RAG Store
ChromaDB-backed vector store for historical IBD weekly snapshots.

Two collections:
- ibd_snapshots: One document per stock-per-week (for rating history, list signals, similarity)
- ibd_sector_snapshots: One document per sector-per-week (for sector trends)

Follows dual-path pattern: ChromaDB when available, empty fallback otherwise.
Never makes buy/sell recommendations — only discovers, scores, and organizes.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ChromaDB optional import (same pattern as crewai)
try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    chromadb = None  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHROMA_DB_PATH = "data/chroma_db"
STOCK_COLLECTION = "ibd_snapshots"
SECTOR_COLLECTION = "ibd_sector_snapshots"

# Date extraction patterns for IBD filenames
# "IBD Smart NYSE + Nasdaq Tables -- Feb 13, 2026 _ Investor's Business Daily.pdf"
_MONTH_MAP = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}
_DATE_PATTERN_MONTH = re.compile(
    r'(' + '|'.join(_MONTH_MAP.keys()) + r')\s+(\d{1,2}),?\s+(\d{4})'
)
_DATE_PATTERN_ISO = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
_DATE_PATTERN_UNDERSCORE = re.compile(r'(\d{2})_(\d{2})_(\d{4})')


def _date_to_int(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to integer YYYYMMDD for ChromaDB numeric filtering."""
    return int(date_str.replace("-", ""))


def _int_to_date(date_int: int) -> str:
    """Convert integer YYYYMMDD back to 'YYYY-MM-DD' string."""
    s = str(date_int)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


# ---------------------------------------------------------------------------
# Date Extraction
# ---------------------------------------------------------------------------

def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Extract YYYY-MM-DD date from an IBD filename.

    Supports:
      - "Feb 13, 2026" -> "2026-02-13"
      - "2026-02-13" -> "2026-02-13"
      - "02_13_2026" -> "2026-02-13"

    Returns None if no date found.
    """
    m = _DATE_PATTERN_MONTH.search(filename)
    if m:
        month_str = _MONTH_MAP.get(m.group(1))
        if month_str:
            day = m.group(2).zfill(2)
            year = m.group(3)
            return f"{year}-{month_str}-{day}"

    m = _DATE_PATTERN_ISO.search(filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    m = _DATE_PATTERN_UNDERSCORE.search(filename)
    if m:
        return f"{m.group(3)}-{m.group(1)}-{m.group(2)}"

    return None


# ---------------------------------------------------------------------------
# ChromaDB Client
# ---------------------------------------------------------------------------

def get_chroma_client(db_path: str = CHROMA_DB_PATH):
    """
    Get or create a persistent ChromaDB client.
    Returns None if chromadb is not installed.
    """
    if not HAS_CHROMADB:
        logger.warning("chromadb not installed — historical store unavailable")
        return None

    Path(db_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)


def _get_or_create_collection(client, name: str):
    """Get or create a ChromaDB collection with cosine similarity."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Document Building
# ---------------------------------------------------------------------------

def _build_stock_document(stock: dict, snapshot_date: str) -> str:
    """Build embedding text for a stock snapshot."""
    ibd_lists = stock.get("ibd_lists", [])
    if not ibd_lists and stock.get("ibd_list"):
        ibd_lists = [stock["ibd_list"]]
    lists_str = ", ".join(ibd_lists) if ibd_lists else "none"

    return (
        f"{stock.get('symbol', '?')} {stock.get('company_name', '')} "
        f"in {stock.get('sector', 'UNKNOWN')} sector. "
        f"Composite {stock.get('composite_rating', 'N/A')}, "
        f"RS {stock.get('rs_rating', 'N/A')}, "
        f"EPS {stock.get('eps_rating', 'N/A')}, "
        f"SMR {stock.get('smr_rating', 'N/A')}, "
        f"Acc/Dis {stock.get('acc_dis_rating', 'N/A')}. "
        f"Lists: {lists_str}. "
        f"Date: {snapshot_date}."
    )


def _build_stock_metadata(
    stock: dict, snapshot_date: str, source_file: str
) -> dict:
    """Build metadata dict for a stock snapshot."""
    ibd_lists = stock.get("ibd_lists", [])
    if not ibd_lists and stock.get("ibd_list"):
        ibd_lists = [stock["ibd_list"]]

    return {
        "symbol": str(stock.get("symbol", "")).strip().upper(),
        "snapshot_date": _date_to_int(snapshot_date),
        "sector": str(stock.get("sector", "UNKNOWN")),
        "composite_rating": int(stock.get("composite_rating") or 0),
        "rs_rating": int(stock.get("rs_rating") or 0),
        "eps_rating": int(stock.get("eps_rating") or 0),
        "smr_rating": str(stock.get("smr_rating") or ""),
        "acc_dis_rating": str(stock.get("acc_dis_rating") or ""),
        "ibd_lists": ",".join(ibd_lists) if ibd_lists else "",
        "source_file": source_file,
        "source_type": "pdf" if source_file.endswith(".pdf") else "xls",
    }


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_snapshot(
    stocks: list[dict],
    snapshot_date: str,
    source_file: str = "",
    db_path: str = CHROMA_DB_PATH,
) -> dict:
    """
    Ingest a list of stock dicts into the historical store.

    Args:
        stocks: List of dicts from read_ibd_pdf() or read_ibd_xls().
        snapshot_date: YYYY-MM-DD date string.
        source_file: Name of the source file.
        db_path: ChromaDB storage path.

    Returns:
        {"stocks_added": int, "sectors_updated": int, "snapshot_date": str}
    """
    client = get_chroma_client(db_path)
    if client is None:
        return {"error": "chromadb not available", "stocks_added": 0,
                "sectors_updated": 0, "snapshot_date": snapshot_date}

    stock_col = _get_or_create_collection(client, STOCK_COLLECTION)
    sector_col = _get_or_create_collection(client, SECTOR_COLLECTION)

    # --- Upsert stock records ---
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for s in stocks:
        sym = str(s.get("symbol", "")).strip().upper()
        if not sym:
            continue
        doc_id = f"{sym}_{snapshot_date}"
        ids.append(doc_id)
        documents.append(_build_stock_document(s, snapshot_date))
        metadatas.append(_build_stock_metadata(s, snapshot_date, source_file))

    if ids:
        stock_col.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # --- Compute and upsert sector aggregates ---
    sector_groups: dict[str, list[dict]] = {}
    for s in stocks:
        sec = str(s.get("sector", "UNKNOWN"))
        sector_groups.setdefault(sec, []).append(s)

    sec_ids: list[str] = []
    sec_docs: list[str] = []
    sec_metas: list[dict] = []

    for sector, sec_stocks in sector_groups.items():
        comps = [s["composite_rating"] for s in sec_stocks
                 if s.get("composite_rating")]
        rss = [s["rs_rating"] for s in sec_stocks if s.get("rs_rating")]

        avg_comp = round(sum(comps) / len(comps), 1) if comps else 0.0
        avg_rs = round(sum(rss) / len(rss), 1) if rss else 0.0

        elite_count = sum(
            1 for s in sec_stocks
            if (s.get("eps_rating") or 0) >= 85
            and (s.get("rs_rating") or 0) >= 75
        )
        elite_pct = (
            round(elite_count / len(sec_stocks) * 100, 1)
            if sec_stocks else 0.0
        )
        keep_count = sum(
            1 for s in sec_stocks
            if (s.get("composite_rating") or 0) >= 93
            and (s.get("rs_rating") or 0) >= 90
        )

        sec_id = f"{sector}_{snapshot_date}"
        sec_ids.append(sec_id)
        sec_docs.append(
            f"{sector} sector on {snapshot_date}: {len(sec_stocks)} stocks, "
            f"avg Composite {avg_comp}, avg RS {avg_rs}, "
            f"{elite_pct}% elite, {keep_count} keeps."
        )
        sec_metas.append({
            "sector": sector,
            "snapshot_date": _date_to_int(snapshot_date),
            "stock_count": len(sec_stocks),
            "avg_composite": avg_comp,
            "avg_rs": avg_rs,
            "elite_count": elite_count,
            "elite_pct": elite_pct,
            "keep_count": keep_count,
        })

    if sec_ids:
        sector_col.upsert(ids=sec_ids, documents=sec_docs, metadatas=sec_metas)

    logger.info(
        f"Historical store: ingested {len(ids)} stocks, "
        f"{len(sec_ids)} sectors for {snapshot_date}"
    )
    return {
        "stocks_added": len(ids),
        "sectors_updated": len(sec_ids),
        "snapshot_date": snapshot_date,
    }


def ingest_file(
    file_path: str,
    snapshot_date: Optional[str] = None,
    db_path: str = CHROMA_DB_PATH,
) -> dict:
    """
    Ingest a single IBD PDF or XLS/CSV file.
    Reuses existing pdf_reader and xls_reader functions.
    Auto-extracts date from filename if not provided.
    """
    from ibd_agents.tools.pdf_reader import read_ibd_pdf
    from ibd_agents.tools.xls_reader import read_ibd_csv, read_ibd_xls

    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}", "stocks_added": 0,
                "sectors_updated": 0, "snapshot_date": ""}

    if snapshot_date is None:
        snapshot_date = extract_date_from_filename(path.name)
    if snapshot_date is None:
        snapshot_date = date.today().isoformat()
        logger.warning(
            f"No date in filename '{path.name}', using today: {snapshot_date}"
        )

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        stocks = read_ibd_pdf(file_path)
    elif suffix in (".xls", ".xlsx"):
        stocks = read_ibd_xls(file_path)
    elif suffix == ".csv":
        stocks = read_ibd_csv(file_path)
    else:
        return {"error": f"Unsupported file type: {suffix}",
                "stocks_added": 0, "sectors_updated": 0, "snapshot_date": ""}

    if not stocks:
        return {"error": f"No stocks extracted from {path.name}",
                "stocks_added": 0, "sectors_updated": 0,
                "snapshot_date": snapshot_date}

    return ingest_snapshot(stocks, snapshot_date, path.name, db_path)


def ingest_directory(
    directory: str,
    db_path: str = CHROMA_DB_PATH,
) -> dict:
    """
    Batch ingest all IBD files in a directory.
    Supports: data/ibd_history/ for past data, data/ibd_pdf/ for current.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return {"error": f"Directory not found: {directory}",
                "files_processed": 0, "total_stocks_added": 0}

    extensions = {".pdf", ".xls", ".xlsx", ".csv"}
    files = sorted(f for f in dir_path.iterdir()
                   if f.suffix.lower() in extensions)

    results: list[dict] = []
    for f in files:
        result = ingest_file(str(f), db_path=db_path)
        results.append({"file": f.name, **result})

    total_stocks = sum(r.get("stocks_added", 0) for r in results)
    return {
        "files_processed": len(results),
        "total_stocks_added": total_stocks,
        "details": results,
    }


# ---------------------------------------------------------------------------
# Query: Rating History
# ---------------------------------------------------------------------------

def search_rating_history(
    symbol: str,
    metric: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db_path: str = CHROMA_DB_PATH,
) -> list[dict]:
    """
    Get rating history for a symbol over time.

    Args:
        symbol: Ticker symbol.
        metric: Optional filter to specific metric (rs/composite/eps).
        date_from: Optional YYYY-MM-DD start date.
        date_to: Optional YYYY-MM-DD end date.

    Returns:
        List of dicts sorted by date, each with snapshot_date + ratings.
    """
    client = get_chroma_client(db_path)
    if client is None:
        return []

    stock_col = _get_or_create_collection(client, STOCK_COLLECTION)

    where_conditions: list[dict] = [{"symbol": symbol.upper()}]
    if date_from:
        where_conditions.append(
            {"snapshot_date": {"$gte": _date_to_int(date_from)}}
        )
    if date_to:
        where_conditions.append(
            {"snapshot_date": {"$lte": _date_to_int(date_to)}}
        )

    where = (
        where_conditions[0]
        if len(where_conditions) == 1
        else {"$and": where_conditions}
    )

    results = stock_col.get(where=where, include=["metadatas"])

    if not results["metadatas"]:
        return []

    records = sorted(results["metadatas"], key=lambda m: m["snapshot_date"])

    output: list[dict] = []
    for m in records:
        date_str = _int_to_date(m["snapshot_date"])
        if metric:
            key = f"{metric}_rating"
            output.append({
                "snapshot_date": date_str,
                key: m.get(key),
            })
        else:
            output.append({
                "snapshot_date": date_str,
                "composite_rating": m.get("composite_rating"),
                "rs_rating": m.get("rs_rating"),
                "eps_rating": m.get("eps_rating"),
                "smr_rating": m.get("smr_rating"),
                "acc_dis_rating": m.get("acc_dis_rating"),
                "sector": m.get("sector"),
                "ibd_lists": (
                    m.get("ibd_lists", "").split(",")
                    if m.get("ibd_lists") else []
                ),
            })

    return output


# ---------------------------------------------------------------------------
# Query: List Entry/Exit Signals
# ---------------------------------------------------------------------------

def search_list_signals(
    symbol: Optional[str] = None,
    list_name: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db_path: str = CHROMA_DB_PATH,
) -> list[dict]:
    """
    Find entry/exit events — when stocks appeared on or dropped off IBD lists.

    Returns:
        List of {symbol, event_type: "ENTRY"/"EXIT", list_name, date, ...}
        sorted by date.
    """
    client = get_chroma_client(db_path)
    if client is None:
        return []

    stock_col = _get_or_create_collection(client, STOCK_COLLECTION)

    where_conditions: list[dict] = []
    if symbol:
        where_conditions.append({"symbol": symbol.upper()})
    if date_from:
        where_conditions.append(
            {"snapshot_date": {"$gte": _date_to_int(date_from)}}
        )
    if date_to:
        where_conditions.append(
            {"snapshot_date": {"$lte": _date_to_int(date_to)}}
        )

    where = None
    if len(where_conditions) == 1:
        where = where_conditions[0]
    elif len(where_conditions) > 1:
        where = {"$and": where_conditions}

    results = stock_col.get(where=where, include=["metadatas"])

    if not results["metadatas"]:
        return []

    # Group by symbol and sort by date
    by_symbol: dict[str, list[dict]] = {}
    for m in results["metadatas"]:
        sym = m["symbol"]
        by_symbol.setdefault(sym, []).append(m)

    for sym in by_symbol:
        by_symbol[sym].sort(key=lambda m: m["snapshot_date"])

    # Detect entry/exit events
    signals: list[dict] = []
    for sym, snapshots in by_symbol.items():
        for i, snap in enumerate(snapshots):
            current_lists = set(snap.get("ibd_lists", "").split(",")) - {""}
            date_str = _int_to_date(snap["snapshot_date"])

            if i == 0:
                # First snapshot: each list is an entry
                for lst in current_lists:
                    if list_name and list_name.lower() not in lst.lower():
                        continue
                    signals.append({
                        "symbol": sym,
                        "event_type": "ENTRY",
                        "list_name": lst,
                        "date": date_str,
                        "composite_rating": snap.get("composite_rating"),
                        "rs_rating": snap.get("rs_rating"),
                    })
            else:
                prev_lists = (
                    set(snapshots[i - 1].get("ibd_lists", "").split(","))
                    - {""}
                )
                entries = current_lists - prev_lists
                exits = prev_lists - current_lists

                for lst in entries:
                    if list_name and list_name.lower() not in lst.lower():
                        continue
                    signals.append({
                        "symbol": sym,
                        "event_type": "ENTRY",
                        "list_name": lst,
                        "date": date_str,
                        "composite_rating": snap.get("composite_rating"),
                        "rs_rating": snap.get("rs_rating"),
                    })
                for lst in exits:
                    if list_name and list_name.lower() not in lst.lower():
                        continue
                    signals.append({
                        "symbol": sym,
                        "event_type": "EXIT",
                        "list_name": lst,
                        "date": date_str,
                        "composite_rating": snap.get("composite_rating"),
                        "rs_rating": snap.get("rs_rating"),
                    })

    signals.sort(key=lambda s: s["date"])
    return signals


# ---------------------------------------------------------------------------
# Query: Similar Historical Setups
# ---------------------------------------------------------------------------

def find_similar_setups(
    profile: dict,
    top_n: int = 10,
    db_path: str = CHROMA_DB_PATH,
) -> list[dict]:
    """
    Find historically similar stock profiles using semantic search.

    Args:
        profile: Dict with symbol, sector, composite_rating, rs_rating, etc.
        top_n: Number of similar results to return.

    Returns:
        List of similar historical stock snapshots with similarity scores.
    """
    client = get_chroma_client(db_path)
    if client is None:
        return []

    stock_col = _get_or_create_collection(client, STOCK_COLLECTION)

    query_text = _build_stock_document(profile, date.today().isoformat())

    try:
        results = stock_col.query(
            query_texts=[query_text],
            n_results=min(top_n + 10, stock_col.count()),
            include=["metadatas", "distances"],
        )
    except Exception as e:
        logger.warning(f"Similarity search failed: {e}")
        return []

    if not results["metadatas"] or not results["metadatas"][0]:
        return []

    output: list[dict] = []
    current_symbol = str(profile.get("symbol", "")).upper()

    for meta, distance in zip(
        results["metadatas"][0], results["distances"][0]
    ):
        # Skip self-matches
        if meta.get("symbol") == current_symbol:
            continue

        similarity = round(1.0 - distance, 4)
        output.append({
            "symbol": meta.get("symbol"),
            "snapshot_date": _int_to_date(meta["snapshot_date"]),
            "sector": meta.get("sector"),
            "composite_rating": meta.get("composite_rating"),
            "rs_rating": meta.get("rs_rating"),
            "eps_rating": meta.get("eps_rating"),
            "smr_rating": meta.get("smr_rating"),
            "acc_dis_rating": meta.get("acc_dis_rating"),
            "ibd_lists": (
                meta.get("ibd_lists", "").split(",")
                if meta.get("ibd_lists") else []
            ),
            "similarity_score": similarity,
        })

        if len(output) >= top_n:
            break

    return output


# ---------------------------------------------------------------------------
# Query: Sector Trends
# ---------------------------------------------------------------------------

def get_sector_trends(
    sector: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db_path: str = CHROMA_DB_PATH,
) -> list[dict]:
    """
    Get sector-level metrics over time.

    Returns:
        List of dicts sorted by date with avg_rs, stock_count, elite_pct, etc.
    """
    client = get_chroma_client(db_path)
    if client is None:
        return []

    sector_col = _get_or_create_collection(client, SECTOR_COLLECTION)

    where_conditions: list[dict] = [{"sector": sector}]
    if date_from:
        where_conditions.append(
            {"snapshot_date": {"$gte": _date_to_int(date_from)}}
        )
    if date_to:
        where_conditions.append(
            {"snapshot_date": {"$lte": _date_to_int(date_to)}}
        )

    where = (
        where_conditions[0]
        if len(where_conditions) == 1
        else {"$and": where_conditions}
    )

    results = sector_col.get(where=where, include=["metadatas"])

    if not results["metadatas"]:
        return []

    records = sorted(results["metadatas"], key=lambda m: m["snapshot_date"])
    # Convert snapshot_date ints back to strings
    for r in records:
        r["snapshot_date"] = _int_to_date(r["snapshot_date"])
    return records


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_snapshot_dates(db_path: str = CHROMA_DB_PATH) -> list[str]:
    """Return all unique snapshot dates in the store, sorted ascending."""
    client = get_chroma_client(db_path)
    if client is None:
        return []

    stock_col = _get_or_create_collection(client, STOCK_COLLECTION)
    results = stock_col.get(include=["metadatas"])

    if not results["metadatas"]:
        return []

    date_ints = sorted(set(
        m["snapshot_date"]
        for m in results["metadatas"]
        if m.get("snapshot_date")
    ))
    return [_int_to_date(d) for d in date_ints]


def get_store_stats(db_path: str = CHROMA_DB_PATH) -> dict:
    """Return basic stats about the historical store."""
    client = get_chroma_client(db_path)
    if client is None:
        return {
            "available": False,
            "total_records": 0,
            "snapshot_count": 0,
            "date_range": [],
            "unique_symbols": 0,
        }

    stock_col = _get_or_create_collection(client, STOCK_COLLECTION)
    count = stock_col.count()

    if count == 0:
        return {
            "available": True,
            "total_records": 0,
            "snapshot_count": 0,
            "date_range": [],
            "unique_symbols": 0,
        }

    results = stock_col.get(include=["metadatas"])
    date_ints = sorted(set(
        m["snapshot_date"]
        for m in results["metadatas"]
        if m.get("snapshot_date")
    ))
    symbols = set(
        m["symbol"]
        for m in results["metadatas"]
        if m.get("symbol")
    )

    date_strs = [_int_to_date(d) for d in date_ints]
    return {
        "available": True,
        "total_records": count,
        "snapshot_count": len(date_strs),
        "date_range": [date_strs[0], date_strs[-1]] if date_strs else [],
        "unique_symbols": len(symbols),
    }


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class HistoricalStoreInput(BaseModel):
    action: str = Field(
        ...,
        description=(
            "Action: 'search_rating_history', 'search_list_signals', "
            "'find_similar_setups', 'get_sector_trends', 'ingest', 'stats'"
        ),
    )
    symbol: Optional[str] = Field(None, description="Ticker symbol")
    sector: Optional[str] = Field(None, description="IBD sector name")
    metric: Optional[str] = Field(None, description="rs/composite/eps")
    date_from: Optional[str] = Field(None, description="YYYY-MM-DD start")
    date_to: Optional[str] = Field(None, description="YYYY-MM-DD end")
    file_path: Optional[str] = Field(None, description="File or dir to ingest")


class HistoricalStoreTool(BaseTool):
    """Query or ingest data from the Historical IBD RAG store."""

    name: str = "historical_ibd_store"
    description: str = (
        "Historical IBD RAG store. Actions: search_rating_history (rating "
        "time series), search_list_signals (entry/exit events), "
        "find_similar_setups (semantic search), get_sector_trends (sector "
        "metrics over time), ingest (add files), stats (store status)."
    )
    args_schema: type[BaseModel] = HistoricalStoreInput

    def _run(
        self,
        action: str,
        symbol: Optional[str] = None,
        sector: Optional[str] = None,
        metric: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> str:
        import json

        if action == "search_rating_history" and symbol:
            result = search_rating_history(
                symbol, metric, date_from, date_to
            )
        elif action == "search_list_signals":
            result = search_list_signals(symbol, date_from=date_from,
                                         date_to=date_to)
        elif action == "find_similar_setups" and symbol:
            profile = {"symbol": symbol, "sector": sector or "UNKNOWN"}
            result = find_similar_setups(profile)
        elif action == "get_sector_trends" and sector:
            result = get_sector_trends(sector, date_from, date_to)
        elif action == "ingest" and file_path:
            p = Path(file_path)
            if p.is_dir():
                result = ingest_directory(str(p))
            else:
                result = ingest_file(str(p))
        elif action == "stats":
            result = get_store_stats()
        else:
            result = {"error": f"Invalid action or missing params: {action}"}

        return json.dumps(result, default=str, indent=2)
