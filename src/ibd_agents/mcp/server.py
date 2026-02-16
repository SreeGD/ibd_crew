"""
IBD Historical RAG — MCP Server
Exposes 5 tools via stdio transport for use with Claude Desktop / Claude Code.

Tools:
1. search_rating_history — rating time series per stock
2. search_list_signals — IBD list ENTRY/EXIT events
3. find_similar_setups — semantic similarity search
4. get_sector_trends — sector metrics over time
5. ingest_snapshot — ingest a file or directory

Run: python -m ibd_agents.mcp.server
"""

from __future__ import annotations

import json
import logging
import sys

logger = logging.getLogger(__name__)

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from ibd_agents.tools.historical_store import (
    CHROMA_DB_PATH,
    find_similar_setups,
    get_sector_trends,
    ingest_directory,
    ingest_file,
    search_list_signals,
    search_rating_history,
)


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_rating_history",
        "description": (
            "Get weekly IBD rating history for a stock symbol over time. "
            "Returns composite, RS, EPS ratings per snapshot date."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol (e.g. NVDA, AAPL)",
                },
                "metric": {
                    "type": "string",
                    "description": "Optional: filter to rs/composite/eps",
                    "enum": ["rs", "composite", "eps"],
                },
                "date_from": {
                    "type": "string",
                    "description": "Optional: YYYY-MM-DD start date",
                },
                "date_to": {
                    "type": "string",
                    "description": "Optional: YYYY-MM-DD end date",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "search_list_signals",
        "description": (
            "Find when stocks entered or exited IBD lists (IBD 50, "
            "Tech Leaders, etc). Returns ENTRY/EXIT events with dates."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Optional: filter to specific ticker",
                },
                "list_name": {
                    "type": "string",
                    "description": "Optional: filter to specific IBD list name",
                },
                "date_from": {
                    "type": "string",
                    "description": "Optional: YYYY-MM-DD start date",
                },
                "date_to": {
                    "type": "string",
                    "description": "Optional: YYYY-MM-DD end date",
                },
            },
        },
    },
    {
        "name": "find_similar_setups",
        "description": (
            "Find historically similar stock setups using semantic search. "
            "Provide a stock profile and get back past stocks with similar "
            "IBD rating patterns."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol to find analogs for",
                },
                "sector": {
                    "type": "string",
                    "description": "IBD sector name",
                },
                "composite_rating": {
                    "type": "integer",
                    "description": "IBD Composite Rating (0-99)",
                },
                "rs_rating": {
                    "type": "integer",
                    "description": "IBD RS Rating (0-99)",
                },
                "eps_rating": {
                    "type": "integer",
                    "description": "IBD EPS Rating (0-99)",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of results (default 10)",
                    "default": 10,
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_sector_trends",
        "description": (
            "Get sector-level metrics over time: average RS, stock count, "
            "elite percentage, keep count per week."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sector": {
                    "type": "string",
                    "description": "IBD sector name (e.g. CHIPS, SOFTWARE)",
                },
                "date_from": {
                    "type": "string",
                    "description": "Optional: YYYY-MM-DD start date",
                },
                "date_to": {
                    "type": "string",
                    "description": "Optional: YYYY-MM-DD end date",
                },
            },
            "required": ["sector"],
        },
    },
    {
        "name": "ingest_snapshot",
        "description": (
            "Ingest an IBD PDF, XLS, or CSV file (or directory of files) "
            "into the historical RAG store. Date is auto-extracted from "
            "the filename."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Path to an IBD file or directory. "
                        "Supports .pdf, .xls, .xlsx, .csv"
                    ),
                },
                "snapshot_date": {
                    "type": "string",
                    "description": (
                        "Optional: YYYY-MM-DD override "
                        "(auto-extracted from filename if omitted)"
                    ),
                },
            },
            "required": ["file_path"],
        },
    },
]


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def create_server(db_path: str = CHROMA_DB_PATH) -> "Server":
    """Create and configure the MCP server with all 5 tools."""
    if not HAS_MCP:
        raise ImportError(
            "mcp package is required. Install with: pip install mcp"
        )

    server = Server("ibd-historical-rag")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [Tool(**t) for t in TOOLS]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            result = _dispatch(name, arguments, db_path)
            text = json.dumps(result, default=str, indent=2)
        except Exception as e:
            text = json.dumps({"error": str(e)})

        return [TextContent(type="text", text=text)]

    return server


def _dispatch(name: str, args: dict, db_path: str) -> dict | list:
    """Route tool calls to the appropriate store function."""
    if name == "search_rating_history":
        return search_rating_history(
            symbol=args["symbol"],
            metric=args.get("metric"),
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
            db_path=db_path,
        )
    elif name == "search_list_signals":
        return search_list_signals(
            symbol=args.get("symbol"),
            list_name=args.get("list_name"),
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
            db_path=db_path,
        )
    elif name == "find_similar_setups":
        profile = {
            "symbol": args["symbol"],
            "sector": args.get("sector", "UNKNOWN"),
            "composite_rating": args.get("composite_rating"),
            "rs_rating": args.get("rs_rating"),
            "eps_rating": args.get("eps_rating"),
        }
        return find_similar_setups(
            profile=profile,
            top_n=args.get("top_n", 10),
            db_path=db_path,
        )
    elif name == "get_sector_trends":
        return get_sector_trends(
            sector=args["sector"],
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
            db_path=db_path,
        )
    elif name == "ingest_snapshot":
        from pathlib import Path

        path = Path(args["file_path"])
        if path.is_dir():
            return ingest_directory(str(path), db_path=db_path)
        else:
            return ingest_file(
                str(path),
                snapshot_date=args.get("snapshot_date"),
                db_path=db_path,
            )
    else:
        return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

async def main():
    """Run the MCP server with stdio transport."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    asyncio.run(main())
