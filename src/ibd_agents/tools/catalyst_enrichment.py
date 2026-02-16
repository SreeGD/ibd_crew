"""
Analyst Agent Tool: Catalyst Enrichment
Look up upcoming corporate catalysts (earnings, FDA approvals, product
launches, conferences) using LLM knowledge and compute conviction
adjustments based on catalyst timing.

Follows the same dual-path pattern as valuation_metrics.py:
deterministic fallback + optional LLM enrichment.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATALYST_TYPES = (
    "earnings", "fda", "product_launch", "conference",
    "guidance", "dividend", "split", "other",
)


# ---------------------------------------------------------------------------
# Conviction Adjustment
# ---------------------------------------------------------------------------

def compute_catalyst_adjustment(days_until: int | None) -> int:
    """
    Conviction adjustment based on days until next catalyst.

    <= 14 days: +2  (imminent catalyst — high urgency)
    <= 30 days: +1  (near-term catalyst)
    > 30 or None: 0 (no timing edge)
    """
    if days_until is None:
        return 0
    if days_until <= 14:
        return 2
    if days_until <= 30:
        return 1
    return 0


# ---------------------------------------------------------------------------
# LLM Catalyst Enrichment
# ---------------------------------------------------------------------------

_CATALYST_PROMPT_TEMPLATE = """\
For each stock below, identify the NEXT upcoming corporate catalyst event.
Return a JSON array with one object per stock. Use ONLY valid JSON — no markdown, no code fences, no extra text.

Fields per stock:
- symbol: str (ticker, uppercase)
- catalyst_date: str (YYYY-MM-DD) or null if unknown
- catalyst_type: str (one of: "earnings", "fda", "product_launch", "conference", "guidance", "dividend", "split", "other")
- catalyst_description: str (1-2 sentence description of the catalyst event)
- days_until: int or null (approximate days from {today} until the catalyst)

Focus on:
1. Next earnings report date (most common catalyst)
2. FDA approval decisions or PDUFA dates (for biotech/pharma)
3. Product launches or major announcements
4. Analyst day / investor conferences
5. Guidance updates

If you don't know the exact date, provide your best estimate. If no catalyst is identifiable, use null for date and days_until, and set catalyst_type to "other" with a brief description.

Stocks:
{stock_list}
"""


def enrich_catalyst_llm(
    stocks: list[dict],
    model: str = "claude-haiku-4-5-20251001",
    batch_size: int = 30,
) -> dict[str, dict]:
    """
    Look up next corporate catalyst from LLM knowledge.

    Args:
        stocks: List of dicts with keys: symbol, sector, company_name,
                composite_rating, rs_rating, eps_rating, tier.
        model: Anthropic model ID.
        batch_size: Stocks per LLM call.

    Returns:
        {symbol: {catalyst_date, catalyst_type, catalyst_description, days_until}}
        for each stock the LLM provided data for.
        Returns empty dict on any failure (graceful fallback).
    """
    if not stocks:
        return {}

    result: dict[str, dict] = {}
    today_str = date.today().isoformat()
    _t0 = time.monotonic()

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0)

        for i in range(0, len(stocks), batch_size):
            if time.monotonic() - _t0 > 60:
                logger.warning(f"LLM catalyst enrichment timed out after 60s, returning {len(result)} partial results")
                break
            batch = stocks[i:i + batch_size]

            stock_lines = []
            for s in batch:
                stock_lines.append(
                    f"  {s['symbol']} | {s.get('company_name', s['symbol'])} | "
                    f"Sector: {s.get('sector', 'UNKNOWN')} | "
                    f"Tier: {s.get('tier', '?')}"
                )
            stock_list_str = "\n".join(stock_lines)
            prompt = _CATALYST_PROMPT_TEMPLATE.format(
                today=today_str,
                stock_list=stock_list_str,
            )

            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else ""

            parsed = _parse_catalyst_response(text, {s["symbol"] for s in batch})
            result.update(parsed)

            logger.info(
                f"LLM catalyst batch {i // batch_size + 1}: "
                f"enriched {len(parsed)}/{len(batch)} stocks"
            )

    except ImportError:
        logger.warning("anthropic SDK not installed — skipping LLM catalyst enrichment")
    except Exception as e:
        logger.warning(f"LLM catalyst enrichment error: {e}")

    return result


def _parse_catalyst_response(text: str, valid_symbols: set[str]) -> dict[str, dict]:
    """Parse and validate LLM JSON response into {symbol: catalyst_info} dict."""
    result: dict[str, dict] = {}

    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.warning("LLM catalyst response did not contain a JSON array")
        return result

    json_str = text[start:end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM catalyst JSON: {e}")
        return result

    if not isinstance(data, list):
        logger.warning("LLM catalyst JSON is not an array")
        return result

    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    for item in data:
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", "")).strip().upper()
        if not sym or sym not in valid_symbols:
            continue

        cleaned: dict = {"symbol": sym}

        # Validate catalyst_date
        cat_date = item.get("catalyst_date")
        if cat_date is not None:
            cat_date = str(cat_date).strip()
            if date_pattern.match(cat_date):
                # Verify it's a real date
                try:
                    datetime.strptime(cat_date, "%Y-%m-%d")
                    cleaned["catalyst_date"] = cat_date
                except ValueError:
                    cleaned["catalyst_date"] = None
            else:
                cleaned["catalyst_date"] = None
        else:
            cleaned["catalyst_date"] = None

        # Validate catalyst_type
        cat_type = item.get("catalyst_type")
        if cat_type is not None and str(cat_type) in CATALYST_TYPES:
            cleaned["catalyst_type"] = str(cat_type)
        else:
            cleaned["catalyst_type"] = "other"

        # catalyst_description
        desc = item.get("catalyst_description")
        cleaned["catalyst_description"] = str(desc) if desc else None

        # days_until
        days = item.get("days_until")
        if days is not None:
            try:
                days = int(days)
                if days < 0:
                    days = 0
                if days > 365:
                    days = 365
                cleaned["days_until"] = days
            except (ValueError, TypeError):
                cleaned["days_until"] = None
        else:
            cleaned["days_until"] = None

        result[sym] = cleaned

    return result
