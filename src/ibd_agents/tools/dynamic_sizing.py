"""
Portfolio Manager Tool: Dynamic Position Sizing
Adjust position sizes based on LLM-assessed volatility risk,
catalyst timing, and sector characteristics.

Follows the same dual-path pattern as catalyst_enrichment.py:
deterministic fallback + optional LLM enrichment.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum adjustment magnitude per tier
MAX_ADJUSTMENT: dict[int, float] = {
    1: 1.5,   # T1 momentum — allow ±1.5%
    2: 1.0,   # T2 quality — allow ±1.0%
    3: 0.5,   # T3 defensive — allow ±0.5%
}


# ---------------------------------------------------------------------------
# Deterministic Helpers
# ---------------------------------------------------------------------------

def compute_size_adjustment(llm_adjustment: float, tier: int) -> float:
    """
    Clamp LLM size adjustment to tier-valid range.
    Returns float in [-2.0, 2.0] clamped to tier max.
    """
    limit = MAX_ADJUSTMENT.get(tier, 1.0)
    return round(max(-limit, min(llm_adjustment, limit)), 2)


# ---------------------------------------------------------------------------
# LLM Dynamic Sizing
# ---------------------------------------------------------------------------

_SIZING_PROMPT_TEMPLATE = """\
For each position below, assess volatility risk and recommend a position size adjustment.
A positive adjustment means INCREASE position size (lower volatility risk).
A negative adjustment means DECREASE position size (higher volatility risk).

Return a JSON array with one object per position. Use ONLY valid JSON — no markdown, no code fences, no extra text.

Fields per position:
- symbol: str (ticker, uppercase)
- volatility_score: int (1-10, where 10 = highest volatility risk)
- size_adjustment_pct: float (between -2.0 and +2.0, the % to add/subtract from target)
- reason: str (1-2 sentence explanation)

Guidelines:
- Stocks with imminent catalysts (earnings in <14 days): consider reducing size (-0.5 to -1.0%)
- Low-volatility defensive names: consider increasing size (+0.5 to +1.0%)
- High-beta momentum stocks: may need size reduction if concentrated
- ETFs generally have lower volatility than individual stocks
- T1 momentum positions can tolerate larger adjustments than T3 defensive

Positions:
{position_list}
"""


def enrich_sizing_llm(
    positions: list[dict],
    model: str = "claude-haiku-4-5-20251001",
    batch_size: int = 30,
) -> dict[str, dict]:
    """
    Get LLM volatility-based position sizing adjustments.

    Args:
        positions: List of dicts with keys: symbol, sector, tier,
            conviction, target_pct, asset_type, catalyst_date.
        model: Anthropic model ID.
        batch_size: Positions per LLM call.

    Returns:
        {symbol: {volatility_score, size_adjustment_pct, reason}}
        Empty dict on failure.
    """
    if not positions:
        return {}

    result: dict[str, dict] = {}
    _t0 = time.monotonic()

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0)

        for i in range(0, len(positions), batch_size):
            if time.monotonic() - _t0 > 60:
                logger.warning(f"LLM dynamic sizing timed out after 60s, returning {len(result)} partial results")
                break
            batch = positions[i:i + batch_size]

            pos_lines = []
            for p in batch:
                catalyst_info = f"Catalyst: {p.get('catalyst_date', 'unknown')}" if p.get("catalyst_date") else "No known catalyst"
                cap_info = p.get('cap_size', 'N/A') or 'N/A'
                pos_lines.append(
                    f"  {p['symbol']} | Sector: {p.get('sector', 'UNKNOWN')} | "
                    f"Cap: {cap_info} | "
                    f"T{p.get('tier', '?')} | Target: {p.get('target_pct', '?')}% | "
                    f"Type: {p.get('asset_type', 'stock')} | "
                    f"Conviction: {p.get('conviction', '?')}/10 | {catalyst_info}"
                )
            position_list_str = "\n".join(pos_lines)
            prompt = _SIZING_PROMPT_TEMPLATE.format(
                position_list=position_list_str,
            )

            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else ""

            parsed = _parse_sizing_response(text, {p["symbol"] for p in batch})
            result.update(parsed)

            logger.info(
                f"LLM sizing batch {i // batch_size + 1}: "
                f"enriched {len(parsed)}/{len(batch)} positions"
            )

    except ImportError:
        logger.warning("anthropic SDK not installed — skipping LLM dynamic sizing")
    except Exception as e:
        logger.warning(f"LLM dynamic sizing error: {e}")

    return result


def _parse_sizing_response(text: str, valid_symbols: set[str]) -> dict[str, dict]:
    """Parse and validate LLM JSON response into {symbol: sizing_info} dict."""
    result: dict[str, dict] = {}

    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.warning("LLM sizing response did not contain a JSON array")
        return result

    json_str = text[start:end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM sizing JSON: {e}")
        return result

    if not isinstance(data, list):
        logger.warning("LLM sizing JSON is not an array")
        return result

    for item in data:
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", "")).strip().upper()
        if not sym or sym not in valid_symbols:
            continue

        cleaned: dict = {"symbol": sym}

        # volatility_score
        vol_score = item.get("volatility_score")
        if vol_score is not None:
            try:
                vol_score = int(vol_score)
                vol_score = max(1, min(vol_score, 10))
                cleaned["volatility_score"] = vol_score
            except (ValueError, TypeError):
                cleaned["volatility_score"] = 5
        else:
            cleaned["volatility_score"] = 5

        # size_adjustment_pct
        adj = item.get("size_adjustment_pct")
        if adj is not None:
            try:
                adj = float(adj)
                adj = max(-2.0, min(adj, 2.0))
                cleaned["size_adjustment_pct"] = round(adj, 2)
            except (ValueError, TypeError):
                cleaned["size_adjustment_pct"] = 0.0
        else:
            cleaned["size_adjustment_pct"] = 0.0

        # reason
        reason = item.get("reason")
        cleaned["reason"] = str(reason) if reason and len(str(reason)) >= 10 else "LLM sizing adjustment based on volatility assessment"

        result[sym] = cleaned

    return result
