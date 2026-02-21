"""
Risk Officer Tool: Smart Stop-Loss Tuning
Recommend per-position trailing stop adjustments using LLM knowledge
of sector volatility, catalyst timing, and beta characteristics.

Follows the same dual-path pattern as catalyst_enrichment.py:
deterministic fallback + optional LLM enrichment.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

from ibd_agents.tools.token_tracker import track as track_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_VOLATILITY_FLAGS = ("high", "normal", "low")

# Allowed stop range per tier (never go below or above these)
STOP_RANGE: dict[int, tuple[float, float]] = {
    1: (10.0, 25.0),
    2: (8.0, 20.0),
    3: (5.0, 15.0),
}


# ---------------------------------------------------------------------------
# Deterministic Helpers
# ---------------------------------------------------------------------------

def compute_stop_recommendation(
    current_stop: float,
    llm_stop: float,
    tier: int,
) -> float:
    """
    Clamp LLM recommendation to tier-valid range.
    Returns the clamped recommended stop percentage.
    """
    low, high = STOP_RANGE.get(tier, (5.0, 25.0))
    return round(max(low, min(llm_stop, high)), 1)


# ---------------------------------------------------------------------------
# LLM Stop-Loss Enrichment
# ---------------------------------------------------------------------------

_STOP_LOSS_PROMPT_TEMPLATE = """\
For each position below, recommend an optimal trailing stop percentage.
Consider the sector's typical volatility, the stock's tier (1=momentum/high-growth, 2=quality, 3=defensive), and any known catalyst timing.

Return a JSON array with one object per position. Use ONLY valid JSON — no markdown, no code fences, no extra text.

Fields per position:
- symbol: str (ticker, uppercase)
- recommended_stop_pct: float (trailing stop %, between 5 and 35)
- reason: str (1-2 sentence explanation)
- volatility_flag: str ("high", "normal", or "low")

Guidelines: High-beta sectors → tighter stops; Defensive sectors → wider stops. T1 needs wider stops (15-22%) to avoid whipsaw, T3 can be tighter (8-12%).

Positions:
{position_list}
"""


def enrich_stop_loss_llm(
    positions: list[dict],
    model: str = "claude-haiku-4-5-20251001",
    batch_size: int = 30,
) -> dict[str, dict]:
    """
    Get LLM stop-loss recommendations for positions.

    Args:
        positions: List of dicts with keys: symbol, sector, tier,
            current_stop_pct, conviction, asset_type.
        model: Anthropic model ID.
        batch_size: Positions per LLM call.

    Returns:
        {symbol: {recommended_stop_pct, reason, volatility_flag}}
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
                logger.warning(f"LLM stop-loss tuning timed out after 60s, returning {len(result)} partial results")
                break
            batch = positions[i:i + batch_size]

            pos_lines = []
            for p in batch:
                pos_lines.append(
                    f"  {p['symbol']} | Sector: {p.get('sector', 'UNKNOWN')} | "
                    f"T{p.get('tier', '?')} | Current stop: {p.get('current_stop_pct', '?')}% | "
                    f"Type: {p.get('asset_type', 'stock')} | "
                    f"Conviction: {p.get('conviction', '?')}/10"
                )
            position_list_str = "\n".join(pos_lines)
            prompt = _STOP_LOSS_PROMPT_TEMPLATE.format(
                position_list=position_list_str,
            )

            response = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            track_tokens("enrich_stop_loss_llm", response)
            text = response.content[0].text if response.content else ""

            parsed = _parse_stop_loss_response(text, {p["symbol"] for p in batch})
            result.update(parsed)

            logger.info(
                f"LLM stop-loss batch {i // batch_size + 1}: "
                f"enriched {len(parsed)}/{len(batch)} positions"
            )

    except ImportError:
        logger.warning("anthropic SDK not installed — skipping LLM stop-loss tuning")
    except Exception as e:
        logger.warning(f"LLM stop-loss tuning error: {e}")

    return result


def _parse_stop_loss_response(text: str, valid_symbols: set[str]) -> dict[str, dict]:
    """Parse and validate LLM JSON response into {symbol: stop_info} dict."""
    result: dict[str, dict] = {}

    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.warning("LLM stop-loss response did not contain a JSON array")
        return result

    json_str = text[start:end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM stop-loss JSON: {e}")
        return result

    if not isinstance(data, list):
        logger.warning("LLM stop-loss JSON is not an array")
        return result

    for item in data:
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", "")).strip().upper()
        if not sym or sym not in valid_symbols:
            continue

        cleaned: dict = {"symbol": sym}

        # recommended_stop_pct
        stop_pct = item.get("recommended_stop_pct")
        if stop_pct is not None:
            try:
                stop_pct = float(stop_pct)
                stop_pct = max(5.0, min(stop_pct, 35.0))
                cleaned["recommended_stop_pct"] = round(stop_pct, 1)
            except (ValueError, TypeError):
                continue  # Skip invalid entries
        else:
            continue  # required field

        # reason
        reason = item.get("reason")
        cleaned["reason"] = str(reason) if reason and len(str(reason)) >= 10 else "LLM stop-loss recommendation based on sector and tier analysis"

        # volatility_flag
        vol_flag = item.get("volatility_flag")
        if vol_flag is not None and str(vol_flag) in VALID_VOLATILITY_FLAGS:
            cleaned["volatility_flag"] = str(vol_flag)
        else:
            cleaned["volatility_flag"] = "normal"

        result[sym] = cleaned

    return result
