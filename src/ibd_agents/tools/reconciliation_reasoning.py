"""
Portfolio Reconciler Tool: Smart Reconciliation Reasoning
Enrich position action rationales with LLM-generated context-aware
explanations incorporating rotation signals, risk warnings, and strategy.

Follows the same dual-path pattern as catalyst_enrichment.py:
deterministic fallback + optional LLM enrichment.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from ibd_agents.tools.token_tracker import track as track_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Reasoning Enrichment
# ---------------------------------------------------------------------------

_REASONING_PROMPT_TEMPLATE = """\
You are a portfolio transition advisor explaining action rationales.

Market context:
- Rotation Verdict: {rotation_verdict}
- Rotation Type: {rotation_type}
- Risk Status: {risk_status}
- Market Regime: {regime}

For each position action below, provide a 1-2 sentence rationale that explains WHY the action is recommended, incorporating the market context above. Be specific and mention rotation, risk, or regime factors where relevant.

Return a JSON array with one object per action. Use ONLY valid JSON — no markdown, no code fences, no extra text.

Fields per action:
- symbol: str (ticker, uppercase)
- rationale: str (1-2 sentence explanation, minimum 15 characters)

Actions:
{action_list}
"""


def enrich_rationale_llm(
    actions: list[dict],
    context: dict,
    model: str = "claude-haiku-4-5-20251001",
    batch_size: int = 30,
) -> dict[str, str]:
    """
    Get LLM-enhanced rationales for position actions.

    Args:
        actions: List of dicts with keys: symbol, action_type, current_pct,
            target_pct, priority.
        context: Dict with keys: rotation_verdict, rotation_type,
            risk_status, regime.
        model: Anthropic model ID.
        batch_size: Actions per LLM call.

    Returns:
        {symbol: enhanced_rationale}
        Empty dict on failure.
    """
    if not actions:
        return {}

    result: dict[str, str] = {}
    _t0 = time.monotonic()

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0)

        for i in range(0, len(actions), batch_size):
            if time.monotonic() - _t0 > 60:
                logger.warning(f"LLM rationale enrichment timed out after 60s, returning {len(result)} partial results")
                break
            batch = actions[i:i + batch_size]

            action_lines = []
            for a in batch:
                action_lines.append(
                    f"  {a['symbol']} | Action: {a.get('action_type', '?')} | "
                    f"Current: {a.get('current_pct', 0):.1f}% → Target: {a.get('target_pct', 0):.1f}% | "
                    f"Priority: {a.get('priority', '?')}"
                )
            action_list_str = "\n".join(action_lines)
            prompt = _REASONING_PROMPT_TEMPLATE.format(
                rotation_verdict=context.get("rotation_verdict", "unknown"),
                rotation_type=context.get("rotation_type", "unknown"),
                risk_status=context.get("risk_status", "unknown"),
                regime=context.get("regime", "unknown"),
                action_list=action_list_str,
            )

            response = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            track_tokens("enrich_rationale_llm", response)
            text = response.content[0].text if response.content else ""

            parsed = _parse_reasoning_response(text, {a["symbol"] for a in batch})
            result.update(parsed)

            logger.info(
                f"LLM reasoning batch {i // batch_size + 1}: "
                f"enriched {len(parsed)}/{len(batch)} actions"
            )

    except ImportError:
        logger.warning("anthropic SDK not installed — skipping LLM reasoning enrichment")
    except Exception as e:
        logger.warning(f"LLM reasoning enrichment error: {e}")

    return result


def _parse_reasoning_response(text: str, valid_symbols: set[str]) -> dict[str, str]:
    """Parse and validate LLM JSON response into {symbol: rationale} dict."""
    result: dict[str, str] = {}

    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.warning("LLM reasoning response did not contain a JSON array")
        return result

    json_str = text[start:end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM reasoning JSON: {e}")
        return result

    if not isinstance(data, list):
        logger.warning("LLM reasoning JSON is not an array")
        return result

    for item in data:
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", "")).strip().upper()
        if not sym or sym not in valid_symbols:
            continue

        rationale = item.get("rationale")
        if rationale and len(str(rationale)) >= 10:
            result[sym] = str(rationale)

    return result
