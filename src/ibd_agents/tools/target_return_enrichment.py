"""
Agent 14 Tool: Target Return LLM Enrichment
Enrich selection rationale, construction narrative, and alternative
reasoning using LLM knowledge of market dynamics and sector positioning.

Follows the same dual-path pattern as stop_loss_tuning.py and
dynamic_sizing.py: deterministic fallback + optional LLM enrichment.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enrichment 1: Selection Rationale (batched, 30 per call)
# ---------------------------------------------------------------------------

_SELECTION_RATIONALE_PROMPT = """\
For each position below, write a 1-2 sentence rationale explaining WHY this \
specific stock was selected for a target return portfolio. Focus on:
- What makes it stand out (RS breakout, EPS momentum, sector leadership)
- Sector momentum context (leading/improving/lagging/declining)
- Conviction factors (multi-source validation, tier positioning)

Return a JSON array with one object per position. Use ONLY valid JSON — \
no markdown, no code fences, no extra text.

Fields per position:
- symbol: str (ticker, uppercase)
- rationale: str (1-2 sentences, minimum 20 characters)

Positions:
{position_list}
"""


def enrich_selection_rationale_llm(
    positions: list[dict],
    model: str = "claude-haiku-4-5-20251001",
    batch_size: int = 30,
) -> dict[str, str]:
    """
    Get LLM-enriched selection rationale for target return positions.

    Args:
        positions: List of dicts with keys: ticker, company_name, tier,
            sector, composite_score, rs_rating, eps_rating, conviction_level,
            allocation_pct, sector_momentum, multi_source_count.
        model: Anthropic model ID.
        batch_size: Positions per LLM call.

    Returns:
        {ticker: rationale_string}. Empty dict on failure.
    """
    if not positions:
        return {}

    result: dict[str, str] = {}
    _t0 = time.monotonic()

    try:
        import anthropic
        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0,
        )

        for i in range(0, len(positions), batch_size):
            if time.monotonic() - _t0 > 60:
                logger.warning(
                    f"LLM selection rationale timed out after 60s, "
                    f"returning {len(result)} partial results"
                )
                break
            batch = positions[i:i + batch_size]

            pos_lines = []
            for p in batch:
                pos_lines.append(
                    f"  {p['ticker']} ({p.get('company_name', '')}) | "
                    f"T{p.get('tier', '?')} | Sector: {p.get('sector', 'UNKNOWN')} "
                    f"({p.get('sector_momentum', 'neutral')}) | "
                    f"Comp: {p.get('composite_score', '?')} | "
                    f"RS: {p.get('rs_rating', '?')} | EPS: {p.get('eps_rating', '?')} | "
                    f"Conv: {p.get('conviction_level', '?')} | "
                    f"Alloc: {p.get('allocation_pct', '?')}% | "
                    f"Multi-src: {p.get('multi_source_count', 0)}"
                )
            position_list_str = "\n".join(pos_lines)
            prompt = _SELECTION_RATIONALE_PROMPT.format(
                position_list=position_list_str,
            )

            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else ""

            parsed = _parse_selection_rationale_response(
                text, {p["ticker"] for p in batch},
            )
            result.update(parsed)

            logger.info(
                f"LLM selection rationale batch {i // batch_size + 1}: "
                f"enriched {len(parsed)}/{len(batch)} positions"
            )

    except ImportError:
        logger.warning(
            "anthropic SDK not installed — skipping LLM selection rationale"
        )
    except Exception as e:
        logger.warning(f"LLM selection rationale error: {e}")

    return result


def _parse_selection_rationale_response(
    text: str, valid_symbols: set[str],
) -> dict[str, str]:
    """Parse LLM JSON response into {symbol: rationale} dict."""
    result: dict[str, str] = {}

    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.warning("LLM selection rationale response has no JSON array")
        return result

    json_str = text[start:end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM selection rationale JSON: {e}")
        return result

    if not isinstance(data, list):
        logger.warning("LLM selection rationale JSON is not an array")
        return result

    for item in data:
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", "")).strip().upper()
        if not sym or sym not in valid_symbols:
            continue

        rationale = item.get("rationale")
        if not rationale or not isinstance(rationale, str):
            continue
        rationale = rationale.strip()
        if len(rationale) < 20:
            continue

        result[sym] = rationale

    return result


# ---------------------------------------------------------------------------
# Enrichment 2: Construction Narrative (single call)
# ---------------------------------------------------------------------------

_CONSTRUCTION_NARRATIVE_PROMPT = """\
You are a portfolio strategist explaining a target return portfolio construction.

Portfolio parameters:
- Target return: {target_return_pct}% annualized
- Market regime: {regime}
- Tier mix: T1={t1_pct}% / T2={t2_pct}% / T3={t3_pct}%
- Positions: {n_positions}
- Probability of achieving target: {prob_achieve}
- Achievability rating: {achievability}
- Sector momentum: {sector_momentum}
- Top sectors: {top_sectors}
- Top holdings: {top_positions}

Write a 3-5 sentence construction rationale (minimum 100 characters) explaining:
1. WHY this tier mix was chosen for this regime and target
2. What sector conditions support this construction
3. What gives the portfolio its edge (concentration, momentum, etc.)
4. What could go wrong

Return ONLY the narrative text — no JSON, no markdown headers, no bullet points.
"""


def enrich_construction_narrative_llm(
    context: dict,
    model: str = "claude-haiku-4-5-20251001",
) -> Optional[str]:
    """
    Get LLM-enriched construction narrative for the target return portfolio.

    Args:
        context: Dict with keys: target_return_pct, regime, t1_pct, t2_pct,
            t3_pct, n_positions, prob_achieve, achievability, sector_momentum,
            top_sectors, top_positions.
        model: Anthropic model ID.

    Returns:
        Narrative string (min 100 chars) or None on failure.
    """
    if not context:
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0,
        )

        prompt = _CONSTRUCTION_NARRATIVE_PROMPT.format(
            target_return_pct=context.get("target_return_pct", 30.0),
            regime=context.get("regime", "neutral"),
            t1_pct=context.get("t1_pct", 40),
            t2_pct=context.get("t2_pct", 35),
            t3_pct=context.get("t3_pct", 25),
            n_positions=context.get("n_positions", 10),
            prob_achieve=context.get("prob_achieve", "N/A"),
            achievability=context.get("achievability", "STRETCH"),
            sector_momentum=context.get("sector_momentum", "neutral"),
            top_sectors=", ".join(context.get("top_sectors", [])),
            top_positions=", ".join(context.get("top_positions", [])),
        )

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else ""
        text = text.strip()

        # Validate minimum length
        if len(text) < 100:
            logger.warning(
                f"LLM construction narrative too short ({len(text)} chars)"
            )
            return None

        # Truncate at 2000 chars
        if len(text) > 2000:
            text = text[:2000]

        logger.info(f"LLM construction narrative: {len(text)} chars")
        return text

    except ImportError:
        logger.warning(
            "anthropic SDK not installed — skipping LLM construction narrative"
        )
    except Exception as e:
        logger.warning(f"LLM construction narrative error: {e}")

    return None


# ---------------------------------------------------------------------------
# Enrichment 3: Alternative Reasoning (single call)
# ---------------------------------------------------------------------------

_ALTERNATIVE_REASONING_PROMPT = """\
You are a portfolio advisor comparing alternative portfolio constructions \
against a primary target return portfolio.

Primary portfolio: {primary_target}% target, {primary_prob} probability \
of success in {regime} market regime.

For each alternative below, explain:
1. key_difference: What makes this alternative fundamentally different \
(1-2 sentences, min 10 characters)
2. tradeoff: What you gain and give up vs the primary portfolio \
(1-2 sentences, min 10 characters)

Be specific about risk/return tradeoffs and regime implications.

Return a JSON array with one object per alternative. Use ONLY valid JSON — \
no markdown, no code fences, no extra text.

Fields per alternative:
- name: str (alternative portfolio name, exactly as given)
- key_difference: str
- tradeoff: str

Alternatives:
{alternative_list}
"""


def enrich_alternative_reasoning_llm(
    alternatives: list[dict],
    context: dict,
    model: str = "claude-haiku-4-5-20251001",
) -> dict[str, dict]:
    """
    Get LLM-enriched reasoning for alternative portfolios.

    Args:
        alternatives: List of dicts with keys: name, target_return_pct,
            prob_achieve_target, t1_pct, t2_pct, t3_pct, max_drawdown_pct.
        context: Dict with keys: primary_target, primary_prob, regime.
        model: Anthropic model ID.

    Returns:
        {name: {key_difference, tradeoff}}. Empty dict on failure.
    """
    if not alternatives:
        return {}

    result: dict[str, dict] = {}

    try:
        import anthropic
        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0,
        )

        alt_lines = []
        for a in alternatives:
            alt_lines.append(
                f"  {a['name']} | Target: {a.get('target_return_pct', '?')}% | "
                f"Prob: {a.get('prob_achieve_target', '?')} | "
                f"T1={a.get('t1_pct', '?')}%/T2={a.get('t2_pct', '?')}%/"
                f"T3={a.get('t3_pct', '?')}% | "
                f"Max drawdown: {a.get('max_drawdown_pct', '?')}%"
            )
        alternative_list_str = "\n".join(alt_lines)

        prompt = _ALTERNATIVE_REASONING_PROMPT.format(
            primary_target=context.get("primary_target", 30.0),
            primary_prob=context.get("primary_prob", "N/A"),
            regime=context.get("regime", "neutral"),
            alternative_list=alternative_list_str,
        )

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else ""

        valid_names = {a["name"] for a in alternatives}
        parsed = _parse_alternative_reasoning_response(text, valid_names)
        result.update(parsed)

        logger.info(
            f"LLM alternative reasoning: "
            f"enriched {len(parsed)}/{len(alternatives)} alternatives"
        )

    except ImportError:
        logger.warning(
            "anthropic SDK not installed — skipping LLM alternative reasoning"
        )
    except Exception as e:
        logger.warning(f"LLM alternative reasoning error: {e}")

    return result


def _parse_alternative_reasoning_response(
    text: str, valid_names: set[str],
) -> dict[str, dict]:
    """Parse LLM JSON response into {name: {key_difference, tradeoff}} dict."""
    result: dict[str, dict] = {}

    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.warning("LLM alternative reasoning response has no JSON array")
        return result

    json_str = text[start:end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM alternative reasoning JSON: {e}")
        return result

    if not isinstance(data, list):
        logger.warning("LLM alternative reasoning JSON is not an array")
        return result

    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name not in valid_names:
            continue

        key_diff = item.get("key_difference")
        if not key_diff or not isinstance(key_diff, str) or len(key_diff.strip()) < 10:
            continue

        tradeoff = item.get("tradeoff")
        if not tradeoff or not isinstance(tradeoff, str) or len(tradeoff.strip()) < 10:
            continue

        result[name] = {
            "key_difference": key_diff.strip(),
            "tradeoff": tradeoff.strip(),
        }

    return result
