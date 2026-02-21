"""
Rotation Detector Tool: Rotation Narrative Intelligence
Generate historical context and implications for detected rotation patterns
using LLM knowledge.

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
# LLM Rotation Narrative
# ---------------------------------------------------------------------------

_ROTATION_NARRATIVE_PROMPT_TEMPLATE = """\
You are a market historian analyzing sector rotation patterns.

Current rotation analysis:
- Verdict: {verdict}
- Rotation Type: {rotation_type}
- Stage: {stage}
- Source Clusters (capital leaving): {source_clusters}
- Destination Clusters (capital flowing to): {dest_clusters}
- Market Regime: {regime}
- Signals Active: {signals_active}/5
- Velocity: {velocity}

Provide a 2-3 paragraph narrative (100-300 words) covering historical context, what typically happens next, and key risks. Focus on factual historical parallels.

Return ONLY the narrative text — no JSON, no markdown headers, no bullet points.
"""


def generate_rotation_narrative_llm(
    rotation_context: dict,
    model: str = "claude-haiku-4-5-20251001",
) -> Optional[str]:
    """
    Generate a rotation narrative from LLM knowledge.

    Args:
        rotation_context: Dict with keys: verdict, rotation_type, stage,
            source_clusters, dest_clusters, regime, signals_active, velocity.
        model: Anthropic model ID.

    Returns:
        Narrative string (100-500 words) or None on failure.
    """
    if not rotation_context:
        return None

    # Skip if no rotation detected
    if rotation_context.get("verdict") == "none":
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0)

        prompt = _ROTATION_NARRATIVE_PROMPT_TEMPLATE.format(
            verdict=rotation_context.get("verdict", "unknown"),
            rotation_type=rotation_context.get("rotation_type", "unknown"),
            stage=rotation_context.get("stage", "unknown"),
            source_clusters=", ".join(rotation_context.get("source_clusters", [])) or "none",
            dest_clusters=", ".join(rotation_context.get("dest_clusters", [])) or "none",
            regime=rotation_context.get("regime", "unknown"),
            signals_active=rotation_context.get("signals_active", 0),
            velocity=rotation_context.get("velocity", "unknown"),
        )

        response = client.messages.create(
            model=model,
            max_tokens=768,
            messages=[{"role": "user", "content": prompt}],
        )
        track_tokens("generate_rotation_narrative_llm", response)
        text = response.content[0].text.strip() if response.content else ""

        if len(text) < 100:
            logger.warning(f"LLM rotation narrative too short ({len(text)} chars)")
            return None

        # Truncate if too long
        if len(text) > 2000:
            text = text[:2000]

        logger.info(f"LLM rotation narrative generated: {len(text)} chars")
        return text

    except ImportError:
        logger.warning("anthropic SDK not installed — skipping LLM rotation narrative")
        return None
    except Exception as e:
        logger.warning(f"LLM rotation narrative error: {e}")
        return None
