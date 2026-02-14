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
from typing import Optional

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

Provide a 2-4 paragraph narrative (100-500 words) that:
1. Places this rotation pattern in historical context (cite similar past rotations, e.g. 2020 growth->value, 2022 growth->commodity)
2. Explains what typically happens NEXT in this type of rotation
3. Identifies key risks and potential sector opportunities
4. Notes any unusual aspects of the current rotation vs historical precedent

Return ONLY the narrative text — no JSON, no markdown headers, no bullet points.
Focus on factual historical parallels and pattern recognition.
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
        client = anthropic.Anthropic(timeout=60.0)

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
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
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
