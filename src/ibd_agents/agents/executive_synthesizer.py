"""
Agent 10: Executive Summary Synthesizer
IBD Momentum Investment Framework v4.0

Reads all 9 agent outputs and produces a unified investment thesis:
- "Here's why your portfolio looks the way it does"
- Connects rotation signals to sector allocation to risk findings
- Highlights contradictions between agents
- Risk: None — read-only synthesis

Supports LLM narrative generation (anthropic SDK) with deterministic
template fallback when the SDK is unavailable or API errors occur.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date
from typing import Optional

try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    Agent = None  # type: ignore
    Task = None  # type: ignore

from ibd_agents.schemas.analyst_output import AnalystOutput
from ibd_agents.schemas.educator_output import EducatorOutput
from ibd_agents.schemas.portfolio_output import PortfolioOutput
from ibd_agents.schemas.reconciliation_output import ReconciliationOutput
from ibd_agents.schemas.research_output import ResearchOutput
from ibd_agents.schemas.returns_projection_output import ReturnsProjectionOutput
from ibd_agents.schemas.risk_output import RiskAssessment
from ibd_agents.schemas.rotation_output import RotationDetectionOutput
from ibd_agents.schemas.strategy_output import SectorStrategyOutput
from ibd_agents.schemas.synthesis_output import (
    Contradiction,
    CrossAgentConnection,
    KeyNumbersDashboard,
    SynthesisOutput,
)
from ibd_agents.tools.token_tracker import track as track_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI Agent Builder
# ---------------------------------------------------------------------------

def build_synthesizer_agent() -> "Agent":
    """Create the Executive Summary Synthesizer Agent. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Agent(
        role="Executive Summary Synthesizer",
        goal=(
            "Read all 9 agent outputs and synthesize a unified 1-page "
            "investment thesis that explains why the portfolio looks the "
            "way it does. Connect rotation signals to sector allocations "
            "to risk findings. Highlight contradictions between agents."
        ),
        backstory=(
            "You are a chief investment strategist who distills complex "
            "multi-agent analysis into a clear, actionable executive summary. "
            "You never make investment decisions — you only synthesize and "
            "explain the collective findings."
        ),
        tools=[],
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        temperature=0.5,
    )


def build_synthesizer_task(
    agent: "Agent",
    all_outputs_json: str = "",
) -> "Task":
    """Create the Synthesis task. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Task(
        description=f"""Synthesize all 9 agent outputs into a unified executive summary.

STEPS:
1. Extract key numbers from each agent output
2. Identify cross-agent connections (rotation → strategy → portfolio)
3. Detect contradictions between agents
4. Generate investment thesis narrative
5. Build action items from reconciler/risk/returns

Agent outputs:
{all_outputs_json}
""",
        expected_output=(
            "JSON with investment_thesis, portfolio_narrative, "
            "risk_reward_assessment, market_context, key_numbers, "
            "cross_agent_connections, contradictions, action_items."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Deterministic Extraction Functions
# ---------------------------------------------------------------------------

def _extract_key_numbers(
    research: ResearchOutput,
    analyst: AnalystOutput,
    rotation: RotationDetectionOutput,
    strategy: SectorStrategyOutput,
    portfolio: PortfolioOutput,
    risk: RiskAssessment,
    returns_output: ReturnsProjectionOutput,
    reconciler: ReconciliationOutput,
) -> KeyNumbersDashboard:
    """Extract key numbers from all agent outputs."""
    # Get scenario returns
    base_return = 0.0
    bull_return = 0.0
    bear_return = 0.0
    for scenario in returns_output.scenarios:
        if scenario.scenario == "base":
            base_return = scenario.portfolio_return_12m
        elif scenario.scenario == "bull":
            bull_return = scenario.portfolio_return_12m
        elif scenario.scenario == "bear":
            bear_return = scenario.portfolio_return_12m

    # Count cap sizes across all portfolio positions
    all_pos = (
        portfolio.tier_1.positions
        + portfolio.tier_2.positions
        + portfolio.tier_3.positions
    )
    large_cap = sum(1 for p in all_pos if p.cap_size == "large")
    mid_cap = sum(1 for p in all_pos if p.cap_size == "mid")
    small_cap = sum(1 for p in all_pos if p.cap_size == "small")

    return KeyNumbersDashboard(
        # Research
        total_stocks_scanned=research.total_securities_scanned,
        stocks_in_universe=len(research.stocks),
        etfs_in_universe=len(research.etfs),
        # Analyst
        stocks_rated=len(analyst.rated_stocks),
        etfs_rated=len(analyst.rated_etfs),
        tier_1_count=analyst.tier_distribution.tier_1_count,
        tier_2_count=analyst.tier_distribution.tier_2_count,
        tier_3_count=analyst.tier_distribution.tier_3_count,
        # Rotation
        rotation_verdict=rotation.verdict.value,
        rotation_confidence=rotation.confidence,
        market_regime=rotation.market_regime.regime,
        # Portfolio
        portfolio_positions=portfolio.total_positions,
        stock_count=portfolio.stock_count,
        etf_count=portfolio.etf_count,
        large_cap_count=large_cap,
        mid_cap_count=mid_cap,
        small_cap_count=small_cap,
        cash_pct=portfolio.cash_pct,
        # Risk
        risk_status=risk.overall_status,
        sleep_well_score=risk.sleep_well_scores.overall_score,
        risk_warnings_count=len(risk.warnings),
        # Returns
        base_case_return_12m=base_return,
        bull_case_return_12m=bull_return,
        bear_case_return_12m=bear_return,
        # Reconciler
        turnover_pct=reconciler.transformation_metrics.turnover_pct,
        actions_count=len(reconciler.actions),
    )


def _detect_contradictions(
    strategy: SectorStrategyOutput,
    risk: RiskAssessment,
    rotation: RotationDetectionOutput,
    portfolio: PortfolioOutput,
    returns_output: ReturnsProjectionOutput,
    reconciler: ReconciliationOutput,
) -> list[Contradiction]:
    """Detect contradictions or tensions between agent findings."""
    contradictions: list[Contradiction] = []

    # Check 1: Strategist recommends sector that Risk Officer flagged
    risk_warned_sectors: set[str] = set()
    for warning in risk.warnings:
        desc_upper = warning.description.upper()
        for sector in portfolio.sector_exposure:
            if sector.upper() in desc_upper:
                risk_warned_sectors.add(sector)

    for sector, pct in strategy.sector_allocations.overall_allocation.items():
        if pct >= 15.0 and sector in risk_warned_sectors:
            contradictions.append(Contradiction(
                agent_a="Sector Strategist (Agent 04)",
                agent_b="Risk Officer (Agent 06)",
                finding_a=f"Strategist allocates {pct:.0f}% to {sector}",
                finding_b=f"Risk Officer flagged {sector} concentration warning",
                resolution=(
                    f"The {sector} allocation reflects high momentum conviction "
                    f"while the risk warning flags concentration risk. Consider "
                    f"monitoring position sizes within {sector} closely."
                ),
            ))

    # Check 2: Bear regime but offensive tier weights
    if rotation.market_regime.regime == "bear":
        t1_pct = portfolio.tier_1.actual_pct
        if t1_pct > 40.0:
            contradictions.append(Contradiction(
                agent_a="Rotation Detector (Agent 03)",
                agent_b="Portfolio Manager (Agent 05)",
                finding_a=f"Market regime is BEAR",
                finding_b=f"Tier 1 (Momentum) allocation is {t1_pct:.0f}%, above 40% threshold",
                resolution=(
                    "The portfolio maintains offensive positioning despite bear signals. "
                    "This may reflect high-conviction momentum names or keep commitments. "
                    "Consider tightening trailing stops on T1 positions."
                ),
            ))

    # Check 3: Risk not APPROVED but returns projection is optimistic
    base_return = 0.0
    for scenario in returns_output.scenarios:
        if scenario.scenario == "base":
            base_return = scenario.portfolio_return_12m
    if risk.overall_status != "APPROVED" and base_return > 15.0:
        contradictions.append(Contradiction(
            agent_a="Risk Officer (Agent 06)",
            agent_b="Returns Projector (Agent 07)",
            finding_a=f"Risk status is {risk.overall_status} (not APPROVED)",
            finding_b=f"Base case projects {base_return:.1f}% 12-month return",
            resolution=(
                "Higher risk accompanies higher expected returns. "
                "The risk warnings should be addressed to improve the "
                "probability of achieving projected returns."
            ),
        ))

    # Check 4: High rotation velocity but low reconciler turnover
    if rotation.velocity == "Fast" and reconciler.transformation_metrics.turnover_pct < 30.0:
        contradictions.append(Contradiction(
            agent_a="Rotation Detector (Agent 03)",
            agent_b="Portfolio Reconciler (Agent 08)",
            finding_a="Rotation velocity is FAST, suggesting rapid sector shifts",
            finding_b=f"Portfolio turnover is only {reconciler.transformation_metrics.turnover_pct:.0f}%",
            resolution=(
                "Low turnover despite fast rotation may indicate the portfolio "
                "is already positioned for the rotation, or that keep commitments "
                "limit the ability to fully rotate."
            ),
        ))

    return contradictions


def _build_cross_agent_connections(
    rotation: RotationDetectionOutput,
    strategy: SectorStrategyOutput,
    portfolio: PortfolioOutput,
    risk: RiskAssessment,
    returns_output: ReturnsProjectionOutput,
) -> list[CrossAgentConnection]:
    """Build connections between agent findings."""
    connections: list[CrossAgentConnection] = []

    # Connection 1: Rotation verdict → Strategy regime adjustment
    connections.append(CrossAgentConnection(
        from_agent="Rotation Detector (Agent 03)",
        to_agent="Sector Strategist (Agent 04)",
        connection=(
            f"Rotation verdict '{rotation.verdict.value}' with "
            f"{rotation.confidence}% confidence in {rotation.market_regime.regime} "
            f"regime drove the strategy response: {strategy.rotation_response[:80]}"
        ),
        implication=(
            f"Regime adjustment: {strategy.regime_adjustment[:80]}. "
            f"This shaped tier targets and sector allocations."
        ),
    ))

    # Connection 2: Strategy tier targets → Portfolio actual allocations
    tier_targets = strategy.sector_allocations.tier_targets
    connections.append(CrossAgentConnection(
        from_agent="Sector Strategist (Agent 04)",
        to_agent="Portfolio Manager (Agent 05)",
        connection=(
            f"Strategy set tier targets T1={tier_targets.get('T1', 0):.0f}%/"
            f"T2={tier_targets.get('T2', 0):.0f}%/"
            f"T3={tier_targets.get('T3', 0):.0f}%/"
            f"Cash={tier_targets.get('Cash', 0):.0f}%. "
            f"Portfolio built {portfolio.total_positions} positions "
            f"({portfolio.stock_count} stocks, {portfolio.etf_count} ETFs)."
        ),
        implication=(
            f"Actual allocation: T1={portfolio.tier_1.actual_pct:.0f}%/"
            f"T2={portfolio.tier_2.actual_pct:.0f}%/"
            f"T3={portfolio.tier_3.actual_pct:.0f}%/"
            f"Cash={portfolio.cash_pct:.0f}%."
        ),
    ))

    # Connection 3: Portfolio sector exposure → Risk concentration findings
    top_sector = max(portfolio.sector_exposure.items(), key=lambda x: x[1])
    concentration_checks = [
        c for c in risk.check_results if c.check_name == "sector_concentration"
    ]
    conc_status = concentration_checks[0].status if concentration_checks else "N/A"
    connections.append(CrossAgentConnection(
        from_agent="Portfolio Manager (Agent 05)",
        to_agent="Risk Officer (Agent 06)",
        connection=(
            f"Portfolio's top sector exposure: {top_sector[0]} at {top_sector[1]:.1f}%. "
            f"Sector concentration check: {conc_status}."
        ),
        implication=(
            f"Risk assessment overall status: {risk.overall_status}. "
            f"Sleep Well score: {risk.sleep_well_scores.overall_score}/10. "
            f"{len(risk.warnings)} warning(s) issued."
        ),
    ))

    # Connection 4: Risk assessment → Returns confidence
    connections.append(CrossAgentConnection(
        from_agent="Risk Officer (Agent 06)",
        to_agent="Returns Projector (Agent 07)",
        connection=(
            f"Risk status '{risk.overall_status}' with Sleep Well score "
            f"{risk.sleep_well_scores.overall_score}/10 informed return projections. "
            f"Max drawdown with stops: "
            f"{returns_output.risk_metrics.max_drawdown_with_stops:.1f}%."
        ),
        implication=(
            f"Expected 12-month return: {returns_output.expected_return.expected_12m:.1f}%. "
            f"Portfolio Sharpe (base): {returns_output.risk_metrics.portfolio_sharpe_base:.2f}."
        ),
    ))

    # Connection 5: Rotation source/dest sectors → Portfolio positioning
    source_sectors = [sf.sector for sf in rotation.source_sectors[:3]]
    dest_sectors = [sf.sector for sf in rotation.destination_sectors[:3]]
    connections.append(CrossAgentConnection(
        from_agent="Rotation Detector (Agent 03)",
        to_agent="Portfolio Manager (Agent 05)",
        connection=(
            f"Rotation shows capital flowing from "
            f"{', '.join(source_sectors) if source_sectors else 'N/A'} "
            f"toward {', '.join(dest_sectors) if dest_sectors else 'N/A'}."
        ),
        implication=(
            "Portfolio positioning should favor destination sectors and "
            "reduce exposure to source sectors where not committed via keeps."
        ),
    ))

    return connections


def _build_action_items(
    reconciler: ReconciliationOutput,
    risk: RiskAssessment,
    returns_output: ReturnsProjectionOutput,
) -> list[str]:
    """Build 3-5 actionable next steps from reconciler, risk, and returns."""
    items: list[str] = []

    # From reconciler: implementation summary
    sells = sum(1 for a in reconciler.actions if a.action_type == "SELL")
    buys = sum(1 for a in reconciler.actions if a.action_type == "BUY")
    if sells > 0 or buys > 0:
        items.append(
            f"Execute portfolio transition: {sells} sell(s) in Week 1, "
            f"{buys} new position(s) across Weeks 2-4 per the 4-week plan."
        )

    # From risk: address warnings
    if risk.warnings:
        items.append(
            f"Address {len(risk.warnings)} risk warning(s) before full deployment: "
            f"{risk.warnings[0].description[:60]}."
        )

    # From returns: expected outcome
    base_return = returns_output.expected_return.expected_12m
    max_dd = returns_output.risk_metrics.max_drawdown_with_stops
    items.append(
        f"Monitor for base-case +{base_return:.1f}% 12-month return "
        f"with max drawdown of {max_dd:.1f}% (trailing stops active)."
    )

    # Keep monitoring
    keeps_found = len(reconciler.keep_verification.keeps_in_current)
    keeps_missing = len(reconciler.keep_verification.keeps_missing)
    if keeps_missing > 0:
        items.append(
            f"Acquire {keeps_missing} missing keep position(s) — "
            f"{keeps_found}/21 currently held."
        )

    # Sleep well check
    if risk.sleep_well_scores.overall_score <= 5:
        items.append(
            f"Review Sleep Well score ({risk.sleep_well_scores.overall_score}/10) — "
            f"consider reducing T1 exposure or tightening stops for comfort."
        )

    # Ensure 3-5 items
    if len(items) < 3:
        items.append(
            "Review full agent reports for detailed analysis before executing trades."
        )
    if len(items) < 3:
        items.append(
            "Set calendar reminders for weekly portfolio review during 4-week transition."
        )

    return items[:5]


# ---------------------------------------------------------------------------
# Template Fallback (No LLM)
# ---------------------------------------------------------------------------

def _build_template_thesis(
    key_numbers: KeyNumbersDashboard,
    connections: list[CrossAgentConnection],
    contradictions: list[Contradiction],
) -> str:
    """Build investment thesis from template using extracted data."""
    parts = [
        f"This portfolio is constructed from a universe of "
        f"{key_numbers.total_stocks_scanned} securities scanned, "
        f"with {key_numbers.stocks_rated} stocks and {key_numbers.etfs_rated} ETFs "
        f"rated through the IBD Momentum Framework's elite screening process. ",

        f"The Rotation Detector identifies a '{key_numbers.rotation_verdict}' "
        f"rotation signal at {key_numbers.rotation_confidence}% confidence "
        f"within a {key_numbers.market_regime} market regime. ",

        f"This drove the Sector Strategist to set tier allocations resulting "
        f"in a {key_numbers.portfolio_positions}-position portfolio "
        f"({key_numbers.stock_count} stocks, {key_numbers.etf_count} ETFs, "
        f"{key_numbers.cash_pct:.0f}% cash). ",

        f"The Risk Officer assessed the portfolio as '{key_numbers.risk_status}' "
        f"with a Sleep Well score of {key_numbers.sleep_well_score}/10 "
        f"and {key_numbers.risk_warnings_count} warning(s). ",

        f"Return projections range from {key_numbers.bear_case_return_12m:.1f}% (bear) "
        f"to {key_numbers.bull_case_return_12m:.1f}% (bull) over 12 months, "
        f"with a base case of {key_numbers.base_case_return_12m:.1f}%. ",
    ]

    if contradictions:
        parts.append(
            f"Notable: {len(contradictions)} tension(s) detected between agents — "
            f"review contradictions for nuanced positioning decisions."
        )

    return "".join(parts)


def _build_template_narrative(
    key_numbers: KeyNumbersDashboard,
    rotation: RotationDetectionOutput,
    strategy: SectorStrategyOutput,
) -> str:
    """Build portfolio narrative from template."""
    source_sectors = [sf.sector for sf in rotation.source_sectors[:3]]
    dest_sectors = [sf.sector for sf in rotation.destination_sectors[:3]]

    return (
        f"The portfolio reflects a {key_numbers.market_regime} regime "
        f"with {key_numbers.rotation_verdict} rotation. "
        f"Capital is flowing from {', '.join(source_sectors) if source_sectors else 'stable sectors'} "
        f"toward {', '.join(dest_sectors) if dest_sectors else 'broad diversification'}. "
        f"The Strategist responded with: {strategy.rotation_response}. "
        f"Regime adjustment: {strategy.regime_adjustment}. "
        f"This resulted in {key_numbers.tier_1_count} T1, "
        f"{key_numbers.tier_2_count} T2, and {key_numbers.tier_3_count} T3 "
        f"rated stocks forming the investment universe."
    )


def _build_template_risk_reward(
    key_numbers: KeyNumbersDashboard,
    risk: RiskAssessment,
    returns_output: ReturnsProjectionOutput,
) -> str:
    """Build risk/reward assessment from template."""
    return (
        f"Risk status: {key_numbers.risk_status}. "
        f"Sleep Well score: {key_numbers.sleep_well_score}/10. "
        f"{key_numbers.risk_warnings_count} warning(s) flagged. "
        f"Stress test resilience: {risk.stress_test_results.overall_resilience}. "
        f"Max drawdown with trailing stops: "
        f"{returns_output.risk_metrics.max_drawdown_with_stops:.1f}%, "
        f"without stops: {returns_output.risk_metrics.max_drawdown_without_stops:.1f}%. "
        f"Base-case Sharpe ratio: {returns_output.risk_metrics.portfolio_sharpe_base:.2f} "
        f"vs SPY Sharpe: {returns_output.risk_metrics.spy_sharpe_base:.2f}. "
        f"Expected 12-month alpha vs SPY: "
        f"{returns_output.alpha_analysis.net_expected_alpha_vs_spy:.1f}%."
    )


def _build_template_market_context(
    key_numbers: KeyNumbersDashboard,
    rotation: RotationDetectionOutput,
) -> str:
    """Build market context from template."""
    regime = rotation.market_regime
    parts = [
        f"Market regime: {regime.regime.upper()}. ",
        f"{regime.regime_note} ",
    ]
    if regime.bull_signals_present:
        parts.append(f"Bull signals: {', '.join(regime.bull_signals_present[:3])}. ")
    if regime.bear_signals_present:
        parts.append(f"Bear signals: {', '.join(regime.bear_signals_present[:3])}. ")

    signals_active = rotation.signals.signals_active
    parts.append(
        f"{signals_active}/5 rotation signals active. "
        f"Rotation type: {rotation.rotation_type.value}. "
    )
    if rotation.velocity:
        parts.append(f"Velocity: {rotation.velocity}. ")
    if rotation.strategist_notes:
        parts.append(f"Key note: {rotation.strategist_notes[0][:80]}.")

    return "".join(parts)


# ---------------------------------------------------------------------------
# LLM Narrative Generation
# ---------------------------------------------------------------------------

def _generate_llm_synthesis(
    key_numbers: KeyNumbersDashboard,
    connections: list[CrossAgentConnection],
    contradictions: list[Contradiction],
    rotation: RotationDetectionOutput,
    strategy: SectorStrategyOutput,
    risk: RiskAssessment,
    returns_output: ReturnsProjectionOutput,
    model: str = "claude-haiku-4-5-20251001",
) -> Optional[dict[str, str]]:
    """Generate narrative sections using Anthropic Claude.

    Returns dict with keys: investment_thesis, portfolio_narrative,
    risk_reward_assessment, market_context. Returns None on failure.
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0)
    except (ImportError, Exception) as e:
        logger.info(f"Anthropic SDK not available, using template fallback: {e}")
        return None

    # Build context for LLM
    numbers_dict = key_numbers.model_dump()
    connections_text = "\n".join(
        f"- {c.from_agent} → {c.to_agent}: {c.connection}"
        for c in connections
    )
    contradictions_text = "\n".join(
        f"- {c.agent_a} vs {c.agent_b}: {c.finding_a} | {c.finding_b}"
        for c in contradictions
    ) if contradictions else "No contradictions detected."

    prompt = f"""You are an investment strategist synthesizing a 9-agent analysis framework.

KEY NUMBERS:
{json.dumps(numbers_dict, indent=2)}

CROSS-AGENT CONNECTIONS:
{connections_text}

CONTRADICTIONS:
{contradictions_text}

ROTATION: {rotation.verdict.value} ({rotation.confidence}% confidence), {rotation.market_regime.regime} regime
STRATEGY: {strategy.rotation_response}
RISK: {risk.overall_status}, Sleep Well {risk.sleep_well_scores.overall_score}/10
RETURNS: Base {key_numbers.base_case_return_12m:.1f}%, Bull {key_numbers.bull_case_return_12m:.1f}%, Bear {key_numbers.bear_case_return_12m:.1f}%

Generate a JSON response with EXACTLY these 4 keys (no markdown, no extra text):
{{
  "investment_thesis": "200+ char thesis explaining WHY the portfolio is constructed this way",
  "portfolio_narrative": "100+ char narrative connecting rotation → strategy → portfolio",
  "risk_reward_assessment": "100+ char risk/reward tradeoff summary",
  "market_context": "100+ char current market environment description"
}}

RULES:
- Never recommend specific buy/sell actions
- Never mention position sizes or dollar amounts
- Focus on the WHY, not the WHAT
- Reference agent findings by name (e.g., "The Rotation Detector found...")
- If contradictions exist, explain how they inform the thesis"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1536,
            messages=[{"role": "user", "content": prompt}],
        )
        track_tokens("_generate_llm_synthesis", response)
        text = response.content[0].text if response.content else ""

        # Try to parse JSON directly or from fenced block
        text = text.strip()
        if text.startswith("```"):
            # Remove fenced block markers
            lines = text.split("\n")
            text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )

        result = json.loads(text)

        # Validate all 4 keys present with minimum lengths
        required = {
            "investment_thesis": 200,
            "portfolio_narrative": 100,
            "risk_reward_assessment": 100,
            "market_context": 100,
        }
        for key, min_len in required.items():
            if key not in result or len(str(result[key])) < min_len:
                logger.warning(
                    f"LLM response missing or short '{key}', falling back to template"
                )
                return None

        logger.info("LLM synthesis generated successfully")
        return {k: str(result[k]) for k in required}

    except Exception as e:
        logger.warning(f"LLM synthesis error: {e}")
        return None


# ---------------------------------------------------------------------------
# Deterministic Pipeline
# ---------------------------------------------------------------------------

def run_synthesizer_pipeline(
    research_output: ResearchOutput,
    analyst_output: AnalystOutput,
    rotation_output: RotationDetectionOutput,
    strategy_output: SectorStrategyOutput,
    portfolio_output: PortfolioOutput,
    risk_output: RiskAssessment,
    returns_output: ReturnsProjectionOutput,
    reconciler_output: ReconciliationOutput,
    educator_output: EducatorOutput,
) -> SynthesisOutput:
    """
    Run the Executive Summary Synthesizer pipeline.

    Deterministic extraction always runs. LLM generates narrative
    sections when available; template fallback otherwise.
    """
    logger.info("[Agent 10] Running Executive Summary Synthesizer pipeline ...")

    # Step 1: Extract key numbers (deterministic)
    key_numbers = _extract_key_numbers(
        research_output, analyst_output, rotation_output,
        strategy_output, portfolio_output, risk_output,
        returns_output, reconciler_output,
    )
    logger.info(
        f"[Agent 10] Key numbers: {key_numbers.stocks_rated} rated stocks, "
        f"{key_numbers.rotation_verdict} rotation, "
        f"{key_numbers.market_regime} regime"
    )

    # Step 2: Detect contradictions (deterministic)
    contradictions = _detect_contradictions(
        strategy_output, risk_output, rotation_output,
        portfolio_output, returns_output, reconciler_output,
    )
    logger.info(f"[Agent 10] Contradictions detected: {len(contradictions)}")

    # Step 3: Build cross-agent connections (deterministic)
    connections = _build_cross_agent_connections(
        rotation_output, strategy_output, portfolio_output,
        risk_output, returns_output,
    )
    logger.info(f"[Agent 10] Cross-agent connections: {len(connections)}")

    # Step 4: Build action items (deterministic)
    action_items = _build_action_items(
        reconciler_output, risk_output, returns_output,
    )
    logger.info(f"[Agent 10] Action items: {len(action_items)}")

    # Step 5: Try LLM narrative, fall back to template
    synthesis_source = "template"
    llm_result = _generate_llm_synthesis(
        key_numbers, connections, contradictions,
        rotation_output, strategy_output, risk_output, returns_output,
    )

    if llm_result:
        investment_thesis = llm_result["investment_thesis"]
        portfolio_narrative = llm_result["portfolio_narrative"]
        risk_reward_assessment = llm_result["risk_reward_assessment"]
        market_context = llm_result["market_context"]
        synthesis_source = "llm"
    else:
        investment_thesis = _build_template_thesis(
            key_numbers, connections, contradictions
        )
        portfolio_narrative = _build_template_narrative(
            key_numbers, rotation_output, strategy_output
        )
        risk_reward_assessment = _build_template_risk_reward(
            key_numbers, risk_output, returns_output
        )
        market_context = _build_template_market_context(
            key_numbers, rotation_output
        )

    # Step 6: Build summary
    summary = _build_summary(key_numbers, contradictions, synthesis_source)

    output = SynthesisOutput(
        investment_thesis=investment_thesis,
        portfolio_narrative=portfolio_narrative,
        risk_reward_assessment=risk_reward_assessment,
        market_context=market_context,
        key_numbers=key_numbers,
        cross_agent_connections=connections,
        contradictions=contradictions,
        action_items=action_items,
        synthesis_source=synthesis_source,
        analysis_date=date.today().isoformat(),
        summary=summary,
    )

    logger.info(
        f"[Agent 10] Done — source={synthesis_source}, "
        f"{len(connections)} connections, {len(contradictions)} contradictions, "
        f"{len(action_items)} action items"
    )

    return output


def _build_summary(
    key_numbers: KeyNumbersDashboard,
    contradictions: list[Contradiction],
    synthesis_source: str,
) -> str:
    """Build summary string (>= 50 chars)."""
    parts = [
        f"Executive synthesis ({synthesis_source}): "
        f"{key_numbers.rotation_verdict} rotation in {key_numbers.market_regime} regime.",
        f"{key_numbers.portfolio_positions} positions, "
        f"base return {key_numbers.base_case_return_12m:.1f}%, "
        f"risk status {key_numbers.risk_status}.",
    ]
    if contradictions:
        parts.append(f"{len(contradictions)} cross-agent contradiction(s) flagged.")
    summary = " ".join(parts)
    if len(summary) < 50:
        summary += " Framework v4.0 multi-agent synthesis complete."
    return summary
