"""
Agent 12: PatternAlpha — Per-Stock Pattern Scoring Engine
IBD Momentum Investment Framework v4.0

Evaluates every security against 5 wealth-creation patterns
(Platform Economics, Self-Cannibalization, Capital Allocation,
Category Creation, Inflection Timing) to produce a 150-point
Enhanced Score per stock.

Never makes buy/sell recommendations — only discovers, scores, and organizes.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import List, Optional

try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    Agent = None
    Task = None

from ibd_agents.schemas.analyst_output import AnalystOutput, RatedStock
from ibd_agents.schemas.pattern_output import (
    PATTERN_MAX_SCORES,
    PatternSubScore,
    PatternScoreBreakdown,
    BaseScoreBreakdown,
    EnhancedStockAnalysis,
    PatternAlert,
    PortfolioPatternSummary,
    PortfolioPatternOutput,
)
from ibd_agents.tools.pattern_analyzer import (
    PatternAnalyzerTool,
    compute_base_score,
    classify_enhanced_rating,
    generate_stock_pattern_alerts,
    assess_tier_fit,
    score_patterns_llm,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI Builders
# ---------------------------------------------------------------------------

def build_pattern_agent() -> "Agent":
    """Build CrewAI PatternAlpha per-stock scoring agent."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode")
    return Agent(
        role="PatternAlpha Scoring Analyst",
        goal=(
            "Score every security against 5 wealth-creation patterns "
            "(Platform Economics, Self-Cannibalization, Capital Allocation, "
            "Category Creation, Inflection Timing) to produce a 150-point "
            "Enhanced Score. Compute 100-point base scores from IBD/analyst/"
            "risk metrics, then overlay 50-point pattern scores via LLM. "
            "Never make buy/sell recommendations — only discover, score, "
            "and organize."
        ),
        backstory=(
            "You are a pattern recognition specialist who evaluates "
            "individual stocks against 5 wealth-creation patterns derived "
            "from decades of market research. You compute deterministic "
            "base scores from quantitative metrics, then score qualitative "
            "pattern dimensions using your broad knowledge of companies, "
            "industries, and business models."
        ),
        tools=[PatternAnalyzerTool()],
        verbose=True,
        allow_delegation=False,
        max_iter=15,
    )


def build_pattern_task(
    agent: "Agent",
    analyst_json: str = "",
) -> "Task":
    """Build CrewAI PatternAlpha scoring task."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode")
    return Task(
        description=f"""Score every stock against 5 wealth-creation patterns.

6 PHASES:

1. COMPUTE BASE SCORES (100 pts deterministic)
   - IBD Score (40): Composite, RS, EPS ratings
   - Analyst Score (30): Validation, IBD Keep, Multi-source
   - Risk Score (30): Sharpe, Alpha, Risk Rating

2. FILTER FOR PATTERN SCORING
   - Stocks with base_total >= 60 proceed to LLM
   - Others get enhanced_score = base_total, no pattern score

3. LLM: SCORE 5 PATTERNS (50 pts)
   - Platform Economics (0-12)
   - Self-Cannibalization (0-10)
   - Capital Allocation (0-10)
   - Category Creation (0-10)
   - Inflection Timing (0-8)

4. ASSEMBLE ENHANCED SCORES
   - enhanced_score = base_total + pattern_total (0-150)
   - Classify rating (stars + label)
   - Generate per-stock pattern alerts

5. BUILD PORTFOLIO SUMMARY
   - Count stocks scored, avg enhanced, tier candidates

6. ASSEMBLE OUTPUT

Analyst data:
{analyst_json}
""",
        expected_output=(
            "JSON object with stock_analyses (per-stock enhanced scores), "
            "pattern_alerts, portfolio_summary, scoring_source, "
            "methodology_notes, analysis_date, summary."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Deterministic Pipeline
# ---------------------------------------------------------------------------

def run_pattern_pipeline(
    analyst_output: AnalystOutput,
) -> PortfolioPatternOutput:
    """
    Run the PatternAlpha per-stock scoring pipeline.

    6 phases:
    1. Compute base scores (100 pts, deterministic)
    2. Filter for pattern scoring (base >= 60)
    3. LLM: Score 5 patterns per stock
    4. Assemble enhanced scores + alerts
    5. Build portfolio summary
    6. Assemble output
    """
    logger.info("[Agent 12] Running PatternAlpha per-stock scoring pipeline ...")

    rated_stocks = analyst_output.rated_stocks
    logger.info(f"[Agent 12] Scoring {len(rated_stocks)} rated stocks")

    # --- Phase 1: Compute base scores ---
    base_scores: dict[str, dict] = {}
    for stock in rated_stocks:
        stock_dict = stock.model_dump()
        base = compute_base_score(stock_dict)
        base_scores[stock.symbol] = base

    logger.info(
        f"[Agent 12] Phase 1: Base scores computed for {len(base_scores)} stocks"
    )

    # --- Phase 2: Filter for pattern scoring ---
    MIN_BASE_FOR_PATTERNS = 60
    stocks_for_llm: list[dict] = []
    for stock in rated_stocks:
        base = base_scores[stock.symbol]
        if base["base_total"] >= MIN_BASE_FOR_PATTERNS:
            stocks_for_llm.append({
                "symbol": stock.symbol,
                "company_name": stock.company_name,
                "sector": stock.sector,
                "tier": stock.tier,
            })

    logger.info(
        f"[Agent 12] Phase 2: {len(stocks_for_llm)}/{len(rated_stocks)} stocks "
        f"qualify for pattern scoring (base >= {MIN_BASE_FOR_PATTERNS})"
    )

    # --- Phase 3: LLM pattern scoring ---
    scoring_source = "deterministic"
    pattern_scores: dict[str, dict] = {}
    try:
        pattern_scores = score_patterns_llm(stocks_for_llm)
        if pattern_scores:
            scoring_source = "llm"
            logger.info(
                f"[Agent 12] Phase 3: LLM scored {len(pattern_scores)} stocks"
            )
        else:
            logger.info("[Agent 12] Phase 3: LLM returned no results, using deterministic only")
    except Exception as e:
        logger.warning(f"[Agent 12] Phase 3: LLM pattern scoring failed: {e}")

    # --- Phase 4: Assemble enhanced scores + alerts ---
    stock_analyses: list[EnhancedStockAnalysis] = []
    all_pattern_alerts: list[PatternAlert] = []

    for stock in rated_stocks:
        base = base_scores[stock.symbol]
        base_breakdown = BaseScoreBreakdown(
            ibd_score=base["ibd_score"],
            analyst_score=base["analyst_score"],
            risk_score=base["risk_score"],
            base_total=base["base_total"],
        )

        # Build pattern score breakdown if LLM provided scores
        pattern_breakdown: Optional[PatternScoreBreakdown] = None
        llm_data = pattern_scores.get(stock.symbol)

        if llm_data:
            p1 = llm_data.get("p1", 0)
            p2 = llm_data.get("p2", 0)
            p3 = llm_data.get("p3", 0)
            p4 = llm_data.get("p4", 0)
            p5 = llm_data.get("p5", 0)
            pattern_total = p1 + p2 + p3 + p4 + p5

            pattern_breakdown = PatternScoreBreakdown(
                p1_platform=PatternSubScore(
                    pattern_name="Platform Economics",
                    score=p1, max_score=12,
                    justification=llm_data.get("p1_justification", ""),
                ),
                p2_cannibalization=PatternSubScore(
                    pattern_name="Self-Cannibalization",
                    score=p2, max_score=10,
                    justification=llm_data.get("p2_justification", ""),
                ),
                p3_capital_allocation=PatternSubScore(
                    pattern_name="Capital Allocation",
                    score=p3, max_score=10,
                    justification=llm_data.get("p3_justification", ""),
                ),
                p4_category_creation=PatternSubScore(
                    pattern_name="Category Creation",
                    score=p4, max_score=10,
                    justification=llm_data.get("p4_justification", ""),
                ),
                p5_inflection_timing=PatternSubScore(
                    pattern_name="Inflection Timing",
                    score=p5, max_score=8,
                    justification=llm_data.get("p5_justification", ""),
                ),
                pattern_total=pattern_total,
                dominant_pattern=llm_data.get("dominant_pattern", "Platform Economics"),
                pattern_narrative=llm_data.get(
                    "pattern_narrative",
                    "Pattern analysis completed for this security."
                ),
            )

        # Compute enhanced score
        pattern_total_val = pattern_breakdown.pattern_total if pattern_breakdown else 0
        enhanced_score = base["base_total"] + pattern_total_val

        # Classify rating
        stars, label = classify_enhanced_rating(enhanced_score)

        # Generate per-stock pattern alerts
        stock_alert_strs: list[str] = []
        if llm_data:
            stock_alert_strs = generate_stock_pattern_alerts(
                stock.symbol, llm_data
            )

            # Also create formal PatternAlert objects for portfolio-level list
            for alert_type in stock_alert_strs:
                pattern_name_for_alert = _alert_type_to_pattern(alert_type)
                score_for_alert = llm_data.get(
                    _alert_type_to_score_key(alert_type), 0
                )
                all_pattern_alerts.append(PatternAlert(
                    alert_type=alert_type,
                    symbol=stock.symbol,
                    description=_build_alert_description(
                        alert_type, stock.symbol, stock.company_name,
                        pattern_name_for_alert, score_for_alert,
                    ),
                    pattern_name=pattern_name_for_alert,
                    pattern_score=score_for_alert,
                ))

        # Assess tier fit
        tier_rec = assess_tier_fit(
            enhanced_score,
            llm_data if llm_data else None,
            stock.tier,
        )

        # Determine per-stock scoring source
        stock_source = "llm" if llm_data else "deterministic"

        stock_analyses.append(EnhancedStockAnalysis(
            symbol=stock.symbol,
            company_name=stock.company_name,
            sector=stock.sector,
            tier=stock.tier,
            base_score=base_breakdown,
            pattern_score=pattern_breakdown,
            enhanced_score=enhanced_score,
            enhanced_rating=stars,
            enhanced_rating_label=label,
            pattern_alerts=stock_alert_strs,
            tier_recommendation=tier_rec,
            scoring_source=stock_source,
        ))

    logger.info(
        f"[Agent 12] Phase 4: {len(stock_analyses)} stocks scored, "
        f"{len(all_pattern_alerts)} pattern alerts generated"
    )

    # --- Phase 5: Build portfolio summary ---
    stocks_with_patterns = sum(
        1 for sa in stock_analyses if sa.pattern_score is not None
    )
    enhanced_scores = [sa.enhanced_score for sa in stock_analyses]
    avg_enhanced = (
        sum(enhanced_scores) / len(enhanced_scores) if enhanced_scores else 0.0
    )

    # Tier 1 candidates: meet T1 pattern requirements (enhanced >= 115, 2+ at 7)
    tier_1_candidates = 0
    category_kings = 0
    inflection_alerts = 0
    disruption_risks = 0

    for sa in stock_analyses:
        if sa.pattern_score:
            scores = [
                sa.pattern_score.p1_platform.score,
                sa.pattern_score.p2_cannibalization.score,
                sa.pattern_score.p3_capital_allocation.score,
                sa.pattern_score.p4_category_creation.score,
                sa.pattern_score.p5_inflection_timing.score,
            ]
            if sa.enhanced_score >= 115 and sum(1 for s in scores if s >= 7) >= 2:
                tier_1_candidates += 1
            if sa.pattern_score.p4_category_creation.score >= 8:
                category_kings += 1
            if sa.pattern_score.p5_inflection_timing.score >= 6:
                inflection_alerts += 1
            if sa.pattern_score.p2_cannibalization.score == 0:
                disruption_risks += 1

    portfolio_summary = PortfolioPatternSummary(
        total_stocks_scored=len(stock_analyses),
        stocks_with_patterns=stocks_with_patterns,
        avg_enhanced_score=round(avg_enhanced, 1),
        tier_1_candidates=tier_1_candidates,
        category_kings=category_kings,
        inflection_alerts=inflection_alerts,
        disruption_risks=disruption_risks,
    )

    logger.info(
        f"[Agent 12] Phase 5: Summary — {stocks_with_patterns} with patterns, "
        f"avg enhanced={avg_enhanced:.1f}, T1 candidates={tier_1_candidates}"
    )

    # --- Phase 6: Assemble output ---
    methodology_notes = (
        "6-phase PatternAlpha per-stock scoring engine: "
        "(1) 100-point base score from IBD ratings (40pts), analyst metrics (30pts), "
        "and risk assessment (30pts). "
        "(2) Stocks with base >= 60 proceed to pattern scoring. "
        "(3) LLM scores 5 wealth-creation patterns: Platform Economics (12pts), "
        "Self-Cannibalization (10pts), Capital Allocation (10pts), "
        "Category Creation (10pts), Inflection Timing (8pts). "
        "(4) Enhanced score = base + pattern (0-150), classified by star rating. "
        "(5) Portfolio summary with tier candidates and alert counts. "
        "(6) Output assembly with per-stock analyses and pattern alerts."
    )

    # Build summary (>= 50 chars)
    summary = (
        f"PatternAlpha scored {len(stock_analyses)} stocks: "
        f"{stocks_with_patterns} with 5-pattern scoring, "
        f"avg enhanced score {avg_enhanced:.1f}/150. "
        f"{tier_1_candidates} tier-1 candidates, "
        f"{category_kings} category kings, "
        f"{inflection_alerts} inflection alerts, "
        f"{disruption_risks} disruption risks. "
        f"Scoring source: {scoring_source}."
    )

    output = PortfolioPatternOutput(
        stock_analyses=stock_analyses,
        pattern_alerts=all_pattern_alerts,
        portfolio_summary=portfolio_summary,
        scoring_source=scoring_source,
        methodology_notes=methodology_notes,
        analysis_date=date.today().isoformat(),
        summary=summary,
    )

    logger.info(
        f"[Agent 12] Pipeline complete: {len(stock_analyses)} stocks, "
        f"{len(all_pattern_alerts)} alerts, source={scoring_source}"
    )
    return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alert_type_to_pattern(alert_type: str) -> str:
    """Map alert type to the relevant pattern name."""
    mapping = {
        "Category King": "Category Creation",
        "Inflection Alert": "Inflection Timing",
        "Disruption Risk": "Self-Cannibalization",
        "Pattern Imbalance": "Multiple Patterns",
    }
    return mapping.get(alert_type, alert_type)


def _alert_type_to_score_key(alert_type: str) -> str:
    """Map alert type to the pattern score key in LLM data."""
    mapping = {
        "Category King": "p4",
        "Inflection Alert": "p5",
        "Disruption Risk": "p2",
        "Pattern Imbalance": "p1",
    }
    return mapping.get(alert_type, "p1")


def _build_alert_description(
    alert_type: str,
    symbol: str,
    company_name: str,
    pattern_name: str,
    score: int,
) -> str:
    """Build a descriptive alert string (>= 20 chars)."""
    if alert_type == "Category King":
        return (
            f"{symbol} ({company_name}) scores {score}/10 on Category Creation, "
            f"indicating strong category ownership and TAM expansion potential"
        )
    elif alert_type == "Inflection Alert":
        return (
            f"{symbol} ({company_name}) scores {score}/8 on Inflection Timing, "
            f"suggesting the stock is near a significant inflection point"
        )
    elif alert_type == "Disruption Risk":
        return (
            f"{symbol} ({company_name}) scores 0/10 on Self-Cannibalization, "
            f"indicating no evidence of proactive disruption readiness"
        )
    elif alert_type == "Pattern Imbalance":
        return (
            f"{symbol} ({company_name}) shows pattern imbalance — "
            f"at least one pattern scores 0 while another scores 8+, "
            f"suggesting uneven competitive positioning"
        )
    return f"{symbol}: {alert_type} alert on {pattern_name} (score={score})"
