"""
Agent 09: Educator
Investment Educator — IBD Momentum Framework v4.0

Receives outputs from all 8 prior agents and produces
EducatorOutput with educational explanations, analogies,
glossary, and action items. Never makes investment decisions
or modifies the analysis — only explains it.
"""

from __future__ import annotations

import logging
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
from ibd_agents.schemas.educator_output import (
    ConceptLesson,
    EducatorOutput,
    ETFExplanation,
    KeepExplanations,
    StockExplanation,
    TierGuide,
    TransitionGuide,
)
from ibd_agents.schemas.portfolio_output import (
    ALL_KEEPS,
    FUNDAMENTAL_KEEPS,
    IBD_KEEPS,
    KEEP_METADATA,
    TRAILING_STOPS,
    PortfolioOutput,
)
from ibd_agents.schemas.reconciliation_output import ReconciliationOutput
from ibd_agents.schemas.returns_projection_output import ReturnsProjectionOutput
from ibd_agents.schemas.risk_output import RiskAssessment
from ibd_agents.schemas.rotation_output import RotationDetectionOutput
from ibd_agents.schemas.strategy_output import SectorStrategyOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI Agent Builder
# ---------------------------------------------------------------------------

def build_educator_agent() -> "Agent":
    """Create the Investment Educator Agent. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Agent(
        role="Investment Educator",
        goal=(
            "Create comprehensive educational content that explains every "
            "decision made by the 7-agent crew using Framework v4.0 "
            "terminology, IBD rating definitions, tier rationale, keep-position "
            "reasoning, and risk findings in clear, accessible terms."
        ),
        backstory=(
            "You are a warm, clear, analogy-driven investment educator. "
            "You translate complex IBD methodology into accessible language "
            "without ever making investment decisions or overriding other agents. "
            "You explain — you never prescribe."
        ),
        tools=[],
        verbose=True,
        allow_delegation=False,
        max_iter=10,
        temperature=0.7,
    )


def build_educator_task(
    agent: "Agent",
    all_outputs_json: str = "",
) -> "Task":
    """Create the Educator task. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Task(
        description=f"""Create educational content explaining the entire portfolio analysis.

STEPS:
1. Write executive summary (2-3 paragraphs, 200+ chars)
2. Explain the 3-tier system with analogies
3. Explain the 14 keep positions across 2 categories
4. Explain top 15-25 stock picks with ratings and context
5. Translate rotation detection verdict into plain language
6. Explain risk findings and Sleep Well scores
7. Explain portfolio transition plan and money flow
8. Teach 5+ IBD concepts with examples from the analysis
9. Provide 3-7 action items
10. Build glossary of 15+ financial terms

Pipeline data:
{all_outputs_json}
""",
        expected_output=(
            "JSON with executive_summary, tier_guide, keep_explanations, "
            "stock_explanations, rotation_explanation, risk_explainer, "
            "transition_guide, concept_lessons, action_items, glossary, "
            "analysis_date, summary."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Deterministic Pipeline
# ---------------------------------------------------------------------------

def run_educator_pipeline(
    portfolio_output: PortfolioOutput,
    analyst_output: AnalystOutput,
    rotation_output: RotationDetectionOutput,
    strategy_output: SectorStrategyOutput,
    risk_output: RiskAssessment,
    reconciliation_output: ReconciliationOutput,
    returns_output: Optional[ReturnsProjectionOutput] = None,
) -> EducatorOutput:
    """
    Run the deterministic educator pipeline without LLM.

    Generates educational content by templating from all prior agent outputs.
    """
    logger.info("[Agent 09] Running Educator pipeline ...")

    # 1. Executive summary
    exec_summary = _build_executive_summary(
        portfolio_output, analyst_output, rotation_output, risk_output
    )

    # 2. Tier guide
    tier_guide = _build_tier_guide(portfolio_output)

    # 3. Keep explanations
    keep_explanations = _build_keep_explanations(portfolio_output)

    # 4. Stock explanations (top 15-25)
    stock_explanations = _build_stock_explanations(
        portfolio_output, analyst_output
    )

    # 5. Rotation explanation
    rotation_explanation = _build_rotation_explanation(rotation_output)

    # 6. Risk explainer
    risk_explainer = _build_risk_explainer(risk_output)

    # 7. Returns explanation (optional, from Agent 07)
    returns_explanation = ""
    if returns_output is not None:
        returns_explanation = _build_returns_explanation(returns_output)

    # 8. Transition guide
    transition_guide = _build_transition_guide(reconciliation_output)

    # 9. Concept lessons
    concept_lessons = _build_concept_lessons(analyst_output, portfolio_output)
    if returns_output is not None:
        concept_lessons.append(_build_returns_concept_lesson(returns_output))

    # 9b. ETF explanations
    etf_explanations = _build_etf_explanations(portfolio_output, analyst_output)

    # 10. Action items
    action_items = _build_action_items(risk_output, reconciliation_output)

    # 11. Glossary
    glossary = _build_glossary()
    if returns_output is not None:
        glossary.update(_returns_glossary_entries())

    # 12. Summary
    summary = (
        f"Educational report explaining {len(stock_explanations)} stock selections, "
        f"3-tier portfolio structure, 14 keeps, {len(concept_lessons)} IBD concepts, "
        f"and 4-week transition plan per Framework v4.0."
    )

    output = EducatorOutput(
        executive_summary=exec_summary,
        tier_guide=tier_guide,
        keep_explanations=keep_explanations,
        stock_explanations=stock_explanations,
        rotation_explanation=rotation_explanation,
        risk_explainer=risk_explainer,
        returns_explanation=returns_explanation,
        transition_guide=transition_guide,
        concept_lessons=concept_lessons,
        action_items=action_items,
        etf_explanations=etf_explanations,
        glossary=glossary,
        analysis_date=date.today().isoformat(),
        summary=summary,
    )

    logger.info(
        f"[Agent 09] Done — {len(stock_explanations)} stocks explained, "
        f"{len(concept_lessons)} concepts taught, {len(glossary)} glossary entries"
    )

    return output


# ---------------------------------------------------------------------------
# Content Builders
# ---------------------------------------------------------------------------

def _build_executive_summary(
    portfolio: PortfolioOutput,
    analyst: AnalystOutput,
    rotation: RotationDetectionOutput,
    risk: RiskAssessment,
) -> str:
    """Build 200+ char executive summary."""
    t1_pct = portfolio.tier_1.actual_pct
    t2_pct = portfolio.tier_2.actual_pct
    t3_pct = portfolio.tier_3.actual_pct
    total_pos = portfolio.total_positions
    n_sectors = len(portfolio.sector_exposure)
    regime = rotation.market_regime.regime
    sleep_score = risk.sleep_well_scores.overall_score
    n_vetoes = len(risk.vetoes)
    n_warnings = len(risk.warnings)

    # Build multi-paragraph summary
    p1 = (
        f"This analysis used the IBD Momentum Investment Framework v4.0 to screen, rate, and organize "
        f"growth stock opportunities into a 3-tier portfolio. From {len(analyst.rated_stocks)} rated securities, "
        f"{total_pos} positions were selected across {n_sectors} sectors. "
        f"Tier 1 Momentum holds {t1_pct:.0f}% for highest-conviction growth picks, "
        f"Tier 2 Quality Growth holds {t2_pct:.0f}% for strong but stable companies, "
        f"and Tier 3 Defensive holds {t3_pct:.0f}% for capital preservation."
    )

    p2 = (
        f"Of the {total_pos} positions, 14 are pre-committed keeps: "
        f"5 fundamental value holds and 9 IBD elite-rated stocks "
        f"with Composite 93+ and RS 90+. The Rotation Detector assessed the current regime "
        f"as {regime} with rotation status {rotation.verdict.value}. "
        f"The Risk Officer ran 10 checks with {n_vetoes} vetoes and {n_warnings} warnings, "
        f"assigning a Sleep Well Score of {sleep_score}/10."
    )

    return f"{p1}\n\n{p2}"


def _build_tier_guide(portfolio: PortfolioOutput) -> TierGuide:
    """Build tier guide from portfolio data."""
    t1 = portfolio.tier_1
    t2 = portfolio.tier_2
    t3 = portfolio.tier_3

    return TierGuide(
        overview=(
            "The 3-tier portfolio system organizes investments by risk and return expectations. "
            "Each tier has its own position limits, trailing stops, and target returns, "
            "creating a balanced structure that combines growth ambition with capital protection."
        ),
        tier_1_description=(
            f"Tier 1 Momentum ({t1.actual_pct:.0f}% of portfolio, {len(t1.positions)} positions): "
            f"Highest-conviction growth picks with Composite 95+ and RS 90+. "
            f"22% trailing stops. Like a sports car — thrilling speed but feels every bump. "
            f"Target return: 25-30%."
        ),
        tier_2_description=(
            f"Tier 2 Quality Growth ({t2.actual_pct:.0f}% of portfolio, {len(t2.positions)} positions): "
            f"Strong companies rated 85+ with proven track records. "
            f"18% trailing stops. Like a luxury sedan — good speed with smooth ride. "
            f"Target return: 18-22%."
        ),
        tier_3_description=(
            f"Tier 3 Defensive ({t3.actual_pct:.0f}% of portfolio, {len(t3.positions)} positions): "
            f"Capital preservation with quality stocks and defensive ETFs. "
            f"12% trailing stops. Like an armored SUV — handles any terrain safely. "
            f"Target return: 12-15%."
        ),
        choosing_advice=(
            "Weight toward T1 if comfortable with volatility and have a long time horizon. "
            "Weight toward T3 if closer to retirement or prefer stability. "
            "The current split reflects a balanced growth profile."
        ),
        current_allocation=(
            f"{t1.actual_pct:.0f}% T1 / {t2.actual_pct:.0f}% T2 / {t3.actual_pct:.0f}% T3 / "
            f"{portfolio.cash_pct:.0f}% Cash"
        ),
        analogy=(
            "Tier 1 is a sports car (fast, exciting), "
            "Tier 2 is a luxury sedan (balanced performance), "
            "Tier 3 is an armored SUV (safety-first)."
        ),
    )


def _build_keep_explanations(portfolio: PortfolioOutput) -> KeepExplanations:
    """Build keep category explanations."""
    fund_syms = ", ".join(FUNDAMENTAL_KEEPS)
    ibd_syms = ", ".join(IBD_KEEPS)

    return KeepExplanations(
        overview=(
            "14 positions are pre-committed across 2 categories: fundamental value holds "
            "and IBD elite-rated stocks. These are kept regardless "
            "of the screening process because they serve specific strategic purposes."
        ),
        fundamental_keeps=(
            f"5 Fundamental Keeps ({fund_syms}): Retained for strong value metrics "
            f"(low PE ratios, strong forward earnings) despite missing some elite IBD thresholds. "
            f"Like keeping a reliable car that may not win races but gets you there safely."
        ),
        ibd_keeps=(
            f"9 IBD Elite Keeps ({ibd_syms}): Stocks with Composite 93+ AND RS 90+ — "
            f"the honor roll students who earned their spot with outstanding IBD ratings."
        ),
        total_keeps=14,
    )


def _build_stock_explanations(
    portfolio: PortfolioOutput,
    analyst: AnalystOutput,
) -> list[StockExplanation]:
    """Build explanations for top 15-25 stocks and ETFs."""
    # Build rated stock and ETF lookups
    rated_map = {rs.symbol: rs for rs in analyst.rated_stocks}
    etf_map = {e.symbol: e for e in analyst.rated_etfs}

    explanations: list[StockExplanation] = []

    # Process all tiers
    for tier_data, tier_num in [
        (portfolio.tier_1, 1),
        (portfolio.tier_2, 2),
        (portfolio.tier_3, 3),
    ]:
        for pos in tier_data.positions:
            # Determine keep category
            keep_cat = None
            if pos.symbol in KEEP_METADATA:
                keep_cat = KEEP_METADATA[pos.symbol]["category"]

            if pos.asset_type == "etf":
                # ETF explanation path
                etf = etf_map.get(pos.symbol)
                if etf is not None:
                    rs_str = f"RS {etf.rs_rating}" if etf.rs_rating else "RS N/A"
                    ad_str = f"Acc/Dis {etf.acc_dis_rating}" if etf.acc_dis_rating else "Acc/Dis N/A"
                    ytd_str = f"YTD {etf.ytd_change:+.1f}%" if etf.ytd_change is not None else ""
                    ratings_text = f"{rs_str}, {ad_str}"
                    if ytd_str:
                        ratings_text += f", {ytd_str}"
                    key_strength = etf.strengths[0] if etf.strengths else "Diversified ETF exposure"
                    key_risk = etf.weaknesses[0] if etf.weaknesses else "Sector-specific concentration risk"
                    company_name = etf.name
                else:
                    ratings_text = "ETF ratings not available from analyst"
                    key_strength = "Diversified ETF exposure for portfolio balance"
                    key_risk = "Sector-specific concentration risk"
                    company_name = pos.company_name or pos.symbol

                # Build why selected for ETFs
                why_parts = []
                if keep_cat:
                    why_parts.append(f"Retained as {keep_cat} keep: {KEEP_METADATA[pos.symbol]['reason']}")
                else:
                    why_parts.append(f"Selected as theme ETF for diversified sector exposure and portfolio balance")
                if etf is not None and etf.rs_rating:
                    why_parts.append(f"IBD Relative Strength of {etf.rs_rating}/99")
                why_selected = ". ".join(why_parts) + "."
                if len(why_selected) < 50:
                    why_selected += f" Placed in Tier {tier_num} for tactical allocation."

                tier_label = ['', 'momentum', 'quality growth', 'defensive'][tier_num]
                one_liner = f"{pos.symbol}: {tier_label} ETF providing diversified exposure"
            else:
                # Stock explanation path
                rs = rated_map.get(pos.symbol)

                # Build ratings explanation
                if rs is not None:
                    comp_str = f"Composite {rs.composite_rating}" if rs.composite_rating else "Composite N/A"
                    rs_str = f"RS {rs.rs_rating}" if rs.rs_rating else "RS N/A"
                    eps_str = f"EPS {rs.eps_rating}" if rs.eps_rating else "EPS N/A"
                    ratings_text = f"{comp_str}, {rs_str}, {eps_str}"
                    if rs.composite_rating and rs.composite_rating >= 95:
                        ratings_text += f" — top {100 - rs.composite_rating}% of all stocks"
                else:
                    ratings_text = "Ratings not available from analyst screening"

                # Build why selected
                why_parts = []
                if keep_cat == "fundamental":
                    why_parts.append(f"Retained as fundamental keep for strong value metrics: {KEEP_METADATA[pos.symbol]['reason']}")
                elif keep_cat == "ibd":
                    why_parts.append(f"Retained as IBD elite keep with outstanding ratings: {KEEP_METADATA[pos.symbol]['reason']}")
                else:
                    why_parts.append(f"Selected through IBD screening with strong fundamentals and momentum characteristics")

                if rs is not None and rs.composite_rating:
                    why_parts.append(f"IBD rates it {rs.composite_rating}/99 overall")
                why_selected = ". ".join(why_parts) + "."

                if len(why_selected) < 50:
                    why_selected += f" Placed in Tier {tier_num} based on fundamental and momentum analysis."

                # Key strength/risk from analyst
                if rs is not None:
                    sector = rs.sector
                    key_strength = rs.strengths[0] if rs.strengths else f"Strong IBD ratings in {sector}"
                    key_risk = rs.weaknesses[0] if rs.weaknesses else f"Market volatility in {sector} sector"
                    company_name = rs.company_name
                else:
                    sector = "Unknown"
                    key_strength = f"Selected for Tier {tier_num} portfolio placement"
                    key_risk = f"Market volatility risk applies to all equity positions"
                    company_name = pos.symbol

                one_liner = f"{pos.symbol}: {sector} sector {['', 'momentum leader', 'quality growth', 'defensive hold'][tier_num]}"

            stop_pct = TRAILING_STOPS[tier_num]
            cap_label = pos.cap_size.capitalize() if pos.cap_size else "Unknown"
            position_context = f"T{tier_num} {cap_label}-cap position at {pos.target_pct:.1f}% allocation, {stop_pct}% trailing stop"

            explanations.append(StockExplanation(
                symbol=pos.symbol,
                company_name=company_name,
                tier=tier_num,
                cap_size=pos.cap_size,
                keep_category=keep_cat,
                one_liner=one_liner,
                why_selected=why_selected,
                ibd_ratings_explained=ratings_text,
                key_strength=key_strength,
                key_risk=key_risk,
                position_context=position_context,
                analogy=None,
            ))

            if len(explanations) >= 25:
                break
        if len(explanations) >= 25:
            break

    return explanations


def _build_rotation_explanation(rotation: RotationDetectionOutput) -> str:
    """Translate rotation verdict into plain language."""
    verdict = rotation.verdict.value
    signals_active = rotation.signals.signals_active
    confidence = rotation.confidence
    total_signals = 5

    if verdict == "ACTIVE":
        status_text = (
            f"The Rotation Detector confirmed active sector rotation with confidence {confidence}%. "
            f"{signals_active} of {total_signals} signals fired — like the seasons clearly changing. "
            f"Market leadership is shifting, and the portfolio has been positioned to benefit."
        )
    elif verdict == "EMERGING":
        status_text = (
            f"The Rotation Detector found emerging signs of sector rotation with confidence {confidence}%. "
            f"{signals_active} of {total_signals} signals fired — like the first cool morning in August. "
            f"Could signal a real shift, or could be temporary. The portfolio is partially adjusted."
        )
    else:
        status_text = (
            f"The Rotation Detector found no significant sector rotation. Confidence {confidence}%. "
            f"Only {signals_active} of {total_signals} signals fired — market leadership is stable, "
            f"like mid-summer. No need to change course based on rotation alone."
        )

    regime = rotation.market_regime.regime
    status_text += f" Current market regime: {regime}."
    return status_text


def _build_risk_explainer(risk: RiskAssessment) -> str:
    """Translate risk findings into non-scary terms."""
    n_checks = len(risk.check_results)
    n_pass = sum(1 for c in risk.check_results if c.status == "PASS")
    n_vetoes = len(risk.vetoes)
    n_warnings = len(risk.warnings)
    sleep = risk.sleep_well_scores.overall_score

    parts = [
        f"The Risk Officer — the portfolio's safety inspector — ran {n_checks} checks. "
        f"{n_pass} passed cleanly."
    ]

    if n_vetoes > 0:
        parts.append(
            f" {n_vetoes} veto(es) were issued — hard violations that must be fixed before proceeding."
        )
    elif n_warnings > 0:
        parts.append(
            f" {n_warnings} warning(s) flagged — areas to monitor but not deal-breakers."
        )
    else:
        parts.append(" No vetoes or warnings — the portfolio meets all framework limits.")

    parts.append(
        f" Sleep Well Score: {sleep}/10 — "
    )
    if sleep >= 8:
        parts.append("like a comfortable hotel bed. Rest easy.")
    elif sleep >= 6:
        parts.append("like a good hotel bed. Comfortable, with occasional awareness of market weather.")
    else:
        parts.append("some turbulence expected. The trailing stops serve as your safety net.")

    return "".join(parts)


def _build_transition_guide(recon: ReconciliationOutput) -> TransitionGuide:
    """Build transition guide from reconciliation output."""
    sells = [a for a in recon.actions if a.action_type == "SELL"]
    buys = [a for a in recon.actions if a.action_type == "BUY"]
    adds = [a for a in recon.actions if a.action_type == "ADD"]
    trims = [a for a in recon.actions if a.action_type == "TRIM"]
    keeps = [a for a in recon.actions if a.action_type == "KEEP"]

    sell_syms = ", ".join(a.symbol for a in sells[:5])
    if len(sells) > 5:
        sell_syms += f" and {len(sells) - 5} more"

    buy_syms = ", ".join(a.symbol for a in buys[:5])
    if len(buys) > 5:
        buy_syms += f" and {len(buys) - 5} more"

    mf = recon.money_flow
    metrics = recon.transformation_metrics

    return TransitionGuide(
        overview=(
            f"The portfolio transition involves {len(recon.actions)} total actions over 4 weeks: "
            f"{len(sells)} sells, {len(buys)} new buys, {len(adds)} adds, {len(trims)} trims, "
            f"and {len(keeps)} keeps. Week 1 focuses on liquidating non-recommended positions "
            f"to generate cash for new investments in weeks 2-4."
        ),
        what_to_sell=(
            f"{len(sells)} positions to sell: {sell_syms}. These are not in the recommended portfolio."
        ),
        what_to_buy=(
            f"{len(buys)} new positions to buy: {buy_syms}. These are new additions from the IBD screening."
        ),
        money_flow_explained=(
            f"Cash sources: selling non-recommended positions generates "
            f"proceeds, trimming oversized positions provides additional funds. "
            f"Cash uses: buying new recommended positions and adding to undersized ones. "
            f"Net cash change maintains the target cash reserve."
        ),
        timeline_explained=(
            "Week 1: LIQUIDATION — sell non-recommended positions to free cash. "
            "Week 2: T1 MOMENTUM — establish highest-conviction growth positions. "
            "Week 3: T2 QUALITY GROWTH — build the balanced growth core. "
            "Week 4: T3 DEFENSIVE — complete the defensive safety net."
        ),
        before_after_summary=(
            f"Before: {metrics.before_position_count} positions across {metrics.before_sector_count} sectors. "
            f"After: {metrics.after_position_count} positions across {metrics.after_sector_count} sectors."
        ),
        key_numbers=(
            f"Turnover: {metrics.turnover_pct:.0f}%, "
            f"{len(sells)} sells, {len(buys)} buys, "
            f"{metrics.before_position_count} to {metrics.after_position_count} positions"
        ),
    )


def _build_concept_lessons(
    analyst: AnalystOutput,
    portfolio: PortfolioOutput,
) -> list[ConceptLesson]:
    """Build 5+ concept lessons using actual portfolio examples."""
    # Find example stocks for each concept
    top_stock = analyst.rated_stocks[0] if analyst.rated_stocks else None
    top_sym = top_stock.symbol if top_stock else "CLS"

    lessons = [
        ConceptLesson(
            concept="Composite Rating",
            simple_explanation=(
                "A single number from 1-99 that combines all of IBD's individual ratings "
                "(earnings, price strength, fundamentals, institutional interest) into one overall grade."
            ),
            analogy="Like a student's GPA — combines math, English, and science into one number.",
            why_it_matters="Quickly identifies the strongest stocks without checking 5 separate ratings.",
            example_from_analysis=(
                f"{top_sym} has Composite {top_stock.composite_rating if top_stock else 99} — "
                f"ranked in the top {100 - (top_stock.composite_rating or 99)}% of all stocks."
            ),
            framework_reference="Framework v4.0 Section 2.1",
        ),
        ConceptLesson(
            concept="Relative Strength (RS) Rating",
            simple_explanation=(
                "Measures how a stock's price has performed relative to all other stocks "
                "over the last 12 months. RS 90 means it outperformed 90% of stocks."
            ),
            analogy="Like class rank — RS 90 means you finished ahead of 90% of your classmates.",
            why_it_matters="IBD research shows most big winners had RS 87+ before their biggest moves.",
            example_from_analysis=(
                f"{top_sym} has RS {top_stock.rs_rating if top_stock else 97} — strong price momentum "
                f"indicating institutional buying and market leadership."
            ),
            framework_reference="Framework v4.0 Section 2.2",
        ),
        ConceptLesson(
            concept="Three-Tier Portfolio System",
            simple_explanation=(
                "The portfolio is organized into three tiers by risk level: Momentum (high growth), "
                "Quality Growth (balanced), and Defensive (capital preservation)."
            ),
            analogy="Like a 3-vehicle garage: sports car for excitement, sedan for daily use, SUV for safety.",
            why_it_matters="Allows investors to balance growth ambition with protection in one portfolio.",
            example_from_analysis=(
                f"T1 holds {len(portfolio.tier_1.positions)} positions at {portfolio.tier_1.actual_pct:.0f}%, "
                f"T2 holds {len(portfolio.tier_2.positions)} at {portfolio.tier_2.actual_pct:.0f}%, "
                f"T3 holds {len(portfolio.tier_3.positions)} at {portfolio.tier_3.actual_pct:.0f}%."
            ),
            framework_reference="Framework v4.0 Section 3.1",
        ),
        ConceptLesson(
            concept="Trailing Stop Protocol",
            simple_explanation=(
                "An automatic sell trigger set below the purchase price (22%/18%/12% by tier) "
                "that rises with the stock but never falls, protecting gains."
            ),
            analogy="Like raising the floor in a building as you climb — you can always go up, but the floor rises too.",
            why_it_matters="Limits downside while allowing unlimited upside. Stops tighten as gains grow.",
            example_from_analysis=(
                "T1 positions start with 22% stops. After a 10% gain, stops tighten to 15-18%. "
                "After 20% gain, stops tighten further to 10-12%."
            ),
            framework_reference="Framework v4.0 Section 4.3",
        ),
        ConceptLesson(
            concept="Multi-Source Validation",
            simple_explanation=(
                "Stocks earn points from multiple independent sources (IBD lists, analyst ratings, "
                "Schwab themes). Score 5+ from 2+ providers means multi-source validated."
            ),
            analogy="Like a job interview — being recommended by 3 people who do not know each other is very strong.",
            why_it_matters="Reduces the risk of relying on any single source for stock selection.",
            example_from_analysis=(
                "Stocks appearing on both IBD 50 and analyst recommended lists "
                "earn higher validation scores and increased confidence ratings."
            ),
            framework_reference="Framework v4.0 Section 5.1",
        ),
        ConceptLesson(
            concept="Sector Rotation",
            simple_explanation=(
                "The pattern of different market sectors taking turns leading performance. "
                "Technology may lead for months, then energy or healthcare takes over."
            ),
            analogy="Like seasons changing — different crops thrive in different seasons.",
            why_it_matters="Detecting rotation early allows the portfolio to shift toward rising sectors.",
            example_from_analysis=(
                "The Rotation Detector uses 5 independent signals to determine "
                "whether rotation is active, emerging, or not present."
            ),
            framework_reference="Framework v4.0 Section 6.1",
        ),
        ConceptLesson(
            concept="EPS Rating",
            simple_explanation=(
                "Earnings Per Share rating from 1-99 comparing a company's profit growth "
                "(last 2 quarters and 3-year trend) to all other stocks."
            ),
            analogy="Like a company's report card for profit growth — measures both recent and sustained performance.",
            why_it_matters="Companies with accelerating earnings are more likely to see price appreciation.",
            example_from_analysis=(
                f"{top_sym} has EPS {top_stock.eps_rating if top_stock else 99} — "
                f"indicating profit growth exceeding most companies in the market."
            ),
            framework_reference="Framework v4.0 Section 2.3",
        ),
    ]

    # ETF concept lessons
    etf_count = sum(
        1 for t in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]
        for p in t.positions if p.asset_type == "etf"
    )
    stock_count = sum(
        1 for t in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]
        for p in t.positions if p.asset_type == "stock"
    )

    lessons.append(ConceptLesson(
        concept="ETF Selection and Ranking",
        simple_explanation=(
            "ETFs (Exchange-Traded Funds) are rated using IBD Relative Strength and "
            "Accumulation/Distribution grades, then scored and ranked within their "
            "focus area (e.g. semiconductors, healthcare) to pick the best in class."
        ),
        analogy=(
            "Like choosing the best team in each league — you rank all basketball teams "
            "against each other, all football teams against each other, then pick the champion from each."
        ),
        why_it_matters=(
            "Not all ETFs in a theme are equal. Ranking ensures the portfolio holds "
            "the strongest performer in each sector or theme."
        ),
        example_from_analysis=(
            f"The portfolio includes {etf_count} ETFs alongside {stock_count} stocks, "
            f"each selected as the top-ranked fund in its focus area."
        ),
        framework_reference="Framework v4.0 Section 4.1 ETF Screening",
    ))

    lessons.append(ConceptLesson(
        concept="Theme ETFs vs Broad Market ETFs",
        simple_explanation=(
            "Theme ETFs focus on a specific trend (AI, clean energy, cybersecurity) while "
            "broad market ETFs (like QQQ or VGT) provide diversified exposure across many companies. "
            "The portfolio uses both for targeted growth and broad stability."
        ),
        analogy=(
            "Theme ETFs are like a specialty restaurant (great at one cuisine), "
            "while broad ETFs are like a buffet (something for everyone)."
        ),
        why_it_matters=(
            "Theme ETFs capture emerging trends with higher upside but more concentration risk. "
            "Broad ETFs reduce single-stock risk and provide a stable base."
        ),
        example_from_analysis=(
            f"The portfolio balances theme and broad ETFs across {etf_count} positions, "
            f"ensuring both targeted growth exposure and diversified protection."
        ),
        framework_reference="Framework v4.0 Section 4.2 ETF Classification",
    ))

    return lessons


def _build_etf_explanations(
    portfolio: PortfolioOutput,
    analyst: AnalystOutput,
) -> list[ETFExplanation]:
    """Build educational explanations for ETF positions."""
    etf_map = {e.symbol: e for e in analyst.rated_etfs}
    explanations: list[ETFExplanation] = []

    for tier_data, tier_num in [
        (portfolio.tier_1, 1),
        (portfolio.tier_2, 2),
        (portfolio.tier_3, 3),
    ]:
        for pos in tier_data.positions:
            if pos.asset_type != "etf":
                continue

            etf = etf_map.get(pos.symbol)
            name = etf.name if etf else (pos.company_name or pos.symbol)

            # Build one-liner
            tier_label = ["", "Momentum", "Quality Growth", "Defensive"][tier_num]
            one_liner = f"{pos.symbol}: {tier_label} ETF at {pos.target_pct:.1f}% allocation"

            # Build why_selected
            why_parts = []
            if pos.symbol in KEEP_METADATA:
                why_parts.append(
                    f"Retained as {KEEP_METADATA[pos.symbol]['category']} keep: "
                    f"{KEEP_METADATA[pos.symbol]['reason']}"
                )
            else:
                why_parts.append("Selected through ETF screening for diversified sector exposure")
            if etf and etf.conviction:
                why_parts.append(f"Conviction score: {etf.conviction}/10")
            if etf and etf.rs_rating:
                why_parts.append(f"RS Rating: {etf.rs_rating}/99")
            why_selected = ". ".join(why_parts) + "."
            if len(why_selected) < 30:
                why_selected += f" Placed in Tier {tier_num}."

            # Theme context
            theme_context = None
            if etf and etf.focus:
                theme_context = f"Focus area: {etf.focus}. "
                if etf.focus_rank:
                    theme_context += f"Ranked #{etf.focus_rank} in its focus group."

            # Conviction explained
            conviction_explained = None
            if etf and etf.conviction:
                conv = etf.conviction
                if conv >= 8:
                    conviction_explained = f"High conviction ({conv}/10) — strong ratings and thematic tailwinds."
                elif conv >= 5:
                    conviction_explained = f"Moderate conviction ({conv}/10) — solid fundamentals with room for growth."
                else:
                    conviction_explained = f"Baseline conviction ({conv}/10) — meets screening thresholds."

            # Strengths / risks
            key_strength = "Diversified ETF exposure for portfolio balance"
            key_risk = "Sector-specific concentration risk"
            if etf:
                if etf.strengths:
                    key_strength = etf.strengths[0]
                if etf.weaknesses:
                    key_risk = etf.weaknesses[0]

            explanations.append(ETFExplanation(
                symbol=pos.symbol,
                name=name,
                tier=tier_num,
                one_liner=one_liner,
                why_selected=why_selected,
                theme_context=theme_context,
                conviction_explained=conviction_explained,
                key_strength=key_strength,
                key_risk=key_risk,
            ))

    return explanations


def _build_action_items(
    risk: RiskAssessment,
    recon: ReconciliationOutput,
) -> list[str]:
    """Build 3-7 actionable next steps."""
    items = [
        "Review the 4-week implementation timeline and execution order before making any changes",
        "Set trailing stops at the specified percentages for each tier immediately after purchase",
        "Monitor sector rotation signals monthly for changes in market leadership",
    ]

    if risk.warnings:
        items.append(
            f"Address the {len(risk.warnings)} risk warning(s) flagged by the Risk Officer"
        )

    if risk.vetoes:
        items.append(
            f"Resolve the {len(risk.vetoes)} veto(es) before proceeding with implementation"
        )

    items.append(
        "Review the Sleep Well Score breakdown to ensure alignment with personal risk tolerance"
    )

    # Cap at 7
    return items[:7]


def _build_glossary() -> dict[str, str]:
    """Build comprehensive glossary of financial terms."""
    return {
        "Composite Rating": (
            "IBD's master score (1-99) combining earnings, relative strength, "
            "fundamentals, and institutional interest. 99 = top 1% of all stocks."
        ),
        "RS Rating": (
            "Relative Strength — how the stock's price compares to all others "
            "over 12 months. RS 90 = outperformed 90% of stocks."
        ),
        "EPS Rating": (
            "Earnings Per Share rating (1-99) measuring profit growth vs all stocks. "
            "Compares last 2 quarters AND 3-year trend."
        ),
        "SMR Rating": (
            "Sales, Margins, Return on equity — grades A through E measuring "
            "business quality. Elite: A, B, or B-."
        ),
        "Acc/Dis Rating": (
            "Accumulation/Distribution — shows whether big institutional investors "
            "are buying (accumulating) or selling (distributing). Elite: A, B, or B-."
        ),
        "CAN SLIM": (
            "IBD's 7-factor checklist: Current earnings, Annual growth, New products, "
            "Supply/demand, Leader, Institutional sponsorship, Market direction."
        ),
        "Tier 1 Momentum": (
            "Highest-growth portfolio tier. Composite 95+, RS 90+. "
            "22% trailing stop. Target return: 25-30%."
        ),
        "Tier 2 Quality Growth": (
            "Balanced growth tier. Composite 85+, RS 80+. "
            "18% trailing stop. Target return: 18-22%."
        ),
        "Tier 3 Defensive": (
            "Capital preservation tier. Composite 80+, RS 75+. "
            "12% trailing stop. Target return: 12-15%."
        ),
        "Trailing Stop": (
            "Automatic sell order that rises with the stock price but never falls. "
            "Protects gains while allowing upside."
        ),
        "Stop Tightening": (
            "As a stock gains 10% or 20%, the trailing stop gets tighter "
            "(closer to current price) to lock in more profit."
        ),
        "Sector Rotation": (
            "The pattern of different market sectors taking turns leading. "
            "Detected by the Rotation Detector using 5 independent signals."
        ),
        "Multi-Source Validated": (
            "A stock recommended by 2+ independent sources with total "
            "validation score of 5 or more. Stronger conviction than single-source."
        ),
        "IBD Keep Threshold": (
            "Composite 93+ AND RS 90+. These elite stocks are kept "
            "unless specific overriding factors exist."
        ),
        "Sleep Well Score": (
            "Risk Officer's rating (1-10) of how comfortable you should feel "
            "about the portfolio. Based on stops, diversification, and alignment."
        ),
        "Position Sizing": (
            "How much of the portfolio to allocate to each position. "
            "Stocks: max 5%/4%/3% by tier. ETFs: max 8%/8%/6% by tier."
        ),
    }


def _build_returns_explanation(returns: ReturnsProjectionOutput) -> str:
    """Build plain-language returns explanation from Agent 07 output."""
    exp = returns.expected_return
    spy = next((b for b in returns.benchmark_comparisons if b.symbol == "SPY"), None)
    spy_alpha = spy.expected_alpha_12m if spy else 0
    dd = returns.risk_metrics.max_drawdown_with_stops
    sharpe = returns.risk_metrics.portfolio_sharpe_base
    ci = returns.confidence_intervals.horizon_12m

    parts = [
        f"The Returns Projector analyzed your portfolio under three scenarios "
        f"(bull, base, and bear) weighted by the current {returns.market_regime} regime.",
        f"Expected returns: {exp.expected_3m:.1f}% over 3 months, "
        f"{exp.expected_6m:.1f}% over 6 months, and {exp.expected_12m:.1f}% over 12 months.",
        f"Compared to the S&P 500, the portfolio is projected to generate "
        f"{spy_alpha:+.1f}% alpha (excess return) over 12 months.",
        f"Risk protection: trailing stops limit the maximum drawdown to {dd:.1f}%. "
        f"The Sharpe ratio of {sharpe:.2f} means you earn {sharpe:.2f}% return "
        f"for every 1% of risk taken.",
        f"In the range of outcomes, there is an 80% chance your 12-month return "
        f"falls between {ci.p10:.1f}% and {ci.p90:.1f}%.",
    ]
    return " ".join(parts)


def _build_returns_concept_lesson(returns: ReturnsProjectionOutput) -> ConceptLesson:
    """Build a concept lesson about returns projections."""
    return ConceptLesson(
        concept="Returns Projection & Scenario Analysis",
        simple_explanation=(
            "Portfolio returns are projected across three scenarios — bull (optimistic), "
            "base (most likely), and bear (pessimistic) — then probability-weighted to "
            "produce an expected return. This accounts for uncertainty rather than "
            "predicting a single number."
        ),
        analogy=(
            "Like a weather forecast: instead of saying 'it will be 72F', we say "
            "'there is a 55% chance of warm, 35% normal, and 10% cold' — "
            "the weighted average gives the best estimate."
        ),
        why_it_matters=(
            "Single-point estimates create false confidence. Scenario analysis shows "
            "the range of possible outcomes and how protected you are in each case."
        ),
        example_from_analysis=(
            f"Expected 12m return: {returns.expected_return.expected_12m:.1f}%. "
            f"Bull scenario: {returns.scenarios[0].portfolio_return_12m:.1f}%, "
            f"Bear scenario: {returns.scenarios[2].portfolio_return_12m:.1f}%."
        ),
        framework_reference="Framework v4.0 Section 7.1",
    )


def _returns_glossary_entries() -> dict[str, str]:
    """Return glossary entries for returns-related terms."""
    return {
        "Alpha": (
            "Excess return above a benchmark. Alpha of +4% vs SPY means the portfolio "
            "returned 4% more than the S&P 500."
        ),
        "Sharpe Ratio": (
            "Return per unit of risk: (return - risk-free rate) / volatility. "
            "Higher is better — 1.0+ is good, 2.0+ is excellent."
        ),
        "Maximum Drawdown": (
            "The largest peak-to-trough decline in portfolio value. With trailing stops, "
            "max drawdown is capped at the weighted stop-loss percentages."
        ),
    }
