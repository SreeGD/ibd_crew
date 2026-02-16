"""
Agent 08: Portfolio Reconciler
Portfolio Transition Specialist — IBD Momentum Framework v4.0

Receives PortfolioOutput (recommended) + builds mock current holdings.
Produces ReconciliationOutput with:
- Current vs recommended diff
- KEEP/SELL/BUY/ADD/TRIM actions
- Money flow calculation
- 4-week implementation plan
- Keep verification

This agent reconciles portfolios. It never picks stocks or makes allocations.
"""

from __future__ import annotations

import logging
from datetime import date

try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    Agent = None  # type: ignore
    Task = None  # type: ignore

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ibd_agents.schemas.value_investor_output import ValueInvestorOutput
    from ibd_agents.schemas.pattern_output import PortfolioPatternOutput

from ibd_agents.schemas.analyst_output import AnalystOutput
from ibd_agents.schemas.portfolio_output import PortfolioOutput
from ibd_agents.schemas.reconciliation_output import ReconciliationOutput
from ibd_agents.schemas.returns_projection_output import ReturnsProjectionOutput
from ibd_agents.schemas.rotation_output import RotationDetectionOutput
from ibd_agents.schemas.strategy_output import SectorStrategyOutput
from ibd_agents.schemas.risk_output import RiskAssessment
from ibd_agents.tools.portfolio_reader import (
    PortfolioReaderTool,
    build_mock_current_holdings,
    read_brokerage_pdfs,
)
from ibd_agents.tools.position_differ import (
    PositionDifferTool,
    build_implementation_plan,
    compute_money_flow,
    compute_transformation_metrics,
    diff_positions,
    verify_keeps,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI Agent Builder
# ---------------------------------------------------------------------------

def build_reconciler_agent() -> "Agent":
    """Create the Portfolio Reconciler Agent. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Agent(
        role="Portfolio Transition Specialist",
        goal=(
            "Compare every position in the current portfolio against the "
            "recommended portfolio. Generate KEEP/SELL/BUY/ADD/TRIM actions "
            "with a 4-week implementation timeline. Calculate money flow "
            "and verify all 14 keeps are present."
        ),
        backstory=(
            "You are a meticulous portfolio reconciler who compares current "
            "holdings against recommended targets. You generate precise "
            "action lists, calculate money flows, and build implementation "
            "timelines. You never make allocation decisions — you only "
            "reconcile and plan transitions."
        ),
        tools=[PortfolioReaderTool(), PositionDifferTool()],
        verbose=True,
        allow_delegation=False,
        max_iter=10,
        temperature=0.3,
    )


def build_reconciler_task(
    agent: "Agent",
    portfolio_json: str = "",
) -> "Task":
    """Create the Reconciliation task. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Task(
        description=f"""Reconcile current holdings vs recommended portfolio.

STEPS:
1. Read current holdings from brokerage files
2. Diff every position: KEEP, SELL, BUY, ADD, TRIM
3. Calculate money flow (sources vs uses)
4. Build 4-week implementation timeline
5. Verify all 14 keeps present
6. Compute transformation metrics

IMPLEMENTATION PHASES:
- Week 1: LIQUIDATION — sell non-recommended positions
- Week 2: T1 MOMENTUM — establish T1 positions
- Week 3: T2 QUALITY GROWTH — establish T2 positions
- Week 4: T3 DEFENSIVE — establish T3 positions

Portfolio data:
{portfolio_json}
""",
        expected_output=(
            "JSON with current_holdings, actions, money_flow, "
            "implementation_plan, keep_verification, transformation_metrics, "
            "analysis_date, summary."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Deterministic Pipeline
# ---------------------------------------------------------------------------

def run_reconciler_pipeline(
    portfolio_output: PortfolioOutput,
    analyst_output: AnalystOutput,
    returns_output: Optional[ReturnsProjectionOutput] = None,
    rotation_output: Optional["RotationDetectionOutput"] = None,
    strategy_output: Optional["SectorStrategyOutput"] = None,
    risk_output: Optional["RiskAssessment"] = None,
    portfolios_dir: Optional[str] = None,
    value_output: Optional["ValueInvestorOutput"] = None,
    pattern_output: Optional["PortfolioPatternOutput"] = None,
) -> ReconciliationOutput:
    """
    Run the deterministic reconciliation pipeline without LLM.

    Args:
        portfolio_output: Validated PortfolioOutput from Agent 05
        analyst_output: Validated AnalystOutput from Agent 02
        returns_output: Optional ReturnsProjectionOutput from Agent 07
        portfolios_dir: Path to brokerage PDFs. If provided and PDFs exist,
                        parses real holdings. Otherwise uses mock data.

    Returns:
        Validated ReconciliationOutput
    """
    logger.info("[Agent 08] Running Reconciliation pipeline ...")

    # Step 1: Read current holdings (real PDFs if provided, else mock)
    current_holdings = None
    if portfolios_dir:
        from pathlib import Path
        pdir = Path(portfolios_dir)
        if pdir.exists() and list(pdir.glob("*.pdf")):
            try:
                current_holdings = read_brokerage_pdfs(portfolios_dir)
                logger.info(
                    f"[Agent 08] Real holdings from PDFs: {len(current_holdings.holdings)} positions, "
                    f"${current_holdings.total_value:,.0f} across {current_holdings.account_count} accounts"
                )
            except Exception as e:
                logger.warning(f"[Agent 08] PDF parsing failed ({e}), falling back to mock")

    if current_holdings is None:
        current_holdings = build_mock_current_holdings(analyst_output.rated_stocks)

    logger.info(
        f"[Agent 08] Current holdings: {len(current_holdings.holdings)} positions, "
        f"${current_holdings.total_value:,.0f}"
    )

    # Step 2: Diff current vs recommended
    actions = diff_positions(current_holdings, portfolio_output)
    logger.info(
        f"[Agent 08] Actions: "
        f"{sum(1 for a in actions if a.action_type == 'SELL')} sells, "
        f"{sum(1 for a in actions if a.action_type == 'BUY')} buys, "
        f"{sum(1 for a in actions if a.action_type == 'ADD')} adds, "
        f"{sum(1 for a in actions if a.action_type == 'TRIM')} trims, "
        f"{sum(1 for a in actions if a.action_type == 'KEEP')} keeps"
    )

    # --- Step 2.5: LLM rationale enrichment ---
    rationale_source = None
    try:
        from ibd_agents.tools.reconciliation_reasoning import enrich_rationale_llm
        context = {
            "rotation_verdict": rotation_output.verdict.value if rotation_output else "unknown",
            "rotation_type": rotation_output.rotation_type.value if rotation_output else "unknown",
            "risk_status": risk_output.overall_status if risk_output else "unknown",
            "regime": strategy_output.regime_adjustment if strategy_output else "unknown",
        }
        action_inputs = [
            {
                "symbol": a.symbol,
                "action_type": a.action_type,
                "cap_size": a.cap_size,
                "current_pct": a.current_pct,
                "target_pct": a.target_pct,
                "priority": a.priority,
            }
            for a in actions
        ]
        enhanced = enrich_rationale_llm(action_inputs, context)
        if enhanced:
            from ibd_agents.schemas.reconciliation_output import PositionAction
            for idx, action in enumerate(actions):
                if action.symbol in enhanced:
                    actions[idx] = PositionAction(
                        symbol=action.symbol,
                        action_type=action.action_type,
                        cap_size=action.cap_size,
                        current_pct=action.current_pct,
                        target_pct=action.target_pct,
                        dollar_change=action.dollar_change,
                        priority=action.priority,
                        week=action.week,
                        rationale=enhanced[action.symbol],
                    )
            rationale_source = "llm"
            logger.info(f"LLM rationale enrichment: {len(enhanced)}/{len(actions)} actions enhanced")
        else:
            rationale_source = "deterministic"
    except Exception as e:
        logger.warning(f"LLM rationale enrichment failed: {e}")
        rationale_source = "deterministic"

    # --- Step 2.7: Review sells against value/pattern scores ---
    actions = _review_sells_against_value_pattern(actions, value_output, pattern_output)

    # --- Step 2.8: Enrich actions with Sharpe/Alpha from analyst data ---
    actions = _enrich_sharpe_alpha(actions, analyst_output)

    # Step 3: Compute money flow
    money_flow = compute_money_flow(actions, current_holdings.total_value)
    logger.info(
        f"[Agent 08] Money flow: sells=${money_flow.sell_proceeds:,.0f}, "
        f"buys=${money_flow.buy_cost:,.0f}, net=${money_flow.net_cash_change:,.0f}"
    )

    # Step 4: Build implementation plan
    impl_plan = build_implementation_plan(actions)

    # Step 5: Verify keeps
    keep_verification = verify_keeps(current_holdings)

    # Step 6: Compute transformation metrics
    metrics = compute_transformation_metrics(
        current_holdings, portfolio_output, actions
    )

    # Step 7: ETF implementation notes
    etf_notes = _build_etf_implementation_notes(actions)

    # Step 8: Build summary
    summary = _build_summary(actions, money_flow, keep_verification, metrics)

    output = ReconciliationOutput(
        current_holdings=current_holdings,
        actions=actions,
        money_flow=money_flow,
        implementation_plan=impl_plan,
        keep_verification=keep_verification,
        transformation_metrics=metrics,
        etf_implementation_notes=etf_notes,
        analysis_date=date.today().isoformat(),
        summary=summary,
        rationale_source=rationale_source,
    )

    logger.info(
        f"[Agent 08] Done — {len(actions)} actions, "
        f"{metrics.turnover_pct:.0f}% turnover, "
        f"{len(keep_verification.keeps_in_current)}/14 keeps found"
    )

    return output


# ---------------------------------------------------------------------------
# String Builder
# ---------------------------------------------------------------------------

def _build_summary(
    actions, money_flow, keep_verification, metrics
) -> str:
    """Build summary string (>= 50 chars)."""
    sells = sum(1 for a in actions if a.action_type == "SELL")
    buys = sum(1 for a in actions if a.action_type == "BUY")
    keeps_found = len(keep_verification.keeps_in_current)

    parts = [
        f"Reconciliation: {len(actions)} actions across 4-week plan.",
        f"{sells} positions to liquidate, {buys} new positions to establish.",
        f"{keeps_found}/14 keeps verified in current holdings.",
        f"Turnover: {metrics.turnover_pct:.0f}%.",
    ]
    summary = " ".join(parts)
    if len(summary) < 50:
        summary += " Portfolio transition planned per Framework v4.0."
    return summary


def _build_etf_implementation_notes(
    actions: list,
) -> list[str]:
    """Build ETF-specific trading sequence and execution guidance."""
    etf_actions = [a for a in actions if _is_etf_symbol(a.symbol)]
    if not etf_actions:
        return []

    notes: list[str] = []
    notes.append(
        "Execute ETF orders before stock orders — ETFs have higher liquidity "
        "and tighter spreads, providing faster execution."
    )

    # Check for thematic ETFs (non-broad market)
    broad_market = {"QQQ", "SPY", "VOO", "VTI", "IWM", "DIA", "VGT", "VUG", "VTV"}
    theme_etf_actions = [a for a in etf_actions if a.symbol not in broad_market]
    if theme_etf_actions:
        notes.append(
            "Consider limit orders for thematic ETFs with lower volume — "
            "market orders on less-liquid ETFs can result in wider spreads."
        )

    etf_buys = [a for a in etf_actions if a.action_type == "BUY"]
    etf_sells = [a for a in etf_actions if a.action_type == "SELL"]
    if len(etf_buys) + len(etf_sells) >= 4:
        notes.append(
            "Stagger ETF orders across implementation weeks to minimize market impact "
            "and allow for dollar-cost averaging on entries."
        )

    if etf_sells:
        notes.append(
            f"Liquidate {len(etf_sells)} ETF position(s) in Week 1 to free capital "
            f"for new ETF and stock purchases in subsequent weeks."
        )

    return notes


def _is_etf_symbol(symbol: str) -> bool:
    """Heuristic: check if symbol is likely an ETF (3+ chars, no single-letter)."""
    from ibd_agents.schemas.portfolio_output import KEEP_METADATA
    meta = KEEP_METADATA.get(symbol)
    if meta and meta.get("asset_type") == "etf":
        return True
    # Common ETF patterns: 3-4 char symbols ending in specific patterns
    etf_symbols = {
        "QQQ", "SPY", "VOO", "VTI", "IWM", "DIA", "VGT", "VUG", "VTV",
        "EWY", "SLV", "XLK", "XLV", "XLF", "XLE", "XLI", "XLB", "XLU",
        "XLP", "XLY", "XLRE", "SMH", "SOXX", "ARKK", "ARKG", "ARKW",
    }
    return symbol in etf_symbols


def _review_sells_against_value_pattern(
    actions: list,
    value_output: Optional["ValueInvestorOutput"],
    pattern_output: Optional["PortfolioPatternOutput"],
) -> list:
    """
    Enrich SELL action rationales with value/pattern context.

    For each SELL action, checks if the symbol has a notable value score
    or pattern score and appends a REVIEW annotation to the rationale.
    Does NOT change the SELL decision — annotation only.
    """
    if value_output is None and pattern_output is None:
        return actions

    from ibd_agents.schemas.reconciliation_output import PositionAction

    # Build lookups
    value_map: dict[str, dict] = {}
    if value_output is not None:
        for vs in value_output.value_stocks:
            if vs.value_category != "Not Value":
                value_map[vs.symbol] = {
                    "category": vs.value_category,
                    "score": vs.value_score,
                    "rank": vs.value_rank,
                }

    pattern_map: dict[str, dict] = {}
    if pattern_output is not None:
        for sa in pattern_output.stock_analyses:
            if sa.enhanced_score >= 85:
                pattern_map[sa.symbol] = {
                    "score": sa.enhanced_score,
                    "label": sa.enhanced_rating_label,
                }

    reviewed_count = 0
    for idx, action in enumerate(actions):
        if action.action_type != "SELL":
            continue

        annotations: list[str] = []
        vinfo = value_map.get(action.symbol)
        if vinfo:
            annotations.append(
                f"Deep Value ({vinfo['category']}, score={vinfo['score']:.0f}, rank #{vinfo['rank']})"
            )

        pinfo = pattern_map.get(action.symbol)
        if pinfo:
            annotations.append(
                f"Pattern {pinfo['label']} (score={pinfo['score']}/150)"
            )

        if annotations:
            review_note = " | REVIEW: " + "; ".join(annotations)
            actions[idx] = PositionAction(
                symbol=action.symbol,
                action_type=action.action_type,
                cap_size=action.cap_size,
                current_pct=action.current_pct,
                target_pct=action.target_pct,
                dollar_change=action.dollar_change,
                priority=action.priority,
                week=action.week,
                rationale=action.rationale + review_note,
            )
            reviewed_count += 1

    if reviewed_count > 0:
        logger.info(
            f"[Agent 08] Value/pattern sell review: {reviewed_count} SELL actions annotated"
        )

    return actions


def _enrich_sharpe_alpha(
    actions: list,
    analyst_output: AnalystOutput,
) -> list:
    """
    Enrich actions with Sharpe ratio and Alpha from analyst output.

    Looks up sharpe_ratio and alpha_pct from RatedStock data for each action.
    """
    from ibd_agents.schemas.reconciliation_output import PositionAction

    # Build lookup: symbol → (sharpe_ratio, alpha_pct)
    metrics_map: dict[str, dict] = {}
    for rs in analyst_output.rated_stocks:
        metrics_map[rs.symbol] = {
            "sharpe": rs.sharpe_ratio,
            "alpha": rs.alpha_pct,
        }

    enriched_count = 0
    for idx, action in enumerate(actions):
        info = metrics_map.get(action.symbol)
        if info and (info["sharpe"] is not None or info["alpha"] is not None):
            actions[idx] = PositionAction(
                symbol=action.symbol,
                action_type=action.action_type,
                cap_size=action.cap_size,
                current_pct=action.current_pct,
                target_pct=action.target_pct,
                dollar_change=action.dollar_change,
                priority=action.priority,
                week=action.week,
                rationale=action.rationale,
                sharpe_ratio=info["sharpe"],
                alpha_pct=info["alpha"],
            )
            enriched_count += 1

    logger.info(
        f"[Agent 08] Sharpe/Alpha enrichment: {enriched_count}/{len(actions)} actions enriched"
    )
    return actions
