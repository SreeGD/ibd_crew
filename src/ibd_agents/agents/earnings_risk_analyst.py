"""
Agent 17: Earnings Risk Analyst
Pre-Earnings Risk Specialist — IBD Momentum Framework v4.0

Receives PortfolioOutput + RegimeDetectorOutput + optional HoldingsSummary.
Produces EarningsRiskOutput with:
- Per-position earnings analysis with cushion ratios
- Strategy options with scenario tables (best/base/worst)
- Portfolio-level concentration assessment
- Executive summary

Two execution paths:
1. Deterministic pipeline: run_earnings_risk_pipeline() — no LLM
2. Agentic pipeline: build_earnings_risk_agent() + build_earnings_risk_task()

This agent quantifies risk. It never predicts beat or miss.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Optional

try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    Agent = None  # type: ignore[assignment, misc]
    Task = None  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    from ibd_agents.schemas.analyst_output import AnalystOutput

from ibd_agents.schemas.earnings_risk_output import (
    LOOKFORWARD_DAYS,
    CushionCategory,
    EarningsRisk,
    EarningsRiskOutput,
    EstimateRevision,
    HistoricalEarnings,
    ImpliedVolSignal,
    PortfolioEarningsConcentration,
    PositionEarningsAnalysis,
    StrategyType,
)
from ibd_agents.schemas.portfolio_output import PortfolioOutput
from ibd_agents.schemas.reconciliation_output import HoldingsSummary
from ibd_agents.tools.earnings_data_fetcher import (
    EarningsCalendarTool,
    HistoricalEarningsTool,
    assess_concentration,
    build_strategy_options,
    classify_risk_level,
    compute_cushion_ratio,
    enrich_estimate_revisions_llm,
    fetch_earnings_calendar_mock,
    fetch_earnings_calendar_real,
    fetch_historical_earnings_mock,
    fetch_historical_earnings_real,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI Agent Builder
# ---------------------------------------------------------------------------

def build_earnings_risk_agent() -> "Agent":
    """Create the Earnings Risk Analyst Agent. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError(
            "crewai is required for agentic mode. pip install crewai[tools]"
        )
    return Agent(
        role="Earnings Risk Analyst",
        goal=(
            "Analyze every portfolio position approaching earnings and produce "
            "a specific, risk-quantified pre-earnings strategy for each. Calculate "
            "cushion ratios, retrieve historical earnings reactions, assess estimate "
            "revisions and implied volatility, and present at minimum 2 strategy "
            "options with scenario tables showing best/base/worst case dollar impacts. "
            "Also assess portfolio-level earnings concentration risk. Never predict "
            "whether a company will beat or miss — quantify risk and present strategies."
        ),
        backstory=(
            "You are a pre-earnings risk specialist who has studied thousands of "
            "earnings reactions across growth stocks. You know that earnings season "
            "is when most momentum investors give back gains — not from bad stock "
            "picks but from poor preparation for binary events. The average gap is "
            "not the worst case — plan for tail risk. Profit cushion is the most "
            "important variable. Portfolio concentration of same-week earnings creates "
            "correlated risk. You never say 'hold and hope.' Every position gets "
            "quantified risk/reward. Your principle: know exactly what you can lose "
            "before the number drops."
        ),
        tools=[EarningsCalendarTool(), HistoricalEarningsTool()],
        verbose=True,
        allow_delegation=False,
        max_iter=20,
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# CrewAI Task Builder
# ---------------------------------------------------------------------------

def build_earnings_risk_task(
    agent: "Agent",
    positions_json: str = "",
) -> "Task":
    """Create the Earnings Risk Analysis task. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError(
            "crewai is required for agentic mode. pip install crewai[tools]"
        )
    return Task(
        description=f"""Analyze all portfolio positions for upcoming earnings risk.

For EACH position:
1. Check if earnings are within 21 days
2. Retrieve last 8 quarters of earnings reactions
3. Calculate cushion ratio (current gain / average earnings move)
4. Classify risk: LOW, MODERATE, HIGH, or CRITICAL
5. Present minimum 2 strategy options with scenario tables
6. Select recommended strategy with rationale

Then assess PORTFOLIO-LEVEL earnings concentration:
- Group positions by earnings week
- Flag if > 30% of portfolio reports in same week

CRITICAL RULES:
- Every scenario table must show the math
- NEVER predict beat or miss
- Factor market regime into strategy aggressiveness
- Cushion ratio < 0.5 + weak market = MUST NOT hold full

Portfolio data:
{positions_json}""",
        expected_output=(
            "JSON with analysis_date, market_regime, analyses (list of "
            "PositionEarningsAnalysis), positions_clear, concentration, "
            "and executive_summary. Must match EarningsRiskOutput schema."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Deterministic Pipeline
# ---------------------------------------------------------------------------

def _collect_positions(
    portfolio_output: PortfolioOutput,
    brokerage_holdings: Optional[HoldingsSummary] = None,
    analyst_output: Optional["AnalystOutput"] = None,
) -> list[dict]:
    """
    Build unified position list from portfolio + brokerage holdings.

    Returns list of dicts with keys:
        symbol, shares, current_price, buy_price, gain_loss_pct,
        portfolio_pct, position_value, account, tier, asset_type
    """
    positions: list[dict] = []
    seen_symbols: set[str] = set()

    # Total portfolio value for portfolio_pct computation
    total_value = 0.0
    if brokerage_holdings is not None:
        total_value = brokerage_holdings.total_value
    if total_value <= 0:
        # Estimate from portfolio positions
        total_value = 100_000.0  # Default fallback

    # 1. From portfolio output tiers
    for tier_portfolio in [
        portfolio_output.tier_1,
        portfolio_output.tier_2,
        portfolio_output.tier_3,
    ]:
        for pos in tier_portfolio.positions:
            sym = pos.symbol.upper()
            if sym in seen_symbols:
                continue
            seen_symbols.add(sym)

            # Estimate current price and buy price
            # Use target_pct * total_value / 100 as position value estimate
            position_value = total_value * pos.target_pct / 100
            shares = max(1, int(position_value / 100))  # Rough share count
            current_price = position_value / shares if shares > 0 else 100.0
            buy_price = current_price * 0.95  # Estimate ~5% gain

            # Look for real data from brokerage holdings
            if brokerage_holdings is not None:
                for h in brokerage_holdings.holdings:
                    if h.symbol.upper() == sym:
                        shares = max(1, int(h.shares))
                        current_price = h.market_value / h.shares if h.shares > 0 else current_price
                        if h.cost_basis is not None and h.cost_basis > 0:
                            buy_price = h.cost_basis / h.shares
                        position_value = h.market_value
                        break

            gain_loss_pct = round(
                (current_price - buy_price) / buy_price * 100, 2
            ) if buy_price > 0 else 0.0

            positions.append({
                "symbol": sym,
                "shares": shares,
                "current_price": round(current_price, 2),
                "buy_price": round(buy_price, 2),
                "gain_loss_pct": gain_loss_pct,
                "portfolio_pct": round(position_value / total_value * 100, 1) if total_value > 0 else 0.0,
                "position_value": round(position_value, 2),
                "account": "Model",
                "tier": tier_portfolio.tier,
                "asset_type": pos.asset_type,
            })

    # 2. From brokerage holdings (positions not in model portfolio)
    if brokerage_holdings is not None:
        # Aggregate same-symbol holdings across accounts
        brokerage_agg: dict[str, dict] = {}
        for h in brokerage_holdings.holdings:
            sym = h.symbol.upper()
            if sym in seen_symbols:
                continue
            if sym not in brokerage_agg:
                brokerage_agg[sym] = {
                    "shares": 0.0,
                    "market_value": 0.0,
                    "cost_basis": 0.0,
                    "account": h.account,
                }
            agg = brokerage_agg[sym]
            agg["shares"] += h.shares
            agg["market_value"] += h.market_value
            if h.cost_basis is not None:
                agg["cost_basis"] += h.cost_basis

        for sym, agg in brokerage_agg.items():
            shares = max(1, int(agg["shares"]))
            current_price = agg["market_value"] / agg["shares"] if agg["shares"] > 0 else 100.0
            buy_price = current_price
            if agg["cost_basis"] > 0 and agg["shares"] > 0:
                buy_price = agg["cost_basis"] / agg["shares"]

            gain_loss_pct = round(
                (current_price - buy_price) / buy_price * 100, 2
            ) if buy_price > 0 else 0.0

            positions.append({
                "symbol": sym,
                "shares": shares,
                "current_price": round(current_price, 2),
                "buy_price": round(buy_price, 2),
                "gain_loss_pct": gain_loss_pct,
                "portfolio_pct": round(agg["market_value"] / total_value * 100, 1) if total_value > 0 else 0.0,
                "position_value": round(agg["market_value"], 2),
                "account": agg["account"],
                "tier": 3,
                "asset_type": "stock",
            })
            seen_symbols.add(sym)

    return positions


def run_earnings_risk_pipeline(
    portfolio_output: PortfolioOutput,
    regime_output=None,
    analyst_output: Optional["AnalystOutput"] = None,
    brokerage_holdings: Optional[HoldingsSummary] = None,
    use_real_data: bool = False,
) -> EarningsRiskOutput:
    """
    Run the deterministic Earnings Risk Analyst pipeline.

    Args:
        portfolio_output: Validated PortfolioOutput from Agent 05.
        regime_output: Optional RegimeDetectorOutput from Agent 16.
        analyst_output: Optional AnalystOutput from Agent 02.
        brokerage_holdings: Optional HoldingsSummary from brokerage PDFs.
        use_real_data: If True and yfinance available, fetch real data.

    Returns:
        Validated EarningsRiskOutput.
    """
    logger.info("[Agent 17] Running Earnings Risk Analyst pipeline ...")

    # Step 1: Map regime
    regime = "CONFIRMED_UPTREND"
    if regime_output is not None:
        regime = getattr(regime_output, "regime", None)
        if regime is not None and hasattr(regime, "value"):
            regime = regime.value
        if regime is None:
            regime = "CONFIRMED_UPTREND"
    logger.info(f"[Agent 17] Market regime: {regime}")

    # Step 2: Collect all positions
    positions = _collect_positions(portfolio_output, brokerage_holdings, analyst_output)
    logger.info(f"[Agent 17] Collected {len(positions)} positions")

    # Step 3: Fetch earnings calendar
    symbols = [p["symbol"] for p in positions]
    use_real = use_real_data
    try:
        from ibd_agents.tools.market_data_fetcher import is_available as has_yfinance
        if not has_yfinance():
            use_real = False
    except ImportError:
        use_real = False

    if use_real:
        calendar = fetch_earnings_calendar_real(symbols)
        data_source = "real"
    else:
        calendar = fetch_earnings_calendar_mock(symbols)
        data_source = "mock"
    logger.info(f"[Agent 17] Earnings calendar: {len(calendar)} symbols with dates ({data_source})")

    # Step 4: Split into approaching vs clear
    approaching: list[dict] = []
    positions_clear: list[str] = []

    for pos in positions:
        sym = pos["symbol"]
        if sym in calendar:
            cal_data = calendar[sym]
            days = cal_data["days_until_earnings"]
            if days <= LOOKFORWARD_DAYS:
                pos["earnings_date"] = cal_data["earnings_date"]
                pos["days_until_earnings"] = days
                pos["reporting_time"] = cal_data.get("reporting_time", "UNKNOWN")
                approaching.append(pos)
            else:
                positions_clear.append(sym)
        else:
            positions_clear.append(sym)

    logger.info(
        f"[Agent 17] {len(approaching)} positions with earnings within "
        f"{LOOKFORWARD_DAYS} days, {len(positions_clear)} clear"
    )

    # Step 5: LLM enrichment for estimate revisions (optional)
    estimate_revisions: dict[str, dict] = {}
    if approaching:
        approaching_symbols = [p["symbol"] for p in approaching]
        estimate_revisions = enrich_estimate_revisions_llm(approaching_symbols)
        if estimate_revisions:
            logger.info(f"[Agent 17] LLM estimate revisions for {len(estimate_revisions)} symbols")

    # Step 6: Analyze each approaching position
    # Count positions per week for concentration context
    week_counts: dict[str, int] = {}
    for pos in approaching:
        ed = pos["earnings_date"]
        if isinstance(ed, date):
            from datetime import timedelta
            monday = ed - timedelta(days=ed.weekday())
            week_key = monday.isoformat()
        else:
            week_key = str(ed)[:10]
        week_counts[week_key] = week_counts.get(week_key, 0) + 1

    analyses: list[PositionEarningsAnalysis] = []
    concentration_data: list[dict] = []

    for pos in approaching:
        sym = pos["symbol"]

        # Fetch historical earnings
        if use_real:
            hist_data = fetch_historical_earnings_real(sym)
        else:
            hist_data = None
        if hist_data is None:
            hist_data = fetch_historical_earnings_mock(sym)

        historical = HistoricalEarnings(**hist_data)

        # Compute cushion ratio
        cushion_ratio, cushion_category = compute_cushion_ratio(
            pos["gain_loss_pct"], historical.avg_move_pct,
        )

        # Estimate revision
        est_data = estimate_revisions.get(sym, {})
        revision_str = est_data.get("revision", "NEUTRAL")
        try:
            estimate_revision = EstimateRevision(revision_str)
        except ValueError:
            estimate_revision = EstimateRevision.NEUTRAL
        estimate_detail = est_data.get(
            "detail", "No recent revision data available"
        )

        # Implied vol (deferred to v1.1 — default NORMAL)
        iv_signal = ImpliedVolSignal.NORMAL
        implied_move_pct = None

        # Classify risk
        risk_level, risk_factors = classify_risk_level(
            cushion_ratio, cushion_category,
            historical.beat_rate_pct, pos["portfolio_pct"],
            regime, estimate_revision, iv_signal,
            historical.quarters_analyzed,
        )

        # Compute same-week count for this position
        ed = pos["earnings_date"]
        if isinstance(ed, date):
            from datetime import timedelta
            monday = ed - timedelta(days=ed.weekday())
            same_week = week_counts.get(monday.isoformat(), 1)
        else:
            same_week = 1

        # Build strategy options
        strategies, recommended, rationale = build_strategy_options(
            shares=pos["shares"],
            current_price=pos["current_price"],
            buy_price=pos["buy_price"],
            gain_loss_pct=pos["gain_loss_pct"],
            portfolio_pct=pos["portfolio_pct"],
            historical=historical,
            cushion_ratio=cushion_ratio,
            cushion_category=cushion_category,
            regime=regime,
            estimate_revision=estimate_revision,
            implied_vol_signal=iv_signal,
            implied_move_pct=implied_move_pct,
            risk_level=risk_level,
            same_week_count=same_week,
        )

        # Format earnings date
        ed = pos["earnings_date"]
        if isinstance(ed, date):
            earnings_date_str = ed.isoformat()
        else:
            earnings_date_str = str(ed)[:10]

        analysis = PositionEarningsAnalysis(
            ticker=sym,
            account=pos["account"],
            earnings_date=earnings_date_str,
            days_until_earnings=pos["days_until_earnings"],
            reporting_time=pos["reporting_time"],
            shares=pos["shares"],
            current_price=pos["current_price"],
            buy_price=pos["buy_price"],
            gain_loss_pct=pos["gain_loss_pct"],
            position_value=pos["position_value"],
            portfolio_pct=pos["portfolio_pct"],
            historical=historical,
            cushion_ratio=cushion_ratio,
            cushion_category=cushion_category,
            estimate_revision=estimate_revision,
            estimate_revision_detail=estimate_detail,
            implied_vol_signal=iv_signal,
            implied_move_pct=implied_move_pct,
            risk_level=risk_level,
            risk_factors=risk_factors,
            strategies=strategies,
            recommended_strategy=recommended,
            recommendation_rationale=rationale,
        )
        analyses.append(analysis)

        concentration_data.append({
            "ticker": sym,
            "earnings_date": pos["earnings_date"],
            "portfolio_pct": pos["portfolio_pct"],
        })

    # Sort analyses by earnings date
    analyses.sort(key=lambda a: a.earnings_date)

    # Step 7: Assess concentration
    concentration = assess_concentration(concentration_data)

    # Step 8: Build executive summary
    n_critical = sum(1 for a in analyses if a.risk_level == EarningsRisk.CRITICAL)
    n_high = sum(1 for a in analyses if a.risk_level == EarningsRisk.HIGH)
    n_mod = sum(1 for a in analyses if a.risk_level == EarningsRisk.MODERATE)
    n_low = sum(1 for a in analyses if a.risk_level == EarningsRisk.LOW)

    if analyses:
        summary_parts = [
            f"Earnings Risk Analysis: {len(analyses)} positions have earnings "
            f"within {LOOKFORWARD_DAYS} days in {regime} regime.",
        ]
        if n_critical > 0:
            critical_tickers = [a.ticker for a in analyses if a.risk_level == EarningsRisk.CRITICAL]
            summary_parts.append(
                f"{n_critical} CRITICAL risk: {', '.join(critical_tickers)}."
            )
        if n_high > 0:
            summary_parts.append(f"{n_high} HIGH risk positions require attention.")
        summary_parts.append(
            f"Risk breakdown: {n_critical} critical, {n_high} high, "
            f"{n_mod} moderate, {n_low} low."
        )
        if concentration.concentration_risk in ("HIGH", "CRITICAL"):
            summary_parts.append(
                f"Portfolio concentration risk: {concentration.concentration_risk}."
            )
        summary = " ".join(summary_parts)
    else:
        summary = (
            f"No positions have earnings within the {LOOKFORWARD_DAYS}-day "
            f"lookforward window. All {len(positions_clear)} positions are clear "
            f"of near-term earnings binary event risk."
        )

    # Step 9: Build and return output
    output = EarningsRiskOutput(
        analysis_date=date.today().isoformat(),
        market_regime=regime,
        lookforward_days=LOOKFORWARD_DAYS,
        analyses=analyses,
        positions_clear=sorted(positions_clear),
        concentration=concentration,
        executive_summary=summary,
        data_source=data_source,
    )

    logger.info(
        f"[Agent 17] Done — {len(analyses)} analyses, "
        f"{n_critical} critical, {n_high} high, {n_mod} moderate, {n_low} low, "
        f"concentration={concentration.concentration_risk}"
    )

    return output
