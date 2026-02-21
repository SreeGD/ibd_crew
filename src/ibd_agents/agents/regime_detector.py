"""
Agent 16: Regime Detector
Market Regime Classification Specialist — IBD Momentum Framework v4.0

Classifies market into exactly one of 5 regimes:
  CONFIRMED_UPTREND, UPTREND_UNDER_PRESSURE, FOLLOW_THROUGH_DAY,
  RALLY_ATTEMPT, CORRECTION

Uses 5 data sources: distribution days, market breadth, leading stock
health, sector rankings, and index price history (for FTD detection).

Two execution paths:
1. Deterministic pipeline: run_regime_detector_pipeline() — no LLM
2. Agentic pipeline: build_regime_detector_agent() + build_regime_detector_task()

This agent classifies the CURRENT regime based on CURRENT evidence.
It never forecasts or predicts when corrections will end.
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

from ibd_agents.schemas.regime_detector_output import (
    MarketRegime5,
    PreviousRegime,
    RegimeDetectorOutput,
)
from ibd_agents.tools.regime_classifier import classify_regime
from ibd_agents.tools.regime_data_fetcher import (
    DistributionDaysTool,
    IndexPriceHistoryTool,
    LeadingStocksHealthTool,
    MarketBreadthTool,
    SectorRankingsTool,
    get_distribution_days_mock,
    get_distribution_days_real,
    get_index_price_history_mock,
    get_index_price_history_real,
    get_leading_stocks_health_mock,
    get_leading_stocks_health_real,
    get_market_breadth_mock,
    get_market_breadth_real,
    get_sector_rankings_mock,
    get_sector_rankings_real,
    is_available as has_yfinance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI Agent Builder
# ---------------------------------------------------------------------------

def build_regime_detector_agent() -> "Agent":
    """Create the Regime Detector Agent with 5 tools. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError(
            "crewai is required for agentic mode. pip install crewai[tools]"
        )
    return Agent(
        role="Regime Detector",
        goal=(
            "Classify the current market regime into exactly one of five states: "
            "CONFIRMED_UPTREND, UPTREND_UNDER_PRESSURE, RALLY_ATTEMPT, "
            "FOLLOW_THROUGH_DAY, or CORRECTION. Synthesize distribution day counts, "
            "market breadth, leading stock health, sector rotation, and follow-through "
            "day analysis to produce an evidence-backed classification with exposure "
            "recommendation. Default to the more conservative classification when "
            "signals conflict. Never classify as CONFIRMED_UPTREND without a valid "
            "Follow-Through Day."
        ),
        backstory=(
            "You are a market regime analyst who has studied every major cycle since 1953. "
            "You know that three out of four stocks follow the general market direction. "
            "Your job is to answer one question: is the environment favorable for buying, "
            "or should we be defensive? You are neither permanently bullish nor bearish — "
            "you follow the evidence. You know the hardest calls happen at transitions: "
            "the first days of a correction feel like pullbacks, and the first "
            "follow-through day after a correction feels like a trap. You handle these "
            "by weighing multiple independent signals. You resist calling bottoms or tops "
            "prematurely. When in doubt, you are defensive. The cost of missing the first "
            "few days of a rally is small. The cost of being fully invested at the start "
            "of a correction is enormous."
        ),
        tools=[
            DistributionDaysTool(),
            MarketBreadthTool(),
            LeadingStocksHealthTool(),
            SectorRankingsTool(),
            IndexPriceHistoryTool(),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=15,
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# CrewAI Task Builder
# ---------------------------------------------------------------------------

def build_regime_detector_task(
    agent: "Agent",
    input_json: str = "",
) -> "Task":
    """Create the Regime Detection task. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError(
            "crewai is required for agentic mode. pip install crewai[tools]"
        )
    return Task(
        description=f"""Classify the current market regime.

Analyze ALL of the following:
1. Distribution day count on S&P 500 and Nasdaq (25-day trailing)
2. Market breadth: % above 200-day/50-day MA, new highs vs lows, A/D line
3. Leading stock health: RS >= 90 stocks above 50-day MA
4. Sector rotation: growth vs defensive leadership
5. Follow-through day analysis (only if in CORRECTION or RALLY_ATTEMPT)

HARD RULES:
- 5+ distribution days on either index = CANNOT be CONFIRMED_UPTREND
- 6+ distribution days = CORRECTION regardless of other signals
- No Follow-Through Day after correction = CANNOT be CONFIRMED_UPTREND
- Do NOT flip regime on a single day's action
- When signals conflict, default to more conservative classification

Provide: regime, confidence, exposure recommendation, all indicator
assessments, transition conditions, and executive summary.

Input context:
{input_json}""",
        expected_output=(
            "Complete regime classification with evidence from all indicators, "
            "exposure recommendation, and specific transition conditions. "
            "Must match RegimeDetectorOutput schema."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Deterministic Pipeline
# ---------------------------------------------------------------------------

def run_regime_detector_pipeline(
    analyst_output: Optional["AnalystOutput"] = None,
    previous_regime: str = "UNKNOWN",
    previous_exposure: Optional[int] = None,
    correction_low_date: Optional[date] = None,
    correction_low_sp500: Optional[float] = None,
    correction_low_nasdaq: Optional[float] = None,
    use_real_data: bool = False,
    scenario: str = "healthy_uptrend",
) -> RegimeDetectorOutput:
    """
    Run the deterministic Regime Detector pipeline without LLM.

    Args:
        analyst_output: Optional AnalystOutput from Agent 02 (for RS ratings).
        previous_regime: Previous regime classification string.
        previous_exposure: Previous exposure recommendation (0-100).
        correction_low_date: Date of most recent correction low.
        correction_low_sp500: S&P 500 close at correction low.
        correction_low_nasdaq: Nasdaq close at correction low.
        use_real_data: If True and yfinance available, fetch real data.
        scenario: Mock scenario name (used when use_real_data=False).

    Returns:
        Validated RegimeDetectorOutput.
    """
    logger.info("[Agent 16] Running Regime Detector pipeline ...")
    logger.info(f"[Agent 16] Previous regime: {previous_regime}, scenario: {scenario}")

    # Validate previous regime
    try:
        prev = PreviousRegime(previous_regime)
    except ValueError:
        logger.warning(f"[Agent 16] Unknown previous regime '{previous_regime}', defaulting to UNKNOWN")
        prev = PreviousRegime.UNKNOWN

    # Determine data source
    use_real = use_real_data and has_yfinance()
    if use_real_data and not has_yfinance():
        logger.warning("[Agent 16] use_real_data=True but yfinance not available — falling back to mock")
    data_source = "real" if use_real else "mock"
    logger.info(f"[Agent 16] Data source: {data_source}")

    # Step 1: Fetch distribution days
    logger.info("[Agent 16] Step 1: Fetching distribution days ...")
    dist_data = get_distribution_days_real() if use_real else get_distribution_days_mock(scenario)

    # Step 2: Fetch market breadth
    logger.info("[Agent 16] Step 2: Fetching market breadth ...")
    breadth_data = get_market_breadth_real() if use_real else get_market_breadth_mock(scenario)

    # Step 3: Fetch leading stocks health
    logger.info("[Agent 16] Step 3: Fetching leading stocks health ...")
    leader_data = (get_leading_stocks_health_real(analyst_output) if use_real
                   else get_leading_stocks_health_mock(scenario))

    # Step 4: Fetch sector rankings
    logger.info("[Agent 16] Step 4: Fetching sector rankings ...")
    sector_data = (get_sector_rankings_real(analyst_output) if use_real
                   else get_sector_rankings_mock(scenario))

    # Step 5: Fetch index price history (only for FTD detection)
    index_sp500 = None
    index_nasdaq = None
    needs_ftd = prev in (
        PreviousRegime.CORRECTION,
        PreviousRegime.RALLY_ATTEMPT,
        PreviousRegime.FOLLOW_THROUGH_DAY,
        PreviousRegime.UNKNOWN,
    )
    if needs_ftd:
        logger.info("[Agent 16] Step 5: Fetching index price history for FTD detection ...")
        if use_real:
            index_sp500 = get_index_price_history_real("SP500")
            index_nasdaq = get_index_price_history_real("NASDAQ")
        else:
            index_sp500 = get_index_price_history_mock("SP500", scenario)
            index_nasdaq = get_index_price_history_mock("NASDAQ", scenario)
    else:
        logger.info("[Agent 16] Step 5: Skipping FTD check (not in correction/rally state)")

    # Step 6: Classify regime
    logger.info("[Agent 16] Step 6: Classifying regime ...")
    output = classify_regime(
        dist_data=dist_data,
        breadth_data=breadth_data,
        leader_data=leader_data,
        sector_data=sector_data,
        index_data_sp500=index_sp500,
        index_data_nasdaq=index_nasdaq,
        previous_regime=prev.value,
        correction_low_date=correction_low_date,
        correction_low_sp500=correction_low_sp500,
        correction_low_nasdaq=correction_low_nasdaq,
    )

    logger.info(
        f"[Agent 16] Done — regime={output.regime.value}, "
        f"confidence={output.confidence.value}, "
        f"health={output.market_health_score}/10, "
        f"exposure={output.exposure_recommendation}%"
    )

    return output
