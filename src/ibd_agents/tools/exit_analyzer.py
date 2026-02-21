"""
Exit Strategist Tool: Exit Analysis & Sell Rule Evaluation
IBD Momentum Investment Framework v4.0

Pure functions for:
- 8 IBD sell rule evaluators (one function per rule)
- Urgency classifier
- Action classifier
- Evidence chain builder
- Portfolio impact calculator
- Health score computation

No LLM, no file I/O.
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional, Tuple

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool

from pydantic import BaseModel, Field

from ibd_agents.schemas.exit_strategy_output import (
    CLIMAX_SURGE_MIN_PCT,
    EARNINGS_MIN_CUSHION_PCT,
    EARNINGS_PROXIMITY_DAYS,
    FAST_GAIN_MAX_DAYS,
    HARD_BACKSTOP_LOSS_PCT,
    HEALTH_SCORE_RANGE,
    REGIME_STOP_THRESHOLDS,
    RS_DROP_THRESHOLD,
    RS_MIN_THRESHOLD,
    RULE_ID_MAP,
    SECTOR_DIST_DAYS_THRESHOLD,
    URGENCY_ORDER,
    EvidenceLink,
    ExitActionType,
    ExitMarketRegime,
    PortfolioImpact,
    PositionSignal,
    SellRule,
    SellRuleResult,
    SellType,
    Urgency,
)
from ibd_agents.tools.position_monitor import MarketHealthData, PositionMarketData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 8 Sell Rule Check Functions
# ---------------------------------------------------------------------------

def check_stop_loss_7_8(
    data: PositionMarketData,
    regime: ExitMarketRegime,
) -> SellRuleResult:
    """
    MUST-M2: 7-8% stop loss rule. Absolute and non-negotiable.
    Also enforces MUST_NOT-S1 hard backstop at 10%.
    """
    threshold = -7.0  # Standard 7% stop loss
    triggered = data.gain_loss_pct <= threshold

    detail: str
    if data.gain_loss_pct <= -HARD_BACKSTOP_LOSS_PCT:
        detail = (
            f"{data.symbol} down {abs(data.gain_loss_pct):.1f}% from buy price "
            f"${data.buy_price:.2f} — HARD BACKSTOP breached (>{HARD_BACKSTOP_LOSS_PCT}% loss)"
        )
        triggered = True
    elif triggered:
        detail = (
            f"{data.symbol} down {abs(data.gain_loss_pct):.1f}% from buy price "
            f"${data.buy_price:.2f} — breached 7-8% stop loss threshold"
        )
    else:
        detail = (
            f"{data.symbol} at {data.gain_loss_pct:+.1f}% from buy price "
            f"${data.buy_price:.2f} — within stop loss range"
        )

    return SellRuleResult(
        rule=SellRule.STOP_LOSS_7_8,
        triggered=triggered,
        value=data.gain_loss_pct,
        threshold=threshold,
        detail=detail,
    )


def check_climax_top(
    data: PositionMarketData,
    regime: ExitMarketRegime,
) -> SellRuleResult:
    """MUST-M3: Climax top — 25-50% surge in 1-3 weeks on heavy volume."""
    triggered = (
        data.price_surge_pct_3w >= CLIMAX_SURGE_MIN_PCT
        and data.price_surge_volume_ratio >= 2.0
        and data.gain_loss_pct > 0
    )

    if triggered:
        detail = (
            f"{data.symbol} surged {data.price_surge_pct_3w:.1f}% in 3 weeks "
            f"on {data.price_surge_volume_ratio:.1f}x average volume — classic climax top"
        )
    else:
        detail = (
            f"{data.symbol} 3-week surge {data.price_surge_pct_3w:.1f}% with "
            f"{data.price_surge_volume_ratio:.1f}x volume — no climax top detected"
        )

    return SellRuleResult(
        rule=SellRule.CLIMAX_TOP,
        triggered=triggered,
        value=data.price_surge_pct_3w,
        threshold=CLIMAX_SURGE_MIN_PCT,
        detail=detail,
    )


def check_below_50_day_ma(
    data: PositionMarketData,
    regime: ExitMarketRegime,
) -> SellRuleResult:
    """MUST-M4: Closed below 50-day MA on heavy volume."""
    triggered = data.pct_from_50ma < 0 and data.volume_ratio >= 1.5

    if triggered:
        detail = (
            f"{data.symbol} closed ${data.current_price:.2f}, "
            f"{abs(data.pct_from_50ma):.1f}% below 50-day MA (${data.ma_50:.2f}) "
            f"on {data.volume_ratio:.1f}x average volume"
        )
    else:
        if data.pct_from_50ma >= 0:
            detail = (
                f"{data.symbol} at ${data.current_price:.2f}, "
                f"{data.pct_from_50ma:.1f}% above 50-day MA (${data.ma_50:.2f})"
            )
        else:
            detail = (
                f"{data.symbol} below 50-day MA but volume ratio "
                f"{data.volume_ratio:.1f}x insufficient (need ≥1.5x)"
            )

    return SellRuleResult(
        rule=SellRule.BELOW_50_DAY_MA,
        triggered=triggered,
        value=data.pct_from_50ma,
        threshold=0.0,
        detail=detail,
    )


def check_rs_deterioration(
    data: PositionMarketData,
    regime: ExitMarketRegime,
) -> SellRuleResult:
    """MUST-M5: RS Rating below 70 or dropped 15+ points from peak."""
    rs_drop = data.rs_rating_peak_since_buy - data.rs_rating_current
    below_min = data.rs_rating_current < RS_MIN_THRESHOLD
    large_drop = rs_drop >= RS_DROP_THRESHOLD

    triggered = below_min or large_drop

    if below_min and large_drop:
        detail = (
            f"{data.symbol} RS Rating {data.rs_rating_current} below {RS_MIN_THRESHOLD} "
            f"AND dropped {rs_drop} points from peak {data.rs_rating_peak_since_buy}"
        )
    elif below_min:
        detail = (
            f"{data.symbol} RS Rating {data.rs_rating_current} "
            f"below minimum threshold {RS_MIN_THRESHOLD}"
        )
    elif large_drop:
        detail = (
            f"{data.symbol} RS Rating dropped {rs_drop} points "
            f"from peak {data.rs_rating_peak_since_buy} to {data.rs_rating_current}"
        )
    else:
        detail = (
            f"{data.symbol} RS Rating {data.rs_rating_current} "
            f"(peak {data.rs_rating_peak_since_buy}) — healthy"
        )

    return SellRuleResult(
        rule=SellRule.RS_DETERIORATION,
        triggered=triggered,
        value=float(rs_drop),
        threshold=float(RS_DROP_THRESHOLD),
        detail=detail,
    )


def check_profit_target_20_25(
    data: PositionMarketData,
    regime: ExitMarketRegime,
) -> SellRuleResult:
    """
    MUST-M6: Profit-taking at 20-25%.
    Exception: 8-week hold rule — if 20%+ in < 3 weeks, NOT triggered.
    """
    at_profit_target = data.gain_loss_pct >= 20.0
    is_fast_gainer = data.days_held <= FAST_GAIN_MAX_DAYS

    # 8-week hold exception: fast gainer NOT triggered for profit-taking
    triggered = at_profit_target and not is_fast_gainer

    if at_profit_target and is_fast_gainer:
        detail = (
            f"{data.symbol} up {data.gain_loss_pct:.1f}% in {data.days_held} days — "
            f"exceptional strength, 8-week hold rule applies (hold from breakout)"
        )
    elif triggered:
        detail = (
            f"{data.symbol} up {data.gain_loss_pct:.1f}% over {data.days_held} days — "
            f"reached 20-25% profit target, consider taking partial profits"
        )
    else:
        detail = (
            f"{data.symbol} at {data.gain_loss_pct:+.1f}% gain — "
            f"below 20% profit target threshold"
        )

    return SellRuleResult(
        rule=SellRule.PROFIT_TARGET_20_25,
        triggered=triggered,
        value=data.gain_loss_pct,
        threshold=20.0,
        detail=detail,
    )


def check_sector_distribution(
    data: PositionMarketData,
    regime: ExitMarketRegime,
    market_health: Optional[MarketHealthData] = None,
) -> SellRuleResult:
    """MUST-M8: 4+ distribution days in stock's sector in 3 weeks."""
    triggered = data.sector_distribution_days_3w >= SECTOR_DIST_DAYS_THRESHOLD

    if triggered:
        detail = (
            f"{data.symbol} sector ({data.sector}) has "
            f"{data.sector_distribution_days_3w} distribution days in 3 weeks "
            f"(threshold: {SECTOR_DIST_DAYS_THRESHOLD}) — elevated risk"
        )
    else:
        detail = (
            f"{data.symbol} sector ({data.sector}) has "
            f"{data.sector_distribution_days_3w} distribution days in 3 weeks — normal"
        )

    return SellRuleResult(
        rule=SellRule.SECTOR_DISTRIBUTION,
        triggered=triggered,
        value=float(data.sector_distribution_days_3w),
        threshold=float(SECTOR_DIST_DAYS_THRESHOLD),
        detail=detail,
    )


def check_regime_tightened_stop(
    data: PositionMarketData,
    regime: ExitMarketRegime,
) -> SellRuleResult:
    """
    MUST-M7: Tightened stops in non-uptrend regimes.
    Only triggers if standard 7-8% stop has NOT triggered.
    """
    # In confirmed uptrend, standard stops apply — this rule doesn't trigger
    if regime == ExitMarketRegime.CONFIRMED_UPTREND:
        return SellRuleResult(
            rule=SellRule.REGIME_TIGHTENED_STOP,
            triggered=False,
            value=data.gain_loss_pct,
            threshold=None,
            detail=(
                f"{data.symbol} in CONFIRMED_UPTREND — standard stops apply, "
                f"no regime tightening needed"
            ),
        )

    low, high = REGIME_STOP_THRESHOLDS[regime.value]
    # Use the low end (tighter) for trigger
    threshold = -low
    triggered = data.gain_loss_pct <= threshold

    # Don't double-count if standard stop already caught it
    if data.gain_loss_pct <= -7.0:
        triggered = False

    if triggered:
        detail = (
            f"{data.symbol} down {abs(data.gain_loss_pct):.1f}% in {regime.value} regime — "
            f"exceeds tightened stop threshold of {low:.0f}% "
            f"(standard 7-8% not yet breached)"
        )
    else:
        detail = (
            f"{data.symbol} at {data.gain_loss_pct:+.1f}% in {regime.value} — "
            f"within tightened {low:.0f}-{high:.0f}% stop range"
        )

    return SellRuleResult(
        rule=SellRule.REGIME_TIGHTENED_STOP,
        triggered=triggered,
        value=data.gain_loss_pct,
        threshold=threshold,
        detail=detail,
    )


def check_earnings_risk(
    data: PositionMarketData,
    regime: ExitMarketRegime,
) -> SellRuleResult:
    """MUST_NOT-S2: Earnings within 3 weeks and insufficient profit cushion."""
    if data.days_until_earnings is None:
        return SellRuleResult(
            rule=SellRule.EARNINGS_RISK,
            triggered=False,
            value=None,
            threshold=None,
            detail=f"{data.symbol} no known earnings date — earnings risk not applicable",
        )

    near_earnings = data.days_until_earnings <= EARNINGS_PROXIMITY_DAYS
    insufficient_cushion = data.gain_loss_pct < EARNINGS_MIN_CUSHION_PCT
    triggered = near_earnings and insufficient_cushion

    if triggered:
        detail = (
            f"{data.symbol} earnings in {data.days_until_earnings} days with only "
            f"{data.gain_loss_pct:.1f}% cushion (need ≥{EARNINGS_MIN_CUSHION_PCT:.0f}%) — "
            f"avg earnings move: {data.avg_earnings_move_pct or '?'}%"
        )
    elif near_earnings:
        detail = (
            f"{data.symbol} earnings in {data.days_until_earnings} days but "
            f"{data.gain_loss_pct:.1f}% cushion sufficient (≥{EARNINGS_MIN_CUSHION_PCT:.0f}%)"
        )
    else:
        detail = (
            f"{data.symbol} earnings in {data.days_until_earnings} days — "
            f"not within {EARNINGS_PROXIMITY_DAYS}-day proximity window"
        )

    return SellRuleResult(
        rule=SellRule.EARNINGS_RISK,
        triggered=triggered,
        value=float(data.days_until_earnings),
        threshold=float(EARNINGS_PROXIMITY_DAYS),
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Urgency Classifier
# ---------------------------------------------------------------------------

def classify_urgency(
    rule_results: list[SellRuleResult],
    data: PositionMarketData,
    regime: ExitMarketRegime,
) -> Urgency:
    """Classify signal urgency based on triggered rules."""
    triggered = {r.rule for r in rule_results if r.triggered}

    # CRITICAL: stop loss, climax top, hard backstop, regime stop in CORRECTION
    if SellRule.STOP_LOSS_7_8 in triggered:
        return Urgency.CRITICAL
    if SellRule.CLIMAX_TOP in triggered:
        return Urgency.CRITICAL
    if data.gain_loss_pct <= -HARD_BACKSTOP_LOSS_PCT:
        return Urgency.CRITICAL
    if (
        SellRule.REGIME_TIGHTENED_STOP in triggered
        and regime == ExitMarketRegime.CORRECTION
    ):
        return Urgency.CRITICAL

    # WARNING: below 50MA, RS deterioration, earnings risk, sector dist,
    # regime stop in non-CORRECTION
    if SellRule.BELOW_50_DAY_MA in triggered:
        return Urgency.WARNING
    if SellRule.RS_DETERIORATION in triggered:
        return Urgency.WARNING
    if SellRule.EARNINGS_RISK in triggered:
        return Urgency.WARNING
    if SellRule.SECTOR_DISTRIBUTION in triggered:
        return Urgency.WARNING
    if SellRule.REGIME_TIGHTENED_STOP in triggered:
        return Urgency.WARNING

    # WATCH: profit target (time to consider trimming)
    if SellRule.PROFIT_TARGET_20_25 in triggered:
        return Urgency.WATCH

    # HEALTHY: nothing triggered
    return Urgency.HEALTHY


# ---------------------------------------------------------------------------
# Action Classifier
# ---------------------------------------------------------------------------

def classify_action(
    rule_results: list[SellRuleResult],
    data: PositionMarketData,
    regime: ExitMarketRegime,
    urgency: Urgency,
) -> Tuple[ExitActionType, SellType, Optional[float]]:
    """
    Classify action, sell type, and sell percentage.

    Returns (action, sell_type, sell_pct).
    """
    triggered = {r.rule for r in rule_results if r.triggered}

    # Priority 1: Stop loss — absolute SELL_ALL
    if SellRule.STOP_LOSS_7_8 in triggered:
        return ExitActionType.SELL_ALL, SellType.DEFENSIVE, 100.0

    # Regime tightened stop
    if SellRule.REGIME_TIGHTENED_STOP in triggered:
        return ExitActionType.SELL_ALL, SellType.DEFENSIVE, 100.0

    # Climax top — offensive SELL_ALL
    if SellRule.CLIMAX_TOP in triggered:
        return ExitActionType.SELL_ALL, SellType.OFFENSIVE, 100.0

    # Below 50-day MA — defensive SELL_ALL
    if SellRule.BELOW_50_DAY_MA in triggered:
        return ExitActionType.SELL_ALL, SellType.DEFENSIVE, 100.0

    # Earnings risk — defensive TRIM 50%
    if SellRule.EARNINGS_RISK in triggered:
        return ExitActionType.TRIM, SellType.DEFENSIVE, 50.0

    # Profit target — check 8-week hold exception
    if SellRule.PROFIT_TARGET_20_25 in triggered:
        return ExitActionType.TRIM, SellType.OFFENSIVE, 50.0

    # Check for 8-week hold (20%+ in < 3 weeks, profit target NOT triggered)
    if (
        data.gain_loss_pct >= 20.0
        and data.days_held <= FAST_GAIN_MAX_DAYS
    ):
        return ExitActionType.HOLD_8_WEEK, SellType.NOT_APPLICABLE, None

    # RS deterioration or sector distribution — tighten stop
    if SellRule.RS_DETERIORATION in triggered or SellRule.SECTOR_DISTRIBUTION in triggered:
        return ExitActionType.TIGHTEN_STOP, SellType.NOT_APPLICABLE, None

    # Healthy — hold
    return ExitActionType.HOLD, SellType.NOT_APPLICABLE, None


# ---------------------------------------------------------------------------
# Evidence Chain Builder
# ---------------------------------------------------------------------------

def build_evidence_chain(
    rule_results: list[SellRuleResult],
    data: PositionMarketData,
) -> list[EvidenceLink]:
    """Create evidence links for triggered rules. Always returns at least one."""
    evidence: list[EvidenceLink] = []

    for r in rule_results:
        if r.triggered:
            rule_id = RULE_ID_MAP.get(r.rule.value, "UNKNOWN")
            evidence.append(EvidenceLink(
                data_point=r.detail,
                rule_triggered=r.rule,
                rule_id=rule_id,
            ))

    # If no rules triggered (HEALTHY), add a healthy evidence link
    if not evidence:
        evidence.append(EvidenceLink(
            data_point=(
                f"{data.symbol} at {data.gain_loss_pct:+.1f}%, "
                f"RS {data.rs_rating_current}, above 50-day MA — no sell signals"
            ),
            rule_triggered=SellRule.NONE,
            rule_id="HEALTHY",
        ))

    return evidence


# ---------------------------------------------------------------------------
# Stop Price Calculator
# ---------------------------------------------------------------------------

def compute_stop_price(
    data: PositionMarketData,
    regime: ExitMarketRegime,
) -> float:
    """Compute the mental stop price based on regime thresholds."""
    low, _high = REGIME_STOP_THRESHOLDS[regime.value]
    stop_price = data.buy_price * (1 - low / 100)
    return round(max(0.0, stop_price), 2)


# ---------------------------------------------------------------------------
# Portfolio Impact Calculator
# ---------------------------------------------------------------------------

def compute_portfolio_impact(
    signals: list[PositionSignal],
    cash_pct: float,
) -> PortfolioImpact:
    """Calculate portfolio impact if all recommendations are executed."""
    n_critical = sum(1 for s in signals if s.urgency == Urgency.CRITICAL)
    n_warning = sum(1 for s in signals if s.urgency == Urgency.WARNING)
    n_watch = sum(1 for s in signals if s.urgency == Urgency.WATCH)
    n_healthy = sum(1 for s in signals if s.urgency == Urgency.HEALTHY)

    # Estimate cash freed by sells/trims
    sell_pct_freed = 0.0
    for s in signals:
        if s.action == ExitActionType.SELL_ALL:
            sell_pct_freed += 3.0  # ~3% per position average
        elif s.action == ExitActionType.TRIM:
            sell_pct_freed += 1.5  # ~1.5% per trim

    projected_cash = min(100.0, cash_pct + sell_pct_freed)

    # Top holding estimate (simplified)
    current_top = 5.0 if signals else 0.0
    # If the top holding is being sold, projected drops
    projected_top = current_top
    if signals and signals[0].action == ExitActionType.SELL_ALL:
        projected_top = max(0.0, current_top - 2.0)

    # Sector concentration
    sectors: dict[str, int] = {}
    for s in signals:
        if s.urgency == Urgency.HEALTHY:
            key = "unknown"  # Simplified; real impl would use sector data
            sectors[key] = sectors.get(key, 0) + 1
    max_sector_count = max(sectors.values()) if sectors else 0
    total_healthy = n_healthy or 1
    concentration = "HIGH" if max_sector_count / total_healthy > 0.4 else "LOW"

    return PortfolioImpact(
        current_cash_pct=round(cash_pct, 1),
        projected_cash_pct=round(projected_cash, 1),
        current_top_holding_pct=round(current_top, 1),
        projected_top_holding_pct=round(projected_top, 1),
        positions_healthy=n_healthy,
        positions_watch=n_watch,
        positions_warning=n_warning,
        positions_critical=n_critical,
        sector_concentration_risk=concentration,
    )


# ---------------------------------------------------------------------------
# Health Score
# ---------------------------------------------------------------------------

def compute_health_score(
    signals: list[PositionSignal],
    regime: ExitMarketRegime,
) -> int:
    """
    Compute portfolio health score (1-10).

    Start at 8. Deduct 2 per CRITICAL, 1 per WARNING.
    Add 0.5 per HEALTHY (max +2). Adjust for regime. Clamp [1, 10].
    """
    score = 8.0

    for s in signals:
        if s.urgency == Urgency.CRITICAL:
            score -= 2.0
        elif s.urgency == Urgency.WARNING:
            score -= 1.0

    # Bonus for healthy positions (max +2)
    healthy_count = sum(1 for s in signals if s.urgency == Urgency.HEALTHY)
    score += min(2.0, healthy_count * 0.5)

    # Regime adjustment
    if regime == ExitMarketRegime.CORRECTION:
        score -= 1.0
    elif regime == ExitMarketRegime.UPTREND_UNDER_PRESSURE:
        score -= 0.5

    return max(HEALTH_SCORE_RANGE[0], min(HEALTH_SCORE_RANGE[1], int(round(score))))


# ---------------------------------------------------------------------------
# Reasoning Text Builders
# ---------------------------------------------------------------------------

def _build_reasoning(
    data: PositionMarketData,
    urgency: Urgency,
    action: ExitActionType,
    rule_results: list[SellRuleResult],
) -> str:
    """Build human-readable reasoning string for a position signal."""
    triggered_rules = [r for r in rule_results if r.triggered]

    if not triggered_rules:
        return (
            f"{data.symbol} is performing within acceptable parameters at "
            f"{data.gain_loss_pct:+.1f}%, RS {data.rs_rating_current}. "
            f"No IBD sell rules triggered. Continue to hold with current stops."
        )

    parts = [f"{data.symbol}:"]
    for r in triggered_rules:
        parts.append(r.detail.split(" — ")[-1] if " — " in r.detail else r.detail)

    action_text = {
        ExitActionType.SELL_ALL: "Recommend immediate full exit.",
        ExitActionType.TRIM: "Recommend selling 50% of position.",
        ExitActionType.TIGHTEN_STOP: "Recommend tightening stop loss.",
        ExitActionType.HOLD: "Continue to hold.",
        ExitActionType.HOLD_8_WEEK: "Exceptional strength — hold for 8 weeks from breakout.",
    }
    parts.append(action_text.get(action, ""))

    return " ".join(parts)


def _build_what_would_change(
    data: PositionMarketData,
    urgency: Urgency,
    action: ExitActionType,
    rule_results: list[SellRuleResult],
) -> str:
    """Build text describing what would change the assessment."""
    if action == ExitActionType.SELL_ALL:
        triggered = [r.rule for r in rule_results if r.triggered]
        if SellRule.STOP_LOSS_7_8 in triggered:
            return (
                "Nothing — once the 7-8% stop loss is breached, the decision is final. "
                "The position should have been exited at -7%."
            )
        if SellRule.CLIMAX_TOP in triggered:
            return (
                "If volume normalizes and the stock consolidates sideways for 2+ weeks "
                "without further parabolic advance, the climax signal weakens."
            )
        if SellRule.BELOW_50_DAY_MA in triggered:
            return (
                f"If {data.symbol} recovers above the 50-day MA (${data.ma_50:.2f}) "
                f"within 2-3 trading days on increasing volume, downgrade to WATCH."
            )
        return (
            "A recovery above the stop threshold with institutional-quality volume "
            "would reduce urgency, but the current signal is definitive."
        )

    if action == ExitActionType.TRIM:
        return (
            f"If {data.symbol} breaks out to new highs with increasing RS Rating "
            f"and institutional accumulation, consider holding the remaining position."
        )

    if action == ExitActionType.TIGHTEN_STOP:
        return (
            f"If RS Rating stabilizes or improves above {RS_MIN_THRESHOLD} with "
            f"sector distribution days declining, loosen stops back to standard levels."
        )

    if action == ExitActionType.HOLD_8_WEEK:
        return (
            f"If {data.symbol} triggers the 7-8% stop loss during the 8-week hold, "
            f"the stop loss overrides this exception. Sell immediately."
        )

    # HOLD
    return (
        f"Monitor {data.symbol} for any deterioration in RS Rating below "
        f"{RS_MIN_THRESHOLD}, a break below the 50-day MA on heavy volume, "
        f"or approaching the 7-8% stop loss level."
    )


# ---------------------------------------------------------------------------
# Full Position Evaluator
# ---------------------------------------------------------------------------

def evaluate_position(
    data: PositionMarketData,
    regime: ExitMarketRegime,
    market_health: Optional[MarketHealthData] = None,
    tier: int = 2,
    asset_type: str = "stock",
) -> PositionSignal:
    """
    Orchestrate full exit evaluation for a single position.

    Runs all 8 sell rule checks, classifies urgency and action,
    builds evidence chain, and produces a PositionSignal.
    """
    # Run all 8 checks
    rule_results = [
        check_stop_loss_7_8(data, regime),
        check_climax_top(data, regime),
        check_below_50_day_ma(data, regime),
        check_rs_deterioration(data, regime),
        check_profit_target_20_25(data, regime),
        check_sector_distribution(data, regime, market_health),
        check_regime_tightened_stop(data, regime),
        check_earnings_risk(data, regime),
    ]

    # Classify
    urgency = classify_urgency(rule_results, data, regime)
    action, sell_type, sell_pct = classify_action(rule_results, data, regime, urgency)

    # Build evidence
    evidence = build_evidence_chain(rule_results, data)

    # Collect triggered rules
    triggered_rules = [r.rule for r in rule_results if r.triggered]
    if not triggered_rules:
        triggered_rules = [SellRule.NONE]

    # Compute stop price
    stop_price = compute_stop_price(data, regime)

    # Build reasoning text
    reasoning = _build_reasoning(data, urgency, action, rule_results)
    what_would_change = _build_what_would_change(data, urgency, action, rule_results)

    return PositionSignal(
        symbol=data.symbol,
        tier=tier,
        asset_type=asset_type,
        current_price=data.current_price,
        buy_price=data.buy_price,
        gain_loss_pct=data.gain_loss_pct,
        days_held=data.days_held,
        urgency=urgency,
        action=action,
        sell_type=sell_type,
        sell_pct=sell_pct,
        stop_price=stop_price,
        rules_triggered=triggered_rules,
        evidence=evidence,
        reasoning=reasoning,
        what_would_change=what_would_change,
    )


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class ExitAnalyzerInput(BaseModel):
    position_json: str = Field(
        ..., description="JSON string of PositionMarketData",
    )
    regime: str = Field(
        "CONFIRMED_UPTREND",
        description="ExitMarketRegime value",
    )


class ExitAnalyzerTool(BaseTool):
    """Evaluate positions against 8 IBD sell rules."""

    name: str = "exit_analyzer"
    description: str = (
        "Evaluate portfolio positions against 8 IBD sell rules: "
        "7-8% stop loss, climax top, 50-day MA break, RS deterioration, "
        "profit target, sector distribution, regime-tightened stops, "
        "and earnings risk. Returns urgency-ranked signals."
    )
    args_schema: type[BaseModel] = ExitAnalyzerInput

    def _run(
        self,
        position_json: str,
        regime: str = "CONFIRMED_UPTREND",
    ) -> str:
        import json
        data = PositionMarketData.model_validate_json(position_json)
        exit_regime = ExitMarketRegime(regime)
        signal = evaluate_position(data, exit_regime)
        return json.dumps(signal.model_dump(), indent=2, default=str)
