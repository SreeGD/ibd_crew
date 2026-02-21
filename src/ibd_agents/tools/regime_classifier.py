"""
Regime Detector Tool: Classification Logic
IBD Momentum Investment Framework v4.0

Pure functions implementing the regime classification decision tree.
No LLM, no file I/O, no market data fetching.

Priority resolution (spec §3.4):
  P1: dist_days >= 6 on either index -> CORRECTION (hard rule)
  P2: dist_days >= 5 on either index -> capped at UPTREND_UNDER_PRESSURE
  P3: No FTD after correction -> cannot be CONFIRMED_UPTREND
  P4: When signals conflict -> default to more conservative
  P5: Regime upgrades need 2+ confirming signals (MUST_NOT-S2)
      Downgrades can trigger on 1 definitive signal (PRIORITY 4)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from ibd_agents.schemas.regime_detector_output import (
    DEFENSIVE_ROTATION_THRESHOLD,
    DEFENSIVE_SECTORS,
    DIST_DAY_CORRECTION_THRESHOLD,
    DIST_DAY_UPTREND_CAP,
    EXPOSURE_RANGES,
    FTD_MIN_GAIN_PCT,
    FTD_MIN_RALLY_DAY,
    FTD_MODERATE_MAX_DAY,
    FTD_STRONG_GAIN_PCT,
    FTD_STRONG_MAX_DAY,
    FTD_STRONG_VOLUME_RATIO,
    HEALTH_SCORE_RANGES,
    INDICATOR_COUNT,
    LEADER_DETERIORATING_PCT,
    LEADER_HEALTHY_PCT,
    BreadthAssessment,
    Confidence,
    DistributionDayAssessment,
    FTDQuality,
    FollowThroughDayAssessment,
    LeaderAssessment,
    LeaderHealth,
    MarketRegime5,
    PreviousRegime,
    RegimeChange,
    RegimeDetectorOutput,
    SectorAssessment,
    SectorCharacter,
    SignalDirection,
    TransitionCondition,
    Trend,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Assessment builders
# ---------------------------------------------------------------------------

def assess_distribution(dist_data: dict) -> DistributionDayAssessment:
    """Convert raw distribution day data into a DistributionDayAssessment."""
    sp500_count = dist_data.get("sp500_dist_count", 0)
    nasdaq_count = dist_data.get("nasdaq_dist_count", 0)

    # Direction: if recent days have more dist days, deteriorating
    sp500_days = dist_data.get("sp500_dist_days", [])
    nasdaq_days = dist_data.get("nasdaq_dist_days", [])
    power_ups = dist_data.get("power_up_days", [])

    sp500_dir = _infer_trend(sp500_count, len(power_ups))
    nasdaq_dir = _infer_trend(nasdaq_count, len(power_ups))

    # Signal
    max_count = max(sp500_count, nasdaq_count)
    if max_count >= DIST_DAY_CORRECTION_THRESHOLD:
        signal = SignalDirection.BEARISH
    elif max_count >= DIST_DAY_UPTREND_CAP:
        signal = SignalDirection.BEARISH
    elif max_count <= 2:
        signal = SignalDirection.BULLISH
    else:
        signal = SignalDirection.NEUTRAL

    detail = (
        f"S&P 500 at {sp500_count}, Nasdaq at {nasdaq_count} distribution days "
        f"in trailing 25 sessions."
    )
    if max_count >= DIST_DAY_UPTREND_CAP:
        detail += f" Approaching/exceeding threshold of {DIST_DAY_UPTREND_CAP}."
    if power_ups:
        detail += f" {len(power_ups)} power-up day(s) removed oldest distribution day(s)."

    return DistributionDayAssessment(
        sp500_count=sp500_count,
        nasdaq_count=nasdaq_count,
        sp500_direction=sp500_dir,
        nasdaq_direction=nasdaq_dir,
        signal=signal,
        detail=detail,
    )


def assess_breadth(breadth_data: dict | str) -> BreadthAssessment:
    """Convert raw breadth data into a BreadthAssessment.
    If breadth_data is a string (error), return DATA UNAVAILABLE assessment.
    """
    if isinstance(breadth_data, str):
        # Error/unavailable — MUST-Q4
        return BreadthAssessment(
            pct_above_200ma=0.0,
            pct_above_50ma=0.0,
            new_highs=0,
            new_lows=0,
            advance_decline_direction=Trend.STABLE,
            breadth_divergence=False,
            signal=SignalDirection.NEUTRAL,
            detail=f"DATA UNAVAILABLE: {breadth_data}. Manual verification recommended.",
        )

    pct_200 = breadth_data.get("pct_above_200ma", 50.0)
    pct_50 = breadth_data.get("pct_above_50ma", 50.0)
    highs = breadth_data.get("new_highs_today", 0)
    lows = breadth_data.get("new_lows_today", 0)
    ad_dir_str = breadth_data.get("adv_decline_line_10d_direction", "flat")

    ad_direction = _str_to_trend(ad_dir_str)

    # Breadth divergence: strong index (proxy: pct_200 > 60) but deteriorating breadth
    divergence = pct_200 > 50 and (pct_50 < 45 or lows > highs) and ad_direction == Trend.DETERIORATING

    # Signal
    if pct_200 >= 65 and pct_50 >= 55 and highs > lows * 2:
        signal = SignalDirection.BULLISH
    elif pct_200 < 45 or pct_50 < 35 or lows > highs * 2:
        signal = SignalDirection.BEARISH
    else:
        signal = SignalDirection.NEUTRAL

    detail = (
        f"{pct_200:.0f}% above 200-day MA, {pct_50:.0f}% above 50-day MA. "
        f"{highs} new highs vs {lows} new lows. A/D line {ad_dir_str}."
    )
    if divergence:
        detail += " Breadth divergence detected."

    return BreadthAssessment(
        pct_above_200ma=pct_200,
        pct_above_50ma=pct_50,
        new_highs=highs,
        new_lows=lows,
        advance_decline_direction=ad_direction,
        breadth_divergence=divergence,
        signal=signal,
        detail=detail,
    )


def assess_leaders(leader_data: dict | str) -> LeaderAssessment:
    """Convert raw leader health data into a LeaderAssessment."""
    if isinstance(leader_data, str):
        return LeaderAssessment(
            rs90_above_50ma_pct=0.0,
            rs90_new_highs=0,
            rs90_new_lows=0,
            health=LeaderHealth.MIXED,
            signal=SignalDirection.NEUTRAL,
            detail=f"DATA UNAVAILABLE: {leader_data}. Manual verification recommended.",
        )

    pct = leader_data.get("rs90_above_50ma_pct", 50.0)
    new_highs = leader_data.get("rs90_new_highs", 0)
    new_lows = leader_data.get("rs90_new_lows", 0)

    # Health classification per MUST-M5
    if pct > LEADER_HEALTHY_PCT:
        health = LeaderHealth.HEALTHY
    elif pct >= LEADER_DETERIORATING_PCT:
        health = LeaderHealth.MIXED
    else:
        health = LeaderHealth.DETERIORATING

    # Signal
    if health == LeaderHealth.HEALTHY and new_highs > new_lows * 2:
        signal = SignalDirection.BULLISH
    elif health == LeaderHealth.DETERIORATING or new_lows > new_highs * 2:
        signal = SignalDirection.BEARISH
    else:
        signal = SignalDirection.NEUTRAL

    detail = (
        f"{pct:.0f}% of RS>=90 stocks above 50-day MA ({health.value}). "
        f"{new_highs} making new highs, {new_lows} making new lows."
    )

    return LeaderAssessment(
        rs90_above_50ma_pct=pct,
        rs90_new_highs=new_highs,
        rs90_new_lows=new_lows,
        health=health,
        signal=signal,
        detail=detail,
    )


def assess_sectors(sector_data: dict | str) -> SectorAssessment:
    """Convert raw sector ranking data into a SectorAssessment."""
    if isinstance(sector_data, str):
        return SectorAssessment(
            top_5_sectors=["DATA UNAVAILABLE"],
            defensive_in_top_5=0,
            character=SectorCharacter.BALANCED,
            signal=SignalDirection.NEUTRAL,
            detail=f"DATA UNAVAILABLE: {sector_data}. Manual verification recommended.",
        )

    # Extract top 5 sector names
    top_10 = sector_data.get("top_10", [])
    top_5_names = [s.get("sector_name", "Unknown") for s in top_10[:5]]
    if not top_5_names:
        top_5_names = ["Unknown"]

    defensive_count = sector_data.get("defensive_in_top_5", 0)

    # Character per MUST-M6
    if defensive_count >= DEFENSIVE_ROTATION_THRESHOLD:
        character = SectorCharacter.DEFENSIVE_ROTATION
    elif defensive_count == 0:
        character = SectorCharacter.GROWTH_LEADING
    else:
        character = SectorCharacter.BALANCED

    # Signal
    if character == SectorCharacter.DEFENSIVE_ROTATION:
        signal = SignalDirection.BEARISH
    elif character == SectorCharacter.GROWTH_LEADING:
        signal = SignalDirection.BULLISH
    else:
        signal = SignalDirection.NEUTRAL

    detail = (
        f"Top 5 sectors: {', '.join(top_5_names)}. "
        f"{defensive_count} defensive sector(s) in top 5 ({character.value})."
    )

    return SectorAssessment(
        top_5_sectors=top_5_names,
        defensive_in_top_5=defensive_count,
        character=character,
        signal=signal,
        detail=detail,
    )


def detect_follow_through_day(
    index_data_sp500: dict | None,
    index_data_nasdaq: dict | None,
    correction_low_date: Optional[date] = None,
) -> FollowThroughDayAssessment:
    """
    Scan OHLCV data for a Follow-Through Day per MUST-M3.

    FTD requires:
    - Day 4+ of rally attempt (Day 1 = first up day after correction low)
    - Gain >= 1.25%
    - Volume > prior day
    - On S&P 500 OR Nasdaq (one sufficient)

    Quality:
    - STRONG: >= 1.7% gain, >= 1.5x avg volume, Day 4-7
    - MODERATE: >= 1.25% gain, above prior day volume, Day 4-10
    - WEAK: barely qualifies or after Day 10
    """
    best_ftd: dict | None = None

    for index_name, index_data in [("S&P 500", index_data_sp500), ("Nasdaq", index_data_nasdaq)]:
        if index_data is None:
            continue
        bars = index_data.get("data", [])
        if len(bars) < 2:
            continue

        ftd = _find_ftd_in_bars(bars, correction_low_date)
        if ftd is not None:
            if best_ftd is None or _ftd_quality_rank(ftd["quality"]) < _ftd_quality_rank(best_ftd["quality"]):
                ftd["index"] = index_name
                best_ftd = ftd

    if best_ftd is None:
        return FollowThroughDayAssessment(
            detected=False,
            quality=FTDQuality.NONE,
            detail="No follow-through day detected in current rally attempt.",
        )

    quality = FTDQuality(best_ftd["quality"])
    return FollowThroughDayAssessment(
        detected=True,
        quality=quality,
        index=best_ftd["index"],
        gain_pct=best_ftd["gain_pct"],
        volume_vs_prior=best_ftd["volume_vs_prior"],
        rally_day_number=best_ftd["rally_day"],
        detail=(
            f"{quality.value} FTD on {best_ftd['index']} Day {best_ftd['rally_day']} "
            f"with {best_ftd['gain_pct']:.1f}% gain and {best_ftd['volume_vs_prior']:.1f}x volume."
        ),
    )


# ---------------------------------------------------------------------------
# Regime Classification — Main Entry Point
# ---------------------------------------------------------------------------

def classify_regime(
    dist_data: dict,
    breadth_data: dict | str,
    leader_data: dict | str,
    sector_data: dict | str,
    index_data_sp500: dict | None,
    index_data_nasdaq: dict | None,
    previous_regime: str = "UNKNOWN",
    correction_low_date: Optional[date] = None,
    correction_low_sp500: Optional[float] = None,
    correction_low_nasdaq: Optional[float] = None,
    analysis_date: Optional[str] = None,
) -> RegimeDetectorOutput:
    """
    Classify the market regime using all available indicator data.
    Implements the Section 7 decision tree with priority resolution.
    """
    if analysis_date is None:
        analysis_date = date.today().isoformat()

    prev = PreviousRegime(previous_regime)

    # Step 1: Build assessments
    dist = assess_distribution(dist_data)
    breadth = assess_breadth(breadth_data)
    leaders = assess_leaders(leader_data)
    sectors = assess_sectors(sector_data)

    # FTD detection — only when coming from CORRECTION/RALLY_ATTEMPT/UNKNOWN
    needs_ftd = prev in (
        PreviousRegime.CORRECTION,
        PreviousRegime.RALLY_ATTEMPT,
        PreviousRegime.FOLLOW_THROUGH_DAY,
        PreviousRegime.UNKNOWN,
    )
    if needs_ftd:
        ftd = detect_follow_through_day(index_data_sp500, index_data_nasdaq, correction_low_date)
    else:
        ftd = FollowThroughDayAssessment(
            detected=False,
            quality=FTDQuality.NONE,
            detail="FTD analysis not applicable — market not in correction/rally state.",
        )

    # Step 2: Count signals
    signals = [dist.signal, breadth.signal, leaders.signal, sectors.signal, ftd_signal(ftd)]
    bullish = sum(1 for s in signals if s == SignalDirection.BULLISH)
    neutral = sum(1 for s in signals if s == SignalDirection.NEUTRAL)
    bearish = sum(1 for s in signals if s == SignalDirection.BEARISH)

    # Step 3: Count data unavailable
    data_unavailable = sum(1 for d in [breadth_data, leader_data, sector_data] if isinstance(d, str))

    # Step 4: Apply decision tree
    regime = _apply_decision_tree(dist, breadth, leaders, sectors, ftd, prev)

    # Step 5: Compute derived values
    confidence = compute_confidence(bullish, bearish, data_unavailable)
    exposure = compute_exposure(regime, dist, breadth, leaders, sectors)
    health = compute_health_score(regime, bullish, bearish, breadth.breadth_divergence)

    # Step 6: Regime change
    regime_change = _build_regime_change(regime, prev, dist, breadth, leaders)

    # Step 7: Transition conditions
    transitions = generate_transition_conditions(regime, dist, breadth, leaders)

    # Step 8: Executive summary
    summary = build_executive_summary(
        regime, confidence, exposure, dist, breadth, leaders, sectors, ftd, regime_change,
    )

    return RegimeDetectorOutput(
        analysis_date=analysis_date,
        regime=regime,
        confidence=confidence,
        market_health_score=health,
        exposure_recommendation=exposure,
        distribution_days=dist,
        breadth=breadth,
        leaders=leaders,
        sectors=sectors,
        follow_through_day=ftd,
        bullish_signals=bullish,
        neutral_signals=neutral,
        bearish_signals=bearish,
        regime_change=regime_change,
        transition_conditions=transitions,
        executive_summary=summary,
        reasoning_source="deterministic",
    )


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------

def _apply_decision_tree(
    dist: DistributionDayAssessment,
    breadth: BreadthAssessment,
    leaders: LeaderAssessment,
    sectors: SectorAssessment,
    ftd: FollowThroughDayAssessment,
    prev: PreviousRegime,
) -> MarketRegime5:
    """Implement Section 7 decision tree with priority resolution."""
    max_dist = max(dist.sp500_count, dist.nasdaq_count)

    # P1: 6+ distribution days on either index → CORRECTION (hard rule)
    if max_dist >= DIST_DAY_CORRECTION_THRESHOLD:
        return MarketRegime5.CORRECTION

    # P2: 5+ distribution days → UPTREND_UNDER_PRESSURE at best
    cap_at_pressure = max_dist >= DIST_DAY_UPTREND_CAP

    # Check if leaders are also deteriorating with high dist → CORRECTION
    if cap_at_pressure and leaders.health == LeaderHealth.DETERIORATING:
        if sectors.character == SectorCharacter.DEFENSIVE_ROTATION:
            return MarketRegime5.CORRECTION

    # Coming from CORRECTION or RALLY_ATTEMPT → check for FTD
    if prev in (PreviousRegime.CORRECTION, PreviousRegime.RALLY_ATTEMPT, PreviousRegime.UNKNOWN):
        if ftd.detected:
            return MarketRegime5.FOLLOW_THROUGH_DAY
        # Rally attempt in progress?
        # If breadth improving and some bullish signals, RALLY_ATTEMPT
        # Otherwise stay CORRECTION
        if prev == PreviousRegime.CORRECTION:
            if breadth.advance_decline_direction == Trend.IMPROVING:
                return MarketRegime5.RALLY_ATTEMPT
            return MarketRegime5.CORRECTION
        if prev == PreviousRegime.RALLY_ATTEMPT:
            return MarketRegime5.RALLY_ATTEMPT
        # UNKNOWN → classify fresh
        if cap_at_pressure:
            return MarketRegime5.UPTREND_UNDER_PRESSURE

    # FOLLOW_THROUGH_DAY → can upgrade to CONFIRMED_UPTREND or fall back
    if prev == PreviousRegime.FOLLOW_THROUGH_DAY:
        if cap_at_pressure:
            return MarketRegime5.UPTREND_UNDER_PRESSURE
        if leaders.health == LeaderHealth.HEALTHY and not breadth.breadth_divergence:
            return MarketRegime5.CONFIRMED_UPTREND
        # Stay transitional if mixed
        if leaders.health == LeaderHealth.DETERIORATING:
            return MarketRegime5.RALLY_ATTEMPT
        return MarketRegime5.UPTREND_UNDER_PRESSURE

    # CONFIRMED_UPTREND → check for deterioration
    if prev == PreviousRegime.CONFIRMED_UPTREND:
        if cap_at_pressure:
            return MarketRegime5.UPTREND_UNDER_PRESSURE
        # 4+ dist AND (breadth divergence OR leaders deteriorating) → downgrade
        if max_dist >= 4:
            if breadth.breadth_divergence or leaders.health == LeaderHealth.DETERIORATING:
                return MarketRegime5.UPTREND_UNDER_PRESSURE
        # Leaders deteriorating alone → downgrade (MUST-M5)
        if leaders.health == LeaderHealth.DETERIORATING:
            return MarketRegime5.UPTREND_UNDER_PRESSURE
        return MarketRegime5.CONFIRMED_UPTREND

    # UPTREND_UNDER_PRESSURE → check for recovery or further deterioration
    if prev == PreviousRegime.UPTREND_UNDER_PRESSURE:
        if cap_at_pressure:
            if leaders.health == LeaderHealth.DETERIORATING:
                return MarketRegime5.CORRECTION
            return MarketRegime5.UPTREND_UNDER_PRESSURE
        # Recovery: needs 2+ improving signals (MUST_NOT-S2)
        improving_count = 0
        if dist.signal == SignalDirection.BULLISH:
            improving_count += 1
        if breadth.signal == SignalDirection.BULLISH:
            improving_count += 1
        if leaders.health == LeaderHealth.HEALTHY:
            improving_count += 1
        if improving_count >= 2:
            return MarketRegime5.CONFIRMED_UPTREND
        # Deteriorating further
        if leaders.health == LeaderHealth.DETERIORATING and sectors.character == SectorCharacter.DEFENSIVE_ROTATION:
            return MarketRegime5.CORRECTION
        return MarketRegime5.UPTREND_UNDER_PRESSURE

    # Default: classify from scratch based on signals
    return _classify_fresh(dist, breadth, leaders, sectors, ftd, cap_at_pressure)


def _classify_fresh(
    dist: DistributionDayAssessment,
    breadth: BreadthAssessment,
    leaders: LeaderAssessment,
    sectors: SectorAssessment,
    ftd: FollowThroughDayAssessment,
    cap_at_pressure: bool,
) -> MarketRegime5:
    """Classify regime from scratch (no previous context)."""
    if cap_at_pressure:
        return MarketRegime5.UPTREND_UNDER_PRESSURE

    # Count signal direction
    signals = [dist.signal, breadth.signal, leaders.signal, sectors.signal]
    bullish = sum(1 for s in signals if s == SignalDirection.BULLISH)
    bearish = sum(1 for s in signals if s == SignalDirection.BEARISH)

    if bullish >= 3 and bearish == 0:
        return MarketRegime5.CONFIRMED_UPTREND
    elif bearish >= 3:
        return MarketRegime5.CORRECTION
    elif bearish >= 2:
        return MarketRegime5.UPTREND_UNDER_PRESSURE
    elif bullish >= 2:
        return MarketRegime5.CONFIRMED_UPTREND
    else:
        return MarketRegime5.UPTREND_UNDER_PRESSURE


# ---------------------------------------------------------------------------
# Derived Computations
# ---------------------------------------------------------------------------

def compute_exposure(
    regime: MarketRegime5,
    dist: DistributionDayAssessment,
    breadth: BreadthAssessment,
    leaders: LeaderAssessment,
    sectors: SectorAssessment,
) -> int:
    """Compute exposure within the regime's allowed range (MUST-M7, MUST_NOT-S4)."""
    lo, hi = EXPOSURE_RANGES[regime.value]
    if lo == hi:
        return lo  # CORRECTION: always 0

    # Start at midpoint, adjust based on signal strength
    mid = (lo + hi) // 2
    adjustment = 0

    if leaders.health == LeaderHealth.HEALTHY:
        adjustment += 5
    elif leaders.health == LeaderHealth.DETERIORATING:
        adjustment -= 5

    if breadth.breadth_divergence:
        adjustment -= 5

    if sectors.character == SectorCharacter.GROWTH_LEADING:
        adjustment += 5
    elif sectors.character == SectorCharacter.DEFENSIVE_ROTATION:
        adjustment -= 5

    # Dist day pressure
    max_dist = max(dist.sp500_count, dist.nasdaq_count)
    if max_dist >= 4:
        adjustment -= 5
    elif max_dist <= 1:
        adjustment += 5

    exposure = max(lo, min(hi, mid + adjustment))
    return exposure


def compute_health_score(
    regime: MarketRegime5,
    bullish: int,
    bearish: int,
    breadth_divergence: bool,
) -> int:
    """Compute 1-10 health score within regime range (MUST-Q6)."""
    lo, hi = HEALTH_SCORE_RANGES[regime.value]

    # Base: midpoint of range
    base = (lo + hi) / 2.0

    # Adjust within range based on signal balance
    if bullish > bearish:
        adjustment = min(bullish - bearish, hi - int(base))
    elif bearish > bullish:
        adjustment = -min(bearish - bullish, int(base) - lo)
    else:
        adjustment = 0

    if breadth_divergence:
        adjustment -= 1

    score = int(round(base + adjustment))
    return max(lo, min(hi, score))


def compute_confidence(
    bullish: int,
    bearish: int,
    data_unavailable: int,
) -> Confidence:
    """
    Compute confidence level (MUST-Q2, MUST-Q4).
    HIGH: 4+ signals agree, no contradictions
    MEDIUM: 3 agree, 1-2 ambiguous
    LOW: mixed/contradictory or 2+ missing data
    """
    if data_unavailable >= 2:
        return Confidence.LOW

    max_agreement = max(bullish, bearish, INDICATOR_COUNT - bullish - bearish)

    if max_agreement >= 4:
        return Confidence.HIGH
    elif max_agreement >= 3:
        if data_unavailable >= 1:
            return Confidence.LOW
        return Confidence.MEDIUM
    else:
        return Confidence.LOW


def generate_transition_conditions(
    regime: MarketRegime5,
    dist: DistributionDayAssessment,
    breadth: BreadthAssessment,
    leaders: LeaderAssessment,
) -> list[TransitionCondition]:
    """Generate specific, measurable transition conditions (MUST-M8)."""
    conditions: list[TransitionCondition] = []
    max_dist = max(dist.sp500_count, dist.nasdaq_count)

    if regime == MarketRegime5.CONFIRMED_UPTREND:
        conditions.append(TransitionCondition(
            direction="DOWNGRADE",
            target_regime=MarketRegime5.UPTREND_UNDER_PRESSURE,
            condition=(
                f"S&P 500 or Nasdaq accumulates {DIST_DAY_UPTREND_CAP}+ distribution days "
                f"in trailing 25 sessions (currently at {max_dist})"
            ),
            likelihood="POSSIBLE" if max_dist >= 3 else "UNLIKELY",
        ))
        if leaders.health == LeaderHealth.MIXED:
            conditions.append(TransitionCondition(
                direction="DOWNGRADE",
                target_regime=MarketRegime5.UPTREND_UNDER_PRESSURE,
                condition=(
                    f"RS>=90 stocks above 50-day MA drops below 40% "
                    f"(currently at {leaders.rs90_above_50ma_pct:.0f}%)"
                ),
                likelihood="POSSIBLE",
            ))

    elif regime == MarketRegime5.UPTREND_UNDER_PRESSURE:
        conditions.append(TransitionCondition(
            direction="UPGRADE",
            target_regime=MarketRegime5.CONFIRMED_UPTREND,
            condition=(
                f"Distribution days drop below 4 on both indices AND "
                f"breadth stabilizes (A/D line turns flat or rising) AND "
                f"RS>=90 stocks above 50-day MA recovers above 60%"
            ),
            likelihood="POSSIBLE" if leaders.rs90_above_50ma_pct > 45 else "UNLIKELY",
        ))
        conditions.append(TransitionCondition(
            direction="DOWNGRADE",
            target_regime=MarketRegime5.CORRECTION,
            condition=(
                f"S&P 500 or Nasdaq reaches {DIST_DAY_CORRECTION_THRESHOLD}+ distribution days "
                f"OR leaders drop below 40% above 50-day MA with defensive rotation"
            ),
            likelihood="POSSIBLE" if max_dist >= 4 else "UNLIKELY",
        ))

    elif regime == MarketRegime5.CORRECTION:
        conditions.append(TransitionCondition(
            direction="UPGRADE",
            target_regime=MarketRegime5.RALLY_ATTEMPT,
            condition="Index posts first up day after making a new low to begin rally attempt",
            likelihood="POSSIBLE",
        ))

    elif regime == MarketRegime5.RALLY_ATTEMPT:
        conditions.append(TransitionCondition(
            direction="UPGRADE",
            target_regime=MarketRegime5.FOLLOW_THROUGH_DAY,
            condition=(
                f"S&P 500 or Nasdaq produces a follow-through day with "
                f">={FTD_MIN_GAIN_PCT}% gain on above-prior-day volume on Day 4+"
            ),
            likelihood="POSSIBLE",
        ))
        conditions.append(TransitionCondition(
            direction="DOWNGRADE",
            target_regime=MarketRegime5.CORRECTION,
            condition="Index undercuts the most recent correction low or rally attempt fails",
            likelihood="POSSIBLE",
        ))

    elif regime == MarketRegime5.FOLLOW_THROUGH_DAY:
        conditions.append(TransitionCondition(
            direction="UPGRADE",
            target_regime=MarketRegime5.CONFIRMED_UPTREND,
            condition=(
                "Leaders recover above 60% above 50-day MA AND "
                "no new distribution days accumulate in the next 1-3 sessions"
            ),
            likelihood="LIKELY" if leaders.rs90_above_50ma_pct > 50 else "POSSIBLE",
        ))
        conditions.append(TransitionCondition(
            direction="DOWNGRADE",
            target_regime=MarketRegime5.RALLY_ATTEMPT,
            condition="Follow-through day fails — index undercuts FTD day low within 3 sessions",
            likelihood="POSSIBLE",
        ))

    return conditions


def build_executive_summary(
    regime: MarketRegime5,
    confidence: Confidence,
    exposure: int,
    dist: DistributionDayAssessment,
    breadth: BreadthAssessment,
    leaders: LeaderAssessment,
    sectors: SectorAssessment,
    ftd: FollowThroughDayAssessment,
    regime_change: RegimeChange,
) -> str:
    """Build 3-4 sentence executive summary (MUST_NOT-S5: no forecasting)."""
    parts = []

    # Sentence 1: Classification
    parts.append(
        f"Market classified as {regime.value} with {confidence.value} confidence."
    )

    # Sentence 2: Key evidence
    evidence = []
    evidence.append(
        f"Distribution days at {dist.sp500_count} (S&P) and {dist.nasdaq_count} (Nasdaq)"
    )
    if breadth.pct_above_200ma > 0 or "DATA UNAVAILABLE" not in breadth.detail:
        evidence.append(f"breadth at {breadth.pct_above_200ma:.0f}% above 200-day MA")
    evidence.append(f"leaders {leaders.health.value.lower()} at {leaders.rs90_above_50ma_pct:.0f}% above 50-day MA")
    parts.append(". ".join(evidence[:3]) + ".")

    # Sentence 3: Sector context and regime change
    if regime_change.changed:
        parts.append(
            f"REGIME CHANGE: {regime_change.previous.value if regime_change.previous else 'UNKNOWN'} to "
            f"{regime_change.current.value if regime_change.current else regime.value}. "
            f"Trigger: {regime_change.trigger}."
        )
    else:
        if sectors.character == SectorCharacter.DEFENSIVE_ROTATION:
            parts.append("Defensive sectors leading — risk-off rotation underway.")
        elif sectors.character == SectorCharacter.GROWTH_LEADING:
            parts.append("Growth sectors leading — uptrend health confirmed by sector leadership.")
        else:
            parts.append("Sector leadership balanced between growth and defensive.")

    # Sentence 4: Exposure recommendation
    parts.append(f"Recommend {exposure}% portfolio exposure.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _build_regime_change(
    current: MarketRegime5,
    previous: PreviousRegime,
    dist: DistributionDayAssessment,
    breadth: BreadthAssessment,
    leaders: LeaderAssessment,
) -> RegimeChange:
    """Build regime change documentation (MUST-Q5)."""
    if previous == PreviousRegime.UNKNOWN:
        return RegimeChange(
            changed=False,
            previous=previous,
            current=current,
            trigger=None,
        )

    # Map to comparable values
    prev_regime_value = previous.value
    curr_regime_value = current.value

    if prev_regime_value == curr_regime_value:
        return RegimeChange(changed=False, previous=previous, current=current, trigger=None)

    # Determine trigger
    trigger = _determine_trigger(current, previous, dist, breadth, leaders)

    return RegimeChange(
        changed=True,
        previous=previous,
        current=current,
        trigger=trigger,
    )


def _determine_trigger(
    current: MarketRegime5,
    previous: PreviousRegime,
    dist: DistributionDayAssessment,
    breadth: BreadthAssessment,
    leaders: LeaderAssessment,
) -> str:
    """Determine what triggered a regime change."""
    max_dist = max(dist.sp500_count, dist.nasdaq_count)

    if current == MarketRegime5.CORRECTION:
        if max_dist >= DIST_DAY_CORRECTION_THRESHOLD:
            idx = "Nasdaq" if dist.nasdaq_count >= DIST_DAY_CORRECTION_THRESHOLD else "S&P 500"
            return f"{idx} reached {max_dist} distribution days, triggering correction classification"
        return "Multiple bearish signals converged: deteriorating leaders and defensive rotation"

    if current == MarketRegime5.UPTREND_UNDER_PRESSURE:
        if max_dist >= DIST_DAY_UPTREND_CAP:
            idx = "S&P 500" if dist.sp500_count >= DIST_DAY_UPTREND_CAP else "Nasdaq"
            return f"{idx} accumulated {max_dist} distribution days, capping regime at under pressure"
        if leaders.health == LeaderHealth.DETERIORATING:
            return f"Leaders deteriorating at {leaders.rs90_above_50ma_pct:.0f}% above 50-day MA"
        return "Distribution accumulation and breadth deterioration triggered downgrade"

    if current == MarketRegime5.CONFIRMED_UPTREND:
        return "Distribution days declined, breadth stabilized, and leaders recovered above healthy threshold"

    if current == MarketRegime5.FOLLOW_THROUGH_DAY:
        return "Follow-through day detected — rally attempt confirmed with qualifying gain on higher volume"

    if current == MarketRegime5.RALLY_ATTEMPT:
        return "Index began recovery attempt with breadth showing signs of improvement"

    return "Regime transition based on multi-indicator assessment"


def ftd_signal(ftd: FollowThroughDayAssessment) -> SignalDirection:
    """Determine signal direction for FTD assessment."""
    if ftd.detected:
        if ftd.quality in (FTDQuality.STRONG, FTDQuality.MODERATE):
            return SignalDirection.BULLISH
        return SignalDirection.NEUTRAL
    # No FTD is neutral when not in correction, bearish when rally attempt
    return SignalDirection.NEUTRAL


def _infer_trend(dist_count: int, power_up_count: int) -> Trend:
    """Infer distribution day trend."""
    if dist_count <= 1 and power_up_count > 0:
        return Trend.IMPROVING
    elif dist_count >= 4:
        return Trend.DETERIORATING
    else:
        return Trend.STABLE


def _str_to_trend(s: str) -> Trend:
    """Convert string to Trend enum."""
    s_lower = s.lower()
    if s_lower in ("rising", "improving"):
        return Trend.IMPROVING
    elif s_lower in ("declining", "deteriorating"):
        return Trend.DETERIORATING
    return Trend.STABLE


def _find_ftd_in_bars(
    bars: list[dict],
    correction_low_date: Optional[date] = None,
) -> dict | None:
    """Find a follow-through day in OHLCV bars."""
    if len(bars) < FTD_MIN_RALLY_DAY + 1:
        return None

    # Find the correction low (lowest close)
    low_idx = 0
    low_close = bars[0].get("close", 0)
    for i, bar in enumerate(bars):
        if bar.get("close", float("inf")) <= low_close:
            low_close = bar["close"]
            low_idx = i

    # Count rally days from Day 1 (first up day after low)
    rally_day = 0
    for i in range(low_idx + 1, len(bars)):
        bar = bars[i]
        change_pct = bar.get("change_pct", 0)
        if change_pct > 0:
            rally_day += 1
        # Check FTD conditions
        if rally_day >= FTD_MIN_RALLY_DAY:
            vol_ratio = bar.get("volume_vs_prior", 0)
            if change_pct >= FTD_MIN_GAIN_PCT and vol_ratio > 1.0:
                quality = _classify_ftd_quality(change_pct, vol_ratio, rally_day)
                return {
                    "gain_pct": change_pct,
                    "volume_vs_prior": vol_ratio,
                    "rally_day": rally_day,
                    "quality": quality,
                    "date": bar.get("date", ""),
                }

    return None


def _classify_ftd_quality(gain_pct: float, volume_ratio: float, rally_day: int) -> str:
    """Classify FTD quality per MUST-M3."""
    if gain_pct >= FTD_STRONG_GAIN_PCT and volume_ratio >= FTD_STRONG_VOLUME_RATIO and rally_day <= FTD_STRONG_MAX_DAY:
        return "STRONG"
    elif rally_day <= FTD_MODERATE_MAX_DAY:
        return "MODERATE"
    else:
        return "WEAK"


def _ftd_quality_rank(quality: str) -> int:
    """Lower rank = better quality."""
    return {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "NONE": 3}.get(quality, 3)
