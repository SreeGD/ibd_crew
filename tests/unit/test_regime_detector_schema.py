"""
Tests for Agent 16: Regime Detector — Schema Validation
Level 1: Pure Pydantic validation, no LLM, no file I/O.

Tests all enums, models, validators, and constants from
regime_detector_output.py.
"""

import pytest
from datetime import date
from pydantic import ValidationError

from ibd_agents.schemas.regime_detector_output import (
    # Constants
    DIST_DAY_DECLINE_PCT,
    DIST_DAY_WINDOW,
    DIST_DAY_UPTREND_CAP,
    DIST_DAY_CORRECTION_THRESHOLD,
    POWER_UP_GAIN_PCT,
    FTD_MIN_GAIN_PCT,
    FTD_STRONG_GAIN_PCT,
    FTD_STRONG_VOLUME_RATIO,
    FTD_MIN_RALLY_DAY,
    FTD_STRONG_MAX_DAY,
    FTD_MODERATE_MAX_DAY,
    LEADER_HEALTHY_PCT,
    LEADER_MIXED_LOW_PCT,
    LEADER_DETERIORATING_PCT,
    DEFENSIVE_SECTORS,
    DEFENSIVE_ROTATION_THRESHOLD,
    EXPOSURE_RANGES,
    HEALTH_SCORE_RANGES,
    REGIME_TO_LEGACY,
    INDICATOR_COUNT,
    # Enums
    MarketRegime5,
    PreviousRegime,
    Confidence,
    SignalDirection,
    Trend,
    LeaderHealth,
    SectorCharacter,
    FTDQuality,
    # Models
    DistributionDayAssessment,
    BreadthAssessment,
    LeaderAssessment,
    SectorAssessment,
    FollowThroughDayAssessment,
    TransitionCondition,
    RegimeChange,
    RegimeDetectorInput,
    RegimeDetectorOutput,
)


# ---------------------------------------------------------------------------
# Helpers: valid model builders
# ---------------------------------------------------------------------------

def _dist_assessment(**overrides) -> DistributionDayAssessment:
    defaults = dict(
        sp500_count=2,
        nasdaq_count=1,
        sp500_direction="STABLE",
        nasdaq_direction="STABLE",
        signal="BULLISH",
        detail="S&P at 2, Nasdaq at 1. Low distribution levels.",
    )
    defaults.update(overrides)
    return DistributionDayAssessment(**defaults)


def _breadth_assessment(**overrides) -> BreadthAssessment:
    defaults = dict(
        pct_above_200ma=72.0,
        pct_above_50ma=65.0,
        new_highs=280,
        new_lows=35,
        advance_decline_direction="IMPROVING",
        breadth_divergence=False,
        signal="BULLISH",
        detail="Broad participation with 280 new highs vs 35 new lows.",
    )
    defaults.update(overrides)
    return BreadthAssessment(**defaults)


def _leader_assessment(**overrides) -> LeaderAssessment:
    defaults = dict(
        rs90_above_50ma_pct=75.0,
        rs90_new_highs=22,
        rs90_new_lows=3,
        health="HEALTHY",
        signal="BULLISH",
        detail="75% of RS>=90 stocks above 50-day MA, healthy leader participation.",
    )
    defaults.update(overrides)
    return LeaderAssessment(**defaults)


def _sector_assessment(**overrides) -> SectorAssessment:
    defaults = dict(
        top_5_sectors=["Technology", "Consumer Discretionary", "Industrials", "Healthcare", "Financial"],
        defensive_in_top_5=0,
        character="GROWTH_LEADING",
        signal="BULLISH",
        detail="Growth sectors dominating top 5, no defensive rotation.",
    )
    defaults.update(overrides)
    return SectorAssessment(**defaults)


def _ftd_assessment(**overrides) -> FollowThroughDayAssessment:
    defaults = dict(
        detected=False,
        quality="NONE",
        index=None,
        gain_pct=None,
        volume_vs_prior=None,
        rally_day_number=None,
        detail="No follow-through day applicable in current uptrend.",
    )
    defaults.update(overrides)
    return FollowThroughDayAssessment(**defaults)


def _regime_change(**overrides) -> RegimeChange:
    defaults = dict(changed=False, previous=None, current=None, trigger=None)
    defaults.update(overrides)
    return RegimeChange(**defaults)


def _transition_condition(**overrides) -> TransitionCondition:
    defaults = dict(
        direction="DOWNGRADE",
        target_regime="UPTREND_UNDER_PRESSURE",
        condition="S&P 500 accumulates 5+ distribution days in trailing 25 sessions",
        likelihood="POSSIBLE",
    )
    defaults.update(overrides)
    return TransitionCondition(**defaults)


def _regime_output(**overrides) -> RegimeDetectorOutput:
    """Build a valid CONFIRMED_UPTREND output."""
    defaults = dict(
        analysis_date="2026-02-21",
        regime="CONFIRMED_UPTREND",
        confidence="HIGH",
        market_health_score=9,
        exposure_recommendation=90,
        distribution_days=_dist_assessment(),
        breadth=_breadth_assessment(),
        leaders=_leader_assessment(),
        sectors=_sector_assessment(),
        follow_through_day=_ftd_assessment(),
        bullish_signals=4,
        neutral_signals=1,
        bearish_signals=0,
        regime_change=_regime_change(),
        transition_conditions=[_transition_condition()],
        executive_summary=(
            "Market classified as CONFIRMED_UPTREND with HIGH confidence. "
            "Distribution days low at 2 (S&P) and 1 (Nasdaq). Breadth healthy "
            "with 72% above 200-day MA. Leaders strong at 75% above 50-day MA. "
            "Growth sectors leading. Recommend 90% exposure."
        ),
        reasoning_source="deterministic",
    )
    defaults.update(overrides)
    return RegimeDetectorOutput(**defaults)


# ===================================================================
# Constants
# ===================================================================

@pytest.mark.schema
class TestConstants:
    """Verify all threshold constants are defined and reasonable."""

    def test_distribution_thresholds(self):
        assert DIST_DAY_DECLINE_PCT == 0.2
        assert DIST_DAY_WINDOW == 25
        assert DIST_DAY_UPTREND_CAP == 5
        assert DIST_DAY_CORRECTION_THRESHOLD == 6
        assert POWER_UP_GAIN_PCT == 1.0

    def test_ftd_thresholds(self):
        assert FTD_MIN_GAIN_PCT == 1.25
        assert FTD_STRONG_GAIN_PCT == 1.7
        assert FTD_STRONG_VOLUME_RATIO == 1.5
        assert FTD_MIN_RALLY_DAY == 4
        assert FTD_STRONG_MAX_DAY == 7
        assert FTD_MODERATE_MAX_DAY == 10

    def test_leader_thresholds(self):
        assert LEADER_HEALTHY_PCT == 60.0
        assert LEADER_MIXED_LOW_PCT == 40.0
        assert LEADER_DETERIORATING_PCT == 40.0

    def test_defensive_sectors(self):
        assert len(DEFENSIVE_SECTORS) >= 4
        assert "Utilities" in DEFENSIVE_SECTORS
        assert "Consumer Staples" in DEFENSIVE_SECTORS
        assert DEFENSIVE_ROTATION_THRESHOLD == 3

    def test_exposure_ranges_cover_all_regimes(self):
        for regime in MarketRegime5:
            assert regime.value in EXPOSURE_RANGES
            lo, hi = EXPOSURE_RANGES[regime.value]
            assert lo <= hi

    def test_health_score_ranges_cover_all_regimes(self):
        for regime in MarketRegime5:
            assert regime.value in HEALTH_SCORE_RANGES
            lo, hi = HEALTH_SCORE_RANGES[regime.value]
            assert 1 <= lo <= hi <= 10

    def test_regime_to_legacy_covers_all_regimes(self):
        for regime in MarketRegime5:
            assert regime.value in REGIME_TO_LEGACY
            assert REGIME_TO_LEGACY[regime.value] in ("bull", "bear", "neutral")

    def test_indicator_count(self):
        assert INDICATOR_COUNT == 5


# ===================================================================
# Enums
# ===================================================================

@pytest.mark.schema
class TestEnums:
    """Verify all enum values exist."""

    def test_market_regime_5_values(self):
        assert len(MarketRegime5) == 5
        assert MarketRegime5("CONFIRMED_UPTREND") == MarketRegime5.CONFIRMED_UPTREND
        assert MarketRegime5("CORRECTION") == MarketRegime5.CORRECTION

    def test_previous_regime_includes_unknown(self):
        assert len(PreviousRegime) == 6
        assert PreviousRegime("UNKNOWN") == PreviousRegime.UNKNOWN

    def test_confidence_values(self):
        assert len(Confidence) == 3
        for v in ["HIGH", "MEDIUM", "LOW"]:
            assert Confidence(v)

    def test_signal_direction_values(self):
        assert len(SignalDirection) == 3
        for v in ["BULLISH", "NEUTRAL", "BEARISH"]:
            assert SignalDirection(v)

    def test_trend_values(self):
        assert len(Trend) == 3

    def test_leader_health_values(self):
        assert len(LeaderHealth) == 3

    def test_sector_character_values(self):
        assert len(SectorCharacter) == 3

    def test_ftd_quality_values(self):
        assert len(FTDQuality) == 4
        assert FTDQuality("NONE") == FTDQuality.NONE


# ===================================================================
# Assessment Models
# ===================================================================

@pytest.mark.schema
class TestDistributionDayAssessment:

    def test_valid_construction(self):
        a = _dist_assessment()
        assert a.sp500_count == 2
        assert a.nasdaq_count == 1
        assert a.signal == SignalDirection.BULLISH

    def test_negative_count_rejected(self):
        with pytest.raises(ValidationError):
            _dist_assessment(sp500_count=-1)

    def test_short_detail_rejected(self):
        with pytest.raises(ValidationError):
            _dist_assessment(detail="too short")


@pytest.mark.schema
class TestBreadthAssessment:

    def test_valid_construction(self):
        b = _breadth_assessment()
        assert b.pct_above_200ma == 72.0
        assert b.breadth_divergence is False

    def test_pct_above_200ma_out_of_range(self):
        with pytest.raises(ValidationError):
            _breadth_assessment(pct_above_200ma=101.0)

    def test_negative_new_lows_rejected(self):
        with pytest.raises(ValidationError):
            _breadth_assessment(new_lows=-1)


@pytest.mark.schema
class TestLeaderAssessment:

    def test_valid_construction(self):
        la = _leader_assessment()
        assert la.health == LeaderHealth.HEALTHY
        assert la.rs90_above_50ma_pct == 75.0

    def test_pct_above_100_rejected(self):
        with pytest.raises(ValidationError):
            _leader_assessment(rs90_above_50ma_pct=101.0)


@pytest.mark.schema
class TestSectorAssessment:

    def test_valid_construction(self):
        sa = _sector_assessment()
        assert len(sa.top_5_sectors) == 5
        assert sa.defensive_in_top_5 == 0

    def test_defensive_count_too_high(self):
        with pytest.raises(ValidationError):
            _sector_assessment(defensive_in_top_5=6)


@pytest.mark.schema
class TestFollowThroughDayAssessment:

    def test_no_ftd(self):
        ftd = _ftd_assessment()
        assert ftd.detected is False
        assert ftd.quality == FTDQuality.NONE

    def test_valid_ftd_detected(self):
        ftd = _ftd_assessment(
            detected=True,
            quality="STRONG",
            index="Nasdaq",
            gain_pct=2.1,
            volume_vs_prior=1.6,
            rally_day_number=5,
            detail="Strong FTD on Nasdaq Day 5 with 2.1% gain and 1.6x volume.",
        )
        assert ftd.detected is True
        assert ftd.quality == FTDQuality.STRONG

    def test_ftd_detected_but_none_quality_rejected(self):
        with pytest.raises(ValidationError, match="quality is NONE"):
            _ftd_assessment(
                detected=True,
                quality="NONE",
                index="Nasdaq",
                detail="FTD detected but quality set to NONE — invalid.",
            )

    def test_ftd_not_detected_but_strong_quality_rejected(self):
        with pytest.raises(ValidationError, match="quality is"):
            _ftd_assessment(
                detected=False,
                quality="STRONG",
                detail="Not detected but strong quality — invalid combination.",
            )

    def test_ftd_detected_without_index_rejected(self):
        with pytest.raises(ValidationError, match="index is not specified"):
            _ftd_assessment(
                detected=True,
                quality="MODERATE",
                index=None,
                detail="FTD detected but missing index — should fail validation.",
            )


@pytest.mark.schema
class TestTransitionCondition:

    def test_valid_downgrade(self):
        tc = _transition_condition()
        assert tc.direction == "DOWNGRADE"
        assert tc.likelihood == "POSSIBLE"

    def test_valid_upgrade(self):
        tc = _transition_condition(
            direction="UPGRADE",
            target_regime="CONFIRMED_UPTREND",
            condition="Distribution days drop below 4 on both indices and breadth stabilizes",
            likelihood="UNLIKELY",
        )
        assert tc.direction == "UPGRADE"

    def test_invalid_direction_rejected(self):
        with pytest.raises(ValidationError):
            _transition_condition(direction="SIDEWAYS")

    def test_short_condition_rejected(self):
        with pytest.raises(ValidationError):
            _transition_condition(condition="Too short")


@pytest.mark.schema
class TestRegimeChange:

    def test_no_change(self):
        rc = _regime_change()
        assert rc.changed is False

    def test_valid_change(self):
        rc = _regime_change(
            changed=True,
            previous="CONFIRMED_UPTREND",
            current="UPTREND_UNDER_PRESSURE",
            trigger="S&P 500 accumulated 5th distribution day on Feb 18",
        )
        assert rc.changed is True
        assert rc.previous == PreviousRegime.CONFIRMED_UPTREND
        assert rc.current == MarketRegime5.UPTREND_UNDER_PRESSURE

    def test_change_without_previous_rejected(self):
        with pytest.raises(ValidationError, match="previous is not set"):
            _regime_change(
                changed=True,
                previous=None,
                current="CORRECTION",
                trigger="Distribution days hit 6 on Nasdaq today.",
            )

    def test_change_without_trigger_rejected(self):
        with pytest.raises(ValidationError, match="trigger is missing"):
            _regime_change(
                changed=True,
                previous="CONFIRMED_UPTREND",
                current="CORRECTION",
                trigger=None,
            )


@pytest.mark.schema
class TestRegimeDetectorInput:

    def test_default_previous_regime(self):
        inp = RegimeDetectorInput(analysis_date=date(2026, 2, 21))
        assert inp.previous_regime == PreviousRegime.UNKNOWN

    def test_with_correction_context(self):
        inp = RegimeDetectorInput(
            analysis_date=date(2026, 2, 21),
            previous_regime="CORRECTION",
            previous_exposure_recommendation=0,
            correction_low_date=date(2026, 2, 10),
            correction_low_sp500=5800.0,
            correction_low_nasdaq=15000.0,
        )
        assert inp.previous_regime == PreviousRegime.CORRECTION
        assert inp.correction_low_sp500 == 5800.0

    def test_exposure_above_100_rejected(self):
        with pytest.raises(ValidationError):
            RegimeDetectorInput(
                analysis_date=date(2026, 2, 21),
                previous_exposure_recommendation=101,
            )


# ===================================================================
# RegimeDetectorOutput — top-level validators
# ===================================================================

@pytest.mark.schema
class TestRegimeDetectorOutput:

    def test_valid_confirmed_uptrend(self):
        out = _regime_output()
        assert out.regime == MarketRegime5.CONFIRMED_UPTREND
        assert out.confidence == Confidence.HIGH
        assert out.market_health_score == 9

    def test_valid_correction(self):
        out = _regime_output(
            regime="CORRECTION",
            confidence="HIGH",
            market_health_score=2,
            exposure_recommendation=0,
            distribution_days=_dist_assessment(
                sp500_count=4, nasdaq_count=6,
                signal="BEARISH",
                sp500_direction="DETERIORATING",
                nasdaq_direction="DETERIORATING",
                detail="Nasdaq at 6 distribution days, severe selling pressure.",
            ),
            breadth=_breadth_assessment(
                pct_above_200ma=35.0, pct_above_50ma=25.0,
                new_highs=20, new_lows=300,
                advance_decline_direction="DETERIORATING",
                breadth_divergence=False,
                signal="BEARISH",
                detail="Only 35% above 200MA, 300 new lows, severe deterioration.",
            ),
            leaders=_leader_assessment(
                rs90_above_50ma_pct=20.0, rs90_new_highs=1, rs90_new_lows=30,
                health="DETERIORATING", signal="BEARISH",
                detail="Only 20% of leaders above 50MA, severe breakdown.",
            ),
            sectors=_sector_assessment(
                top_5_sectors=["Utilities", "Consumer Staples", "Healthcare", "REITs", "Bonds/Treasuries"],
                defensive_in_top_5=5, character="DEFENSIVE_ROTATION", signal="BEARISH",
                detail="All top 5 sectors defensive — full risk-off rotation.",
            ),
            follow_through_day=_ftd_assessment(
                detail="No follow-through day — market in correction.",
            ),
            bullish_signals=0,
            neutral_signals=0,
            bearish_signals=5,
            transition_conditions=[
                _transition_condition(
                    direction="UPGRADE",
                    target_regime="RALLY_ATTEMPT",
                    condition="Index posts first up day after making a new low to begin rally attempt",
                    likelihood="POSSIBLE",
                )
            ],
            executive_summary=(
                "Market classified as CORRECTION with HIGH confidence. "
                "Nasdaq has 6 distribution days. Breadth severely deteriorated "
                "with 300 new lows. Leaders broken at 20% above 50-day MA. "
                "All top 5 sectors are defensive. Exposure at 0%."
            ),
        )
        assert out.regime == MarketRegime5.CORRECTION
        assert out.exposure_recommendation == 0

    def test_valid_follow_through_day(self):
        out = _regime_output(
            regime="FOLLOW_THROUGH_DAY",
            confidence="MEDIUM",
            market_health_score=5,
            exposure_recommendation=30,
            distribution_days=_dist_assessment(sp500_count=3, nasdaq_count=2),
            follow_through_day=_ftd_assessment(
                detected=True,
                quality="STRONG",
                index="Nasdaq",
                gain_pct=2.1,
                volume_vs_prior=1.6,
                rally_day_number=5,
                detail="Strong FTD on Nasdaq Day 5 with 2.1% gain.",
            ),
        )
        assert out.regime == MarketRegime5.FOLLOW_THROUGH_DAY
        assert out.follow_through_day.detected is True

    def test_valid_rally_attempt(self):
        out = _regime_output(
            regime="RALLY_ATTEMPT",
            confidence="MEDIUM",
            market_health_score=3,
            exposure_recommendation=10,
            distribution_days=_dist_assessment(sp500_count=2, nasdaq_count=1),
            leaders=_leader_assessment(
                rs90_above_50ma_pct=50.0, health="MIXED", signal="NEUTRAL",
                detail="Mixed leader health, 50% above 50MA during rally.",
            ),
            breadth=_breadth_assessment(signal="NEUTRAL",
                detail="Neutral breadth during rally attempt, improving slowly.",
            ),
            sectors=_sector_assessment(signal="NEUTRAL",
                detail="Balanced sector leadership during rally attempt.",
            ),
            follow_through_day=_ftd_assessment(
                detail="No follow-through day yet, rally attempt Day 3.",
            ),
            bullish_signals=1,
            neutral_signals=3,
            bearish_signals=1,
        )
        assert out.regime == MarketRegime5.RALLY_ATTEMPT
        assert 0 <= out.exposure_recommendation <= 20


# ===================================================================
# Validator Edge Cases
# ===================================================================

@pytest.mark.schema
class TestExposureValidator:

    def test_correction_with_nonzero_exposure_rejected(self):
        with pytest.raises(ValidationError, match="Exposure.*outside range.*\\[0, 0\\]"):
            _regime_output(
                regime="CORRECTION",
                market_health_score=2,
                exposure_recommendation=10,
                distribution_days=_dist_assessment(sp500_count=6, nasdaq_count=5, signal="BEARISH",
                    detail="6 distribution days, severe selling pressure on S&P."),
                breadth=_breadth_assessment(signal="BEARISH",
                    detail="Severe breadth deterioration during correction."),
                leaders=_leader_assessment(health="DETERIORATING", signal="BEARISH",
                    detail="Leaders broken down, only 20% above 50-day MA."),
                sectors=_sector_assessment(character="DEFENSIVE_ROTATION", signal="BEARISH",
                    detail="Full defensive rotation in top sectors."),
                bullish_signals=0, neutral_signals=0, bearish_signals=5,
            )

    def test_confirmed_uptrend_exposure_below_80_rejected(self):
        with pytest.raises(ValidationError, match="Exposure.*outside range.*\\[80, 100\\]"):
            _regime_output(exposure_recommendation=50)

    def test_rally_attempt_exposure_above_20_rejected(self):
        with pytest.raises(ValidationError, match="Exposure.*outside range.*\\[0, 20\\]"):
            _regime_output(
                regime="RALLY_ATTEMPT",
                market_health_score=3,
                exposure_recommendation=30,
                bullish_signals=1, neutral_signals=3, bearish_signals=1,
            )


@pytest.mark.schema
class TestHealthScoreValidator:

    def test_confirmed_uptrend_low_health_rejected(self):
        with pytest.raises(ValidationError, match="Health score.*outside range"):
            _regime_output(market_health_score=3)

    def test_correction_high_health_rejected(self):
        with pytest.raises(ValidationError, match="Health score.*outside range"):
            _regime_output(
                regime="CORRECTION",
                market_health_score=8,
                exposure_recommendation=0,
                distribution_days=_dist_assessment(sp500_count=6, nasdaq_count=5, signal="BEARISH",
                    detail="6 distribution days, correction-level selling."),
                breadth=_breadth_assessment(signal="BEARISH",
                    detail="Severely deteriorated breadth during correction."),
                leaders=_leader_assessment(health="DETERIORATING", signal="BEARISH",
                    detail="Leaders broken down during correction."),
                sectors=_sector_assessment(character="DEFENSIVE_ROTATION", signal="BEARISH",
                    detail="Defensive rotation during correction."),
                bullish_signals=0, neutral_signals=0, bearish_signals=5,
            )


@pytest.mark.schema
class TestSignalCountValidator:

    def test_signals_sum_not_5_rejected(self):
        with pytest.raises(ValidationError, match="Signal counts must sum to 5"):
            _regime_output(bullish_signals=3, neutral_signals=1, bearish_signals=0)

    def test_signals_exactly_5_accepted(self):
        out = _regime_output(bullish_signals=3, neutral_signals=2, bearish_signals=0)
        assert out.bullish_signals + out.neutral_signals + out.bearish_signals == 5


@pytest.mark.schema
class TestFTDRegimeConsistency:

    def test_ftd_regime_without_detected_rejected(self):
        with pytest.raises(ValidationError, match="follow_through_day.detected is False"):
            _regime_output(
                regime="FOLLOW_THROUGH_DAY",
                market_health_score=5,
                exposure_recommendation=30,
                follow_through_day=_ftd_assessment(),  # detected=False
            )


@pytest.mark.schema
class TestDistDayCapValidator:

    def test_5_dist_days_confirmed_uptrend_rejected(self):
        with pytest.raises(ValidationError, match="MUST_NOT-S3"):
            _regime_output(
                distribution_days=_dist_assessment(sp500_count=5, nasdaq_count=3),
            )

    def test_4_dist_days_confirmed_uptrend_accepted(self):
        out = _regime_output(
            distribution_days=_dist_assessment(sp500_count=4, nasdaq_count=3),
        )
        assert out.distribution_days.sp500_count == 4

    def test_5_dist_nasdaq_confirmed_uptrend_rejected(self):
        with pytest.raises(ValidationError, match="MUST_NOT-S3"):
            _regime_output(
                distribution_days=_dist_assessment(sp500_count=2, nasdaq_count=5),
            )


# ===================================================================
# Backward Compatibility
# ===================================================================

@pytest.mark.schema
class TestBackwardCompatibility:

    def test_confirmed_uptrend_to_bull(self):
        out = _regime_output(regime="CONFIRMED_UPTREND")
        assert out.to_legacy_regime() == "bull"

    def test_uptrend_under_pressure_to_neutral(self):
        out = _regime_output(
            regime="UPTREND_UNDER_PRESSURE",
            market_health_score=5,
            exposure_recommendation=50,
            distribution_days=_dist_assessment(sp500_count=4, nasdaq_count=3),
            breadth=_breadth_assessment(signal="NEUTRAL",
                detail="Mixed breadth with declining participation."),
            leaders=_leader_assessment(health="MIXED", signal="NEUTRAL",
                detail="Leaders mixed at 48% above 50-day MA."),
            bullish_signals=1, neutral_signals=3, bearish_signals=1,
        )
        assert out.to_legacy_regime() == "neutral"

    def test_follow_through_day_to_neutral(self):
        out = _regime_output(
            regime="FOLLOW_THROUGH_DAY",
            confidence="MEDIUM",
            market_health_score=5,
            exposure_recommendation=30,
            distribution_days=_dist_assessment(sp500_count=3, nasdaq_count=2),
            follow_through_day=_ftd_assessment(
                detected=True, quality="STRONG", index="Nasdaq",
                gain_pct=2.1, volume_vs_prior=1.6, rally_day_number=5,
                detail="Strong FTD detected on Nasdaq Day 5.",
            ),
            bullish_signals=2, neutral_signals=2, bearish_signals=1,
        )
        assert out.to_legacy_regime() == "neutral"

    def test_rally_attempt_to_bear(self):
        out = _regime_output(
            regime="RALLY_ATTEMPT",
            confidence="MEDIUM",
            market_health_score=3,
            exposure_recommendation=10,
            distribution_days=_dist_assessment(sp500_count=2, nasdaq_count=1),
            breadth=_breadth_assessment(signal="NEUTRAL",
                detail="Breadth beginning to stabilize during rally."),
            leaders=_leader_assessment(health="MIXED", signal="NEUTRAL",
                detail="Leaders mixed during rally attempt."),
            sectors=_sector_assessment(signal="NEUTRAL",
                detail="Mixed sector leadership during rally."),
            bullish_signals=1, neutral_signals=3, bearish_signals=1,
        )
        assert out.to_legacy_regime() == "bear"

    def test_correction_to_bear(self):
        out = _regime_output(
            regime="CORRECTION",
            confidence="HIGH",
            market_health_score=2,
            exposure_recommendation=0,
            distribution_days=_dist_assessment(sp500_count=6, nasdaq_count=5, signal="BEARISH",
                detail="6 distribution days on S&P, correction level."),
            breadth=_breadth_assessment(signal="BEARISH",
                detail="Severe breadth deterioration."),
            leaders=_leader_assessment(health="DETERIORATING", signal="BEARISH",
                detail="Leaders broken at 20% above 50-day MA."),
            sectors=_sector_assessment(character="DEFENSIVE_ROTATION", signal="BEARISH",
                detail="Full defensive rotation."),
            bullish_signals=0, neutral_signals=0, bearish_signals=5,
            transition_conditions=[
                _transition_condition(
                    direction="UPGRADE", target_regime="RALLY_ATTEMPT",
                    condition="Index posts first up day after new low to begin rally attempt",
                    likelihood="POSSIBLE",
                )
            ],
            executive_summary=(
                "Market classified as CORRECTION. Six distribution days on S&P 500. "
                "Severe breadth deterioration. Leaders broken at 20%. "
                "Full defensive sector rotation. Exposure at 0%."
            ),
        )
        assert out.to_legacy_regime() == "bear"

    def test_all_legacy_values_are_valid(self):
        """Every legacy mapping produces bull/bear/neutral."""
        for regime in MarketRegime5:
            legacy = REGIME_TO_LEGACY[regime.value]
            assert legacy in ("bull", "bear", "neutral"), f"{regime.value} -> {legacy}"
