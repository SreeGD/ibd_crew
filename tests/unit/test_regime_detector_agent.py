"""
Tests for Agent 16: Regime Detector — Pipeline and Tool Tests
Level 2: Deterministic pipeline, classifier functions, golden dataset.

All tests use mock data (no yfinance/network).
"""

import json
import pytest
from datetime import date
from pathlib import Path

from ibd_agents.schemas.regime_detector_output import (
    DIST_DAY_CORRECTION_THRESHOLD,
    DIST_DAY_UPTREND_CAP,
    EXPOSURE_RANGES,
    HEALTH_SCORE_RANGES,
    INDICATOR_COUNT,
    REGIME_TO_LEGACY,
    BreadthAssessment,
    Confidence,
    DistributionDayAssessment,
    FTDQuality,
    FollowThroughDayAssessment,
    LeaderAssessment,
    LeaderHealth,
    MarketRegime5,
    PreviousRegime,
    RegimeDetectorOutput,
    SectorAssessment,
    SectorCharacter,
    SignalDirection,
    Trend,
)
from ibd_agents.tools.regime_classifier import (
    assess_breadth,
    assess_distribution,
    assess_leaders,
    assess_sectors,
    build_executive_summary,
    classify_regime,
    compute_confidence,
    compute_exposure,
    compute_health_score,
    detect_follow_through_day,
    generate_transition_conditions,
)
from ibd_agents.tools.regime_data_fetcher import (
    get_distribution_days_mock,
    get_index_price_history_mock,
    get_leading_stocks_health_mock,
    get_market_breadth_mock,
    get_sector_rankings_mock,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def _force_mock_data():
    """Tests always use mock data — disable yfinance."""
    import ibd_agents.tools.regime_data_fetcher as fetcher
    fetcher.HAS_YFINANCE = False


GOLDEN_PATH = Path(__file__).resolve().parents[2] / "golden_datasets" / "regime_detector_golden.json"


@pytest.fixture(scope="session")
def golden():
    """Load golden dataset."""
    return json.loads(GOLDEN_PATH.read_text())


@pytest.fixture(scope="session")
def pipeline_output():
    """Cache default pipeline output (healthy_uptrend)."""
    from ibd_agents.agents.regime_detector import run_regime_detector_pipeline
    return run_regime_detector_pipeline(scenario="healthy_uptrend")


@pytest.fixture(scope="session")
def all_scenarios():
    """Cache pipeline outputs for all 5 scenarios."""
    from ibd_agents.agents.regime_detector import run_regime_detector_pipeline
    scenarios = {
        "healthy_uptrend": ("CONFIRMED_UPTREND", "CONFIRMED_UPTREND"),
        "under_pressure": ("CONFIRMED_UPTREND", "CONFIRMED_UPTREND"),
        "correction": ("CORRECTION", "CORRECTION"),
        "rally_attempt": ("CORRECTION", "CORRECTION"),
        "follow_through_day": ("RALLY_ATTEMPT", "RALLY_ATTEMPT"),
    }
    results = {}
    for scenario, (prev, _) in scenarios.items():
        results[scenario] = run_regime_detector_pipeline(
            scenario=scenario,
            previous_regime=prev,
        )
    return results


# ===================================================================
# Mock Data Functions
# ===================================================================

@pytest.mark.schema
class TestMockDataFunctions:
    """Verify mock data functions return valid dicts for all scenarios."""

    @pytest.mark.parametrize("scenario", [
        "healthy_uptrend", "under_pressure", "correction", "rally_attempt", "follow_through_day",
    ])
    def test_distribution_days_mock(self, scenario):
        data = get_distribution_days_mock(scenario)
        assert isinstance(data, dict)
        assert "sp500_dist_count" in data
        assert "nasdaq_dist_count" in data
        assert data["sp500_dist_count"] >= 0
        assert data["nasdaq_dist_count"] >= 0

    @pytest.mark.parametrize("scenario", [
        "healthy_uptrend", "under_pressure", "correction", "rally_attempt", "follow_through_day",
    ])
    def test_breadth_mock(self, scenario):
        data = get_market_breadth_mock(scenario)
        assert isinstance(data, dict)
        assert "pct_above_200ma" in data
        assert 0 <= data["pct_above_200ma"] <= 100

    @pytest.mark.parametrize("scenario", [
        "healthy_uptrend", "under_pressure", "correction", "rally_attempt", "follow_through_day",
    ])
    def test_leaders_mock(self, scenario):
        data = get_leading_stocks_health_mock(scenario)
        assert isinstance(data, dict)
        assert "rs90_above_50ma_pct" in data

    @pytest.mark.parametrize("scenario", [
        "healthy_uptrend", "under_pressure", "correction", "rally_attempt", "follow_through_day",
    ])
    def test_sectors_mock(self, scenario):
        data = get_sector_rankings_mock(scenario)
        assert isinstance(data, dict)
        assert "defensive_in_top_5" in data

    @pytest.mark.parametrize("scenario", ["rally_attempt", "follow_through_day"])
    def test_index_history_mock(self, scenario):
        for index in ["SP500", "NASDAQ"]:
            data = get_index_price_history_mock(index, scenario)
            assert isinstance(data, dict)
            assert "data" in data

    def test_unknown_scenario_falls_back(self):
        data = get_distribution_days_mock("nonexistent")
        assert data["sp500_dist_count"] == 2  # Falls back to healthy_uptrend


# ===================================================================
# Assessment Functions
# ===================================================================

@pytest.mark.schema
class TestAssessDistribution:

    def test_low_dist_is_bullish(self):
        dist = assess_distribution(get_distribution_days_mock("healthy_uptrend"))
        assert dist.signal == SignalDirection.BULLISH
        assert dist.sp500_count == 2
        assert dist.nasdaq_count == 1

    def test_high_dist_is_bearish(self):
        dist = assess_distribution(get_distribution_days_mock("correction"))
        assert dist.signal == SignalDirection.BEARISH
        assert max(dist.sp500_count, dist.nasdaq_count) >= DIST_DAY_UPTREND_CAP

    def test_moderate_dist_is_neutral(self):
        data = get_distribution_days_mock("follow_through_day")
        dist = assess_distribution(data)
        assert dist.sp500_count == 3
        assert dist.signal in (SignalDirection.NEUTRAL, SignalDirection.BEARISH)


@pytest.mark.schema
class TestAssessBreadth:

    def test_healthy_breadth_is_bullish(self):
        breadth = assess_breadth(get_market_breadth_mock("healthy_uptrend"))
        assert breadth.signal == SignalDirection.BULLISH
        assert breadth.breadth_divergence is False

    def test_weak_breadth_is_bearish(self):
        breadth = assess_breadth(get_market_breadth_mock("correction"))
        assert breadth.signal == SignalDirection.BEARISH

    def test_error_string_produces_data_unavailable(self):
        breadth = assess_breadth("ERROR: Service unavailable")
        assert "DATA UNAVAILABLE" in breadth.detail
        assert breadth.signal == SignalDirection.NEUTRAL


@pytest.mark.schema
class TestAssessLeaders:

    def test_healthy_leaders(self):
        leaders = assess_leaders(get_leading_stocks_health_mock("healthy_uptrend"))
        assert leaders.health == LeaderHealth.HEALTHY
        assert leaders.signal == SignalDirection.BULLISH

    def test_deteriorating_leaders(self):
        leaders = assess_leaders(get_leading_stocks_health_mock("correction"))
        assert leaders.health == LeaderHealth.DETERIORATING
        assert leaders.signal == SignalDirection.BEARISH

    def test_mixed_leaders(self):
        leaders = assess_leaders(get_leading_stocks_health_mock("under_pressure"))
        assert leaders.health == LeaderHealth.MIXED

    def test_error_string_produces_data_unavailable(self):
        leaders = assess_leaders("ERROR: Network timeout")
        assert "DATA UNAVAILABLE" in leaders.detail


@pytest.mark.schema
class TestAssessSectors:

    def test_growth_leading(self):
        sectors = assess_sectors(get_sector_rankings_mock("healthy_uptrend"))
        assert sectors.character == SectorCharacter.GROWTH_LEADING
        assert sectors.defensive_in_top_5 == 0
        assert sectors.signal == SignalDirection.BULLISH

    def test_defensive_rotation(self):
        sectors = assess_sectors(get_sector_rankings_mock("correction"))
        assert sectors.character == SectorCharacter.DEFENSIVE_ROTATION
        assert sectors.defensive_in_top_5 >= 3
        assert sectors.signal == SignalDirection.BEARISH

    def test_balanced(self):
        sectors = assess_sectors(get_sector_rankings_mock("under_pressure"))
        assert sectors.character == SectorCharacter.BALANCED
        assert sectors.defensive_in_top_5 > 0
        assert sectors.defensive_in_top_5 < 3


# ===================================================================
# FTD Detection
# ===================================================================

@pytest.mark.schema
class TestFollowThroughDay:

    def test_strong_ftd_detected(self):
        """REG-004: Day 5 of rally, 2.1% gain, 1.6x volume = STRONG FTD."""
        sp500 = get_index_price_history_mock("SP500", "follow_through_day")
        nasdaq = get_index_price_history_mock("NASDAQ", "follow_through_day")
        ftd = detect_follow_through_day(sp500, nasdaq)
        assert ftd.detected is True
        assert ftd.quality in (FTDQuality.STRONG, FTDQuality.MODERATE)
        assert ftd.gain_pct is not None
        assert ftd.gain_pct >= 1.25

    def test_no_ftd_in_rally_attempt(self):
        """REG-005: Day 3, gains under 1.25% — no FTD."""
        sp500 = get_index_price_history_mock("SP500", "rally_attempt")
        nasdaq = get_index_price_history_mock("NASDAQ", "rally_attempt")
        ftd = detect_follow_through_day(sp500, nasdaq)
        assert ftd.detected is False
        assert ftd.quality == FTDQuality.NONE

    def test_no_data_means_no_ftd(self):
        ftd = detect_follow_through_day(None, None)
        assert ftd.detected is False

    def test_empty_bars_no_ftd(self):
        ftd = detect_follow_through_day({"data": []}, {"data": []})
        assert ftd.detected is False


# ===================================================================
# Exposure Computation
# ===================================================================

@pytest.mark.schema
class TestComputeExposure:

    def test_correction_always_zero(self):
        dist = assess_distribution(get_distribution_days_mock("correction"))
        breadth = assess_breadth(get_market_breadth_mock("correction"))
        leaders = assess_leaders(get_leading_stocks_health_mock("correction"))
        sectors = assess_sectors(get_sector_rankings_mock("correction"))
        exp = compute_exposure(MarketRegime5.CORRECTION, dist, breadth, leaders, sectors)
        assert exp == 0

    def test_confirmed_uptrend_in_range(self):
        dist = assess_distribution(get_distribution_days_mock("healthy_uptrend"))
        breadth = assess_breadth(get_market_breadth_mock("healthy_uptrend"))
        leaders = assess_leaders(get_leading_stocks_health_mock("healthy_uptrend"))
        sectors = assess_sectors(get_sector_rankings_mock("healthy_uptrend"))
        exp = compute_exposure(MarketRegime5.CONFIRMED_UPTREND, dist, breadth, leaders, sectors)
        assert 80 <= exp <= 100

    def test_rally_attempt_in_range(self):
        dist = assess_distribution(get_distribution_days_mock("rally_attempt"))
        breadth = assess_breadth(get_market_breadth_mock("rally_attempt"))
        leaders = assess_leaders(get_leading_stocks_health_mock("rally_attempt"))
        sectors = assess_sectors(get_sector_rankings_mock("rally_attempt"))
        exp = compute_exposure(MarketRegime5.RALLY_ATTEMPT, dist, breadth, leaders, sectors)
        assert 0 <= exp <= 20


# ===================================================================
# Health Score
# ===================================================================

@pytest.mark.schema
class TestComputeHealthScore:

    @pytest.mark.parametrize("regime,lo,hi", [
        (MarketRegime5.CONFIRMED_UPTREND, 7, 10),
        (MarketRegime5.UPTREND_UNDER_PRESSURE, 4, 6),
        (MarketRegime5.CORRECTION, 1, 3),
        (MarketRegime5.RALLY_ATTEMPT, 2, 4),
        (MarketRegime5.FOLLOW_THROUGH_DAY, 4, 6),
    ])
    def test_health_score_in_regime_range(self, regime, lo, hi):
        score = compute_health_score(regime, bullish=3, bearish=1, breadth_divergence=False)
        assert lo <= score <= hi


# ===================================================================
# Confidence
# ===================================================================

@pytest.mark.schema
class TestComputeConfidence:

    def test_high_confidence(self):
        assert compute_confidence(bullish=4, bearish=0, data_unavailable=0) == Confidence.HIGH

    def test_medium_confidence(self):
        assert compute_confidence(bullish=3, bearish=1, data_unavailable=0) == Confidence.MEDIUM

    def test_low_confidence_from_missing_data(self):
        assert compute_confidence(bullish=4, bearish=0, data_unavailable=2) == Confidence.LOW

    def test_low_confidence_from_mixed_signals(self):
        assert compute_confidence(bullish=2, bearish=2, data_unavailable=0) == Confidence.LOW


# ===================================================================
# Transition Conditions
# ===================================================================

@pytest.mark.schema
class TestTransitionConditions:

    def test_confirmed_uptrend_has_downgrade(self):
        dist = assess_distribution(get_distribution_days_mock("healthy_uptrend"))
        breadth = assess_breadth(get_market_breadth_mock("healthy_uptrend"))
        leaders = assess_leaders(get_leading_stocks_health_mock("healthy_uptrend"))
        conditions = generate_transition_conditions(
            MarketRegime5.CONFIRMED_UPTREND, dist, breadth, leaders,
        )
        assert len(conditions) >= 1
        assert any(c.direction == "DOWNGRADE" for c in conditions)

    def test_correction_has_upgrade(self):
        dist = assess_distribution(get_distribution_days_mock("correction"))
        breadth = assess_breadth(get_market_breadth_mock("correction"))
        leaders = assess_leaders(get_leading_stocks_health_mock("correction"))
        conditions = generate_transition_conditions(
            MarketRegime5.CORRECTION, dist, breadth, leaders,
        )
        assert len(conditions) >= 1
        assert any(c.direction == "UPGRADE" for c in conditions)

    def test_rally_attempt_has_both(self):
        dist = assess_distribution(get_distribution_days_mock("rally_attempt"))
        breadth = assess_breadth(get_market_breadth_mock("rally_attempt"))
        leaders = assess_leaders(get_leading_stocks_health_mock("rally_attempt"))
        conditions = generate_transition_conditions(
            MarketRegime5.RALLY_ATTEMPT, dist, breadth, leaders,
        )
        assert any(c.direction == "UPGRADE" for c in conditions)
        assert any(c.direction == "DOWNGRADE" for c in conditions)

    def test_ftd_has_both_directions(self):
        dist = assess_distribution(get_distribution_days_mock("follow_through_day"))
        breadth = assess_breadth(get_market_breadth_mock("follow_through_day"))
        leaders = assess_leaders(get_leading_stocks_health_mock("follow_through_day"))
        conditions = generate_transition_conditions(
            MarketRegime5.FOLLOW_THROUGH_DAY, dist, breadth, leaders,
        )
        assert any(c.direction == "UPGRADE" for c in conditions)
        assert any(c.direction == "DOWNGRADE" for c in conditions)


# ===================================================================
# Decision Tree — classify_regime
# ===================================================================

@pytest.mark.behavior
class TestDecisionTree:

    def test_6_dist_forces_correction(self):
        """P1: 6+ distribution days = CORRECTION regardless."""
        out = classify_regime(
            dist_data=get_distribution_days_mock("correction"),
            breadth_data=get_market_breadth_mock("correction"),
            leader_data=get_leading_stocks_health_mock("correction"),
            sector_data=get_sector_rankings_mock("correction"),
            index_data_sp500=None,
            index_data_nasdaq=None,
            previous_regime="CONFIRMED_UPTREND",
        )
        assert out.regime == MarketRegime5.CORRECTION

    def test_5_dist_caps_at_under_pressure(self):
        """P2: 5 dist days = cannot be CONFIRMED_UPTREND."""
        out = classify_regime(
            dist_data=get_distribution_days_mock("under_pressure"),
            breadth_data=get_market_breadth_mock("under_pressure"),
            leader_data=get_leading_stocks_health_mock("under_pressure"),
            sector_data=get_sector_rankings_mock("under_pressure"),
            index_data_sp500=None,
            index_data_nasdaq=None,
            previous_regime="CONFIRMED_UPTREND",
        )
        assert out.regime != MarketRegime5.CONFIRMED_UPTREND
        assert out.regime == MarketRegime5.UPTREND_UNDER_PRESSURE

    def test_correction_stays_without_ftd(self):
        """MUST_NOT-S1: No FTD = no CONFIRMED_UPTREND after correction."""
        out = classify_regime(
            dist_data=get_distribution_days_mock("rally_attempt"),
            breadth_data=get_market_breadth_mock("rally_attempt"),
            leader_data=get_leading_stocks_health_mock("rally_attempt"),
            sector_data=get_sector_rankings_mock("rally_attempt"),
            index_data_sp500=get_index_price_history_mock("SP500", "rally_attempt"),
            index_data_nasdaq=get_index_price_history_mock("NASDAQ", "rally_attempt"),
            previous_regime="CORRECTION",
        )
        assert out.regime != MarketRegime5.CONFIRMED_UPTREND

    def test_ftd_produces_follow_through_day(self):
        """Valid FTD detected from RALLY_ATTEMPT = FOLLOW_THROUGH_DAY."""
        out = classify_regime(
            dist_data=get_distribution_days_mock("follow_through_day"),
            breadth_data=get_market_breadth_mock("follow_through_day"),
            leader_data=get_leading_stocks_health_mock("follow_through_day"),
            sector_data=get_sector_rankings_mock("follow_through_day"),
            index_data_sp500=get_index_price_history_mock("SP500", "follow_through_day"),
            index_data_nasdaq=get_index_price_history_mock("NASDAQ", "follow_through_day"),
            previous_regime="RALLY_ATTEMPT",
        )
        assert out.regime == MarketRegime5.FOLLOW_THROUGH_DAY
        assert out.follow_through_day.detected is True

    def test_confirmed_uptrend_stays_when_healthy(self):
        """Healthy signals in confirmed uptrend = stay confirmed."""
        out = classify_regime(
            dist_data=get_distribution_days_mock("healthy_uptrend"),
            breadth_data=get_market_breadth_mock("healthy_uptrend"),
            leader_data=get_leading_stocks_health_mock("healthy_uptrend"),
            sector_data=get_sector_rankings_mock("healthy_uptrend"),
            index_data_sp500=None,
            index_data_nasdaq=None,
            previous_regime="CONFIRMED_UPTREND",
        )
        assert out.regime == MarketRegime5.CONFIRMED_UPTREND


# ===================================================================
# Safety Rules (MUST_NOT)
# ===================================================================

@pytest.mark.behavior
class TestSafetyRules:

    def test_must_not_s1_no_uptrend_without_ftd(self):
        """MUST_NOT-S1: After correction, cannot jump to CONFIRMED_UPTREND without FTD."""
        out = classify_regime(
            dist_data={"sp500_dist_count": 1, "nasdaq_dist_count": 1,
                       "sp500_dist_days": [], "nasdaq_dist_days": [], "power_up_days": []},
            breadth_data=get_market_breadth_mock("healthy_uptrend"),
            leader_data=get_leading_stocks_health_mock("healthy_uptrend"),
            sector_data=get_sector_rankings_mock("healthy_uptrend"),
            index_data_sp500=get_index_price_history_mock("SP500", "rally_attempt"),  # No FTD
            index_data_nasdaq=get_index_price_history_mock("NASDAQ", "rally_attempt"),
            previous_regime="CORRECTION",
        )
        assert out.regime != MarketRegime5.CONFIRMED_UPTREND

    def test_must_not_s3_five_dist_blocks_uptrend(self):
        """MUST_NOT-S3: 5+ dist days = cannot be CONFIRMED_UPTREND."""
        out = classify_regime(
            dist_data={"sp500_dist_count": 5, "nasdaq_dist_count": 3,
                       "sp500_dist_days": [], "nasdaq_dist_days": [], "power_up_days": []},
            breadth_data=get_market_breadth_mock("healthy_uptrend"),
            leader_data=get_leading_stocks_health_mock("healthy_uptrend"),
            sector_data=get_sector_rankings_mock("healthy_uptrend"),
            index_data_sp500=None,
            index_data_nasdaq=None,
            previous_regime="CONFIRMED_UPTREND",
        )
        assert out.regime != MarketRegime5.CONFIRMED_UPTREND

    def test_must_not_s4_correction_exposure_zero(self):
        """MUST_NOT-S4: CORRECTION exposure must be 0."""
        out = classify_regime(
            dist_data=get_distribution_days_mock("correction"),
            breadth_data=get_market_breadth_mock("correction"),
            leader_data=get_leading_stocks_health_mock("correction"),
            sector_data=get_sector_rankings_mock("correction"),
            index_data_sp500=None,
            index_data_nasdaq=None,
            previous_regime="CONFIRMED_UPTREND",
        )
        assert out.exposure_recommendation == 0

    def test_must_not_s4_rally_exposure_max_20(self):
        """MUST_NOT-S4: RALLY_ATTEMPT exposure max 20%."""
        out = classify_regime(
            dist_data=get_distribution_days_mock("rally_attempt"),
            breadth_data=get_market_breadth_mock("rally_attempt"),
            leader_data=get_leading_stocks_health_mock("rally_attempt"),
            sector_data=get_sector_rankings_mock("rally_attempt"),
            index_data_sp500=get_index_price_history_mock("SP500", "rally_attempt"),
            index_data_nasdaq=get_index_price_history_mock("NASDAQ", "rally_attempt"),
            previous_regime="CORRECTION",
        )
        assert out.exposure_recommendation <= 20

    def test_must_not_s5_no_forecasting_in_summary(self):
        """MUST_NOT-S5: Executive summary must not predict bottoms/tops."""
        out = classify_regime(
            dist_data=get_distribution_days_mock("correction"),
            breadth_data=get_market_breadth_mock("correction"),
            leader_data=get_leading_stocks_health_mock("correction"),
            sector_data=get_sector_rankings_mock("correction"),
            index_data_sp500=None,
            index_data_nasdaq=None,
            previous_regime="CORRECTION",
        )
        summary_lower = out.executive_summary.lower()
        assert "bottom is in" not in summary_lower
        assert "worst is over" not in summary_lower
        assert "expect a recovery" not in summary_lower


# ===================================================================
# Data Unavailable (MUST-Q4)
# ===================================================================

@pytest.mark.behavior
class TestDataUnavailable:

    def test_breadth_error_lowers_confidence(self):
        """MUST-Q4: Error in breadth = DATA UNAVAILABLE, confidence LOW if 2+ fail."""
        out = classify_regime(
            dist_data=get_distribution_days_mock("healthy_uptrend"),
            breadth_data="ERROR: Service unavailable",
            leader_data="ERROR: Network timeout",
            sector_data=get_sector_rankings_mock("healthy_uptrend"),
            index_data_sp500=None,
            index_data_nasdaq=None,
            previous_regime="CONFIRMED_UPTREND",
        )
        assert out.confidence == Confidence.LOW
        assert "DATA UNAVAILABLE" in out.breadth.detail
        assert "DATA UNAVAILABLE" in out.leaders.detail

    def test_single_error_keeps_classification(self):
        """One error still allows classification."""
        out = classify_regime(
            dist_data=get_distribution_days_mock("healthy_uptrend"),
            breadth_data="ERROR: Service unavailable",
            leader_data=get_leading_stocks_health_mock("healthy_uptrend"),
            sector_data=get_sector_rankings_mock("healthy_uptrend"),
            index_data_sp500=None,
            index_data_nasdaq=None,
            previous_regime="CONFIRMED_UPTREND",
        )
        assert out.regime is not None
        assert "DATA UNAVAILABLE" in out.breadth.detail


# ===================================================================
# Pipeline Tests
# ===================================================================

@pytest.mark.schema
class TestRegimeDetectorPipeline:

    def test_default_scenario_produces_confirmed_uptrend(self, pipeline_output):
        assert pipeline_output.regime == MarketRegime5.CONFIRMED_UPTREND

    def test_output_is_valid_schema(self, pipeline_output):
        assert isinstance(pipeline_output, RegimeDetectorOutput)

    def test_all_scenarios_produce_valid_output(self, all_scenarios):
        for scenario, output in all_scenarios.items():
            assert isinstance(output, RegimeDetectorOutput), f"{scenario} failed"

    def test_scenario_regimes_match_expected(self, all_scenarios):
        expected = {
            "healthy_uptrend": MarketRegime5.CONFIRMED_UPTREND,
            "under_pressure": MarketRegime5.UPTREND_UNDER_PRESSURE,
            "correction": MarketRegime5.CORRECTION,
            "rally_attempt": MarketRegime5.RALLY_ATTEMPT,
            "follow_through_day": MarketRegime5.FOLLOW_THROUGH_DAY,
        }
        for scenario, exp_regime in expected.items():
            assert all_scenarios[scenario].regime == exp_regime, (
                f"{scenario}: expected {exp_regime}, got {all_scenarios[scenario].regime}"
            )

    def test_signal_counts_sum_to_5(self, all_scenarios):
        for scenario, output in all_scenarios.items():
            total = output.bullish_signals + output.neutral_signals + output.bearish_signals
            assert total == INDICATOR_COUNT, f"{scenario}: signals sum {total}"

    def test_exposure_within_regime_range(self, all_scenarios):
        for scenario, output in all_scenarios.items():
            lo, hi = EXPOSURE_RANGES[output.regime.value]
            assert lo <= output.exposure_recommendation <= hi, (
                f"{scenario}: exposure {output.exposure_recommendation} not in [{lo}, {hi}]"
            )

    def test_health_within_regime_range(self, all_scenarios):
        for scenario, output in all_scenarios.items():
            lo, hi = HEALTH_SCORE_RANGES[output.regime.value]
            assert lo <= output.market_health_score <= hi, (
                f"{scenario}: health {output.market_health_score} not in [{lo}, {hi}]"
            )

    def test_executive_summary_min_length(self, all_scenarios):
        for scenario, output in all_scenarios.items():
            assert len(output.executive_summary) >= 100, (
                f"{scenario}: summary too short ({len(output.executive_summary)} chars)"
            )

    def test_legacy_regime_valid(self, all_scenarios):
        for scenario, output in all_scenarios.items():
            legacy = output.to_legacy_regime()
            assert legacy in ("bull", "bear", "neutral"), (
                f"{scenario}: invalid legacy regime '{legacy}'"
            )


# ===================================================================
# Golden Dataset
# ===================================================================

@pytest.mark.schema
class TestGoldenDataset:

    def test_golden_file_exists(self, golden):
        assert golden is not None

    def test_default_regime(self, golden, pipeline_output):
        assert pipeline_output.regime.value == golden["regime"]

    def test_default_confidence(self, golden, pipeline_output):
        assert pipeline_output.confidence.value == golden["confidence"]

    def test_default_exposure_range(self, golden, pipeline_output):
        assert golden["exposure_min"] <= pipeline_output.exposure_recommendation <= golden["exposure_max"]

    def test_default_health_range(self, golden, pipeline_output):
        assert golden["health_score_min"] <= pipeline_output.market_health_score <= golden["health_score_max"]

    def test_default_bullish_signals(self, golden, pipeline_output):
        assert pipeline_output.bullish_signals >= golden["bullish_signals_min"]

    def test_default_bearish_signals(self, golden, pipeline_output):
        assert pipeline_output.bearish_signals <= golden["bearish_signals_max"]

    def test_default_legacy(self, golden, pipeline_output):
        assert pipeline_output.to_legacy_regime() == golden["legacy_regime"]

    def test_default_ftd(self, golden, pipeline_output):
        assert pipeline_output.follow_through_day.detected == golden["ftd_detected"]

    def test_default_regime_change(self, golden, pipeline_output):
        assert pipeline_output.regime_change.changed == golden["regime_change_changed"]

    def test_transition_conditions_min(self, golden, pipeline_output):
        assert len(pipeline_output.transition_conditions) >= golden["transition_conditions_min"]

    def test_summary_min_length(self, golden, pipeline_output):
        assert len(pipeline_output.executive_summary) >= golden["executive_summary_min_length"]

    def test_all_scenario_expectations(self, golden, all_scenarios):
        for scenario_name, expectations in golden.get("scenarios", {}).items():
            if scenario_name not in all_scenarios:
                continue
            output = all_scenarios[scenario_name]
            assert output.regime.value == expectations["regime"], (
                f"{scenario_name}: expected {expectations['regime']}, got {output.regime.value}"
            )
            assert output.to_legacy_regime() == expectations["legacy"], (
                f"{scenario_name}: expected legacy {expectations['legacy']}, got {output.to_legacy_regime()}"
            )
            assert output.exposure_recommendation >= expectations["exposure_min"], (
                f"{scenario_name}: exposure {output.exposure_recommendation} < {expectations['exposure_min']}"
            )
            assert output.exposure_recommendation <= expectations["exposure_max"], (
                f"{scenario_name}: exposure {output.exposure_recommendation} > {expectations['exposure_max']}"
            )


# ===================================================================
# End-to-End Chain
# ===================================================================

@pytest.mark.integration
class TestEndToEnd:

    def test_pipeline_chain_research_analyst_regime(self):
        """Research -> Analyst -> Regime Detector chain."""
        from ibd_agents.agents.regime_detector import run_regime_detector_pipeline

        # Run without analyst_output (standalone)
        output = run_regime_detector_pipeline(
            analyst_output=None,
            scenario="healthy_uptrend",
        )
        assert isinstance(output, RegimeDetectorOutput)
        assert output.regime in MarketRegime5
        assert output.to_legacy_regime() in ("bull", "bear", "neutral")

    def test_regime_to_legacy_feeds_downstream(self):
        """Verify legacy regime is compatible with downstream consumers."""
        from ibd_agents.agents.regime_detector import run_regime_detector_pipeline

        output = run_regime_detector_pipeline(scenario="correction", previous_regime="CORRECTION")
        legacy = output.to_legacy_regime()

        # Simulate downstream consumption
        from ibd_agents.tools.scenario_weighter import get_scenario_weights
        weights = get_scenario_weights(legacy)
        assert abs(sum(weights.values()) - 1.0) < 0.01
