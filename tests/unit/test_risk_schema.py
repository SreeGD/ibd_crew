"""
Agent 06: Risk Officer — Schema Tests
Level 1: Pure Pydantic validation, no LLM, no file I/O.
"""

from __future__ import annotations

import pytest

from ibd_agents.schemas.risk_output import (
    CHECK_NAMES,
    SLEEP_WELL_RANGE,
    STRESS_SCENARIOS,
    VALID_CHECK_STATUS,
    VALID_OVERALL_STATUS,
    VALID_SEVERITY,
    KeepValidation,
    RiskAssessment,
    RiskCheck,
    RiskWarning,
    SleepWellScores,
    StopLossRecommendation,
    StressScenario,
    StressTestReport,
    Veto,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_check(name: str, status: str = "PASS") -> RiskCheck:
    return RiskCheck(
        check_name=name,
        status=status,
        findings=f"{name} check completed successfully with no issues found",
        details=[],
    )


def _make_all_checks(status: str = "PASS") -> list[RiskCheck]:
    return [_make_check(name, status) for name in CHECK_NAMES]


def _make_stress_report() -> StressTestReport:
    return StressTestReport(
        scenarios=[
            StressScenario(
                scenario_name="Market Crash",
                impact_description="Broad market decline of 20% with momentum names hit hardest",
                estimated_drawdown_pct=18.0,
                positions_most_affected=["NVDA", "CLS"],
            ),
            StressScenario(
                scenario_name="Sector Correction",
                impact_description="Leading sector corrects 30% with concentrated positions affected",
                estimated_drawdown_pct=12.0,
                positions_most_affected=["MU", "AVGO"],
            ),
            StressScenario(
                scenario_name="Rate Hike",
                impact_description="Unexpected 50bp rate increase impacts growth names",
                estimated_drawdown_pct=6.0,
                positions_most_affected=["PLTR", "APP"],
            ),
        ],
        overall_resilience="Moderate resilience: diversified but momentum-heavy portfolio",
    )


def _make_sleep_well() -> SleepWellScores:
    return SleepWellScores(
        tier_1_score=7,
        tier_2_score=8,
        tier_3_score=9,
        overall_score=8,
        factors=["Strong diversification", "All keeps present"],
    )


def _make_keep_validation() -> KeepValidation:
    return KeepValidation(
        total_keeps_found=14,
        missing_keeps=[],
        status="PASS",
    )


def _make_risk_assessment(**overrides) -> RiskAssessment:
    defaults = dict(
        check_results=_make_all_checks(),
        vetoes=[],
        warnings=[],
        stress_test_results=_make_stress_report(),
        sleep_well_scores=_make_sleep_well(),
        keep_validation=_make_keep_validation(),
        overall_status="APPROVED",
        analysis_date="2025-01-15",
        summary="Risk Assessment: APPROVED. 11/11 checks passed. Sleep Well: 8/10. Portfolio within all framework limits.",
    )
    defaults.update(overrides)
    return RiskAssessment(**defaults)


# ---------------------------------------------------------------------------
# Constant Tests
# ---------------------------------------------------------------------------

class TestConstants:

    @pytest.mark.schema
    def test_11_check_names(self):
        """Exactly 11 risk check names."""
        assert len(CHECK_NAMES) == 11

    @pytest.mark.schema
    def test_3_check_statuses(self):
        """PASS, WARNING, VETO statuses."""
        assert set(VALID_CHECK_STATUS) == {"PASS", "WARNING", "VETO"}

    @pytest.mark.schema
    def test_3_severity_levels(self):
        """LOW, MEDIUM, HIGH severity."""
        assert set(VALID_SEVERITY) == {"LOW", "MEDIUM", "HIGH"}

    @pytest.mark.schema
    def test_3_overall_statuses(self):
        """APPROVED, CONDITIONAL, REJECTED."""
        assert set(VALID_OVERALL_STATUS) == {"APPROVED", "CONDITIONAL", "REJECTED"}

    @pytest.mark.schema
    def test_3_stress_scenarios(self):
        """3 stress scenarios defined."""
        assert len(STRESS_SCENARIOS) == 3

    @pytest.mark.schema
    def test_sleep_well_range(self):
        """Sleep Well range is 1-10."""
        assert SLEEP_WELL_RANGE == (1, 10)


# ---------------------------------------------------------------------------
# RiskCheck Tests
# ---------------------------------------------------------------------------

class TestRiskCheck:

    @pytest.mark.schema
    def test_valid_check(self):
        c = _make_check("position_sizing", "PASS")
        assert c.status == "PASS"

    @pytest.mark.schema
    def test_invalid_check_name_fails(self):
        with pytest.raises(ValueError, match="check_name"):
            RiskCheck(
                check_name="invalid_check",
                status="PASS",
                findings="This check name does not exist in the framework",
                details=[],
            )

    @pytest.mark.schema
    def test_invalid_status_fails(self):
        with pytest.raises(ValueError, match="status"):
            RiskCheck(
                check_name="position_sizing",
                status="FAIL",
                findings="Invalid status value used for risk check",
                details=[],
            )


# ---------------------------------------------------------------------------
# Veto Tests
# ---------------------------------------------------------------------------

class TestVeto:

    @pytest.mark.schema
    def test_valid_veto(self):
        v = Veto(
            check_name="position_sizing",
            reason="Position exceeds limit",
            required_fix="Reduce NVDA position from 6% to 5% maximum in T1 to comply with framework position sizing limits",
        )
        assert v.check_name == "position_sizing"

    @pytest.mark.schema
    def test_short_required_fix_fails(self):
        with pytest.raises(ValueError):
            Veto(
                check_name="position_sizing",
                reason="Position exceeds limit",
                required_fix="Fix it",
            )


# ---------------------------------------------------------------------------
# SleepWellScores Tests
# ---------------------------------------------------------------------------

class TestSleepWellScores:

    @pytest.mark.schema
    def test_valid_scores(self):
        s = _make_sleep_well()
        assert 1 <= s.overall_score <= 10

    @pytest.mark.schema
    def test_score_below_1_fails(self):
        with pytest.raises(ValueError):
            SleepWellScores(
                tier_1_score=0, tier_2_score=5,
                tier_3_score=5, overall_score=3,
            )

    @pytest.mark.schema
    def test_score_above_10_fails(self):
        with pytest.raises(ValueError):
            SleepWellScores(
                tier_1_score=11, tier_2_score=5,
                tier_3_score=5, overall_score=3,
            )


# ---------------------------------------------------------------------------
# RiskAssessment Tests
# ---------------------------------------------------------------------------

class TestRiskAssessment:

    @pytest.mark.schema
    def test_valid_approved(self):
        ra = _make_risk_assessment()
        assert ra.overall_status == "APPROVED"

    @pytest.mark.schema
    def test_valid_conditional(self):
        ra = _make_risk_assessment(
            warnings=[RiskWarning(
                check_name="correlation",
                severity="MEDIUM",
                description="Potential correlation cluster in CHIPS sector detected",
                suggestion="Consider reducing CHIPS sector exposure for better diversification",
            )],
            overall_status="CONDITIONAL",
        )
        assert ra.overall_status == "CONDITIONAL"

    @pytest.mark.schema
    def test_valid_rejected(self):
        ra = _make_risk_assessment(
            vetoes=[Veto(
                check_name="position_sizing",
                reason="Position exceeds limit",
                required_fix="Reduce oversized position to comply with framework T1 stock limit of 5% maximum allocation",
            )],
            overall_status="REJECTED",
        )
        assert ra.overall_status == "REJECTED"

    @pytest.mark.schema
    def test_missing_check_fails(self):
        """Missing a check name fails validation."""
        incomplete = [_make_check(n) for n in CHECK_NAMES[:10]]
        incomplete.append(_make_check(CHECK_NAMES[0]))  # Duplicate instead of last
        with pytest.raises(ValueError, match="Missing risk checks"):
            _make_risk_assessment(check_results=incomplete)

    @pytest.mark.schema
    def test_vetoes_require_rejected(self):
        """Having vetoes but not REJECTED fails."""
        with pytest.raises(ValueError, match="vetoes"):
            _make_risk_assessment(
                vetoes=[Veto(
                    check_name="position_sizing",
                    reason="Position exceeds limit",
                    required_fix="Reduce oversized position to comply with framework T1 stock limit of 5% maximum allocation",
                )],
                overall_status="APPROVED",
            )

    @pytest.mark.schema
    def test_no_issues_require_approved(self):
        """No vetoes or warnings but not APPROVED fails."""
        with pytest.raises(ValueError, match="not APPROVED"):
            _make_risk_assessment(overall_status="CONDITIONAL")

    @pytest.mark.schema
    def test_summary_min_length(self):
        ra = _make_risk_assessment()
        assert len(ra.summary) >= 50

    @pytest.mark.schema
    def test_exactly_11_checks(self):
        ra = _make_risk_assessment()
        assert len(ra.check_results) == 11


# ---------------------------------------------------------------------------
# StopLossRecommendation Tests
# ---------------------------------------------------------------------------

class TestStopLossRecommendation:

    @pytest.mark.schema
    def test_valid_recommendation(self):
        rec = StopLossRecommendation(
            symbol="NVDA",
            current_stop_pct=22.0,
            recommended_stop_pct=18.5,
            reason="High-beta semiconductor with elevated volatility profile.",
            volatility_flag="high",
        )
        assert rec.symbol == "NVDA"
        assert rec.recommended_stop_pct == 18.5

    @pytest.mark.schema
    def test_symbol_uppercased(self):
        rec = StopLossRecommendation(
            symbol="nvda",
            current_stop_pct=22.0,
            recommended_stop_pct=15.0,
            reason="Symbol should be uppercased by validator.",
        )
        assert rec.symbol == "NVDA"

    @pytest.mark.schema
    def test_stop_below_5_fails(self):
        with pytest.raises(ValueError):
            StopLossRecommendation(
                symbol="AAPL",
                current_stop_pct=22.0,
                recommended_stop_pct=3.0,
                reason="Stop below minimum threshold of 5 percent.",
            )

    @pytest.mark.schema
    def test_stop_above_35_fails(self):
        with pytest.raises(ValueError):
            StopLossRecommendation(
                symbol="AAPL",
                current_stop_pct=22.0,
                recommended_stop_pct=40.0,
                reason="Stop above maximum threshold of 35 percent.",
            )

    @pytest.mark.schema
    def test_short_reason_fails(self):
        with pytest.raises(ValueError):
            StopLossRecommendation(
                symbol="AAPL",
                current_stop_pct=22.0,
                recommended_stop_pct=18.0,
                reason="Short",
            )

    @pytest.mark.schema
    def test_valid_volatility_flags(self):
        for flag in ("high", "normal", "low"):
            rec = StopLossRecommendation(
                symbol="TEST",
                current_stop_pct=18.0,
                recommended_stop_pct=15.0,
                reason=f"Testing {flag} volatility flag value.",
                volatility_flag=flag,
            )
            assert rec.volatility_flag == flag

    @pytest.mark.schema
    def test_invalid_volatility_flag_fails(self):
        with pytest.raises(ValueError, match="volatility_flag"):
            StopLossRecommendation(
                symbol="TEST",
                current_stop_pct=18.0,
                recommended_stop_pct=15.0,
                reason="Testing invalid volatility flag value.",
                volatility_flag="extreme",
            )

    @pytest.mark.schema
    def test_none_volatility_flag_accepted(self):
        rec = StopLossRecommendation(
            symbol="TEST",
            current_stop_pct=18.0,
            recommended_stop_pct=15.0,
            reason="Testing with no volatility flag set.",
            volatility_flag=None,
        )
        assert rec.volatility_flag is None


# ---------------------------------------------------------------------------
# Stop-Loss Source Field Tests
# ---------------------------------------------------------------------------

class TestStopLossSourceField:

    @pytest.mark.schema
    def test_stop_loss_source_defaults_to_none(self):
        ra = _make_risk_assessment()
        assert ra.stop_loss_source is None

    @pytest.mark.schema
    def test_stop_loss_source_llm_accepted(self):
        ra = _make_risk_assessment(stop_loss_source="llm")
        assert ra.stop_loss_source == "llm"

    @pytest.mark.schema
    def test_stop_loss_source_deterministic_accepted(self):
        ra = _make_risk_assessment(stop_loss_source="deterministic")
        assert ra.stop_loss_source == "deterministic"

    @pytest.mark.schema
    def test_stop_loss_source_invalid_rejected(self):
        with pytest.raises(ValueError, match="stop_loss_source"):
            _make_risk_assessment(stop_loss_source="other")

    @pytest.mark.schema
    def test_stop_loss_recommendations_default_empty(self):
        ra = _make_risk_assessment()
        assert ra.stop_loss_recommendations == []

    @pytest.mark.schema
    def test_stop_loss_recommendations_populated(self):
        recs = [
            StopLossRecommendation(
                symbol="NVDA",
                current_stop_pct=22.0,
                recommended_stop_pct=18.0,
                reason="High-beta stock needs tighter trailing stop.",
                volatility_flag="high",
            ),
        ]
        ra = _make_risk_assessment(stop_loss_recommendations=recs, stop_loss_source="llm")
        assert len(ra.stop_loss_recommendations) == 1
        assert ra.stop_loss_recommendations[0].symbol == "NVDA"
