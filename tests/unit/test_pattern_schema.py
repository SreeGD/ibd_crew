"""
Agent 12: PatternAlpha — Per-Stock Pattern Scoring Schema & Tool Tests
Level 1: Pure function tests + Pydantic validation, no LLM calls.

Tests pattern analysis tool functions from pattern_analyzer.py and
schema models from pattern_output.py.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ibd_agents.tools.pattern_analyzer import (
    compute_base_score,
    classify_enhanced_rating,
    generate_stock_pattern_alerts,
    assess_tier_fit,
)
from ibd_agents.schemas.pattern_output import (
    PATTERN_MAX_SCORES,
    ENHANCED_RATING_BANDS,
    ENHANCED_RATING_LABELS,
    PATTERN_ALERT_TYPES,
    TIER_PATTERN_REQUIREMENTS,
    PatternSubScore,
    PatternScoreBreakdown,
    BaseScoreBreakdown,
    EnhancedStockAnalysis,
    PatternAlert,
    PortfolioPatternSummary,
    PortfolioPatternOutput,
)


# ---------------------------------------------------------------------------
# compute_base_score tests
# ---------------------------------------------------------------------------


@pytest.mark.schema
class TestComputeBaseScore:
    """Test compute_base_score() deterministic scoring from pattern_analyzer.py."""

    def test_base_score_t1_stock(self):
        """High-rated T1 stock should produce base_total in 85-100 range."""
        stock = {
            "composite_rating": 97,
            "rs_rating": 95,
            "eps_rating": 90,
            "validation_score": 8,
            "is_ibd_keep": True,
            "is_multi_source_validated": True,
            "sharpe_category": "Excellent",
            "alpha_category": "Strong Outperformer",
            "risk_rating": "Excellent",
        }
        result = compute_base_score(stock)
        assert 85 <= result["base_total"] <= 100, (
            f"T1 stock base_total={result['base_total']} not in 85-100"
        )

    def test_base_score_t3_stock(self):
        """Moderate-rated T3 stock should produce base_total in 30-60 range."""
        stock = {
            "composite_rating": 82,
            "rs_rating": 78,
            "eps_rating": 72,
            "validation_score": 3,
            "is_ibd_keep": False,
            "is_multi_source_validated": False,
            "sharpe_category": "Moderate",
            "alpha_category": "Slight Underperformer",
            "risk_rating": "Moderate",
        }
        result = compute_base_score(stock)
        assert 30 <= result["base_total"] <= 60, (
            f"T3 stock base_total={result['base_total']} not in 30-60"
        )

    def test_base_score_minimal(self):
        """Missing keys default gracefully; still produces base_total 0-100."""
        stock = {}
        result = compute_base_score(stock)
        assert 0 <= result["base_total"] <= 100, (
            f"Minimal stock base_total={result['base_total']} not in 0-100"
        )

    def test_base_score_has_required_keys(self):
        """Return dict has exactly keys ibd_score, analyst_score, risk_score, base_total."""
        stock = {
            "composite_rating": 90,
            "rs_rating": 85,
            "eps_rating": 80,
        }
        result = compute_base_score(stock)
        expected_keys = {"ibd_score", "analyst_score", "risk_score", "base_total"}
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# classify_enhanced_rating tests
# ---------------------------------------------------------------------------


@pytest.mark.schema
class TestClassifyEnhancedRating:
    """Test classify_enhanced_rating() star/label mapping."""

    def test_rating_max_conviction(self):
        """Score 135 -> Max Conviction band."""
        stars, label = classify_enhanced_rating(135)
        assert stars == "★★★★★+"
        assert label == "Max Conviction"

    def test_rating_strong(self):
        """Score 120 -> Strong band."""
        stars, label = classify_enhanced_rating(120)
        assert stars == "★★★★★"
        assert label == "Strong"

    def test_rating_favorable(self):
        """Score 107 -> Favorable band."""
        stars, label = classify_enhanced_rating(107)
        assert stars == "★★★★"
        assert label == "Favorable"

    def test_rating_notable(self):
        """Score 92 -> Notable band."""
        stars, label = classify_enhanced_rating(92)
        assert stars == "★★★"
        assert label == "Notable"

    def test_rating_monitor(self):
        """Score 77 -> Monitor band."""
        stars, label = classify_enhanced_rating(77)
        assert stars == "★★"
        assert label == "Monitor"

    def test_rating_review(self):
        """Score 50 -> Review band (lowest)."""
        stars, label = classify_enhanced_rating(50)
        assert stars == "★"
        assert label == "Review"

    def test_rating_zero(self):
        """Score 0 -> Review band (lowest)."""
        stars, label = classify_enhanced_rating(0)
        assert stars == "★"
        assert label == "Review"


# ---------------------------------------------------------------------------
# generate_stock_pattern_alerts tests
# ---------------------------------------------------------------------------


@pytest.mark.schema
class TestGenerateStockPatternAlerts:
    """Test generate_stock_pattern_alerts() per-stock alert generation."""

    def test_category_king_alert(self):
        """P4 >= 8 triggers 'Category King' alert."""
        alerts = generate_stock_pattern_alerts(
            "TEST", {"p1": 5, "p2": 5, "p3": 5, "p4": 9, "p5": 4}
        )
        assert "Category King" in alerts

    def test_inflection_alert(self):
        """P5 >= 6 triggers 'Inflection Alert'."""
        alerts = generate_stock_pattern_alerts(
            "TEST", {"p1": 5, "p2": 5, "p3": 5, "p4": 5, "p5": 7}
        )
        assert "Inflection Alert" in alerts

    def test_disruption_risk_alert(self):
        """P2 = 0 triggers 'Disruption Risk'."""
        alerts = generate_stock_pattern_alerts(
            "TEST", {"p1": 5, "p2": 0, "p3": 5, "p4": 5, "p5": 4}
        )
        assert "Disruption Risk" in alerts

    def test_pattern_imbalance(self):
        """Any pattern = 0 while another >= 8 triggers 'Pattern Imbalance'."""
        alerts = generate_stock_pattern_alerts(
            "TEST", {"p1": 10, "p2": 0, "p3": 5, "p4": 5, "p5": 4}
        )
        assert "Pattern Imbalance" in alerts

    def test_no_alerts(self):
        """All moderate scores (no extremes) -> empty alert list."""
        alerts = generate_stock_pattern_alerts(
            "TEST", {"p1": 5, "p2": 5, "p3": 5, "p4": 5, "p5": 4}
        )
        assert alerts == []


# ---------------------------------------------------------------------------
# assess_tier_fit tests
# ---------------------------------------------------------------------------


@pytest.mark.schema
class TestAssessTierFit:
    """Test assess_tier_fit() tier pattern requirement assessments."""

    def test_tier_1_strong_fit(self):
        """Enhanced >= 115 and 3 patterns at 7+ -> 'Strong fit' for T1."""
        breakdown = {"p1": 8, "p2": 7, "p3": 7, "p4": 5, "p5": 4}
        result = assess_tier_fit(120, breakdown, current_tier=1)
        assert "Strong fit" in result

    def test_tier_1_gap(self):
        """Enhanced < 115 -> 'Gap' for T1."""
        breakdown = {"p1": 5, "p2": 5, "p3": 5, "p4": 5, "p5": 4}
        result = assess_tier_fit(100, breakdown, current_tier=1)
        assert "Gap" in result

    def test_tier_2_strong_fit(self):
        """Enhanced >= 95 and 3 patterns at 5+ -> 'Strong fit' for T2."""
        breakdown = {"p1": 6, "p2": 5, "p3": 5, "p4": 4, "p5": 3}
        result = assess_tier_fit(100, breakdown, current_tier=2)
        assert "Strong fit" in result

    def test_tier_3_strong_fit(self):
        """Enhanced >= 80 and P3 >= 7 -> 'Strong fit' for T3."""
        breakdown = {"p1": 4, "p2": 3, "p3": 8, "p4": 3, "p5": 2}
        result = assess_tier_fit(85, breakdown, current_tier=3)
        assert "Strong fit" in result

    def test_tier_fit_no_breakdown(self):
        """breakdown=None -> 'unavailable' in result."""
        result = assess_tier_fit(80, None, current_tier=1)
        assert "unavailable" in result.lower()


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


@pytest.mark.schema
class TestPatternSchemaValidation:
    """Test Pydantic model validators on pattern output models."""

    def test_pattern_sub_score_valid(self):
        """Valid PatternSubScore instantiates without error."""
        sub = PatternSubScore(
            pattern_name="Platform Economics",
            score=8,
            max_score=12,
            justification="Strong network effects",
        )
        assert sub.score == 8
        assert sub.max_score == 12

    def test_pattern_sub_score_exceeds_max(self):
        """score > max_score raises ValidationError."""
        with pytest.raises(ValidationError, match="exceeds max_score"):
            PatternSubScore(
                pattern_name="Platform Economics",
                score=15,
                max_score=12,
                justification="Invalid score",
            )

    def test_base_score_breakdown_valid(self):
        """Valid BaseScoreBreakdown where base_total = round(sum of components)."""
        bsb = BaseScoreBreakdown(
            ibd_score=30.0,
            analyst_score=20.0,
            risk_score=15.0,
            base_total=65,
        )
        assert bsb.base_total == 65

    def test_base_score_breakdown_mismatch(self):
        """base_total != round(ibd + analyst + risk) raises ValidationError."""
        with pytest.raises(ValidationError, match="base_total"):
            BaseScoreBreakdown(
                ibd_score=30.0,
                analyst_score=20.0,
                risk_score=15.0,
                base_total=70,  # Should be 65
            )

    def test_enhanced_stock_analysis_valid(self):
        """Full valid EnhancedStockAnalysis instantiates without error."""
        base = BaseScoreBreakdown(
            ibd_score=30.0,
            analyst_score=20.0,
            risk_score=15.0,
            base_total=65,
        )
        esa = EnhancedStockAnalysis(
            symbol="NVDA",
            company_name="NVIDIA Corporation",
            sector="CHIPS",
            tier=1,
            base_score=base,
            pattern_score=None,
            enhanced_score=65,  # base_total + 0 (no pattern)
            enhanced_rating="★",
            enhanced_rating_label="Review",
            pattern_alerts=[],
            tier_recommendation="Tier 1: Gap",
            scoring_source="deterministic",
        )
        assert esa.symbol == "NVDA"
        assert esa.enhanced_score == 65

    def test_enhanced_score_mismatch(self):
        """enhanced_score != base_total + pattern_total raises ValidationError."""
        base = BaseScoreBreakdown(
            ibd_score=30.0,
            analyst_score=20.0,
            risk_score=15.0,
            base_total=65,
        )
        with pytest.raises(ValidationError, match="enhanced_score"):
            EnhancedStockAnalysis(
                symbol="NVDA",
                company_name="NVIDIA Corporation",
                sector="CHIPS",
                tier=1,
                base_score=base,
                pattern_score=None,
                enhanced_score=100,  # Should be 65 (base only)
                enhanced_rating="★★★",
                enhanced_rating_label="Notable",
                pattern_alerts=[],
                tier_recommendation="Tier 1: Gap",
                scoring_source="deterministic",
            )
