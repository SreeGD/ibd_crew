"""
Agent 11: Value Investor — Schema & Scoring Tests
Level 1: Pure function tests + Pydantic validation, no LLM calls.

Tests value scoring functions from value_screener.py and
schema models from value_investor_output.py.
"""

from __future__ import annotations

import pytest

from ibd_agents.tools.value_screener import (
    compute_morningstar_score,
    compute_pe_value_score,
    compute_peg_value_score,
    compute_moat_quality_score,
    compute_discount_score,
    compute_value_score,
    classify_value_category,
    detect_value_trap,
    assess_momentum_value_alignment,
)
from ibd_agents.schemas.value_investor_output import (
    VALUE_CATEGORIES,
    ValueStock,
    ValueInvestorOutput,
    SectorValueAnalysis,
    ValueCategorySummary,
    MoatAnalysisSummary,
)


# ---------------------------------------------------------------------------
# Morningstar Score Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestMorningstarScore:
    """Test compute_morningstar_score() component scoring."""

    def test_morningstar_score_5star_wide_moat_max(self):
        """5-star(40) + Wide(30) + P/FV 0.5 discount(20) + Low(10) = 100."""
        result = compute_morningstar_score("5-star", "Wide", 0.5, "Low")
        assert result["star_points"] == 40
        assert result["moat_points"] == 30
        assert result["discount_points"] == 20
        assert result["certainty_points"] == 10
        assert result["morningstar_score"] == 100

    def test_morningstar_score_no_data(self):
        """All None inputs produce score of 0."""
        result = compute_morningstar_score(None, None, None, None)
        assert result["morningstar_score"] == 0
        assert result["star_points"] == 0
        assert result["moat_points"] == 0
        assert result["discount_points"] == 0
        assert result["certainty_points"] == 0

    def test_morningstar_score_4star_narrow(self):
        """4-star(25) + Narrow(15) + Medium(7) = 47."""
        result = compute_morningstar_score("4-star", "Narrow", None, "Medium")
        assert result["star_points"] == 25
        assert result["moat_points"] == 15
        assert result["certainty_points"] == 7
        assert result["discount_points"] == 0
        assert result["morningstar_score"] == 47


# ---------------------------------------------------------------------------
# P/E Value Score Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestPEValueScore:
    """Test compute_pe_value_score() category mapping."""

    def test_pe_value_score_deep_value(self):
        """Deep Value category scores 100."""
        score = compute_pe_value_score(5.0, "Deep Value", "CHIPS")
        assert score == 100.0

    def test_pe_value_score_reasonable(self):
        """Reasonable category scores 50."""
        score = compute_pe_value_score(20.0, "Reasonable", "SOFTWARE")
        assert score == 50.0

    def test_pe_value_score_speculative(self):
        """Speculative category scores 0."""
        score = compute_pe_value_score(200.0, "Speculative", "INTERNET")
        assert score == 0.0

    def test_pe_value_score_none(self):
        """None pe_category scores 0."""
        score = compute_pe_value_score(None, None, "CHIPS")
        assert score == 0.0


# ---------------------------------------------------------------------------
# PEG Value Score Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestPEGValueScore:
    """Test compute_peg_value_score() PEG ratio scoring."""

    def test_peg_value_score_below_05(self):
        """PEG 0.3 (below 0.5) scores 100."""
        score = compute_peg_value_score(0.3, None)
        assert score == 100.0

    def test_peg_value_score_1_to_15(self):
        """PEG 1.2 (1.0-1.5 range) scores 60."""
        score = compute_peg_value_score(1.2, None)
        assert score == 60.0

    def test_peg_value_score_above_2(self):
        """PEG 3.0 (above 2.0) scores 10."""
        score = compute_peg_value_score(3.0, None)
        assert score == 10.0

    def test_peg_value_score_none(self):
        """None PEG ratio scores 0."""
        score = compute_peg_value_score(None, None)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Moat Quality Score Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestMoatQualityScore:
    """Test compute_moat_quality_score() moat/quality scoring."""

    def test_moat_quality_wide_5star_smrA(self):
        """Wide(80) + 5-star(+10) + SMR A(+10) = 100."""
        score = compute_moat_quality_score("Wide", "5-star", "A")
        assert score == 100.0

    def test_moat_quality_narrow_no_bonuses(self):
        """Narrow moat with no bonuses = 40."""
        score = compute_moat_quality_score("Narrow", None, None)
        assert score == 40.0

    def test_moat_quality_none(self):
        """No moat with no bonuses = 0."""
        score = compute_moat_quality_score(None, None, None)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Discount Score Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestDiscountScore:
    """Test compute_discount_score() price-to-fair-value scoring."""

    def test_discount_score_50pct_below(self):
        """P/FV 0.5 (50% below fair value) scores 100."""
        score = compute_discount_score(0.5)
        assert score == 100.0

    def test_discount_score_at_fair_value(self):
        """P/FV 1.0 (at fair value) scores 0."""
        score = compute_discount_score(1.0)
        assert score == 0.0

    def test_discount_score_above_fair_value(self):
        """P/FV 1.2 (above fair value) scores 0."""
        score = compute_discount_score(1.2)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Composite Value Score Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestCompositeValueScore:
    """Test compute_value_score() weighted composite calculation."""

    def test_value_score_all_max(self):
        """All components at 100 -> composite = 100."""
        score = compute_value_score(100, 100, 100, 100, 100)
        assert score == 100.0

    def test_value_score_all_zero(self):
        """All components at 0 -> composite = 0."""
        score = compute_value_score(0, 0, 0, 0, 0)
        assert score == 0.0

    def test_value_score_weight_redistribution(self):
        """When M* and discount are 0, weight shifts to PE/PEG."""
        # M*=0, discount=0 -> redistributed: PE 42.5%, PEG 42.5%, Moat 15%
        # PE=100, PEG=100, Moat=0 -> 42.5 + 42.5 + 0 = 85.0
        score = compute_value_score(0, 100, 100, 0, 0)
        assert score == 85.0


# ---------------------------------------------------------------------------
# Category Classification Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestCategoryClassification:
    """Test classify_value_category() category assignment logic."""

    def test_quality_value(self):
        """Wide moat + 5-star + P/FV=0.90 -> Quality Value."""
        cat, _ = classify_value_category(
            pe_category="Reasonable",
            peg_ratio=1.5,
            peg_category="Fair Value",
            rs_rating=85,
            eps_rating=90,
            economic_moat="Wide",
            morningstar_rating="5-star",
            price_to_fair_value=0.90,
            llm_dividend_yield=None,
        )
        assert cat == "Quality Value"

    def test_deep_value_pe(self):
        """pe_category='Deep Value' -> Deep Value."""
        cat, _ = classify_value_category(
            pe_category="Deep Value",
            peg_ratio=0.8,
            peg_category="Undervalued",
            rs_rating=60,
            eps_rating=70,
            economic_moat=None,
            morningstar_rating=None,
            price_to_fair_value=None,
            llm_dividend_yield=None,
        )
        assert cat == "Deep Value"

    def test_deep_value_pfv(self):
        """P/FV 0.70 (< 0.80) -> Deep Value."""
        cat, _ = classify_value_category(
            pe_category="Value",
            peg_ratio=1.0,
            peg_category="Fair Value",
            rs_rating=70,
            eps_rating=75,
            economic_moat=None,
            morningstar_rating=None,
            price_to_fair_value=0.70,
            llm_dividend_yield=None,
        )
        assert cat == "Deep Value"

    def test_garp(self):
        """PEG 1.5, PE 'Reasonable', RS 85, EPS 80 -> GARP."""
        cat, _ = classify_value_category(
            pe_category="Reasonable",
            peg_ratio=1.5,
            peg_category="Fair Value",
            rs_rating=85,
            eps_rating=80,
            economic_moat=None,
            morningstar_rating=None,
            price_to_fair_value=None,
            llm_dividend_yield=None,
        )
        assert cat == "GARP"

    def test_not_value_default(self):
        """Growth Premium, no moat -> Not Value."""
        cat, _ = classify_value_category(
            pe_category="Growth Premium",
            peg_ratio=3.5,
            peg_category="Expensive",
            rs_rating=95,
            eps_rating=99,
            economic_moat=None,
            morningstar_rating=None,
            price_to_fair_value=None,
            llm_dividend_yield=None,
        )
        assert cat == "Not Value"


# ---------------------------------------------------------------------------
# Value Trap Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestValueTrapDetection:
    """Test detect_value_trap() risk signal detection."""

    def test_value_trap_high_risk(self):
        """RS=40, EPS=50, SMR='D' -> 'High' with 3 signals."""
        risk, signals = detect_value_trap(
            pe_category="Deep Value",
            rs_rating=40,
            eps_rating=50,
            smr_rating="D",
            acc_dis_rating=None,
            price_to_fair_value=None,
            value_category="Deep Value",
        )
        assert risk == "High"
        assert len(signals) == 3

    def test_value_trap_none_strong(self):
        """RS=90, EPS=85, SMR='A' -> 'None' (no signals)."""
        risk, signals = detect_value_trap(
            pe_category="Value",
            rs_rating=90,
            eps_rating=85,
            smr_rating="A",
            acc_dis_rating="B",
            price_to_fair_value=0.90,
            value_category="Quality Value",
        )
        assert risk == "None"
        assert len(signals) == 0

    def test_value_trap_not_value_exempt(self):
        """category='Not Value' -> 'None' always, even with bad ratings."""
        risk, signals = detect_value_trap(
            pe_category="Speculative",
            rs_rating=30,
            eps_rating=20,
            smr_rating="E",
            acc_dis_rating="E",
            price_to_fair_value=None,
            value_category="Not Value",
        )
        assert risk == "None"
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Momentum-Value Alignment Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestMomentumValueAlignment:
    """Test assess_momentum_value_alignment() classification."""

    def test_aligned_high_both(self):
        """RS=90, value_score=70 -> 'Aligned'."""
        label, _ = assess_momentum_value_alignment(90, 70.0, "GARP")
        assert label == "Aligned"

    def test_strong_mismatch(self):
        """RS=40, value_score=70 -> 'Strong Mismatch'."""
        label, _ = assess_momentum_value_alignment(40, 70.0, "Deep Value")
        assert label == "Strong Mismatch"

    def test_mild_mismatch(self):
        """RS=90, value_score=20 -> 'Mild Mismatch'."""
        label, _ = assess_momentum_value_alignment(90, 20.0, "Not Value")
        assert label == "Mild Mismatch"
