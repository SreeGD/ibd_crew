"""
Agent 09: Educator — Schema Tests
Level 1: Pure Pydantic validation, no LLM, no file I/O.
"""

from __future__ import annotations

import pytest

from ibd_agents.schemas.educator_output import (
    REQUIRED_GLOSSARY_TERMS,
    IBD_CONCEPTS_TO_TEACH,
    MIN_STOCK_EXPLANATIONS,
    MIN_CONCEPT_LESSONS,
    MIN_GLOSSARY_ENTRIES,
    MIN_ACTION_ITEMS,
    MAX_ACTION_ITEMS,
    TierGuide,
    KeepExplanations,
    StockExplanation,
    TransitionGuide,
    ConceptLesson,
    EducatorOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tier_guide(**overrides) -> dict:
    base = {
        "overview": "The 3-tier portfolio system balances growth ambition with capital protection across momentum, quality, and defensive positions.",
        "tier_1_description": "Tier 1 Momentum: highest-conviction growth picks with 22% trailing stops. Target 25-30% returns.",
        "tier_2_description": "Tier 2 Quality Growth: strong companies rated 85+ with 18% trailing stops. Target 18-22% returns.",
        "tier_3_description": "Tier 3 Defensive: capital preservation with ETFs and defensive stocks. 12% stops. Target 12-15%.",
        "choosing_advice": "Choose your tier weighting based on risk tolerance, time horizon, and market conditions.",
        "current_allocation": "39% T1 / 37% T2 / 22% T3 / 2% Cash",
        "analogy": "T1 is a sports car, T2 is a sedan, T3 is an armored SUV.",
    }
    base.update(overrides)
    return base


def _make_keep_explanations(**overrides) -> dict:
    base = {
        "overview": "14 positions are pre-committed across 2 categories: fundamental value and IBD elite ratings.",
        "fundamental_keeps": "5 stocks (UNH, MU, AMZN, MRK, COP) kept for strong value metrics despite missing some elite thresholds.",
        "ibd_keeps": "9 stocks (CLS, AGI, TFPM, GMED, SCCO, HWM, KLAC, TPR, GS) kept for elite IBD ratings: Composite 93+ AND RS 90+.",
        "total_keeps": 14,
    }
    base.update(overrides)
    return base


def _make_stock_explanation(**overrides) -> dict:
    base = {
        "symbol": "NVDA",
        "company_name": "NVIDIA Corp",
        "tier": 1,
        "keep_category": None,
        "one_liner": "NVDA: The AI gold rush leader powering data centers worldwide",
        "why_selected": "Elite IBD ratings with Composite 98 places NVDA in the top 2% of all stocks. Strong AI demand drives growth.",
        "ibd_ratings_explained": "Composite 98 means top 2% of all stocks in combined earnings, price strength, and fundamentals.",
        "key_strength": "Dominant market position in AI chips and data center GPUs",
        "key_risk": "Valuation stretched after massive run-up, customer concentration risk",
        "position_context": "T1 Momentum position with 22% trailing stop",
        "analogy": "Like the company that sold pickaxes during the Gold Rush",
    }
    base.update(overrides)
    return base


def _make_concept_lesson(**overrides) -> dict:
    base = {
        "concept": "Composite Rating",
        "simple_explanation": "A single number from 1-99 that combines all of IBD's individual ratings into one overall grade.",
        "analogy": "Like a student's GPA that combines math, English, and science grades into one number.",
        "why_it_matters": "Quickly identifies the strongest stocks without checking 5 separate ratings.",
        "example_from_analysis": "CLS has Composite 99 — ranked in the top 1% of all 7000+ stocks tracked by IBD.",
        "framework_reference": "Framework v4.0 Section 2.1",
    }
    base.update(overrides)
    return base


def _make_transition_guide(**overrides) -> dict:
    base = {
        "overview": "The portfolio transition moves from the current scattered holdings to the recommended 3-tier structure over 4 weeks, selling non-recommended positions first.",
        "what_to_sell": "15 positions not in the recommended portfolio including CAT, AMD, PFE, and DIS will be sold in Week 1.",
        "what_to_buy": "New positions to establish include momentum leaders identified by the IBD screening process.",
        "money_flow_explained": "Cash from selling non-recommended positions funds new buys. Sell proceeds cover buy costs with a small cash reserve for the 2% cash target.",
        "timeline_explained": "Week 1: Liquidation of non-recommended positions. Weeks 2-4: Build T1, T2, T3 in order.",
        "before_after_summary": "Portfolio narrows from scattered to focused 3-tier structure with improved diversification.",
        "key_numbers": "Turnover ~50%, 15 sells, 20+ new buys across 3 tiers",
    }
    base.update(overrides)
    return base


def _make_glossary() -> dict:
    return {
        "Composite Rating": "IBD's master score (1-99) combining earnings, relative strength, fundamentals, and institutional interest.",
        "RS Rating": "Relative Strength — how the stock's price compares to all others over 12 months.",
        "EPS Rating": "Earnings Per Share rating (1-99) measuring profit growth vs all stocks.",
        "SMR Rating": "Sales, Margins, Return on equity — grades A through E measuring business quality.",
        "Acc/Dis Rating": "Accumulation/Distribution — shows whether big investors are buying or selling.",
        "CAN SLIM": "IBD's 7-factor checklist: Current earnings, Annual growth, New, Supply/demand, Leader, Institutional, Market.",
        "Tier 1 Momentum": "Highest-growth portfolio tier. Composite 95+, RS 90+. 22% trailing stop.",
        "Tier 2 Quality Growth": "Balanced growth tier. Composite 85+, RS 80+. 18% trailing stop.",
        "Tier 3 Defensive": "Capital preservation tier. Composite 80+, RS 75+. 12% trailing stop.",
        "Trailing Stop": "Automatic sell order that rises with stock price but never falls.",
        "Stop Tightening": "As a stock gains 10-20%, the trailing stop gets closer to lock in more profit.",
        "Sector Rotation": "The pattern of different market sectors taking turns leading performance.",
        "Multi-Source Validated": "A stock recommended by 2+ independent sources with score 5+.",
        "IBD Keep Threshold": "Composite 93+ AND RS 90+. Elite stocks kept unless overriding factors.",
        "Sleep Well Score": "Risk Officer rating (1-10) of portfolio comfort. 10 = most comfortable.",
    }


def _make_valid_educator_output(**overrides) -> dict:
    stock_explanations = [_make_stock_explanation(symbol=f"SYM{i}", company_name=f"Company {i}") for i in range(15)]
    concept_lessons = [
        _make_concept_lesson(concept=f"Concept {i}", framework_reference=f"Framework v4.0 Section {i}.1")
        for i in range(5)
    ]

    base = {
        "executive_summary": (
            "This portfolio analysis used the IBD Momentum Investment Framework v4.0 to screen, rate, and organize "
            "growth stock opportunities into a 3-tier structure. The analysis identified 50+ positions across 14 sectors, "
            "with 14 pre-committed keeps and trailing stops at every level. The rotation detector found sector leadership "
            "shifting toward energy and mining, and the portfolio has been positioned to benefit from this shift."
        ),
        "tier_guide": _make_tier_guide(),
        "keep_explanations": _make_keep_explanations(),
        "stock_explanations": stock_explanations,
        "rotation_explanation": (
            "The Rotation Detector found early signs of sector rotation from Technology toward Energy and Mining. "
            "This means market leadership is shifting, and the portfolio has been adjusted to capture this trend."
        ),
        "risk_explainer": (
            "The Risk Officer ran 10 checks on the portfolio. No vetoes were issued. The Sleep Well Score is 7/10, "
            "meaning the portfolio is well-diversified with appropriate stops at every position level for protection."
        ),
        "transition_guide": _make_transition_guide(),
        "concept_lessons": concept_lessons,
        "action_items": [
            "Review the 4-week implementation timeline before executing any trades",
            "Set trailing stops at the specified percentages for each tier immediately after purchase",
            "Monitor sector rotation signals monthly for changes in market leadership",
        ],
        "glossary": _make_glossary(),
        "analysis_date": "2025-01-15",
        "summary": "Educational report covering IBD methodology, 3-tier portfolio structure, 14 keeps, and 4-week transition plan.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Constant Tests
# ---------------------------------------------------------------------------

class TestConstants:

    @pytest.mark.schema
    def test_15_required_glossary_terms(self):
        assert len(REQUIRED_GLOSSARY_TERMS) == 15

    @pytest.mark.schema
    def test_12_concepts_to_teach(self):
        assert len(IBD_CONCEPTS_TO_TEACH) == 12

    @pytest.mark.schema
    def test_min_stock_explanations(self):
        assert MIN_STOCK_EXPLANATIONS == 15

    @pytest.mark.schema
    def test_min_concept_lessons(self):
        assert MIN_CONCEPT_LESSONS == 5

    @pytest.mark.schema
    def test_min_glossary_entries(self):
        assert MIN_GLOSSARY_ENTRIES == 15

    @pytest.mark.schema
    def test_action_item_bounds(self):
        assert MIN_ACTION_ITEMS == 3
        assert MAX_ACTION_ITEMS == 7


# ---------------------------------------------------------------------------
# TierGuide Tests
# ---------------------------------------------------------------------------

class TestTierGuide:

    @pytest.mark.schema
    def test_valid_tier_guide(self):
        guide = TierGuide(**_make_tier_guide())
        assert "3-tier" in guide.overview.lower() or "tier" in guide.overview.lower()

    @pytest.mark.schema
    def test_short_overview_fails(self):
        with pytest.raises(Exception):
            TierGuide(**_make_tier_guide(overview="Too short"))


# ---------------------------------------------------------------------------
# KeepExplanations Tests
# ---------------------------------------------------------------------------

class TestKeepExplanations:

    @pytest.mark.schema
    def test_valid_keep_explanations(self):
        keeps = KeepExplanations(**_make_keep_explanations())
        assert keeps.total_keeps == 14

    @pytest.mark.schema
    def test_wrong_total_fails(self):
        with pytest.raises(Exception, match="14"):
            KeepExplanations(**_make_keep_explanations(total_keeps=20))


# ---------------------------------------------------------------------------
# StockExplanation Tests
# ---------------------------------------------------------------------------

class TestStockExplanation:

    @pytest.mark.schema
    def test_valid_stock_explanation(self):
        stock = StockExplanation(**_make_stock_explanation())
        assert stock.symbol == "NVDA"
        assert stock.tier == 1

    @pytest.mark.schema
    def test_symbol_uppercased(self):
        stock = StockExplanation(**_make_stock_explanation(symbol="nvda"))
        assert stock.symbol == "NVDA"

    @pytest.mark.schema
    def test_invalid_keep_category_fails(self):
        with pytest.raises(Exception, match="keep_category"):
            StockExplanation(**_make_stock_explanation(keep_category="unknown"))

    @pytest.mark.schema
    def test_valid_keep_categories(self):
        for cat in ("fundamental", "ibd", None):
            stock = StockExplanation(**_make_stock_explanation(keep_category=cat))
            assert stock.keep_category == cat

    @pytest.mark.schema
    def test_short_why_selected_fails(self):
        with pytest.raises(Exception):
            StockExplanation(**_make_stock_explanation(why_selected="Too short"))


# ---------------------------------------------------------------------------
# ConceptLesson Tests
# ---------------------------------------------------------------------------

class TestConceptLesson:

    @pytest.mark.schema
    def test_valid_concept_lesson(self):
        lesson = ConceptLesson(**_make_concept_lesson())
        assert lesson.concept == "Composite Rating"

    @pytest.mark.schema
    def test_short_explanation_fails(self):
        with pytest.raises(Exception):
            ConceptLesson(**_make_concept_lesson(simple_explanation="Short"))


# ---------------------------------------------------------------------------
# EducatorOutput Tests
# ---------------------------------------------------------------------------

class TestEducatorOutput:

    @pytest.mark.schema
    def test_valid_output_parses(self):
        output = EducatorOutput(**_make_valid_educator_output())
        assert isinstance(output, EducatorOutput)

    @pytest.mark.schema
    def test_executive_summary_length(self):
        output = EducatorOutput(**_make_valid_educator_output())
        assert len(output.executive_summary) >= 200

    @pytest.mark.schema
    def test_minimum_stock_explanations(self):
        output = EducatorOutput(**_make_valid_educator_output())
        assert len(output.stock_explanations) >= 15

    @pytest.mark.schema
    def test_too_few_stock_explanations_fails(self):
        stocks = [_make_stock_explanation(symbol=f"S{i}", company_name=f"Co {i}") for i in range(5)]
        with pytest.raises(Exception):
            EducatorOutput(**_make_valid_educator_output(stock_explanations=stocks))

    @pytest.mark.schema
    def test_minimum_concept_lessons(self):
        output = EducatorOutput(**_make_valid_educator_output())
        assert len(output.concept_lessons) >= 5

    @pytest.mark.schema
    def test_minimum_glossary_entries(self):
        output = EducatorOutput(**_make_valid_educator_output())
        assert len(output.glossary) >= 15

    @pytest.mark.schema
    def test_too_few_glossary_entries_fails(self):
        small_glossary = {"Term1": "Def1", "Term2": "Def2"}
        with pytest.raises(Exception, match="Glossary"):
            EducatorOutput(**_make_valid_educator_output(glossary=small_glossary))

    @pytest.mark.schema
    def test_minimum_action_items(self):
        output = EducatorOutput(**_make_valid_educator_output())
        assert len(output.action_items) >= 3

    @pytest.mark.schema
    def test_too_few_action_items_fails(self):
        with pytest.raises(Exception):
            EducatorOutput(**_make_valid_educator_output(action_items=["One item"]))

    @pytest.mark.schema
    def test_short_executive_summary_fails(self):
        with pytest.raises(Exception):
            EducatorOutput(**_make_valid_educator_output(executive_summary="Too short"))

    @pytest.mark.schema
    def test_short_rotation_explanation_fails(self):
        with pytest.raises(Exception):
            EducatorOutput(**_make_valid_educator_output(rotation_explanation="Short"))

    @pytest.mark.schema
    def test_short_risk_explainer_fails(self):
        with pytest.raises(Exception):
            EducatorOutput(**_make_valid_educator_output(risk_explainer="Short"))

    @pytest.mark.schema
    def test_summary_min_length(self):
        output = EducatorOutput(**_make_valid_educator_output())
        assert len(output.summary) >= 50


# ---------------------------------------------------------------------------
# Cap Size Field Tests
# ---------------------------------------------------------------------------

class TestCapSizeField:

    @pytest.mark.schema
    def test_stock_explanation_accepts_cap_size(self):
        """StockExplanation accepts cap_size."""
        stock = StockExplanation(**_make_stock_explanation(cap_size="mid"))
        assert stock.cap_size == "mid"

    @pytest.mark.schema
    def test_stock_explanation_cap_size_default_none(self):
        """cap_size defaults to None."""
        stock = StockExplanation(**_make_stock_explanation())
        assert stock.cap_size is None
