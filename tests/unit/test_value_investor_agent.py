"""
Agent 11: Value Investor — Pipeline Tests
Level 2: Deterministic pipeline tests with mock AnalystOutput.

Tests the full run_value_investor_pipeline() flow.
"""

from __future__ import annotations

import re
from datetime import date

import pytest

from ibd_agents.agents.value_investor_agent import run_value_investor_pipeline
from ibd_agents.schemas.analyst_output import (
    AnalystOutput,
    EliteFilterSummary,
    IBDKeep,
    RatedStock,
    SectorRank,
    TierDistribution,
)
from ibd_agents.schemas.value_investor_output import (
    MoatAnalysisSummary,
    ValueInvestorOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BUY_SELL_PATTERN = re.compile(
    r"\b(buy|sell|hold|strong buy|strong sell|outperform|underperform"
    r"|overweight|underweight|accumulate|reduce|add shares"
    r"|position size|allocation)\b",
    re.IGNORECASE,
)


def _make_rated_stock(**overrides) -> dict:
    """Minimal valid RatedStock dict for testing."""
    base = {
        "symbol": "TEST",
        "company_name": "Test Corp",
        "sector": "CHIPS",
        "security_type": "stock",
        "tier": 1,
        "tier_label": "Momentum",
        "composite_rating": 95,
        "rs_rating": 90,
        "eps_rating": 85,
        "smr_rating": "A",
        "acc_dis_rating": "B",
        "passes_eps_filter": True,
        "passes_rs_filter": True,
        "passes_smr_filter": True,
        "passes_acc_dis_filter": True,
        "passes_all_elite": True,
        "is_ibd_keep": True,
        "is_multi_source_validated": False,
        "ibd_lists": ["IBD 50"],
        "schwab_themes": [],
        "validation_score": 3,
        "conviction": 8,
        "strengths": ["Strong momentum", "High RS rating"],
        "weaknesses": ["Premium valuation"],
        "catalyst": "Upcoming earnings release with strong expected growth",
        "reasoning": (
            "High-conviction momentum play with top-tier IBD ratings "
            "across all categories and strong sector positioning"
        ),
        "sector_rank_in_sector": 1,
        "estimated_pe": 25.0,
        "pe_category": "Growth Premium",
        "peg_ratio": 1.5,
        "peg_category": "Fair Value",
        "estimated_beta": 1.2,
        "estimated_return_pct": 18.0,
        "return_category": "Strong",
        "estimated_volatility_pct": 28.0,
        "sharpe_ratio": 0.5,
        "sharpe_category": "Good",
        "alpha_pct": 5.0,
        "alpha_category": "Outperformer",
        "risk_rating": "Good",
        "catalyst_source": "template",
        "catalyst_conviction_adjustment": 0,
        # Morningstar fields (default None)
        "morningstar_rating": None,
        "economic_moat": None,
        "fair_value": None,
        "morningstar_price": None,
        "price_to_fair_value": None,
        "morningstar_uncertainty": None,
    }
    base.update(overrides)
    return base


def _build_analyst_output(stock_dicts: list[dict]) -> AnalystOutput:
    """Build a valid AnalystOutput from a list of RatedStock dicts."""
    rated_stocks = [RatedStock(**d) for d in stock_dicts]

    # Build sector rankings
    sectors: dict[str, list[RatedStock]] = {}
    for s in rated_stocks:
        sectors.setdefault(s.sector, []).append(s)

    sector_rankings = []
    for rank, (sector, stocks) in enumerate(
        sorted(sectors.items(), key=lambda x: -len(x[1])), 1
    ):
        avg_comp = sum(s.composite_rating for s in stocks) / len(stocks)
        avg_rs = sum(s.rs_rating for s in stocks) / len(stocks)
        sector_rankings.append(SectorRank(
            rank=rank,
            sector=sector,
            stock_count=len(stocks),
            avg_composite=round(avg_comp, 1),
            avg_rs=round(avg_rs, 1),
            elite_pct=50.0,
            multi_list_pct=30.0,
            sector_score=round(avg_comp * 0.5 + avg_rs * 0.3, 1),
            top_stocks=[s.symbol for s in stocks[:3]],
            ibd_keep_count=sum(1 for s in stocks if s.is_ibd_keep),
            avg_pe=25.0,
            avg_beta=1.1,
            avg_volatility=25.0,
        ))

    keep_candidates = [
        IBDKeep(
            symbol=s.symbol,
            composite_rating=s.composite_rating,
            rs_rating=s.rs_rating,
            eps_rating=s.eps_rating,
            ibd_lists=s.ibd_lists,
            tier=s.tier,
            keep_rationale=f"{s.symbol}: Comp {s.composite_rating}, RS {s.rs_rating}",
        )
        for s in rated_stocks
        if s.is_ibd_keep
    ]

    tier_counts = {1: 0, 2: 0, 3: 0}
    for s in rated_stocks:
        tier_counts[s.tier] = tier_counts.get(s.tier, 0) + 1

    return AnalystOutput(
        rated_stocks=rated_stocks,
        unrated_stocks=[],
        sector_rankings=sector_rankings,
        ibd_keeps=keep_candidates,
        elite_filter_summary=EliteFilterSummary(
            total_screened=len(rated_stocks),
            passed_all_four=sum(1 for s in rated_stocks if s.passes_all_elite),
            failed_eps=0,
            failed_rs=0,
            failed_smr=0,
            failed_acc_dis=0,
            missing_ratings=0,
        ),
        tier_distribution=TierDistribution(
            tier_1_count=tier_counts.get(1, 0),
            tier_2_count=tier_counts.get(2, 0),
            tier_3_count=tier_counts.get(3, 0),
            below_threshold_count=0,
            unrated_count=0,
        ),
        methodology_notes=(
            "Elite Screening gates: EPS>=85, RS>=75, SMR A/B/B-, Acc/Dis A/B/B-. "
            "Tier assignment: T1 (Comp>=95, RS>=90, EPS>=80), "
            "T2 (>=85, >=80, >=75), T3 (>=80, >=75, >=70)."
        ),
        analysis_date=date.today().isoformat(),
        summary=(
            "Analyst pipeline processed 10 rated stocks across multiple sectors "
            "with full tier assignment, elite screening, and sector ranking analysis"
        ),
    )


# ---------------------------------------------------------------------------
# Test Data: 10 stocks spanning different value characteristics
# ---------------------------------------------------------------------------

_VALUE_STOCKS = [
    # Wide moat, 5-star, below fair value -> Quality Value
    _make_rated_stock(
        symbol="BRK.B", company_name="Berkshire Hathaway", sector="FINANCE",
        tier=1, tier_label="Momentum",
        composite_rating=95, rs_rating=90, eps_rating=88,
        is_ibd_keep=True,
        pe_category="Value", peg_ratio=1.2, peg_category="Fair Value",
        morningstar_rating="5-star", economic_moat="Wide",
        price_to_fair_value=0.85, morningstar_uncertainty="Low",
        sector_rank_in_sector=1,
    ),
    # Deep Value P/E
    _make_rated_stock(
        symbol="CVS", company_name="CVS Health", sector="MEDICAL",
        tier=2, tier_label="Quality Growth",
        composite_rating=88, rs_rating=82, eps_rating=78,
        is_ibd_keep=False,
        pe_category="Deep Value", peg_ratio=0.8, peg_category="Undervalued",
        morningstar_rating="4-star", economic_moat="Narrow",
        price_to_fair_value=0.75, morningstar_uncertainty="Medium",
        sector_rank_in_sector=1,
    ),
    # GARP candidate
    _make_rated_stock(
        symbol="GOOGL", company_name="Alphabet", sector="INTERNET",
        tier=1, tier_label="Momentum",
        composite_rating=99, rs_rating=94, eps_rating=92,
        is_ibd_keep=True,
        pe_category="Reasonable", peg_ratio=1.3, peg_category="Fair Value",
        sector_rank_in_sector=1,
    ),
    # Another GARP candidate
    _make_rated_stock(
        symbol="MSFT", company_name="Microsoft", sector="SOFTWARE",
        tier=1, tier_label="Momentum",
        composite_rating=96, rs_rating=85, eps_rating=90,
        is_ibd_keep=False,
        pe_category="Reasonable", peg_ratio=1.8, peg_category="Fair Value",
        morningstar_rating="4-star", economic_moat="Wide",
        price_to_fair_value=1.05, morningstar_uncertainty="Low",
        sector_rank_in_sector=1,
    ),
    # Not Value — high PE, high PEG, growth premium
    _make_rated_stock(
        symbol="PLTR", company_name="Palantir", sector="SOFTWARE",
        tier=1, tier_label="Momentum",
        composite_rating=99, rs_rating=93, eps_rating=99,
        is_ibd_keep=True,
        pe_category="Speculative", peg_ratio=5.0, peg_category="Expensive",
        sector_rank_in_sector=2,
    ),
    # Not Value — growth premium
    _make_rated_stock(
        symbol="NVDA", company_name="NVIDIA", sector="CHIPS",
        tier=1, tier_label="Momentum",
        composite_rating=98, rs_rating=81, eps_rating=99,
        is_ibd_keep=False,
        pe_category="Growth Premium", peg_ratio=2.5, peg_category="Expensive",
        sector_rank_in_sector=1,
    ),
    # Narrow moat value
    _make_rated_stock(
        symbol="JPM", company_name="JPMorgan Chase", sector="FINANCE",
        tier=2, tier_label="Quality Growth",
        composite_rating=90, rs_rating=85, eps_rating=80,
        is_ibd_keep=False,
        pe_category="Value", peg_ratio=1.1, peg_category="Fair Value",
        morningstar_rating="3-star", economic_moat="Narrow",
        price_to_fair_value=0.95, morningstar_uncertainty="Medium",
        sector_rank_in_sector=2,
    ),
    # Weak momentum potential value trap
    _make_rated_stock(
        symbol="WBA", company_name="Walgreens Boots", sector="CONSUMER",
        tier=3, tier_label="Defensive",
        composite_rating=80, rs_rating=45, eps_rating=55,
        smr_rating="D", acc_dis_rating="D",
        is_ibd_keep=False,
        pe_category="Deep Value", peg_ratio=0.5, peg_category="Undervalued",
        passes_eps_filter=False, passes_rs_filter=False,
        passes_smr_filter=False, passes_acc_dis_filter=False,
        passes_all_elite=False,
        conviction=3,
        strengths=["Low P/E"],
        weaknesses=["Declining earnings", "Weak momentum", "Poor fundamentals"],
        sector_rank_in_sector=1,
    ),
    # Mid-range GARP with Morningstar data
    _make_rated_stock(
        symbol="ABBV", company_name="AbbVie", sector="MEDICAL",
        tier=2, tier_label="Quality Growth",
        composite_rating=89, rs_rating=83, eps_rating=82,
        is_ibd_keep=False,
        pe_category="Reasonable", peg_ratio=1.6, peg_category="Fair Value",
        morningstar_rating="4-star", economic_moat="Wide",
        price_to_fair_value=0.92, morningstar_uncertainty="Medium",
        llm_dividend_yield=3.5,
        sector_rank_in_sector=2,
    ),
    # Growth premium, no moat -> Not Value
    _make_rated_stock(
        symbol="APP", company_name="AppLovin", sector="SOFTWARE",
        tier=1, tier_label="Momentum",
        composite_rating=98, rs_rating=95, eps_rating=80,
        is_ibd_keep=True,
        pe_category="Growth Premium", peg_ratio=3.0, peg_category="Expensive",
        sector_rank_in_sector=3,
    ),
]


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

@pytest.mark.schema
class TestValueInvestorPipeline:
    """Test the deterministic Value Investor pipeline."""

    @pytest.fixture
    def analyst_output(self) -> AnalystOutput:
        """Build AnalystOutput from the 10 test stocks."""
        return _build_analyst_output(_VALUE_STOCKS)

    @pytest.fixture
    def pipeline_output(self, analyst_output) -> ValueInvestorOutput:
        """Run the pipeline and return the output (shared across tests)."""
        return run_value_investor_pipeline(analyst_output)

    def test_pipeline_produces_valid_output(self, pipeline_output):
        """Pipeline produces valid ValueInvestorOutput."""
        assert isinstance(pipeline_output, ValueInvestorOutput)
        assert len(pipeline_output.value_stocks) > 0

    def test_value_ranking_descending(self, pipeline_output):
        """Rank 1 has the highest value_score."""
        stocks = pipeline_output.value_stocks
        rank_1 = [s for s in stocks if s.value_rank == 1]
        assert len(rank_1) == 1
        # Rank 1 should have the highest value_score
        max_score = max(s.value_score for s in stocks)
        assert rank_1[0].value_score == max_score

    def test_all_stocks_get_assessment(self, pipeline_output, analyst_output):
        """Every rated stock from Analyst gets a value assessment."""
        assert len(pipeline_output.value_stocks) == len(analyst_output.rated_stocks)

    def test_sector_analysis_present(self, pipeline_output):
        """At least 1 sector analysis is produced."""
        assert len(pipeline_output.sector_value_analysis) >= 1

    def test_category_summaries_present(self, pipeline_output):
        """At least 1 value category summary is produced."""
        assert len(pipeline_output.value_category_summaries) >= 1

    def test_value_traps_subset_of_stocks(self, pipeline_output):
        """Value trap symbols are a subset of value_stocks symbols."""
        stock_symbols = {s.symbol for s in pipeline_output.value_stocks}
        for trap in pipeline_output.value_traps:
            assert trap in stock_symbols, f"Trap '{trap}' not in value_stocks"

    def test_top_picks_subset_of_stocks(self, pipeline_output):
        """Top value picks are a subset of value_stocks symbols."""
        stock_symbols = {s.symbol for s in pipeline_output.value_stocks}
        for pick in pipeline_output.top_value_picks:
            assert pick in stock_symbols, f"Pick '{pick}' not in value_stocks"

    def test_moat_analysis_present(self, pipeline_output):
        """Moat analysis summary is populated."""
        moat = pipeline_output.moat_analysis
        assert isinstance(moat, MoatAnalysisSummary)
        # We have stocks with Wide, Narrow, and no moat data
        total = (
            moat.wide_moat_count + moat.narrow_moat_count
            + moat.no_moat_count + moat.no_data_count
        )
        assert total == len(pipeline_output.value_stocks)

    def test_no_buy_sell_language(self, pipeline_output):
        """No buy/sell recommendation language in any text field."""
        for vs in pipeline_output.value_stocks:
            for field_name in (
                "value_category_reasoning", "alignment_detail",
            ):
                text = getattr(vs, field_name, "")
                assert not BUY_SELL_PATTERN.search(text), (
                    f"{vs.symbol}.{field_name} contains buy/sell language: '{text}'"
                )
            for signal in vs.value_trap_signals:
                assert not BUY_SELL_PATTERN.search(signal), (
                    f"{vs.symbol} trap signal contains buy/sell language: '{signal}'"
                )
        # Check summary and methodology too
        assert not BUY_SELL_PATTERN.search(pipeline_output.summary)
        assert not BUY_SELL_PATTERN.search(pipeline_output.methodology_notes)

    def test_ranks_are_unique(self, pipeline_output):
        """No duplicate value ranks."""
        ranks = [s.value_rank for s in pipeline_output.value_stocks]
        assert len(ranks) == len(set(ranks)), f"Duplicate ranks found: {ranks}"
