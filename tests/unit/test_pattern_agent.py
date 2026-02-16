"""
Agent 12: PatternAlpha — Per-Stock Scoring Pipeline Tests
Level 2: Deterministic pipeline tests with mock AnalystOutput.

Tests the full run_pattern_pipeline() flow which computes base scores,
(optionally) LLM pattern scores, and assembles PortfolioPatternOutput.
"""

from __future__ import annotations

import re
from datetime import date

import pytest

from ibd_agents.agents.pattern_agent import run_pattern_pipeline
from ibd_agents.schemas.analyst_output import (
    AnalystOutput,
    EliteFilterSummary,
    IBDKeep,
    RatedStock,
    SectorRank,
    TierDistribution,
)
from ibd_agents.schemas.pattern_output import (
    ENHANCED_RATING_LABELS,
    PortfolioPatternOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BUY_SELL_PATTERN = re.compile(
    r"\b(buy|sell|hold|strong buy|strong sell|outperform|underperform"
    r"|overweight|underweight|accumulate|reduce|add shares"
    r"|position size)\b",
    re.IGNORECASE,
)


def _make_rated_stock(
    symbol="AAPL",
    company_name="Apple Inc",
    sector="CHIPS",
    tier=1,
    composite=97,
    rs=95,
    eps=90,
    validation_score=8,
    is_ibd_keep=True,
    is_multi_source=True,
    sharpe="Good",
    alpha="Outperformer",
    risk="Good",
    cap_size="large",
):
    """Build a valid RatedStock for testing."""
    from ibd_agents.schemas.research_output import is_ibd_keep_candidate

    TIER_LABELS = {1: "Momentum", 2: "Quality Growth", 3: "Defensive"}

    actual_keep = is_ibd_keep_candidate(composite, rs)

    return RatedStock(
        symbol=symbol,
        company_name=company_name,
        sector=sector,
        tier=tier,
        tier_label=TIER_LABELS[tier],
        composite_rating=composite,
        rs_rating=rs,
        eps_rating=eps,
        passes_eps_filter=eps >= 85,
        passes_rs_filter=rs >= 75,
        passes_all_elite=(eps >= 85 and rs >= 75),
        is_ibd_keep=actual_keep,
        is_multi_source_validated=is_multi_source,
        validation_score=validation_score,
        conviction=8,
        strengths=["Strong momentum"],
        weaknesses=["High valuation"],
        catalyst="Upcoming earnings",
        reasoning=(
            "Strong growth stock with solid momentum indicators "
            "and consistent earnings growth trajectory"
        ),
        sector_rank_in_sector=1,
        sharpe_category=sharpe,
        alpha_category=alpha,
        risk_rating=risk,
        cap_size=cap_size,
    )


def _make_analyst_output() -> AnalystOutput:
    """Build a valid AnalystOutput with 9 stocks across 5+ sectors."""
    stocks = [
        # Tier 1 - Momentum (4 stocks, 4 sectors)
        _make_rated_stock(
            symbol="NVDA", company_name="NVIDIA", sector="CHIPS",
            tier=1, composite=98, rs=95, eps=99,
            sharpe="Excellent", alpha="Strong Outperformer", risk="Good",
        ),
        _make_rated_stock(
            symbol="MSFT", company_name="Microsoft", sector="SOFTWARE",
            tier=1, composite=96, rs=91, eps=90,
            sharpe="Good", alpha="Outperformer", risk="Good",
        ),
        _make_rated_stock(
            symbol="GOOGL", company_name="Alphabet", sector="INTERNET",
            tier=1, composite=99, rs=94, eps=92,
            sharpe="Good", alpha="Outperformer", risk="Excellent",
        ),
        _make_rated_stock(
            symbol="CLS", company_name="Celestica", sector="ELECTRONICS",
            tier=1, composite=99, rs=97, eps=88,
            sharpe="Good", alpha="Strong Outperformer", risk="Good",
        ),
        # Tier 2 - Quality Growth (3 stocks, 3 sectors)
        _make_rated_stock(
            symbol="JPM", company_name="JPMorgan Chase", sector="BANKS",
            tier=2, composite=90, rs=85, eps=80,
            is_multi_source=False,
            sharpe="Moderate", alpha="Outperformer", risk="Good",
        ),
        _make_rated_stock(
            symbol="GS", company_name="Goldman Sachs", sector="FINANCE",
            tier=2, composite=93, rs=91, eps=82,
            sharpe="Good", alpha="Outperformer", risk="Moderate",
        ),
        _make_rated_stock(
            symbol="GMED", company_name="Globus Medical", sector="MEDICAL",
            tier=2, composite=88, rs=82, eps=78,
            is_multi_source=False,
            sharpe="Moderate", alpha="Slight Underperformer", risk="Moderate",
        ),
        # Tier 3 - Defensive (2 stocks, 2 sectors)
        _make_rated_stock(
            symbol="UNH", company_name="UnitedHealth", sector="HEALTHCARE",
            tier=3, composite=85, rs=75, eps=80,
            is_multi_source=False,
            sharpe="Moderate", alpha="Slight Underperformer", risk="Good",
        ),
        _make_rated_stock(
            symbol="MRK", company_name="Merck", sector="CONSUMER",
            tier=3, composite=82, rs=72, eps=78,
            is_multi_source=False,
            sharpe="Moderate", alpha="Slight Underperformer", risk="Moderate",
        ),
    ]

    # Build sector rankings from actual stocks
    sectors: dict[str, list[RatedStock]] = {}
    for s in stocks:
        sectors.setdefault(s.sector, []).append(s)

    sector_rankings = []
    for rank, (sector, sector_stocks) in enumerate(
        sorted(sectors.items(), key=lambda x: -len(x[1])), 1
    ):
        avg_comp = sum(s.composite_rating for s in sector_stocks) / len(sector_stocks)
        avg_rs = sum(s.rs_rating for s in sector_stocks) / len(sector_stocks)
        sector_rankings.append(SectorRank(
            rank=rank,
            sector=sector,
            stock_count=len(sector_stocks),
            avg_composite=round(avg_comp, 1),
            avg_rs=round(avg_rs, 1),
            elite_pct=50.0,
            multi_list_pct=30.0,
            sector_score=round(avg_comp * 0.5 + avg_rs * 0.3, 1),
            top_stocks=[s.symbol for s in sector_stocks[:3]],
            ibd_keep_count=sum(1 for s in sector_stocks if s.is_ibd_keep),
        ))

    keep_candidates = [
        IBDKeep(
            symbol=s.symbol,
            composite_rating=s.composite_rating,
            rs_rating=s.rs_rating,
            eps_rating=s.eps_rating,
            ibd_lists=[],
            tier=s.tier,
            keep_rationale=f"{s.symbol}: Comp {s.composite_rating}, RS {s.rs_rating}",
        )
        for s in stocks
        if s.is_ibd_keep
    ]

    tier_counts = {1: 0, 2: 0, 3: 0}
    for s in stocks:
        tier_counts[s.tier] = tier_counts.get(s.tier, 0) + 1

    return AnalystOutput(
        rated_stocks=stocks,
        unrated_stocks=[],
        sector_rankings=sector_rankings,
        ibd_keeps=keep_candidates,
        elite_filter_summary=EliteFilterSummary(
            total_screened=len(stocks),
            passed_all_four=sum(1 for s in stocks if s.passes_all_elite),
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
            "Analyst pipeline processed 9 rated stocks across multiple sectors "
            "with full tier assignment, elite screening, and sector ranking analysis"
        ),
    )


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------


@pytest.mark.behavior
class TestPatternPipeline:
    """Test the deterministic PatternAlpha per-stock scoring pipeline."""

    @pytest.fixture
    def analyst_output(self) -> AnalystOutput:
        return _make_analyst_output()

    @pytest.fixture
    def pipeline_output(self, analyst_output) -> PortfolioPatternOutput:
        """Run the pipeline and return the output (shared across tests)."""
        return run_pattern_pipeline(analyst_output)

    def test_pipeline_produces_valid_output(self, pipeline_output):
        """run_pattern_pipeline(analyst_output) returns PortfolioPatternOutput."""
        assert isinstance(pipeline_output, PortfolioPatternOutput)

    def test_all_stocks_have_base_scores(self, pipeline_output):
        """Every stock_analysis has base_score with base_total > 0."""
        for sa in pipeline_output.stock_analyses:
            assert sa.base_score is not None, (
                f"{sa.symbol} missing base_score"
            )
            assert sa.base_score.base_total > 0, (
                f"{sa.symbol} base_total={sa.base_score.base_total} should be > 0"
            )

    def test_enhanced_equals_base_plus_pattern(self, pipeline_output):
        """For all stocks, enhanced_score = base_total + pattern_total (or base only)."""
        for sa in pipeline_output.stock_analyses:
            base = sa.base_score.base_total
            pattern = sa.pattern_score.pattern_total if sa.pattern_score else 0
            expected = base + pattern
            assert sa.enhanced_score == expected, (
                f"{sa.symbol}: enhanced_score={sa.enhanced_score} != "
                f"base({base}) + pattern({pattern}) = {expected}"
            )

    def test_enhanced_ratings_valid(self, pipeline_output):
        """All enhanced_rating_labels are in ENHANCED_RATING_LABELS."""
        for sa in pipeline_output.stock_analyses:
            assert sa.enhanced_rating_label in ENHANCED_RATING_LABELS, (
                f"{sa.symbol} has invalid enhanced_rating_label "
                f"'{sa.enhanced_rating_label}'"
            )

    def test_scoring_source_valid(self, pipeline_output):
        """scoring_source is 'llm' or 'deterministic' at both levels."""
        assert pipeline_output.scoring_source in ("llm", "deterministic"), (
            f"Portfolio scoring_source='{pipeline_output.scoring_source}' invalid"
        )
        for sa in pipeline_output.stock_analyses:
            assert sa.scoring_source in ("llm", "deterministic"), (
                f"{sa.symbol} scoring_source='{sa.scoring_source}' invalid"
            )

    def test_analysis_date_format(self, pipeline_output):
        """analysis_date matches YYYY-MM-DD regex."""
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", pipeline_output.analysis_date), (
            f"analysis_date '{pipeline_output.analysis_date}' doesn't match YYYY-MM-DD"
        )

    def test_summary_min_length(self, pipeline_output):
        """Summary length >= 50 characters."""
        assert len(pipeline_output.summary) >= 50, (
            f"Summary too short ({len(pipeline_output.summary)} chars): "
            f"'{pipeline_output.summary}'"
        )

    def test_methodology_notes_present(self, pipeline_output):
        """methodology_notes is non-empty."""
        assert pipeline_output.methodology_notes, (
            "methodology_notes should be non-empty"
        )
        assert len(pipeline_output.methodology_notes) > 0

    def test_portfolio_summary_counts(self, pipeline_output):
        """portfolio_summary.total_stocks_scored matches len(stock_analyses)."""
        assert pipeline_output.portfolio_summary.total_stocks_scored == len(
            pipeline_output.stock_analyses
        ), (
            f"total_stocks_scored="
            f"{pipeline_output.portfolio_summary.total_stocks_scored} "
            f"!= len(stock_analyses)={len(pipeline_output.stock_analyses)}"
        )

    def test_no_buy_sell_language(self, pipeline_output):
        """No 'buy', 'sell', 'hold' in summary or methodology_notes."""
        assert not BUY_SELL_PATTERN.search(pipeline_output.summary), (
            f"Summary contains buy/sell language: '{pipeline_output.summary}'"
        )
        assert not BUY_SELL_PATTERN.search(pipeline_output.methodology_notes), (
            f"methodology_notes contains buy/sell language: "
            f"'{pipeline_output.methodology_notes}'"
        )
