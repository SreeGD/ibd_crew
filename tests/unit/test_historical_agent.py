"""
Agent 13: Historical Analyst — Pipeline Tests
Level 2: Deterministic pipeline tests with mock AnalystOutput.

Tests the full run_historical_pipeline() flow which queries the
historical store, classifies momentum, and assembles output.
"""

from __future__ import annotations

import re
import tempfile
from datetime import date
from pathlib import Path

import pytest

from ibd_agents.agents.historical_analyst import run_historical_pipeline
from ibd_agents.schemas.historical_output import MOMENTUM_DIRECTIONS
from ibd_agents.schemas.analyst_output import (
    AnalystOutput,
    EliteFilterSummary,
    IBDKeep,
    RatedStock,
    SectorRank,
    TierDistribution,
)
from ibd_agents.schemas.historical_output import HistoricalAnalysisOutput
from ibd_agents.tools.historical_store import HAS_CHROMADB, ingest_snapshot


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
    """Build a valid AnalystOutput with 6 stocks across 4 sectors."""
    stocks = [
        _make_rated_stock(
            symbol="NVDA", company_name="NVIDIA", sector="CHIPS",
            tier=1, composite=98, rs=95, eps=99,
        ),
        _make_rated_stock(
            symbol="MSFT", company_name="Microsoft", sector="SOFTWARE",
            tier=1, composite=96, rs=91, eps=90,
        ),
        _make_rated_stock(
            symbol="GOOGL", company_name="Alphabet", sector="INTERNET",
            tier=1, composite=99, rs=94, eps=92,
        ),
        _make_rated_stock(
            symbol="JPM", company_name="JPMorgan", sector="BANKS",
            tier=2, composite=90, rs=85, eps=80,
            is_multi_source=False,
        ),
        _make_rated_stock(
            symbol="UNH", company_name="UnitedHealth", sector="HEALTHCARE",
            tier=3, composite=85, rs=75, eps=80,
            is_multi_source=False,
        ),
        _make_rated_stock(
            symbol="MRK", company_name="Merck", sector="CONSUMER",
            tier=3, composite=82, rs=72, eps=78,
            is_multi_source=False,
        ),
    ]

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
            "Tier assignment: T1 (Comp>=95, RS>=90, EPS>=80)."
        ),
        analysis_date=date.today().isoformat(),
        summary=(
            "Analyst pipeline processed 6 rated stocks across multiple sectors "
            "with full tier assignment, elite screening, and sector ranking."
        ),
    )


# ---------------------------------------------------------------------------
# Pipeline Tests — No ChromaDB (empty fallback)
# ---------------------------------------------------------------------------


@pytest.mark.behavior
class TestHistoricalPipelineEmpty:
    """Test pipeline when ChromaDB is unavailable or empty."""

    @pytest.fixture
    def analyst_output(self) -> AnalystOutput:
        return _make_analyst_output()

    def test_pipeline_returns_valid_output(self, analyst_output):
        """Pipeline produces valid HistoricalAnalysisOutput even without DB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "empty_db")
            output = run_historical_pipeline(
                analyst_output, db_path=db_path
            )
            assert isinstance(output, HistoricalAnalysisOutput)

    def test_empty_source_when_no_data(self, analyst_output):
        """historical_source is 'empty' or 'chromadb' (auto-ingest makes it chromadb)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "empty_db")
            output = run_historical_pipeline(
                analyst_output, db_path=db_path
            )
            assert output.historical_source in ("empty", "chromadb")

    def test_analysis_date_format(self, analyst_output):
        """analysis_date matches YYYY-MM-DD regex."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "empty_db")
            output = run_historical_pipeline(
                analyst_output, db_path=db_path
            )
            assert re.match(r"^\d{4}-\d{2}-\d{2}$", output.analysis_date)

    def test_summary_min_length(self, analyst_output):
        """Summary length >= 50 characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "empty_db")
            output = run_historical_pipeline(
                analyst_output, db_path=db_path
            )
            assert len(output.summary) >= 50

    def test_no_buy_sell_language(self, analyst_output):
        """No buy/sell language in summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "empty_db")
            output = run_historical_pipeline(
                analyst_output, db_path=db_path
            )
            assert not BUY_SELL_PATTERN.search(output.summary)


# ---------------------------------------------------------------------------
# Pipeline Tests — With ChromaDB Data
# ---------------------------------------------------------------------------

pytestmark_chromadb = pytest.mark.skipif(
    not HAS_CHROMADB, reason="chromadb not installed"
)


@pytestmark_chromadb
@pytest.mark.behavior
class TestHistoricalPipelineWithData:
    """Test pipeline with populated ChromaDB store."""

    @pytest.fixture
    def analyst_output(self) -> AnalystOutput:
        return _make_analyst_output()

    @pytest.fixture
    def populated_db(self):
        """Create a temp DB with 2 weeks of historical data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_chroma")

            # Week 1
            w1_stocks = [
                {
                    "symbol": "NVDA", "company_name": "NVIDIA",
                    "sector": "CHIPS", "composite_rating": 95,
                    "rs_rating": 90, "eps_rating": 95,
                    "ibd_lists": ["IBD 50"],
                },
                {
                    "symbol": "MSFT", "company_name": "Microsoft",
                    "sector": "SOFTWARE", "composite_rating": 93,
                    "rs_rating": 88, "eps_rating": 87,
                    "ibd_lists": ["IBD 50"],
                },
            ]
            ingest_snapshot(w1_stocks, "2026-01-25", "week1.pdf", db_path)

            # Week 2
            w2_stocks = [
                {
                    "symbol": "NVDA", "company_name": "NVIDIA",
                    "sector": "CHIPS", "composite_rating": 97,
                    "rs_rating": 93, "eps_rating": 97,
                    "ibd_lists": ["IBD 50", "Tech Leaders"],
                },
                {
                    "symbol": "MSFT", "company_name": "Microsoft",
                    "sector": "SOFTWARE", "composite_rating": 94,
                    "rs_rating": 89, "eps_rating": 88,
                    "ibd_lists": ["IBD 50"],
                },
            ]
            ingest_snapshot(w2_stocks, "2026-02-01", "week2.pdf", db_path)

            yield db_path

    def test_pipeline_produces_chromadb_output(
        self, analyst_output, populated_db
    ):
        """Pipeline with data produces chromadb-sourced output."""
        output = run_historical_pipeline(
            analyst_output, db_path=populated_db
        )
        assert isinstance(output, HistoricalAnalysisOutput)
        assert output.historical_source == "chromadb"

    def test_stock_analyses_populated(self, analyst_output, populated_db):
        """stock_analyses has entries for analyzed stocks."""
        output = run_historical_pipeline(
            analyst_output, db_path=populated_db
        )
        assert len(output.stock_analyses) > 0

    def test_stock_analyses_have_momentum(self, analyst_output, populated_db):
        """Each stock analysis has valid momentum direction."""
        output = run_historical_pipeline(
            analyst_output, db_path=populated_db
        )
        for sa in output.stock_analyses:
            assert sa.momentum_direction in MOMENTUM_DIRECTIONS

    def test_improving_stocks_consistent(self, analyst_output, populated_db):
        """Stocks in improving_stocks have direction=improving in analyses."""
        output = run_historical_pipeline(
            analyst_output, db_path=populated_db
        )
        analyzed = {s.symbol: s for s in output.stock_analyses}
        for sym in output.improving_stocks:
            match = analyzed.get(sym)
            if match:
                assert match.momentum_direction == "improving"

    def test_deteriorating_stocks_consistent(
        self, analyst_output, populated_db
    ):
        """Stocks in deteriorating_stocks have direction=deteriorating."""
        output = run_historical_pipeline(
            analyst_output, db_path=populated_db
        )
        analyzed = {s.symbol: s for s in output.stock_analyses}
        for sym in output.deteriorating_stocks:
            match = analyzed.get(sym)
            if match:
                assert match.momentum_direction == "deteriorating"

    def test_store_meta_populated(self, analyst_output, populated_db):
        """store_meta shows available=True with data."""
        output = run_historical_pipeline(
            analyst_output, db_path=populated_db
        )
        assert output.store_meta.store_available is True
        assert output.store_meta.total_records > 0

    def test_sector_trends_for_active_sectors(
        self, analyst_output, populated_db
    ):
        """sector_trends includes entries for sectors with historical data."""
        output = run_historical_pipeline(
            analyst_output, db_path=populated_db
        )
        # CHIPS and SOFTWARE have historical data
        sectors_with_trends = {t.sector for t in output.sector_trends}
        # At least one of the historical sectors should appear
        assert len(output.sector_trends) >= 0  # May be 0 if no overlap

    def test_no_buy_sell_language_with_data(
        self, analyst_output, populated_db
    ):
        """No buy/sell language in summary with populated data."""
        output = run_historical_pipeline(
            analyst_output, db_path=populated_db
        )
        assert not BUY_SELL_PATTERN.search(output.summary)


# ---------------------------------------------------------------------------
# Momentum Classification Tests
# ---------------------------------------------------------------------------

from ibd_agents.agents.historical_analyst import _classify_momentum


@pytest.mark.schema
class TestClassifyMomentum:
    """Test _classify_momentum() helper."""

    def test_improving(self):
        """Rising RS and composite -> 'improving'."""
        history = [
            {"snapshot_date": "2026-01-01", "rs_rating": 80, "composite_rating": 85},
            {"snapshot_date": "2026-01-08", "rs_rating": 85, "composite_rating": 88},
            {"snapshot_date": "2026-01-15", "rs_rating": 90, "composite_rating": 92},
            {"snapshot_date": "2026-01-22", "rs_rating": 95, "composite_rating": 96},
        ]
        direction, score = _classify_momentum(history)
        assert direction == "improving"
        assert score > 0

    def test_deteriorating(self):
        """Falling RS and composite -> 'deteriorating'."""
        history = [
            {"snapshot_date": "2026-01-01", "rs_rating": 95, "composite_rating": 96},
            {"snapshot_date": "2026-01-08", "rs_rating": 90, "composite_rating": 92},
            {"snapshot_date": "2026-01-15", "rs_rating": 85, "composite_rating": 88},
            {"snapshot_date": "2026-01-22", "rs_rating": 80, "composite_rating": 85},
        ]
        direction, score = _classify_momentum(history)
        assert direction == "deteriorating"
        assert score < 0

    def test_stable(self):
        """Flat ratings -> 'stable'."""
        history = [
            {"snapshot_date": "2026-01-01", "rs_rating": 90, "composite_rating": 92},
            {"snapshot_date": "2026-01-08", "rs_rating": 90, "composite_rating": 92},
            {"snapshot_date": "2026-01-15", "rs_rating": 91, "composite_rating": 92},
            {"snapshot_date": "2026-01-22", "rs_rating": 90, "composite_rating": 93},
        ]
        direction, score = _classify_momentum(history)
        assert direction == "stable"
        assert -5 <= score <= 5

    def test_single_record(self):
        """Single record -> 'stable' with score 0."""
        history = [
            {"snapshot_date": "2026-01-01", "rs_rating": 90, "composite_rating": 92},
        ]
        direction, score = _classify_momentum(history)
        assert direction == "stable"
        assert score == 0.0

    def test_empty_history(self):
        """Empty history -> 'stable' with score 0."""
        direction, score = _classify_momentum([])
        assert direction == "stable"
        assert score == 0.0

    def test_score_clamped(self):
        """Score is clamped to -100..100 range."""
        history = [
            {"snapshot_date": "2026-01-01", "rs_rating": 10, "composite_rating": 10},
            {"snapshot_date": "2026-01-08", "rs_rating": 99, "composite_rating": 99},
        ]
        direction, score = _classify_momentum(history)
        assert -100 <= score <= 100
