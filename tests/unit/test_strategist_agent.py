"""
Agent 04: Sector Strategist — Pipeline & Behavioral Tests
Level 2: Deterministic pipeline tests with mock data.
Level 3: Behavioral boundary tests.

Tests the full run_strategist_pipeline() flow + tool functions.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from ibd_agents.agents.analyst_agent import run_analyst_pipeline
from ibd_agents.agents.rotation_detector import run_rotation_pipeline
from ibd_agents.agents.sector_strategist import run_strategist_pipeline
from ibd_agents.schemas.analyst_output import AnalystOutput
from ibd_agents.schemas.research_output import (
    ResearchOutput,
    ResearchStock,
    compute_preliminary_tier,
    is_ibd_keep_candidate,
)
from ibd_agents.schemas.rotation_output import (
    RotationDetectionOutput,
    RotationStatus,
)
from ibd_agents.schemas.strategy_output import (
    SECTOR_LIMITS,
    THEME_ETFS,
    SectorStrategyOutput,
)
from ibd_agents.tools.sector_allocator import (
    compute_cash_recommendation,
    compute_regime_adjusted_targets,
)
from ibd_agents.tools.theme_mapper import (
    generate_rotation_action_signals,
    map_themes_to_recommendations,
)

# Import shared fixtures
from tests.fixtures.conftest import PADDING_STOCKS, SAMPLE_IBD_STOCKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_research_output(stocks_data: list[dict]) -> ResearchOutput:
    """Build a valid ResearchOutput from raw stock dicts for testing."""
    research_stocks = []
    for s in stocks_data:
        tier = compute_preliminary_tier(
            s.get("composite_rating"), s.get("rs_rating"), s.get("eps_rating")
        )
        keep = is_ibd_keep_candidate(s.get("composite_rating"), s.get("rs_rating"))

        has_ratings = sum(1 for r in [s.get("composite_rating"), s.get("rs_rating"),
                                       s.get("eps_rating"), s.get("smr_rating"),
                                       s.get("acc_dis_rating")] if r is not None)
        confidence = min(1.0, 0.3 + (has_ratings * 0.14))

        research_stocks.append(ResearchStock(
            symbol=s["symbol"],
            company_name=s.get("company_name", s["symbol"]),
            sector=s.get("sector", "UNKNOWN"),
            security_type="stock",
            composite_rating=s.get("composite_rating"),
            rs_rating=s.get("rs_rating"),
            eps_rating=s.get("eps_rating"),
            smr_rating=s.get("smr_rating"),
            acc_dis_rating=s.get("acc_dis_rating"),
            ibd_lists=s.get("ibd_lists", ["IBD 50"]),
            schwab_themes=s.get("schwab_themes", []),
            fool_status=None,
            other_ratings={},
            validation_score=3,
            validation_providers=1,
            is_multi_source_validated=False,
            is_ibd_keep_candidate=keep,
            preliminary_tier=tier,
            sources=["test_data.xls"],
            confidence=confidence,
            reasoning="Test stock for strategist pipeline testing",
        ))

    keep_candidates = [s.symbol for s in research_stocks if s.is_ibd_keep_candidate]
    multi_validated = [s.symbol for s in research_stocks if s.is_multi_source_validated]

    return ResearchOutput(
        stocks=research_stocks,
        etfs=[],
        sector_patterns=[],
        data_sources_used=["test_data.xls"],
        data_sources_failed=[],
        total_securities_scanned=len(research_stocks),
        ibd_keep_candidates=keep_candidates,
        multi_source_validated=multi_validated,
        analysis_date=date.today().isoformat(),
        summary="Test research output for sector strategist pipeline testing with sample data",
    )


def _build_full_pipeline():
    """Run Research → Analyst → Rotation → return all outputs for strategy testing."""
    research_output = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
    analyst_output = run_analyst_pipeline(research_output)
    rotation_output = run_rotation_pipeline(analyst_output, research_output)
    return analyst_output, rotation_output, research_output


# Cache the pipeline outputs (they're deterministic for same input)
_cached_outputs = None


def _get_pipeline_outputs():
    global _cached_outputs
    if _cached_outputs is None:
        _cached_outputs = _build_full_pipeline()
    return _cached_outputs


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

class TestStrategistPipeline:

    @pytest.fixture
    def pipeline_outputs(self):
        return _get_pipeline_outputs()

    @pytest.mark.schema
    def test_pipeline_produces_valid_output(self, pipeline_outputs):
        """Pipeline produces valid SectorStrategyOutput."""
        analyst, rotation, _ = pipeline_outputs
        output = run_strategist_pipeline(rotation, analyst)
        assert isinstance(output, SectorStrategyOutput)

    @pytest.mark.schema
    def test_min_8_sectors_in_overall(self, pipeline_outputs):
        """Overall allocation has at least 8 sectors."""
        analyst, rotation, _ = pipeline_outputs
        output = run_strategist_pipeline(rotation, analyst)
        sector_count = len(output.sector_allocations.overall_allocation)
        assert sector_count >= SECTOR_LIMITS["min_sectors"], (
            f"Only {sector_count} sectors, need >= {SECTOR_LIMITS['min_sectors']}"
        )

    @pytest.mark.schema
    def test_no_sector_exceeds_40pct(self, pipeline_outputs):
        """No single sector exceeds 40% in overall allocation."""
        analyst, rotation, _ = pipeline_outputs
        output = run_strategist_pipeline(rotation, analyst)
        max_pct = SECTOR_LIMITS["max_single_sector"]
        for sector, pct in output.sector_allocations.overall_allocation.items():
            assert pct <= max_pct, (
                f"Sector '{sector}' at {pct}% exceeds max {max_pct}%"
            )

    @pytest.mark.schema
    def test_tier_targets_sum_to_100(self, pipeline_outputs):
        """Tier targets should sum to approximately 100%."""
        analyst, rotation, _ = pipeline_outputs
        output = run_strategist_pipeline(rotation, analyst)
        total = sum(output.sector_allocations.tier_targets.values())
        assert 95.0 <= total <= 105.0, f"Tier targets sum = {total}"

    @pytest.mark.schema
    def test_cash_in_range(self, pipeline_outputs):
        """Cash recommendation is 2-15%."""
        analyst, rotation, _ = pipeline_outputs
        output = run_strategist_pipeline(rotation, analyst)
        cash = output.sector_allocations.cash_recommendation
        assert 2.0 <= cash <= 15.0, f"Cash = {cash}%"

    @pytest.mark.schema
    def test_rotation_response_reflects_verdict(self, pipeline_outputs):
        """Rotation response text reflects the actual rotation verdict."""
        analyst, rotation, _ = pipeline_outputs
        output = run_strategist_pipeline(rotation, analyst)
        if rotation.verdict == RotationStatus.ACTIVE:
            assert "active" in output.rotation_response.lower()
        elif rotation.verdict == RotationStatus.EMERGING:
            assert "emerging" in output.rotation_response.lower()
        else:
            assert "no rotation" in output.rotation_response.lower()

    @pytest.mark.schema
    def test_regime_adjustment_reflects_regime(self, pipeline_outputs):
        """Regime adjustment text mentions the actual regime."""
        analyst, rotation, _ = pipeline_outputs
        output = run_strategist_pipeline(rotation, analyst)
        regime = rotation.market_regime.regime
        assert regime in output.regime_adjustment.lower()

    @pytest.mark.schema
    def test_theme_etfs_valid(self, pipeline_outputs):
        """All recommended ETFs exist in THEME_ETFS for their theme."""
        analyst, rotation, _ = pipeline_outputs
        output = run_strategist_pipeline(rotation, analyst)
        for rec in output.theme_recommendations:
            valid_etfs = set(THEME_ETFS.get(rec.theme, []))
            for etf in rec.recommended_etfs:
                assert etf in valid_etfs, (
                    f"ETF '{etf}' not valid for theme '{rec.theme}'"
                )

    @pytest.mark.schema
    def test_rotation_signals_present(self, pipeline_outputs):
        """At least one rotation action signal is generated."""
        analyst, rotation, _ = pipeline_outputs
        output = run_strategist_pipeline(rotation, analyst)
        assert len(output.rotation_signals) >= 1

    @pytest.mark.schema
    def test_analysis_date_is_today(self, pipeline_outputs):
        """Analysis date matches today."""
        analyst, rotation, _ = pipeline_outputs
        output = run_strategist_pipeline(rotation, analyst)
        assert output.analysis_date == date.today().isoformat()

    @pytest.mark.schema
    def test_summary_min_length(self, pipeline_outputs):
        """Summary must be at least 50 characters."""
        analyst, rotation, _ = pipeline_outputs
        output = run_strategist_pipeline(rotation, analyst)
        assert len(output.summary) >= 50

    @pytest.mark.schema
    def test_each_tier_allocation_sums_95_to_105(self, pipeline_outputs):
        """Each tier's sector allocation sums to 95-105%."""
        analyst, rotation, _ = pipeline_outputs
        output = run_strategist_pipeline(rotation, analyst)
        for tier_name, alloc in [
            ("T1", output.sector_allocations.tier_1_allocation),
            ("T2", output.sector_allocations.tier_2_allocation),
            ("T3", output.sector_allocations.tier_3_allocation),
        ]:
            if alloc:
                total = sum(alloc.values())
                assert 95.0 <= total <= 105.0, (
                    f"{tier_name} allocation sums to {total}%"
                )


# ---------------------------------------------------------------------------
# Tool Function Tests
# ---------------------------------------------------------------------------

class TestToolFunctions:

    @pytest.mark.schema
    def test_regime_targets_bull(self):
        """Bull regime adjusts T1 up, T3 down."""
        targets = compute_regime_adjusted_targets("bull")
        assert targets["T1"] > 39.0  # baseline + 5
        assert targets["T3"] < 22.0  # baseline - 5

    @pytest.mark.schema
    def test_regime_targets_bear(self):
        """Bear regime adjusts T1 down, T3 up."""
        targets = compute_regime_adjusted_targets("bear")
        assert targets["T1"] < 39.0  # baseline - 10
        assert targets["T3"] > 22.0  # baseline + 10

    @pytest.mark.schema
    def test_regime_targets_neutral(self):
        """Neutral regime keeps baseline targets."""
        targets = compute_regime_adjusted_targets("neutral")
        assert targets["T1"] == 39.0
        assert targets["T3"] == 22.0

    @pytest.mark.schema
    def test_regime_targets_sum_to_100(self):
        """All regime targets sum to ~100%."""
        for regime in ("bull", "bear", "neutral"):
            targets = compute_regime_adjusted_targets(regime)
            total = sum(targets.values())
            assert 95.0 <= total <= 105.0, (
                f"{regime} targets sum = {total}"
            )

    @pytest.mark.schema
    def test_cash_bull_no_rotation(self):
        """Bull + NONE rotation → low cash."""
        cash = compute_cash_recommendation("bull", RotationStatus.NONE)
        assert 2.0 <= cash <= 5.0

    @pytest.mark.schema
    def test_cash_bear_active_rotation(self):
        """Bear + ACTIVE rotation → high cash."""
        cash = compute_cash_recommendation("bear", RotationStatus.ACTIVE)
        assert cash >= 7.0

    @pytest.mark.schema
    def test_theme_mapper_returns_recommendations(self):
        """Theme mapper returns valid ThemeRecommendation list."""
        analyst, rotation, _ = _get_pipeline_outputs()
        recs = map_themes_to_recommendations(
            sector_rankings=analyst.sector_rankings,
            rated_stocks=analyst.rated_stocks,
            dest_sectors=rotation.destination_sectors,
            regime=rotation.market_regime.regime,
            verdict=rotation.verdict,
        )
        for rec in recs:
            assert rec.theme in THEME_ETFS
            assert rec.conviction in ("HIGH", "MEDIUM", "LOW")
            assert rec.tier_fit in (1, 2, 3)

    @pytest.mark.schema
    def test_rotation_signals_have_criteria(self):
        """Action signals have confirmation and invalidation lists."""
        analyst, rotation, _ = _get_pipeline_outputs()
        signals = generate_rotation_action_signals(
            verdict=rotation.verdict,
            rotation_type=rotation.rotation_type,
            stage=rotation.rotation_stage,
            source_sectors=rotation.source_sectors,
            dest_sectors=rotation.destination_sectors,
            regime=rotation.market_regime.regime,
        )
        for sig in signals:
            assert len(sig.confirmation) >= 1
            assert len(sig.invalidation) >= 1
            assert len(sig.action) >= 10


# ---------------------------------------------------------------------------
# Behavioral Boundary Tests
# ---------------------------------------------------------------------------

class TestBehavioralBoundaries:

    @pytest.fixture
    def strategy_output(self):
        analyst, rotation, _ = _get_pipeline_outputs()
        return run_strategist_pipeline(rotation, analyst)

    @pytest.mark.behavior
    def test_no_stock_picks(self, strategy_output):
        """Strategy output should not contain specific stock ticker recommendations."""
        text = strategy_output.model_dump_json()
        # The output may reference ETFs (that's OK), but should not say "buy NVDA" etc.
        forbidden = ["buy NVDA", "sell AAPL", "purchase TSLA", "short MSFT"]
        text_lower = text.lower()
        for phrase in forbidden:
            assert phrase not in text_lower

    @pytest.mark.behavior
    def test_no_position_sizes(self, strategy_output):
        """No dollar amounts or share counts."""
        text = strategy_output.model_dump_json()
        assert not re.search(r"\$\d+[,.]?\d*\s*(shares|position|worth)", text.lower())
        assert not re.search(r"\d+\s+shares", text.lower())

    @pytest.mark.behavior
    def test_no_buy_sell_language(self, strategy_output):
        """No direct buy/sell/trade recommendations."""
        # Check rotation_response, regime_adjustment, summary
        for text in [
            strategy_output.rotation_response,
            strategy_output.regime_adjustment,
            strategy_output.summary,
        ]:
            text_lower = text.lower()
            for word in ["buy", "sell", "purchase", "short position"]:
                assert word not in text_lower, (
                    f"Found '{word}' in: {text}"
                )

    @pytest.mark.behavior
    def test_no_trailing_stops(self, strategy_output):
        """No trailing stop recommendations."""
        text = strategy_output.model_dump_json().lower()
        assert "trailing stop" not in text
        assert "stop loss" not in text

    @pytest.mark.behavior
    def test_no_pm_override(self, strategy_output):
        """Output should not override PM discretion."""
        text = strategy_output.model_dump_json().lower()
        assert "you must" not in text
        assert "you should immediately" not in text


# ---------------------------------------------------------------------------
# End-to-End Chain Test
# ---------------------------------------------------------------------------

class TestEndToEndChain:

    @pytest.mark.integration
    def test_research_analyst_rotation_strategy_chain(self):
        """Full Research → Analyst → Rotation → Strategy chain produces valid output."""
        research_output = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
        analyst_output = run_analyst_pipeline(research_output)
        rotation_output = run_rotation_pipeline(analyst_output, research_output)
        strategy_output = run_strategist_pipeline(rotation_output, analyst_output)

        assert isinstance(strategy_output, SectorStrategyOutput)
        assert len(strategy_output.sector_allocations.overall_allocation) >= 8
        assert strategy_output.sector_allocations.cash_recommendation >= 2.0
        assert len(strategy_output.rotation_signals) >= 1


# ---------------------------------------------------------------------------
# Golden Dataset Test
# ---------------------------------------------------------------------------

class TestGoldenDataset:

    @pytest.mark.schema
    def test_golden_strategy_output(self):
        """Verify strategy output against golden dataset expectations."""
        golden_path = Path(__file__).parent.parent.parent / "golden_datasets" / "strategy_golden.json"
        if not golden_path.exists():
            pytest.skip("Golden dataset not yet created")

        with open(golden_path) as f:
            golden = json.load(f)

        analyst, rotation, _ = _get_pipeline_outputs()
        output = run_strategist_pipeline(rotation, analyst)

        # Verify deterministic properties
        assert len(output.sector_allocations.overall_allocation) >= golden["min_sectors"]
        assert output.sector_allocations.cash_recommendation >= golden["cash_min"]
        assert output.sector_allocations.cash_recommendation <= golden["cash_max"]
        assert rotation.market_regime.regime == golden["regime"]

        # Verify tier targets are within ranges
        for key in ("T1", "T2", "T3", "Cash"):
            val = output.sector_allocations.tier_targets[key]
            lo, hi = golden["tier_target_ranges"][key]
            assert lo <= val <= hi, (
                f"Tier target {key}={val} not in [{lo}, {hi}]"
            )
