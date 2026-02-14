"""
Agent 03: Rotation Detector — Pipeline & Behavioral Tests
Level 2: Deterministic pipeline tests with mock data.
Level 3: Behavioral boundary tests.

Tests the full run_rotation_pipeline() flow + individual signal detectors.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from ibd_agents.agents.analyst_agent import run_analyst_pipeline
from ibd_agents.agents.rotation_detector import run_rotation_pipeline
from ibd_agents.schemas.analyst_output import AnalystOutput
from ibd_agents.schemas.research_output import (
    ResearchOutput,
    ResearchStock,
    SectorPattern,
    compute_preliminary_tier,
    is_ibd_keep_candidate,
)
from ibd_agents.schemas.rotation_output import (
    RotationDetectionOutput,
    RotationStatus,
    RotationType,
)
from ibd_agents.tools.rotation_signals import (
    compute_all_signals,
    detect_breadth_shift,
    detect_elite_concentration_shift,
    detect_ibd_keep_migration,
    detect_leadership_change,
    detect_rs_divergence,
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
            reasoning="Test stock from golden dataset — awaiting deeper analysis from Analyst Agent",
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
        summary="Test research output for rotation detector pipeline testing with sample data",
    )


def _build_full_pipeline():
    """Run Research → Analyst → return both outputs for rotation testing."""
    research_output = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
    analyst_output = run_analyst_pipeline(research_output)
    return analyst_output, research_output


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

class TestRotationPipeline:

    @pytest.fixture
    def pipeline_outputs(self):
        return _get_pipeline_outputs()

    @pytest.mark.schema
    def test_pipeline_produces_valid_output(self, pipeline_outputs):
        """Pipeline produces valid RotationDetectionOutput."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        assert isinstance(output, RotationDetectionOutput)

    @pytest.mark.schema
    def test_verdict_consistent_with_signals(self, pipeline_outputs):
        """Verdict matches signal count rules."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        sa = output.signals.signals_active
        if output.verdict == RotationStatus.ACTIVE:
            assert sa >= 3
            assert output.confidence >= 50
        elif output.verdict == RotationStatus.EMERGING:
            assert sa == 2
        else:
            assert sa <= 1

    @pytest.mark.schema
    def test_all_5_signals_computed(self, pipeline_outputs):
        """All 5 signal readings are present."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        assert output.signals.rs_divergence is not None
        assert output.signals.leadership_change is not None
        assert output.signals.breadth_shift is not None
        assert output.signals.elite_concentration_shift is not None
        assert output.signals.ibd_keep_migration is not None

    @pytest.mark.schema
    def test_source_sectors_have_outflow(self, pipeline_outputs):
        """All source sectors must have direction='outflow'."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        for sf in output.source_sectors:
            assert sf.direction == "outflow"

    @pytest.mark.schema
    def test_dest_sectors_have_inflow(self, pipeline_outputs):
        """All destination sectors must have direction='inflow'."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        for sf in output.destination_sectors:
            assert sf.direction == "inflow"

    @pytest.mark.schema
    def test_strategist_notes_present(self, pipeline_outputs):
        """Strategist notes are non-empty."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        assert len(output.strategist_notes) >= 1

    @pytest.mark.schema
    def test_market_regime_valid(self, pipeline_outputs):
        """Market regime is bull, bear, or neutral."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        assert output.market_regime.regime in ("bull", "bear", "neutral")

    @pytest.mark.schema
    def test_analysis_date_is_today(self, pipeline_outputs):
        """Analysis date matches today."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        assert output.analysis_date == date.today().isoformat()

    @pytest.mark.schema
    def test_rotation_type_consistent_with_verdict(self, pipeline_outputs):
        """NONE verdict has NONE type; non-NONE has non-NONE type."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        if output.verdict == RotationStatus.NONE:
            assert output.rotation_type == RotationType.NONE
        else:
            assert output.rotation_type != RotationType.NONE

    @pytest.mark.schema
    def test_stage_consistent_with_verdict(self, pipeline_outputs):
        """ACTIVE/EMERGING must have stage; NONE must not."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        if output.verdict != RotationStatus.NONE:
            assert output.rotation_stage is not None
        else:
            assert output.rotation_stage is None

    @pytest.mark.schema
    def test_velocity_consistent_with_verdict(self, pipeline_outputs):
        """ACTIVE/EMERGING should have velocity; NONE should not."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        if output.verdict != RotationStatus.NONE:
            assert output.velocity in ("Slow", "Moderate", "Fast")
        else:
            assert output.velocity is None

    @pytest.mark.schema
    def test_confidence_in_range(self, pipeline_outputs):
        """Confidence is 0-100."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        assert 0 <= output.confidence <= 100


# ---------------------------------------------------------------------------
# Individual Signal Tests
# ---------------------------------------------------------------------------

class TestIndividualSignals:

    @pytest.fixture
    def pipeline_outputs(self):
        return _get_pipeline_outputs()

    @pytest.mark.schema
    def test_rs_divergence_returns_signal(self, pipeline_outputs):
        """RS divergence detector returns a valid SignalReading."""
        analyst, _ = pipeline_outputs
        sig = detect_rs_divergence(analyst.sector_rankings)
        assert sig.signal_name == "RS Divergence"
        assert isinstance(sig.triggered, bool)
        assert len(sig.evidence) >= 10

    @pytest.mark.schema
    def test_rs_divergence_with_few_sectors(self):
        """RS divergence handles < 6 sectors gracefully."""
        from ibd_agents.schemas.analyst_output import SectorRank
        few = [SectorRank(
            rank=i+1, sector=f"SECTOR{i}", stock_count=5,
            avg_composite=85.0, avg_rs=80.0 + i, elite_pct=30.0,
            multi_list_pct=20.0, sector_score=60.0, top_stocks=["TST"],
        ) for i in range(3)]
        sig = detect_rs_divergence(few)
        assert sig.triggered is False

    @pytest.mark.schema
    def test_leadership_change_returns_signal(self, pipeline_outputs):
        """Leadership change detector returns valid SignalReading."""
        analyst, research = pipeline_outputs
        sig = detect_leadership_change(
            analyst.sector_rankings, research.sector_patterns
        )
        assert sig.signal_name == "Leadership Change"

    @pytest.mark.schema
    def test_breadth_shift_returns_signal(self, pipeline_outputs):
        """Breadth shift detector returns valid SignalReading."""
        analyst, _ = pipeline_outputs
        sig = detect_breadth_shift(
            analyst.sector_rankings, analyst.rated_stocks
        )
        assert sig.signal_name == "Breadth Shift"

    @pytest.mark.schema
    def test_elite_concentration_returns_signal(self, pipeline_outputs):
        """Elite concentration shift detector returns valid SignalReading."""
        analyst, _ = pipeline_outputs
        sig = detect_elite_concentration_shift(analyst.sector_rankings)
        assert sig.signal_name == "Elite Concentration Shift"

    @pytest.mark.schema
    def test_ibd_keep_migration_returns_signal(self, pipeline_outputs):
        """IBD keep migration detector returns valid SignalReading."""
        analyst, _ = pipeline_outputs
        sig = detect_ibd_keep_migration(
            analyst.ibd_keeps, analyst.rated_stocks
        )
        assert sig.signal_name == "IBD Keep Migration"

    @pytest.mark.schema
    def test_ibd_keep_migration_empty_keeps(self):
        """IBD keep migration handles empty keeps list."""
        sig = detect_ibd_keep_migration([], [])
        assert sig.triggered is False
        assert sig.signal_name == "IBD Keep Migration"


# ---------------------------------------------------------------------------
# Behavioral Boundary Tests
# ---------------------------------------------------------------------------

class TestBehavioralBoundaries:

    @pytest.fixture
    def pipeline_outputs(self):
        return _get_pipeline_outputs()

    @pytest.mark.behavior
    def test_no_buy_sell_language_in_notes(self, pipeline_outputs):
        """Strategist notes must not contain buy/sell language."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        forbidden = ["buy", "sell", "purchase", "short", "long position"]
        for note in output.strategist_notes:
            note_lower = note.lower()
            for word in forbidden:
                assert word not in note_lower, (
                    f"Found '{word}' in strategist note: {note}"
                )

    @pytest.mark.behavior
    def test_no_allocation_percentages_in_notes(self, pipeline_outputs):
        """Strategist notes must not contain allocation percentages."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        import re
        for note in output.strategist_notes:
            # Check for patterns like "allocate 30%", "20% allocation"
            assert not re.search(r"allocat\w*\s+\d+%", note.lower()), (
                f"Found allocation percentage in note: {note}"
            )

    @pytest.mark.behavior
    def test_no_duration_predictions_in_notes(self, pipeline_outputs):
        """Strategist notes should not predict how long rotation lasts."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        forbidden = ["will last", "will continue for", "expect it to end",
                      "weeks remaining", "months left"]
        for note in output.strategist_notes:
            note_lower = note.lower()
            for phrase in forbidden:
                assert phrase not in note_lower, (
                    f"Found duration prediction '{phrase}' in note: {note}"
                )

    @pytest.mark.behavior
    def test_summary_min_length(self, pipeline_outputs):
        """Summary must be at least 50 characters."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        assert len(output.summary) >= 50


# ---------------------------------------------------------------------------
# End-to-End Chain Test
# ---------------------------------------------------------------------------

class TestEndToEndChain:

    @pytest.mark.integration
    def test_research_analyst_rotation_chain(self):
        """Full Research → Analyst → Rotation chain produces valid output."""
        research_output = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
        analyst_output = run_analyst_pipeline(research_output)
        rotation_output = run_rotation_pipeline(analyst_output, research_output)

        assert isinstance(rotation_output, RotationDetectionOutput)
        assert rotation_output.verdict in (
            RotationStatus.ACTIVE, RotationStatus.EMERGING, RotationStatus.NONE
        )
        assert rotation_output.signals.signals_active >= 0
        assert rotation_output.signals.signals_active <= 5


# ---------------------------------------------------------------------------
# Golden Dataset Test
# ---------------------------------------------------------------------------

class TestGoldenDataset:

    @pytest.mark.schema
    def test_golden_rotation_output(self):
        """Verify rotation output against golden dataset expectations."""
        golden_path = Path(__file__).parent.parent.parent / "golden_datasets" / "rotation_golden.json"
        if not golden_path.exists():
            pytest.skip("Golden dataset not yet created")

        with open(golden_path) as f:
            golden = json.load(f)

        research_output = _build_research_output(SAMPLE_IBD_STOCKS + PADDING_STOCKS)
        analyst_output = run_analyst_pipeline(research_output)
        rotation_output = run_rotation_pipeline(analyst_output, research_output)

        assert rotation_output.verdict.value == golden["verdict"]
        assert rotation_output.signals.signals_active == golden["signals_active"]
        assert rotation_output.rotation_type.value == golden["rotation_type"]
        assert rotation_output.market_regime.regime == golden["regime"]


# ---------------------------------------------------------------------------
# Rotation Narrative Pipeline Tests
# ---------------------------------------------------------------------------

class TestRotationNarrativePipeline:

    @pytest.fixture
    def pipeline_outputs(self):
        return _get_pipeline_outputs()

    @pytest.mark.schema
    def test_narrative_source_set(self, pipeline_outputs):
        """After pipeline, narrative_source is set (template without LLM)."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        # Without API key, should be "template" if rotation detected, or None
        if output.verdict != RotationStatus.NONE:
            assert output.narrative_source in ("llm", "template")
        else:
            assert output.narrative_source in ("template", None)

    @pytest.mark.schema
    def test_narrative_none_when_no_rotation(self, pipeline_outputs):
        """Narrative is None when verdict is NONE."""
        analyst, research = pipeline_outputs
        output = run_rotation_pipeline(analyst, research)
        if output.verdict == RotationStatus.NONE:
            assert output.rotation_narrative is None
