"""
Agent 03: Rotation Detector — Schema Tests
Level 1: Pure Pydantic validation, no LLM, no file I/O.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ibd_agents.schemas.research_output import IBD_SECTORS
from ibd_agents.schemas.rotation_output import (
    SECTOR_CLUSTERS,
    SECTOR_TO_CLUSTER,
    VALID_CLUSTER_NAMES,
    MarketRegime,
    RotationDetectionOutput,
    RotationSignals,
    RotationStage,
    RotationStatus,
    RotationType,
    SectorFlow,
    SignalReading,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(name: str = "test_signal", triggered: bool = False, **kw) -> SignalReading:
    return SignalReading(
        signal_name=kw.get("signal_name", name),
        triggered=triggered,
        value=kw.get("value", "10.5"),
        threshold=kw.get("threshold", "15.0"),
        evidence=kw.get("evidence", "Test evidence for this signal reading"),
    )


def _make_signals(active_count: int = 0) -> RotationSignals:
    """Make RotationSignals with exactly active_count signals triggered."""
    names = ["RS Divergence", "Leadership Change", "Breadth Shift",
             "Elite Concentration", "IBD Keep Migration"]
    readings = []
    for i, name in enumerate(names):
        readings.append(_make_signal(name=name, triggered=(i < active_count)))
    return RotationSignals(
        rs_divergence=readings[0],
        leadership_change=readings[1],
        breadth_shift=readings[2],
        elite_concentration_shift=readings[3],
        ibd_keep_migration=readings[4],
        signals_active=active_count,
    )


def _make_regime(**kw) -> MarketRegime:
    return MarketRegime(
        regime=kw.get("regime", "bull"),
        bull_signals_present=kw.get("bull_signals_present", ["strong breadth"]),
        bear_signals_present=kw.get("bear_signals_present", []),
        sector_breadth_pct=kw.get("sector_breadth_pct", 75.0),
        regime_note=kw.get("regime_note", "Market showing bull signals with strong breadth"),
    )


def _make_sector_flow(direction: str = "inflow", **kw) -> SectorFlow:
    return SectorFlow(
        sector=kw.get("sector", "MINING"),
        cluster=kw.get("cluster", "commodity"),
        direction=direction,
        current_rank=kw.get("current_rank", 1),
        avg_rs=kw.get("avg_rs", 93.0),
        elite_pct=kw.get("elite_pct", 75.0),
        stock_count=kw.get("stock_count", 6),
        magnitude=kw.get("magnitude", "Strong"),
        evidence=kw.get("evidence", "MINING showing strong inflow with rising RS"),
    )


def _make_output(**kw) -> RotationDetectionOutput:
    verdict = kw.get("verdict", RotationStatus.ACTIVE)
    signals_active = kw.get("signals_active", 3)
    return RotationDetectionOutput(
        verdict=verdict,
        confidence=kw.get("confidence", 60),
        rotation_type=kw.get("rotation_type", RotationType.CYCLICAL),
        rotation_stage=kw.get("rotation_stage", RotationStage.MID),
        market_regime=kw.get("market_regime", _make_regime()),
        signals=kw.get("signals", _make_signals(signals_active)),
        source_sectors=kw.get("source_sectors", [_make_sector_flow("outflow", sector="SOFTWARE", cluster="growth")]),
        destination_sectors=kw.get("destination_sectors", [_make_sector_flow("inflow")]),
        stable_sectors=kw.get("stable_sectors", ["BANKS"]),
        velocity=kw.get("velocity", "Moderate"),
        strategist_notes=kw.get("strategist_notes", ["Commodity cluster gaining leadership"]),
        analysis_date=kw.get("analysis_date", "2026-02-10"),
        summary=kw.get("summary", "Rotation ACTIVE: 3/5 signals triggered. Cyclical rotation from growth to commodity cluster detected."),
        rotation_narrative=kw.get("rotation_narrative", None),
        narrative_source=kw.get("narrative_source", None),
    )


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------

class TestRotationEnums:

    @pytest.mark.schema
    def test_rotation_status_values(self):
        assert RotationStatus.ACTIVE.value == "active"
        assert RotationStatus.EMERGING.value == "emerging"
        assert RotationStatus.NONE.value == "none"

    @pytest.mark.schema
    def test_rotation_type_values(self):
        assert len(RotationType) == 7
        assert RotationType.CYCLICAL.value == "cyclical"
        assert RotationType.NONE.value == "none"

    @pytest.mark.schema
    def test_rotation_stage_values(self):
        assert len(RotationStage) == 4
        assert RotationStage.EARLY.value == "early"
        assert RotationStage.EXHAUSTING.value == "exhausting"


# ---------------------------------------------------------------------------
# SignalReading Tests
# ---------------------------------------------------------------------------

class TestSignalReading:

    @pytest.mark.schema
    def test_valid_signal_reading(self):
        s = _make_signal(triggered=True)
        assert s.triggered is True

    @pytest.mark.schema
    def test_evidence_min_length(self):
        with pytest.raises(ValidationError):
            _make_signal(evidence="short")


# ---------------------------------------------------------------------------
# RotationSignals Tests
# ---------------------------------------------------------------------------

class TestRotationSignals:

    @pytest.mark.schema
    def test_signals_active_matches_count(self):
        signals = _make_signals(active_count=3)
        assert signals.signals_active == 3

    @pytest.mark.schema
    def test_signals_active_mismatch_rejects(self):
        with pytest.raises(ValidationError, match="signals_active"):
            RotationSignals(
                rs_divergence=_make_signal(triggered=True),
                leadership_change=_make_signal(triggered=True),
                breadth_shift=_make_signal(triggered=False),
                elite_concentration_shift=_make_signal(triggered=False),
                ibd_keep_migration=_make_signal(triggered=False),
                signals_active=3,  # Wrong: only 2 triggered
            )

    @pytest.mark.schema
    def test_all_5_signals_present(self):
        signals = _make_signals(0)
        assert signals.rs_divergence is not None
        assert signals.leadership_change is not None
        assert signals.breadth_shift is not None
        assert signals.elite_concentration_shift is not None
        assert signals.ibd_keep_migration is not None


# ---------------------------------------------------------------------------
# SectorFlow Tests
# ---------------------------------------------------------------------------

class TestSectorFlow:

    @pytest.mark.schema
    def test_valid_sector_flow(self):
        sf = _make_sector_flow("inflow")
        assert sf.direction == "inflow"
        assert sf.cluster == "commodity"

    @pytest.mark.schema
    def test_invalid_cluster_rejects(self):
        with pytest.raises(ValidationError, match="cluster"):
            _make_sector_flow(cluster="invalid_cluster")

    @pytest.mark.schema
    def test_invalid_magnitude_rejects(self):
        with pytest.raises(ValidationError, match="magnitude"):
            _make_sector_flow(magnitude="SuperStrong")

    @pytest.mark.schema
    def test_valid_outflow(self):
        sf = _make_sector_flow("outflow", sector="SOFTWARE", cluster="growth")
        assert sf.direction == "outflow"


# ---------------------------------------------------------------------------
# MarketRegime Tests
# ---------------------------------------------------------------------------

class TestMarketRegime:

    @pytest.mark.schema
    def test_valid_bull_regime(self):
        r = _make_regime(regime="bull")
        assert r.regime == "bull"

    @pytest.mark.schema
    def test_valid_bear_regime(self):
        r = _make_regime(regime="bear", bear_signals_present=["weak breadth"])
        assert r.regime == "bear"

    @pytest.mark.schema
    def test_invalid_regime_rejects(self):
        with pytest.raises(ValidationError):
            _make_regime(regime="sideways")

    @pytest.mark.schema
    def test_breadth_range(self):
        r = _make_regime(sector_breadth_pct=85.0)
        assert r.sector_breadth_pct == 85.0
        with pytest.raises(ValidationError):
            _make_regime(sector_breadth_pct=105.0)


# ---------------------------------------------------------------------------
# RotationDetectionOutput Tests
# ---------------------------------------------------------------------------

class TestRotationDetectionOutput:

    @pytest.mark.schema
    def test_valid_active_output(self):
        output = _make_output()
        assert output.verdict == RotationStatus.ACTIVE

    @pytest.mark.schema
    def test_valid_none_output(self):
        output = _make_output(
            verdict=RotationStatus.NONE,
            signals_active=0,
            confidence=10,
            rotation_type=RotationType.NONE,
            rotation_stage=None,
            velocity=None,
            source_sectors=[],
            destination_sectors=[],
            summary="No rotation detected. 0/5 signals triggered. Market leadership stable across all sector clusters.",
        )
        assert output.verdict == RotationStatus.NONE

    @pytest.mark.schema
    def test_valid_emerging_output(self):
        output = _make_output(
            verdict=RotationStatus.EMERGING,
            signals_active=2,
            confidence=40,
            rotation_stage=RotationStage.EARLY,
        )
        assert output.verdict == RotationStatus.EMERGING

    @pytest.mark.schema
    def test_active_requires_3_signals(self):
        with pytest.raises(ValidationError, match="ACTIVE requires signals_active >= 3"):
            _make_output(verdict=RotationStatus.ACTIVE, signals_active=2)

    @pytest.mark.schema
    def test_active_requires_confidence_50(self):
        with pytest.raises(ValidationError, match="ACTIVE requires confidence >= 50"):
            _make_output(verdict=RotationStatus.ACTIVE, signals_active=3, confidence=40)

    @pytest.mark.schema
    def test_emerging_requires_2_signals(self):
        with pytest.raises(ValidationError, match="EMERGING requires signals_active == 2"):
            _make_output(
                verdict=RotationStatus.EMERGING,
                signals_active=3,
                rotation_stage=RotationStage.EARLY,
            )

    @pytest.mark.schema
    def test_none_requires_1_or_fewer(self):
        with pytest.raises(ValidationError, match="NONE requires signals_active <= 1"):
            _make_output(
                verdict=RotationStatus.NONE,
                signals_active=2,
                confidence=10,
                rotation_type=RotationType.NONE,
                rotation_stage=None,
                velocity=None,
                source_sectors=[],
                destination_sectors=[],
                summary="No rotation detected but signals mismatch in this invalid test case padding.",
            )

    @pytest.mark.schema
    def test_source_sectors_have_outflow(self):
        with pytest.raises(ValidationError, match="outflow"):
            _make_output(
                source_sectors=[_make_sector_flow("inflow", sector="SOFTWARE", cluster="growth")],
            )

    @pytest.mark.schema
    def test_dest_sectors_have_inflow(self):
        with pytest.raises(ValidationError, match="inflow"):
            _make_output(
                destination_sectors=[_make_sector_flow("outflow")],
            )

    @pytest.mark.schema
    def test_none_verdict_requires_none_type(self):
        with pytest.raises(ValidationError, match="NONE verdict requires NONE rotation_type"):
            _make_output(
                verdict=RotationStatus.NONE,
                signals_active=0,
                confidence=10,
                rotation_type=RotationType.CYCLICAL,
                rotation_stage=None,
                velocity=None,
                source_sectors=[],
                destination_sectors=[],
                summary="No rotation detected but type mismatch in this invalid test case for padding.",
            )

    @pytest.mark.schema
    def test_active_verdict_forbids_none_type(self):
        with pytest.raises(ValidationError, match="requires a rotation_type other than NONE"):
            _make_output(rotation_type=RotationType.NONE)

    @pytest.mark.schema
    def test_active_requires_stage(self):
        with pytest.raises(ValidationError, match="requires a rotation_stage"):
            _make_output(rotation_stage=None)

    @pytest.mark.schema
    def test_none_forbids_stage(self):
        with pytest.raises(ValidationError, match="should not have rotation_stage"):
            _make_output(
                verdict=RotationStatus.NONE,
                signals_active=0,
                confidence=10,
                rotation_type=RotationType.NONE,
                rotation_stage=RotationStage.EARLY,
                velocity=None,
                source_sectors=[],
                destination_sectors=[],
                summary="No rotation detected but stage present in this invalid test case padding text.",
            )

    @pytest.mark.schema
    def test_summary_min_length(self):
        with pytest.raises(ValidationError):
            _make_output(summary="Too short")

    @pytest.mark.schema
    def test_velocity_valid_values(self):
        for v in ("Slow", "Moderate", "Fast"):
            output = _make_output(velocity=v)
            assert output.velocity == v

    @pytest.mark.schema
    def test_velocity_invalid_rejects(self):
        with pytest.raises(ValidationError, match="velocity"):
            _make_output(velocity="VeryFast")

    @pytest.mark.schema
    def test_analysis_date_format(self):
        output = _make_output()
        assert output.analysis_date == "2026-02-10"
        with pytest.raises(ValidationError):
            _make_output(analysis_date="Feb 10, 2026")


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    @pytest.mark.schema
    def test_all_28_sectors_mapped(self):
        """Every IBD sector is in exactly one cluster."""
        mapped = set(SECTOR_TO_CLUSTER.keys())
        assert mapped == set(IBD_SECTORS)

    @pytest.mark.schema
    def test_cluster_names_valid(self):
        assert len(VALID_CLUSTER_NAMES) == 7
        assert "growth" in VALID_CLUSTER_NAMES
        assert "commodity" in VALID_CLUSTER_NAMES

    @pytest.mark.schema
    def test_sector_to_cluster_reverse_lookup(self):
        assert SECTOR_TO_CLUSTER["CHIPS"] == "growth"
        assert SECTOR_TO_CLUSTER["MINING"] == "commodity"
        assert SECTOR_TO_CLUSTER["UTILITIES"] == "defensive"
        assert SECTOR_TO_CLUSTER["BANKS"] == "financial"
        assert SECTOR_TO_CLUSTER["AEROSPACE"] == "industrial"
        assert SECTOR_TO_CLUSTER["RETAIL"] == "consumer_cyclical"
        assert SECTOR_TO_CLUSTER["TELECOM"] == "services"


# ---------------------------------------------------------------------------
# Rotation Narrative Field Tests
# ---------------------------------------------------------------------------

class TestRotationNarrativeFields:

    @pytest.mark.schema
    def test_narrative_defaults_to_none(self):
        """rotation_narrative defaults to None."""
        output = _make_output()
        assert output.rotation_narrative is None

    @pytest.mark.schema
    def test_narrative_source_defaults_to_none(self):
        """narrative_source defaults to None."""
        output = _make_output()
        assert output.narrative_source is None

    @pytest.mark.schema
    def test_narrative_source_llm_accepted(self):
        """narrative_source='llm' is valid."""
        output = _make_output(narrative_source="llm")
        assert output.narrative_source == "llm"

    @pytest.mark.schema
    def test_narrative_source_template_accepted(self):
        """narrative_source='template' is valid."""
        output = _make_output(narrative_source="template")
        assert output.narrative_source == "template"

    @pytest.mark.schema
    def test_narrative_source_invalid_rejected(self):
        """Invalid narrative_source is rejected."""
        with pytest.raises(ValueError, match="narrative_source"):
            _make_output(narrative_source="other")

    @pytest.mark.schema
    def test_narrative_with_sufficient_length(self):
        """rotation_narrative with >= 100 chars is accepted."""
        long_text = "A" * 150
        output = _make_output(rotation_narrative=long_text)
        assert output.rotation_narrative == long_text

    @pytest.mark.schema
    def test_narrative_too_short_rejected(self):
        """rotation_narrative with < 100 chars is rejected."""
        with pytest.raises(ValueError):
            _make_output(rotation_narrative="Too short")
