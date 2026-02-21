"""
Agent 15: Exit Strategist — Schema Tests
Level 1: Pure Pydantic validation, no LLM, no file I/O.
"""

from __future__ import annotations

import pytest

from ibd_agents.schemas.exit_strategy_output import (
    CLIMAX_SURGE_MIN_PCT,
    EARNINGS_MIN_CUSHION_PCT,
    EARNINGS_PROXIMITY_DAYS,
    EIGHT_WEEK_HOLD_DAYS,
    FAST_GAIN_MAX_DAYS,
    HARD_BACKSTOP_LOSS_PCT,
    HEALTH_SCORE_RANGE,
    PROFIT_TARGET_RANGE,
    REGIME_STOP_THRESHOLDS,
    RS_DROP_THRESHOLD,
    RS_MIN_THRESHOLD,
    RULE_ID_MAP,
    SECTOR_DIST_DAYS_THRESHOLD,
    SELL_RULE_NAMES,
    URGENCY_ORDER,
    EvidenceLink,
    ExitActionType,
    ExitMarketRegime,
    ExitStrategyOutput,
    PortfolioImpact,
    PositionSignal,
    SellRule,
    SellRuleResult,
    SellType,
    Urgency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evidence(
    rule: SellRule = SellRule.NONE,
    rule_id: str = "HEALTHY",
) -> EvidenceLink:
    return EvidenceLink(
        data_point="AAPL at +5.0%, RS 85, above 50-day MA — no sell signals triggered",
        rule_triggered=rule,
        rule_id=rule_id,
    )


def _make_signal(
    symbol: str = "AAPL",
    urgency: Urgency = Urgency.HEALTHY,
    action: ExitActionType = ExitActionType.HOLD,
    sell_type: SellType = SellType.NOT_APPLICABLE,
    sell_pct: float | None = None,
    rules_triggered: list[SellRule] | None = None,
    evidence: list[EvidenceLink] | None = None,
) -> PositionSignal:
    if rules_triggered is None:
        rules_triggered = [SellRule.NONE]
    if evidence is None:
        evidence = [_make_evidence()]
    return PositionSignal(
        symbol=symbol,
        tier=2,
        asset_type="stock",
        current_price=150.0,
        buy_price=142.86,
        gain_loss_pct=5.0,
        days_held=30,
        urgency=urgency,
        action=action,
        sell_type=sell_type,
        sell_pct=sell_pct,
        stop_price=132.86,
        rules_triggered=rules_triggered,
        evidence=evidence,
        reasoning="Position is within acceptable parameters, no sell rules triggered.",
        what_would_change="Monitor for RS deterioration below 70 or break below 50-day MA.",
    )


def _make_impact(
    n_critical: int = 0,
    n_warning: int = 0,
    n_watch: int = 0,
    n_healthy: int = 1,
) -> PortfolioImpact:
    return PortfolioImpact(
        current_cash_pct=2.0,
        projected_cash_pct=5.0,
        current_top_holding_pct=5.0,
        projected_top_holding_pct=4.0,
        positions_healthy=n_healthy,
        positions_watch=n_watch,
        positions_warning=n_warning,
        positions_critical=n_critical,
        sector_concentration_risk="LOW",
    )


def _make_output(
    signals: list[PositionSignal] | None = None,
    impact: PortfolioImpact | None = None,
) -> ExitStrategyOutput:
    if signals is None:
        signals = [_make_signal()]
    if impact is None:
        n_c = sum(1 for s in signals if s.urgency == Urgency.CRITICAL)
        n_w = sum(1 for s in signals if s.urgency == Urgency.WARNING)
        n_wa = sum(1 for s in signals if s.urgency == Urgency.WATCH)
        n_h = sum(1 for s in signals if s.urgency == Urgency.HEALTHY)
        impact = _make_impact(n_c, n_w, n_wa, n_h)
    return ExitStrategyOutput(
        analysis_date="2026-02-21",
        market_regime=ExitMarketRegime.CONFIRMED_UPTREND,
        portfolio_health_score=8,
        signals=signals,
        portfolio_impact=impact,
        summary="Exit Strategy: 1 positions evaluated in CONFIRMED_UPTREND regime. 0 critical, 0 warning, 0 watch, 1 healthy. Portfolio health score: 8/10.",
        reasoning_source="deterministic",
    )


# ===========================================================================
# TestConstants
# ===========================================================================

class TestConstants:
    """Validate constant definitions match spec."""

    def test_sell_rule_count(self):
        assert len(SELL_RULE_NAMES) == 8

    def test_urgency_levels(self):
        assert set(URGENCY_ORDER.keys()) == {"CRITICAL", "WARNING", "WATCH", "HEALTHY"}

    def test_urgency_order(self):
        assert URGENCY_ORDER["CRITICAL"] < URGENCY_ORDER["WARNING"]
        assert URGENCY_ORDER["WARNING"] < URGENCY_ORDER["WATCH"]
        assert URGENCY_ORDER["WATCH"] < URGENCY_ORDER["HEALTHY"]

    def test_action_types(self):
        actions = {a.value for a in ExitActionType}
        assert actions == {"SELL_ALL", "TRIM", "TIGHTEN_STOP", "HOLD", "HOLD_8_WEEK"}

    def test_market_regimes(self):
        regimes = {r.value for r in ExitMarketRegime}
        assert regimes == {
            "CONFIRMED_UPTREND",
            "UPTREND_UNDER_PRESSURE",
            "CORRECTION",
            "RALLY_ATTEMPT",
        }

    def test_regime_stop_thresholds(self):
        assert len(REGIME_STOP_THRESHOLDS) == 4
        for regime_str, (low, high) in REGIME_STOP_THRESHOLDS.items():
            assert low < high, f"{regime_str}: low {low} >= high {high}"

    def test_profit_target_range(self):
        assert PROFIT_TARGET_RANGE == (20.0, 25.0)

    def test_eight_week_hold_days(self):
        assert EIGHT_WEEK_HOLD_DAYS == 56

    def test_fast_gain_max_days(self):
        assert FAST_GAIN_MAX_DAYS == 21

    def test_climax_surge_min(self):
        assert CLIMAX_SURGE_MIN_PCT == 25.0

    def test_rs_thresholds(self):
        assert RS_MIN_THRESHOLD == 70
        assert RS_DROP_THRESHOLD == 15

    def test_sector_dist_threshold(self):
        assert SECTOR_DIST_DAYS_THRESHOLD == 4

    def test_earnings_constants(self):
        assert EARNINGS_PROXIMITY_DAYS == 21
        assert EARNINGS_MIN_CUSHION_PCT == 15.0

    def test_hard_backstop(self):
        assert HARD_BACKSTOP_LOSS_PCT == 10.0

    def test_health_score_range(self):
        assert HEALTH_SCORE_RANGE == (1, 10)

    def test_rule_id_map_covers_all_rules(self):
        for rule in SellRule:
            assert rule.value in RULE_ID_MAP


# ===========================================================================
# TestEvidenceLink
# ===========================================================================

class TestEvidenceLink:
    """Test EvidenceLink model validation."""

    def test_valid_evidence_link(self):
        e = _make_evidence()
        assert e.rule_triggered == SellRule.NONE
        assert e.rule_id == "HEALTHY"

    def test_data_point_min_length(self):
        with pytest.raises(Exception):
            EvidenceLink(
                data_point="short",
                rule_triggered=SellRule.NONE,
                rule_id="HEALTHY",
            )

    def test_rule_id_min_length(self):
        with pytest.raises(Exception):
            EvidenceLink(
                data_point="A sufficiently long data point observation string",
                rule_triggered=SellRule.NONE,
                rule_id="AB",
            )


# ===========================================================================
# TestSellRuleResult
# ===========================================================================

class TestSellRuleResult:
    """Test SellRuleResult model."""

    def test_valid_triggered(self):
        r = SellRuleResult(
            rule=SellRule.STOP_LOSS_7_8,
            triggered=True,
            value=-8.5,
            threshold=-7.0,
            detail="AAPL down 8.5% from buy price — breached 7-8% stop loss",
        )
        assert r.triggered is True
        assert r.rule == SellRule.STOP_LOSS_7_8

    def test_valid_not_triggered(self):
        r = SellRuleResult(
            rule=SellRule.CLIMAX_TOP,
            triggered=False,
            value=5.0,
            threshold=25.0,
            detail="AAPL 3-week surge 5.0% with 1.0x volume — no climax top detected",
        )
        assert r.triggered is False

    def test_detail_min_length(self):
        with pytest.raises(Exception):
            SellRuleResult(
                rule=SellRule.STOP_LOSS_7_8,
                triggered=True,
                value=-8.0,
                threshold=-7.0,
                detail="too short",
            )


# ===========================================================================
# TestPositionSignal
# ===========================================================================

class TestPositionSignal:
    """Test PositionSignal validators."""

    def test_valid_healthy_hold(self):
        sig = _make_signal()
        assert sig.urgency == Urgency.HEALTHY
        assert sig.action == ExitActionType.HOLD
        assert sig.sell_type == SellType.NOT_APPLICABLE

    def test_sell_all_defensive(self):
        sig = _make_signal(
            urgency=Urgency.CRITICAL,
            action=ExitActionType.SELL_ALL,
            sell_type=SellType.DEFENSIVE,
            sell_pct=100.0,
            rules_triggered=[SellRule.STOP_LOSS_7_8],
            evidence=[_make_evidence(SellRule.STOP_LOSS_7_8, "MUST-M2")],
        )
        assert sig.action == ExitActionType.SELL_ALL
        assert sig.sell_type == SellType.DEFENSIVE
        assert sig.sell_pct == 100.0

    def test_sell_all_offensive(self):
        sig = _make_signal(
            urgency=Urgency.CRITICAL,
            action=ExitActionType.SELL_ALL,
            sell_type=SellType.OFFENSIVE,
            sell_pct=100.0,
            rules_triggered=[SellRule.CLIMAX_TOP],
            evidence=[_make_evidence(SellRule.CLIMAX_TOP, "MUST-M3")],
        )
        assert sig.sell_type == SellType.OFFENSIVE

    def test_trim_requires_sell_pct_1_99(self):
        sig = _make_signal(
            urgency=Urgency.WARNING,
            action=ExitActionType.TRIM,
            sell_type=SellType.DEFENSIVE,
            sell_pct=50.0,
            rules_triggered=[SellRule.EARNINGS_RISK],
            evidence=[_make_evidence(SellRule.EARNINGS_RISK, "MUST_NOT-S2")],
        )
        assert sig.sell_pct == 50.0

    def test_trim_rejects_100_pct(self):
        with pytest.raises(Exception):
            _make_signal(
                urgency=Urgency.WARNING,
                action=ExitActionType.TRIM,
                sell_type=SellType.DEFENSIVE,
                sell_pct=100.0,
                rules_triggered=[SellRule.EARNINGS_RISK],
                evidence=[_make_evidence(SellRule.EARNINGS_RISK, "MUST_NOT-S2")],
            )

    def test_sell_requires_non_na_sell_type(self):
        with pytest.raises(Exception):
            _make_signal(
                urgency=Urgency.CRITICAL,
                action=ExitActionType.SELL_ALL,
                sell_type=SellType.NOT_APPLICABLE,
                sell_pct=100.0,
                rules_triggered=[SellRule.STOP_LOSS_7_8],
                evidence=[_make_evidence(SellRule.STOP_LOSS_7_8, "MUST-M2")],
            )

    def test_hold_requires_na_sell_type(self):
        with pytest.raises(Exception):
            _make_signal(
                urgency=Urgency.HEALTHY,
                action=ExitActionType.HOLD,
                sell_type=SellType.OFFENSIVE,
            )

    def test_tighten_stop_requires_na_sell_type(self):
        sig = _make_signal(
            urgency=Urgency.WARNING,
            action=ExitActionType.TIGHTEN_STOP,
            sell_type=SellType.NOT_APPLICABLE,
            rules_triggered=[SellRule.RS_DETERIORATION],
            evidence=[_make_evidence(SellRule.RS_DETERIORATION, "MUST-M5")],
        )
        assert sig.action == ExitActionType.TIGHTEN_STOP

    def test_hold_8_week(self):
        sig = _make_signal(
            urgency=Urgency.HEALTHY,
            action=ExitActionType.HOLD_8_WEEK,
            sell_type=SellType.NOT_APPLICABLE,
        )
        assert sig.action == ExitActionType.HOLD_8_WEEK

    def test_evidence_required_for_triggered_rules(self):
        """Rules in rules_triggered must have matching evidence."""
        with pytest.raises(Exception):
            _make_signal(
                urgency=Urgency.WARNING,
                action=ExitActionType.TIGHTEN_STOP,
                sell_type=SellType.NOT_APPLICABLE,
                rules_triggered=[SellRule.RS_DETERIORATION, SellRule.SECTOR_DISTRIBUTION],
                evidence=[_make_evidence(SellRule.RS_DETERIORATION, "MUST-M5")],
                # Missing evidence for SECTOR_DISTRIBUTION
            )

    def test_symbol_uppercased(self):
        sig = _make_signal(symbol="aapl")
        assert sig.symbol == "AAPL"


# ===========================================================================
# TestPortfolioImpact
# ===========================================================================

class TestPortfolioImpact:
    """Test PortfolioImpact model validation."""

    def test_valid_impact(self):
        impact = _make_impact()
        assert impact.sector_concentration_risk == "LOW"

    def test_concentration_risk_high(self):
        impact = _make_impact()
        impact_high = PortfolioImpact(
            **{**impact.model_dump(), "sector_concentration_risk": "HIGH"}
        )
        assert impact_high.sector_concentration_risk == "HIGH"

    def test_concentration_risk_medium(self):
        impact_med = PortfolioImpact(
            **{**_make_impact().model_dump(), "sector_concentration_risk": "MEDIUM"}
        )
        assert impact_med.sector_concentration_risk == "MEDIUM"

    def test_invalid_concentration_risk(self):
        with pytest.raises(Exception):
            PortfolioImpact(
                **{**_make_impact().model_dump(), "sector_concentration_risk": "NONE"}
            )


# ===========================================================================
# TestExitStrategyOutput
# ===========================================================================

class TestExitStrategyOutput:
    """Test top-level output validators."""

    def test_valid_output(self):
        out = _make_output()
        assert out.portfolio_health_score == 8
        assert out.reasoning_source == "deterministic"

    def test_signals_must_be_ordered_by_urgency(self):
        """CRITICAL must come before WARNING before WATCH before HEALTHY."""
        critical_sig = _make_signal(
            symbol="NVDA",
            urgency=Urgency.CRITICAL,
            action=ExitActionType.SELL_ALL,
            sell_type=SellType.DEFENSIVE,
            sell_pct=100.0,
            rules_triggered=[SellRule.STOP_LOSS_7_8],
            evidence=[_make_evidence(SellRule.STOP_LOSS_7_8, "MUST-M2")],
        )
        healthy_sig = _make_signal(symbol="AAPL")

        # Correct order: CRITICAL then HEALTHY
        out = _make_output(signals=[critical_sig, healthy_sig])
        assert out.signals[0].urgency == Urgency.CRITICAL

    def test_signals_wrong_order_rejected(self):
        critical_sig = _make_signal(
            symbol="NVDA",
            urgency=Urgency.CRITICAL,
            action=ExitActionType.SELL_ALL,
            sell_type=SellType.DEFENSIVE,
            sell_pct=100.0,
            rules_triggered=[SellRule.STOP_LOSS_7_8],
            evidence=[_make_evidence(SellRule.STOP_LOSS_7_8, "MUST-M2")],
        )
        healthy_sig = _make_signal(symbol="AAPL")

        with pytest.raises(Exception):
            _make_output(signals=[healthy_sig, critical_sig])

    def test_impact_counts_must_match_signals(self):
        """Impact urgency counts must match actual signal counts."""
        sig = _make_signal()
        wrong_impact = _make_impact(n_critical=1, n_healthy=1)  # Wrong: no CRITICAL signal
        with pytest.raises(Exception):
            _make_output(signals=[sig], impact=wrong_impact)

    def test_health_score_min(self):
        with pytest.raises(Exception):
            ExitStrategyOutput(
                analysis_date="2026-02-21",
                market_regime=ExitMarketRegime.CONFIRMED_UPTREND,
                portfolio_health_score=0,  # Below 1
                signals=[_make_signal()],
                portfolio_impact=_make_impact(),
                summary="Exit Strategy: 1 positions evaluated. Portfolio health score: 0/10. This is too low.",
                reasoning_source="deterministic",
            )

    def test_health_score_max(self):
        with pytest.raises(Exception):
            ExitStrategyOutput(
                analysis_date="2026-02-21",
                market_regime=ExitMarketRegime.CONFIRMED_UPTREND,
                portfolio_health_score=11,  # Above 10
                signals=[_make_signal()],
                portfolio_impact=_make_impact(),
                summary="Exit Strategy: 1 positions evaluated. Portfolio health score: 11/10. Too high.",
                reasoning_source="deterministic",
            )

    def test_reasoning_source_llm(self):
        out = _make_output()
        out2 = ExitStrategyOutput(**{**out.model_dump(), "reasoning_source": "llm"})
        assert out2.reasoning_source == "llm"

    def test_reasoning_source_invalid(self):
        with pytest.raises(Exception):
            ExitStrategyOutput(**{**_make_output().model_dump(), "reasoning_source": "magic"})

    def test_analysis_date_format(self):
        with pytest.raises(Exception):
            ExitStrategyOutput(
                **{**_make_output().model_dump(), "analysis_date": "02-21-2026"}
            )

    def test_min_one_signal(self):
        with pytest.raises(Exception):
            ExitStrategyOutput(
                analysis_date="2026-02-21",
                market_regime=ExitMarketRegime.CONFIRMED_UPTREND,
                portfolio_health_score=8,
                signals=[],
                portfolio_impact=_make_impact(n_healthy=0),
                summary="Exit Strategy: 0 positions evaluated. No signals. Empty. Nothing to report here.",
                reasoning_source="deterministic",
            )
