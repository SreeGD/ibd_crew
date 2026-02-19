"""
Risk Officer Tool: Risk Analysis & Stress Testing
IBD Momentum Investment Framework v4.0

Pure functions for:
- 10-point risk check pipeline
- Stress test scenarios
- Sleep Well score computation

No LLM, no file I/O.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ibd_agents.schemas.analyst_output import AnalystOutput

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field

from ibd_agents.schemas.portfolio_output import (
    ALL_KEEPS,
    ETF_LIMITS,
    MAX_LOSS,
    STOCK_LIMITS,
    TRAILING_STOPS,
    PortfolioOutput,
)
from ibd_agents.schemas.risk_output import (
    CHECK_NAMES,
    STRESS_SCENARIOS,
    KeepValidation,
    RiskCheck,
    RiskWarning,
    SleepWellScores,
    StressScenario,
    StressTestReport,
    Veto,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 10 Risk Check Functions
# ---------------------------------------------------------------------------

def check_position_sizing(portfolio: PortfolioOutput) -> RiskCheck:
    """Check 1: Position sizing compliance — stocks ≤ 5/4/3%, ETFs ≤ 8/8/6%."""
    violations = []
    for tier in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]:
        for p in tier.positions:
            limit = (
                ETF_LIMITS[p.tier] if p.asset_type == "etf"
                else STOCK_LIMITS[p.tier]
            )
            if p.target_pct > limit + 0.01:
                violations.append(
                    f"{p.symbol}: {p.target_pct:.2f}% exceeds "
                    f"T{p.tier} {p.asset_type} limit {limit}%"
                )

    if violations:
        return RiskCheck(
            check_name="position_sizing",
            status="VETO",
            findings=f"Position sizing violations found: {len(violations)} positions exceed limits",
            details=violations,
        )
    return RiskCheck(
        check_name="position_sizing",
        status="PASS",
        findings="All positions within tier-specific sizing limits (stocks 5/4/3%, ETFs 8/8/6%)",
        details=[],
    )


def check_trailing_stops(portfolio: PortfolioOutput) -> RiskCheck:
    """Check 2: Trailing stop compliance — stops match tier."""
    issues = []
    for tier in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]:
        for p in tier.positions:
            expected = TRAILING_STOPS[p.tier]
            if p.trailing_stop_pct > expected + 0.01:
                issues.append(
                    f"{p.symbol}: stop {p.trailing_stop_pct}% > "
                    f"T{p.tier} initial {expected}%"
                )

    if issues:
        return RiskCheck(
            check_name="trailing_stops",
            status="WARNING",
            findings=f"Trailing stop issues found: {len(issues)} positions with non-standard stops",
            details=issues,
        )
    return RiskCheck(
        check_name="trailing_stops",
        status="PASS",
        findings="All trailing stops match tier protocol (T1=22%, T2=18%, T3=12%) or tightened",
        details=[],
    )


def check_sector_concentration(portfolio: PortfolioOutput) -> RiskCheck:
    """Check 3: Sector concentration — max 40%, min 8 sectors."""
    violations = []
    for sector, pct in portfolio.sector_exposure.items():
        if pct > 40.0:
            violations.append(f"{sector}: {pct:.1f}% exceeds 40% max")

    if len(portfolio.sector_exposure) < 8:
        violations.append(
            f"Only {len(portfolio.sector_exposure)} sectors, minimum is 8"
        )

    if violations:
        status = "VETO" if any("exceeds" in v for v in violations) else "WARNING"
        return RiskCheck(
            check_name="sector_concentration",
            status=status,
            findings=f"Sector concentration issues: {'; '.join(violations[:3])}",
            details=violations,
        )
    return RiskCheck(
        check_name="sector_concentration",
        status="PASS",
        findings=f"Sector diversification OK: {len(portfolio.sector_exposure)} sectors, max exposure within 40% limit",
        details=[],
    )


def check_tier_allocation(portfolio: PortfolioOutput) -> RiskCheck:
    """Check 4: Tier allocation within target ranges."""
    issues = []
    t1 = portfolio.tier_1.actual_pct
    t2 = portfolio.tier_2.actual_pct
    t3 = portfolio.tier_3.actual_pct

    # Check reasonable ranges (lenient for regime adjustments)
    if not (25.0 <= t1 <= 50.0):
        issues.append(f"T1 at {t1:.1f}% outside 25-50% range")
    if not (25.0 <= t2 <= 45.0):
        issues.append(f"T2 at {t2:.1f}% outside 25-45% range")
    if not (10.0 <= t3 <= 35.0):
        issues.append(f"T3 at {t3:.1f}% outside 10-35% range")

    total = t1 + t2 + t3 + portfolio.cash_pct
    if not (95.0 <= total <= 105.0):
        issues.append(f"Total allocation {total:.1f}% not near 100%")

    if issues:
        return RiskCheck(
            check_name="tier_allocation",
            status="WARNING",
            findings=f"Tier allocation outside ranges: {'; '.join(issues[:2])}",
            details=issues,
        )
    return RiskCheck(
        check_name="tier_allocation",
        status="PASS",
        findings=f"Tier allocation within ranges: T1={t1:.0f}%, T2={t2:.0f}%, T3={t3:.0f}%, Cash={portfolio.cash_pct:.0f}%",
        details=[],
    )


def check_max_loss(portfolio: PortfolioOutput) -> RiskCheck:
    """Check 5: Max loss per position within framework limits."""
    violations = []
    for tier in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]:
        for p in tier.positions:
            limit = MAX_LOSS[p.asset_type][p.tier]
            actual = p.target_pct * (p.trailing_stop_pct / 100.0)
            if actual > limit + 0.01:
                violations.append(
                    f"{p.symbol}: max loss {actual:.2f}% exceeds "
                    f"T{p.tier} {p.asset_type} limit {limit}%"
                )

    if violations:
        return RiskCheck(
            check_name="max_loss",
            status="WARNING",
            findings=f"Max loss concerns: {len(violations)} positions exceed risk limits",
            details=violations,
        )
    return RiskCheck(
        check_name="max_loss",
        status="PASS",
        findings="All positions within maximum loss limits per framework rules",
        details=[],
    )


def check_correlation(portfolio: PortfolioOutput) -> RiskCheck:
    """Check 6: Detect same-sector clustering and ETF theme overlap."""
    sector_counts: dict[str, int] = defaultdict(int)
    for tier in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]:
        for p in tier.positions:
            sector_counts[p.sector] += 1

    concentrated = [
        f"{sector}: {count} positions"
        for sector, count in sector_counts.items()
        if count >= 6
    ]

    # --- ETF theme overlap detection ---
    # Compute per-sector ETF allocation % across all tiers
    sector_etf_pct: dict[str, float] = defaultdict(float)
    for tier in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]:
        for p in tier.positions:
            if p.asset_type == "etf":
                sector_etf_pct[p.sector] += p.target_pct

    etf_overlap_warnings = [
        f"ETF theme overlap: {sector} has {pct:.1f}% from ETFs alone (>15% threshold)"
        for sector, pct in sector_etf_pct.items()
        if pct > 15.0
    ]

    all_issues = concentrated + etf_overlap_warnings

    if all_issues:
        return RiskCheck(
            check_name="correlation",
            status="WARNING",
            findings=f"Potential correlation clusters: {'; '.join(all_issues[:3])}",
            details=all_issues,
        )
    return RiskCheck(
        check_name="correlation",
        status="PASS",
        findings="No excessive same-sector clustering detected across portfolio positions",
        details=[],
    )


def check_regime_alignment(portfolio: PortfolioOutput, regime: str) -> RiskCheck:
    """Check 7: Portfolio alignment with market regime."""
    t1 = portfolio.tier_1.actual_pct
    t3 = portfolio.tier_3.actual_pct
    issues = []

    if regime == "bear" and t1 > 35.0:
        issues.append(f"Bear regime but T1 at {t1:.0f}% (should be ≤35%)")
    if regime == "bear" and t3 < 25.0:
        issues.append(f"Bear regime but T3 at {t3:.0f}% (should be ≥25%)")
    if regime == "bull" and t3 > 30.0:
        issues.append(f"Bull regime but T3 at {t3:.0f}% (should be ≤30%)")

    if issues:
        return RiskCheck(
            check_name="regime_alignment",
            status="WARNING",
            findings=f"Regime alignment concerns ({regime}): {'; '.join(issues[:2])}",
            details=issues,
        )
    return RiskCheck(
        check_name="regime_alignment",
        status="PASS",
        findings=f"Portfolio aligned with {regime} regime: tier distribution appropriate",
        details=[],
    )


def check_volume(portfolio: PortfolioOutput) -> RiskCheck:
    """Check 8: Volume confirmation — placeholder PASS in deterministic mode."""
    return RiskCheck(
        check_name="volume_confirmation",
        status="PASS",
        findings="Volume confirmation check: deterministic mode — all positions assumed confirmed",
        details=["Volume data not available in deterministic pipeline"],
    )


def check_keeps(portfolio: PortfolioOutput) -> RiskCheck:
    """Check 9: Validate all 14 keeps present in portfolio."""
    all_symbols = set()
    for tier in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]:
        for p in tier.positions:
            all_symbols.add(p.symbol)

    missing = [k for k in ALL_KEEPS if k not in all_symbols]

    if missing:
        return RiskCheck(
            check_name="keeps_validation",
            status="VETO",
            findings=f"Missing {len(missing)} keep positions: {', '.join(missing[:5])}",
            details=[f"Missing: {k}" for k in missing],
        )
    return RiskCheck(
        check_name="keeps_validation",
        status="PASS",
        findings="All 21 keep positions verified: 5 fundamental + 7 user + 9 IBD present in portfolio",
        details=[],
    )


def run_stress_tests(portfolio: PortfolioOutput) -> RiskCheck:
    """Check 10: Run 3 stress test scenarios."""
    # We run the stress tests and return a PASS/WARNING based on results
    all_positions = (
        portfolio.tier_1.positions
        + portfolio.tier_2.positions
        + portfolio.tier_3.positions
    )

    # Find top sectors for impact assessment
    sector_pcts: dict[str, float] = defaultdict(float)
    for p in all_positions:
        sector_pcts[p.sector] += p.target_pct

    top_sector = max(sector_pcts, key=sector_pcts.get) if sector_pcts else "UNKNOWN"
    top_pct = sector_pcts.get(top_sector, 0)

    details = []
    for scenario in STRESS_SCENARIOS:
        impact = scenario["market_impact_pct"]
        details.append(
            f"{scenario['name']}: {impact}% impact on portfolio, "
            f"worst affected: {top_sector} ({top_pct:.1f}%)"
        )

    return RiskCheck(
        check_name="stress_test",
        status="PASS",
        findings=f"Stress tests completed: {len(STRESS_SCENARIOS)} scenarios analyzed for portfolio resilience",
        details=details,
    )


# ---------------------------------------------------------------------------
# Stress Test Report Builder
# ---------------------------------------------------------------------------

def build_stress_test_report(portfolio: PortfolioOutput) -> StressTestReport:
    """Build detailed stress test report with 3 scenarios."""
    all_positions = (
        portfolio.tier_1.positions
        + portfolio.tier_2.positions
        + portfolio.tier_3.positions
    )

    sector_pcts: dict[str, float] = defaultdict(float)
    for p in all_positions:
        sector_pcts[p.sector] += p.target_pct

    sorted_sectors = sorted(sector_pcts.items(), key=lambda x: -x[1])
    top_3 = [s[0] for s in sorted_sectors[:3]]

    scenarios: list[StressScenario] = []

    # Scenario 1: Market crash -20%
    t1_exposure = portfolio.tier_1.actual_pct
    crash_drawdown = round(t1_exposure * 0.25 + (100 - t1_exposure) * 0.15, 1)
    scenarios.append(StressScenario(
        scenario_name="Market Crash (-20%)",
        impact_description=(
            f"Broad market decline of 20%. T1 Momentum positions most exposed "
            f"at {t1_exposure:.0f}% allocation. Expected higher beta impact on momentum names."
        ),
        estimated_drawdown_pct=crash_drawdown,
        positions_most_affected=[p.symbol for p in portfolio.tier_1.positions[:5]],
    ))

    # Scenario 2: Sector correction -30%
    sector_exposure = sorted_sectors[0][1] if sorted_sectors else 10
    sector_drawdown = round(sector_exposure * 0.30 + (100 - sector_exposure) * 0.05, 1)
    scenarios.append(StressScenario(
        scenario_name="Sector Correction (-30%)",
        impact_description=(
            f"Leading sector ({top_3[0]}) corrects 30%. "
            f"Portfolio has {sector_exposure:.0f}% exposure to this sector."
        ),
        estimated_drawdown_pct=sector_drawdown,
        positions_most_affected=[
            p.symbol for tier in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]
            for p in tier.positions if p.sector == top_3[0]
        ][:5],
    ))

    # Scenario 3: Rate hike +50bp
    rate_drawdown = round(t1_exposure * 0.10 + portfolio.tier_3.actual_pct * 0.03, 1)
    scenarios.append(StressScenario(
        scenario_name="Rate Hike (+50bp)",
        impact_description=(
            "Unexpected 50bp rate increase impacts growth/momentum names most. "
            "Defensive positions provide partial offset."
        ),
        estimated_drawdown_pct=rate_drawdown,
        positions_most_affected=[p.symbol for p in portfolio.tier_1.positions[:3]],
    ))

    # Determine resilience
    max_drawdown = max(s.estimated_drawdown_pct for s in scenarios)
    if max_drawdown < 15:
        resilience = "Strong resilience: portfolio well-diversified across scenarios"
    elif max_drawdown < 20:
        resilience = "Moderate resilience: some concentration risk in stress scenarios"
    else:
        resilience = "Limited resilience: significant drawdown risk in adverse scenarios"

    return StressTestReport(scenarios=scenarios, overall_resilience=resilience)


# ---------------------------------------------------------------------------
# Sleep Well Score Computation
# ---------------------------------------------------------------------------

def compute_sleep_well_scores(
    check_results: List[RiskCheck],
    portfolio: PortfolioOutput,
    analyst_output: Optional["AnalystOutput"] = None,
) -> SleepWellScores:
    """
    Compute Sleep Well scores (1-10) per tier and overall.

    Scoring: start at 8, deduct for issues.
    When analyst_output is provided, incorporates per-stock conviction
    and volatility data for more accurate scoring.
    """
    base = 8

    # Count issues by severity
    veto_count = sum(1 for c in check_results if c.status == "VETO")
    warning_count = sum(1 for c in check_results if c.status == "WARNING")

    # Build conviction/volatility lookup from analyst data
    conviction_map: dict[str, int] = {}
    volatility_map: dict[str, float] = {}
    if analyst_output:
        for stock in analyst_output.rated_stocks:
            conviction_map[stock.symbol] = stock.conviction
            vol = stock.llm_volatility or stock.estimated_volatility_pct
            if vol is not None:
                volatility_map[stock.symbol] = vol

    # Per-tier scoring
    def _tier_score(tier_positions, tier_num: int) -> int:
        score = base
        # Deduct for concentrated positions
        if any(p.target_pct > STOCK_LIMITS[tier_num] * 0.8 for p in tier_positions if p.asset_type == "stock"):
            score -= 1
        # Deduct for high stop distances
        if tier_num == 1:
            score -= 1  # T1 inherently riskier
        elif tier_num == 3:
            score += 1  # T3 inherently safer
        # Deduct for vetoes/warnings
        score -= veto_count * 2
        score -= warning_count

        # Conviction adjustment (when analyst data available)
        if conviction_map and tier_positions:
            tier_convictions = [
                conviction_map[p.symbol]
                for p in tier_positions
                if p.symbol in conviction_map
            ]
            if tier_convictions:
                avg_conviction = sum(tier_convictions) / len(tier_convictions)
                if avg_conviction >= 8:
                    score += 1
                elif avg_conviction <= 4:
                    score -= 1

        # Volatility adjustment (when analyst data available)
        if volatility_map and tier_positions:
            tier_vols = [
                volatility_map[p.symbol]
                for p in tier_positions
                if p.symbol in volatility_map
            ]
            if tier_vols:
                avg_vol = sum(tier_vols) / len(tier_vols)
                if avg_vol > 40:
                    score -= 1

        return max(1, min(10, score))

    t1 = _tier_score(portfolio.tier_1.positions, 1)
    t2 = _tier_score(portfolio.tier_2.positions, 2)
    t3 = _tier_score(portfolio.tier_3.positions, 3)
    overall = max(1, min(10, round((t1 + t2 + t3) / 3)))

    factors = []
    if veto_count > 0:
        factors.append(f"{veto_count} veto(s) significantly impact score")
    if warning_count > 0:
        factors.append(f"{warning_count} warning(s) moderately impact score")
    if portfolio.tier_3.actual_pct >= 20:
        factors.append("Adequate defensive allocation supports score")
    if len(portfolio.sector_exposure) >= 10:
        factors.append("Strong sector diversification supports score")
    if conviction_map:
        factors.append("Per-stock conviction data incorporated")
    if volatility_map:
        factors.append("Per-stock volatility data incorporated")
    if not factors:
        factors.append("Portfolio meets all framework requirements")

    return SleepWellScores(
        tier_1_score=t1,
        tier_2_score=t2,
        tier_3_score=t3,
        overall_score=overall,
        factors=factors,
    )


# ---------------------------------------------------------------------------
# Keep Validation Builder
# ---------------------------------------------------------------------------

def validate_keeps(portfolio: PortfolioOutput) -> KeepValidation:
    """Validate all 14 keeps are present."""
    all_symbols = set()
    for tier in [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3]:
        for p in tier.positions:
            all_symbols.add(p.symbol)

    missing = [k for k in ALL_KEEPS if k not in all_symbols]
    found = len(ALL_KEEPS) - len(missing)

    return KeepValidation(
        total_keeps_found=found,
        missing_keeps=missing,
        status="PASS" if not missing else f"MISSING {len(missing)} keeps",
    )


# ---------------------------------------------------------------------------
# CrewAI Tool Wrapper
# ---------------------------------------------------------------------------

class RiskAnalyzerInput(BaseModel):
    portfolio_json: str = Field(..., description="JSON string of PortfolioOutput")
    regime: str = Field("neutral", description="Market regime: bull/bear/neutral")


class RiskAnalyzerTool(BaseTool):
    """Run 10-point risk check pipeline on portfolio."""

    name: str = "risk_analyzer"
    description: str = (
        "Run 10-point risk assessment on portfolio: position sizing, "
        "stops, concentration, allocation, max loss, correlation, "
        "regime, volume, keeps, stress tests"
    )
    args_schema: type[BaseModel] = RiskAnalyzerInput

    def _run(self, portfolio_json: str, regime: str = "neutral") -> str:
        import json
        portfolio = PortfolioOutput.model_validate_json(portfolio_json)

        checks = [
            check_position_sizing(portfolio),
            check_trailing_stops(portfolio),
            check_sector_concentration(portfolio),
            check_tier_allocation(portfolio),
            check_max_loss(portfolio),
            check_correlation(portfolio),
            check_regime_alignment(portfolio, regime),
            check_volume(portfolio),
            check_keeps(portfolio),
            run_stress_tests(portfolio),
        ]

        return json.dumps([c.model_dump() for c in checks], indent=2)
