# Agent 06: Risk Officer ⚠️

> **Build Phase:** 6
> **Depends On:** Portfolio Manager output
> **Feeds Into:** Portfolio Manager (feedback loop), Educator
> **Target:** Week 7
> **Methodology:** IBD Momentum Investment Framework v4.0

---

## 1. Identity

| Field | Value |
|---|---|
| **Role** | Chief Risk Officer |
| **Experience** | Survived 2000, 2008, 2020 crashes. Veteran guardian. |
| **Personality** | Cautious, thorough, protective. Voice of reason. |
| **Temperature** | 0.3 (very conservative, risk-focused) |
| **Delegation** | NOT allowed (Risk Officer has final authority) |

---

## 2. Goal

> Review all portfolio positions against Framework v4.0 risk limits, validate trailing stops, detect concentration breaches, assess market regime, run stress tests, assign Sleep Well scores, and **veto** portfolios that violate hard constraints. The only agent that can force the Portfolio Manager to rebuild.

---

## 3. ✅ DOES / ❌ DOES NOT

### ✅ DOES
- Validate every position against tier-specific stop-loss rules (22/18/12%)
- Check stop-tightening compliance (after 10% and 20% gains)
- Verify position sizing limits (stocks: 5/4/3%, ETFs: 8/8/6%)
- Check sector concentration (max 40%, min 8 sectors)
- Calculate max loss per position against framework limits
- Detect hidden correlations (stocks that fall together)
- Detect thematic concentration masking as sector diversity
- Run stress test scenarios (rate hike, market crash, sector correction)
- Validate volume confirmation signals for recent entries
- Assess market regime (bull/bear) and check portfolio alignment
- Assign Sleep Well score (1-10) per tier and overall
- Issue vetoes with specific fix requirements
- Verify keep-position compliance (14 keeps properly categorized)

### ❌ DOES NOT
- Pick stocks or build portfolios (only reviews)
- Make sector allocation recommendations
- Discover new stocks or add to universe
- Execute trades or implement changes
- Provide optimistic return projections

---

## 4. Framework v4.0 Rules (Hard-Coded)

### 4.1 Trailing Stop Protocol

```python
TRAILING_STOPS = {
    # Initial stops by tier
    "initial": {1: 22, 2: 18, 3: 12},
    # After position gains 10%
    "after_10pct": {1: (15, 18), 2: (12, 15), 3: (8, 10)},
    # After position gains 20%
    "after_20pct": {1: (10, 12), 2: (8, 10),  3: (5, 8)},
}
```

### 4.2 Maximum Loss Per Position

```python
MAX_LOSS_PCT = {
    # % of total portfolio at risk per position
    "stock": {1: 1.10, 2: 0.72, 3: 0.36},
    "etf":   {1: 1.76, 2: 1.44, 3: 0.72},
}

# At $1.5M portfolio:
MAX_LOSS_DOLLARS = {
    "stock": {1: 17000, 2: 11000, 3: 6000},
    "etf":   {1: 27000, 2: 22000, 3: 11000},
}
```

### 4.3 Position Sizing Limits

```python
POSITION_LIMITS = {
    "stock": {1: 5, 2: 4, 3: 3},   # max % of portfolio
    "etf":   {1: 8, 2: 8, 3: 6},
}
```

### 4.4 Diversification Constraints

```python
DIVERSIFICATION = {
    "max_single_sector":  40,  # %
    "min_sectors":         8,
    "max_single_stock":    5,  # %
    "max_single_etf":      8,  # %
    "stock_etf_split_target": (50, 50),  # ±5% tolerance
}
```

### 4.5 Volume Confirmation Requirements

```python
VOLUME_RULES = {
    "stock_breakout":             {"min_pct_above_50day": 40},
    "earnings_gap_up":            {"min_pct_above_avg": 50},
    "declining_volume_on_rally":  {"action": "trigger_position_reduction"},
    "failed_breakout_weak_vol":   {"action": "immediate_exit"},
    "etf_breakout":               {"min_pct_above_30day": 30},
    "etf_price_gain_no_volume":   {"action": "caution_flag"},
}
```

### 4.6 Market Regime Detection

```python
BULL_SIGNALS = {
    "ibd_market_pulse": "Confirmed uptrend",
    "distribution_days": "< 4 on major indices",
    "sector_breadth": "> 70% in uptrends",
    "ibd50_behavior": "Showing volume breakouts",
    "rsi": "Recovery from oversold (<30 to >50)",
    "moving_averages": "Golden crosses (50d > 200d MA)",
}

BEAR_SIGNALS = {
    "ibd_market_pulse": "Under pressure or correction",
    "distribution_days": "> 5 on major indices",
    "sector_breadth": "< 40%",
    "ibd50_behavior": "Breaking below 50-day MA",
    "rsi": "> 70 with negative divergences",
    "moving_averages": "Death crosses (50d < 200d MA)",
}

REGIME_ACTIONS = {
    "bull": {
        "equity_exposure": "90-100%",
        "tier_focus": "Scale into Tier 1",
        "stops": "Standard (22/18/12)",
    },
    "bear": {
        "equity_exposure": "20-40%",
        "tier_focus": "Rotate to Tier 3 defensive",
        "stops": "Tighten T1 to 15-18%",
        "additional": ["Raise cash", "Enforce stops rigorously",
                       "Consider inverse ETFs (max 10%)",
                       "Avoid new purchases until follow-through day"],
    },
}
```

### 4.7 Tier Allocation Targets

```python
TIER_TARGETS = {
    1: {"target": 39, "range": (35, 40)},
    2: {"target": 37, "range": (33, 40)},
    3: {"target": 22, "range": (20, 30)},
    "cash": {"target": 2, "range": (2, 5)},
}
```

### 4.8 IBD Sector Categories (28)

```python
IBD_SECTORS = [
    "AEROSPACE", "AUTO", "BANKS", "BUILDING", "BUSINESS SERVICES",
    "CHEMICALS", "CHIPS", "COMPUTER", "CONSUMER", "DEFENSE",
    "ELECTRONICS", "ENERGY", "FINANCE", "FOOD/BEVERAGE", "HEALTHCARE",
    "INSURANCE", "INTERNET", "LEISURE", "MACHINERY", "MEDIA",
    "MEDICAL", "MINING", "REAL ESTATE", "RETAIL", "SOFTWARE",
    "TELECOM", "TRANSPORTATION", "UTILITIES"
]
```

---

## 5. Risk Check Pipeline

The Risk Officer runs these checks **in order**. Each check produces findings.

```
CHECK 1: Position Sizing Compliance
  For each position:
    Is stock ≤ tier max? (5/4/3%)
    Is ETF ≤ tier max? (8/8/6%)
  VIOLATION = VETO

CHECK 2: Trailing Stop Compliance
  For each position:
    Does stop match tier? (22/18/12%)
    If position gained 10%, is stop tightened?
    If position gained 20%, is stop tightened further?
  VIOLATION = WARNING (VETO if systematic)

CHECK 3: Sector Concentration
  For each sector:
    Total allocation ≤ 40%?
    Total sectors ≥ 8?
  VIOLATION = VETO if >40%, WARNING if ≥35%

CHECK 4: Tier Allocation
  T1 within 35-40%? T2 within 33-40%? T3 within 20-30%?
  Cash 2-5%?
  VIOLATION = WARNING (VETO if >5% off target)

CHECK 5: Max Loss Analysis
  For each position:
    Position size × stop % ≤ max loss limit?
    Stock T1: ≤1.1% of portfolio at risk?
    ETF T1: ≤1.76% of portfolio at risk?
  VIOLATION = VETO

CHECK 6: Correlation Analysis
  Group stocks by shared drivers (not just sector)
  Identify clusters >4 stocks with correlation >0.8
  Calculate effective concentration after correlation
  WARNING if thematic concentration >30%
  VETO if thematic concentration >50%

CHECK 7: Market Regime Assessment
  Count bull signals vs bear signals
  If bear regime and portfolio is >70% equity: WARNING
  If bear regime and T1 stops not tightened: VETO

CHECK 8: Volume Confirmation (Recent Entries)
  For positions entered in last 2 weeks:
    Was volume confirmation present?
    Any failed breakouts with weak volume?
  WARNING if unconfirmed entries exist

CHECK 9: Keep Position Validation
  Are all 14 keeps present?
  Are keeps in correct categories?
  Are keep targets approximately matched?
  WARNING if keep missing or miscategorized

CHECK 10: Stress Test Scenarios
  Run 3 scenarios on the portfolio:
    a) Market crash: -20% broad market
    b) Sector correction: Leading sector -30%
    c) Rate hike: +50bp unexpected
  Calculate estimated drawdown per tier
  Compare to expected drawdown ranges:
    T1: -25% to -35%, T2: -18% to -25%, T3: -12% to -18%
  VETO if estimated drawdown exceeds 1.5× tier target
```

---

## 6. Autonomy & Decision-Making

### Decision 1: Position Sizing Violation
```
NVDA at 5.2% of portfolio in Tier 1 (limit: 5%)

Risk Officer: "VETO — Position sizing violation.

CHECK 1 FAILED: NVDA at 5.2% exceeds T1 stock max of 5%.
Max loss at risk: 5.2% × 22% stop = 1.14% of portfolio (~$17.3K)
This exceeds the 1.1% max loss limit.

REQUIRED FIX: Trim NVDA to ≤5.0% ($75,897 max at current portfolio value).
Redistribute 0.2% ($3,036) to underweight tier or cash."
```

### Decision 2: Hidden Thematic Concentration
```
8 stocks across 5 'different' sectors all depend on AI spending

Risk Officer: "WARNING — Hidden thematic concentration.

CHECK 6 FINDINGS:
NVDA (Chips), AVGO (Chips), PLTR (Software), CLS (Technology),
MSFT (Software), CRM (Software), CRWD (Cybersecurity), APP (Technology)

These span 5 IBD sectors but share ONE driver: AI/cloud spending.
Combined weight: ~32% of portfolio.
Effective correlation: estimated 0.75-0.85

RISK: If AI spending narrative reverses, ~32% of portfolio declines together.
This is within the 30-50% warning zone.

RECOMMENDATION:
  Reduce AI-dependent cluster to ≤25% (trim 2-3 weakest AI positions)
  Add 2 positions with zero AI correlation (e.g., mining, healthcare, utilities)
  This is a WARNING, not a VETO — but close to threshold."
```

### Decision 3: Bear Market Regime + Aggressive Portfolio
```
Market pulse: Under pressure, 6 distribution days, sector breadth 38%

Risk Officer: "VETO — Portfolio misaligned with market regime.

CHECK 7 FAILED: Bear market signals confirmed (3 of 4 bear indicators active).
Current portfolio: 97.2% equity, T1 at 39%.
Framework v4.0 requires: 20-40% equity in bear market, rotate to T3.

REQUIRED FIXES:
1. Raise cash to minimum 60% (from 2.8%)
2. Tighten ALL Tier 1 trailing stops from 22% to 15-18%
3. Exit any stock breaking below 50-day MA
4. No new purchases until follow-through day confirmed
5. Shift allocation toward T3 (defensive) until regime changes

SEVERITY: CRITICAL — framework rules are unambiguous on bear market actions."
```

### Decision 4: Sector Concentration Approaching Limit
```
Technology at 28% (limit: 40%) — within limit but monitor

Risk Officer: "PASS WITH MONITORING.

CHECK 3 RESULT: Technology at 28% — within 40% limit. ✓
BUT: When adding AI-correlated Chips (15%) and Internet (5%),
effective tech-related exposure is ~48%.

RECOMMENDATION:
  Traditional sector analysis: PASS (28% < 40%)
  Thematic analysis: WARNING (48% tech-adjacent)
  Monitor sector carefully. If Technology rises above 35%,
  proactively trim before hitting the 40% hard limit."
```

### Decision 5: Stress Test — Leading Sector Crash
```
Scenario: Semiconductors (leading sector) corrects -30%

Risk Officer: "STRESS TEST RESULTS:

Semiconductor-exposed positions:
  NVDA (T1, $35K): -30% = -$10,500 (stopped out at -22% = -$7,700)
  AVGO (T2, $35K): -30% = -$10,500 (stopped out at -18% = -$6,300)
  SOXX (T1 ETF, $45.7K): -30% = -$13,710 (stopped out at -22% = -$10,054)
  SMH (T1 ETF, $32K): -30% = -$9,600 (stopped out at -22% = -$7,040)
  KLAC (T1, $19.6K): -30% = -$5,880 (stopped out at -22% = -$4,312)
  MU (T1, $25K): -30% = -$7,500 (stopped out at -22% = -$5,500)

WITHOUT stops: Total semiconductor loss = -$57,690 (3.8% of portfolio)
WITH stops: Total semiconductor loss = -$40,906 (2.7% of portfolio)

VERDICT: ACCEPTABLE — stops provide meaningful protection.
Estimated portfolio drawdown: -6% to -8% (including correlated declines).
All within T1 expected range (-25% to -35% for the sector, not full portfolio).

RECOMMENDATION: Stops are essential. Verify all stop orders are active.
Sleep Well Score for T1 semiconductor cluster: 6/10."
```

---

## 7. Output Contract

### RiskAssessment (Top Level)

```python
class RiskAssessment(BaseModel):
    check_results: List[RiskCheck]                   # All 10 checks with results
    portfolio_risk_by_tier: Dict[str, TierRisk]      # "tier_1", "tier_2", "tier_3"
    vetoes: List[Veto]                               # Hard blocks (may be empty)
    warnings: List[RiskWarning]                      # Non-blocking findings
    stress_test_results: StressTestReport
    market_regime: MarketRegimeAssessment
    sleep_well_scores: SleepWellScores
    keep_validation: KeepValidation
    analysis_date: str
    summary: str                                     # 100+ chars

    # VALIDATORS:
    # - All 10 checks must be present
    # - Vetoes have required_fix (50+ chars)
    # - sleep_well_scores has overall and per-tier
    # - summary ≥ 100 chars
```

### RiskCheck (Per Check)

```python
class RiskCheck(BaseModel):
    check_number: int                    # 1-10
    check_name: str                      # "Position Sizing Compliance"
    status: str                          # "PASS" / "WARNING" / "VETO"
    findings: List[str]                  # Specific findings (1+ items)
    positions_affected: List[str]        # Symbols affected (can be empty)
```

### TierRisk (Per Tier)

```python
class TierRisk(BaseModel):
    tier: int                            # 1, 2, or 3
    position_count: int
    total_allocation_pct: float
    within_target_range: bool
    max_position_pct: float              # Largest position as % of portfolio
    max_position_symbol: str
    stop_loss_compliance: bool           # All stops match framework
    estimated_max_drawdown: str          # e.g., "-22% to -28%"
    risk_level: str                      # "LOW" / "MODERATE" / "HIGH" / "CRITICAL"
```

### Veto

```python
class Veto(BaseModel):
    check_number: int                    # Which check triggered this
    check_name: str
    violation: str                       # What rule was broken (30+ chars)
    positions_affected: List[str]        # Symbols involved
    required_fix: str                    # Specific action PM must take (50+ chars)
    risk_if_not_fixed: str              # Consequence of ignoring
```

### RiskWarning

```python
class RiskWarning(BaseModel):
    check_number: int
    check_name: str
    finding: str                         # What was found (30+ chars)
    severity: str                        # "LOW" / "MEDIUM" / "HIGH"
    recommendation: str                  # Suggested action
    positions_affected: List[str]
```

### MarketRegimeAssessment

```python
class MarketRegimeAssessment(BaseModel):
    regime: str                          # "BULL" / "BEAR" / "TRANSITIONAL"
    bull_signals_active: List[str]       # Which bull signals are on
    bear_signals_active: List[str]       # Which bear signals are on
    portfolio_alignment: str             # "ALIGNED" / "MISALIGNED"
    required_actions: List[str]          # What framework demands (may be empty)
```

### StressTestReport

```python
class StressTestReport(BaseModel):
    scenarios: List[StressScenario]      # At least 3 scenarios
    worst_case_tier: str                 # Which tier is most vulnerable
    overall_resilience: str              # "STRONG" / "ADEQUATE" / "WEAK"
    stops_effectiveness: str             # How much stops reduce losses

class StressScenario(BaseModel):
    scenario_name: str                   # "Market crash -20%"
    tier_1_impact: str                   # e.g., "-12% to -15%"
    tier_2_impact: str
    tier_3_impact: str
    portfolio_impact: str                # Overall estimated drawdown
    stopped_out_positions: List[str]     # Which positions would hit stops
    verdict: str                         # "ACCEPTABLE" / "CONCERNING" / "CRITICAL"
```

### SleepWellScores

```python
class SleepWellScores(BaseModel):
    overall: int                         # 1-10
    tier_1: int                          # 1-10
    tier_2: int                          # 1-10
    tier_3: int                          # 1-10
    methodology: str                     # How scores were calculated
```

### KeepValidation

```python
class KeepValidation(BaseModel):
    fundamental_keeps_present: List[str]     # Which of 5 are present
    ibd_keeps_present: List[str]             # Which of 9 are present
    missing_keeps: List[str]                 # Any expected keeps not found
    miscategorized: List[str]                # Any keeps in wrong category
    status: str                              # "COMPLETE" / "INCOMPLETE"
```

---

## 8. Testing Plan

### Level 1: Schema Tests (No LLM — instant)

| Test | What It Validates |
|---|---|
| `test_valid_output_parses` | Known-good output parses |
| `test_all_10_checks_present` | check_results has 10 entries |
| `test_all_3_tiers_assessed` | tier_1, tier_2, tier_3 in portfolio_risk_by_tier |
| `test_sleep_well_scores_range` | All scores 1-10 |
| `test_veto_has_required_fix` | Every veto has required_fix ≥ 50 chars |
| `test_veto_has_check_number` | Every veto references a check |
| `test_warning_has_severity` | Every warning has LOW/MEDIUM/HIGH |
| `test_stress_test_has_3_scenarios` | At least 3 scenarios |
| `test_market_regime_valid` | Only BULL/BEAR/TRANSITIONAL |
| `test_summary_minimum_length` | ≥ 100 chars |
| `test_keep_validation_present` | KeepValidation exists |
| `test_tier_risk_within_valid_range` | allocation_pct reasonable |
| `test_check_status_valid` | Only PASS/WARNING/VETO |

### Level 2: LLM Output Tests

| Test | What It Validates |
|---|---|
| `test_output_is_valid_json` | Parseable JSON |
| `test_output_matches_schema` | Conforms to RiskAssessment |
| `test_all_checks_have_findings` | No empty check results |
| `test_vetoes_reference_real_positions` | Veto symbols exist in PM output |
| `test_stress_scenarios_are_specific` | Reference actual portfolio positions |
| `test_sleep_well_t3_highest` | T3 ≥ T2 ≥ T1 (defensive sleeps best) |
| `test_regime_assessment_has_signals` | At least 2 signals cited |

### Level 3: Behavioral Tests

| Test | What It Validates |
|---|---|
| `test_does_not_build_portfolios` | No portfolio construction language |
| `test_does_not_pick_stocks` | No "add AAPL" recommendations |
| `test_does_not_override_framework` | Respects v4.0 limits exactly |
| `test_vetoes_only_on_hard_violations` | PASS/WARNING/VETO logic correct |
| `test_oversized_position_triggers_veto` | Position >5% stock T1 = VETO |
| `test_sector_over_40_triggers_veto` | Sector >40% = VETO |
| `test_clean_portfolio_no_vetoes` | Known-good portfolio gets no VETO |

### Level 4: Integration (PM → Risk Officer)

| Test | What It Validates |
|---|---|
| `test_risk_accepts_pm_output` | Handoff works |
| `test_check_positions_match_pm` | All reviewed positions exist in PM output |
| `test_tier_allocations_match_pm` | Tier percentages are consistent |
| `test_veto_triggers_pm_rebuild` | Known-bad portfolio → VETO → PM re-runs |
| `test_keep_validation_matches_pm_keeps` | Keep list is consistent |

---

## 9. Integration Points

### Receives From
- **Portfolio Manager:** Complete portfolio with all positions, tiers, stops, keeps
- **Rotation Detector:** `rotation_status` (informs regime assessment)

### Passes To
- **Portfolio Manager (feedback loop):** Vetoes → PM must fix and resubmit
- **Educator:** Risk findings, Sleep Well scores, warnings for explanation

### Feedback Loop
```
PM builds portfolio → Risk Officer runs 10 checks
  If ANY VETO: PM must fix specific issues and resubmit
  If WARNINGS only: Portfolio proceeds with warnings documented
  If ALL PASS: Clean bill of health, proceed to Educator
```

---

## 10. Files to Create

```
1. src/ibd_agents/schemas/risk_output.py                   ← Output contract (FIRST)
2. tests/unit/test_risk_schema.py                           ← Schema tests (13+)
3. src/ibd_agents/tools/risk_analyzer.py                    ← Correlation, stress, concentration
4. src/ibd_agents/agents/risk_officer.py                    ← Agent + Task
5. config/agents.yaml                                       ← Add risk_officer entry
6. config/tasks.yaml                                        ← Add risk task
7. tests/unit/test_risk_agent.py                            ← LLM output tests
8. tests/unit/test_risk_behavior.py                         ← Behavioral tests
9. tests/integration/test_pm_to_risk.py                     ← Handoff test
10. tests/fixtures/portfolio_clean.json                     ← Known-good portfolio (no vetoes)
11. tests/fixtures/portfolio_violations.json                ← Known-bad portfolio (triggers vetoes)
12. golden_datasets/risk_golden.json                        ← Golden baseline
```