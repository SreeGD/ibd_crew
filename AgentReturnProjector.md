# Agent 07: Returns Projector 📈

> **Build Phase:** 7
> **Depends On:** Risk Officer (approved portfolio), Rotation Detector (regime)
> **Feeds Into:** Reconciler, Educator
> **Target:** Week 8
> **Methodology:** IBD Momentum Investment Framework v4.0

---

## 1. Identity

| Field | Value |
|---|---|
| **Role** | Quantitative Returns Strategist |
| **Experience** | 12 years in portfolio analytics, factor-based return modeling |
| **Personality** | Data-driven, probabilistic thinker, always presents ranges not points |
| **Temperature** | 0.4 (analytical, but needs judgment for scenario weighting) |
| **Delegation** | NOT allowed |

---

## 2. Goal

> Project expected portfolio returns across bull/base/bear scenarios based on the tier composition (T1 Momentum / T2 Quality Growth / T3 Defensive), individual position characteristics (IBD ratings, sector, validation scores), and market regime. Benchmark against S&P 500, DOW, and NASDAQ to quantify expected alpha. Always present ranges with probabilities — never single-point predictions.

---

## 3. ✅ DOES / ❌ DOES NOT

### ✅ DOES
- Project 3-month, 6-month, and 12-month return ranges for the portfolio
- Model 3 scenarios: Bull, Base, Bear with probability weights
- Calculate expected returns per tier (T1/T2/T3) based on historical IBD momentum performance
- Benchmark against S&P 500 (SPY), DOW (DIA), and NASDAQ-100 (QQQ)
- Calculate projected alpha (portfolio return − benchmark return) per scenario
- Model tier contribution to total return (how much each tier adds/drags)
- Factor in market regime from Rotation Detector (bull amplifies T1, bear protects via T3)
- Account for trailing stop impact on downside (stops cap max loss per position)
- Calculate portfolio Sharpe ratio estimates per scenario
- Show risk-adjusted return comparison vs benchmarks
- Compute max drawdown projections per scenario
- Present confidence intervals (25th/50th/75th/90th percentile)
- Factor in sector momentum (leading sectors boost returns, lagging drag)

### ❌ DOES NOT
- Make buy/sell recommendations (only projects what the current portfolio could return)
- Modify portfolio composition or suggest changes
- Set stop losses or position sizes
- Guarantee or promise any specific returns
- Provide tax advice or account-specific projections
- Use real-time market data (uses historical performance characteristics)

---

## 4. Framework v4.0 Rules (Hard-Coded)

### 4.1 Historical Tier Performance Baselines

Based on IBD methodology backtesting — annualized returns for stocks meeting tier criteria:

```python
TIER_HISTORICAL_RETURNS = {
    # Annualized return assumptions (based on IBD universe characteristics)
    1: {  # Momentum: Comp≥95, RS≥90, EPS≥80
        "bull":  {"mean": 35.0, "std": 18.0},  # High RS stocks in bull markets
        "base":  {"mean": 18.0, "std": 15.0},
        "bear":  {"mean": -8.0, "std": 22.0},  # High-beta, falls harder
    },
    2: {  # Quality Growth: Comp≥85, RS≥80, EPS≥75
        "bull":  {"mean": 25.0, "std": 14.0},
        "base":  {"mean": 14.0, "std": 12.0},
        "bear":  {"mean": -5.0, "std": 16.0},
    },
    3: {  # Defensive: Comp≥80, RS≥75, EPS≥70
        "bull":  {"mean": 15.0, "std": 10.0},
        "base":  {"mean": 10.0, "std": 8.0},
        "bear":  {"mean": 2.0, "std": 12.0},   # Defensive holds up
    },
}
```

### 4.2 Benchmark Historical Returns

```python
BENCHMARK_RETURNS = {
    "SPY": {  # S&P 500
        "bull":  {"mean": 22.0, "std": 12.0},
        "base":  {"mean": 10.5, "std": 14.0},
        "bear":  {"mean": -15.0, "std": 18.0},
    },
    "DIA": {  # Dow Jones
        "bull":  {"mean": 18.0, "std": 11.0},
        "base":  {"mean": 9.0, "std": 13.0},
        "bear":  {"mean": -12.0, "std": 16.0},
    },
    "QQQ": {  # NASDAQ-100
        "bull":  {"mean": 30.0, "std": 16.0},
        "base":  {"mean": 14.0, "std": 18.0},
        "bear":  {"mean": -22.0, "std": 24.0},
    },
}
```

### 4.3 Scenario Probability Weights

```python
# Default weights — adjusted by Rotation Detector regime
SCENARIO_WEIGHTS = {
    "bull_regime":    {"bull": 0.55, "base": 0.35, "bear": 0.10},
    "neutral_regime": {"bull": 0.25, "base": 0.50, "bear": 0.25},
    "bear_regime":    {"bull": 0.10, "base": 0.35, "bear": 0.55},
}
```

### 4.4 Trailing Stop Downside Cap

```python
# From Framework v4.0 — stops limit max loss per position
STOP_LOSS_BY_TIER = {1: 0.22, 2: 0.18, 3: 0.12}

# Max portfolio loss with stops in place:
# T1 (39%) × 22% + T2 (37%) × 18% + T3 (22%) × 12% + Cash (2%) × 0%
# = 8.58% + 6.66% + 2.64% = 17.88% max portfolio drawdown (if ALL stops hit)
MAX_PORTFOLIO_LOSS_WITH_STOPS = 17.88  # %
```

### 4.5 Sector Momentum Multipliers

```python
# Leading sectors get return boost, lagging sectors get drag
SECTOR_MOMENTUM_MULTIPLIER = {
    "leading":   1.15,   # +15% return boost
    "improving": 1.05,   # +5% boost
    "lagging":   0.95,   # -5% drag
    "declining": 0.85,   # -15% drag
}
```

### 4.6 Time Horizon Scaling

```python
# Scale annualized returns to different periods
TIME_HORIZONS = {
    "3_month":  {"fraction": 0.25, "volatility_scale": 0.50},  # √(0.25)
    "6_month":  {"fraction": 0.50, "volatility_scale": 0.71},  # √(0.50)
    "12_month": {"fraction": 1.00, "volatility_scale": 1.00},  # √(1.00)
}
```

---

## 5. Tools

| Tool | Purpose | Input | Output |
|---|---|---|---|
| `tier_return_calculator` | Compute weighted return per tier | Tier allocations + scenario | Return % per tier |
| `benchmark_fetcher` | Get benchmark return assumptions | Benchmark symbol + scenario | Return stats |
| `scenario_weighter` | Apply regime-adjusted scenario weights | Regime + raw scenarios | Weighted expected return |
| `alpha_calculator` | Portfolio return − benchmark return | Portfolio + benchmark returns | Alpha per benchmark |
| `drawdown_estimator` | Project max drawdown with stops | Portfolio + stops + scenario | Max drawdown % |
| `confidence_interval_builder` | Build percentile ranges | Mean, std, horizon | P25/P50/P75/P90 |

---

## 6. Autonomy & Decision-Making

### Decision 1: Regime-Adjusted Scenario Weights
```
Rotation Detector says: "Bull regime — strong rotation into momentum sectors"
Agent: "Shifting scenario weights to bull_regime: 55% bull / 35% base / 10% bear.
This amplifies T1 Momentum return expectations and reduces bear probability.
Expected portfolio return rises from 14.2% → 19.8% (12-month)."
```

### Decision 2: Sector Momentum Impact
```
Portfolio is 35% CHIPS + MINING (both 'leading' sectors).
Agent: "Applying 1.15x multiplier to 35% of T1/T2 positions in leading sectors.
This adds ~1.8% to base-case portfolio return projection.
But concentration in two sectors increases volatility estimate by 2pp."
```

### Decision 3: Stop-Loss Downside Protection
```
Agent: "With all tier stops in place (22/18/12%), max portfolio drawdown
is capped at 17.88% even if every stop triggers simultaneously.
Compared to SPY bear case of -15% without stops and potential -30% tail,
this portfolio has a defined worst-case. However, stop-outs lock in losses
and miss recovery — factoring 40% whipsaw probability in volatile markets."
```

### Decision 4: Alpha Decomposition
```
12-month base case:
  Portfolio: +14.7%  |  SPY: +10.5%  |  Alpha: +4.2%
  
  Alpha sources:
    T1 Momentum stocks outperform SPY:    +2.8%
    Sector momentum tilt (leading):       +1.1%
    Multi-source validated picks:         +0.8%
    T3 defensive drag vs SPY:            -0.5%
    
Agent: "Most alpha comes from T1 momentum concentration. This is
intentional per IBD methodology but means alpha disappears in bear markets
where T1 stocks underperform. The T3 defensive tier partially hedges this."
```

### Decision 5: Comparing Tier Mixes
```
Current mix: T1=39% / T2=37% / T3=22% / Cash=2%
Agent models 3 alternatives:
  Aggressive:    T1=50% / T2=35% / T3=13% / Cash=2%  → +21.5% bull / -9.8% bear
  Balanced:      T1=39% / T2=37% / T3=22% / Cash=2%  → +18.4% bull / -6.2% bear
  Conservative:  T1=25% / T2=35% / T3=35% / Cash=5%  → +14.1% bull / -2.8% bear

"Current balanced mix delivers +18.4% in bull case with -6.2% bear downside.
Aggressive gains +3.1% in bull but costs -3.6% more in bear.
Risk/reward ratio favors the balanced mix for the current neutral-to-bull regime."
```

---

## 7. Output Contract

### ReturnsProjectionOutput

```python
class ReturnsProjectionOutput(BaseModel):
    portfolio_value: float                        # Total portfolio $ value
    analysis_date: str                            # YYYY-MM-DD
    market_regime: str                            # "bull" / "neutral" / "bear"
    regime_source: str                            # "Rotation Detector Agent 03"
    
    # Tier composition
    tier_allocation: TierAllocation               # Actual T1/T2/T3/Cash %
    
    # Scenario projections
    scenarios: List[ScenarioProjection]           # Bull, Base, Bear (3)
    expected_return: ExpectedReturn               # Probability-weighted blend
    
    # Benchmark comparison
    benchmark_comparisons: List[BenchmarkComparison]  # SPY, DIA, QQQ (3)
    
    # Alpha analysis
    alpha_analysis: AlphaAnalysis
    
    # Risk metrics
    risk_metrics: RiskMetrics
    
    # Alternative tier mixes
    tier_mix_comparison: List[TierMixScenario]    # 3 alternatives
    
    # Confidence intervals
    confidence_intervals: ConfidenceIntervals
    
    # Summary
    summary: str                                  # 100+ chars
    caveats: List[str]                           # Required disclaimers
    
    # VALIDATORS:
    # - Exactly 3 scenarios (bull/base/bear)
    # - Scenario weights sum to 1.0
    # - Exactly 3 benchmarks (SPY/DIA/QQQ)
    # - tier_allocation sums to 100% (±0.5% rounding)
    # - caveats non-empty (must include disclaimers)
    # - All return values are percentages (not decimals)
```

### TierAllocation

```python
class TierAllocation(BaseModel):
    tier_1_pct: float           # 35-40% typically
    tier_2_pct: float           # 33-40% typically
    tier_3_pct: float           # 20-30% typically
    cash_pct: float             # 2-5%
    tier_1_value: float         # $ amount
    tier_2_value: float
    tier_3_value: float
    cash_value: float
    
    # Validators: all pcts sum to ~100%
```

### ScenarioProjection

```python
class ScenarioProjection(BaseModel):
    scenario: Literal["bull", "base", "bear"]
    probability: float                 # 0.0-1.0 (regime-adjusted weight)
    
    # Per-tier returns
    tier_1_return_pct: float           # Projected return for T1 slice
    tier_2_return_pct: float
    tier_3_return_pct: float
    
    # Blended portfolio return
    portfolio_return_3m: float         # 3-month projected %
    portfolio_return_6m: float         # 6-month projected %
    portfolio_return_12m: float        # 12-month projected %
    
    # Dollar impact
    portfolio_gain_12m: float          # $ gain/loss at 12-month return
    ending_value_12m: float            # Starting + gain
    
    # Tier contribution to return
    tier_1_contribution: float         # T1 weight × T1 return
    tier_2_contribution: float
    tier_3_contribution: float
    
    reasoning: str                     # 30+ chars explaining scenario
```

### BenchmarkComparison

```python
class BenchmarkComparison(BaseModel):
    symbol: Literal["SPY", "DIA", "QQQ"]
    name: str                           # "S&P 500", "Dow Jones", "NASDAQ-100"
    
    # Benchmark projected returns
    benchmark_return_3m: float
    benchmark_return_6m: float
    benchmark_return_12m: float
    
    # Alpha (portfolio − benchmark) per scenario
    alpha_bull_12m: float
    alpha_base_12m: float
    alpha_bear_12m: float
    
    # Expected alpha (probability-weighted)
    expected_alpha_12m: float
    
    # Outperformance probability
    outperform_probability: float       # 0.0-1.0, likelihood of beating benchmark
```

### AlphaAnalysis

```python
class AlphaAnalysis(BaseModel):
    primary_alpha_sources: List[AlphaSource]      # What generates alpha
    primary_alpha_drags: List[AlphaSource]         # What costs alpha
    net_expected_alpha_vs_spy: float               # % vs S&P 500
    net_expected_alpha_vs_qqq: float               # % vs NASDAQ
    net_expected_alpha_vs_dia: float               # % vs DOW
    alpha_persistence_note: str                    # When alpha works/fails
    
class AlphaSource(BaseModel):
    source: str                                    # "T1 Momentum outperformance"
    contribution_pct: float                        # +2.8% or -0.5%
    confidence: Literal["high", "medium", "low"]
    reasoning: str                                 # 20+ chars
```

### RiskMetrics

```python
class RiskMetrics(BaseModel):
    # Drawdown
    max_drawdown_with_stops: float                 # % (capped by trailing stops)
    max_drawdown_without_stops: float              # % (if no stops in place)
    max_drawdown_dollar: float                     # $ max loss with stops
    
    # Volatility
    projected_annual_volatility: float             # Portfolio std dev %
    spy_annual_volatility: float                   # SPY comparison std dev %
    
    # Sharpe ratio
    portfolio_sharpe_bull: float
    portfolio_sharpe_base: float
    portfolio_sharpe_bear: float
    spy_sharpe_base: float
    
    # Risk-adjusted
    return_per_unit_risk: float                    # Expected return / volatility
    
    # Stop analysis
    probability_any_stop_triggers_12m: float       # 0.0-1.0
    expected_stops_triggered_12m: int              # Count of positions likely stopped
    whipsaw_risk_note: str                         # Warning about false triggers
```

### TierMixScenario

```python
class TierMixScenario(BaseModel):
    label: str                          # "Aggressive", "Balanced", "Conservative"
    tier_1_pct: float
    tier_2_pct: float
    tier_3_pct: float
    cash_pct: float
    
    expected_return_12m: float          # Probability-weighted
    bull_return_12m: float
    bear_return_12m: float
    max_drawdown_with_stops: float
    sharpe_ratio_base: float
    
    comparison_note: str                # "vs current mix: +2.1% bull / -3.2% bear"
```

### ConfidenceIntervals

```python
class ConfidenceIntervals(BaseModel):
    horizon_3m: PercentileRange
    horizon_6m: PercentileRange
    horizon_12m: PercentileRange
    
class PercentileRange(BaseModel):
    p10: float                 # 10th percentile (bad case)
    p25: float                 # 25th percentile
    p50: float                 # Median
    p75: float                 # 75th percentile
    p90: float                 # 90th percentile (great case)
    p10_dollar: float          # $ value at p10
    p50_dollar: float
    p90_dollar: float
```

---

## 8. Example Output Summary

```
══════════════════════════════════════════════════════════
PORTFOLIO RETURNS PROJECTION — February 7, 2026
Portfolio Value: $1,500,000  |  Regime: Neutral-to-Bull
══════════════════════════════════════════════════════════

TIER COMPOSITION
  T1 Momentum:      39% ($585,000)  — 12 stocks, 4 ETFs
  T2 Quality:       37% ($555,000)  — 14 stocks, 5 ETFs
  T3 Defensive:     22% ($330,000)  — 8 stocks, 3 ETFs
  Cash:              2% ($30,000)

SCENARIO PROJECTIONS (12-Month)
  ┌─────────┬─────────┬────────────┬──────────────┐
  │ Scenario│  Prob   │ Return %   │ Ending Value │
  ├─────────┼─────────┼────────────┼──────────────┤
  │ Bull    │  40%    │  +21.6%    │ $1,824,000   │
  │ Base    │  40%    │  +14.7%    │ $1,720,500   │
  │ Bear    │  20%    │   -4.2%    │ $1,437,000   │
  ├─────────┼─────────┼────────────┼──────────────┤
  │ EXPECTED│ 100%    │  +14.0%    │ $1,710,000   │
  └─────────┴─────────┴────────────┴──────────────┘

vs BENCHMARKS (12-Month Expected)
  ┌──────────┬──────────┬───────────┬──────────────┐
  │ Benchmark│ Return % │ Alpha %   │ Beat Prob    │
  ├──────────┼──────────┼───────────┼──────────────┤
  │ S&P 500  │  +10.5%  │  +3.5%    │    68%       │
  │ DOW      │   +9.0%  │  +5.0%    │    72%       │
  │ NASDAQ   │  +14.0%  │  +0.0%    │    51%       │
  └──────────┴──────────┴───────────┴──────────────┘

TIER CONTRIBUTION TO RETURN (Base Case)
  T1 Momentum:    39% × 18.0% = +7.0%  ██████████████▌
  T2 Quality:     37% × 14.0% = +5.2%  ██████████▌
  T3 Defensive:   22% × 10.0% = +2.2%  ████▌
  Cash:            2% ×  4.5% = +0.1%  ▌
  PORTFOLIO TOTAL:             = +14.5%

RISK METRICS
  Max drawdown (with stops):     -17.9%  ($268,200)
  Max drawdown (without stops):  -28.4%  ($426,000)
  Stop protection saves:          $157,800
  Projected volatility:          14.2% (vs SPY 14.0%)
  Sharpe ratio (base):           0.73  (vs SPY 0.54)

CONFIDENCE INTERVALS (12-Month Return %)
  10th percentile:   -8.2%  ($1,377,000)  ← worst realistic
  25th percentile:   +4.1%  ($1,561,500)
  50th percentile:  +14.0%  ($1,710,000)  ← median
  75th percentile:  +23.8%  ($1,857,000)
  90th percentile:  +32.5%  ($1,987,500)  ← best realistic

ALTERNATIVE TIER MIXES
  Aggressive  (50/35/13/2):  +17.2% expected | -9.8% bear | Sharpe 0.68
  Balanced    (39/37/22/2):  +14.0% expected | -4.2% bear | Sharpe 0.73  ← CURRENT
  Conservative(25/35/35/5):  +10.8% expected | -1.1% bear | Sharpe 0.71

CAVEATS:
• Projections based on historical IBD tier characteristics, not guarantees
• Past performance does not predict future results
• Trailing stops may trigger during volatile periods causing realized losses
• Actual returns depend on entry timing, market conditions, and execution
• These projections assume no portfolio changes during the holding period
```

---

## 9. Testing Plan

### Level 1: Schema Tests (No LLM — instant)

| Test | What It Validates |
|---|---|
| `test_valid_output_parses` | Known-good output parses |
| `test_exactly_3_scenarios` | Bull, Base, Bear — no more, no less |
| `test_scenario_weights_sum_to_1` | Probabilities add to 1.0 |
| `test_exactly_3_benchmarks` | SPY, DIA, QQQ |
| `test_tier_allocation_sums_to_100` | T1+T2+T3+Cash ≈ 100% |
| `test_caveats_non_empty` | Must include disclaimers |
| `test_returns_are_percentages` | Not decimals (14.0 not 0.14) |
| `test_ending_value_math` | start × (1 + return/100) = ending |
| `test_alpha_equals_portfolio_minus_benchmark` | Alpha arithmetic |
| `test_tier_contribution_sums_to_total` | T1+T2+T3+cash ≈ portfolio return |
| `test_drawdown_with_stops_less_than_without` | Stops always reduce drawdown |
| `test_confidence_intervals_ordered` | p10 < p25 < p50 < p75 < p90 |
| `test_max_drawdown_within_stop_cap` | ≤ 17.88% with stops |
| `test_outperform_probability_range` | 0.0-1.0 |
| `test_sharpe_ratio_math` | (return − risk_free) / volatility |

### Level 2: LLM Output Tests
- `test_output_valid_json`, `test_matches_schema`
- `test_bull_returns_highest`, `test_bear_returns_lowest`
- `test_t1_returns_highest_in_bull`, `test_t3_returns_highest_in_bear`
- `test_alpha_positive_base_case_vs_spy`
- `test_regime_affects_weights`, `test_sector_momentum_applied`

### Level 3: Behavioral Tests
- `test_no_buy_sell_language` — projections only, no recommendations
- `test_no_guarantees` — never says "will return" or "guaranteed"
- `test_caveats_present` — always includes disclaimers
- `test_no_portfolio_modifications` — doesn't suggest changes
- `test_no_tax_advice` — no account-specific tax projections
- `test_ranges_not_points` — always presents ranges, never single numbers

---

## 10. Integration Points

### Receives From Risk Officer (Agent 06)
- Approved portfolio: positions, tiers, trailing stops, tier allocations
- Sleep Well score (context for risk narrative)

### Receives From Rotation Detector (Agent 03)
- Market regime: bull/neutral/bear
- Sector rotation signals (leading/lagging sectors)

### Passes To Reconciler (Agent 08)
- Expected return context for implementation priority
- Alpha analysis to justify transition urgency

### Passes To Educator (Agent 09)
- All projections for plain-English explanation
- Benchmark comparisons for "here's how you compare"
- Tier mix scenarios for "here are your options"
- Confidence intervals for "here's the realistic range"

---

## 11. Files to Create (In Order)

```
1. src/ibd_agents/schemas/returns_projection_output.py
2. tests/unit/test_returns_projection_schema.py
3. src/ibd_agents/tools/tier_return_calculator.py
4. src/ibd_agents/tools/benchmark_fetcher.py
5. src/ibd_agents/tools/scenario_weighter.py
6. src/ibd_agents/tools/alpha_calculator.py
7. src/ibd_agents/tools/drawdown_estimator.py
8. src/ibd_agents/tools/confidence_builder.py
9. src/ibd_agents/agents/returns_projector.py
10. src/ibd_agents/config/agents.yaml        (add entry)
11. src/ibd_agents/config/tasks.yaml         (add entry)
12. tests/unit/test_returns_projector.py
13. tests/unit/test_returns_projector_behavior.py
14. tests/integration/test_risk_to_returns.py
15. golden_datasets/returns_projection_golden.json
```