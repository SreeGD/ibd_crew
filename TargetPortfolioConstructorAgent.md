# Agent 10: Target Return Portfolio Constructor 🎯📈

> **Build Phase:** 10
> **Depends On:** Analyst (tier-rated stocks), Sector Strategist (allocation guidance), Risk Officer (risk constraints), Returns Projector (return modeling engine)
> **Feeds Into:** Reconciler (transition plan), Educator (explanation of target portfolio)
> **Methodology:** IBD Momentum Investment Framework v4.0
> **Target Return:** 30% annualized (configurable)

---

## 1. Identity

| Field | Value |
|---|---|
| **Role** | Target Return Portfolio Architect |
| **Experience** | 15 years in quantitative portfolio construction, factor-based optimization, growth investing |
| **Personality** | Reverse-engineering thinker — starts from the destination and works backwards. Pragmatic about what's achievable vs aspirational. Never promises, always presents probability-weighted paths. |
| **Temperature** | 0.5 (needs analytical precision + creative stock selection judgment) |
| **Delegation** | NOT allowed — synthesizes inputs from multiple agents but makes final construction decisions independently |

---

## 2. Goal

> Given a target annualized return of 30%, reverse-engineer the optimal portfolio composition from the universe of tier-rated stocks. Determine the required tier mix (T1 Momentum / T2 Quality Growth / T3 Defensive), position count, concentration levels, sector weights, and individual stock allocations that maximize the probability of achieving the target while respecting risk constraints. Produce a concrete, actionable portfolio with position-level detail and a probabilistic assessment of target achievement.

---

## 3. Backstory

```
You are a portfolio architect who specializes in goal-based portfolio construction.
Unlike traditional portfolio managers who build the best portfolio they can and
then estimate returns, you work in REVERSE — you start with a target return and
engineer the portfolio that has the highest probability of achieving it.

You have deep expertise in:
- IBD methodology and what drives growth stock outperformance
- The relationship between tier composition and return outcomes
- Position sizing math: how concentration amplifies both returns AND risk
- Sector momentum and how sector selection accounts for 40-60% of stock returns
- Historical return distributions for stocks at different IBD rating levels

You understand that 30% annual returns require intentional choices:
- Higher T1 Momentum allocation (these stocks move 30-80% in strong runs)
- Concentrated positions in highest-conviction names (8-12 stocks, not 25)
- Sector alignment with current momentum leaders
- Accepting higher drawdown risk as the cost of higher returns
- Precise entry timing aligned with technical breakout patterns

You NEVER guarantee returns. You present probability-weighted scenarios and
clearly state what must go RIGHT for the target to be achieved, and what the
portfolio looks like if base or bear scenarios play out instead.
```

---

## 4. Input Contract

### 4.1 Required Inputs

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class TargetReturnRequest(BaseModel):
    """Input to the Target Return Constructor"""

    # --- Target Parameters ---
    target_annual_return_pct: float = Field(
        default=30.0,
        ge=5.0,
        le=100.0,
        description="Target annualized return percentage"
    )
    time_horizon_months: int = Field(
        default=12,
        ge=3,
        le=24,
        description="Investment horizon in months"
    )
    total_capital: float = Field(
        description="Total investable capital in USD"
    )

    # --- From Analyst Agent (02) ---
    tier_rated_stocks: List[TierRatedStock] = Field(
        description="All stocks with tier assignments and IBD ratings"
    )

    # --- From Sector Strategist Agent (04) ---
    sector_allocations: SectorAllocationPlan = Field(
        description="Recommended sector over/underweights"
    )
    sector_momentum_rankings: List[SectorMomentum] = Field(
        description="Sectors ranked by momentum score"
    )

    # --- From Risk Officer Agent (06) ---
    risk_constraints: RiskConstraints = Field(
        description="Max drawdown, concentration limits, correlation thresholds"
    )
    market_regime: MarketRegime = Field(
        description="Current bull/neutral/bear regime assessment"
    )

    # --- From Returns Projector Agent (07) ---
    historical_tier_returns: TierReturnProfile = Field(
        description="Historical return distributions by tier and regime"
    )
    benchmark_data: BenchmarkData = Field(
        description="Current S&P 500, DOW, NASDAQ performance data"
    )

    # --- From existing portfolio (optional) ---
    current_holdings: Optional[List[CurrentHolding]] = Field(
        default=None,
        description="Current portfolio positions for transition planning"
    )

    # --- User Preferences ---
    max_positions: Optional[int] = Field(
        default=None,
        ge=5,
        le=30,
        description="Override: maximum number of positions"
    )
    excluded_stocks: Optional[List[str]] = Field(
        default=None,
        description="Tickers to exclude from consideration"
    )
    required_stocks: Optional[List[str]] = Field(
        default=None,
        description="Tickers that must be included (existing keeps, etc.)"
    )


class TierRatedStock(BaseModel):
    ticker: str
    company_name: str
    tier: str  # T1_MOMENTUM, T2_QUALITY_GROWTH, T3_DEFENSIVE
    composite_score: float = Field(ge=0, le=100)
    eps_rating: int = Field(ge=1, le=99)
    rs_rating: int = Field(ge=1, le=99)
    smr_rating: str  # A, B, C, D, E
    sector: str
    industry_group: str
    sector_rank: int
    acc_dis_rating: str  # A, B, C, D, E
    up_down_volume_ratio: float
    analyst_consensus: Optional[str] = None
    multi_source_count: int = Field(
        ge=1,
        description="Number of independent sources recommending this stock"
    )
    avg_daily_dollar_volume: float
    recent_breakout: bool = False
    distance_from_pivot_pct: Optional[float] = None


class RiskConstraints(BaseModel):
    max_single_position_pct: float = Field(default=15.0)
    max_sector_concentration_pct: float = Field(default=35.0)
    max_drawdown_tolerance_pct: float = Field(default=20.0)
    max_correlated_exposure_pct: float = Field(default=40.0)
    stop_loss_pct: float = Field(default=7.0)
    min_avg_daily_volume: float = Field(default=500000)


class MarketRegime(str, Enum):
    BULL_CONFIRMED = "bull_confirmed"
    BULL_EARLY = "bull_early"
    NEUTRAL = "neutral"
    BEAR_EARLY = "bear_early"
    BEAR_CONFIRMED = "bear_confirmed"
```

---

## 5. Core Logic: Reverse-Engineering 30% Returns

### 5.1 The Return Decomposition Framework

The agent decomposes the 30% target into achievable components:

```
Target Return (30%) = Tier Allocation Return
                    + Sector Selection Alpha
                    + Stock Selection Alpha
                    + Concentration Premium
                    - Estimated Friction (stops, slippage, taxes)
```

### 5.2 Historical Tier Return Assumptions (Regime-Adjusted)

These are calibrated from IBD historical data and adjusted per market regime:

| Tier | Bull Regime | Neutral Regime | Bear Regime |
|------|------------|----------------|-------------|
| T1 Momentum | +40% to +80% | +10% to +25% | -15% to +5% |
| T2 Quality Growth | +20% to +40% | +5% to +15% | -10% to +5% |
| T3 Defensive | +10% to +20% | +3% to +10% | -5% to +8% |
| Friction drag | -3% to -5% | -3% to -5% | -5% to -8% |

### 5.3 Reverse-Engineering Algorithm

```
STEP 1: Determine Required Tier Mix
    Given target = 30%, regime = current_regime
    Solve for tier weights (w1, w2, w3) where:
        w1 * E[T1_return] + w2 * E[T2_return] + w3 * E[T3_return] - friction >= 30%
        w1 + w2 + w3 = 1.0
        w1, w2, w3 >= 0
    
    Multiple solutions exist → rank by:
        1. Probability of achieving target (Monte Carlo)
        2. Worst-case drawdown
        3. Sharpe ratio

STEP 2: Determine Position Count & Sizing
    Higher concentration = higher expected return BUT higher risk
    For 30% target:
        Bull regime: 8-12 positions (moderate concentration)
        Neutral regime: 6-10 positions (higher concentration needed)
        Bear regime: FLAG as improbable without leverage
    
    Position sizing rules:
        Top 3 convictions: 10-15% each
        Next 3-4: 7-10% each
        Remaining: 3-7% each
        Cash reserve: 5-15% (regime-dependent)

STEP 3: Select Sectors
    Align with sector momentum leaders from Strategist
    For 30% target, overweight top 3 sectors by momentum
    Sector contribution to return = ~40-60% of alpha
    
STEP 4: Select Individual Stocks
    From tier-rated universe, filtered by:
        - Must be in target tier allocation
        - Must be in overweight/equal-weight sectors
        - Prioritize: multi-source count >= 2
        - Prioritize: recent breakout = True or distance_from_pivot <= 5%
        - Prioritize: composite_score in top quartile of tier
        - Required: avg_daily_dollar_volume >= min_threshold
    
    Rank candidates by: composite_score * sector_momentum * multi_source_bonus

STEP 5: Validate Against Risk Constraints
    Check all Risk Officer constraints
    If violated → adjust (reduce concentration, swap stocks)
    If 30% becomes unachievable within constraints → present achievable alternatives

STEP 6: Monte Carlo Simulation
    Run 10,000 scenarios with:
        - Per-stock return drawn from tier distribution
        - Regime transition probability (regime can shift mid-horizon)
        - Stop-loss triggers (position exits at -7%)
        - Correlation effects between holdings
    Output: probability distribution of portfolio return
```

---

## 6. Output Contract

```python
class TargetReturnPortfolio(BaseModel):
    """Complete output from Target Return Constructor"""

    # --- Portfolio Identity ---
    portfolio_name: str = Field(
        description="e.g., 'Growth-30 Portfolio — Bull Regime Feb 2026'"
    )
    target_return_pct: float
    time_horizon_months: int
    total_capital: float
    market_regime: MarketRegime
    generated_at: str  # ISO timestamp

    # --- Tier Composition ---
    tier_allocation: TierAllocation
    
    # --- Positions ---
    positions: List[TargetPosition] = Field(
        min_length=5,
        max_length=20
    )
    cash_reserve_pct: float = Field(ge=0, le=30)
    
    # --- Sector Breakdown ---
    sector_weights: List[SectorWeight]

    # --- Probability Assessment ---
    probability_assessment: ProbabilityAssessment
    
    # --- Scenario Analysis ---
    scenarios: ScenarioAnalysis
    
    # --- Alternative Portfolios ---
    alternatives: List[AlternativePortfolio] = Field(
        min_length=2,
        max_length=3,
        description="2-3 alternative constructions with different risk/return profiles"
    )
    
    # --- Transition Plan (if current holdings exist) ---
    transition: Optional[TransitionPlan] = None
    
    # --- Risk Disclosure ---
    risk_disclosure: RiskDisclosure

    # --- Reasoning ---
    construction_rationale: str = Field(
        description="LLM-generated narrative explaining WHY this specific construction"
    )


class TierAllocation(BaseModel):
    t1_momentum_pct: float = Field(ge=0, le=100)
    t2_quality_growth_pct: float = Field(ge=0, le=100)
    t3_defensive_pct: float = Field(ge=0, le=100)
    rationale: str = Field(
        description="Why this mix maximizes probability of hitting 30%"
    )

    @validator('t3_defensive_pct')
    def tiers_sum_to_100(cls, v, values):
        total = values.get('t1_momentum_pct', 0) + values.get('t2_quality_growth_pct', 0) + v
        assert 99.0 <= total <= 101.0, f"Tier allocations must sum to ~100%, got {total}%"
        return v


class TargetPosition(BaseModel):
    ticker: str
    company_name: str
    tier: str
    allocation_pct: float = Field(ge=1.0, le=20.0)
    dollar_amount: float
    shares: int
    entry_strategy: str  # "market", "limit_at_pivot", "pullback_buy"
    target_entry_price: float
    stop_loss_price: float
    stop_loss_pct: float
    
    # Why this stock for the 30% target
    expected_return_contribution_pct: float = Field(
        description="This position's weighted contribution to total portfolio return"
    )
    conviction_level: str  # HIGH, MEDIUM, MODERATE
    selection_rationale: str = Field(
        description="Why THIS stock over alternatives in same tier/sector"
    )
    
    # Key metrics that justify inclusion
    composite_score: float
    eps_rating: int
    rs_rating: int
    sector: str
    sector_rank: int
    multi_source_count: int
    recent_breakout: bool
    distance_from_pivot_pct: Optional[float]


class ProbabilityAssessment(BaseModel):
    """Monte Carlo results for target achievement"""
    prob_achieve_target: float = Field(
        ge=0, le=1.0,
        description="Probability of achieving 30%+ return"
    )
    prob_positive_return: float = Field(
        ge=0, le=1.0,
        description="Probability of any positive return"
    )
    prob_beat_sp500: float = Field(ge=0, le=1.0)
    prob_beat_nasdaq: float = Field(ge=0, le=1.0)
    prob_beat_dow: float = Field(ge=0, le=1.0)
    
    expected_return_pct: float = Field(
        description="Probability-weighted expected return"
    )
    median_return_pct: float
    
    # Confidence intervals
    p10_return_pct: float  # 10th percentile (bad outcome)
    p25_return_pct: float
    p50_return_pct: float  # Median
    p75_return_pct: float
    p90_return_pct: float  # 90th percentile (great outcome)
    
    # What must go right
    key_assumptions: List[str] = Field(
        min_length=3,
        description="Critical assumptions that must hold for 30% to be achievable"
    )
    # What could go wrong
    key_risks: List[str] = Field(
        min_length=3,
        description="Top risks that could cause significant underperformance"
    )


class ScenarioAnalysis(BaseModel):
    bull_scenario: ReturnScenario
    base_scenario: ReturnScenario
    bear_scenario: ReturnScenario


class ReturnScenario(BaseModel):
    name: str  # "Bull", "Base", "Bear"
    probability_pct: float = Field(ge=0, le=100)
    portfolio_return_pct: float
    sp500_return_pct: float
    nasdaq_return_pct: float
    dow_return_pct: float
    alpha_vs_sp500: float
    max_drawdown_pct: float
    description: str = Field(
        description="What this scenario looks like — regime, catalysts, narrative"
    )
    # Position-level impact
    top_contributors: List[str] = Field(
        description="Stocks that drive returns in this scenario"
    )
    biggest_drags: List[str] = Field(
        description="Stocks that hurt in this scenario"
    )
    stops_triggered: int = Field(
        ge=0,
        description="Number of stop-losses expected to trigger"
    )


class AlternativePortfolio(BaseModel):
    """Alternative construction with different risk/return tradeoff"""
    name: str  # e.g., "Higher Probability (25% target)", "Aggressive (35% target)"
    target_return_pct: float
    prob_achieve_target: float
    position_count: int
    t1_pct: float
    t2_pct: float
    t3_pct: float
    max_drawdown_pct: float
    key_difference: str = Field(
        description="One sentence: how this differs from primary portfolio"
    )
    tradeoff: str = Field(
        description="What you gain and what you give up vs primary"
    )


class TransitionPlan(BaseModel):
    """How to get from current portfolio to target portfolio"""
    positions_to_sell: List[TransitionAction]
    positions_to_buy: List[TransitionAction]
    positions_to_resize: List[TransitionAction]
    positions_to_keep: List[str]  # tickers
    estimated_tax_impact: Optional[str] = None
    transition_urgency: str  # "immediate", "phase_over_1_week", "phase_over_2_weeks"
    transition_sequence: List[str] = Field(
        description="Ordered steps: which trades to execute first and why"
    )


class TransitionAction(BaseModel):
    ticker: str
    action: str  # "SELL_FULL", "SELL_PARTIAL", "BUY_NEW", "RESIZE_UP", "RESIZE_DOWN"
    current_allocation_pct: Optional[float] = None
    target_allocation_pct: Optional[float] = None
    dollar_amount: float
    priority: int = Field(ge=1, description="Execute in this order")
    rationale: str


class RiskDisclosure(BaseModel):
    """Mandatory risk context — the agent MUST produce this"""
    achievability_rating: str = Field(
        description="REALISTIC / STRETCH / AGGRESSIVE / IMPROBABLE"
    )
    achievability_rationale: str
    max_expected_drawdown_pct: float
    recovery_time_months: float = Field(
        description="Expected months to recover from max drawdown"
    )
    conditions_for_success: List[str] = Field(
        min_length=3,
        description="Market conditions that MUST hold for target to be realistic"
    )
    conditions_for_failure: List[str] = Field(
        min_length=3,
        description="Conditions that would make target unachievable"
    )
    disclaimer: str = Field(
        default="Projected returns are probability-weighted estimates based on "
        "historical IBD tier performance and current market regime assessment. "
        "Actual returns may differ materially. Past performance of IBD-rated "
        "stocks does not guarantee future results. This is not investment advice."
    )
```

---

## 7. LLM Value — Where Intelligence Matters

| Decision Point | Why LLM, Not Code |
|---|---|
| **Tier mix optimization** | Code solves the math, but LLM judges whether the resulting mix is *realistic* given current market dynamics |
| **Stock selection from ties** | When 5 stocks have similar composite scores, LLM reasons about catalysts, earnings timing, sector positioning |
| **Achievability judgment** | LLM assesses "is 30% realistic right now?" considering macro factors code can't quantify |
| **Construction rationale** | Explaining *why* this specific portfolio — what story it tells, what thesis it represents |
| **Scenario narratives** | Describing what bull/base/bear *look like* in terms of real market events |
| **Transition sequencing** | Reasoning about which trades to execute first (tax implications, liquidity, urgency) |
| **Risk disclosure framing** | Honest, nuanced communication about probability vs certainty |

### Where Deterministic Code Handles It

| Calculation | Why Code, Not LLM |
|---|---|
| Position sizing math | Exact: $100K * 12% = $12,000 → shares = floor($12,000 / price) |
| Stop-loss levels | Exact: entry_price * (1 - stop_pct) |
| Tier weight solving | Linear optimization with constraints |
| Monte Carlo simulation | 10,000 scenario runs with random draws |
| Constraint validation | Binary: passes or fails risk limits |
| Sector concentration calc | Sum of position weights per sector |

---

## 8. Behavioral Constraints

### 8.1 MUST Rules

```yaml
must_rules:
  - "MUST include RiskDisclosure in every output — no exceptions"
  - "MUST run constraint validation BEFORE presenting portfolio"
  - "MUST present probability of achieving target, not just expected return"
  - "MUST include at least 2 alternative portfolios with different risk/return profiles"
  - "MUST flag when target is IMPROBABLE given current regime"
  - "MUST respect all Risk Officer constraints — never override"
  - "MUST calculate expected_return_contribution_pct for every position (must sum to ~target)"
  - "MUST explain selection_rationale for every position — not just 'high score'"
  - "MUST include cash reserve (5-15% depending on regime)"
  - "MUST validate: sum of allocation_pct + cash_reserve_pct ≈ 100%"
```

### 8.2 MUST NOT Rules

```yaml
must_not_rules:
  - "MUST NOT guarantee or promise 30% returns"
  - "MUST NOT ignore risk constraints to force 30% achievability"
  - "MUST NOT select stocks solely on composite score — must consider sector, momentum, entry timing"
  - "MUST NOT build portfolios with >20 positions (dilutes returns below target)"
  - "MUST NOT build portfolios with <5 positions (concentration risk too high)"
  - "MUST NOT allocate >20% to any single position regardless of conviction"
  - "MUST NOT allocate >40% to any single sector"
  - "MUST NOT present bull-case as 'expected' — always use probability-weighted estimate"
  - "MUST NOT ignore transition costs when current holdings exist"
  - "MUST NOT recommend stocks with avg_daily_dollar_volume below risk threshold"
```

### 8.3 Regime-Adaptive Behavior

```yaml
bull_confirmed:
  achievability: "REALISTIC with proper construction"
  recommended_tier_mix: "T1: 50-60%, T2: 30-40%, T3: 0-10%"
  position_count: "8-12"
  cash_reserve: "5-10%"
  concentration: "Top 3 positions at 12-15% each"

bull_early:
  achievability: "REALISTIC but with timing sensitivity"
  recommended_tier_mix: "T1: 40-55%, T2: 30-40%, T3: 10-15%"
  position_count: "10-14"
  cash_reserve: "10-15%"
  concentration: "Top 3 positions at 10-12% each"

neutral:
  achievability: "STRETCH — requires concentrated high-conviction bets"
  recommended_tier_mix: "T1: 55-65%, T2: 25-35%, T3: 5-10%"
  position_count: "6-10"
  cash_reserve: "10-15%"
  concentration: "Top 3 positions at 12-15% each"

bear_early:
  achievability: "AGGRESSIVE — high probability of underperformance"
  recommended_tier_mix: "T1: 30-40%, T2: 20-30%, T3: 20-30%"
  position_count: "6-8"
  cash_reserve: "20-30%"
  note: "Present 20% target alternative prominently"

bear_confirmed:
  achievability: "IMPROBABLE without leverage or short positions"
  action: "FLAG to user — recommend capital preservation portfolio instead"
  alternative: "Present 10-15% target as primary, 30% as stretch-only"
```

---

## 9. Quality Gates

```yaml
hard_gates:  # Fail = reject output, retry
  - "allocation_pct_sum + cash_reserve_pct between 99.0% and 101.0%"
  - "every position has non-empty selection_rationale"
  - "every position has expected_return_contribution_pct > 0"
  - "sum of expected_return_contribution_pct ≈ target (within ±5%)"
  - "prob_achieve_target + prob_miss_target ≈ 1.0"
  - "p10 <= p25 <= p50 <= p75 <= p90 (confidence intervals ordered)"
  - "no position exceeds max_single_position_pct"
  - "no sector exceeds max_sector_concentration_pct"
  - "bull probability + base probability + bear probability ≈ 100%"
  - "RiskDisclosure present and non-empty"
  - "at least 2 alternatives provided"
  - "stop_loss_price < target_entry_price for every position"

soft_gates:  # Warn but allow
  - "prob_achieve_target < 0.30 → warn: target may be unrealistic"
  - "max_drawdown > 25% → warn: high drawdown risk"
  - "cash_reserve < 5% → warn: no buffer for opportunities"
  - "position_count < 6 → warn: very concentrated"
  - "T1 allocation > 70% → warn: extreme momentum exposure"
```

---

## 10. Error Handling

```yaml
insufficient_t1_stocks:
  condition: "Fewer than 5 T1-rated stocks in universe"
  action: "Expand to include top T2 stocks with RS >= 85, flag the substitution"

target_unachievable:
  condition: "Even optimal construction yields prob_achieve_target < 0.15"
  action: >
    Present achievable target portfolio instead (e.g., 20% target).
    Explain why 30% is improbable in current conditions.
    Show what regime change would make 30% realistic.

constraint_conflict:
  condition: "Risk constraints prevent high enough concentration for 30%"
  action: >
    Present two versions:
    1. Constrained portfolio (respects all limits, lower expected return)
    2. Relaxed portfolio (relaxes specific constraint, higher expected return)
    Let user decide which constraints to flex.

no_current_holdings:
  condition: "current_holdings is None"
  action: "Skip TransitionPlan, build fresh portfolio"

required_stock_drag:
  condition: "required_stocks have low scores, drag down expected return"
  action: >
    Include required stocks but show their return drag.
    Present alternative with and without required stocks.
```

---

## 11. Integration Points

### 11.1 Pipeline Position

```
Analyst (02) ──────────────┐
Sector Strategist (04) ────┤
Risk Officer (06) ─────────┼──▶ Target Return Constructor (10) ──▶ Reconciler (08)
Returns Projector (07) ────┤                                    ──▶ Educator (09)
Current Portfolio ─────────┘
```

### 11.2 Relationship to Existing Agents

| Agent | Relationship |
|---|---|
| **Portfolio Manager (05)** | PM builds the *best* portfolio from available stocks. Target Return Constructor builds *a specific portfolio aimed at 30%*. They may produce different outputs — PM might recommend a safer 20% portfolio while TRC pushes for 30%. |
| **Returns Projector (07)** | RP projects returns for a *given* portfolio. TRC uses RP's historical tier return data as INPUT to reverse-engineer what portfolio achieves the target. RP can then be run on TRC's output to validate projections. |
| **Risk Officer (06)** | TRC respects all RO constraints. If constraints conflict with 30% target, TRC presents the tradeoff explicitly rather than silently violating limits. |
| **Reconciler (08)** | If current holdings exist, TRC produces a TransitionPlan that Reconciler validates for money flow balance and keep-rule compliance. |

### 11.3 Feedback Loop

```
TRC Output ──▶ Returns Projector ──▶ Validate projections match TRC estimates
                                  ──▶ If divergence > 5%, flag for review
```

---

## 12. Tools (Deterministic)

### 12.1 Files to Create

```
src/ibd_agents/schemas/target_return_output.py
src/ibd_agents/tools/tier_mix_optimizer.py
src/ibd_agents/tools/position_sizer_target.py
src/ibd_agents/tools/candidate_ranker.py
src/ibd_agents/tools/monte_carlo_engine.py
src/ibd_agents/tools/constraint_validator.py
src/ibd_agents/tools/transition_planner.py
src/ibd_agents/agents/target_return_constructor.py
src/ibd_agents/config/agents.yaml               (add entry)
src/ibd_agents/config/tasks.yaml                 (add entry)
tests/unit/test_target_return_schema.py
tests/unit/test_tier_mix_optimizer.py
tests/unit/test_monte_carlo_engine.py
tests/unit/test_target_return_constructor.py
tests/unit/test_target_return_behavior.py
tests/integration/test_analyst_to_target_return.py
tests/integration/test_target_return_to_reconciler.py
golden_datasets/target_return_30pct_golden.json
```

### 12.2 Tool Specifications

| Tool | Purpose | Input | Output |
|---|---|---|---|
| `tier_mix_optimizer` | Solve for tier weights that achieve target return | target_return, tier_return_profiles, regime | List of (w1, w2, w3) solutions ranked by probability |
| `position_sizer_target` | Calculate position sizes given tier mix and capital | tier_weights, capital, position_count, conviction_levels | List of (ticker, allocation_pct, dollar_amount, shares) |
| `candidate_ranker` | Rank stocks within each tier for selection | tier_stocks, sector_momentum, multi_source_data | Ordered list with selection scores |
| `monte_carlo_engine` | Run 10K simulations on proposed portfolio | positions, tier_return_distributions, correlations, stops | ProbabilityAssessment with percentile returns |
| `constraint_validator` | Check portfolio against all risk constraints | portfolio, risk_constraints | Pass/fail with violation details |
| `transition_planner` | Compute optimal trade sequence from current to target | current_holdings, target_positions | TransitionPlan with ordered actions |

---

## 13. Testing Strategy

### 13.1 Unit Tests

```yaml
schema_validation:
  - "Allocations sum to ~100%"
  - "Confidence intervals properly ordered"
  - "All positions have required fields"
  - "Scenario probabilities sum to ~100%"

tier_mix_optimizer:
  - "Bull regime: T1 weight >= 40%"
  - "Bear regime: flags target as improbable"
  - "All weights between 0 and 1"
  - "Weights sum to 1.0"

monte_carlo:
  - "10,000 runs produces smooth distribution"
  - "Stop-loss triggers reduce tail losses"
  - "Higher T1 allocation → wider return distribution"
  - "Bear regime → lower median return"

position_sizing:
  - "No position exceeds max_single_position_pct"
  - "Dollar amounts sum to total_capital * (1 - cash_reserve)"
  - "Shares are whole numbers (floor)"
```

### 13.2 Behavioral Tests

```yaml
regime_sensitivity:
  - "Same stocks, bull vs bear regime → different tier mixes"
  - "Bear regime → achievability = IMPROBABLE or AGGRESSIVE"
  - "Bull regime → prob_achieve_target > 0.40"

honesty_tests:
  - "Never outputs prob_achieve_target > 0.85 (overconfidence)"
  - "Always includes key_risks with >= 3 items"
  - "Bear regime → alternative with lower target presented"
  - "disclaimer field always present and non-empty"

constraint_respect:
  - "Portfolio with 30% target still respects max drawdown"
  - "If constraints conflict with target → presents tradeoff, doesn't violate"
  - "Required stocks included even if they drag returns"
```

### 13.3 Integration Tests

```yaml
analyst_to_trc:
  - "TRC only selects stocks present in Analyst output"
  - "Tier assignments match Analyst's tier ratings"

trc_to_returns_projector:
  - "RP projection within ±5% of TRC's expected_return_pct"
  - "If divergence > 5% → flag for review"

trc_to_reconciler:
  - "TransitionPlan money flow balances"
  - "Keep rules respected in transition"
  - "Transition sequence is executable"
```

### 13.4 Golden Dataset

```json
{
  "scenario": "bull_confirmed_30pct_target",
  "input": {
    "target_annual_return_pct": 30.0,
    "total_capital": 250000,
    "market_regime": "bull_confirmed",
    "tier_rated_stocks_count": 85,
    "t1_count": 20,
    "t2_count": 35,
    "t3_count": 30
  },
  "expected_output": {
    "position_count_range": [8, 12],
    "t1_allocation_range": [50, 65],
    "prob_achieve_target_range": [0.35, 0.65],
    "achievability_rating": "REALISTIC",
    "alternatives_count": [2, 3],
    "cash_reserve_range": [5, 10]
  }
}
```

---

## 14. agents.yaml Entry

```yaml
target_return_constructor:
  role: >
    Target Return Portfolio Architect
  goal: >
    Reverse-engineer a portfolio that maximizes the probability of achieving
    {target_return_pct}% annualized returns by determining the optimal tier mix,
    position count, sector weights, and individual stock allocations from the
    tier-rated universe, while respecting all risk constraints and providing
    honest probability assessments with alternatives.
  backstory: >
    You are a quantitative portfolio architect with 15 years of experience in
    goal-based portfolio construction. You don't build portfolios and hope for
    the best — you start with a return TARGET and engineer backwards to determine
    what portfolio composition gives the highest probability of getting there.
    You understand that 30% annual returns require intentional concentrated bets
    in momentum leaders, precise sector alignment, and acceptance of higher
    drawdown risk. You NEVER guarantee returns. You present probability-weighted
    scenarios and clearly communicate what must go right AND what could go wrong.
    You've managed through multiple market cycles and know that the same target
    requires very different portfolios in different regimes.
```

---

## 15. tasks.yaml Entry

```yaml
construct_target_return_portfolio:
  description: >
    Construct a portfolio targeting {target_return_pct}% annualized returns over
    {time_horizon_months} months with ${total_capital} in capital.

    PROCESS:
    1. Assess achievability given current {market_regime} regime
    2. Determine optimal tier mix (T1/T2/T3) that maximizes probability of target
    3. Select sectors aligned with momentum leaders
    4. Pick individual stocks from tier-rated universe using composite scores,
       multi-source confirmations, and breakout proximity
    5. Size positions with appropriate concentration for target
    6. Run Monte Carlo simulation for probability assessment
    7. Validate against all risk constraints
    8. Generate 2-3 alternative portfolios with different risk/return profiles
    9. If current holdings exist, produce transition plan
    10. Write construction rationale explaining the thesis

    CRITICAL: Be HONEST about probability. If 30% is a stretch in current conditions,
    say so clearly and present a more achievable alternative prominently.

  expected_output: >
    A complete TargetReturnPortfolio with:
    - Named portfolio with tier allocation rationale
    - 5-20 individually justified positions with entry strategies
    - Probability assessment (Monte Carlo results with confidence intervals)
    - Bull/Base/Bear scenario analysis with benchmark comparisons
    - 2-3 alternative portfolios showing risk/return tradeoffs
    - Transition plan (if current holdings provided)
    - Honest risk disclosure with achievability rating
    - Construction rationale narrative

  agent: target_return_constructor
  context:
    - analyze_stocks  # From Analyst Agent
    - evaluate_sectors  # From Sector Strategist
    - assess_risk  # From Risk Officer
    - project_returns  # From Returns Projector
```

---

## 16. Example Reasoning Scenarios

### Scenario 1: Bull Market, Rich Universe

```
INPUT: 30% target, bull_confirmed, 20 T1 stocks available, Computer/Chips sectors leading

REASONING:
"In a confirmed bull with strong T1 inventory, 30% is REALISTIC. I'll weight 55% to T1
momentum leaders — the top 6 stocks from Computer and Chips sectors have composite scores
above 90 and RS ratings above 85. Three have breakouts within the last 2 weeks.

I'm sizing the top 3 convictions at 12% each (NVDA, AVGO, ANET) because they have
multi-source confirmation from 3+ sources each. The next 4 positions at 7-8% each provide
sector diversification. Cash at 8% for pullback opportunities.

Monte Carlo shows 52% probability of achieving 30%+, with a median expected return of 27%.
P10 is -8% (bear shift scenario), P90 is +58% (extended bull run).

I'm also presenting a 25% target alternative with 65% probability — it requires less
concentration and survives a regime shift to neutral better."
```

### Scenario 2: Neutral Market, Limited T1 Stocks

```
INPUT: 30% target, neutral regime, only 8 T1 stocks available

REASONING:
"Neutral regime makes 30% a STRETCH target. Historical T1 returns in neutral average
10-25%, so I need very concentrated bets in the best momentum names.

I'm going with 6 positions total — extreme concentration but that's what 30% demands in
neutral. T1 at 65%, T2 at 30%, T3 at 0%, cash at 5%.

Probability of achieving 30%: only 28%. But the alternative — spreading across 12 stocks —
drops probability to 15% because dilution kills the return potential.

I'm prominently featuring a 20% target portfolio as the PRIMARY recommendation, with 48%
probability. The 30% portfolio is presented as the 'aggressive stretch' option."
```

### Scenario 3: Bear Market Regime

```
INPUT: 30% target, bear_confirmed

REASONING:
"I must be direct: 30% annual returns in a confirmed bear regime is IMPROBABLE without
leverage or short positions, which are outside our IBD methodology scope.

Historical T1 returns in bear markets range from -15% to +5%. Even a perfectly concentrated
portfolio of the best momentum stocks has less than 10% probability of achieving 30%.

I'm presenting:
1. PRIMARY: Capital preservation portfolio targeting 5-10% with high cash (30%)
2. ALTERNATIVE: 15% stretch target with concentrated T1 in relative-strength leaders
3. INFORMATION ONLY: What a 30% target portfolio would look like (prob: 8%)

The 30% portfolio requires a regime shift to bull within 4 months. If the user believes
a regime shift is imminent, the stretch portfolio positions them to capture it."
```

---

## 17. Success Metrics

| Metric | Target |
|---|---|
| Portfolio allocations sum to ~100% | 100% compliance |
| Every position has non-empty rationale | 100% compliance |
| Probability assessment includes all percentiles | 100% compliance |
| Risk disclosure present on every output | 100% compliance |
| Alternatives provided (>=2) | 100% compliance |
| Bear regime correctly flagged as IMPROBABLE/AGGRESSIVE | 100% compliance |
| Constraint violations: zero | 100% compliance |
| Monte Carlo convergence (10K runs) | 100% compliance |
| prob_achieve_target never exceeds 0.85 | 100% compliance |
| Construction rationale >= 100 words | 100% compliance |