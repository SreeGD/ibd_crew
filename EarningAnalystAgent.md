# Earnings Risk Analyst Agent — Specification

**Version:** 1.0
**Author:** Sree
**Project:** IBD_Crew Multi-Agent Investment System
**Framework:** CrewAI
**Date:** February 2026
**Status:** Draft

---

## 1. Problem Statement

Growth stocks live and die by earnings. A stock with a perfect IBD setup — RS Rating 95, Composite 98, sector rank #2 — can gap down 25% overnight on an earnings miss. Conversely, a strong beat can launch a stock 15% higher before the market opens. This binary event risk is **the single largest unhedged exposure in any momentum portfolio.**

The IBD_Crew system currently handles earnings reactively. The Exit Strategist checks if earnings are within 3 weeks and flags insufficient cushion. The Portfolio Manager gets a generic warning. But nobody is doing the deep analysis required to make a **specific, position-level pre-earnings decision.**

The problems with the current approach:

1. **Binary thinking** — "hold" or "sell before earnings" ignores the full spectrum of strategies: trim to half, add a protective put, sell a covered call to offset risk, or even add to a winner if the setup is exceptional.

2. **No historical context** — a stock that has beaten estimates 8 consecutive quarters and gapped up every time is a very different risk than a stock with erratic post-earnings moves. The current system treats them identically.

3. **No position-level customization** — a stock up 35% with a 10% average earnings move can absorb a miss. A stock up 4% with a 15% average move cannot. The strategy should be completely different, but the current system doesn't do this math.

4. **Portfolio-level earnings clustering** — if 4 of your 8 positions report in the same week, that's concentrated binary event risk. Even if each individual position is manageable, the portfolio-level exposure to simultaneous earnings is not being assessed.

The Earnings Risk Analyst Agent replaces this reactive approach with a proactive, position-specific analysis that runs weekly (or whenever earnings are within a 3-week window), producing tailored strategies for every affected holding.

---

## 2. Agent Identity

### Role
Earnings Risk Analyst

### Goal
Analyze every portfolio position approaching an earnings report and produce a specific, risk-quantified pre-earnings strategy for each. Factor in historical earnings reactions, current profit cushion, position size, options pricing, and portfolio-level earnings concentration to recommend the optimal approach — ranging from "hold full" to "exit completely" — with clear risk/reward math for every option presented.

### Backstory
You are a pre-earnings risk specialist who has studied thousands of earnings reactions across growth stocks. You know that **earnings season is when most momentum investors give back their gains** — not because they pick bad stocks, but because they don't prepare for the binary event.

You've learned three things from experience. First, the average earnings gap is not the worst case — you plan for the tail risk, not the average. Second, profit cushion is the most important variable — a stock can absorb a gap if the cushion exceeds the expected move. Third, portfolio concentration matters — four positions all reporting in the same week creates correlated binary risk that no amount of individual position analysis can address.

You never say "hold and hope." Every position gets a strategy with quantified risk/reward. You present at minimum two options so the Portfolio Manager can make an informed choice based on their risk tolerance. Your principle: **know exactly what you can lose before the number drops.**

---

## 3. Behavioral Constraints

### 3.1 Methodology Rules — How We Analyze Earnings Risk

```
MUST-M1: Analyze every portfolio position with earnings within 
         21 calendar days (3 weeks). Begin analysis at 21 days out 
         to allow time for strategy execution (options need liquidity, 
         position trimming needs good exit prices).

MUST-M2: Retrieve historical earnings reaction data for at minimum 
         the last 8 quarters. Calculate:
         - Average post-earnings move (absolute value)
         - Average gap UP when beating estimates
         - Average gap DOWN when missing estimates
         - Beat rate (% of quarters with positive surprise)
         - Maximum single-quarter adverse move (worst case)
         - Standard deviation of moves (consistency)

MUST-M3: Calculate the "cushion ratio" for every position:
         Cushion Ratio = Current Gain % / Average Earnings Move %
         - Cushion Ratio > 2.0: COMFORTABLE — can likely absorb a miss
         - Cushion Ratio 1.0-2.0: MANAGEABLE — position survives average 
           miss but not worst case
         - Cushion Ratio 0.5-1.0: THIN — a miss likely turns the 
           position into a loss
         - Cushion Ratio < 0.5: INSUFFICIENT — even a small miss 
           erases the entire gain

MUST-M4: Present at minimum 2 strategy options for every position, 
         each with quantified risk/reward:
         Option A: The aggressive choice (hold full, or hold and add)
         Option B: The conservative choice (trim or exit)
         Option C (when applicable): The hedged choice (protective put)
         Each option must include:
         - Best case outcome (stock gaps up by average beat amount)
         - Base case outcome (stock moves by average absolute move)
         - Worst case outcome (stock moves by max historical adverse move)
         - Dollar impact on the portfolio for each scenario

MUST-M5: Assess earnings estimate revisions (the "whisper" signal):
         - Estimates revised UP in last 30 days: POSITIVE signal
         - Estimates unchanged: NEUTRAL
         - Estimates revised DOWN in last 30 days: NEGATIVE signal
         Combine with beat rate: a stock that beats 7/8 quarters 
         AND has rising estimates is a much lower risk than one 
         that beats 5/8 with flat estimates.

MUST-M6: Assess implied volatility and options pricing when available:
         - High IV relative to historical: market expects a big move
         - Compare implied move to historical average move
         - If options-based implied move > 1.5x historical average, 
           flag as ELEVATED_EXPECTATIONS
         - Calculate cost of protective put as % of position value

MUST-M7: Assess portfolio-level earnings concentration:
         - Count positions with earnings in the same week
         - Calculate aggregate portfolio % exposed to same-week earnings
         - If > 30% of portfolio value has earnings in the same week, 
           flag as CONCENTRATED_EARNINGS_RISK
         - If > 50%, flag as CRITICAL_CONCENTRATION

MUST-M8: Factor market regime into earnings strategy:
         - CONFIRMED_UPTREND: Standard analysis, full range of strategies
         - UPTREND_UNDER_PRESSURE: Bias toward conservative strategies. 
           Post-earnings reactions tend to skew negative in weakening markets.
         - CORRECTION: Recommend exiting or maximum hedging for all 
           positions approaching earnings. Don't hold binary risk 
           in a hostile market environment.
```

### 3.2 Safety Rules — Lines Never Crossed

```
MUST_NOT-S1: MUST NOT recommend holding full position through earnings 
             if the cushion ratio is below 0.5 AND the market regime is 
             UPTREND_UNDER_PRESSURE or CORRECTION. This combination — 
             thin cushion plus weak market — is the highest-risk scenario.

MUST_NOT-S2: MUST NOT recommend adding to a position before earnings 
             unless ALL of the following are true:
             - Beat rate ≥ 7/8 quarters (87.5%+)
             - Cushion ratio ≥ 2.0
             - Market regime is CONFIRMED_UPTREND
             - Position is < 5% of portfolio
             Adding before earnings is the most aggressive strategy 
             and requires the highest conviction threshold.

MUST_NOT-S3: MUST NOT ignore position size when assessing risk. 
             A 2% portfolio position with thin cushion is manageable. 
             A 12% portfolio position with the same cushion is a 
             portfolio-level threat. Always factor position size into 
             strategy recommendations.

MUST_NOT-S4: MUST NOT present dollar-impact calculations without 
             showing the math. Every scenario (best/base/worst) must 
             include: share count × price move = dollar impact. 
             The Portfolio Manager needs to verify the arithmetic.

MUST_NOT-S5: MUST NOT recommend specific options strikes, expirations, 
             or complex multi-leg options strategies. The agent can 
             recommend "consider a protective put" and estimate its cost, 
             but the specific execution should be done by the investor 
             or a qualified options advisor.

MUST_NOT-S6: MUST NOT make predictions about whether the company will 
             beat or miss earnings. The agent quantifies risk and 
             presents strategies — it does not forecast results. 
             Phrases like "I think they'll beat" or "this company 
             will likely miss" are prohibited.

MUST_NOT-S7: MUST NOT use analyst consensus as a certainty signal. 
             "All 30 analysts say buy" does not reduce earnings risk. 
             Consensus can be — and frequently is — wrong. Only 
             historical reaction data and cushion math drive strategy.
```

### 3.3 Quality Rules — Standards We Maintain

```
MUST-Q1: Present a pre-earnings summary dashboard for each position:
         Ticker | Earnings Date | Days Until | Cushion % | Cushion Ratio |
         Avg Move | Worst Case Move | Beat Rate | Position % | Risk Level

MUST-Q2: Classify every position's earnings risk:
         - LOW: Cushion ratio > 2.0, beat rate > 75%, healthy market
         - MODERATE: Cushion ratio 1.0-2.0, or mixed signals
         - HIGH: Cushion ratio 0.5-1.0, or large position with moderate cushion
         - CRITICAL: Cushion ratio < 0.5, or > 8% portfolio position 
           with cushion ratio < 1.0

MUST-Q3: For every strategy option, present a scenario table:
         
         Strategy: Hold Full Position
         ┌──────────┬───────────┬──────────┬──────────────┐
         │ Scenario │ Move %    │ New P&L  │ Dollar Impact│
         ├──────────┼───────────┼──────────┼──────────────┤
         │ Best     │ +12%      │ +32%     │ +$12,000     │
         │ Base     │ -8%       │ +12%     │ -$8,000      │
         │ Worst    │ -18%      │ +2%      │ -$18,000     │
         └──────────┴───────────┴──────────┴──────────────┘
         
         This is non-negotiable. No strategy without a scenario table.

MUST-Q4: Say "DATA UNAVAILABLE" for any metric that cannot be retrieved.
         If historical earnings data is unavailable (e.g., recent IPO), 
         flag the position as HIGH risk by default with note: 
         "Insufficient history — treating as elevated risk."

MUST-Q5: Provide a portfolio-level earnings calendar:
         Week of Feb 24: CRM (8% of portfolio), NVDA (12%)
         Week of Mar 3: PANW (5%)
         Week of Mar 10: (none)
         Total portfolio % exposed in next 3 weeks: 25%

MUST-Q6: Include a recommended strategy for each position — the agent's
         top recommendation among the options presented, with 1-2 
         sentence rationale. The Portfolio Manager can override, 
         but the agent must take a position (not just present options).
```

### 3.4 Priority Resolution

```
PRIORITY 1: Safety rules override methodology rules. If MUST_NOT-S1 
            says don't hold full position (thin cushion + weak market), 
            this overrides any methodology-based "hold" logic.

PRIORITY 2: Portfolio-level concentration (MUST-M7) can override 
            individual position analysis. Even if each individual 
            position has adequate cushion, if 4 positions report in 
            the same week representing 40% of portfolio, the agent 
            must recommend reducing exposure in at least some of them.

PRIORITY 3: Market regime (MUST-M8) adjusts the base strategy 
            aggressiveness. In CORRECTION, all strategies shift 
            one level conservative (hold → trim, trim → exit).

PRIORITY 4: Position size (MUST_NOT-S3) amplifies risk classification. 
            A stock with MODERATE individual risk becomes HIGH risk 
            if it's > 10% of portfolio.
```

---

## 4. Pydantic Contracts

### 4.1 Input Schema

```python
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from datetime import date


class MarketRegime(str, Enum):
    CONFIRMED_UPTREND = "CONFIRMED_UPTREND"
    UPTREND_UNDER_PRESSURE = "UPTREND_UNDER_PRESSURE"
    CORRECTION = "CORRECTION"
    RALLY_ATTEMPT = "RALLY_ATTEMPT"
    FOLLOW_THROUGH_DAY = "FOLLOW_THROUGH_DAY"


class AccountName(str, Enum):
    SCHWAB = "SCHWAB"
    ETRADE = "ETRADE"
    HSA_BANK = "HSA_BANK"


class Position(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    account: AccountName = Field(description="Which account holds the position")
    shares: int = Field(description="Number of shares held")
    buy_date: date = Field(description="Date position was opened")
    buy_price: float = Field(description="Average cost basis per share")
    current_price: float = Field(description="Current market price per share")
    position_value: float = Field(description="Current market value of position")
    portfolio_pct: float = Field(description="Position as % of total portfolio value")
    gain_loss_pct: float = Field(description="Current unrealized gain/loss %")


class EarningsRiskInput(BaseModel):
    positions: list[Position] = Field(description="All open positions across accounts")
    total_portfolio_value: float = Field(description="Total portfolio value across all accounts")
    market_regime: MarketRegime = Field(description="Current regime from Regime Detector")
    analysis_date: date = Field(description="Date of analysis")
    lookforward_days: int = Field(
        default=21,
        description="How far ahead to scan for earnings (default 21 calendar days)"
    )
```

### 4.2 Output Schema

```python
class EarningsRisk(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CushionCategory(str, Enum):
    COMFORTABLE = "COMFORTABLE"       # Ratio > 2.0
    MANAGEABLE = "MANAGEABLE"         # Ratio 1.0-2.0
    THIN = "THIN"                     # Ratio 0.5-1.0
    INSUFFICIENT = "INSUFFICIENT"     # Ratio < 0.5


class EstimateRevision(str, Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


class ImpliedVolSignal(str, Enum):
    NORMAL = "NORMAL"
    ELEVATED_EXPECTATIONS = "ELEVATED_EXPECTATIONS"


class StrategyType(str, Enum):
    HOLD_FULL = "HOLD_FULL"
    HOLD_AND_ADD = "HOLD_AND_ADD"
    TRIM_TO_HALF = "TRIM_TO_HALF"
    TRIM_TO_QUARTER = "TRIM_TO_QUARTER"
    EXIT_BEFORE_EARNINGS = "EXIT_BEFORE_EARNINGS"
    HEDGE_WITH_PUT = "HEDGE_WITH_PUT"


class ScenarioOutcome(BaseModel):
    scenario: str = Field(description="'BEST', 'BASE', or 'WORST'")
    expected_move_pct: float = Field(description="Expected stock price move %")
    resulting_gain_loss_pct: float = Field(description="Position gain/loss after move")
    dollar_impact: float = Field(description="Dollar impact on portfolio")
    math: str = Field(
        description="Show the calculation: '100 shares × $15.00 move = $1,500'"
    )


class StrategyOption(BaseModel):
    strategy: StrategyType
    description: str = Field(description="1-2 sentence description of what to do")
    shares_to_sell: Optional[int] = Field(
        default=None,
        description="Shares to sell before earnings. None for HOLD strategies."
    )
    estimated_hedge_cost: Optional[float] = Field(
        default=None,
        description="Cost of protective put if HEDGE strategy. None otherwise."
    )
    scenarios: list[ScenarioOutcome] = Field(
        description="Best, base, and worst case outcomes for this strategy"
    )
    risk_reward_summary: str = Field(
        description="1 sentence: 'Risk $X to make $Y' or 'Max loss $X, max gain $Y'"
    )


class HistoricalEarnings(BaseModel):
    quarters_analyzed: int = Field(description="Number of quarters with data (target 8)")
    beat_count: int = Field(description="Quarters with positive surprise")
    miss_count: int = Field(description="Quarters with negative surprise")
    beat_rate_pct: float = Field(description="Beat percentage")
    avg_move_pct: float = Field(description="Average absolute post-earnings move %")
    avg_gap_up_pct: float = Field(description="Average move on beats")
    avg_gap_down_pct: float = Field(description="Average move on misses (negative)")
    max_adverse_move_pct: float = Field(description="Worst single-quarter move (negative)")
    move_std_dev: float = Field(description="Standard deviation — consistency measure")
    recent_trend: str = Field(description="Trending better, worse, or consistent")


class PositionEarningsAnalysis(BaseModel):
    ticker: str
    account: AccountName
    earnings_date: date
    days_until_earnings: int
    reporting_time: str = Field(description="'BEFORE_OPEN', 'AFTER_CLOSE', or 'UNKNOWN'")
    
    # Current position context
    shares: int
    current_price: float
    gain_loss_pct: float
    position_value: float
    portfolio_pct: float
    
    # Historical analysis
    historical: HistoricalEarnings
    
    # Cushion analysis
    cushion_ratio: float = Field(description="gain_loss_pct / avg_move_pct")
    cushion_category: CushionCategory
    
    # Forward signals
    estimate_revision: EstimateRevision
    estimate_revision_detail: str = Field(
        description="Specific: 'EPS revised from $1.85 to $1.92 (+3.8%) in last 30 days'"
    )
    implied_vol_signal: ImpliedVolSignal
    implied_move_pct: Optional[float] = Field(
        default=None,
        description="Options-implied expected move %. None if unavailable."
    )
    
    # Risk classification
    risk_level: EarningsRisk
    risk_factors: list[str] = Field(
        description="Specific factors: 'Thin cushion (ratio 0.8)', 'Large position (11%)'"
    )
    
    # Strategies
    strategies: list[StrategyOption] = Field(
        description="Minimum 2 strategy options, each with scenario tables"
    )
    recommended_strategy: StrategyType
    recommendation_rationale: str = Field(
        description="2-3 sentences: why this strategy over the others"
    )


class EarningsWeek(BaseModel):
    week_start: date
    week_label: str = Field(description="'Week of Feb 24'")
    positions_reporting: list[str] = Field(description="Tickers reporting this week")
    aggregate_portfolio_pct: float
    concentration_flag: Optional[str] = Field(
        default=None,
        description="'CONCENTRATED' if > 30%, 'CRITICAL' if > 50%. None otherwise."
    )


class PortfolioEarningsConcentration(BaseModel):
    total_positions_approaching: int
    total_portfolio_pct_exposed: float
    earnings_calendar: list[EarningsWeek]
    concentration_risk: str = Field(description="'LOW', 'MODERATE', 'HIGH', or 'CRITICAL'")
    concentration_recommendation: Optional[str] = Field(
        default=None,
        description="Portfolio-level recommendation if concentration is HIGH or CRITICAL"
    )


class EarningsRiskOutput(BaseModel):
    analysis_date: date
    market_regime: MarketRegime
    lookforward_days: int
    
    analyses: list[PositionEarningsAnalysis] = Field(
        description="One analysis per position with upcoming earnings, ordered by date"
    )
    positions_clear: list[str] = Field(
        description="Tickers with no earnings in the lookforward window"
    )
    concentration: PortfolioEarningsConcentration
    
    executive_summary: str = Field(
        description="3-4 sentences: positions affected, highest risk, "
                    "key actions, concentration status"
    )
```

---

## 5. Tools

### 5.1 Tool: Get Earnings Calendar

```python
@tool("Get Earnings Calendar")
def get_earnings_calendar(ticker: str) -> dict:
    """Get the next scheduled earnings report date for a stock, 
    whether it reports before market open or after close, and the 
    number of calendar days until the report. Use this as the first 
    check to determine which positions need pre-earnings analysis."""
    # Returns: next_earnings_date, reporting_time, days_until_earnings, confirmed
```

### 5.2 Tool: Get Historical Earnings Reactions

```python
@tool("Get Historical Earnings Reactions")
def get_historical_earnings_reactions(ticker: str, quarters: int = 8) -> dict:
    """Get historical post-earnings stock price reactions for the last N quarters. 
    Returns the EPS estimate, actual EPS, surprise %, and the stock's 
    next-day price change for each quarter. This is the most important data 
    for earnings risk quantification — a stock that gaps 5% on average is 
    very different from one that gaps 20%."""
    # Returns: reactions: [{quarter, eps_estimate, eps_actual, surprise_pct, 
    #          stock_move_pct, volume_ratio}],
    #          summary: {beat_count, miss_count, avg_abs_move, avg_gap_up, 
    #          avg_gap_down, max_adverse_move, std_dev}
```

### 5.3 Tool: Get Estimate Revisions

```python
@tool("Get Estimate Revisions")
def get_estimate_revisions(ticker: str) -> dict:
    """Get analyst estimate revisions over the last 30, 60, and 90 days. 
    Rising estimates correlate with higher beat rates. Falling estimates 
    are a warning even if the stock has historically beaten."""
    # Returns: current_eps_estimate, estimate_30d_ago, estimate_60d_ago,
    #          revisions_up_30d, revisions_down_30d, net_direction,
    #          revenue_estimate, revenue_revisions_direction
```

### 5.4 Tool: Get Options Implied Move

```python
@tool("Get Options Implied Move")
def get_options_implied_move(ticker: str) -> dict:
    """Get the options-implied expected move for the upcoming earnings 
    report, derived from the at-the-money straddle price. Also returns 
    the cost of a protective put. Use this to compare market expectations 
    vs historical moves — if implied move is much larger than average, 
    the market is pricing in elevated uncertainty."""
    # Returns: implied_move_pct, implied_move_dollars, straddle_cost,
    #          protective_put_strike, protective_put_cost, 
    #          protective_put_cost_pct, current_iv_rank
```

### 5.5 Tool: Get Stock Ratings

```python
@tool("Get Stock Ratings")
def get_stock_ratings(ticker: str) -> dict:
    """Get IBD ratings including RS Rating, EPS Rating, Composite Rating, 
    and Accumulation/Distribution grade. Use as supplemental context — 
    strong ratings suggest institutional support that may cushion weakness, 
    while deteriorating ratings suggest institutions are already reducing."""
    # Returns: rs_rating, eps_rating, composite_rating, accum_dist_grade, sector_rank
```

---

## 6. Tool Selection Strategy

```
For each position in portfolio:
│
├─ Step 1: get_earnings_calendar(ticker)
│   └─ Earnings > 21 days away? → SKIP. Add to positions_clear list.
│   └─ Earnings ≤ 21 days? → Continue to full analysis.
│
├─ Step 2: get_historical_earnings_reactions(ticker, 8)
│   └─ Calculate beat rate, avg move, worst case, consistency
│   └─ If < 4 quarters available: flag HIGH risk, note gaps
│
├─ Step 3: Calculate cushion ratio (pure math, no tool call)
│   └─ Cushion Ratio = gain_loss_pct / avg_abs_move
│   └─ Classify: COMFORTABLE / MANAGEABLE / THIN / INSUFFICIENT
│
├─ Step 4: get_estimate_revisions(ticker)
│   └─ Forward signal: rising, flat, or falling estimates
│
├─ Step 5: get_options_implied_move(ticker) — CONDITIONAL
│   └─ Only if stock has liquid options
│   └─ Compare implied move to historical average
│   └─ Get protective put cost for hedge strategy
│   └─ If unavailable: present strategies without hedge option
│
├─ Step 6: get_stock_ratings(ticker) — CONDITIONAL
│   └─ Only if risk level is MODERATE or higher
│   └─ Supplemental institutional support context
│
└─ Step 7: Build strategy options and scenario tables
    └─ Minimum 2 options with best/base/worst math
    └─ Factor in regime and position size

After all positions:
└─ Step 8: Portfolio-level concentration analysis
    └─ Group by earnings week
    └─ Flag if > 30% in one week

Efficiency:
- No earnings in window: 1 tool call per position
- LOW risk: 3 tool calls (calendar, history, revisions)
- MODERATE+ risk: 4-5 tool calls (add options + ratings)
```

---

## 7. Strategy Decision Framework

```
                            Cushion Ratio
                   <0.5    0.5-1.0   1.0-2.0   >2.0
                 ┌─────────┬─────────┬─────────┬─────────┐
CORRECTION       │ EXIT    │ EXIT    │ TRIM    │ TRIM    │
                 │         │         │ TO HALF │ TO HALF │
                 ├─────────┼─────────┼─────────┼─────────┤
UNDER            │ EXIT    │ TRIM    │ TRIM    │ HOLD    │
PRESSURE         │         │ TO HALF │ TO HALF │ OR HEDGE│
                 ├─────────┼─────────┼─────────┼─────────┤
CONFIRMED        │ TRIM    │ TRIM    │ HOLD    │ HOLD    │
UPTREND          │ TO HALF │ OR HEDGE│ OR HEDGE│ FULL    │
                 └─────────┴─────────┴─────────┴─────────┘

Modifiers that shift one level conservative:
  - Position > 10% of portfolio
  - Beat rate < 50%
  - Estimates revised down
  - Implied move > 1.5x historical average
  - 3+ positions reporting same week

Modifiers that shift one level aggressive:
  - Beat rate ≥ 87.5% (7/8+)
  - Estimates revised up
  - Implied move < historical average
  - Position < 3% of portfolio

Special: HOLD_AND_ADD requires ALL conditions from MUST_NOT-S2
```

---

## 8. CrewAI Implementation

### 8.1 Agent Definition

```python
from crewai import Agent

earnings_risk_analyst = Agent(
    role="Earnings Risk Analyst",
    goal=(
        "Analyze every portfolio position approaching earnings and produce "
        "a specific, risk-quantified pre-earnings strategy for each. Calculate "
        "cushion ratios, retrieve historical earnings reactions, assess estimate "
        "revisions and implied volatility, and present at minimum 2 strategy "
        "options with scenario tables showing best/base/worst case dollar impacts. "
        "Also assess portfolio-level earnings concentration risk. Never predict "
        "whether a company will beat or miss — quantify risk and present strategies."
    ),
    backstory=(
        "You are a pre-earnings risk specialist who has studied thousands of "
        "earnings reactions across growth stocks. You know that earnings season "
        "is when most momentum investors give back gains — not from bad stock "
        "picks but from poor preparation for binary events. The average gap is "
        "not the worst case — plan for tail risk. Profit cushion is the most "
        "important variable. Portfolio concentration of same-week earnings creates "
        "correlated risk. You never say 'hold and hope.' Every position gets "
        "quantified risk/reward. Your principle: know exactly what you can lose "
        "before the number drops."
    ),
    tools=[
        get_earnings_calendar,
        get_historical_earnings_reactions,
        get_estimate_revisions,
        get_options_implied_move,
        get_stock_ratings,
    ],
    verbose=True,
    allow_delegation=False,
    max_iter=30,
    max_rpm=10,
)
```

### 8.2 Task Definition

```python
from crewai import Task

earnings_risk_task = Task(
    description=(
        "Analyze all portfolio positions for upcoming earnings risk.\n\n"
        "Positions:\n{positions}\n\n"
        "Total Portfolio Value: ${total_portfolio_value:,.0f}\n"
        "Market Regime: {market_regime}\n"
        "Analysis Date: {analysis_date}\n"
        "Lookforward Window: {lookforward_days} days\n\n"
        "For EACH position:\n"
        "1. Check if earnings are within the lookforward window\n"
        "2. Retrieve last 8 quarters of earnings reactions\n"
        "3. Calculate cushion ratio (current gain / average earnings move)\n"
        "4. Check analyst estimate revisions\n"
        "5. Get options-implied expected move if available\n"
        "6. Classify risk: LOW, MODERATE, HIGH, or CRITICAL\n"
        "7. Present minimum 2 strategy options with scenario tables\n"
        "8. Select recommended strategy with rationale\n\n"
        "Then assess PORTFOLIO-LEVEL earnings concentration:\n"
        "- Group positions by earnings week\n"
        "- Flag if > 30% of portfolio reports in same week\n\n"
        "CRITICAL RULES:\n"
        "- Every scenario table must show the math\n"
        "- NEVER predict beat or miss\n"
        "- In CORRECTION: bias all strategies one level conservative\n"
        "- Cushion ratio < 0.5 + weak market = MUST NOT hold full\n"
    ),
    expected_output=(
        "Complete earnings risk analysis with per-position strategies, "
        "scenario tables, risk classifications, portfolio concentration, "
        "and executive summary."
    ),
    agent=earnings_risk_analyst,
    output_pydantic=EarningsRiskOutput,
)
```

### 8.3 Crew Integration

```python
# Runs AFTER Regime Detector (needs regime) 
# and BEFORE Portfolio Manager (PM uses earnings risk for sizing)

ibd_crew = Crew(
    agents=[
        regime_detector,
        research_agent,
        analyst_agent,
        sector_strategist,
        watchlist_monitor,
        exit_strategist,
        earnings_risk_analyst,     # After analysis, before PM
        portfolio_manager,
        risk_officer,
        educator,
    ],
    tasks=[
        regime_task,
        research_task,
        analysis_task,
        sector_task,
        watchlist_task,
        exit_strategy_task,
        earnings_risk_task,
        portfolio_task,
        risk_task,
        education_task,
    ],
    process=Process.hierarchical,
    manager_agent=portfolio_manager,
    verbose=True,
)
```

---

## 9. Golden Dataset

```python
golden_earnings_tests = [

    # ── COMFORTABLE CUSHION, HEALTHY MARKET ──────────────────────────

    {
        "id": "ERN-001",
        "name": "large_cushion_strong_history_hold",
        "description": "Stock up 35%, avg move 8%, beat 7/8. Safe to hold full.",
        "rules_tested": ["MUST-M3", "MUST-M4"],
        "input": {
            "positions": [{
                "ticker": "NVDA", "account": "SCHWAB", "shares": 100,
                "buy_price": 120.00, "current_price": 162.00,
                "gain_loss_pct": 35.0, "portfolio_pct": 10.0,
                "position_value": 16200.00,
            }],
            "total_portfolio_value": 162000,
            "market_regime": "CONFIRMED_UPTREND",
        },
        "mock_tool_returns": {
            "get_earnings_calendar": {"days_until_earnings": 14, "reporting_time": "AFTER_CLOSE"},
            "get_historical_earnings_reactions": {
                "summary": {
                    "beat_count": 7, "miss_count": 1, "avg_abs_move": 8.0,
                    "avg_gap_up": 10.5, "avg_gap_down": -12.0,
                    "max_adverse_move": -15.0, "std_dev": 4.2,
                }
            },
            "get_estimate_revisions": {"net_direction": "POSITIVE"},
        },
        "expected": {
            "cushion_ratio_approx": 4.375,
            "cushion_category": "COMFORTABLE",
            "risk_level": "LOW",
            "recommended_strategy": "HOLD_FULL",
            "strategies_count_min": 2,
            "scenarios_per_strategy": 3,
        },
    },

    # ── THIN CUSHION, HEALTHY MARKET ─────────────────────────────────

    {
        "id": "ERN-002",
        "name": "thin_cushion_uptrend_trim_recommended",
        "description": "Stock up 6%, avg move 12%. Cushion ratio 0.5 — thin.",
        "rules_tested": ["MUST-M3", "MUST-M4"],
        "input": {
            "positions": [{
                "ticker": "CRM", "account": "ETRADE", "shares": 50,
                "buy_price": 320.00, "current_price": 339.20,
                "gain_loss_pct": 6.0, "portfolio_pct": 8.0,
                "position_value": 16960.00,
            }],
            "total_portfolio_value": 212000,
            "market_regime": "CONFIRMED_UPTREND",
        },
        "mock_tool_returns": {
            "get_earnings_calendar": {"days_until_earnings": 10},
            "get_historical_earnings_reactions": {
                "summary": {
                    "beat_count": 5, "miss_count": 3, "avg_abs_move": 12.0,
                    "avg_gap_down": -16.0, "max_adverse_move": -22.0,
                }
            },
            "get_estimate_revisions": {"net_direction": "NEUTRAL"},
            "get_options_implied_move": {"implied_move_pct": 14.0},
        },
        "expected": {
            "cushion_ratio_approx": 0.5,
            "cushion_category": "THIN",
            "risk_level": "HIGH",
            "recommended_strategy_one_of": ["TRIM_TO_HALF", "HEDGE_WITH_PUT"],
            "recommended_must_not_be": "HOLD_FULL",
        },
    },

    # ── INSUFFICIENT CUSHION + WEAK MARKET (MUST_NOT-S1) ─────────────

    {
        "id": "ERN-003",
        "name": "insufficient_cushion_weak_market_must_not_hold",
        "description": "Cushion ratio 0.3, market under pressure. MUST NOT hold full.",
        "rules_tested": ["MUST_NOT-S1", "MUST-M8"],
        "input": {
            "positions": [{
                "ticker": "DDOG", "account": "HSA_BANK", "shares": 60,
                "buy_price": 150.00, "current_price": 154.50,
                "gain_loss_pct": 3.0, "portfolio_pct": 5.0,
                "position_value": 9270.00,
            }],
            "total_portfolio_value": 185400,
            "market_regime": "UPTREND_UNDER_PRESSURE",
        },
        "mock_tool_returns": {
            "get_earnings_calendar": {"days_until_earnings": 8},
            "get_historical_earnings_reactions": {
                "summary": {
                    "beat_count": 6, "miss_count": 2, "avg_abs_move": 10.0,
                    "max_adverse_move": -18.0,
                }
            },
            "get_estimate_revisions": {"net_direction": "NEGATIVE"},
        },
        "expected": {
            "cushion_ratio_approx": 0.3,
            "cushion_category": "INSUFFICIENT",
            "risk_level": "CRITICAL",
            "recommended_strategy_one_of": ["EXIT_BEFORE_EARNINGS", "TRIM_TO_QUARTER"],
            "strategy_must_not_include": "HOLD_FULL",
            "strategy_must_not_include_2": "HOLD_AND_ADD",
        },
    },

    # ── HOLD_AND_ADD ALL CONDITIONS MET (MUST_NOT-S2) ────────────────

    {
        "id": "ERN-004",
        "name": "hold_and_add_all_conditions_met",
        "description": "Beat 8/8, cushion 3.0, uptrend, small position. Can add.",
        "rules_tested": ["MUST_NOT-S2"],
        "input": {
            "positions": [{
                "ticker": "AXON", "account": "SCHWAB", "shares": 20,
                "buy_price": 400.00, "current_price": 520.00,
                "gain_loss_pct": 30.0, "portfolio_pct": 4.0,
                "position_value": 10400.00,
            }],
            "total_portfolio_value": 260000,
            "market_regime": "CONFIRMED_UPTREND",
        },
        "mock_tool_returns": {
            "get_earnings_calendar": {"days_until_earnings": 12},
            "get_historical_earnings_reactions": {
                "summary": {
                    "beat_count": 8, "miss_count": 0, "avg_abs_move": 10.0,
                    "max_adverse_move": -5.0,
                }
            },
            "get_estimate_revisions": {"net_direction": "POSITIVE"},
        },
        "expected": {
            "cushion_ratio_approx": 3.0,
            "risk_level": "LOW",
            "strategy_may_include": "HOLD_AND_ADD",
        },
    },

    {
        "id": "ERN-005",
        "name": "hold_and_add_blocked_large_position",
        "description": "Beat 8/8, good cushion, but 12% of portfolio. Cannot add.",
        "rules_tested": ["MUST_NOT-S2"],
        "input": {
            "positions": [{
                "ticker": "NVDA", "account": "SCHWAB", "shares": 200,
                "buy_price": 120.00, "current_price": 156.00,
                "gain_loss_pct": 30.0, "portfolio_pct": 12.0,
                "position_value": 31200.00,
            }],
            "total_portfolio_value": 260000,
            "market_regime": "CONFIRMED_UPTREND",
        },
        "mock_tool_returns": {
            "get_earnings_calendar": {"days_until_earnings": 15},
            "get_historical_earnings_reactions": {
                "summary": {"beat_count": 8, "miss_count": 0, "avg_abs_move": 8.0}
            },
        },
        "expected": {
            "strategy_must_not_include": "HOLD_AND_ADD",
            "risk_factors_must_mention": "position size",
        },
    },

    # ── SCENARIO TABLE MATH (MUST-Q3, MUST_NOT-S4) ──────────────────

    {
        "id": "ERN-006",
        "name": "scenario_tables_show_math",
        "description": "Every strategy must have 3 scenarios showing arithmetic.",
        "rules_tested": ["MUST-Q3", "MUST_NOT-S4"],
        "input": {
            "positions": [{
                "ticker": "PANW", "account": "ETRADE", "shares": 40,
                "buy_price": 390.00, "current_price": 429.00,
                "gain_loss_pct": 10.0, "portfolio_pct": 7.0,
                "position_value": 17160.00,
            }],
            "total_portfolio_value": 245143,
            "market_regime": "CONFIRMED_UPTREND",
        },
        "mock_tool_returns": {
            "get_earnings_calendar": {"days_until_earnings": 18},
            "get_historical_earnings_reactions": {
                "summary": {"avg_abs_move": 9.0, "max_adverse_move": -16.0}
            },
        },
        "expected": {
            "strategies_count_min": 2,
            "each_strategy_has_3_scenarios": True,
            "each_scenario_has_math_field": True,
            "math_field_contains_multiplication": True,
        },
    },

    # ── NO PREDICTIONS (MUST_NOT-S6) ─────────────────────────────────

    {
        "id": "ERN-007",
        "name": "no_beat_miss_predictions",
        "description": "Even with 8/8 beat rate, agent must not predict outcomes.",
        "rules_tested": ["MUST_NOT-S6"],
        "input": {
            "positions": [{
                "ticker": "AAPL", "account": "SCHWAB", "shares": 100,
                "buy_price": 200.00, "current_price": 230.00,
                "gain_loss_pct": 15.0, "portfolio_pct": 9.0,
            }],
            "market_regime": "CONFIRMED_UPTREND",
        },
        "mock_tool_returns": {
            "get_earnings_calendar": {"days_until_earnings": 7},
            "get_historical_earnings_reactions": {
                "summary": {"beat_count": 8, "miss_count": 0, "avg_abs_move": 5.0}
            },
            "get_estimate_revisions": {"net_direction": "POSITIVE"},
        },
        "expected": {
            "reasoning_must_not_mention": "will beat",
            "reasoning_must_not_mention_2": "likely to beat",
            "reasoning_must_not_mention_3": "expect a beat",
            "reasoning_must_not_mention_4": "will miss",
            "reasoning_must_not_mention_5": "should report strong",
        },
    },

    # ── ELEVATED IMPLIED VOL (MUST-M6) ───────────────────────────────

    {
        "id": "ERN-008",
        "name": "elevated_implied_vol_flagged",
        "description": "Implied move 18% vs historical 10%. 1.8x ratio flags elevated.",
        "rules_tested": ["MUST-M6"],
        "input": {
            "positions": [{
                "ticker": "SMCI", "account": "SCHWAB", "shares": 30,
                "buy_price": 800.00, "current_price": 920.00,
                "gain_loss_pct": 15.0, "portfolio_pct": 11.0,
                "position_value": 27600.00,
            }],
            "total_portfolio_value": 250909,
            "market_regime": "CONFIRMED_UPTREND",
        },
        "mock_tool_returns": {
            "get_earnings_calendar": {"days_until_earnings": 5},
            "get_historical_earnings_reactions": {
                "summary": {"avg_abs_move": 10.0, "max_adverse_move": -25.0}
            },
            "get_options_implied_move": {"implied_move_pct": 18.0},
        },
        "expected": {
            "implied_vol_signal": "ELEVATED_EXPECTATIONS",
            "risk_factors_must_mention": "implied",
            "risk_level_one_of": ["HIGH", "CRITICAL"],
        },
    },

    # ── PORTFOLIO CONCENTRATION (MUST-M7) ────────────────────────────

    {
        "id": "ERN-009",
        "name": "same_week_concentration_flagged",
        "description": "3 positions (35%) report same week. Must flag concentrated.",
        "rules_tested": ["MUST-M7"],
        "input": {
            "positions": [
                {"ticker": "NVDA", "portfolio_pct": 14.0},
                {"ticker": "CRM", "portfolio_pct": 12.0},
                {"ticker": "PANW", "portfolio_pct": 9.0},
                {"ticker": "AXON", "portfolio_pct": 5.0},
            ],
            "market_regime": "CONFIRMED_UPTREND",
        },
        "mock_tool_returns": {
            "get_earnings_calendar": {
                "NVDA": {"earnings_date": "2026-03-03"},
                "CRM": {"earnings_date": "2026-03-05"},
                "PANW": {"earnings_date": "2026-03-04"},
                "AXON": {"earnings_date": "2026-03-13"},
            },
        },
        "expected": {
            "concentration_risk_one_of": ["HIGH", "CRITICAL"],
            "same_week_pct_approx": 35.0,
            "concentration_flag": "CONCENTRATED",
        },
    },

    {
        "id": "ERN-010",
        "name": "critical_concentration_50_pct",
        "description": "4 positions (52%) report same week. CRITICAL concentration.",
        "rules_tested": ["MUST-M7"],
        "input": {
            "positions": [
                {"ticker": "NVDA", "portfolio_pct": 15.0},
                {"ticker": "MSFT", "portfolio_pct": 14.0},
                {"ticker": "GOOGL", "portfolio_pct": 13.0},
                {"ticker": "META", "portfolio_pct": 10.0},
            ],
            "market_regime": "CONFIRMED_UPTREND",
        },
        "mock_tool_returns": {
            "get_earnings_calendar": {
                "NVDA": {"earnings_date": "2026-03-04"},
                "MSFT": {"earnings_date": "2026-03-03"},
                "GOOGL": {"earnings_date": "2026-03-05"},
                "META": {"earnings_date": "2026-03-05"},
            },
        },
        "expected": {
            "concentration_flag": "CRITICAL",
            "concentration_recommendation_must_mention": "reduce",
        },
    },

    # ── CORRECTION SHIFTS STRATEGIES (MUST-M8) ──────────────────────

    {
        "id": "ERN-011",
        "name": "correction_shifts_conservative",
        "description": "Manageable cushion becomes TRIM in correction (not HOLD).",
        "rules_tested": ["MUST-M8"],
        "input": {
            "positions": [{
                "ticker": "NOW", "account": "ETRADE", "shares": 15,
                "buy_price": 950.00, "current_price": 1092.50,
                "gain_loss_pct": 15.0, "portfolio_pct": 7.0,
            }],
            "market_regime": "CORRECTION",
        },
        "mock_tool_returns": {
            "get_earnings_calendar": {"days_until_earnings": 12},
            "get_historical_earnings_reactions": {
                "summary": {"avg_abs_move": 10.0, "max_adverse_move": -16.0}
            },
        },
        "expected": {
            "cushion_ratio_approx": 1.5,
            "recommended_strategy_one_of": ["TRIM_TO_HALF", "EXIT_BEFORE_EARNINGS"],
            "recommended_must_not_be": "HOLD_FULL",
            "reasoning_must_mention": "correction",
        },
    },

    # ── RECENT IPO, NO HISTORY (MUST-Q4) ─────────────────────────────

    {
        "id": "ERN-012",
        "name": "no_history_defaults_high_risk",
        "description": "IPO'd 6 months ago. Only 2 quarters. Default HIGH risk.",
        "rules_tested": ["MUST-Q4"],
        "input": {
            "positions": [{
                "ticker": "NEWIPO", "account": "SCHWAB", "shares": 200,
                "buy_price": 45.00, "current_price": 52.00,
                "gain_loss_pct": 15.6, "portfolio_pct": 4.0,
            }],
            "market_regime": "CONFIRMED_UPTREND",
        },
        "mock_tool_returns": {
            "get_earnings_calendar": {"days_until_earnings": 9},
            "get_historical_earnings_reactions": {
                "summary": {"quarters_analyzed": 2, "avg_abs_move": 20.0}
            },
        },
        "expected": {
            "risk_level_one_of": ["HIGH", "CRITICAL"],
            "risk_factors_must_mention": "insufficient history",
        },
    },

    # ── NEGATIVE REVISIONS ELEVATE RISK (MUST-M5) ───────────────────

    {
        "id": "ERN-013",
        "name": "negative_revisions_elevate_risk",
        "description": "Estimates revised down pushes moderate to high risk.",
        "rules_tested": ["MUST-M5"],
        "input": {
            "positions": [{
                "ticker": "SHOP", "account": "SCHWAB", "shares": 80,
                "buy_price": 100.00, "current_price": 118.00,
                "gain_loss_pct": 18.0, "portfolio_pct": 6.0,
            }],
            "market_regime": "CONFIRMED_UPTREND",
        },
        "mock_tool_returns": {
            "get_earnings_calendar": {"days_until_earnings": 11},
            "get_historical_earnings_reactions": {
                "summary": {"beat_count": 5, "miss_count": 3, "avg_abs_move": 12.0}
            },
            "get_estimate_revisions": {
                "net_direction": "NEGATIVE",
                "revisions_down_30d": 6, "revisions_up_30d": 1,
            },
        },
        "expected": {
            "estimate_revision": "NEGATIVE",
            "risk_factors_must_mention": "estimate",
            "risk_level_one_of": ["MODERATE", "HIGH"],
        },
    },

    # ── LARGE POSITION AMPLIFIES RISK (MUST_NOT-S3) ─────────────────

    {
        "id": "ERN-014",
        "name": "large_position_amplifies_risk",
        "description": "Comfortable cushion but 15% of portfolio. Size amplifies risk.",
        "rules_tested": ["MUST_NOT-S3"],
        "input": {
            "positions": [{
                "ticker": "NVDA", "account": "SCHWAB", "shares": 250,
                "buy_price": 120.00, "current_price": 144.00,
                "gain_loss_pct": 20.0, "portfolio_pct": 15.0,
            }],
            "market_regime": "CONFIRMED_UPTREND",
        },
        "mock_tool_returns": {
            "get_earnings_calendar": {"days_until_earnings": 6},
            "get_historical_earnings_reactions": {
                "summary": {"avg_abs_move": 8.0, "max_adverse_move": -15.0}
            },
        },
        "expected": {
            "cushion_ratio_approx": 2.5,
            "risk_level_one_of": ["MODERATE", "HIGH"],
            "risk_factors_must_mention": "position size",
            "recommended_must_not_be": "HOLD_AND_ADD",
        },
    },

    # ── NO EARNINGS IN WINDOW (MUST-M1) ──────────────────────────────

    {
        "id": "ERN-015",
        "name": "no_earnings_in_window_skip",
        "description": "Earnings 45 days out. Should be in positions_clear.",
        "rules_tested": ["MUST-M1"],
        "input": {
            "positions": [{
                "ticker": "AAPL", "account": "SCHWAB", "shares": 100,
                "buy_price": 200.00, "current_price": 230.00,
                "gain_loss_pct": 15.0, "portfolio_pct": 9.0,
            }],
            "market_regime": "CONFIRMED_UPTREND",
        },
        "mock_tool_returns": {
            "get_earnings_calendar": {"days_until_earnings": 45},
        },
        "expected": {
            "analyses_count": 0,
            "positions_clear_includes": "AAPL",
        },
    },
]
```

---

## 10. Error Handling

```
SCENARIO                          BEHAVIOR
─────────────────────────────────────────────────────────────────
Earnings calendar tool fails      Flag: "Earnings date unknown — manual check 
                                  required." Include with risk_level = HIGH.

Historical data < 4 quarters      Flag: "Insufficient history." Default HIGH 
                                  risk. Use available data, note low confidence.

Historical data fully missing     risk_level = CRITICAL. Recommend TRIM or EXIT.
                                  Note: "No data — maximum uncertainty."

Options data unavailable          Present HOLD and TRIM only, no HEDGE option.
                                  Note: "Options data unavailable."

Estimate revisions tool fails     Classify as NEUTRAL (conservative default).

Position data inconsistency       Use calculated values. Flag discrepancy.

Portfolio value missing           Cannot calculate portfolio_pct. Use absolute 
                                  values. Flag: "Concentration risk unknown."

Pydantic validation fails         CrewAI retries. After 3 failures, partial 
                                  analysis with error note.
```

---

## 11. Performance Metrics

```
METRIC                            TARGET          HOW MEASURED
──────────────────────────────────────────────────────────────────
Earnings window coverage          100%            Every position within 21 days analyzed
Strategy completeness             100%            Every analysis has ≥ 2 strategies
Scenario table completeness       100%            Every strategy has 3 scenarios with math
Cushion ratio accuracy            100%            Arithmetic verified against input
No-prediction compliance          100%            Zero beat/miss predictions
Concentration detection           100%            All > 30% same-week clusters flagged
Risk classification accuracy      > 85%           Manual review
Tool calls per position           2-5             Calendar-only=1, full=4-5
Latency (10 positions)            < 90 seconds    All tool calls
Token cost per run                < $0.60         On Sonnet
```

---

## 12. Future Enhancements

```
VERSION   ENHANCEMENT                                  VALUE
──────────────────────────────────────────────────────────────────
v1.1      Sector earnings clustering                   Correlated risk within sectors
v1.2      Post-earnings reaction tracker               Grade strategy vs outcomes
v1.3      Revenue and guidance analysis                Forward guidance drives stocks
v1.4      Cross-account tax optimization               Feed to Tax-Aware Rebalancer
v2.0      Real-time earnings call monitor              Immediate post-report analysis
```

---

## Appendix A: Cushion Ratio Quick Reference

```
RATIO      CATEGORY        MEANING                         TYPICAL STRATEGY
──────────────────────────────────────────────────────────────────────────────
> 2.0      COMFORTABLE     Absorbs worst case, stays profitable    HOLD FULL
1.0-2.0    MANAGEABLE      Survives average miss, not worst case   HOLD or HEDGE
0.5-1.0    THIN            Average miss erases most gain           TRIM TO HALF
< 0.5      INSUFFICIENT    Even small miss creates a loss          EXIT or TRIM ¼

Example: Stock up 20%, avg move 8% → 20/8 = 2.5 → COMFORTABLE
Example: Stock up 6%, avg move 12% → 6/12 = 0.5 → THIN
```

---

## Appendix B: Example Agent Reasoning Trace

```
[Earnings Risk Analyst] Analyzing CRM for upcoming earnings.

Action: get_earnings_calendar(ticker="CRM")
Result: earnings_date=2026-03-05, days_until=12, reporting_time=AFTER_CLOSE

Thought: CRM reports in 12 days — within the 21-day window. 
         Need full analysis.

Action: get_historical_earnings_reactions(ticker="CRM", quarters=8)
Result: beat_count=5, miss_count=3, avg_abs_move=12.0%, 
        avg_gap_up=9.0%, avg_gap_down=-16.0%, max_adverse=-22.0%

Thought: Beat rate is 62.5% (5/8) — below 75%. Average move is 12%.
         CRM's current gain is 6.25%.
         Cushion Ratio = 6.25% / 12.0% = 0.52 → THIN (barely)
         
         Worst case: -22% move → 6.25% gain becomes -15.75% loss.
         That's a $5,420 loss on a $16,960 position.

Action: get_estimate_revisions(ticker="CRM")
Result: net_direction=NEUTRAL, estimates flat at $2.85

Thought: No positive or negative signal from revisions.

Action: get_options_implied_move(ticker="CRM")
Result: implied_move_pct=14.0%, protective_put_cost_pct=2.8%

Thought: Implied move 14% vs historical 12% = 1.17x. Not ELEVATED 
         (needs 1.5x), but higher than average.
         
         Risk classification:
         - Cushion ratio 0.52 → THIN → starts at HIGH
         - Beat rate 62.5% < 75% → confirms HIGH
         - Position 8% → moderate, no amplification
         - Estimates NEUTRAL → no modifier
         - Implied vol normal → no modifier
         → Risk Level: HIGH

Strategy A: HOLD FULL (50 shares at $339.20)
  Best:  +9.0% = +$1,526  (50 × $339.20 × 0.09)  → new gain 15.25%
  Base:  -12.0% = -$2,035  (50 × $339.20 × 0.12)  → new loss -5.75%
  Worst: -22.0% = -$3,731  (50 × $339.20 × 0.22)  → new loss -15.75%
  Risk/Reward: Risk $3,731 to make $1,526

Strategy B: TRIM TO HALF (sell 25, hold 25)
  Lock $531 profit on 25 sold shares
  Best:  +$763 on remaining + $531 locked = +$1,294 net
  Base:  -$1,018 on remaining + $531 locked = -$487 net
  Worst: -$1,866 on remaining + $531 locked = -$1,335 net
  Risk/Reward: Risk $1,335 to make $1,294

Strategy C: HEDGE (hold 50, buy put at ~$329)
  Put cost: $475 (2.8% of position)
  Best:  +$1,526 - $475 = +$1,051 net
  Base:  Put limits loss to ~$950 + $475 = -$1,425
  Worst: Put limits loss to ~$950 + $475 = -$1,425
  Risk/Reward: Max risk $1,425, max gain $1,051

RECOMMENDED: TRIM TO HALF

Rationale: Cushion ratio of 0.52 and 62.5% beat rate make the 
risk/reward of holding full unfavorable — risking $3,731 to make 
$1,526 (2.4:1 downside/upside). Trimming locks profit on half 
the position and cuts worst-case impact by 64%. If CRM beats, 
the remaining 25 shares still capture upside.
```