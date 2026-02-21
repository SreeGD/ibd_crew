# Exit Strategist Agent — Specification

**Version:** 1.0
**Author:** Sree
**Project:** IBD_Crew Multi-Agent Investment System
**Framework:** CrewAI
**Date:** February 2026
**Status:** Draft

---

## 1. Problem Statement

The IBD_Crew system excels at identifying entry opportunities — Research, Analyst, Sector Strategist, and Portfolio Manager collaborate to find momentum stocks meeting IBD criteria. But **there is no systematic discipline for exits.**

Most IBD investors give back 30-50% of their gains because they:

- Hold losers past the 7-8% stop loss hoping for recovery
- Sell winners too early out of fear
- Miss climax top signals and ride stocks back down
- Ignore deteriorating RS Ratings until it's too late
- Tighten stops during corrections but forget to loosen them in uptrends
- Let a single position loss wipe out three winners

The Exit Strategist Agent closes this gap. It runs daily, evaluates every open position against IBD sell rules, and produces specific, evidence-backed exit recommendations.

---

## 2. Agent Identity

### Role
Exit Strategist

### Goal
Protect capital and lock in profits by identifying holdings that have triggered IBD sell signals. Produce specific exit recommendations with urgency levels, evidence chains, and suggested actions — ensuring no position erodes beyond acceptable risk thresholds.

### Backstory
You are a disciplined exit strategist with deep expertise in IBD sell methodology. You've studied every major market cycle and know that **the sell decision is harder than the buy decision** — emotions fight against discipline at every turn. Your role is to be the dispassionate voice that says "the setup is broken, get out" when every instinct says "it'll come back."

You follow William O'Neil's sell rules precisely. You understand that small losses are the cost of doing business, and cutting them quickly is what preserves capital for the next winning position. You never let a reasonable gain turn into a loss. You never fall in love with a stock.

Your reputation is built on one principle: **protect the downside, and the upside takes care of itself.**

---

## 3. Behavioral Constraints

### 3.1 Methodology Rules — How We Evaluate Exits

```
MUST-M1: Evaluate every open position against all 8 IBD sell rules on every run.
          Never skip a position. Never skip a rule.

MUST-M2: Apply the 7-8% stop loss rule as absolute. If a stock is down 7-8% 
          from its buy point, recommend SELL regardless of other factors.
          This is the #1 capital preservation rule in IBD methodology.

MUST-M3: Detect climax top signals: stock surges 25-50% in 1-3 weeks on 
          heavy volume after an extended advance. Recommend SELL or TRIM.

MUST-M4: Track the 50-day moving average relationship. If a stock closes 
          below its 50-day MA on heavy volume, flag as WARNING. 
          If it fails to recover within 2-3 days, escalate to SELL.

MUST-M5: Monitor RS Rating trajectory. If RS Rating drops below 70 
          or falls more than 15 points from its peak since purchase, 
          flag as DETERIORATING.

MUST-M6: Apply profit-taking rules: when a stock reaches 20-25% profit, 
          recommend TRIM (sell partial) unless the stock reached 20% in 
          under 3 weeks (IBD "8-week hold" rule — exceptional strength).

MUST-M7: Adjust sell discipline based on market regime:
          - CONFIRMED_UPTREND: Standard rules apply
          - UPTREND_UNDER_PRESSURE: Tighten stops to 3-5%
          - CORRECTION: Tighten stops to 2-3%, take profits aggressively

MUST-M8: Track the number of distribution days in the position's sector.
          If sector shows 4+ distribution days in 3 weeks, flag all 
          positions in that sector as ELEVATED_RISK.
```

### 3.2 Safety Rules — Lines Never Crossed

```
MUST_NOT-S1: MUST NOT allow any single position loss to exceed 10% 
             without a CRITICAL alert. The 7-8% rule should catch this 
             first, but this is the hard backstop.

MUST_NOT-S2: MUST NOT recommend holding through earnings if the position 
             has less than 15% profit cushion. Defer to the Earnings Risk 
             Analyst if available, otherwise flag as EARNINGS_RISK.

MUST_NOT-S3: MUST NOT override the 7-8% stop loss rule for any reason.
             Not for a strong RS Rating. Not for sector strength. 
             Not for "it's near support." The stop loss is absolute.

MUST_NOT-S4: MUST NOT recommend adding to a losing position (averaging down).
             IBD methodology explicitly prohibits this.

MUST_NOT-S5: MUST NOT provide specific tax advice about whether to sell 
             for tax-loss harvesting. Flag tax implications but recommend 
             consulting a tax professional.

MUST_NOT-S6: MUST NOT issue a SELL signal based on a single intraday move.
             All analysis must use closing prices and end-of-day volume.
```

### 3.3 Quality Rules — Standards We Maintain

```
MUST-Q1: Provide an evidence chain for every recommendation. 
         Format: [DATA POINT] → [RULE TRIGGERED] → [RECOMMENDATION]
         Example: "NVDA closed below 50-day MA on 2x average volume 
         → MUST-M4 triggered → SELL within 2-3 days if no recovery"

MUST-Q2: Classify every signal with urgency:
         - CRITICAL: Act today. Stop loss breached, climax top, 
           catastrophic breakdown.
         - WARNING: Act within 2-3 days. Below 50-day MA, RS deteriorating, 
           sector distribution.
         - WATCH: Monitor closely. Approaching stop, volume drying up, 
           nearing profit target.
         - HEALTHY: No action needed. Position performing within parameters.

MUST-Q3: For every SELL or TRIM recommendation, specify:
         - Which account holds the position (Schwab, E*TRADE, HSA)
         - Exact shares or percentage to sell
         - Whether it's an offensive sell (locking profit) or 
           defensive sell (cutting loss)

MUST-Q4: Say "DATA UNAVAILABLE" rather than estimating if any data point 
         is missing. Never hallucinate a price, volume, or rating.

MUST-Q5: Include a portfolio impact summary: if all recommended actions 
         are taken, what is the resulting cash position, sector exposure, 
         and top holding concentration?

MUST-Q6: Provide a daily portfolio health score (1-10) based on:
         - Number of positions in HEALTHY status
         - Average profit cushion across portfolio
         - Sector concentration risk
         - Market regime alignment
```

### 3.4 Priority Resolution

When rules conflict, apply this priority order:

```
PRIORITY 1: Safety rules always override methodology rules.
            If MUST_NOT-S3 (absolute stop loss) conflicts with anything, 
            the stop loss wins.

PRIORITY 2: Market regime tightening (MUST-M7) overrides standard thresholds.
            In CORRECTION, a stock down 3% triggers action, even though 
            MUST-M2 says 7-8% normally.

PRIORITY 3: The 8-week hold rule (MUST-M6 exception) overrides normal 
            profit-taking, but NOT stop loss rules. A stock up 25% in 
            2 weeks is held for 8 weeks FROM BREAKOUT — unless it 
            triggers a stop loss.

PRIORITY 4: Urgency determines order of recommendations. 
            CRITICAL signals appear first, then WARNING, then WATCH.
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
    buy_point: float = Field(description="IBD pivot/buy point at entry")


class ExitStrategyInput(BaseModel):
    positions: list[Position] = Field(description="All open positions across accounts")
    market_regime: MarketRegime = Field(description="Current market regime from Regime Detector")
    market_exposure_model: int = Field(
        description="IBD market exposure model 0-100",
        ge=0, le=100
    )
    analysis_date: date = Field(description="Date of analysis (use closing prices)")
```

### 4.2 Output Schema

```python
class Urgency(str, Enum):
    CRITICAL = "CRITICAL"     # Act today
    WARNING = "WARNING"       # Act within 2-3 days
    WATCH = "WATCH"           # Monitor closely
    HEALTHY = "HEALTHY"       # No action needed


class ActionType(str, Enum):
    SELL_ALL = "SELL_ALL"           # Exit entire position
    TRIM = "TRIM"                   # Sell partial position
    TIGHTEN_STOP = "TIGHTEN_STOP"  # Raise mental stop price
    HOLD = "HOLD"                   # No action, position healthy
    HOLD_8_WEEK = "HOLD_8_WEEK"    # Exceptional strength, hold 8 weeks


class SellType(str, Enum):
    OFFENSIVE = "OFFENSIVE"   # Taking profit
    DEFENSIVE = "DEFENSIVE"   # Cutting loss
    NOT_APPLICABLE = "N/A"    # For HOLD actions


class SellRule(str, Enum):
    STOP_LOSS_7_8 = "STOP_LOSS_7_8"
    CLIMAX_TOP = "CLIMAX_TOP"
    BELOW_50_DAY_MA = "BELOW_50_DAY_MA"
    RS_DETERIORATION = "RS_DETERIORATION"
    PROFIT_TARGET_20_25 = "PROFIT_TARGET_20_25"
    SECTOR_DISTRIBUTION = "SECTOR_DISTRIBUTION"
    REGIME_TIGHTENED_STOP = "REGIME_TIGHTENED_STOP"
    EARNINGS_RISK = "EARNINGS_RISK"
    NONE = "NONE"


class EvidenceLink(BaseModel):
    data_point: str = Field(description="Specific data: 'NVDA closed at $138.50, down 8.2% from buy point $150.82'")
    rule_triggered: SellRule = Field(description="Which MUST rule this evidence triggers")
    rule_id: str = Field(description="Rule reference: 'MUST-M2'")


class PositionSignal(BaseModel):
    ticker: str
    account: AccountName
    current_price: float
    gain_loss_pct: float = Field(description="Current gain/loss from buy price as percentage")
    days_held: int = Field(description="Calendar days since buy date")
    urgency: Urgency
    action: ActionType
    sell_type: SellType
    sell_quantity: Optional[int] = Field(
        default=None,
        description="Number of shares to sell. None for HOLD actions."
    )
    stop_price: float = Field(description="Current mental stop price for this position")
    rules_triggered: list[SellRule] = Field(description="All sell rules triggered for this position")
    evidence: list[EvidenceLink] = Field(
        description="Evidence chain: data → rule → recommendation. Minimum 1 per triggered rule."
    )
    reasoning: str = Field(
        description="1-3 sentence explanation of the recommendation in plain language"
    )
    what_would_change: str = Field(
        description="What signal would upgrade or downgrade this assessment"
    )


class PortfolioImpact(BaseModel):
    current_cash_pct: float = Field(description="Current cash as % of total portfolio")
    projected_cash_pct: float = Field(description="Cash % if all recommendations executed")
    current_top_holding_pct: float = Field(description="Largest position as % of portfolio")
    projected_top_holding_pct: float = Field(description="Largest position after actions")
    positions_healthy: int
    positions_warning: int
    positions_critical: int
    sector_concentration_risk: str = Field(description="HIGH/MEDIUM/LOW — any sector > 30%?")


class ExitStrategyOutput(BaseModel):
    analysis_date: date
    market_regime: MarketRegime
    market_exposure_model: int
    portfolio_health_score: int = Field(
        description="1-10 score. 10=all healthy, 1=multiple critical signals",
        ge=1, le=10
    )
    signals: list[PositionSignal] = Field(
        description="One signal per position, ordered by urgency (CRITICAL first)"
    )
    portfolio_impact: PortfolioImpact
    executive_summary: str = Field(
        description="2-3 sentence summary: how many actions needed, "
                    "total capital at risk, most urgent action"
    )
```

---

## 5. Tools

### 5.1 Tool: Get Position Details

```python
@tool("Get Position Details")
def get_position_details(ticker: str, account: str) -> dict:
    """Get current market data for a held position including current price, 
    today's volume vs 50-day average volume, and current gain/loss from 
    cost basis. Use this to check if a position has hit a stop loss 
    or profit target."""
    # Returns: current_price, volume_today, avg_volume_50d, volume_ratio,
    #          gain_loss_pct, gain_loss_dollars, days_held
```

**Schema:**
```json
{
    "name": "get_position_details",
    "description": "Get current market data for a held position including current price, today's volume vs 50-day average volume, and current gain/loss from cost basis. Use this to check if a position has hit a stop loss or profit target.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol, e.g. 'NVDA'"
            },
            "account": {
                "type": "string",
                "enum": ["SCHWAB", "ETRADE", "HSA_BANK"],
                "description": "Account holding the position"
            }
        },
        "required": ["ticker", "account"]
    }
}
```

### 5.2 Tool: Get Technical Indicators

```python
@tool("Get Technical Indicators")
def get_technical_indicators(ticker: str) -> dict:
    """Get key technical indicators for exit analysis: 50-day and 200-day 
    moving averages, distance from each MA, RS Rating (current and 4 weeks ago), 
    and number of distribution days in the stock's sector. Use this to check 
    if a stock is breaking below key support levels or showing RS deterioration."""
    # Returns: ma_50, ma_200, pct_from_50ma, pct_from_200ma,
    #          rs_rating_current, rs_rating_4w_ago, rs_rating_peak_since_buy,
    #          sector_name, sector_rank, sector_distribution_days_3w
```

**Schema:**
```json
{
    "name": "get_technical_indicators",
    "description": "Get key technical indicators for exit analysis: 50-day and 200-day moving averages, distance from each MA, RS Rating (current and 4 weeks ago), and number of distribution days in the stock's sector. Use this to check if a stock is breaking below key support levels or showing RS deterioration.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol"
            }
        },
        "required": ["ticker"]
    }
}
```

### 5.3 Tool: Get Price History

```python
@tool("Get Price History")
def get_price_history(ticker: str, days: int = 20) -> dict:
    """Get recent daily price and volume history for a stock. 
    Returns the last N trading days of OHLCV data. Use this to detect 
    climax top patterns (25-50% surge in 1-3 weeks), volume dry-up, 
    or gap-downs that signal institutional selling."""
    # Returns: list of {date, open, high, low, close, volume}
```

**Schema:**
```json
{
    "name": "get_price_history",
    "description": "Get recent daily price and volume history for a stock. Returns the last N trading days of OHLCV data. Use this to detect climax top patterns (25-50% surge in 1-3 weeks), volume dry-up, or gap-downs that signal institutional selling.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol"
            },
            "days": {
                "type": "integer",
                "description": "Number of trading days of history. Default 20. Use 60 for climax top detection.",
                "default": 20
            }
        },
        "required": ["ticker"]
    }
}
```

### 5.4 Tool: Get Earnings Calendar

```python
@tool("Get Earnings Calendar")
def get_earnings_calendar(ticker: str) -> dict:
    """Check when a stock's next earnings report is scheduled.
    Use this to flag positions where earnings are within 3 weeks 
    and the profit cushion may be insufficient to absorb an 
    earnings gap-down."""
    # Returns: next_earnings_date, days_until_earnings, 
    #          avg_earnings_move_pct (historical average gap)
```

**Schema:**
```json
{
    "name": "get_earnings_calendar",
    "description": "Check when a stock's next earnings report is scheduled. Use this to flag positions where earnings are within 3 weeks and the profit cushion may be insufficient to absorb an earnings gap-down.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol"
            }
        },
        "required": ["ticker"]
    }
}
```

### 5.5 Tool: Get Market Health

```python
@tool("Get Market Health")
def get_market_health() -> dict:
    """Get overall market health indicators: distribution day count on 
    S&P 500 and Nasdaq, percentage of stocks above 200-day MA, 
    new highs vs new lows, and the IBD market exposure model reading.
    Use this to determine if sell rules should be tightened based 
    on market regime."""
    # Returns: sp500_dist_days, nasdaq_dist_days, pct_above_200ma,
    #          new_highs, new_lows, exposure_model, regime_signal
```

**Schema:**
```json
{
    "name": "get_market_health",
    "description": "Get overall market health indicators: distribution day count on S&P 500 and Nasdaq, percentage of stocks above 200-day MA, new highs vs new lows, and the IBD market exposure model reading. Use this to determine if sell rules should be tightened based on market regime.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}
```

---

## 6. Tool Selection Strategy

The agent should follow this decision tree — **not as a rigid script, but as a general approach** that it can adapt based on what it finds at each step:

```
For each position:
│
├─ Step 1: get_position_details(ticker, account)
│   └─ Is loss ≥ 7%? → CRITICAL SELL. No further analysis needed. (MUST-M2)
│   └─ Is gain ≥ 20%? → Continue but flag for profit-taking analysis (MUST-M6)
│
├─ Step 2: get_technical_indicators(ticker)
│   └─ Below 50-day MA? → Check volume (need heavy volume for SELL signal)
│   └─ RS Rating dropped 15+ points? → Flag DETERIORATING (MUST-M5)
│   └─ Sector has 4+ distribution days? → Flag ELEVATED_RISK (MUST-M8)
│
├─ Step 3: get_price_history(ticker, 60) — only if gain > 20% or pattern needed
│   └─ 25-50% surge in 1-3 weeks? → CLIMAX TOP signal (MUST-M3)
│   └─ Volume drying up at highs? → Potential top forming
│
├─ Step 4: get_earnings_calendar(ticker)
│   └─ Earnings within 3 weeks AND cushion < 15%? → EARNINGS_RISK (MUST_NOT-S2)
│
└─ Step 5: get_market_health() — once per run, applies to all positions
    └─ Adjust all stop thresholds per MUST-M7

Goal-driven behavior:
- If Step 1 triggers a stop loss, STOP. Don't waste tokens on Steps 2-4.
- If a position is up 5% in an uptrend with strong RS, minimal analysis needed.
- Spend more time (more tool calls) on ambiguous situations.
```

This is where goal-driven reasoning matters. A scripted pipeline calls all 5 tools for every position. The Exit Strategist reasons about whether it needs more data based on what it's already found.

---

## 7. CrewAI Implementation

### 7.1 Agent Definition

```python
from crewai import Agent

exit_strategist = Agent(
    role="Exit Strategist",
    goal=(
        "Protect capital and lock in profits by evaluating every open position "
        "against IBD sell methodology. Produce specific, evidence-backed exit "
        "recommendations with urgency levels. The 7-8% stop loss is absolute "
        "and non-negotiable. No position should ever be allowed to erode "
        "beyond acceptable risk thresholds."
    ),
    backstory=(
        "You are a disciplined exit strategist with deep expertise in IBD sell "
        "methodology. You know that the sell decision is harder than the buy — "
        "emotions fight discipline at every turn. You are the dispassionate voice "
        "that says 'the setup is broken, get out' when every instinct says "
        "'it will come back.' You follow William O'Neil's sell rules precisely. "
        "Small losses are the cost of business. Cutting them quickly preserves "
        "capital for the next winner. You never let a reasonable gain turn into "
        "a loss. You never fall in love with a stock. Your principle: protect "
        "the downside, and the upside takes care of itself."
    ),
    tools=[
        get_position_details,
        get_technical_indicators,
        get_price_history,
        get_earnings_calendar,
        get_market_health,
    ],
    verbose=True,
    allow_delegation=False,
    max_iter=25,
    max_rpm=10,
)
```

### 7.2 Task Definition

```python
from crewai import Task

exit_strategy_task = Task(
    description=(
        "Evaluate all open positions for exit signals.\n\n"
        "Positions:\n{positions}\n\n"
        "Market Regime: {market_regime}\n"
        "Exposure Model: {market_exposure_model}%\n"
        "Analysis Date: {analysis_date}\n\n"
        "For EACH position:\n"
        "1. Check gain/loss against stop loss thresholds "
        "   (7-8% standard, tighter in corrections)\n"
        "2. Evaluate technical health: 50-day MA, RS Rating trend, "
        "   sector distribution days\n"
        "3. Check for climax top signals if stock has gained significantly\n"
        "4. Check earnings proximity and profit cushion adequacy\n"
        "5. Assign urgency: CRITICAL, WARNING, WATCH, or HEALTHY\n"
        "6. Provide evidence chain for every recommendation\n\n"
        "ABSOLUTE RULE: The 7-8% stop loss is non-negotiable. "
        "If a stock is down 7-8% from its buy point, recommend SELL "
        "regardless of any other factor.\n\n"
        "MARKET REGIME ADJUSTMENT:\n"
        "- CONFIRMED_UPTREND: Standard 7-8% stops\n"
        "- UPTREND_UNDER_PRESSURE: Tighten to 3-5%\n"
        "- CORRECTION: Tighten to 2-3%, take profits aggressively\n\n"
        "Order all signals by urgency (CRITICAL first). Include portfolio "
        "impact summary and overall health score."
    ),
    expected_output=(
        "Structured exit analysis with urgency-ranked signals for every "
        "position, evidence chains, portfolio impact summary, and health score."
    ),
    agent=exit_strategist,
    output_pydantic=ExitStrategyOutput,
)
```

### 7.3 Crew Integration

```python
# The Exit Strategist runs AFTER the Portfolio Manager has current positions
# and AFTER the Regime Detector has determined market conditions.

# Option A: Daily standalone run (lightweight, just exit signals)
daily_exit_crew = Crew(
    agents=[exit_strategist],
    tasks=[exit_strategy_task],
    verbose=True,
)

# Option B: Integrated into full IBD_Crew (weekly full analysis)
ibd_crew = Crew(
    agents=[
        regime_detector,       # Determines market regime first
        research_agent,
        analyst_agent,
        sector_strategist,
        watchlist_monitor,
        exit_strategist,       # Evaluates current positions
        portfolio_manager,     # Synthesizes entries + exits
        risk_officer,          # Validates exit recommendations
        educator,
    ],
    tasks=[
        regime_task,
        research_task,
        analysis_task,
        sector_task,
        watchlist_task,
        exit_strategy_task,    # Runs after analysis, before PM decision
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

## 8. Golden Dataset

### 8.1 Test Case Design Philosophy

Each test case targets a specific MUST/MUST NOT rule. Boundary cases test the edges where rules overlap or conflict.

### 8.2 Test Cases

```python
golden_exit_tests = [

    # ── STOP LOSS TESTS (MUST-M2, MUST_NOT-S3) ──────────────────────

    {
        "id": "EXIT-001",
        "name": "clear_stop_loss_breach",
        "description": "Stock down 8.5% from buy point. Must trigger absolute sell.",
        "rules_tested": ["MUST-M2", "MUST_NOT-S3"],
        "input": {
            "positions": [{
                "ticker": "CRWD",
                "account": "SCHWAB",
                "shares": 50,
                "buy_date": "2026-01-15",
                "buy_price": 380.00,
                "buy_point": 378.50,
            }],
            "market_regime": "CONFIRMED_UPTREND",
            "market_exposure_model": 80,
        },
        "mock_tool_returns": {
            "get_position_details": {"current_price": 347.30, "gain_loss_pct": -8.6},
        },
        "expected": {
            "urgency": "CRITICAL",
            "action": "SELL_ALL",
            "sell_type": "DEFENSIVE",
            "rules_triggered": ["STOP_LOSS_7_8"],
            "must_not_continue_analysis": True,
        },
        "notes": "Agent should stop analysis after Step 1. No need for technical indicators.",
    },

    {
        "id": "EXIT-002",
        "name": "stop_loss_boundary_6_9_pct",
        "description": "Stock down 6.9% — just under the 7% threshold. Should be WARNING, not SELL.",
        "rules_tested": ["MUST-M2"],
        "input": {
            "positions": [{
                "ticker": "ANET",
                "account": "ETRADE",
                "shares": 30,
                "buy_date": "2026-02-01",
                "buy_price": 420.00,
                "buy_point": 418.00,
            }],
            "market_regime": "CONFIRMED_UPTREND",
            "market_exposure_model": 75,
        },
        "mock_tool_returns": {
            "get_position_details": {"current_price": 391.02, "gain_loss_pct": -6.9},
        },
        "expected": {
            "urgency": "WATCH",
            "action": "TIGHTEN_STOP",
            "action_must_not_be": "SELL_ALL",
            "reasoning_must_mention": "approaching stop loss",
        },
        "notes": "Boundary test. 6.9% is close but has not triggered. Agent should tighten, not sell.",
    },

    {
        "id": "EXIT-003",
        "name": "stop_loss_never_overridden_by_strong_rs",
        "description": "Stock down 8% but has RS Rating of 95 and top sector. Must still sell.",
        "rules_tested": ["MUST_NOT-S3"],
        "input": {
            "positions": [{
                "ticker": "SMCI",
                "account": "SCHWAB",
                "shares": 25,
                "buy_date": "2026-01-20",
                "buy_price": 900.00,
                "buy_point": 895.00,
            }],
            "market_regime": "CONFIRMED_UPTREND",
            "market_exposure_model": 80,
        },
        "mock_tool_returns": {
            "get_position_details": {"current_price": 828.00, "gain_loss_pct": -8.0},
            "get_technical_indicators": {"rs_rating_current": 95, "sector_rank": 2},
        },
        "expected": {
            "urgency": "CRITICAL",
            "action": "SELL_ALL",
            "reasoning_must_mention": "stop loss is absolute",
        },
        "notes": "Tests that strong fundamentals do NOT override the stop loss. This is the #1 rule.",
    },

    # ── REGIME-ADJUSTED STOPS (MUST-M7) ──────────────────────────────

    {
        "id": "EXIT-004",
        "name": "correction_regime_tightened_stop",
        "description": "Stock down only 3% but market is in CORRECTION. Tightened rules should trigger.",
        "rules_tested": ["MUST-M7"],
        "input": {
            "positions": [{
                "ticker": "META",
                "account": "SCHWAB",
                "shares": 20,
                "buy_date": "2026-02-10",
                "buy_price": 650.00,
                "buy_point": 648.00,
            }],
            "market_regime": "CORRECTION",
            "market_exposure_model": 25,
        },
        "mock_tool_returns": {
            "get_position_details": {"current_price": 630.50, "gain_loss_pct": -3.0},
        },
        "expected": {
            "urgency": "CRITICAL",
            "action": "SELL_ALL",
            "sell_type": "DEFENSIVE",
            "rules_triggered": ["REGIME_TIGHTENED_STOP"],
        },
        "notes": "3% loss is fine in uptrend but triggers sell in correction (2-3% threshold).",
    },

    {
        "id": "EXIT-005",
        "name": "uptrend_under_pressure_tightened",
        "description": "Stock down 4% in UPTREND_UNDER_PRESSURE. Should trigger tightened 3-5% rule.",
        "rules_tested": ["MUST-M7"],
        "input": {
            "positions": [{
                "ticker": "NOW",
                "account": "ETRADE",
                "shares": 15,
                "buy_date": "2026-02-05",
                "buy_price": 1050.00,
                "buy_point": 1045.00,
            }],
            "market_regime": "UPTREND_UNDER_PRESSURE",
            "market_exposure_model": 50,
        },
        "mock_tool_returns": {
            "get_position_details": {"current_price": 1008.00, "gain_loss_pct": -4.0},
        },
        "expected": {
            "urgency": "WARNING",
            "action": "SELL_ALL",
            "rules_triggered": ["REGIME_TIGHTENED_STOP"],
        },
    },

    # ── CLIMAX TOP (MUST-M3) ─────────────────────────────────────────

    {
        "id": "EXIT-006",
        "name": "climax_top_signal",
        "description": "Stock surged 35% in 2 weeks on massive volume. Classic climax top.",
        "rules_tested": ["MUST-M3"],
        "input": {
            "positions": [{
                "ticker": "NVDA",
                "account": "SCHWAB",
                "shares": 100,
                "buy_date": "2025-11-01",
                "buy_price": 120.00,
                "buy_point": 119.50,
            }],
            "market_regime": "CONFIRMED_UPTREND",
            "market_exposure_model": 80,
        },
        "mock_tool_returns": {
            "get_position_details": {"current_price": 180.00, "gain_loss_pct": 50.0},
            "get_price_history": {
                "data": [
                    {"date": "2026-02-07", "close": 133.00, "volume": 40_000_000},
                    {"date": "2026-02-14", "close": 155.00, "volume": 80_000_000},
                    {"date": "2026-02-21", "close": 180.00, "volume": 120_000_000},
                ]
            },
        },
        "expected": {
            "urgency": "CRITICAL",
            "action": "SELL_ALL",
            "sell_type": "OFFENSIVE",
            "rules_triggered": ["CLIMAX_TOP"],
            "reasoning_must_mention": "surge",
        },
    },

    # ── 50-DAY MA BREAKDOWN (MUST-M4) ───────────────────────────────

    {
        "id": "EXIT-007",
        "name": "below_50_day_heavy_volume",
        "description": "Stock closed below 50-day MA on 2x average volume.",
        "rules_tested": ["MUST-M4"],
        "input": {
            "positions": [{
                "ticker": "PANW",
                "account": "ETRADE",
                "shares": 40,
                "buy_date": "2025-12-15",
                "buy_price": 390.00,
                "buy_point": 388.00,
            }],
            "market_regime": "CONFIRMED_UPTREND",
            "market_exposure_model": 75,
        },
        "mock_tool_returns": {
            "get_position_details": {"current_price": 375.00, "gain_loss_pct": -3.8, "volume_ratio": 2.1},
            "get_technical_indicators": {"ma_50": 380.00, "pct_from_50ma": -1.3},
        },
        "expected": {
            "urgency": "WARNING",
            "action": "SELL_ALL",
            "rules_triggered": ["BELOW_50_DAY_MA"],
            "what_would_change_must_mention": "recover above 50-day",
        },
    },

    # ── RS DETERIORATION (MUST-M5) ───────────────────────────────────

    {
        "id": "EXIT-008",
        "name": "rs_rating_drop_15_points",
        "description": "RS Rating was 92 at purchase, now 75. Dropped 17 points.",
        "rules_tested": ["MUST-M5"],
        "input": {
            "positions": [{
                "ticker": "DDOG",
                "account": "HSA_BANK",
                "shares": 60,
                "buy_date": "2025-12-01",
                "buy_price": 150.00,
                "buy_point": 148.50,
            }],
            "market_regime": "CONFIRMED_UPTREND",
            "market_exposure_model": 75,
        },
        "mock_tool_returns": {
            "get_position_details": {"current_price": 148.00, "gain_loss_pct": -1.3},
            "get_technical_indicators": {
                "rs_rating_current": 75,
                "rs_rating_peak_since_buy": 92,
            },
        },
        "expected": {
            "urgency": "WARNING",
            "rules_triggered": ["RS_DETERIORATION"],
            "evidence_must_mention": "RS dropped from 92 to 75",
        },
    },

    # ── PROFIT TAKING (MUST-M6) ──────────────────────────────────────

    {
        "id": "EXIT-009",
        "name": "profit_target_standard",
        "description": "Stock up 22% over 8 weeks. Standard profit-taking applies.",
        "rules_tested": ["MUST-M6"],
        "input": {
            "positions": [{
                "ticker": "AXON",
                "account": "SCHWAB",
                "shares": 35,
                "buy_date": "2025-12-20",
                "buy_price": 550.00,
                "buy_point": 548.00,
            }],
            "market_regime": "CONFIRMED_UPTREND",
            "market_exposure_model": 80,
        },
        "mock_tool_returns": {
            "get_position_details": {
                "current_price": 671.00,
                "gain_loss_pct": 22.0,
                "days_held": 63,
            },
        },
        "expected": {
            "urgency": "WATCH",
            "action": "TRIM",
            "sell_type": "OFFENSIVE",
            "rules_triggered": ["PROFIT_TARGET_20_25"],
        },
    },

    {
        "id": "EXIT-010",
        "name": "profit_target_exception_8_week_hold",
        "description": "Stock up 22% in just 12 days. Exceptional strength — 8-week hold rule.",
        "rules_tested": ["MUST-M6", "PRIORITY-3"],
        "input": {
            "positions": [{
                "ticker": "PLTR",
                "account": "SCHWAB",
                "shares": 200,
                "buy_date": "2026-02-09",
                "buy_price": 100.00,
                "buy_point": 99.50,
            }],
            "market_regime": "CONFIRMED_UPTREND",
            "market_exposure_model": 80,
        },
        "mock_tool_returns": {
            "get_position_details": {
                "current_price": 122.00,
                "gain_loss_pct": 22.0,
                "days_held": 12,
            },
        },
        "expected": {
            "urgency": "HEALTHY",
            "action": "HOLD_8_WEEK",
            "action_must_not_be": "TRIM",
            "reasoning_must_mention": "8-week hold",
        },
        "notes": "20%+ in under 3 weeks triggers the exception. Hold 8 weeks from breakout.",
    },

    # ── EARNINGS RISK (MUST_NOT-S2) ──────────────────────────────────

    {
        "id": "EXIT-011",
        "name": "earnings_risk_insufficient_cushion",
        "description": "Earnings in 10 days, only 8% profit cushion. Must flag.",
        "rules_tested": ["MUST_NOT-S2"],
        "input": {
            "positions": [{
                "ticker": "CRM",
                "account": "ETRADE",
                "shares": 25,
                "buy_date": "2026-01-15",
                "buy_price": 320.00,
                "buy_point": 318.00,
            }],
            "market_regime": "CONFIRMED_UPTREND",
            "market_exposure_model": 80,
        },
        "mock_tool_returns": {
            "get_position_details": {"current_price": 345.60, "gain_loss_pct": 8.0},
            "get_earnings_calendar": {"days_until_earnings": 10, "avg_earnings_move_pct": 12.5},
        },
        "expected": {
            "urgency": "WARNING",
            "rules_triggered": ["EARNINGS_RISK"],
            "reasoning_must_mention": "cushion",
            "reasoning_must_mention_2": "earnings",
        },
    },

    {
        "id": "EXIT-012",
        "name": "earnings_safe_large_cushion",
        "description": "Earnings in 8 days, but 25% profit cushion. Sufficient to hold.",
        "rules_tested": ["MUST_NOT-S2"],
        "input": {
            "positions": [{
                "ticker": "AAPL",
                "account": "SCHWAB",
                "shares": 100,
                "buy_date": "2025-10-01",
                "buy_price": 220.00,
                "buy_point": 218.00,
            }],
            "market_regime": "CONFIRMED_UPTREND",
            "market_exposure_model": 80,
        },
        "mock_tool_returns": {
            "get_position_details": {"current_price": 275.00, "gain_loss_pct": 25.0},
            "get_earnings_calendar": {"days_until_earnings": 8, "avg_earnings_move_pct": 5.0},
        },
        "expected": {
            "urgency": "HEALTHY",
            "rules_triggered_must_not_include": "EARNINGS_RISK",
        },
    },

    # ── AVERAGING DOWN PROHIBITION (MUST_NOT-S4) ────────────────────

    {
        "id": "EXIT-013",
        "name": "never_recommend_averaging_down",
        "description": "Stock down 5% but 'near support'. Must not suggest adding shares.",
        "rules_tested": ["MUST_NOT-S4"],
        "input": {
            "positions": [{
                "ticker": "GOOGL",
                "account": "SCHWAB",
                "shares": 50,
                "buy_date": "2026-02-01",
                "buy_price": 200.00,
                "buy_point": 198.00,
            }],
            "market_regime": "CONFIRMED_UPTREND",
            "market_exposure_model": 75,
        },
        "mock_tool_returns": {
            "get_position_details": {"current_price": 190.00, "gain_loss_pct": -5.0},
            "get_technical_indicators": {"ma_50": 189.00, "pct_from_50ma": 0.5},
        },
        "expected": {
            "action_must_not_be": "ADD",
            "reasoning_must_not_mention": "add shares",
            "reasoning_must_not_mention_2": "average down",
        },
        "notes": "Even though it's at the 50-day MA ('support'), must never suggest buying more.",
    },

    # ── PORTFOLIO-LEVEL TEST ─────────────────────────────────────────

    {
        "id": "EXIT-014",
        "name": "multi_position_mixed_signals",
        "description": "5 positions: 1 stop loss, 1 climax top, 1 RS deteriorating, 2 healthy.",
        "rules_tested": ["MUST-M1", "MUST-Q2", "MUST-Q5"],
        "input": {
            "positions": [
                {"ticker": "CRWD", "account": "SCHWAB", "shares": 50,
                 "buy_date": "2026-01-15", "buy_price": 380.00, "buy_point": 378.50},
                {"ticker": "NVDA", "account": "SCHWAB", "shares": 100,
                 "buy_date": "2025-11-01", "buy_price": 120.00, "buy_point": 119.50},
                {"ticker": "DDOG", "account": "HSA_BANK", "shares": 60,
                 "buy_date": "2025-12-01", "buy_price": 150.00, "buy_point": 148.50},
                {"ticker": "AXON", "account": "SCHWAB", "shares": 35,
                 "buy_date": "2025-12-20", "buy_price": 550.00, "buy_point": 548.00},
                {"ticker": "AAPL", "account": "ETRADE", "shares": 100,
                 "buy_date": "2025-10-01", "buy_price": 220.00, "buy_point": 218.00},
            ],
            "market_regime": "CONFIRMED_UPTREND",
            "market_exposure_model": 80,
        },
        "expected": {
            "total_signals": 5,
            "critical_count": 2,
            "warning_count": 1,
            "healthy_count": 2,
            "ordered_by_urgency": True,
            "portfolio_impact_present": True,
            "executive_summary_present": True,
            "health_score_between": [3, 6],
        },
        "notes": "Tests MUST-M1 (every position evaluated) and MUST-Q2 (ordered by urgency).",
    },

    # ── DATA UNAVAILABLE (MUST-Q4) ───────────────────────────────────

    {
        "id": "EXIT-015",
        "name": "missing_data_graceful_handling",
        "description": "Tool returns incomplete data for one position. Agent must not hallucinate.",
        "rules_tested": ["MUST-Q4"],
        "input": {
            "positions": [{
                "ticker": "NEWIPO",
                "account": "SCHWAB",
                "shares": 100,
                "buy_date": "2026-02-15",
                "buy_price": 50.00,
                "buy_point": 49.50,
            }],
            "market_regime": "CONFIRMED_UPTREND",
            "market_exposure_model": 80,
        },
        "mock_tool_returns": {
            "get_position_details": {"current_price": 48.00, "gain_loss_pct": -4.0},
            "get_technical_indicators": {"rs_rating_current": "DATA_UNAVAILABLE", "ma_50": "DATA_UNAVAILABLE"},
        },
        "expected": {
            "must_contain_text": "DATA UNAVAILABLE",
            "must_not_hallucinate_rs_rating": True,
            "must_not_hallucinate_ma": True,
        },
        "notes": "New IPO may not have 50-day MA or stable RS Rating. Agent must acknowledge gaps.",
    },
]
```

### 8.3 Evaluation Criteria

```python
def evaluate_exit_signal(test_case, actual_output):
    """Evaluate a single test case against actual agent output."""
    results = {}
    expected = test_case["expected"]
    signal = actual_output.signals[0]  # For single-position tests

    # Exact match checks
    if "urgency" in expected:
        results["urgency_correct"] = signal.urgency.value == expected["urgency"]

    if "action" in expected:
        results["action_correct"] = signal.action.value == expected["action"]

    if "sell_type" in expected:
        results["sell_type_correct"] = signal.sell_type.value == expected["sell_type"]

    # Must-not checks
    if "action_must_not_be" in expected:
        results["action_prohibited"] = signal.action.value != expected["action_must_not_be"]

    # Rules triggered
    if "rules_triggered" in expected:
        actual_rules = [r.value for r in signal.rules_triggered]
        results["rules_correct"] = all(r in actual_rules for r in expected["rules_triggered"])

    # Evidence quality
    if "reasoning_must_mention" in expected:
        results["reasoning_quality"] = expected["reasoning_must_mention"].lower() in signal.reasoning.lower()

    # Portfolio-level checks
    if "total_signals" in expected:
        results["all_positions_evaluated"] = len(actual_output.signals) == expected["total_signals"]

    if "ordered_by_urgency" in expected:
        urgency_order = {"CRITICAL": 0, "WARNING": 1, "WATCH": 2, "HEALTHY": 3}
        urgencies = [urgency_order[s.urgency.value] for s in actual_output.signals]
        results["correctly_ordered"] = urgencies == sorted(urgencies)

    return results
```

---

## 9. Error Handling

```
SCENARIO                          BEHAVIOR
─────────────────────────────────────────────────────────────────
Tool returns error                Report "DATA UNAVAILABLE" for affected fields.
                                  Continue analysis with available data.
                                  If position_details fails, flag position as
                                  CRITICAL with note: "Unable to verify price —
                                  manual check required."

Tool returns stale data           Check data timestamp. If > 1 trading day old,
                                  flag as WARNING: "Analysis based on stale data
                                  from {date}. Verify current prices manually."

Position not found in account     Report as CRITICAL: "Position {ticker} in
                                  {account} not found. May have been sold or
                                  transferred. Manual verification required."

Pydantic validation fails         CrewAI retries with feedback. After 3 failures,
                                  return partial analysis with error note.

Market regime input missing       Default to UPTREND_UNDER_PRESSURE (conservative).
                                  Flag: "Market regime unavailable — using
                                  conservative assumptions."

Conflicting data between tools    Report both values. Flag: "Conflicting data —
                                  get_position_details shows $X, get_price_history
                                  shows $Y. Using more conservative (worse) value."
```

---

## 10. Performance Metrics

Track these metrics over time to evaluate the agent's effectiveness:

```
METRIC                            TARGET          HOW MEASURED
──────────────────────────────────────────────────────────────────
Stop loss compliance              100%            Zero positions ever exceed 10% loss
Urgency accuracy                  > 85%           Manual review: was urgency appropriate?
False CRITICAL rate               < 10%           CRITICALs that didn't need immediate action
Missed sell signals               < 5%            Stocks that dropped 15%+ without prior WARNING
Evidence chain completeness       100%            Every recommendation has ≥ 1 evidence link
Golden dataset pass rate          > 85%           Automated evaluation on 15 test cases
Average tool calls per position   2.5             Goal-driven efficiency (not calling all 5 every time)
Latency per portfolio             < 60 seconds    For a 10-position portfolio
Token cost per run                < $0.50         For a 10-position portfolio on Sonnet
```

---

## 11. Future Enhancements

```
VERSION   ENHANCEMENT                                  VALUE
──────────────────────────────────────────────────────────────────
v1.1      Integration with Tax-Aware Rebalancer        Cross-account exit optimization
v1.2      Historical backtesting mode                  Run against past market data 
                                                       to validate sell rules
v1.3      Notification system                          Push CRITICAL alerts via SMS/email
v1.4      Post-Trade Reviewer integration              Feed exit outcomes back for 
                                                       self-improving feedback loop
v2.0      Options-aware exits                          Factor in covered call obligations
                                                       and protective puts
```

---

## Appendix A: IBD Sell Rules Reference

The 8 core IBD sell rules this agent implements:

1. **7-8% Stop Loss** — Cut every loss at 7-8% below the buy point. No exceptions. (O'Neil Rule #1)
2. **Climax Top** — Sell when stock surges 25-50% in 1-3 weeks on massive volume after extended advance
3. **50-Day MA Break** — Sell when stock closes below 50-day MA on heavy volume and fails to recover
4. **RS Rating Deterioration** — Sell when RS Rating drops below 70 or falls 15+ points from peak
5. **20-25% Profit Target** — Take at least partial profits when gain reaches 20-25% (exception: 8-week hold)
6. **Sector Distribution** — Sell when stock's sector shows 4+ distribution days in 3 weeks
7. **Earnings Risk** — Reduce position if earnings are within 2 weeks and profit cushion is insufficient
8. **Market Regime** — Tighten all stops during corrections and uptrend-under-pressure conditions

---

## Appendix B: Example Agent Reasoning Trace

```
[Exit Strategist] Analyzing position: CRWD in SCHWAB (50 shares)

Thought: Let me check the current price and gain/loss first.

Action: get_position_details(ticker="CRWD", account="SCHWAB")
Result: current_price=347.30, gain_loss_pct=-8.6%, volume_ratio=1.8

Thought: This position is down 8.6% from the buy price of $380.00. 
         This exceeds the 7-8% stop loss threshold (MUST-M2). 
         This is an ABSOLUTE rule (MUST_NOT-S3) — I cannot override it 
         regardless of RS Rating, sector strength, or any other factor.
         
         I don't need to check technical indicators, price history, 
         or earnings. The stop loss is definitive.

Action: None needed. I have sufficient evidence.

Signal: CRWD
  Urgency: CRITICAL
  Action: SELL_ALL (50 shares)
  Sell Type: DEFENSIVE
  Stop Price: $350.82 (7% below buy point — already breached)
  Rules Triggered: STOP_LOSS_7_8
  Evidence: 
    - "CRWD closed at $347.30, down 8.6% from buy price $380.00" 
      → MUST-M2 (7-8% stop loss) → SELL_ALL
  Reasoning: "CRWD has breached the 7-8% stop loss rule. Current loss 
    is 8.6%. The stop loss is absolute and non-negotiable regardless of 
    other factors. Recommend immediate exit to preserve capital."
  What Would Change: "Nothing. Once the stop loss is breached, the 
    decision is final. The position should have been exited at -7%."
```