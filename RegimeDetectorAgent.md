# Regime Detector Agent — Specification

**Version:** 1.0
**Author:** Sree
**Project:** IBD_Crew Multi-Agent Investment System
**Framework:** CrewAI
**Date:** February 2026
**Status:** Draft

---

## 1. Problem Statement

Every agent in the IBD_Crew depends on knowing the current market regime. The Portfolio Manager needs it to size positions. The Exit Strategist needs it to tighten or loosen stops. The Watchlist Monitor needs it to decide whether to signal entries at all. The Risk Officer needs it to set portfolio-level exposure.

Today, **market regime is a manual input** — Sree reads IBD's Big Picture column, checks distribution day counts, looks at leading stock behavior, and makes a judgment call. This creates three problems:

1. **Subjectivity** — the same data gets interpreted differently depending on mood, recent experience, or anchoring bias. After a big win, corrections look like buying opportunities. After a loss, uptrends look fragile.

2. **Latency** — manual regime assessment happens once a week (at best). Markets can shift from uptrend to correction in 2-3 days. A weekly check misses intra-week regime changes that should trigger defensive action.

3. **Inconsistency** — without a systematic framework, the same market conditions might be classified as UPTREND_UNDER_PRESSURE one week and CONFIRMED_UPTREND the next, depending on which signals get the most attention.

The Regime Detector Agent replaces this manual process with a systematic, evidence-based assessment that runs daily, weighs multiple signals with explicit priority, and produces a classification that every downstream agent consumes.

---

## 2. Agent Identity

### Role
Regime Detector

### Goal
Classify the current market regime and recommend a portfolio exposure level by synthesizing multiple market health indicators. Provide clear reasoning about which signals drove the classification and what specific conditions would change it — so the Portfolio Manager and other agents can act with confidence.

### Backstory
You are a market regime analyst who has studied every major market cycle since 1953. You know that the single biggest determinant of whether a stock purchase succeeds is the direction of the general market — not the individual stock's quality. William O'Neil proved that three out of four stocks follow the market's direction. Your job is to answer one question definitively: **is the market environment favorable for buying, or should we be defensive?**

You are neither a permanent bull nor a permanent bear. You follow the evidence. You know that the hardest regime calls happen at transitions — the first few days of a correction feel like a pullback, and the first follow-through day after a correction feels like a trap. You handle these transitions by weighing multiple independent signals rather than relying on any single indicator. You resist the temptation to call bottoms or tops prematurely. You wait for confirmation.

Your principle: **when in doubt, be defensive. The cost of missing the first few days of a rally is small. The cost of being fully invested at the start of a correction is enormous.**

---

## 3. Behavioral Constraints

### 3.1 Methodology Rules — How We Classify Regimes

```
MUST-M1: Classify the market into exactly ONE of five regimes:
         - CONFIRMED_UPTREND: Follow-through day confirmed, distribution 
           days low, leaders healthy. Full offense.
         - UPTREND_UNDER_PRESSURE: Uptrend intact but distribution days 
           accumulating or breadth narrowing. Reduce new buys, tighten stops.
         - RALLY_ATTEMPT: Market bouncing after correction but NO 
           follow-through day yet. No new buys until confirmed.
         - CORRECTION: Distribution day cluster, failed rally attempts, 
           or follow-through day failure. Full defense.
         - FOLLOW_THROUGH_DAY: Special transitional state — a rally 
           attempt just produced a valid FTD. Signals potential new uptrend. 
           Valid for 1-3 days then must transition to CONFIRMED_UPTREND 
           or back to RALLY_ATTEMPT if it fails.

MUST-M2: Count distribution days using IBD methodology:
         - A distribution day = index down ≥ 0.2% on higher volume 
           than the previous session
         - Count separately for S&P 500 and Nasdaq
         - Distribution days expire after 25 trading sessions
         - A strong up day (index up ≥ 1% on higher volume) removes 
           the oldest distribution day ("stalling" count resets too)
         - Track as a rolling window, not a cumulative total

MUST-M3: Validate Follow-Through Days with all four conditions:
         - Must occur on Day 4 or later of a rally attempt 
           (Day 1 = first up day after making a new low)
         - Index must gain ≥ 1.25% (prefer ≥ 1.7% for stronger signal)
         - Volume must be higher than the previous session
         - Must occur on S&P 500 OR Nasdaq (one is sufficient)
         Classify the FTD quality:
         - STRONG: ≥ 1.7% gain, ≥ 1.5x average volume, Day 4-7
         - MODERATE: ≥ 1.25% gain, above prior day volume, Day 4-10
         - WEAK: Barely qualifies, or occurs after Day 10

MUST-M4: Assess market breadth using at minimum:
         - Percentage of S&P 500 stocks above 200-day MA
         - Percentage of S&P 500 stocks above 50-day MA
         - New 52-week highs vs new 52-week lows (NYSE + Nasdaq)
         - Advance/decline line direction (rising, flat, declining)
         Breadth divergence (index at highs but breadth deteriorating) 
         is a WARNING signal even during CONFIRMED_UPTREND.

MUST-M5: Assess leading stock health:
         - Count IBD RS ≥ 90 stocks that are above their 50-day MA
         - Count IBD RS ≥ 90 stocks making new highs vs new lows
         - If > 60% of leaders are above 50-day MA: HEALTHY
         - If 40-60%: MIXED
         - If < 40%: DETERIORATING
         Leading stock health DETERIORATING during a CONFIRMED_UPTREND 
         should trigger transition to UPTREND_UNDER_PRESSURE.

MUST-M6: Assess sector rotation for defensiveness:
         - If 3 or more of the top 5 sectors are defensive 
           (Utilities, Consumer Staples, Healthcare, REITs, 
           Bonds/Treasuries), flag as DEFENSIVE_ROTATION
         - Defensive rotation during an uptrend is an early 
           warning of regime change
         - Growth sectors leading (Tech, Consumer Discretionary, 
           Industrials) confirms uptrend health

MUST-M7: Set exposure model recommendation based on regime:
         - CONFIRMED_UPTREND: 80-100% exposure
         - UPTREND_UNDER_PRESSURE: 40-60% exposure
         - FOLLOW_THROUGH_DAY: 20-40% exposure (test the water)
         - RALLY_ATTEMPT: 0-20% exposure (no new buys)
         - CORRECTION: 0% exposure (fully defensive)
         Within each range, adjust based on signal strength 
         and breadth/leader health.

MUST-M8: Provide transition conditions — specific, measurable 
         statements about what would change the classification:
         - "Regime would upgrade to CONFIRMED_UPTREND if: Nasdaq 
           produces a follow-through day with ≥ 1.7% gain on 
           above-average volume"
         - "Regime would downgrade to CORRECTION if: S&P 500 
           accumulates 6+ distribution days in 5 weeks, or the 
           Nasdaq undercuts its most recent correction low"
         These must be falsifiable — someone could check tomorrow 
         whether the condition was met.
```

### 3.2 Safety Rules — Lines Never Crossed

```
MUST_NOT-S1: MUST NOT classify as CONFIRMED_UPTREND without a valid 
             Follow-Through Day after the most recent correction. 
             A rally off lows without a FTD is still RALLY_ATTEMPT, 
             no matter how strong it looks.

MUST_NOT-S2: MUST NOT flip regime classification based on a single 
             day's action. One bad day in an uptrend is not a correction. 
             One good day in a correction is not a rally. Require 
             at minimum 2 confirming signals across different indicators 
             to change regime.

MUST_NOT-S3: MUST NOT classify as CONFIRMED_UPTREND if distribution 
             day count is 5 or more on EITHER the S&P 500 or Nasdaq 
             in the trailing 25 sessions. 5+ distribution days 
             automatically caps the regime at UPTREND_UNDER_PRESSURE.

MUST_NOT-S4: MUST NOT recommend exposure above 60% during 
             UPTREND_UNDER_PRESSURE, above 40% during RALLY_ATTEMPT, 
             or above 0% during CORRECTION. These are hard ceilings.

MUST_NOT-S5: MUST NOT predict when a correction will end or when 
             the next uptrend will begin. The agent classifies 
             the CURRENT regime based on CURRENT evidence. 
             No forecasting. No "I think the bottom is in."

MUST_NOT-S6: MUST NOT use individual stock performance as a primary 
             regime indicator. A single stock (even AAPL or NVDA) 
             hitting a new high does not confirm an uptrend. 
             Use aggregate leading stock health (MUST-M5) instead.
```

### 3.3 Quality Rules — Standards We Maintain

```
MUST-Q1: Provide a structured evidence summary for the classification.
         For each major indicator, report:
         - Current value
         - Direction (improving, stable, deteriorating)
         - Signal (bullish, neutral, bearish)
         Example: "S&P 500 distribution days: 3 in 25 sessions 
         (stable, neutral)"

MUST-Q2: Assign a confidence level to the regime classification:
         - HIGH: 4+ indicators agree, no contradictions
         - MEDIUM: 3 indicators agree, 1-2 ambiguous
         - LOW: Indicators are mixed or contradictory
         LOW confidence should always be accompanied by 
         "what to watch" guidance.

MUST-Q3: Report all indicators even when they agree. Don't skip 
         breadth data because distribution days already look bad. 
         The downstream agents need the full picture. Omitting 
         a healthy signal because the regime is CORRECTION deprives 
         other agents of useful context.

MUST-Q4: Say "DATA UNAVAILABLE" for any indicator that cannot be 
         retrieved. Never fabricate distribution day counts, breadth 
         percentages, or leading stock numbers. If data is missing 
         for 2+ indicators, classify with LOW confidence and 
         recommend manual verification.

MUST-Q5: Include a regime history — what the classification was 
         in the previous run and whether it changed. If the regime 
         changed, explain WHAT evidence drove the transition. 
         Format: "REGIME CHANGE: CONFIRMED_UPTREND → 
         UPTREND_UNDER_PRESSURE. Trigger: S&P 500 accumulated 
         5th distribution day on Feb 18."

MUST-Q6: Provide a daily market health score (1-10):
         10: Strong uptrend, low distribution, broad participation
         7-9: Healthy uptrend, minor concerns
         4-6: Transitional or under pressure
         1-3: Correction or severe deterioration
```

### 3.4 Priority Resolution

```
PRIORITY 1: Distribution day count thresholds are hard limits. 
            5+ distribution days = cannot be CONFIRMED_UPTREND, 
            regardless of how strong breadth or leaders look.
            (MUST_NOT-S3 overrides MUST-M4 and MUST-M5)

PRIORITY 2: Follow-Through Day requirement is absolute. 
            No FTD = no CONFIRMED_UPTREND. Period.
            (MUST_NOT-S1 overrides everything)

PRIORITY 3: When indicators conflict (e.g., low distribution but 
            deteriorating breadth), default to the more conservative 
            classification. "When in doubt, be defensive."

PRIORITY 4: The 2-signal confirmation rule (MUST_NOT-S2) applies 
            to regime UPGRADES only. Regime DOWNGRADES can trigger 
            on 1 signal if that signal is definitive (e.g., 
            distribution day count hitting 6+, or the index 
            undercutting the correction low).
```

---

## 4. Pydantic Contracts

### 4.1 Input Schema

```python
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from datetime import date


class PreviousRegime(str, Enum):
    CONFIRMED_UPTREND = "CONFIRMED_UPTREND"
    UPTREND_UNDER_PRESSURE = "UPTREND_UNDER_PRESSURE"
    CORRECTION = "CORRECTION"
    RALLY_ATTEMPT = "RALLY_ATTEMPT"
    FOLLOW_THROUGH_DAY = "FOLLOW_THROUGH_DAY"
    UNKNOWN = "UNKNOWN"  # First run, no prior classification


class RegimeDetectorInput(BaseModel):
    analysis_date: date = Field(description="Date of analysis (use end-of-day data)")
    previous_regime: PreviousRegime = Field(
        default=PreviousRegime.UNKNOWN,
        description="Regime classification from the previous run"
    )
    previous_exposure_recommendation: Optional[int] = Field(
        default=None,
        description="Exposure % recommended in the previous run",
        ge=0, le=100
    )
    correction_low_date: Optional[date] = Field(
        default=None,
        description="Date of the most recent correction low (for FTD counting)"
    )
    correction_low_sp500: Optional[float] = Field(
        default=None,
        description="S&P 500 closing price at the most recent correction low"
    )
    correction_low_nasdaq: Optional[float] = Field(
        default=None,
        description="Nasdaq closing price at the most recent correction low"
    )
```

### 4.2 Output Schema

```python
class MarketRegime(str, Enum):
    CONFIRMED_UPTREND = "CONFIRMED_UPTREND"
    UPTREND_UNDER_PRESSURE = "UPTREND_UNDER_PRESSURE"
    CORRECTION = "CORRECTION"
    RALLY_ATTEMPT = "RALLY_ATTEMPT"
    FOLLOW_THROUGH_DAY = "FOLLOW_THROUGH_DAY"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SignalDirection(str, Enum):
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"


class Trend(str, Enum):
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DETERIORATING = "DETERIORATING"


class LeaderHealth(str, Enum):
    HEALTHY = "HEALTHY"         # > 60% of RS ≥ 90 stocks above 50-day MA
    MIXED = "MIXED"             # 40-60%
    DETERIORATING = "DETERIORATING"  # < 40%


class SectorCharacter(str, Enum):
    GROWTH_LEADING = "GROWTH_LEADING"
    BALANCED = "BALANCED"
    DEFENSIVE_ROTATION = "DEFENSIVE_ROTATION"


class FTDQuality(str, Enum):
    STRONG = "STRONG"           # ≥ 1.7% gain, ≥ 1.5x avg volume, Day 4-7
    MODERATE = "MODERATE"       # ≥ 1.25% gain, above prior day volume, Day 4-10
    WEAK = "WEAK"               # Barely qualifies or late (after Day 10)
    NONE = "NONE"               # No FTD detected


class DistributionDayAssessment(BaseModel):
    sp500_count: int = Field(description="Distribution days on S&P 500 in trailing 25 sessions")
    nasdaq_count: int = Field(description="Distribution days on Nasdaq in trailing 25 sessions")
    sp500_direction: Trend = Field(description="Are distribution days accumulating or expiring?")
    nasdaq_direction: Trend = Field(description="Are distribution days accumulating or expiring?")
    signal: SignalDirection
    detail: str = Field(description="Brief explanation: 'S&P at 3, Nasdaq at 4. Nasdaq approaching threshold.'")


class BreadthAssessment(BaseModel):
    pct_above_200ma: float = Field(description="% of S&P 500 stocks above 200-day MA")
    pct_above_50ma: float = Field(description="% of S&P 500 stocks above 50-day MA")
    new_highs: int = Field(description="New 52-week highs (NYSE + Nasdaq)")
    new_lows: int = Field(description="New 52-week lows (NYSE + Nasdaq)")
    advance_decline_direction: Trend
    breadth_divergence: bool = Field(
        description="True if index at/near highs but breadth deteriorating"
    )
    signal: SignalDirection
    detail: str


class LeaderAssessment(BaseModel):
    rs90_above_50ma_pct: float = Field(description="% of RS ≥ 90 stocks above their 50-day MA")
    rs90_new_highs: int = Field(description="RS ≥ 90 stocks making new 52-week highs")
    rs90_new_lows: int = Field(description="RS ≥ 90 stocks making new 52-week lows")
    health: LeaderHealth
    signal: SignalDirection
    detail: str


class SectorAssessment(BaseModel):
    top_5_sectors: list[str] = Field(description="Current top 5 sectors by IBD ranking")
    defensive_in_top_5: int = Field(description="Count of defensive sectors in top 5")
    character: SectorCharacter
    signal: SignalDirection
    detail: str


class FollowThroughDayAssessment(BaseModel):
    detected: bool = Field(description="Was a follow-through day detected in this analysis?")
    quality: FTDQuality
    index: Optional[str] = Field(default=None, description="Which index: 'S&P 500' or 'Nasdaq'")
    gain_pct: Optional[float] = Field(default=None, description="Index gain on the FTD")
    volume_vs_prior: Optional[float] = Field(default=None, description="Volume ratio vs prior day")
    rally_day_number: Optional[int] = Field(default=None, description="Which day of the rally attempt")
    detail: str


class TransitionCondition(BaseModel):
    direction: str = Field(description="'UPGRADE' or 'DOWNGRADE'")
    target_regime: MarketRegime
    condition: str = Field(
        description="Specific, measurable condition: "
        "'S&P 500 produces FTD with ≥ 1.25% gain on above-avg volume'"
    )
    likelihood: str = Field(description="'LIKELY', 'POSSIBLE', or 'UNLIKELY' based on current trend")


class RegimeChange(BaseModel):
    changed: bool = Field(description="Did the regime change from the previous run?")
    previous: Optional[PreviousRegime] = None
    current: Optional[MarketRegime] = None
    trigger: Optional[str] = Field(
        default=None,
        description="What specific evidence triggered the change"
    )


class RegimeDetectorOutput(BaseModel):
    analysis_date: date
    regime: MarketRegime
    confidence: Confidence
    market_health_score: int = Field(
        description="1-10 market health. 10=strong uptrend, 1=severe correction",
        ge=1, le=10
    )
    exposure_recommendation: int = Field(
        description="Recommended portfolio exposure as percentage",
        ge=0, le=100
    )
    
    # Evidence — one assessment per indicator
    distribution_days: DistributionDayAssessment
    breadth: BreadthAssessment
    leaders: LeaderAssessment
    sectors: SectorAssessment
    follow_through_day: FollowThroughDayAssessment
    
    # Signal summary
    bullish_signals: int = Field(description="Count of indicators showing BULLISH")
    neutral_signals: int = Field(description="Count of indicators showing NEUTRAL")
    bearish_signals: int = Field(description="Count of indicators showing BEARISH")
    
    # Transitions
    regime_change: RegimeChange
    transition_conditions: list[TransitionCondition] = Field(
        description="What would cause the regime to change — both upgrade and downgrade"
    )
    
    executive_summary: str = Field(
        description="3-4 sentence summary: regime, key evidence, exposure, "
                    "and what to watch. Written for the Portfolio Manager."
    )
```

---

## 5. Tools

### 5.1 Tool: Get Distribution Days

```python
@tool("Get Distribution Days")
def get_distribution_days() -> dict:
    """Get the current distribution day count for S&P 500 and Nasdaq 
    over the trailing 25 trading sessions. Returns each distribution day's 
    date, index decline %, and volume. Also returns any 'power up' days 
    (index up ≥ 1% on higher volume) that remove distribution days. 
    Use this as the PRIMARY regime indicator — 5+ distribution days 
    on either index signals an uptrend under serious pressure."""
    # Returns: sp500_dist_count, nasdaq_dist_count,
    #          sp500_dist_days: [{date, decline_pct, volume}],
    #          nasdaq_dist_days: [{date, decline_pct, volume}],
    #          power_up_days: [{date, index, gain_pct}],
    #          sp500_close, nasdaq_close, sp500_volume, nasdaq_volume
```

**Schema:**
```json
{
    "name": "get_distribution_days",
    "description": "Get the current distribution day count for S&P 500 and Nasdaq over the trailing 25 trading sessions. Returns each distribution day's date, index decline %, and volume. Also returns any 'power up' days (index up ≥ 1% on higher volume) that remove distribution days. Use this as the PRIMARY regime indicator — 5+ distribution days on either index signals an uptrend under serious pressure.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}
```

### 5.2 Tool: Get Market Breadth

```python
@tool("Get Market Breadth")
def get_market_breadth() -> dict:
    """Get market breadth indicators: percentage of S&P 500 stocks above 
    their 200-day and 50-day moving averages, new 52-week highs vs lows 
    on NYSE and Nasdaq combined, and advance/decline line direction over 
    the past 10 trading days. Use this to detect breadth divergence — 
    when the index is at highs but fewer stocks are participating, 
    it's an early warning of regime deterioration."""
    # Returns: pct_above_200ma, pct_above_50ma, 
    #          new_highs_today, new_lows_today,
    #          new_highs_10d_avg, new_lows_10d_avg,
    #          adv_decline_line_10d_direction: "rising"|"flat"|"declining",
    #          breadth_vs_10d_ago: "improving"|"stable"|"deteriorating"
```

**Schema:**
```json
{
    "name": "get_market_breadth",
    "description": "Get market breadth indicators: percentage of S&P 500 stocks above their 200-day and 50-day moving averages, new 52-week highs vs lows on NYSE and Nasdaq combined, and advance/decline line direction over the past 10 trading days. Use this to detect breadth divergence — when the index is at highs but fewer stocks are participating, it's an early warning of regime deterioration.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}
```

### 5.3 Tool: Get Leading Stocks Health

```python
@tool("Get Leading Stocks Health")
def get_leading_stocks_health() -> dict:
    """Get the health of market-leading stocks: count and percentage of 
    stocks with RS Rating ≥ 90 that are above their 50-day moving average, 
    how many RS ≥ 90 stocks made new 52-week highs vs new lows today, 
    and the 10-day trend. Use this to assess whether institutional money 
    is still flowing into growth leaders or rotating out. Leader 
    deterioration often precedes index deterioration by 1-2 weeks."""
    # Returns: total_rs90_stocks, rs90_above_50ma_count, rs90_above_50ma_pct,
    #          rs90_new_highs, rs90_new_lows,
    #          rs90_health_10d_ago_pct, health_trend: "improving"|"stable"|"deteriorating"
```

**Schema:**
```json
{
    "name": "get_leading_stocks_health",
    "description": "Get the health of market-leading stocks: count and percentage of stocks with RS Rating ≥ 90 that are above their 50-day moving average, how many RS ≥ 90 stocks made new 52-week highs vs new lows today, and the 10-day trend. Use this to assess whether institutional money is still flowing into growth leaders or rotating out. Leader deterioration often precedes index deterioration by 1-2 weeks.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}
```

### 5.4 Tool: Get Sector Rankings

```python
@tool("Get Sector Rankings")
def get_sector_rankings() -> dict:
    """Get the current IBD sector rankings (1-33) with the top 10 and 
    bottom 10 sectors. Each sector includes its rank, 3-month performance, 
    and whether it's classified as 'growth' or 'defensive'. Use this to 
    detect defensive rotation — when utilities, staples, and healthcare 
    dominate the top rankings, institutions are getting cautious even 
    if the index hasn't broken down yet."""
    # Returns: top_10: [{rank, sector_name, performance_3m, type: "growth"|"defensive"}],
    #          bottom_10: [{rank, sector_name, performance_3m, type}],
    #          defensive_in_top_5: count,
    #          growth_in_top_5: count
```

**Schema:**
```json
{
    "name": "get_sector_rankings",
    "description": "Get the current IBD sector rankings (1-33) with the top 10 and bottom 10 sectors. Each sector includes its rank, 3-month performance, and whether it's classified as 'growth' or 'defensive'. Use this to detect defensive rotation — when utilities, staples, and healthcare dominate the top rankings, institutions are getting cautious even if the index hasn't broken down yet.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}
```

### 5.5 Tool: Get Index Price History

```python
@tool("Get Index Price History")
def get_index_price_history(index: str, days: int = 25) -> dict:
    """Get daily OHLCV data for a major index. Use this to detect 
    follow-through days by checking if any day in the rally attempt 
    meets the FTD criteria (≥ 1.25% gain on higher volume than prior day, 
    on Day 4 or later of the rally). Also use to identify the correction 
    low and count rally attempt days."""
    # Returns: data: [{date, open, high, low, close, volume, 
    #          change_pct, volume_vs_prior}]
```

**Schema:**
```json
{
    "name": "get_index_price_history",
    "description": "Get daily OHLCV data for a major index (S&P 500 or Nasdaq). Use this to detect follow-through days by checking if any day in the rally attempt meets the FTD criteria (≥ 1.25% gain on higher volume than prior day, on Day 4 or later of the rally). Also use to identify the correction low and count rally attempt days.",
    "parameters": {
        "type": "object",
        "properties": {
            "index": {
                "type": "string",
                "enum": ["SP500", "NASDAQ"],
                "description": "Which index to retrieve"
            },
            "days": {
                "type": "integer",
                "description": "Number of trading days of history. Default 25.",
                "default": 25
            }
        },
        "required": ["index"]
    }
}
```

---

## 6. Tool Selection Strategy

Unlike the Exit Strategist (which can short-circuit on stop loss), the Regime Detector **must check all indicators** on every run. The market regime is a synthesis — no single indicator is sufficient.

```
Every run:
│
├─ Step 1: get_distribution_days()
│   └─ Establishes the hard constraints (5+ = cannot be CONFIRMED_UPTREND)
│   └─ Also provides current index close and volume for FTD check
│
├─ Step 2: get_market_breadth()
│   └─ Detects divergence (index at highs, breadth weak)
│   └─ Confirms or challenges the distribution day signal
│
├─ Step 3: get_leading_stocks_health()
│   └─ Leading indicator — leaders break down before the index
│   └─ Healthy leaders can offset moderate distribution
│
├─ Step 4: get_sector_rankings()
│   └─ Defensive rotation is an early warning
│   └─ Growth sectors leading confirms uptrend health
│
├─ Step 5: get_index_price_history("SP500", 25) + ("NASDAQ", 25)
│   └─ ONLY needed if current regime is CORRECTION or RALLY_ATTEMPT
│   └─ Checks for follow-through day conditions
│   └─ In CONFIRMED_UPTREND, skip this step (save tokens)
│
└─ Step 6: Synthesize all signals → Classify regime
    └─ Count bullish/neutral/bearish signals
    └─ Apply priority rules (distribution count is hard limit)
    └─ Apply 2-signal confirmation for upgrades (MUST_NOT-S2)
    └─ Set exposure recommendation within regime band
    └─ Generate transition conditions

Goal-driven efficiency:
- Steps 1-4 always run (4 tool calls)
- Step 5 only runs during CORRECTION or RALLY_ATTEMPT (2 more calls)
- In a stable CONFIRMED_UPTREND: 4 tool calls per run
- During regime transitions: 6 tool calls per run
```

---

## 7. Classification Logic

The agent reasons through this decision tree, but with judgment — not as rigid if/else code:

```
START
│
├─ Distribution days (S&P or Nasdaq) ≥ 6?
│   └─ YES → CORRECTION (hard rule, regardless of other signals)
│
├─ Distribution days ≥ 5 on either index?
│   └─ YES → UPTREND_UNDER_PRESSURE at best (MUST_NOT-S3)
│         └─ If leaders DETERIORATING too → likely CORRECTION
│
├─ Previous regime was CORRECTION or RALLY_ATTEMPT?
│   └─ YES → Check for Follow-Through Day
│         ├─ Valid FTD detected? → FOLLOW_THROUGH_DAY
│         │   └─ FTD quality STRONG + leaders HEALTHY → 
│         │      may upgrade to CONFIRMED_UPTREND on next run
│         ├─ Rally attempt in progress (Day 1-3)? → RALLY_ATTEMPT
│         └─ No rally attempt? → CORRECTION continues
│
├─ Previous regime was CONFIRMED_UPTREND?
│   └─ Check for deterioration:
│       ├─ Distribution 4+ AND breadth divergence? 
│       │   → UPTREND_UNDER_PRESSURE
│       ├─ Distribution 4+ AND leaders DETERIORATING? 
│       │   → UPTREND_UNDER_PRESSURE
│       ├─ Distribution < 4 AND leaders HEALTHY AND breadth healthy? 
│       │   → CONFIRMED_UPTREND (confirmed)
│       └─ Distribution < 4 BUT defensive rotation? 
│          → CONFIRMED_UPTREND with early WARNING flag
│
└─ Previous regime was UPTREND_UNDER_PRESSURE?
    └─ Check for recovery or further deterioration:
        ├─ Distribution dropping + leaders recovering + breadth improving?
        │   → Upgrade to CONFIRMED_UPTREND (needs 2+ improving signals)
        ├─ Distribution stable + mixed signals?
        │   → Stay UPTREND_UNDER_PRESSURE
        └─ Distribution increasing OR leaders DETERIORATING further?
           → Downgrade to CORRECTION
```

---

## 8. CrewAI Implementation

### 8.1 Agent Definition

```python
from crewai import Agent

regime_detector = Agent(
    role="Regime Detector",
    goal=(
        "Classify the current market regime into exactly one of five states: "
        "CONFIRMED_UPTREND, UPTREND_UNDER_PRESSURE, RALLY_ATTEMPT, "
        "FOLLOW_THROUGH_DAY, or CORRECTION. Synthesize distribution day counts, "
        "market breadth, leading stock health, sector rotation, and follow-through "
        "day analysis to produce an evidence-backed classification with exposure "
        "recommendation. Default to the more conservative classification when "
        "signals conflict. Never classify as CONFIRMED_UPTREND without a valid "
        "Follow-Through Day."
    ),
    backstory=(
        "You are a market regime analyst who has studied every major cycle since 1953. "
        "You know that three out of four stocks follow the general market direction. "
        "Your job is to answer one question: is the environment favorable for buying, "
        "or should we be defensive? You are neither permanently bullish nor bearish — "
        "you follow the evidence. You know the hardest calls happen at transitions: "
        "the first days of a correction feel like pullbacks, and the first "
        "follow-through day after a correction feels like a trap. You handle these "
        "by weighing multiple independent signals. You resist calling bottoms or tops "
        "prematurely. When in doubt, you are defensive. The cost of missing the first "
        "few days of a rally is small. The cost of being fully invested at the start "
        "of a correction is enormous."
    ),
    tools=[
        get_distribution_days,
        get_market_breadth,
        get_leading_stocks_health,
        get_sector_rankings,
        get_index_price_history,
    ],
    verbose=True,
    allow_delegation=False,
    max_iter=15,
    max_rpm=10,
)
```

### 8.2 Task Definition

```python
from crewai import Task

regime_task = Task(
    description=(
        "Classify the current market regime as of {analysis_date}.\n\n"
        "Previous regime: {previous_regime}\n"
        "Previous exposure recommendation: {previous_exposure_recommendation}%\n"
        "Correction low date: {correction_low_date}\n"
        "Correction low S&P 500: {correction_low_sp500}\n"
        "Correction low Nasdaq: {correction_low_nasdaq}\n\n"
        "Analyze ALL of the following:\n"
        "1. Distribution day count on S&P 500 and Nasdaq (25-day trailing)\n"
        "2. Market breadth: % above 200-day/50-day MA, new highs vs lows, A/D line\n"
        "3. Leading stock health: RS ≥ 90 stocks above 50-day MA\n"
        "4. Sector rotation: growth vs defensive leadership\n"
        "5. Follow-through day analysis (only if in CORRECTION or RALLY_ATTEMPT)\n\n"
        "HARD RULES:\n"
        "- 5+ distribution days on either index = CANNOT be CONFIRMED_UPTREND\n"
        "- No Follow-Through Day after correction = CANNOT be CONFIRMED_UPTREND\n"
        "- Do NOT flip regime on a single day's action\n"
        "- When signals conflict, default to more conservative classification\n\n"
        "Provide: regime, confidence, exposure recommendation, all indicator "
        "assessments, transition conditions, and executive summary."
    ),
    expected_output=(
        "Complete regime classification with evidence from all indicators, "
        "exposure recommendation, and specific transition conditions."
    ),
    agent=regime_detector,
    output_pydantic=RegimeDetectorOutput,
)
```

### 8.3 Crew Integration

```python
# The Regime Detector MUST run FIRST — every other agent depends on its output.

ibd_crew = Crew(
    agents=[
        regime_detector,       # ← Runs first, always
        research_agent,
        analyst_agent,
        sector_strategist,
        watchlist_monitor,     # Consumes regime → decides whether to signal entries
        exit_strategist,       # Consumes regime → adjusts stop thresholds
        portfolio_manager,     # Consumes regime → sizes positions and exposure
        risk_officer,          # Consumes regime → sets portfolio risk limits
        educator,
    ],
    tasks=[
        regime_task,           # ← First task, output feeds all downstream
        research_task,
        analysis_task,
        sector_task,
        watchlist_task,
        exit_strategy_task,
        portfolio_task,
        risk_task,
        education_task,
    ],
    process=Process.hierarchical,
    manager_agent=portfolio_manager,
    verbose=True,
)

# Daily lightweight run (just regime check):
daily_regime_crew = Crew(
    agents=[regime_detector],
    tasks=[regime_task],
    verbose=True,
)
```

---

## 9. Golden Dataset

### 9.1 Test Cases

```python
golden_regime_tests = [

    # ── CONFIRMED UPTREND ────────────────────────────────────────────

    {
        "id": "REG-001",
        "name": "clear_confirmed_uptrend",
        "description": "Low distribution, healthy breadth, strong leaders, growth sectors leading.",
        "rules_tested": ["MUST-M1", "MUST-M7"],
        "mock_tool_returns": {
            "get_distribution_days": {
                "sp500_dist_count": 2, "nasdaq_dist_count": 1,
            },
            "get_market_breadth": {
                "pct_above_200ma": 72, "pct_above_50ma": 65,
                "new_highs": 280, "new_lows": 35,
                "adv_decline_direction": "rising",
            },
            "get_leading_stocks_health": {
                "rs90_above_50ma_pct": 75, "rs90_new_highs": 22, "rs90_new_lows": 3,
            },
            "get_sector_rankings": {
                "top_5": ["Technology", "Consumer Discretionary", "Industrials", 
                          "Healthcare", "Financial"],
                "defensive_in_top_5": 0,
            },
        },
        "expected": {
            "regime": "CONFIRMED_UPTREND",
            "confidence": "HIGH",
            "exposure_min": 80,
            "exposure_max": 100,
            "market_health_score_min": 8,
            "bullish_signals_min": 4,
        },
    },

    # ── DISTRIBUTION DAY HARD LIMIT ──────────────────────────────────

    {
        "id": "REG-002",
        "name": "five_distribution_days_caps_regime",
        "description": "5 distribution days on S&P despite healthy breadth and leaders. Cannot be CONFIRMED_UPTREND.",
        "rules_tested": ["MUST_NOT-S3"],
        "mock_tool_returns": {
            "get_distribution_days": {
                "sp500_dist_count": 5, "nasdaq_dist_count": 3,
            },
            "get_market_breadth": {
                "pct_above_200ma": 68, "pct_above_50ma": 60,
                "new_highs": 200, "new_lows": 50,
                "adv_decline_direction": "flat",
            },
            "get_leading_stocks_health": {
                "rs90_above_50ma_pct": 65, "rs90_new_highs": 15, "rs90_new_lows": 5,
            },
            "get_sector_rankings": {
                "top_5": ["Technology", "Financial", "Industrials", 
                          "Consumer Discretionary", "Energy"],
                "defensive_in_top_5": 0,
            },
        },
        "expected": {
            "regime": "UPTREND_UNDER_PRESSURE",
            "regime_must_not_be": "CONFIRMED_UPTREND",
            "exposure_max": 60,
            "reasoning_must_mention": "distribution",
        },
    },

    {
        "id": "REG-003",
        "name": "six_distribution_days_forces_correction",
        "description": "6 distribution days on Nasdaq. Hard rule: CORRECTION.",
        "rules_tested": ["MUST-M2", "MUST_NOT-S3"],
        "mock_tool_returns": {
            "get_distribution_days": {
                "sp500_dist_count": 4, "nasdaq_dist_count": 6,
            },
            "get_market_breadth": {
                "pct_above_200ma": 55, "pct_above_50ma": 42,
                "new_highs": 80, "new_lows": 120,
                "adv_decline_direction": "declining",
            },
            "get_leading_stocks_health": {
                "rs90_above_50ma_pct": 35, "rs90_new_highs": 5, "rs90_new_lows": 18,
            },
            "get_sector_rankings": {
                "top_5": ["Utilities", "Consumer Staples", "Healthcare",
                          "REITs", "Financial"],
                "defensive_in_top_5": 4,
            },
        },
        "expected": {
            "regime": "CORRECTION",
            "confidence": "HIGH",
            "exposure_max": 0,
            "market_health_score_max": 3,
            "bearish_signals_min": 4,
        },
    },

    # ── FOLLOW-THROUGH DAY ───────────────────────────────────────────

    {
        "id": "REG-004",
        "name": "valid_strong_ftd",
        "description": "Day 5 of rally attempt, Nasdaq up 2.1% on 1.6x volume. Strong FTD.",
        "rules_tested": ["MUST-M3"],
        "input": {
            "previous_regime": "RALLY_ATTEMPT",
            "correction_low_date": "2026-02-10",
        },
        "mock_tool_returns": {
            "get_distribution_days": {
                "sp500_dist_count": 3, "nasdaq_dist_count": 2,
            },
            "get_index_price_history": {
                "NASDAQ": [
                    {"date": "2026-02-10", "close": 15000, "change_pct": -1.2},  # Low
                    {"date": "2026-02-11", "close": 15100, "change_pct": 0.67},  # Day 1
                    {"date": "2026-02-12", "close": 15050, "change_pct": -0.33}, # Pause
                    {"date": "2026-02-13", "close": 15120, "change_pct": 0.46},  # Day 3
                    {"date": "2026-02-14", "close": 15200, "change_pct": 0.53},  # Day 4
                    {"date": "2026-02-17", "close": 15519, "change_pct": 2.1,    # Day 5 - FTD!
                     "volume": 6_200_000_000, "volume_vs_prior": 1.6},
                ]
            },
            "get_market_breadth": {
                "pct_above_200ma": 48, "pct_above_50ma": 38,
                "new_highs": 90, "new_lows": 85,
                "adv_decline_direction": "rising",
            },
            "get_leading_stocks_health": {
                "rs90_above_50ma_pct": 45, "rs90_new_highs": 8, "rs90_new_lows": 10,
            },
            "get_sector_rankings": {
                "top_5": ["Technology", "Consumer Discretionary", "Industrials",
                          "Healthcare", "Financial"],
                "defensive_in_top_5": 0,
            },
        },
        "expected": {
            "regime": "FOLLOW_THROUGH_DAY",
            "ftd_quality": "STRONG",
            "exposure_min": 20,
            "exposure_max": 40,
            "transition_must_include_upgrade": "CONFIRMED_UPTREND",
        },
    },

    {
        "id": "REG-005",
        "name": "invalid_ftd_day_3",
        "description": "Rally attempt only on Day 3. Even with 1.5% gain on volume, not a valid FTD.",
        "rules_tested": ["MUST-M3", "MUST_NOT-S1"],
        "input": {
            "previous_regime": "RALLY_ATTEMPT",
            "correction_low_date": "2026-02-17",
        },
        "mock_tool_returns": {
            "get_index_price_history": {
                "SP500": [
                    {"date": "2026-02-17", "close": 5800, "change_pct": -0.8},  # Low
                    {"date": "2026-02-18", "close": 5850, "change_pct": 0.86},  # Day 1
                    {"date": "2026-02-19", "close": 5870, "change_pct": 0.34},  # Day 2
                    {"date": "2026-02-20", "close": 5958, "change_pct": 1.5,    # Day 3
                     "volume": 4_500_000_000, "volume_vs_prior": 1.3},
                ]
            },
        },
        "expected": {
            "regime": "RALLY_ATTEMPT",
            "regime_must_not_be": "FOLLOW_THROUGH_DAY",
            "regime_must_not_be_2": "CONFIRMED_UPTREND",
            "ftd_quality": "NONE",
            "reasoning_must_mention": "Day 4",
        },
    },

    # ── NO SINGLE-DAY FLIPS (MUST_NOT-S2) ────────────────────────────

    {
        "id": "REG-006",
        "name": "single_bad_day_no_regime_change",
        "description": "One big down day in a confirmed uptrend. Should NOT flip to correction.",
        "rules_tested": ["MUST_NOT-S2"],
        "input": {
            "previous_regime": "CONFIRMED_UPTREND",
        },
        "mock_tool_returns": {
            "get_distribution_days": {
                "sp500_dist_count": 3, "nasdaq_dist_count": 3,
            },
            "get_market_breadth": {
                "pct_above_200ma": 62, "pct_above_50ma": 55,
                "new_highs": 120, "new_lows": 90,
                "adv_decline_direction": "flat",
            },
            "get_leading_stocks_health": {
                "rs90_above_50ma_pct": 58, "rs90_new_highs": 10, "rs90_new_lows": 8,
            },
            "get_sector_rankings": {
                "top_5": ["Technology", "Financial", "Healthcare",
                          "Consumer Discretionary", "Industrials"],
                "defensive_in_top_5": 0,
            },
        },
        "expected": {
            "regime": "CONFIRMED_UPTREND",
            "regime_must_not_be": "CORRECTION",
            "regime_change_changed": False,
        },
        "notes": "3 distribution days is elevated but below the 5-day threshold. "
                 "Breadth mixed but not broken. Leaders mixed. Should stay uptrend.",
    },

    # ── BREADTH DIVERGENCE WARNING ───────────────────────────────────

    {
        "id": "REG-007",
        "name": "breadth_divergence_early_warning",
        "description": "Index at highs but breadth deteriorating. Should flag divergence.",
        "rules_tested": ["MUST-M4"],
        "mock_tool_returns": {
            "get_distribution_days": {
                "sp500_dist_count": 2, "nasdaq_dist_count": 2,
            },
            "get_market_breadth": {
                "pct_above_200ma": 52, "pct_above_50ma": 40,
                "new_highs": 100, "new_lows": 110,
                "adv_decline_direction": "declining",
            },
            "get_leading_stocks_health": {
                "rs90_above_50ma_pct": 55, "rs90_new_highs": 12, "rs90_new_lows": 9,
            },
            "get_sector_rankings": {
                "top_5": ["Technology", "Utilities", "Consumer Staples",
                          "Financial", "Healthcare"],
                "defensive_in_top_5": 2,
            },
        },
        "expected": {
            "regime": "CONFIRMED_UPTREND",
            "breadth_divergence": True,
            "confidence": "MEDIUM",
            "reasoning_must_mention": "divergence",
            "transition_must_include_downgrade": "UPTREND_UNDER_PRESSURE",
        },
        "notes": "Distribution is low, so uptrend holds. But breadth divergence "
                 "and 2 defensive sectors in top 5 should lower confidence and "
                 "flag potential downgrade.",
    },

    # ── DEFENSIVE ROTATION ───────────────────────────────────────────

    {
        "id": "REG-008",
        "name": "defensive_rotation_warning",
        "description": "3 defensive sectors in top 5. Classic risk-off signal.",
        "rules_tested": ["MUST-M6"],
        "mock_tool_returns": {
            "get_distribution_days": {
                "sp500_dist_count": 3, "nasdaq_dist_count": 4,
            },
            "get_market_breadth": {
                "pct_above_200ma": 55, "pct_above_50ma": 45,
                "new_highs": 80, "new_lows": 95,
                "adv_decline_direction": "declining",
            },
            "get_leading_stocks_health": {
                "rs90_above_50ma_pct": 42, "rs90_new_highs": 6, "rs90_new_lows": 14,
            },
            "get_sector_rankings": {
                "top_5": ["Utilities", "Consumer Staples", "REITs",
                          "Technology", "Healthcare"],
                "defensive_in_top_5": 3,
            },
        },
        "expected": {
            "regime": "UPTREND_UNDER_PRESSURE",
            "sector_character": "DEFENSIVE_ROTATION",
            "exposure_max": 60,
            "confidence": "HIGH",
        },
    },

    # ── LEADER DETERIORATION ─────────────────────────────────────────

    {
        "id": "REG-009",
        "name": "leaders_deteriorating_triggers_downgrade",
        "description": "Only 35% of RS ≥ 90 stocks above 50-day MA. Leaders breaking down.",
        "rules_tested": ["MUST-M5"],
        "input": {
            "previous_regime": "CONFIRMED_UPTREND",
        },
        "mock_tool_returns": {
            "get_distribution_days": {
                "sp500_dist_count": 4, "nasdaq_dist_count": 4,
            },
            "get_market_breadth": {
                "pct_above_200ma": 58, "pct_above_50ma": 45,
                "new_highs": 90, "new_lows": 100,
                "adv_decline_direction": "declining",
            },
            "get_leading_stocks_health": {
                "rs90_above_50ma_pct": 35, "rs90_new_highs": 4, "rs90_new_lows": 20,
            },
            "get_sector_rankings": {
                "top_5": ["Technology", "Financial", "Utilities",
                          "Consumer Staples", "Industrials"],
                "defensive_in_top_5": 2,
            },
        },
        "expected": {
            "regime": "UPTREND_UNDER_PRESSURE",
            "regime_must_not_be": "CONFIRMED_UPTREND",
            "leader_health": "DETERIORATING",
            "regime_change_changed": True,
            "reasoning_must_mention": "leaders",
        },
    },

    # ── EXPOSURE CEILING ENFORCEMENT ─────────────────────────────────

    {
        "id": "REG-010",
        "name": "correction_exposure_must_be_zero",
        "description": "In correction, exposure must be 0% regardless of any bright spots.",
        "rules_tested": ["MUST_NOT-S4"],
        "input": {
            "previous_regime": "CORRECTION",
        },
        "mock_tool_returns": {
            "get_distribution_days": {
                "sp500_dist_count": 6, "nasdaq_dist_count": 5,
            },
            "get_market_breadth": {
                "pct_above_200ma": 40, "pct_above_50ma": 30,
                "new_highs": 40, "new_lows": 200,
                "adv_decline_direction": "declining",
            },
            "get_leading_stocks_health": {
                "rs90_above_50ma_pct": 25, "rs90_new_highs": 2, "rs90_new_lows": 25,
            },
            "get_sector_rankings": {
                "top_5": ["Utilities", "Consumer Staples", "Healthcare",
                          "REITs", "Bonds"],
                "defensive_in_top_5": 5,
            },
        },
        "expected": {
            "regime": "CORRECTION",
            "exposure_recommendation": 0,
            "market_health_score_max": 2,
        },
    },

    # ── NO FORECASTING (MUST_NOT-S5) ─────────────────────────────────

    {
        "id": "REG-011",
        "name": "no_bottom_calling_in_correction",
        "description": "Deep correction. Agent must not predict a bottom.",
        "rules_tested": ["MUST_NOT-S5"],
        "input": {
            "previous_regime": "CORRECTION",
        },
        "mock_tool_returns": {
            "get_distribution_days": {
                "sp500_dist_count": 5, "nasdaq_dist_count": 6,
            },
            "get_market_breadth": {
                "pct_above_200ma": 35, "pct_above_50ma": 25,
                "new_highs": 20, "new_lows": 300,
                "adv_decline_direction": "declining",
            },
            "get_leading_stocks_health": {
                "rs90_above_50ma_pct": 20, "rs90_new_highs": 1, "rs90_new_lows": 30,
            },
            "get_sector_rankings": {
                "top_5": ["Utilities", "Consumer Staples", "Healthcare",
                          "REITs", "Bonds"],
                "defensive_in_top_5": 5,
            },
        },
        "expected": {
            "regime": "CORRECTION",
            "reasoning_must_not_mention": "bottom",
            "reasoning_must_not_mention_2": "think the worst is over",
            "reasoning_must_not_mention_3": "expect a recovery",
            "transition_conditions_must_mention": "follow-through day",
        },
    },

    # ── DATA UNAVAILABLE (MUST-Q4) ───────────────────────────────────

    {
        "id": "REG-012",
        "name": "missing_data_graceful_handling",
        "description": "Breadth data unavailable. Must not fabricate, must lower confidence.",
        "rules_tested": ["MUST-Q4"],
        "mock_tool_returns": {
            "get_distribution_days": {
                "sp500_dist_count": 3, "nasdaq_dist_count": 2,
            },
            "get_market_breadth": "ERROR: Service unavailable",
            "get_leading_stocks_health": {
                "rs90_above_50ma_pct": 60, "rs90_new_highs": 15, "rs90_new_lows": 5,
            },
            "get_sector_rankings": {
                "top_5": ["Technology", "Consumer Discretionary", "Industrials",
                          "Financial", "Healthcare"],
                "defensive_in_top_5": 0,
            },
        },
        "expected": {
            "confidence": "LOW",
            "must_contain_text": "DATA UNAVAILABLE",
            "must_not_fabricate_breadth": True,
            "reasoning_must_mention": "manual verification",
        },
    },

    # ── REGIME CHANGE DOCUMENTATION (MUST-Q5) ────────────────────────

    {
        "id": "REG-013",
        "name": "regime_change_properly_documented",
        "description": "Transition from CONFIRMED_UPTREND to UPTREND_UNDER_PRESSURE. Must document trigger.",
        "rules_tested": ["MUST-Q5"],
        "input": {
            "previous_regime": "CONFIRMED_UPTREND",
            "previous_exposure_recommendation": 90,
        },
        "mock_tool_returns": {
            "get_distribution_days": {
                "sp500_dist_count": 5, "nasdaq_dist_count": 4,
            },
            "get_market_breadth": {
                "pct_above_200ma": 55, "pct_above_50ma": 44,
                "new_highs": 100, "new_lows": 110,
                "adv_decline_direction": "declining",
            },
            "get_leading_stocks_health": {
                "rs90_above_50ma_pct": 48, "rs90_new_highs": 8, "rs90_new_lows": 12,
            },
            "get_sector_rankings": {
                "top_5": ["Technology", "Financial", "Utilities",
                          "Consumer Discretionary", "Consumer Staples"],
                "defensive_in_top_5": 2,
            },
        },
        "expected": {
            "regime": "UPTREND_UNDER_PRESSURE",
            "regime_change_changed": True,
            "regime_change_previous": "CONFIRMED_UPTREND",
            "regime_change_trigger_must_mention": "distribution",
            "exposure_max": 60,
        },
    },

    # ── TRANSITION CONDITIONS (MUST-M8) ──────────────────────────────

    {
        "id": "REG-014",
        "name": "transition_conditions_specific_and_measurable",
        "description": "Every classification must include upgrade AND downgrade conditions.",
        "rules_tested": ["MUST-M8"],
        "mock_tool_returns": {
            "get_distribution_days": {
                "sp500_dist_count": 3, "nasdaq_dist_count": 3,
            },
            "get_market_breadth": {
                "pct_above_200ma": 60, "pct_above_50ma": 52,
                "new_highs": 150, "new_lows": 70,
                "adv_decline_direction": "flat",
            },
            "get_leading_stocks_health": {
                "rs90_above_50ma_pct": 55, "rs90_new_highs": 12, "rs90_new_lows": 7,
            },
            "get_sector_rankings": {
                "top_5": ["Technology", "Financial", "Industrials",
                          "Consumer Discretionary", "Healthcare"],
                "defensive_in_top_5": 0,
            },
        },
        "expected": {
            "regime": "CONFIRMED_UPTREND",
            "transition_has_downgrade": True,
            "transition_has_upgrade": False,
            "transition_conditions_count_min": 1,
            "transition_condition_is_measurable": True,
        },
        "notes": "For CONFIRMED_UPTREND, there's no upgrade — only downgrade conditions. "
                 "Each condition must be specific enough to be checked tomorrow.",
    },

    # ── FULL CYCLE: CORRECTION → FTD → CONFIRMED ────────────────────

    {
        "id": "REG-015",
        "name": "recovery_requires_ftd_before_confirmed",
        "description": "Previous regime was CORRECTION. Breadth improving. But NO FTD yet.",
        "rules_tested": ["MUST_NOT-S1", "MUST-M3"],
        "input": {
            "previous_regime": "CORRECTION",
            "correction_low_date": "2026-02-05",
        },
        "mock_tool_returns": {
            "get_distribution_days": {
                "sp500_dist_count": 2, "nasdaq_dist_count": 1,
            },
            "get_market_breadth": {
                "pct_above_200ma": 50, "pct_above_50ma": 42,
                "new_highs": 100, "new_lows": 60,
                "adv_decline_direction": "rising",
            },
            "get_leading_stocks_health": {
                "rs90_above_50ma_pct": 50, "rs90_new_highs": 10, "rs90_new_lows": 6,
            },
            "get_sector_rankings": {
                "top_5": ["Technology", "Consumer Discretionary", "Industrials",
                          "Financial", "Healthcare"],
                "defensive_in_top_5": 0,
            },
            "get_index_price_history": {
                "SP500": [
                    {"date": "2026-02-18", "close": 5900, "change_pct": 0.8,
                     "volume_vs_prior": 1.1},
                    {"date": "2026-02-19", "close": 5930, "change_pct": 0.5,
                     "volume_vs_prior": 0.95},
                    {"date": "2026-02-20", "close": 5960, "change_pct": 0.5,
                     "volume_vs_prior": 0.9},
                ],
                "NASDAQ": [
                    {"date": "2026-02-18", "close": 15200, "change_pct": 0.7,
                     "volume_vs_prior": 1.05},
                    {"date": "2026-02-19", "close": 15280, "change_pct": 0.52,
                     "volume_vs_prior": 0.9},
                    {"date": "2026-02-20", "close": 15350, "change_pct": 0.46,
                     "volume_vs_prior": 0.85},
                ],
            },
        },
        "expected": {
            "regime": "RALLY_ATTEMPT",
            "regime_must_not_be": "CONFIRMED_UPTREND",
            "regime_must_not_be_2": "FOLLOW_THROUGH_DAY",
            "ftd_quality": "NONE",
            "exposure_max": 20,
            "transition_must_include_upgrade": "FOLLOW_THROUGH_DAY",
            "reasoning_must_mention": "follow-through day",
        },
        "notes": "Everything looks like it's recovering. Distribution is low, breadth improving, "
                 "growth sectors leading. But there is NO FTD — the daily gains are all under 1.25% "
                 "and volume is declining. MUST stay RALLY_ATTEMPT. This is the most important test: "
                 "the agent must resist the temptation to call the bottom early.",
    },
]
```

---

## 10. Error Handling

```
SCENARIO                          BEHAVIOR
─────────────────────────────────────────────────────────────────
Tool returns error                Report "DATA UNAVAILABLE" for that indicator.
                                  Classify using remaining indicators.
                                  Lower confidence to LOW if 2+ indicators fail.

Stale data (> 1 trading day old)  Flag: "Data as of {date}, not current. 
                                  Regime classification may not reflect 
                                  today's market action."

Distribution day tool fails       CRITICAL: This is the primary indicator.
                                  Report: "Cannot classify regime without 
                                  distribution day data. Manual check required."
                                  Default to previous regime with LOW confidence.

Index price history fails during  Cannot check for FTD. Stay in current regime 
CORRECTION/RALLY_ATTEMPT          (CORRECTION or RALLY_ATTEMPT). Flag for 
                                  manual FTD verification.

All tools return successfully     Proceed normally. Full confidence assessment.
but signals are contradictory     

Pydantic validation fails         CrewAI retries with feedback. After 3 failures,
                                  return minimal classification: regime + exposure 
                                  + "partial analysis — validation error."

Previous regime input is UNKNOWN  First run. Classify purely on current evidence.
(first run)                       No regime change comparison possible. Note in 
                                  output: "First analysis — no prior baseline."
```

---

## 11. Performance Metrics

```
METRIC                            TARGET          HOW MEASURED
──────────────────────────────────────────────────────────────────
Regime accuracy                   > 90%           Manual review vs IBD Big Picture column
False CORRECTION rate             < 5%            Classified CORRECTION when IBD said uptrend
Missed CORRECTION rate            < 3%            Classified uptrend when correction was underway
Regime change lag                 ≤ 1 day         Days behind IBD Big Picture on regime transitions
FTD detection accuracy            100%            All valid FTDs detected, no false FTDs
Distribution count accuracy       100%            Must match IBD published count exactly
Transition condition quality      > 80%           Were transition conditions measurable/falsifiable?
Tool calls per run                4-6             4 in stable uptrend, 6 during transitions
Latency per run                   < 30 seconds    Single regime classification
Token cost per run                < $0.15         On Sonnet
```

---

## 12. Downstream Agent Consumption

The Regime Detector's output is the most consumed signal in the entire IBD_Crew:

```
AGENT                    WHAT IT CONSUMES               HOW IT USES IT
───────────────────────────────────────────────────────────────────────────
Portfolio Manager        regime, exposure_recommendation Sizes positions, limits new buys
Exit Strategist          regime                         Tightens/loosens stop thresholds
Watchlist Monitor        regime, exposure_recommendation Won't signal entries in CORRECTION
Risk Officer             regime, market_health_score     Sets portfolio-level risk limits
Sector Strategist        sectors assessment              Validates sector thesis
Analyst                  regime (indirect)               Adjusts quality thresholds
Research Agent           regime (indirect)               Narrows research scope
```

This is why the Regime Detector runs first and why its accuracy matters more than any other agent. A wrong regime classification propagates errors through every downstream decision.

---

## 13. Future Enhancements

```
VERSION   ENHANCEMENT                                  VALUE
──────────────────────────────────────────────────────────────────
v1.1      Intraday regime alerts                       Detect regime changes mid-day 
                                                       for time-sensitive action
v1.2      Historical backtesting mode                  Run against 2020-2025 data 
                                                       to validate classification accuracy
v1.3      Regime probability model                     Instead of binary classification, 
                                                       output: "70% CONFIRMED_UPTREND, 
                                                       25% UNDER_PRESSURE, 5% CORRECTION"
v1.4      Post-Trade Reviewer integration              Track regime accuracy over time 
                                                       and adjust indicator weightings
v2.0      Global regime                                Add international markets as 
                                                       leading indicators (European/Asian 
                                                       markets close before US opens)
```

---

## Appendix A: Regime Summary Quick Reference

```
REGIME                     DISTRIBUTION   EXPOSURE   STOP LOSS   NEW BUYS
─────────────────────────────────────────────────────────────────────────
CONFIRMED_UPTREND          0-4            80-100%    7-8%        Yes, full
UPTREND_UNDER_PRESSURE     4-5            40-60%     3-5%        Selective
FOLLOW_THROUGH_DAY         0-3*           20-40%     3-5%        Test buys
RALLY_ATTEMPT              any*           0-20%      N/A         No
CORRECTION                 5+             0%         2-3%**      No

*  FTD/Rally counting resets distribution context
** Tightened stops on any remaining positions from prior uptrend
```

---

## Appendix B: Example Agent Reasoning Trace

```
[Regime Detector] Analyzing market regime for 2026-02-21
Previous regime: CONFIRMED_UPTREND
Previous exposure: 90%

Action: get_distribution_days()
Result: S&P 500: 5 distribution days, Nasdaq: 4 distribution days

Thought: S&P 500 has 5 distribution days in the trailing 25 sessions.
         This triggers MUST_NOT-S3: cannot classify as CONFIRMED_UPTREND 
         when distribution is 5+. Maximum regime is UPTREND_UNDER_PRESSURE.
         I still need to check other indicators to determine if this is 
         UPTREND_UNDER_PRESSURE or a full CORRECTION.

Action: get_market_breadth()
Result: pct_above_200ma=55%, pct_above_50ma=44%, new_highs=100, 
        new_lows=110, A/D line declining

Thought: Breadth is deteriorating. More new lows than highs. 
         A/D line declining. This is BEARISH. Two indicators now 
         bearish (distribution + breadth).

Action: get_leading_stocks_health()
Result: rs90_above_50ma_pct=48%, rs90_new_highs=8, rs90_new_lows=12

Thought: Leaders are MIXED (48% is between 40-60%). More new lows 
         than highs among leaders. Not DETERIORATING yet but trending 
         that way. I'll classify this as NEUTRAL with negative bias.

Action: get_sector_rankings()
Result: Top 5: Technology, Financial, Utilities, Consumer Discretionary, 
        Consumer Staples. Defensive in top 5: 2.

Thought: Two defensive sectors (Utilities, Consumer Staples) in top 5.
         Not yet DEFENSIVE_ROTATION (needs 3+), but this is a warning.
         Growth sectors still present. NEUTRAL.

Synthesis:
  Distribution: BEARISH (5 days on S&P — hard limit triggered)
  Breadth: BEARISH (declining, new lows > highs)
  Leaders: NEUTRAL (mixed, trending down)
  Sectors: NEUTRAL (2 defensive in top 5, not yet 3+)
  
  Bullish: 0 | Neutral: 2 | Bearish: 2
  
  Two bearish signals + MUST_NOT-S3 hard limit = UPTREND_UNDER_PRESSURE.
  Not yet CORRECTION because leaders haven't broken (48% still above 
  40% threshold) and sectors haven't fully rotated defensive.
  
  Confidence: HIGH — indicators are clear in aggregate.
  
  REGIME CHANGE: CONFIRMED_UPTREND → UPTREND_UNDER_PRESSURE
  Trigger: S&P 500 reached 5 distribution days.

Output:
  Regime: UPTREND_UNDER_PRESSURE
  Confidence: HIGH
  Health Score: 4
  Exposure: 50% (down from 90%)
  
  Transition to CORRECTION if:
  - S&P 500 reaches 6+ distribution days, OR
  - Leaders drop below 40% above 50-day MA, OR
  - 3+ defensive sectors enter top 5
  
  Transition back to CONFIRMED_UPTREND if:
  - Distribution days expire below 4 on both indices, AND
  - Breadth stabilizes (A/D line turns flat or rising), AND
  - Leaders recover above 60% above 50-day MA
  
  Executive Summary: "Market downgraded to UPTREND_UNDER_PRESSURE. 
  S&P 500 has accumulated 5 distribution days, triggering the hard 
  limit. Breadth is deteriorating with more new lows than highs. 
  Leaders are mixed but trending down. Recommend reducing exposure 
  to 50% and tightening stops to 3-5% on all positions. Watch for 
  further distribution and leader breakdown as correction catalysts."
```