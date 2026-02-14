# Agent 07: Portfolio Reconciler 🔀

> **Build Phase:** 7
> **Depends On:** Risk Officer (approved portfolio) + Current Holdings files
> **Feeds Into:** Educator
> **Target:** Week 8
> **Methodology:** IBD Momentum Investment Framework v4.0

---

## 1. Identity

| Field | Value |
|---|---|
| **Role** | Portfolio Implementation Strategist |
| **Experience** | 15+ years executing large portfolio transitions; specialist in minimizing tax impact and market impact |
| **Personality** | Methodical, detail-oriented, transition-focused |
| **Temperature** | 0.3 (precise — this is about exact positions and dollar amounts) |
| **Delegation** | NOT allowed (must independently verify every position) |

---

## 2. Goal

> Compare the **current actual portfolio holdings** (from brokerage export files) against the **recommended portfolio** (Risk Officer–approved output from Pipeline). Identify every difference — positions to sell, buy, add, trim, and keep — and produce a **phased implementation plan** with dollar amounts, priority ordering, tax considerations, and weekly execution timeline.

### Why This Agent Exists

The Pipeline builds an *ideal* portfolio. But the investor already holds positions. This agent answers:

1. **"What do I actually own right now?"** — Read brokerage files
2. **"How does that differ from the recommendation?"** — Position-by-position diff
3. **"What's the transition plan?"** — Phased, prioritized, with dollar amounts
4. **"What are the money flows?"** — Sources (sells/trims) → Uses (buys/adds) → Cash remainder

Without this agent, the investor gets a recommendation but no roadmap to get there.

---

## 3. ✅ DOES / ❌ DOES NOT

### ✅ DOES
- Read current holdings from portfolio data files (Schwab, E*TRADE PDFs/CSVs)
- Build complete inventory of current positions (symbol, shares, value, account)
- Compare every current position to the recommended portfolio
- Classify each position: KEEP, SELL, BUY NEW, ADD (increase), TRIM (decrease)
- Calculate exact dollar amounts for every action (sell proceeds, buy costs)
- Identify positions that exist in current but NOT in recommended (sells)
- Identify positions that exist in recommended but NOT in current (new buys)
- Identify positions that exist in both but at different sizes (adds/trims)
- Verify all 14 keep-positions are present in current holdings
- Calculate money flow: total sources, total uses, net cash
- Produce phased weekly implementation timeline (Framework v4.0 §5.3)
- Flag tax implications (retirement vs taxable account, wash sale risk)
- Prioritize execution order (sells first, then buys by tier priority)
- Track implementation progress metrics

### ❌ DOES NOT
- Change the recommended portfolio (that's PM + Risk Officer)
- Override any position the Risk Officer approved
- Pick new stocks or modify tier assignments
- Make market timing decisions ("wait for a dip")
- Provide tax advice (flags considerations, not advice)
- Execute trades (produces the plan, not the execution)

---

## 4. Framework v4.0 Rules (Hard-Coded)

### 4.1 Two Categories of Keeps (Must Verify All 14)

```python
EXPECTED_KEEPS = {
    "fundamental_review": {
        "UNH":  {"target": 30270, "tier": 3},
        "MU":   {"target": 25000, "tier": 1},
        "AMZN": {"target": 33591, "tier": 2},
        "MRK":  {"target": 20000, "tier": 3},
        "COP":  {"target": 15000, "tier": 2},
    },
    "ibd_high_rated": {
        "CLS":  {"target": 16324, "comp": 99, "rs": 97, "tier": 1},
        "AGI":  {"target": 16599, "comp": 99, "rs": 92, "tier": 1},
        "TFPM": {"target": 10109, "comp": 99, "rs": 92, "tier": 1},
        "GMED": {"target": 15588, "comp": 98, "rs": 91, "tier": 2},
        "SCCO": {"target": 12448, "comp": 98, "rs": 94, "tier": 1},
        "HWM":  {"target": 10914, "comp": 98, "rs": 90, "tier": 1},
        "KLAC": {"target": 19600, "comp": 97, "rs": 94, "tier": 1},
        "TPR":  {"target": 10614, "comp": 97, "rs": 93, "tier": 2},
        "GS":   {"target": 19719, "comp": 93, "rs": 91, "tier": 2},
    },
}
# Total: 5 + 9 = 14 expected keeps
```

### 4.2 Position Sizing Reference

```python
POSITION_LIMITS = {
    "stock": {1: 5, 2: 4, 3: 3},   # max % of portfolio
    "etf":   {1: 8, 2: 8, 3: 6},
}

TRAILING_STOPS = {1: 22, 2: 18, 3: 12}
```

### 4.3 Tier Allocation Targets

```python
TIER_TARGETS = {
    "tier_1": {"target_pct": 39, "range": (35, 40)},
    "tier_2": {"target_pct": 37, "range": (33, 40)},
    "tier_3": {"target_pct": 22, "range": (20, 30)},
    "cash":   {"target_pct": 2,  "range": (2, 5)},
}
```

### 4.4 Implementation Timeline Framework (v4.0 §5.3)

```python
IMPLEMENTATION_PHASES = {
    "week_1": {
        "focus": "LIQUIDATION",
        "actions": ["Sell all positions not in recommended portfolio",
                    "Sell mutual funds (longer settlement)",
                    "Execute trims on oversized positions",
                    "Generate cash for redeployment"],
        "priority": "Sells first — generate proceeds before buying",
    },
    "week_2": {
        "focus": "TIER 1 MOMENTUM + DEFENSIVE FACTORS",
        "actions": ["Buy T1 momentum positions",
                    "Buy defensive factor ETFs (QUAL, SPLV, USMV)",
                    "Add to existing positions that need size increase"],
        "priority": "Establish core momentum + safety net",
    },
    "week_3": {
        "focus": "TIER 2 QUALITY GROWTH",
        "actions": ["Buy T2 quality growth positions",
                    "Buy thematic ETFs (CIBR, REMX, COPX)",
                    "Complete adds to existing positions"],
        "priority": "Build growth engine",
    },
    "week_4": {
        "focus": "TIER 3 DEFENSIVE + FINALIZE",
        "actions": ["Buy remaining T3 defensive positions",
                    "Set ALL trailing stop orders",
                    "Verify final position count and allocation",
                    "Document final state"],
        "priority": "Complete protection layer, activate all stops",
    },
}
```

### 4.5 Portfolio Files to Read

```python
PORTFOLIO_FILES = {
    "schwab": "data/portfolios/Positions - Charles Schwab.pdf",
    "etrade_1": "data/portfolios/Etrade1157.pdf",
    "etrade_2": "data/portfolios/Etrade6270.pdf",
}
```

### 4.6 Diversification Constraints (Verify After Transition)

```python
DIVERSIFICATION = {
    "max_single_sector": 40,
    "min_sectors": 8,
    "max_single_stock": 5,
    "max_single_etf": 8,
    "stock_etf_split": (50, 50),  # ±5% tolerance
}
```

---

## 5. Tools

| Tool Name | Purpose | Input | Output |
|---|---|---|---|
| `read_portfolio_holdings` | Parse brokerage PDF/CSV files | File path | List of positions (symbol, shares, value, account) |
| `position_differ` | Compare current vs recommended | Two position lists | Diff report (keeps, sells, buys, adds, trims) |
| `money_flow_calculator` | Calculate sources and uses | Diff report | Sources total, uses total, cash remainder |
| `timeline_builder` | Assign actions to 4-week phases | Diff report + priorities | Weekly action plan |

---

## 6. Autonomy & Decision-Making

### Decision 1: Clean Match — Position Exists at Correct Size
```
Current: NVDA — $34,200 (Schwab)
Recommended: NVDA — $35,000 (T1 Momentum)

Agent reasoning: "NVDA exists in current holdings at $34,200.
Target is $35,000. Difference is $800 (2.3% of position).

ACTION: ADD $800 to NVDA
PRIORITY: Low (position is 97.7% of target — close enough)
PHASE: Week 2 (T1 Momentum phase)
CATEGORY: Minor adjustment"
```

### Decision 2: Position to Sell — Not in Recommended
```
Current: CAT — $18,500 (Schwab)
Recommended: CAT not found in approved portfolio

Agent reasoning: "CAT exists at $18,500 but is NOT in the recommended
portfolio. This is a full liquidation.

ACTION: SELL $18,500 of CAT
PRIORITY: HIGH (generates cash for redeployment)
PHASE: Week 1 (Liquidation phase)
CATEGORY: Full sell
PROCEEDS: $18,500 → contributes to buy pool
NOTE: Check if held >1 year for capital gains treatment"
```

### Decision 3: New Buy — Not in Current Holdings
```
Current: FIX — not found
Recommended: FIX — $30,000 (T1 Momentum, IBD 99/96)

Agent reasoning: "FIX is a new T1 Momentum position not currently held.
This is a new purchase.

ACTION: BUY $30,000 of FIX
PRIORITY: HIGH (T1 momentum, IBD 99 Composite)
PHASE: Week 2 (T1 Momentum phase)
CATEGORY: New buy
FUNDED BY: Sell proceeds from Week 1
STOP ORDER: Set 22% trailing stop immediately after purchase"
```

### Decision 4: Trim — Current Position Oversized
```
Current: AVGO — $40,000 (Schwab)
Recommended: AVGO — $35,000 (T2, Quality Growth)

Agent reasoning: "AVGO exists at $40,000 but target is $35,000.
This is a $5,000 reduction.

ACTION: TRIM AVGO by $5,000 (sell $5K, keep $35K)
PRIORITY: MEDIUM (generates cash, right-sizes position)
PHASE: Week 1 (Liquidation phase — trims alongside full sells)
CATEGORY: Trim (reduce to target)
PROCEEDS: $5,000 → contributes to buy pool
NOTE: AVGO remains in portfolio at $35K target"
```

### Decision 5: Keep Verification — Missing Keep Position
```
Current: AGI — not found in any brokerage file
Expected: AGI is an IBD High-Rated Keep (Comp 99, RS 92, target $16,599)

Agent reasoning: "WARNING: AGI is expected as an IBD Keep but NOT found
in current holdings. This should not happen — keeps are positions that
already exist.

POSSIBLE EXPLANATIONS:
1. AGI was sold since the keep analysis was done
2. AGI is held under a different symbol or in an unread account
3. Data extraction missed it

ACTION: FLAG as 'Missing Keep — requires manual verification'
RECOMMENDATION: If AGI was sold, treat as New Buy ($16,599, T1)
If AGI exists elsewhere, update holdings data"
```

### Decision 6: Multi-Account Reconciliation
```
Current: GOOGL — $25,000 (Schwab) + $15,000 (E*TRADE 1)
Recommended: GOOGL — $35,000 (T1 Momentum)

Agent reasoning: "GOOGL exists across TWO accounts totaling $40,000.
Target is $35,000. This is an OVER-allocation of $5,000.

ACTION: TRIM GOOGL by $5,000
ACCOUNT: Trim from E*TRADE 1 ($15K → $10K) to consolidate toward Schwab
PRIORITY: LOW
PHASE: Week 1 (alongside other trims)
NOTE: Consider tax lot selection — trim highest cost basis first"
```

---

## 7. Output Contract

### ReconciliationOutput (Top Level)

```python
class ReconciliationOutput(BaseModel):
    # Current state
    current_holdings: HoldingsSummary            # What you own NOW
    recommended_portfolio: PortfolioSummary       # What the pipeline recommends
    
    # The diff
    positions_to_sell: List[SellAction]           # Full liquidations
    positions_to_buy: List[BuyAction]             # New purchases
    positions_to_add: List[AddAction]             # Increase existing
    positions_to_trim: List[TrimAction]           # Decrease existing
    positions_unchanged: List[str]                # Already at target (symbols only)
    
    # Keep verification
    keep_verification: KeepVerification           # All 14 keeps accounted for
    
    # Money flow
    money_flow: MoneyFlow                         # Sources, uses, net cash
    
    # Implementation plan
    implementation_plan: ImplementationPlan       # 4-week phased timeline
    
    # Metrics
    transformation_metrics: TransformationMetrics # Before/after comparison
    
    analysis_date: str
    summary: str                                  # 150+ chars

    # VALIDATORS:
    # - At least 1 action (sell, buy, add, or trim) unless portfolios match perfectly
    # - money_flow.net_cash ≥ 0 (can't spend more than available)
    # - keep_verification accounts for all 14 keeps
    # - implementation_plan has 4 weeks
    # - summary ≥ 150 chars
```

### HoldingsSummary

```python
class CurrentPosition(BaseModel):
    symbol: str
    account: str                                  # "Schwab", "E*TRADE 1157", "E*TRADE 6270"
    current_value: float                          # Current $ value
    asset_type: str                               # "stock" / "etf" / "mutual_fund"

class HoldingsSummary(BaseModel):
    total_value: float
    total_positions: int
    positions_by_account: Dict[str, int]          # account → count
    stock_count: int
    etf_count: int
    mutual_fund_count: int
    positions: List[CurrentPosition]
```

### Action Types

```python
class SellAction(BaseModel):
    symbol: str
    account: str
    current_value: float
    reason: str                                   # "Not in recommended portfolio"
    priority: str                                 # "HIGH" / "MEDIUM" / "LOW"
    phase: str                                    # "Week 1" / "Week 2" etc.
    proceeds: float                               # Cash generated
    tax_note: Optional[str]                       # "Retirement — no tax impact"

class BuyAction(BaseModel):
    symbol: str
    target_account: str                           # Where to buy
    target_value: float
    tier: int                                     # 1, 2, or 3
    asset_type: str                               # "stock" / "etf"
    priority: str
    phase: str
    stop_loss_pct: int                            # 22, 18, or 12
    category: str                                 # "new_buy" / "ibd_keep_missing"
    reason: str                                   # "T1 Momentum, IBD 99/96"

class AddAction(BaseModel):
    symbol: str
    account: str
    current_value: float
    target_value: float
    add_amount: float                             # target - current
    tier: int
    priority: str
    phase: str
    reason: str                                   # "Increase from $19K to $35K per target"

class TrimAction(BaseModel):
    symbol: str
    account: str
    current_value: float
    target_value: float
    trim_amount: float                            # current - target
    tier: int
    priority: str
    phase: str
    proceeds: float
    reason: str                                   # "Reduce oversized position to target"
```

### KeepVerification

```python
class KeepStatus(BaseModel):
    symbol: str
    keep_category: str                            # "fundamental" / "ibd"
    found_in_current: bool
    current_value: Optional[float]
    target_value: float
    action_needed: str                            # "none" / "add" / "trim" / "buy_new" / "missing"
    note: Optional[str]

class KeepVerification(BaseModel):
    fundamental_keeps: List[KeepStatus]           # 5 expected
    ibd_keeps: List[KeepStatus]                   # 9 expected
    total_found: int                              # How many of 14 found
    total_missing: int                            # How many not found
    status: str                                   # "ALL_PRESENT" / "MISSING_KEEPS"
```

### MoneyFlow

```python
class MoneyFlow(BaseModel):
    # Sources (cash generated)
    total_sell_proceeds: float
    total_trim_proceeds: float
    existing_cash: float
    total_sources: float                          # sells + trims + cash
    
    # Uses (cash needed)
    total_new_buys: float
    total_adds: float
    total_uses: float                             # buys + adds
    
    # Result
    net_cash: float                               # sources - uses (must be ≥ 0)
    cash_reserve_pct: float                       # net_cash / total_portfolio_value
    within_cash_target: bool                      # 2-5% per framework
    
    # Breakdown
    sell_count: int
    trim_count: int
    buy_count: int
    add_count: int
```

### ImplementationPlan

```python
class WeekPlan(BaseModel):
    week_number: int                              # 1-4
    focus: str                                    # "LIQUIDATION" / "T1 MOMENTUM" etc.
    actions: List[WeekAction]
    estimated_cash_flow: float                    # Net flow for this week
    cumulative_cash: float                        # Running cash balance

class WeekAction(BaseModel):
    day: str                                      # "Monday" / "Tuesday" etc.
    action_type: str                              # "SELL" / "BUY" / "ADD" / "TRIM" / "SET_STOP"
    symbols: List[str]
    estimated_amount: float
    notes: Optional[str]

class ImplementationPlan(BaseModel):
    weeks: List[WeekPlan]                         # Exactly 4 weeks
    total_sell_transactions: int
    total_buy_transactions: int
    total_transactions: int
    execution_notes: List[str]                    # Tips and warnings
```

### TransformationMetrics

```python
class TransformationMetrics(BaseModel):
    # Before (current)
    before_total_positions: int
    before_stock_count: int
    before_etf_count: int
    before_mutual_fund_count: int
    before_sector_concentration: Dict[str, float]  # sector → %
    before_tech_pct: float
    before_defensive_pct: float
    
    # After (recommended)
    after_total_positions: int
    after_stock_count: int
    after_etf_count: int
    after_mutual_fund_count: int
    after_sector_concentration: Dict[str, float]
    after_tech_pct: float
    after_defensive_pct: float
    
    # Changes
    positions_eliminated: int
    positions_added: int
    positions_unchanged: int
    positions_resized: int                         # adds + trims
    turnover_pct: float                            # (sells + buys) / total_value × 100
```

---

## 8. Example Output (Partial)

```json
{
  "money_flow": {
    "total_sell_proceeds": 277653,
    "total_trim_proceeds": 63043,
    "existing_cash": 20237,
    "total_sources": 360933,
    "total_new_buys": 322000,
    "total_adds": 86976,
    "total_uses": 408976,
    "net_cash": 41834,
    "cash_reserve_pct": 2.8,
    "within_cash_target": true,
    "sell_count": 23,
    "trim_count": 4,
    "buy_count": 14,
    "add_count": 10
  },
  "keep_verification": {
    "fundamental_keeps": [
      {"symbol": "UNH", "keep_category": "fundamental", "found_in_current": true,
       "current_value": 28500, "target_value": 30270, "action_needed": "add",
       "note": "Add $1,770 to reach target"},
      {"symbol": "MU", "keep_category": "fundamental", "found_in_current": true,
       "current_value": 21000, "target_value": 25000, "action_needed": "add",
       "note": "Add $4,000 — AI memory leader, IBD 99/99"}
    ],
    "total_found": 14,
    "total_missing": 0,
    "status": "ALL_PRESENT"
  },
  "transformation_metrics": {
    "before_total_positions": 70,
    "before_stock_count": 47,
    "before_etf_count": 23,
    "before_mutual_fund_count": 2,
    "before_tech_pct": 55.0,
    "before_defensive_pct": 5.0,
    "after_total_positions": 59,
    "after_stock_count": 38,
    "after_etf_count": 21,
    "after_mutual_fund_count": 0,
    "after_tech_pct": 28.0,
    "after_defensive_pct": 22.0,
    "positions_eliminated": 36,
    "positions_added": 23,
    "positions_unchanged": 2,
    "positions_resized": 14,
    "turnover_pct": 54.5
  },
  "implementation_plan": {
    "weeks": [
      {
        "week_number": 1,
        "focus": "LIQUIDATION — Generate cash",
        "actions": [
          {"day": "Monday", "action_type": "SELL", "symbols": ["ACAAX", "FBGRX", "CAT", "IDXX", "AMD"],
           "estimated_amount": 192000, "notes": "Mutual funds first (settlement delay)"},
          {"day": "Monday", "action_type": "SELL", "symbols": ["VEA", "XLV", "ICLN"],
           "estimated_amount": 183000, "notes": "Largest ETF sells — generates bulk of cash"},
          {"day": "Tuesday", "action_type": "TRIM", "symbols": ["AVGO", "MSFT"],
           "estimated_amount": 15000, "notes": "Reduce oversized positions to target"},
          {"day": "Wednesday", "action_type": "SELL", "symbols": ["EXPE", "WMT", "GTLS", "LIVN", "BAP"],
           "estimated_amount": 55000, "notes": null}
        ],
        "estimated_cash_flow": 807000,
        "cumulative_cash": 827237
      }
    ],
    "total_sell_transactions": 36,
    "total_buy_transactions": 24,
    "total_transactions": 60,
    "execution_notes": [
      "SELL mutual funds (ACAAX, FBGRX) first — they take 1-2 days to settle",
      "Complete ALL sells in Week 1 before starting buys in Week 2",
      "Set trailing stop orders on EVERY new purchase within 24 hours",
      "All accounts are retirement (Schwab IRA) — no capital gains tax impact",
      "Monitor market conditions — if IBD Market Pulse shifts to 'Under Pressure', pause Week 2 buys",
      "Verify cash settlement before placing buy orders (T+1 for stocks, T+1 for ETFs)"
    ]
  },
  "summary": "Portfolio transformation from 70 positions to 59 requires 36 sells ($278K stock + $351K ETF + $116K mutual funds), 4 trims ($63K from oversized positions), 14 new buys ($322K across all tiers), and 10 adds ($87K to bring existing positions to target). All 14 keep positions verified present in current holdings. Net cash reserve: $42K (2.8%) within the 2-5% target. Execution over 4 weeks: Week 1 liquidation, Week 2 T1 momentum, Week 3 T2 quality, Week 4 T3 defensive + stop activation. Total turnover: 54.5%. Technology concentration drops from 55% to 28%. Defensive allocation increases from 5% to 22%."
}
```

---

## 9. Testing Plan

### Level 1: Schema Tests (No LLM — instant)

| Test | What It Validates |
|---|---|
| `test_valid_output_parses` | Known-good output parses |
| `test_money_flow_balanced` | total_sources ≥ total_uses |
| `test_net_cash_non_negative` | net_cash ≥ 0 |
| `test_cash_reserve_within_target` | 2-5% (or flagged) |
| `test_keep_verification_14_total` | fundamental(5) + ibd(9) = 14 |
| `test_sell_has_proceeds` | Every sell has proceeds > 0 |
| `test_buy_has_tier` | Every buy has tier 1/2/3 |
| `test_buy_has_stop_loss` | Every buy has stop (22/18/12) |
| `test_implementation_4_weeks` | Exactly 4 week plans |
| `test_week1_is_liquidation` | Week 1 focus contains sells |
| `test_add_amount_positive` | add_amount > 0 for all adds |
| `test_trim_amount_positive` | trim_amount > 0 for all trims |
| `test_transformation_metrics_consistent` | eliminated + added + unchanged + resized ≈ total |
| `test_summary_minimum_length` | ≥ 150 chars |
| `test_action_priorities_valid` | Only HIGH/MEDIUM/LOW |

### Level 2: LLM Output Tests

| Test | What It Validates |
|---|---|
| `test_output_is_valid_json` | Parseable JSON |
| `test_output_matches_schema` | Conforms to ReconciliationOutput |
| `test_sells_not_in_recommended` | Every sell symbol absent from recommended |
| `test_buys_not_in_current` | Every new buy absent from current |
| `test_adds_in_both` | Every add exists in both current and recommended |
| `test_trims_in_both` | Every trim exists in both |
| `test_stop_matches_tier` | T1→22%, T2→18%, T3→12% |
| `test_all_keeps_verified` | 14 keeps all accounted for |
| `test_money_flow_math_correct` | Sources - uses = net_cash |

### Level 3: Behavioral Tests

| Test | What It Validates |
|---|---|
| `test_does_not_change_recommendations` | No new symbols, no tier changes |
| `test_does_not_provide_tax_advice` | Flags considerations, no "you should" |
| `test_does_not_time_market` | No "wait for dip" language |
| `test_does_not_override_risk_officer` | Respects approved portfolio exactly |
| `test_sells_before_buys` | Week 1 sells, Week 2+ buys |
| `test_reads_portfolio_files` | References actual brokerage data |

### Level 4: Integration (Risk Officer → Reconciler)

| Test | What It Validates |
|---|---|
| `test_reconciler_accepts_risk_output` | Handoff works |
| `test_recommended_positions_match_pm` | Recommended = Risk-approved PM output |
| `test_current_holdings_from_files` | Actually reads portfolio files |
| `test_known_portfolio_diff` | Given known current + recommended, produces correct diff |
| `test_output_feeds_educator` | Educator can consume reconciliation output |

---

## 10. Integration Points

### Receives From
- **Risk Officer:** Approved `PortfolioOutput` (the target state) + `RiskAssessment`
- **Portfolio files:** `data/portfolios/*.pdf` (the current state)
- **Portfolio Manager:** Keep categories and targets

### Passes To
- **Educator:** `ReconciliationOutput` — for explaining the transition plan
  - Educator uses `money_flow` to explain sources/uses
  - Educator uses `implementation_plan` for weekly action explanation
  - Educator uses `transformation_metrics` for before/after comparison
  - Educator uses `keep_verification` to confirm kept positions

### Data Flow

```
Risk-Approved Portfolio ──┐
                          ├──▶ RECONCILER ──▶ Educator
Current Holdings Files ───┘        │
                                   ▼
                          Implementation Plan
                          (sell/buy/add/trim)
                          + Money Flow
                          + 4-Week Timeline
```

---

## 11. CrewAI Configuration

### agents.yaml

```yaml
portfolio_reconciler:
  role: >
    Portfolio Implementation Strategist
  goal: >
    Compare current actual holdings against the recommended portfolio.
    Identify every difference and produce a phased 4-week implementation
    plan with exact dollar amounts, priority ordering, keep verification,
    and money flow analysis. You bridge the gap between recommendation
    and execution.
  backstory: >
    You are a meticulous portfolio transition specialist who has
    executed hundreds of large portfolio restructurings. You know
    that the best investment plan is worthless without a clear
    implementation roadmap. You read brokerage statements, calculate
    exact dollar differences, verify that all keep-positions are
    accounted for, and produce day-by-day action plans. You never
    modify the investment recommendations — only translate them into
    executable steps. You are the GPS that gets the investor from
    Point A (current portfolio) to Point B (recommended portfolio).
```

### tasks.yaml

```yaml
reconciliation_task:
  description: >
    Read the current portfolio holdings from brokerage files and compare
    against the Risk Officer-approved recommended portfolio.

    For every position:
    1. If in current but NOT recommended → SELL (full liquidation)
    2. If in recommended but NOT current → BUY (new purchase)
    3. If in both but current < target → ADD (increase position)
    4. If in both but current > target → TRIM (decrease position)
    5. If in both at ~same size → UNCHANGED

    Verify all 14 keep positions (5 Fundamental + 9 IBD).
    Calculate complete money flow (sources - uses = cash reserve).
    Build 4-week implementation timeline per Framework v4.0.
    
    Return ONLY a valid JSON object matching the ReconciliationOutput schema.
  expected_output: >
    JSON object with: current_holdings, recommended_portfolio,
    positions_to_sell, positions_to_buy, positions_to_add,
    positions_to_trim, keep_verification, money_flow,
    implementation_plan, transformation_metrics, summary.
  agent: portfolio_reconciler
```

---

## 12. Files to Create

```
1. src/ibd_agents/schemas/reconciliation_output.py          ← Output contract (FIRST)
2. tests/unit/test_reconciliation_schema.py                  ← Schema tests (15+)
3. src/ibd_agents/tools/portfolio_reader.py                  ← Read brokerage PDFs/CSVs
4. src/ibd_agents/tools/position_differ.py                   ← Diff engine
5. src/ibd_agents/tools/money_flow_calculator.py             ← Sources/uses math
6. src/ibd_agents/agents/portfolio_reconciler.py             ← Agent + Task
7. config/agents.yaml                                        ← Add reconciler entry
8. config/tasks.yaml                                         ← Add reconciler task
9. tests/unit/test_reconciler_agent.py                       ← LLM output tests
10. tests/unit/test_reconciler_behavior.py                   ← Behavioral tests
11. tests/integration/test_risk_to_reconciler.py             ← Handoff from Risk
12. tests/integration/test_reconciler_to_educator.py         ← Handoff to Educator
13. tests/fixtures/mock_current_holdings.json                ← Test current state
14. tests/fixtures/mock_recommended_portfolio.json           ← Test target state
15. golden_datasets/reconciler_golden.json                   ← Golden baseline
```