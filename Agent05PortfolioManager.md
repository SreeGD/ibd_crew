# Agent 05: Portfolio Manager 💼

> **Build Phase:** 5
> **Depends On:** Analyst Agent + Sector Strategist
> **Feeds Into:** Risk Officer
> **Framework v4.0 Sections:** Part 3 Keeps, Part 4 Architecture, 4.5 Sizing, 6.1 Stops, 6.3 Limits

---

## 1. Identity

| Field | Value |
|---|---|
| **Role** | Portfolio Construction Specialist |
| **Experience** | $500M+ AUM, institutional growth portfolios |
| **Temperature** | 0.5 (balanced judgment for capital allocation) |

---

## 2. Goal

> Construct a **3-tier portfolio** (Momentum / Quality Growth / Defensive) using the Analyst's top 50 stocks, Strategist's sector allocations, and Framework v4.0's position sizing rules. Handle both keep categories. Ensure compliance with all concentration limits, trailing stop rules, and stock/ETF allocation targets.

---

## 3. ✅ DOES / ❌ DOES NOT

### ✅ DOES
- Build portfolio with T1/T2/T3 tier structure per Framework
- Apply exact position sizing limits per tier (stocks AND ETFs)
- Handle both keep categories (Fundamental/IBD)
- Assign trailing stops per tier (22%/18%/12%)
- Maintain 50/50 stock/ETF allocation target
- Respect 40% max single sector, 8 min sectors
- Generate buy/sell/add/trim order list
- Respond to Risk Officer veto by rebalancing

### ❌ DOES NOT
- Discover new stocks (only uses Analyst's rated + unrated)
- Override Risk Officer veto
- Detect rotation or make sector predictions
- Provide macroeconomic analysis

---

## 4. Framework v4.0 Rules (Hard-Coded)

### 4.1 Three-Tier Architecture

```python
TIER_TARGETS = {
    "tier_1": {"target_pct": 39, "range": (35, 40), "label": "Momentum"},
    "tier_2": {"target_pct": 37, "range": (33, 40), "label": "Quality Growth"},
    "tier_3": {"target_pct": 22, "range": (20, 30), "label": "Defensive"},
    "cash":   {"target_pct": 2,  "range": (2, 5)},
}
```

### 4.2 Position Sizing Limits

```python
STOCK_LIMITS = {
    1: {"max_pct": 5, "typical_pct": "2-3", "stop_pct": 22},
    2: {"max_pct": 4, "typical_pct": "1.5-2.5", "stop_pct": 18},
    3: {"max_pct": 3, "typical_pct": "1-2", "stop_pct": 12},
}

ETF_LIMITS = {
    1: {"max_pct": 8, "typical_pct": "3-4", "stop_pct": 22},
    2: {"max_pct": 8, "typical_pct": "2-3", "stop_pct": 18},
    3: {"max_pct": 6, "typical_pct": "2-3", "stop_pct": 12},
}
```

### 4.3 Trailing Stop Tightening Protocol

```python
STOP_TIGHTENING = {
    1: {"initial": 22, "after_10pct_gain": (15, 18), "after_20pct_gain": (10, 12)},
    2: {"initial": 18, "after_10pct_gain": (12, 15), "after_20pct_gain": (8, 10)},
    3: {"initial": 12, "after_10pct_gain": (8, 10),  "after_20pct_gain": (5, 8)},
}
```

### 4.4 Maximum Loss Per Position

```python
MAX_LOSS = {
    "stock": {1: 1.1, 2: 0.72, 3: 0.36},   # % of total portfolio
    "etf":   {1: 1.76, 2: 1.44, 3: 0.72},
}
# At $1.5M portfolio: T1 stock max loss = ~$17K, T2 = ~$11K, T3 = ~$6K
```

### 4.5 Diversification Constraints

```python
DIVERSIFICATION = {
    "max_single_sector": 40,   # % of total portfolio
    "min_sectors": 8,
    "max_single_stock": 5,     # % of total portfolio
    "max_single_etf": 8,       # % of total portfolio
    "stock_etf_split": (50, 50),  # target 50/50, tolerance ±5%
}
```

### 4.6 Two Keep Categories

```python
KEEP_CATEGORIES = {
    "fundamental_review": {
        "description": "Retained based on PE analysis, forward earnings, analyst ratings",
        "stocks": {
            "UNH":  {"target": 30270, "reason": "PE 17.9 vs 22 historical; defensive quality"},
            "MU":   {"target": 25000, "reason": "Forward PE 12 vs trailing 32; AI memory leader"},
            "AMZN": {"target": 33591, "reason": "Forward PE 29; AWS cloud leader"},
            "MRK":  {"target": 20000, "reason": "PE 74% below 10-yr avg; Keytruda"},
            "COP":  {"target": 15000, "reason": "PE 13.5 vs sector 16.4; energy quality"},
        },
    },
    "ibd_high_rated": {
        "description": "Composite ≥ 93 AND RS ≥ 90 — elite momentum leaders",
        "stocks": {
            "CLS":  {"target": 16324, "comp": 99, "rs": 97},
            "AGI":  {"target": 16599, "comp": 99, "rs": 92},
            "TFPM": {"target": 10109, "comp": 99, "rs": 92},
            "GMED": {"target": 15588, "comp": 98, "rs": 91},
            "SCCO": {"target": 12448, "comp": 98, "rs": 94},
            "HWM":  {"target": 10914, "comp": 98, "rs": 90},
            "KLAC": {"target": 19600, "comp": 97, "rs": 94},
            "TPR":  {"target": 10614, "comp": 97, "rs": 93},
            "GS":   {"target": 19719, "comp": 93, "rs": 91},
        },
    },
}
# Total keeps: 5 + 9 = 14 positions pre-committed
```

### 4.7 Asset Type Targets

```python
ASSET_TARGETS = {
    "individual_stocks": {"target_pct": 50, "tolerance": 5},  # 45-55%
    "etfs":              {"target_pct": 50, "tolerance": 5},   # 45-55%
}
```

### 4.8 Reference: Current Portfolio (59 Positions)

**T1 Momentum (39%): 16 stocks + 5 ETFs = $587,915**
**T2 Quality (37%): 14 stocks + 9 ETFs = $562,591**
**T3 Defensive (22%): 8 stocks + 7 ETFs = $328,970**
**Cash: 2.8% = $41,834**

---

## 5. Autonomy & Decision-Making

### Decision 1: Handling Keep Constraints
```
14 positions are pre-committed keeps. 59 total target.
Remaining: 45 positions to fill from Analyst's top 50 + ETF universe.

Agent reasoning: "Keeps consume ~$406K of $1.52M (27%).
Remaining ~$1.1M for new/add positions.
Must ensure keeps don't violate tier allocation:
  IBD Keeps → mostly T1 (CLS, KLAC, AGI, etc.)
  Fundamental Keeps → split T2/T3 (UNH defensive, MU momentum)"
```

### Decision 2: Position Sizing Within Limits
```
NVDA in T1: target $35K out of $1.52M = 2.3%
Max for T1 stock = 5%. Current = 2.3%. ✅ Within limit.

Agent reasoning: "All positions sized within Framework limits.
Apply conviction-based sizing within the limit range:
  High conviction (9-10): Upper end of typical range
  Medium conviction (6-8): Middle of typical range
  Lower conviction (4-5): Lower end of typical range"
```

### Decision 3: Sector Concentration Check
```
Technology positions: NVDA + GOOGL + MU + CLS + PLTR + KLAC + AVGO + MSFT + CRM + CRWD + APP + SOXX + SMH
Estimated tech: ~26% of portfolio

Agent reasoning: "26% < 40% max. ✅ Compliant.
Flag for Risk Officer: check effective sector exposure including ETF holdings."
```

### Decision 4: Stop-Loss Assignment
```
NVDA: T1 stock, current price $X
  Initial trailing stop: 22%
  Stop price: $X × 0.78

UNH: T3 stock (Fundamental Keep), current price $Y
  Initial trailing stop: 12%
  Stop price: $Y × 0.88

Agent reasoning: "Stops are deterministic from tier assignment:
T1 = 22%, T2 = 18%, T3 = 12%.
These tighten as gains accumulate per the tightening protocol."
```

### Decision 5: Money Flow — Sell Orders
```
Agent reasoning: "Framework identified 36 sells (23 stocks, 11 ETFs, 2 mutual funds).
Must generate ordered sell list:
  Priority: Mutual funds first (ACAAX, FBGRX — liquidate entirely)
  Then: Stocks with no IBD ratings (CAT, IDXX, AMD, etc.)
  Then: ETFs that don't fit tier/theme strategy (VEA, ICLN, etc.)
Also generate buy orders by week:
  Week 1: Liquidation (~$807K)
  Week 2: T1 Momentum + Factors (~$337K)
  Week 3: T2 Quality Growth (~$259K)
  Week 4: T3 Defensive + Finalize (~$225K)"
```

---

## 6. Output Contract

### PortfolioOutput

```python
class PortfolioOutput(BaseModel):
    # Portfolio Tiers
    tier_1: TierPortfolio
    tier_2: TierPortfolio
    tier_3: TierPortfolio
    cash_reserve: CashAllocation
    # Orders
    sell_orders: List[Order]
    buy_orders: List[Order]
    trim_orders: List[Order]
    add_orders: List[Order]
    # Keeps Handling
    keeps_placement: KeepsPlacement
    # Summary
    total_positions: int                     # Target ~59
    stock_count: int
    etf_count: int
    sector_allocation: Dict[str, float]      # Sector → % of total
    implementation_timeline: ImplementationPlan
    construction_methodology: str
    deviation_notes: List[str]               # Where PM deviated from Strategist
    analysis_date: str
    summary: str

    # VALIDATORS:
    # - T1+T2+T3+Cash ≈ 100%
    # - No stock > 5% (T1), 4% (T2), 3% (T3)
    # - No ETF > 8% (T1), 8% (T2), 6% (T3)
    # - No sector > 40%
    # - At least 8 sectors
    # - Stock/ETF ratio within 45-55% / 55-45%
    # - All 14 keeps placed in appropriate tiers
```

### TierPortfolio

```python
class TierPortfolio(BaseModel):
    tier: int
    label: str                               # "Momentum" / "Quality Growth" / "Defensive"
    target_pct: float                        # e.g., 39.0
    actual_pct: float                        # Calculated
    positions: List[Position]
    total_value: float
    stock_count: int
    etf_count: int
```

### Position

```python
class Position(BaseModel):
    symbol: str
    name: str
    security_type: str                       # "stock" / "etf"
    tier: int
    sector: str
    # Sizing
    target_value: float                      # Dollar amount
    weight_pct: float                        # % of total portfolio
    # Ratings (if available)
    composite_rating: Optional[int]
    rs_rating: Optional[int]
    eps_rating: Optional[int]
    ibd_lists: List[str]
    # Keep Status
    keep_category: Optional[str]             # "fundamental" / "ibd" / None
    keep_reason: Optional[str]
    # Risk
    trailing_stop_pct: float                 # 22, 18, or 12
    max_loss_pct: float                      # Portfolio % max loss
    # Status
    order_action: str                        # "NEW BUY" / "ADD" / "HOLD" / "TRIM" / "IBD KEEP" / "FUND KEEP"
    conviction: Optional[int]                # 1-10 from Analyst
    reasoning: str                           # 30+ chars
```

### Order

```python
class Order(BaseModel):
    symbol: str
    action: str                              # "SELL" / "BUY" / "ADD" / "TRIM"
    amount: float                            # Dollar amount
    reason: str
    priority: int                            # 1 = highest
    week: int                                # Implementation week (1-4)
```

### KeepsPlacement

```python
class KeepsPlacement(BaseModel):
    fundamental_keeps: List[KeepDetail]      # 5 stocks
    ibd_keeps: List[KeepDetail]              # 9 stocks
    total_keeps: int                         # 14
    keeps_value: float                       # Total $ in keeps
    keeps_pct: float                         # % of portfolio in keeps
```

### KeepDetail

```python
class KeepDetail(BaseModel):
    symbol: str
    category: str                            # "fundamental" / "ibd"
    tier_placed: int                         # Which tier
    target_value: float
    reason: str
```

### ImplementationPlan

```python
class ImplementationPlan(BaseModel):
    week_1: WeekPlan                         # Liquidation
    week_2: WeekPlan                         # T1 Momentum
    week_3: WeekPlan                         # T2 Quality Growth
    week_4: WeekPlan                         # T3 Defensive + Finalize

class WeekPlan(BaseModel):
    week_number: int
    focus: str                               # "Liquidation", "T1 Momentum", etc.
    total_sells: float                       # $ amount
    total_buys: float
    orders: List[Order]
```

---

## 7. Testing Plan

### Level 1: Schema Tests

| Test | What It Validates |
|---|---|
| `test_tier_allocation_sums` | T1+T2+T3+Cash ≈ 100% |
| `test_t1_stock_max_5pct` | No T1 stock > 5% |
| `test_t2_stock_max_4pct` | No T2 stock > 4% |
| `test_t3_stock_max_3pct` | No T3 stock > 3% |
| `test_t1_etf_max_8pct` | No T1 ETF > 8% |
| `test_t3_etf_max_6pct` | No T3 ETF > 6% |
| `test_no_sector_over_40pct` | Max sector ≤ 40% |
| `test_min_8_sectors` | ≥ 8 sectors |
| `test_stock_etf_ratio` | Each within 45-55% |
| `test_trailing_stops_match_tier` | T1=22%, T2=18%, T3=12% |
| `test_all_14_keeps_placed` | All keeps appear in portfolio |
| `test_keep_categories_correct` | 5 fundamental + 9 IBD |
| `test_max_loss_limits` | T1 stock ≤ 1.1%, T2 ≤ 0.72%, T3 ≤ 0.36% |

### Level 2-4: LLM, behavioral (no stock discovery, no macro), integration with Strategist + Analyst

---

## 8. Integration Points

### Receives From
- **Analyst:** `rated_stocks` (top 50), `unrated_stocks`, `ibd_keeps`, `sector_rankings`
- **Strategist:** `SectorStrategyOutput` (allocations, themes, tier targets)

### Passes To
- **Risk Officer:** Full `PortfolioOutput` for review

### Feedback Loop
- **Risk Officer → PM:** Vetoes → PM must fix and resubmit

---

## 9. Files to Create

```
1. src/ibd_agents/schemas/portfolio_output.py
2. tests/unit/test_portfolio_schema.py
3. src/ibd_agents/tools/position_sizer.py          ← Sizing + stop calculator
4. src/ibd_agents/tools/keeps_manager.py            ← 2 keep categories
5. src/ibd_agents/tools/order_generator.py          ← Buy/sell/trim orders
6. src/ibd_agents/agents/portfolio_manager.py
7. config updates
8. tests/unit/test_portfolio_agent.py
9. tests/unit/test_portfolio_behavior.py
10. tests/integration/test_strategist_to_pm.py
11. golden_datasets/portfolio_golden.json
```