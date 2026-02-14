# Agent 04: Sector Strategist 🎯

> **Build Phase:** 4
> **Depends On:** Rotation Detector + Analyst Agent
> **Feeds Into:** Portfolio Manager
> **Framework v4.0 Sections:** 2.8 Sector Score, 2.10 Portfolio Architecture, 2.7 Themes, 6.5 Regime

---

## 1. Identity

| Field | Value |
|---|---|
| **Role** | Sector Rotation & Allocation Specialist |
| **Experience** | 20+ years, traded through multiple cycles |
| **Temperature** | 0.5 (balanced: analytical + predictive judgment) |

---

## 2. Goal

> Using the Rotation Detector's verdict and Analyst's sector rankings, produce **sector allocation recommendations** for the 3-tier portfolio architecture, incorporating Schwab thematic opportunities and adjusting for market regime.

---

## 3. ✅ DOES / ❌ DOES NOT

### ✅ DOES
- Take Rotation Detector verdict and translate it into allocation action
- Recommend sector % allocations for T1, T2, T3 tiers
- Identify Schwab thematic opportunities (21 themes, key ETFs)
- Adjust allocation for market regime (bull/bear from Detector)
- Ensure all allocations respect 40% single-sector max
- Provide rotation timing signals with confirmation/invalidation criteria
- Recommend which Schwab theme ETFs fit each tier

### ❌ DOES NOT
- Pick individual stocks (sectors and themes only)
- Set specific position sizes (that's PM)
- Detect whether rotation is happening (that's Detector)
- Set trailing stops or risk limits (that's Risk Officer)
- Override Portfolio Manager

---

## 4. Framework v4.0 Rules

### 4.1 Portfolio Architecture Targets (Moderate Profile)

```python
TIER_ALLOCATION_TARGETS = {
    "tier_1_momentum":   {"target_pct": 39, "range": (35, 40)},
    "tier_2_quality":    {"target_pct": 37, "range": (33, 40)},
    "tier_3_defensive":  {"target_pct": 22, "range": (20, 30)},
    "cash":              {"target_pct": 2,  "range": (2, 5)},
}
```

### 4.2 Sector Score Formula (from Framework v4.0 §2.8)

```python
def sector_score(avg_composite, avg_rs, elite_pct, multi_list_pct):
    """Calculate sector strength score for ranking."""
    return (avg_composite * 0.25) + (avg_rs * 0.25) + (elite_pct * 0.30) + (multi_list_pct * 0.20)

# Where:
# avg_composite = Average Composite Rating of all stocks in sector
# avg_rs = Average Relative Strength Rating
# elite_pct = % of stocks passing Elite criteria (normalized to 100)
# multi_list_pct = % of stocks on 2+ IBD lists (normalized to 100)
```

### 4.3 28 IBD Sector Categories

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

### 4.4 Sector Concentration Limits

```python
SECTOR_LIMITS = {
    "max_single_sector": 40,   # % of total portfolio
    "min_sectors": 8,          # Minimum sectors represented
}
```

### 4.5 Schwab Theme → ETF Mapping

```python
THEME_ETFS = {
    # T1/T2 candidates (growth/momentum themes)
    "Artificial Intelligence":    ["BOTZ", "ROBO", "AIQ"],
    "Robotics & Automation":      ["ROBO", "BOTZ"],
    "Cybersecurity":              ["CIBR", "HACK", "BUG"],
    "Big Data & Analytics":       ["CLOU", "WCLD"],
    "E-Commerce":                 ["IBUY", "ONLN"],
    "Digital Payments":           ["IPAY", "FINX"],
    "Space Economy":              ["UFO", "ARKX"],
    "Defense & Aerospace":        ["ITA", "PPA", "XAR"],
    "Electric Vehicles":          ["DRIV", "IDRV", "LIT"],
    "China Growth":               ["MCHI", "FXI", "KWEB"],
    # T2/T3 candidates (quality/defensive themes)
    "Healthcare Innovation":      ["XBI", "IBB", "ARKG"],
    "Aging Population":           [],
    "Renewable Energy":           ["ICLN", "TAN", "QCLN"],
    "Climate Change":             [],
    "Cannabis":                   ["MJ", "MSOS"],
    "Blockchain/Crypto":          ["BLOK", "BITO"],
}
```

### 4.6 Regime-Adjusted Strategy

```python
REGIME_ACTIONS = {
    "bull": {
        "tier_1_bias": +5,      # Add 5% to T1 target
        "tier_3_bias": -5,      # Reduce T3 target
        "equity_target": "90-100%",
        "action": "Scale into T1 from IBD 50/Sector Leaders, add momentum ETFs",
    },
    "bear": {
        "tier_1_bias": -10,     # Reduce T1 target
        "tier_3_bias": +10,     # Increase T3 target
        "equity_target": "20-40%",
        "action": "Rotate to T3 focus, raise cash, tighten T1 stops to 15-18%",
    },
    "neutral": {
        "tier_1_bias": 0,
        "tier_3_bias": 0,
        "equity_target": "70-90%",
        "action": "Maintain target allocations (39/37/22)",
    },
}
```

### 4.7 How Strategist Uses Detector Verdict

```
IF verdict == ACTIVE:
    → Build rotation-adjusted allocation
    → Overweight destination sectors, underweight source sectors
    → Recommend specific theme ETFs aligned with rotation direction

IF verdict == EMERGING:
    → Prepare contingency allocation (smaller shifts)
    → Define confirmation triggers

IF verdict == NONE:
    → Maintain current sector outlook
    → Optimize within existing leadership
```

---

## 5. Autonomy & Decision-Making

### Decision 1: Active Rotation + Bull Regime
```
Detector says: ACTIVE cyclical rotation, bull regime
Source: Technology/Software, Destination: Mining/Energy

Agent reasoning: "Bull regime + active rotation = aggressive allocation shift.
T1 (39% + 5% bull bias = 44%): Heavy in MINING, ENERGY destination sectors
T2 (37%): Quality growth across diversified sectors
T3 (22% - 5% bull bias = 17%): Minimal defensive

SECTOR ALLOCATION:
  Mining: 18% (up from 10%) — destination sector, bull regime
  Technology: 22% (down from 28%) — source sector, still strong absolute
  Healthcare: 12% — stable, quality anchor
  Financial: 10% — stable
  Aerospace: 8% — thematic (Defense & Space)
  Energy: 5% — emerging destination
  Others: 25% diversified
  
THEME ETF RECOMMENDATIONS:
  T1: GDX (Gold Miners), GDXJ (Junior Miners), COPX (Copper)
  T2: ITA (Aerospace/Defense), CIBR (Cybersecurity), LIT (EV/Lithium)
  T3: XBI (Biotech), IBB (Biotech Large), QUAL (Quality Factor)"
```

### Decision 2: Bear Market — Defensive Shift
```
Detector says: ACTIVE defensive rotation, bear regime
Distribution days = 6, breadth = 35%

Agent reasoning: "Framework v4.0 bear protocol: reduce equity to 20-40%.
T1 (-10% bear bias = 29%): Only highest conviction momentum
T2 (37%): Quality names that can weather storm
T3 (+10% bear bias = 32%): Max defensive allocation

SECTOR REALLOCATION:
  Healthcare: 18% (up — defensive haven)
  Utilities: 8% (up — XLU)
  Consumer Staples: 6% (up — XLP)
  Technology: 15% (down — trim source sector)
  Mining: 10% (keep some gold as hedge)
  Cash: 10-15% (raise from 2.8%)

THEME ETF RECOMMENDATIONS:
  T3 focus: SPLV (Low Vol), USMV (Min Vol), QUAL (Quality Factor)
  Trim: Growth-oriented theme ETFs"
```

### Decision 3: No Rotation — Optimize Existing
```
Detector says: NONE, neutral regime

Agent reasoning: "No rotation signals. Maintain current allocation.
Focus on sector optimization within existing leadership.
Current allocation close to targets: 39/37/22 is fine.

ONLY ADJUSTMENT: Check if Technology at 28% is optimal
given strong CHIPS and COMPUTER sectors. Could shift 2-3%
to underrepresented sectors like BUILDING (FIX is top-rated)."
```

---

## 6. Output Contract

### SectorStrategyOutput

```python
class SectorStrategyOutput(BaseModel):
    # Rotation Response
    rotation_response: str                   # What action taken based on detector verdict
    regime_adjustment: str                   # How market regime affected allocation
    # Sector Allocations
    sector_allocations: SectorAllocationPlan
    # Thematic Opportunities
    theme_recommendations: List[ThemeRecommendation]
    # Timing
    rotation_signals: List[RotationActionSignal] # Confirmation/invalidation triggers
    # Context
    analysis_date: str
    summary: str

    # VALIDATORS:
    # - All sector allocations sum to 95-105% per tier
    # - No single sector > 40%
    # - At least 8 sectors represented
```

### SectorAllocationPlan

```python
class SectorAllocationPlan(BaseModel):
    tier_1_allocation: Dict[str, float]      # Sector → % of T1 bucket
    tier_2_allocation: Dict[str, float]      # Sector → % of T2 bucket
    tier_3_allocation: Dict[str, float]      # Sector → % of T3 bucket
    overall_allocation: Dict[str, float]     # Sector → % of total portfolio
    tier_targets: Dict[str, float]           # {"T1": 39, "T2": 37, "T3": 22, "Cash": 2}
    cash_recommendation: float               # 2-15% depending on regime
    rationale: str
```

### ThemeRecommendation

```python
class ThemeRecommendation(BaseModel):
    theme: str                               # Schwab theme name
    recommended_etfs: List[str]              # Key ETFs for this theme
    tier_fit: int                            # 1, 2, or 3
    allocation_suggestion: str               # "3-5% of T1 bucket"
    conviction: str                          # "HIGH" / "MEDIUM" / "LOW"
    rationale: str
```

### RotationActionSignal

```python
class RotationActionSignal(BaseModel):
    action: str                              # What to do
    trigger: str                             # When to do it
    confirmation: List[str]                  # Proves right
    invalidation: List[str]                  # Proves wrong
```

---

## 7. Testing Plan

### Level 1: Schema Tests

| Test | What It Validates |
|---|---|
| `test_no_sector_exceeds_40pct` | Max single sector ≤ 40% |
| `test_min_8_sectors` | At least 8 sectors in overall allocation |
| `test_tier_targets_sum` | T1+T2+T3+Cash ≈ 100% |
| `test_tier_1_range` | T1 target between 25-45% |
| `test_tier_3_range` | T3 target between 15-35% |
| `test_cash_range` | Cash 2-15% |
| `test_theme_etfs_valid` | All recommended ETFs exist in theme mapping |
| `test_theme_tier_fit` | Growth themes → T1/T2, Defensive themes → T2/T3 |

### Level 2-4: LLM output, behavioral (no stock picks, no position sizes), integration with Detector

---

## 8. Integration Points

### Receives From
- **Rotation Detector:** Full `RotationDetectionOutput` (verdict, regime, signals, sectors)
- **Analyst Agent:** `sector_rankings`, `rated_stocks`, `tier_distribution`

### Passes To
- **Portfolio Manager:** `SectorStrategyOutput` (allocations, themes, targets)

---

## 9. Files to Create

```
1. src/ibd_agents/schemas/strategy_output.py
2. tests/unit/test_strategy_schema.py
3. src/ibd_agents/tools/sector_allocator.py       ← Allocation calculator
4. src/ibd_agents/tools/theme_mapper.py            ← Theme → ETF mapping
5. src/ibd_agents/agents/sector_strategist.py
6. config updates
7. tests/unit/test_strategist_agent.py
8. tests/unit/test_strategist_behavior.py
9. tests/integration/test_rotation_to_strategist.py
10. golden_datasets/strategy_golden.json
```