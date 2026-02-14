# Agent 03: Rotation Detector 🔄

> **Build Phase:** 3
> **Depends On:** Research Agent + Analyst Agent
> **Feeds Into:** Sector Strategist, Risk Officer
> **Framework v4.0 Sections:** 2.9 Sectors, 6.5 Market Regime, 6.1 Sector Breadth

---

## 1. Identity

| Field | Value |
|---|---|
| **Role** | Sector Rotation Detection Specialist |
| **Temperature** | 0.2 (factual detection, not creativity) |

---

## 2. Goal

> Analyze sector momentum using Analyst's sector rankings and stock data to determine whether **sector rotation is currently in effect**. Classify rotation type and stage. Provide objective signals for the Strategist. This agent is the **smoke detector** — it detects the fire, it doesn't fight it.

---

## 3. ✅ DOES / ❌ DOES NOT

### ✅ DOES
- Calculate per-sector momentum metrics (avg RS, avg Composite, breadth)
- Detect leadership changes across 28 IBD sectors
- Apply 5-signal detection framework to determine rotation status
- Integrate Framework v4.0 market regime signals (bull/bear indicators)
- Classify rotation type (Cyclical/Defensive/Thematic/Broad)
- Provide binary verdict: ACTIVE / EMERGING / NONE with evidence
- Feed objective signals to Sector Strategist

### ❌ DOES NOT
- Recommend allocations or portfolio changes
- Pick individual stocks
- Predict how long rotation lasts (only detects current state)
- Make buy/sell recommendations
- Set trailing stops or position sizes

---

## 4. Framework v4.0 Rules

### 4.1 Market Regime Signals (From Section 6.5)

```python
BULL_SIGNALS = {
    "market_pulse": "Confirmed uptrend",
    "distribution_days": "< 4 on major indices",
    "sector_breadth": "> 70% in uptrends",
    "ibd_50_status": "Showing volume breakouts",
    "golden_cross": "50-day above 200-day MA",
    "rsi": "Recovery from oversold (<30 to >50)",
}

BEAR_SIGNALS = {
    "market_pulse": "Under pressure or correction",
    "distribution_days": "> 5 on major indices",
    "sector_breadth": "< 40% in uptrends",
    "ibd_50_status": "Breaking below 50-day MA",
    "death_cross": "50-day below 200-day MA",
    "rsi": "> 70 with negative divergences",
}
```

### 4.2 Sector Breadth Thresholds (From Framework)

```python
BREADTH_THRESHOLDS = {
    "bullish": 70,     # > 70% = bull market breadth
    "neutral_high": 60,
    "neutral_low": 40,
    "bearish": 40,     # < 40% = bear market breadth
}
```

### 4.3 Sector Categories for Rotation Analysis

Use the 28 IBD sectors, but group into rotation-relevant clusters:

```python
SECTOR_CLUSTERS = {
    "growth": ["CHIPS", "COMPUTER", "SOFTWARE", "INTERNET", "ELECTRONICS"],
    "commodity": ["MINING", "ENERGY", "CHEMICALS"],
    "defensive": ["HEALTHCARE", "MEDICAL", "UTILITIES", "FOOD/BEVERAGE", "CONSUMER"],
    "financial": ["BANKS", "FINANCE", "INSURANCE", "REAL ESTATE"],
    "industrial": ["AEROSPACE", "BUILDING", "MACHINERY", "TRANSPORTATION", "DEFENSE"],
    "consumer_cyclical": ["RETAIL", "LEISURE", "MEDIA", "AUTO"],
    "services": ["BUSINESS SERVICES", "TELECOM"],
}
```

---

## 5. 5-Signal Detection Framework

```
SIGNAL 1: RS DIVERGENCE
  Measure: Avg RS of top 3 sectors vs bottom 3 sectors
  Threshold: Gap widening by 5+ points vs prior period
  Weight: HIGH

SIGNAL 2: LEADERSHIP CHANGE
  Measure: Sector rank shifts of 3+ positions
  Threshold: 2+ sectors moved 3+ ranks
  Weight: HIGH

SIGNAL 3: BREADTH SHIFT
  Measure: Per-sector breadth (% stocks with rising RS)
  Threshold: Source sector breadth < 40%, destination > 60%
  Weight: MEDIUM

SIGNAL 4: ELITE CONCENTRATION SHIFT
  Measure: Where are elite-qualifying stocks clustering?
  Threshold: Elite% shifting between sector clusters
  Weight: MEDIUM (uses Framework elite criteria)

SIGNAL 5: IBD KEEP MIGRATION
  Measure: Which sectors contain IBD Keep candidates (Comp≥93, RS≥90)?
  Threshold: Keeps migrating to new sectors vs prior period
  Weight: LOW-MEDIUM

ROTATION VERDICT:
  3-5 signals ON → ACTIVE
  2 signals ON   → EMERGING
  0-1 signals ON → NONE
```

---

## 6. Autonomy & Decision-Making

### Decision 1: Clear Cyclical Rotation
```
PRIOR: Growth cluster (CHIPS, COMPUTER, SOFTWARE) ranked #1-3
CURRENT: Commodity cluster (MINING, ENERGY) surged to #1-2

Agent reasoning: "RS Divergence ✅ (gap widened 8 points)
Leadership Change ✅ (MINING went from #6 to #1, ENERGY #8 to #3)
Breadth Shift ✅ (MINING breadth 35%→78%, SOFTWARE breadth 65%→38%)
Elite Concentration ✅ (Elite% in MINING now 87%, SOFTWARE dropped to 25%)
IBD Keep Migration ✅ (6 of 9 IBD Keeps now in commodity sectors)

VERDICT: ACTIVE (5/5 signals). Cyclical rotation confirmed.
SOURCE: Growth cluster. DESTINATION: Commodity cluster."
```

### Decision 2: Defensive Rotation with Market Regime
```
Market showing bear signals: distribution days = 6, breadth = 38%
Healthcare and Utilities rising, Technology declining

Agent reasoning: "Market regime: BEAR signals present.
Framework v4.0 says distribution days > 5 = bear warning.
Sector breadth 38% < 40% threshold = bearish.

RS Divergence ✅ (defensive sectors rising, growth falling)
Leadership Change ✅ (UTILITIES jumped 8 ranks)
Breadth Shift ✅
Elite Concentration ❌ (Elite stocks still mostly in growth — lagging indicator)
IBD Keep Migration ❌ (Keeps haven't shifted yet)

VERDICT: ACTIVE (3/5). Defensive rotation.
NOTE: Market regime supports rotation thesis. Framework says
rotate to T3 focus, reduce equity to 20-40%, raise cash."
```

### Decision 3: Thematic Rotation (AI theme)
```
AI-related stocks rising across CHIPS, COMPUTER, BUILDING (data centers)
Non-AI stocks in same sectors declining

Agent reasoning: "Traditional sector-level analysis shows:
  CHIPS: RS stable. COMPUTER: RS stable. BUILDING: RS rising.
  Traditional signals: 1/5 → would say NONE.

BUT within sectors, AI stocks outperform non-AI by 15+ RS points.
This is a THEMATIC rotation, not a sector rotation.

VERDICT: EMERGING (thematic). Type: THEMATIC.
Standard sector analysis understates this shift.
NOTE TO STRATEGIST: Consider Schwab theme analysis for AI,
Robotics, Cybersecurity themes rather than pure sector allocation."
```

### Decision 4: No Rotation — Stable Market
```
All sector ranks within ±1 position of prior period
No breadth shifts, no elite concentration change

Agent reasoning: "0/5 signals triggered.
VERDICT: NONE (90% confidence). Market leadership stable.
STRATEGIST: Current allocations remain valid."
```

---

## 7. Output Contract

### RotationDetectionOutput

```python
class RotationStatus(str, Enum):
    ACTIVE = "active"
    EMERGING = "emerging"
    NONE = "none"

class RotationType(str, Enum):
    CYCLICAL = "cyclical"
    DEFENSIVE = "defensive"
    OFFENSIVE = "offensive"
    THEMATIC = "thematic"
    BROAD = "broad"
    REVERSAL = "reversal"
    NONE = "none"

class RotationStage(str, Enum):
    EARLY = "early"
    MID = "mid"
    LATE = "late"
    EXHAUSTING = "exhausting"

class RotationDetectionOutput(BaseModel):
    verdict: RotationStatus
    confidence: int                          # 0-100
    rotation_type: Optional[RotationType]
    rotation_stage: Optional[RotationStage]
    # Market Regime Integration
    market_regime: MarketRegime
    # 5 Signals
    signals: RotationSignals
    # Sector Flows
    source_sectors: List[SectorFlow]         # Losing leadership
    destination_sectors: List[SectorFlow]    # Gaining leadership
    stable_sectors: List[str]
    # Context
    velocity: Optional[str]                  # "Slow" / "Moderate" / "Fast"
    strategist_notes: List[str]              # Guidance for Strategist
    analysis_date: str
    summary: str                             # 50+ chars

    # VALIDATORS:
    # - ACTIVE requires confidence ≥ 50 AND signals_active ≥ 3
    # - EMERGING requires signals_active == 2
    # - NONE requires signals_active ≤ 1
```

### MarketRegime

```python
class MarketRegime(BaseModel):
    regime: str                              # "bull" / "bear" / "neutral"
    bull_signals_present: List[str]          # Which bull signals are active
    bear_signals_present: List[str]          # Which bear signals are active
    sector_breadth_pct: Optional[float]      # Overall market breadth
    distribution_day_count: Optional[int]    # If known
    regime_note: str                         # Context for strategist
```

### RotationSignals

```python
class SignalReading(BaseModel):
    signal_name: str
    triggered: bool
    value: str
    threshold: str
    evidence: str

class RotationSignals(BaseModel):
    rs_divergence: SignalReading
    leadership_change: SignalReading
    breadth_shift: SignalReading
    elite_concentration_shift: SignalReading
    ibd_keep_migration: SignalReading
    signals_active: int                      # 0-5
```

### SectorFlow

```python
class SectorFlow(BaseModel):
    sector: str                              # IBD sector
    cluster: str                             # growth/commodity/defensive/financial/industrial
    direction: str                           # "outflow" / "inflow"
    current_rank: int
    prior_rank: Optional[int]
    avg_rs: float
    rs_change: Optional[float]
    elite_pct: float
    stock_count: int
    magnitude: str                           # "Strong" / "Moderate" / "Weak"
    evidence: str
```

---

## 8. Testing Plan

### Level 1: Schema Tests

| Test | What It Validates |
|---|---|
| `test_active_requires_3_signals` | verdict=ACTIVE → signals_active ≥ 3 |
| `test_emerging_requires_2_signals` | verdict=EMERGING → signals_active == 2 |
| `test_none_requires_1_or_fewer` | verdict=NONE → signals_active ≤ 1 |
| `test_active_requires_confidence_50` | ACTIVE → confidence ≥ 50 |
| `test_all_5_signals_present` | All signal readings exist |
| `test_signals_active_matches_count` | Sum of triggered == signals_active |
| `test_source_sectors_have_outflow` | direction == "outflow" |
| `test_dest_sectors_have_inflow` | direction == "inflow" |
| `test_valid_cluster_names` | Only valid cluster values |
| `test_market_regime_valid` | Only "bull"/"bear"/"neutral" |
| `test_breadth_threshold_bull` | breadth > 70 noted as bull |
| `test_breadth_threshold_bear` | breadth < 40 noted as bear |

### Level 2-4: Same structure (LLM, behavioral, integration)
- Behavioral: no allocation recs, no stock picks, no duration predictions
- Integration: accepts Analyst output, sectors match data

---

## 9. Integration Points

### Receives From
- **Research Agent:** `sector_patterns`, `stocks` (per-sector RS data)
- **Analyst Agent:** `sector_rankings`, `rated_stocks`, `ibd_keeps`

### Passes To
- **Sector Strategist:** Full `RotationDetectionOutput` — verdict + regime + signals
- **Risk Officer:** `market_regime` (bear signals = risk escalation)
- **Educator:** Rotation verdict for explanation

---

## 10. Files to Create

```
1. src/ibd_agents/schemas/rotation_output.py
2. tests/unit/test_rotation_schema.py
3. src/ibd_agents/tools/rotation_signals.py        ← 5-signal detection
4. src/ibd_agents/tools/market_regime.py            ← Bull/bear detection
5. src/ibd_agents/agents/rotation_detector.py
6. config updates
7. tests/unit/test_rotation_agent.py
8. tests/unit/test_rotation_behavior.py
9. tests/integration/test_analyst_to_rotation.py
10. golden_datasets/rotation_golden_active.json
11. golden_datasets/rotation_golden_none.json
```