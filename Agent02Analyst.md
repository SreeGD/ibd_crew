# Agent 02: Analyst Agent 📊

> **Build Phase:** 2
> **Depends On:** Research Agent output
> **Feeds Into:** Rotation Detector, Sector Strategist, Portfolio Manager
> **Framework v4.0 Sections:** 2.3 Elite Screening, 2.4 Tiers, 2.5 Keep, 2.5 Sector Ranking

---

## 1. Identity

| Field | Value |
|---|---|
| **Role** | Quantitative Investment Analyst (CFA, IBD methodology expert) |
| **Temperature** | 0.3 (analytical precision — tier assignment is largely deterministic) |

---

## 2. Goal

> Apply Framework v4.0 **Elite Screening**, **Tier Assignment**, and **Sector Ranking** to Research Agent's stocks. Flag IBD Keeps. Calculate sector scores. Produce **top 50 stocks** with conviction, strengths, weaknesses, and catalysts.

---

## 3. ✅ DOES / ❌ DOES NOT

### ✅ DOES
- Apply all 4 elite screening filters (EPS≥85, RS≥75, SMR A/B/B-, Acc/Dis A/B/B-)
- Assign tiers (T1/T2/T3) using exact Framework thresholds
- Honor IBD Keep threshold — Comp≥93 AND RS≥90 → KEEP unless override
- Calculate sector scores using Framework formula
- Rank stocks within each sector (Composite → RS → EPS)
- Assign conviction (1-10) with CAN SLIM reasoning
- Identify strengths, weaknesses, catalyst per stock

### ❌ DOES NOT
- Discover new stocks (evaluates Research output only)
- Set position sizes, trailing stops, or portfolio weights
- Make sector allocation recommendations
- Apply keep categories (Fundamental/User/IBD) — that's PM
- Make buy/sell decisions

---

## 4. Framework v4.0 Rules (Hard-Coded)

### 4.1 Elite Screening — ALL 4 Must Pass

```python
ELITE_FILTERS = {
    "eps_rating":    {"min": 85,  "type": "numeric"},
    "rs_rating":     {"min": 75,  "type": "numeric"},
    "smr_rating":    {"allowed": ["A", "B", "B-"], "type": "letter"},
    "acc_dis_rating": {"allowed": ["A", "B", "B-"], "type": "letter"},
}

def passes_all_elite(stock) -> bool:
    """Stock must pass ALL 4 filters."""
    return (
        stock.eps_rating is not None and stock.eps_rating >= 85
        and stock.rs_rating is not None and stock.rs_rating >= 75
        and stock.smr_rating in ["A", "B", "B-"]
        and stock.acc_dis_rating in ["A", "B", "B-"]
    )
```

### 4.2 Tier Assignment — Deterministic

```python
def assign_tier(composite, rs, eps) -> Optional[int]:
    """Tier assignment is deterministic from Framework v4.0 thresholds."""
    if composite is None or rs is None or eps is None:
        return None  # Cannot assign without all 3 ratings
    if composite >= 95 and rs >= 90 and eps >= 80:
        return 1  # Momentum
    elif composite >= 85 and rs >= 80 and eps >= 75:
        return 2  # Quality Growth
    elif composite >= 80 and rs >= 75 and eps >= 70:
        return 3  # Defensive
    else:
        return None  # Below threshold
```

**Tier Assignment Table:**

| Tier | Composite | RS | EPS | Label | Target Return |
|---|---|---|---|---|---|
| T1 | ≥ 95 | ≥ 90 | ≥ 80 | Momentum | 25-30% |
| T2 | ≥ 85 | ≥ 80 | ≥ 75 | Quality Growth | 18-22% |
| T3 | ≥ 80 | ≥ 75 | ≥ 70 | Defensive | 12-15% |

### 4.3 IBD Keep Threshold

```python
def is_ibd_keep(composite, rs) -> bool:
    """Composite ≥ 93 AND RS ≥ 90 → KEEP unless other factors override."""
    return composite is not None and rs is not None and composite >= 93 and rs >= 90
```

**IBD Keeps from current portfolio (9 stocks):**
```
CLS (99/97), AGI (99/92), TFPM (99/92), GMED (98/91),
SCCO (98/94), HWM (98/90), KLAC (97/94), TPR (97/93), GS (93/91)
```

### 4.4 Sector Score Formula

```python
def sector_score(avg_composite, avg_rs, elite_pct, multi_list_pct) -> float:
    """
    Framework v4.0 sector scoring.
    elite_pct and multi_list_pct are 0-100 scale.
    """
    return (avg_composite * 0.25) + (avg_rs * 0.25) + (elite_pct * 0.30) + (multi_list_pct * 0.20)
```

### 4.5 Rank Within Sector

```
Primary:   Composite Rating (descending)
Secondary: RS Rating (descending)
Tertiary:  EPS Rating (descending)
```

---

## 5. Reference Data: Real IBD Ratings (23 Stocks)

These are actual ratings from the framework. Use as golden dataset reference.

| Symbol | Comp | RS | EPS | Lists | Correct Tier |
|---|---|---|---|---|---|
| MU | 99 | 99 | 81 | RS New High, Rising Est | T1 |
| CLS | 99 | 97 | 99 | Big Cap 20 | T1 |
| FIX | 99 | 96 | 99 | Big Cap 20, Sector Ldrs | T1 |
| KGC | 99 | 96 | 96 | Big Cap 20, Sector Ldrs | T1 |
| NEM | 99 | 95 | 91 | Rising Estimates | T1 |
| GOOGL | 99 | 94 | 92 | Big Cap 20, Rising Est | T1 |
| PLTR | 99 | 93 | 99 | Big Cap 20, Sector Ldrs | T1 |
| AEM | 99 | 92 | 98 | Big Cap 20, Sector Ldrs | T1 |
| AGI | 99 | 92 | 97 | Big Cap 20, Sector Ldrs | T1 |
| TFPM | 99 | 92 | 99 | Sector Leaders | T1 |
| RGLD | 99 | 90 | 95 | Big Cap 20 | T1* |
| GE | 99 | 89 | 98 | Big Cap 20, Rising Est | T2 |
| LLY | 99 | 86 | 98 | Big Cap 20, Sector Ldrs | T2 |
| ISRG | 99 | 80 | 96 | Big Cap 20 | T2** |
| APP | 98 | 95 | 80 | Rising Estimates | T1*** |
| SCCO | 98 | 94 | 79 | RS New High, Rising Est | T2**** |
| GMED | 98 | 91 | 95 | Sector Ldrs, Rising Est | T1 |
| HWM | 98 | 90 | 99 | RS New High | T1 |
| AVGO | 98 | 88 | 99 | Big Cap 20, Rising Est | T2 |
| NVDA | 98 | 81 | 99 | Sector Ldrs, Rising Est | T2 |
| KLAC | 97 | 94 | 96 | RS New High, Rising Est | T1 |
| TPR | 97 | 93 | 94 | Rising Estimates | T1 |
| GS | 93 | 91 | 76 | Rising Estimates | T2***** |

**Interesting edge cases for agent reasoning:**
- `*RGLD`: Comp 99, RS 90 — exactly on T1 boundary (RS=90 ≥ 90). Framework placed it T3 (defensive choice). Agent should note this discrepancy.
- `**ISRG`: Comp 99, RS 80 — T2 by ratings but placed in T3 (defensive). Agent should note.
- `***APP`: Comp 98, RS 95, but EPS only 80 — T1 by thresholds (98≥95, 95≥90, 80≥80).
- `****SCCO`: Comp 98, RS 94, but EPS 79 — Fails T1 (EPS 79 < 80). T2 correct.
- `*****GS`: Comp 93, RS 91, EPS 76 — Fails T1 (Comp 93 < 95). T2 correct (93≥85, 91≥80, 76≥75).

---

## 6. Autonomy & Decision-Making

### Decision 1: Tier vs Framework Placement Discrepancy
```
RGLD: Comp 99, RS 90, EPS 95 → Tier thresholds say T1
Framework portfolio says T3 (Defensive)

Agent reasoning: "By strict threshold rules, RGLD qualifies for T1
(99≥95, 90≥90, 95≥80). However, the framework placed it in T3.
This could be a portfolio construction decision (PM chose defensive
placement for diversification). As Analyst, I assign the THRESHOLD
tier (T1) and flag the discrepancy. Portfolio Manager can override
for strategic reasons."
```

### Decision 2: Missing Ratings
```
AMZN: No IBD ratings in our data (Fundamental Keep)

Agent reasoning: "Cannot apply elite screening or tier assignment
without Composite/RS/EPS ratings. Mark as 'unrated'.
AMZN is a Fundamental Keep with strong qualitative case.
Pass to PM with note: 'Requires external rating lookup or
manual assessment for tier placement.'"
```

### Decision 3: Conviction with CAN SLIM
```
FIX: Comp 99, RS 96, EPS 99, Big Cap 20 + Sector Leaders

Agent reasoning: "CAN SLIM Analysis:
  C (Current earnings): EPS 99 — top 1% ✅
  A (Annual growth): Sustained multi-year growth ✅
  N (New): Building sector benefiting from infrastructure spend ✅
  S (Supply/Demand): Acc/Dis rating would confirm (check if available)
  L (Leader): Sector rank likely #1 given 99/96/99 ratings ✅
  I (Institutional): On Big Cap 20 + Sector Leaders = institutional attention ✅
  M (Market): Current regime assessment needed from Detector

Conviction: 10/10 — near-perfect ratings, multi-list validation, leading sector"
```

### Decision 4: Sector Ranking Calculation
```
Sector MINING: 6 stocks: NEM(99/95), KGC(99/96), AEM(99/92), AGI(99/92), TFPM(99/92), SCCO(98/94)
  Avg Composite: 98.83
  Avg RS: 93.50
  Elite%: 5/6 = 83.3% (SCCO EPS 79 < 85 may fail elite EPS filter)
  Multi-list%: 4/6 = 66.7% (on 2+ IBD lists)

  Sector Score = (98.83×0.25) + (93.50×0.25) + (83.3×0.30) + (66.7×0.20)
               = 24.71 + 23.38 + 24.99 + 13.34 = 86.42

This would be one of the highest sector scores.
```

---

## 7. Output Contract

### AnalystOutput

```python
class AnalystOutput(BaseModel):
    rated_stocks: List[RatedStock]           # Top 50 with full analysis
    unrated_stocks: List[UnratedStock]       # Stocks missing ratings
    sector_rankings: List[SectorRank]        # All sectors scored
    ibd_keeps: List[IBDKeep]                 # Comp≥93 AND RS≥90
    elite_filter_summary: EliteFilterSummary
    tier_distribution: TierDistribution
    methodology_notes: str
    analysis_date: str
    summary: str                             # 50+ chars
```

### RatedStock

```python
class RatedStock(BaseModel):
    symbol: str
    company_name: str
    sector: str                              # IBD sector
    security_type: Literal["stock"]
    # Tier Assignment (deterministic)
    tier: int                                # 1, 2, or 3
    tier_label: str                          # "Momentum" / "Quality Growth" / "Defensive"
    # IBD Ratings
    composite_rating: int                    # 1-99 (required for rated stocks)
    rs_rating: int                           # 1-99
    eps_rating: int                          # 1-99
    smr_rating: Optional[str]                # A/B/B-/C/D/E
    acc_dis_rating: Optional[str]            # A/B/B-/C/D/E
    # Elite Filter Results
    passes_eps_filter: bool                  # EPS ≥ 85
    passes_rs_filter: bool                   # RS ≥ 75
    passes_smr_filter: Optional[bool]        # SMR = A/B/B-
    passes_acc_dis_filter: Optional[bool]    # Acc/Dis = A/B/B-
    passes_all_elite: bool                   # All applicable pass
    # Flags
    is_ibd_keep: bool                        # Comp≥93 AND RS≥90
    is_multi_source_validated: bool          # From Research
    # From Research
    ibd_lists: List[str]
    schwab_themes: List[str]
    validation_score: int
    # Analyst Assessment
    conviction: int                          # 1-10
    strengths: List[str]                     # 1-5 items
    weaknesses: List[str]                    # 1-5 items
    catalyst: str
    reasoning: str                           # 50+ chars
    sector_rank: int                         # Rank within sector
```

### UnratedStock

```python
class UnratedStock(BaseModel):
    symbol: str
    company_name: str
    sector: Optional[str]
    reason_unrated: str                      # "No IBD ratings available"
    schwab_themes: List[str]
    validation_score: int
    sources: List[str]
    note: str                                # What PM should know
```

### SectorRank

```python
class SectorRank(BaseModel):
    rank: int                                # 1 = strongest
    sector: str                              # IBD sector
    stock_count: int
    avg_composite: float
    avg_rs: float
    elite_pct: float                         # 0-100
    multi_list_pct: float                    # 0-100
    sector_score: float                      # Formula result
    top_stocks: List[str]                    # Top 3-5 symbols
    ibd_keep_count: int                      # Keeps in this sector
```

### IBDKeep

```python
class IBDKeep(BaseModel):
    symbol: str
    composite_rating: int                    # ≥ 93
    rs_rating: int                           # ≥ 90
    eps_rating: int
    ibd_lists: List[str]
    tier: int
    keep_rationale: str
    override_risk: Optional[str]             # Any reason to question
```

### TierDistribution

```python
class TierDistribution(BaseModel):
    tier_1_count: int
    tier_2_count: int
    tier_3_count: int
    below_threshold_count: int
    unrated_count: int
```

### EliteFilterSummary

```python
class EliteFilterSummary(BaseModel):
    total_screened: int
    passed_all_four: int
    failed_eps: int
    failed_rs: int
    failed_smr: int
    failed_acc_dis: int
    missing_ratings: int                     # Couldn't apply filters
```

---

## 8. Testing Plan

### Level 1: Schema Tests (No LLM — deterministic verification)

| Test | What It Validates |
|---|---|
| `test_tier_1_threshold_exact` | Comp=95, RS=90, EPS=80 → T1 |
| `test_tier_1_below_composite` | Comp=94, RS=90, EPS=80 → NOT T1 |
| `test_tier_1_below_rs` | Comp=95, RS=89, EPS=80 → NOT T1 |
| `test_tier_2_threshold_exact` | Comp=85, RS=80, EPS=75 → T2 |
| `test_tier_3_threshold_exact` | Comp=80, RS=75, EPS=70 → T3 |
| `test_below_all_thresholds` | Comp=79 → None |
| `test_ibd_keep_boundary` | Comp=93, RS=90 → keep=True |
| `test_ibd_keep_miss_by_one` | Comp=92, RS=90 → keep=False |
| `test_elite_eps_filter` | EPS=84 → fails |
| `test_elite_smr_filter` | SMR="C" → fails |
| `test_elite_all_pass` | All 4 above threshold → passes_all=True |
| `test_sector_score_formula` | Known inputs → known result |
| `test_sector_ranking_order` | Rank 1 has highest sector_score |
| `test_conviction_range` | 1-10 |
| `test_real_data_MU` | MU(99/99/81) → T1, keep=True |
| `test_real_data_SCCO` | SCCO(98/94/79) → T2 (EPS 79 < 80) |
| `test_real_data_GS` | GS(93/91/76) → T2, keep=True |

---

## 9. Integration Points

### Receives From Research
- `stocks` (100-200 with ratings, validation scores, themes)
- `sector_patterns`, `ibd_keep_candidates`, `multi_source_validated`

### Passes To Rotation Detector
- `rated_stocks` (with sectors, ratings for per-sector RS analysis)
- `sector_rankings` (for rotation signal detection)

### Passes To Sector Strategist
- `sector_rankings`, `rated_stocks`, `tier_distribution`

### Passes To Portfolio Manager
- `rated_stocks` (top 50), `unrated_stocks`, `ibd_keeps`
- `sector_rankings`, `tier_distribution`

---

## 10. Files to Create

```
1. src/ibd_agents/schemas/analyst_output.py
2. tests/unit/test_analyst_schema.py
3. src/ibd_agents/tools/elite_screener.py        ← 4 filters + tier assignment
4. src/ibd_agents/tools/sector_scorer.py          ← Sector score formula
5. src/ibd_agents/agents/analyst_agent.py
6. config/agents.yaml, config/tasks.yaml
7. tests/unit/test_analyst_agent.py
8. tests/unit/test_analyst_behavior.py
9. tests/integration/test_research_to_analyst.py
10. golden_datasets/analyst_golden.json           ← Use 23 real ratings
```