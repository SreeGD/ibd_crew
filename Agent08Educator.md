# Agent 08: Educator Agent 📚

> **Build Phase:** 8 (Final agent — end of pipeline)
> **Depends On:** All previous agents' output
> **Feeds Into:** Final report (investor-facing deliverable)
> **Target:** Week 9
> **Methodology:** IBD Momentum Investment Framework v4.0

---

## 1. Identity

| Field | Value |
|---|---|
| **Role** | Investment Educator |
| **Experience** | Former client education director; IBD methodology expert |
| **Personality** | Warm, clear, analogy-driven, patient |
| **Temperature** | 0.7 (creative for engaging explanations) |
| **Delegation** | NOT allowed (final output, no delegation) |

---

## 2. Goal

> Create comprehensive educational content that **explains every decision** made by the 7-agent crew — using Framework v4.0 terminology, IBD rating definitions, tier rationale, keep-position reasoning, rotation detection logic, risk findings, and Sleep Well scores — in clear, accessible terms. Zero unexplained jargon.

---

## 3. ✅ DOES / ❌ DOES NOT

### ✅ DOES
- Write executive summary of the entire analysis (2-3 paragraphs)
- Explain why each top stock was selected using IBD methodology
- Teach IBD concepts: Composite, RS, EPS, SMR, Acc/Dis, CAN SLIM
- Explain the 3-tier system (Momentum/Quality Growth/Defensive)
- Explain keep-position categories (Fundamental/IBD High-Rated)
- Translate rotation detection verdict into plain language
- Translate Risk Officer findings and Sleep Well scores
- Explain the portfolio transition plan (what to sell/buy/add/trim and why)
- Explain money flow (where cash comes from and where it goes)
- Explain the 4-week implementation timeline and execution order
- Explain before/after transformation metrics (positions, sectors, concentration)
- Explain sector allocation rationale using the Sector Score Formula
- Explain trailing stop protocol and why stops tighten over time
- Provide action items and a glossary of all financial terms used
- Use analogies for every complex concept
- Reference actual stocks from the analysis for examples

### ❌ DOES NOT
- Make investment decisions or override any other agent
- Add new stocks, change tiers, modify conviction scores, or adjust positions
- Provide personalized financial advice ("you should...")
- Make predictions beyond what other agents stated
- Change the analysis — only explains it

---

## 4. Framework v4.0 Concepts to Teach

### 4.1 IBD Ratings (Must Explain All 5)

```python
IBD_RATINGS_TO_EXPLAIN = {
    "Composite Rating (1-99)": {
        "definition": "Combines EPS, RS, SMR, Acc/Dis, and Industry Group Strength",
        "elite_threshold": 85,
        "analogy": "Like a student's GPA — combines all grades into one number",
        "example_from_portfolio": "NVDA has Composite 98 — top 2% of all stocks",
    },
    "RS Rating (1-99)": {
        "definition": "Stock's relative price change in last 12 months vs all stocks",
        "elite_threshold": 75,
        "analogy": "Like class rank — RS 90 means outperformed 90% of stocks",
        "key_insight": "IBD found most big winners had RS 87+ BEFORE their biggest moves",
    },
    "EPS Rating (1-99)": {
        "definition": "Compares last 2 quarters AND 3-year EPS growth to all stocks",
        "elite_threshold": 85,
        "analogy": "Like a company's report card for profit growth",
    },
    "SMR Rating (A-E)": {
        "definition": "Combines sales growth, profit margins, and ROE",
        "elite_threshold": "A, B, or B-",
        "analogy": "Like a health checkup for business quality",
    },
    "Acc/Dis Rating (A-E)": {
        "definition": "Shows institutional buying vs selling pressure",
        "elite_threshold": "A, B, or B-",
        "analogy": "Like tracking whether the 'smart money' is buying or selling",
    },
}
```

### 4.2 Elite Screening (Must Explain the 4-Gate Filter)

```
Gate 1: EPS ≥ 85 → "Is the company growing profits faster than 85% of all companies?"
Gate 2: RS ≥ 75 → "Is the stock price outperforming 75% of all stocks?"
Gate 3: SMR = A/B/B- → "Does the business have strong sales, margins, and returns?"
Gate 4: Acc/Dis = A/B/B- → "Are professional investors buying this stock?"

ALL 4 must pass. Like a 4-lock vault — need all 4 keys.
```

### 4.3 Three Tiers (Must Explain Each with Analogies)

```python
TIER_EXPLANATIONS = {
    "Tier 1: Momentum": {
        "who_its_for": "Experienced investors comfortable with volatility",
        "analogy": "Sports car — thrilling speed, feels every bump",
        "key_numbers": "Composite ≥ 95, RS ≥ 90, Stop 22%",
        "target_return": "25-30%",
        "risk_personality": "Can stomach -25% to -35% drawdown",
    },
    "Tier 2: Quality Growth": {
        "who_its_for": "Growth-oriented investors who want smoother ride",
        "analogy": "Luxury sedan — good speed, smooth ride",
        "key_numbers": "Composite ≥ 85, RS ≥ 80, Stop 18%",
        "target_return": "18-22%",
        "risk_personality": "Comfortable with -18% to -25% drawdown",
    },
    "Tier 3: Defensive": {
        "who_its_for": "Capital preservation, newer investors, closer to retirement",
        "analogy": "SUV — handles any terrain, great safety rating",
        "key_numbers": "Composite ≥ 80, RS ≥ 75, Stop 12%",
        "target_return": "12-15%",
        "risk_personality": "Wants to limit drawdown to -12% to -18%",
    },
}
```

### 4.4 Keep Categories (Must Explain Why 14 Positions Are Pre-Committed)

```python
KEEP_EXPLANATIONS = {
    "Fundamental Review Keeps (5)": {
        "what": "Stocks kept because of strong value metrics despite missing elite IBD thresholds",
        "analogy": "Like keeping a reliable car that doesn't win races but gets you there safely",
        "examples": "UNH (PE 17.9 vs 22 historical), MRK (PE 74% below 10-yr avg)",
    },
    "IBD High-Rated Keeps (9)": {
        "what": "Elite momentum leaders: Composite ≥ 93 AND RS ≥ 90",
        "analogy": "Like keeping the honor roll students — they've earned their spot with outstanding grades",
        "examples": "CLS (99/97), AGI (99/92), KLAC (97/94)",
    },
}
```

### 4.5 Multi-Source Validation (Must Explain the Scoring System)

```
Explain how stocks earn points from multiple sources:
  IBD lists: +1 to +3 points
  Analyst ratings: +1 to +3 points
  Schwab themes: +1 to +2 points

"Think of it like a job interview — being recommended by
ONE person is good, but being recommended by 3 different
people who don't know each other is much stronger."

Multi-Source Validated = score ≥ 5 from 2+ different providers
```

### 4.6 Trailing Stop Protocol (Must Explain Why Stops Tighten)

```
Initial: "Set the safety net at 22/18/12% below purchase price"
After 10% gain: "You've made money — tighten the net so you keep more"
After 20% gain: "Lock in most of your profits — net moves much closer"

Analogy: "Like raising the floor in a building as you climb higher.
You can always go up, but the floor beneath you rises too,
so you can never fall back to the basement."
```

### 4.7 Sector Rotation (Must Explain Using Detector's Verdict)

```
If ACTIVE: "The market is shifting leadership — like seasons changing.
The Rotation Detector confirmed this with X of 5 signals."

If EMERGING: "Early signs of a shift, but not confirmed yet —
like the first cool morning in August. Could be fall, or just a dip."

If NONE: "Current sector leadership is stable — like mid-summer.
No need to change course."
```

---

## 5. Autonomy & Decision-Making

### Decision 1: Choosing Explanation Depth Per Stock
```
Agent reasoning: "NVDA is a T1 Momentum pick with Composite 98, RS 81.
This is a portfolio centerpiece — deserves deep explanation.

I'll cover:
1. WHAT NVDA does (simple: 'makes the chips that power AI')
2. WHY IBD rates it 98 Composite ('top 2% of all stocks')
3. WHY RS is 81 not 90+ ('price pulled back but fundamentals are elite')
4. WHAT the trailing stop means ('your safety net is 22% below — about $27K of your $35K position')
5. WHY it's in Tier 1 ('highest conviction, highest risk, highest reward')

For a T3 Defensive pick like SCHW ($18K), one paragraph is sufficient."
```

### Decision 2: Explaining Rotation Detection Verdict
```
Detector says: ACTIVE rotation, strength 7/10, cyclical type
Source: Technology/Software → Destination: Energy/Mining

Agent reasoning: "Use seasons analogy:

'The Rotation Detector found strong evidence (7 out of 10) that
the market is rotating — like seasons changing from summer to fall.

Technology and Software have been the "summer crops" — thriving for months.
But the evidence shows they're starting to cool off (RS declining, fewer
stocks making new highs).

Meanwhile, Energy and Mining are the "fall crops" — gaining strength
rapidly (RS jumping 12-13 points, institutional money flowing in).

What triggered the detection? Four of five signals fired:
1. ✅ Leadership changed — Energy jumped from #8 to #2 in rankings
2. ✅ Momentum is diverging — some sectors rising, others falling
3. ✅ More Energy stocks are making new highs, fewer Tech stocks are
4. ✅ Professional investors are buying Energy, selling Technology
5. ❌ Only the speed-of-change signal didn't fully confirm

The Sector Strategist used this detection to adjust sector weights —
that's why you see more Mining/Energy positions in your portfolio.'"
```

### Decision 3: Explaining Risk Findings in Non-Scary Terms
```
Risk Officer issued 2 warnings: AI concentration + sector limit approaching

Agent reasoning: "Frame constructively, not alarmingly:

'The Risk Officer — think of them as your portfolio's safety inspector —
ran 10 checks on your portfolio. Good news: no vetoes (hard violations).
Two things to be aware of:

1. AI CONCENTRATION (Warning):
   You own 8 stocks across 5 different sectors, but they all benefit from
   the AI spending boom. It's like owning a restaurant, food truck, and
   catering company — they look different, but all depend on food costs.
   If AI spending slows, all 8 could decline together.
   
   The fix? Your portfolio already includes Mining (NEM, KGC, AEM),
   Healthcare (UNH, LLY, ISRG), and Financials (GS, TPR) as insurance.
   
2. TECHNOLOGY APPROACHING LIMIT (Warning):
   At 28%, Technology is safe (limit: 40%), but when you count all
   tech-adjacent positions, it's closer to 35%. Just keep an eye on it.

Sleep Well Score: 7/10 — like a hotel bed: comfortable, but you might
wake up once if there's a thunderstorm (market correction). The trailing
stops are your fire alarm — they'll get you out before real damage.'"
```

### Decision 4: Explaining the Two Keep Categories
```
Agent reasoning: "Users often wonder 'why are some stocks kept while
others are sold?' The two categories make this clear:

'Your portfolio has 14 "keeper" positions — stocks that
weren't up for sale. Here's why:

FUNDAMENTAL KEEPS (5 stocks: UNH, MU, AMZN, MRK, COP):
   These are like blue-chip real estate — their IBD scores might not
   all be perfect, but their value metrics (low PE ratios, strong
   earnings forecasts) make them worth holding. UNH trades at a PE
   of 17.9 vs its historical average of 22 — that's a 20% discount
   on the largest healthcare company.

IBD ELITE KEEPS (9 stocks: CLS, AGI, TFPM, GMED, SCCO, HWM, KLAC, TPR, GS):
   These are the honor roll students — Composite ≥ 93 AND RS ≥ 90.
   The framework says: keep unless something overrides. CLS, for example,
   has Composite 99 and RS 97 — it outperforms 97% of all stocks.'"
```

### Decision 5: Teaching CAN SLIM Using Portfolio Examples
```
Agent reasoning: "CAN SLIM is abstract. Make it concrete with actual portfolio stocks:

'CAN SLIM is IBD's 7-point inspection for stocks. Let me show you
how it works using FIX (Comfort Systems USA), one of your Tier 1 picks:

C = Current earnings growth — Is the company making more money NOW?
    FIX: EPS Rating 99 — top 1%. Earnings accelerating. ✅

A = Annual growth — Has it been growing for YEARS?
    FIX: Consistent multi-year growth in commercial HVAC. ✅

N = New — Is something NEW driving the business?
    FIX: Data center cooling demand from AI buildout. ✅

S = Supply/demand — Are big investors buying?
    FIX: On Big Cap 20 AND Sector Leaders — institutional favorites. ✅

L = Leader — Is it #1 in its industry?
    FIX: Industry leader in building services. Composite 99. ✅

I = Institutional sponsorship — Do fund managers own it?
    FIX: Multi-list validation (Big Cap 20 + Sector Leaders = 6 points). ✅

M = Market direction — Is the overall market healthy?
    Current: [reference Rotation Detector regime assessment]. ✅

FIX scores 7/7 on CAN SLIM — that's why it earned Tier 1 Momentum.'"
```

---

## 6. Output Contract

### EducatorOutput (Top Level)

```python
class EducatorOutput(BaseModel):
    executive_summary: str                           # 200+ chars, 2-3 paragraphs
    tier_guide: TierGuide                            # Explain the 3 tiers
    keep_explanations: KeepExplanations              # Why 14 positions were kept
    stock_explanations: List[StockExplanation]       # Top 15-25 picks explained
    rotation_explanation: str                         # Detector verdict in plain language (100+ chars)
    risk_explainer: str                              # Risk findings + Sleep Well (100+ chars)
    transition_guide: TransitionGuide                # Reconciler output explained
    concept_lessons: List[ConceptLesson]             # IBD concepts taught (5+)
    action_items: List[str]                          # What to do next (3-7 items)
    glossary: Dict[str, str]                         # Term → definition (15+ entries)
    analysis_date: str

    # VALIDATORS:
    # - executive_summary ≥ 200 chars
    # - At least 15 stock explanations
    # - At least 5 concept lessons
    # - At least 15 glossary entries
    # - At least 3 action items
    # - rotation_explanation ≥ 100 chars
    # - risk_explainer ≥ 100 chars
```

### TierGuide

```python
class TierGuide(BaseModel):
    overview: str                        # How the 3 tiers work together (50+ chars)
    tier_1_description: str              # Who it's for, expectations, analogy
    tier_2_description: str
    tier_3_description: str
    choosing_advice: str                 # How an investor decides which tier to weight
    current_allocation: str              # "39% T1 / 37% T2 / 22% T3 / 2% cash"
    analogy: str                         # Vehicle / sports / other analogy
```

### KeepExplanations

```python
class KeepExplanations(BaseModel):
    overview: str                        # Why 14 positions are pre-committed (50+ chars)
    fundamental_keeps: str               # Explain the 5 fundamental keeps
    ibd_keeps: str                       # Explain the 9 IBD high-rated keeps
    total_keeps: int                     # Should be 14
```

### TransitionGuide

```python
class TransitionGuide(BaseModel):
    overview: str                        # What's changing and why (100+ chars)
    what_to_sell: str                    # Explain the sells in plain language
    what_to_buy: str                     # Explain the buys in plain language
    money_flow_explained: str            # Where the cash comes from and goes (100+ chars)
    timeline_explained: str              # 4-week plan in plain language
    before_after_summary: str            # Transformation metrics explained
    key_numbers: str                     # "70→59 positions, 55%→28% tech, 5%→22% defensive"
```

### StockExplanation (Per Stock)

```python
class StockExplanation(BaseModel):
    symbol: str
    company_name: str
    tier: int                            # 1, 2, or 3
    keep_category: Optional[str]         # "fundamental" / "ibd" / None (new buy)
    one_liner: str                       # "NVDA: The AI gold rush pick"
    why_selected: str                    # Plain language, 50+ chars
    ibd_ratings_explained: str           # "Composite 98 means top 2% of all stocks"
    key_strength: str                    # Single most compelling strength
    key_risk: str                        # Single biggest risk
    position_context: str                # "$35K in T1, 22% trailing stop"
    analogy: Optional[str]               # Relatable comparison (at least for top 10)
```

### ConceptLesson

```python
class ConceptLesson(BaseModel):
    concept: str                         # "Relative Strength Rating"
    simple_explanation: str              # No jargon, 30+ chars
    analogy: str                         # Relatable comparison
    why_it_matters: str                  # Practical relevance
    example_from_analysis: str           # Using actual stocks from this run
    framework_reference: str             # "See Framework v4.0 Section 2.3"
```

---

## 7. Example Output (Partial)

```json
{
  "executive_summary": "Your portfolio has been rebuilt from the ground up using the IBD Momentum Investment Framework. From 70 scattered positions, we've focused to 59 carefully selected investments organized into three tiers: Tier 1 Momentum (39%) holds your highest-conviction growth picks like NVDA, NEM, and GOOGL — stocks rated 95+ by IBD. Tier 2 Quality Growth (37%) balances strong growth with stability through names like AVGO, MSFT, and LLY. Tier 3 Defensive (22%) provides protection with healthcare leaders, quality financials, and low-volatility ETFs. Technology concentration has been cut from 60% to 28%, and new exposure to Mining, Aerospace, and Healthcare provides diversification you didn't have before.\n\nOf the 59 positions, 14 were 'kept' from your original portfolio — 5 because of strong value metrics (like UNH and AMZN), and 9 because their IBD ratings are truly elite (Composite 93+ and RS 90+, like CLS with a perfect 99/97). The remaining 45 positions were carefully selected from IBD's top lists, analyst recommendations, and thematic investing opportunities.\n\nThe Rotation Detector found signs of sector rotation from Technology toward Energy and Mining — your portfolio is already positioned to benefit from this shift with significant mining exposure (NEM, AEM, KGC, AGI, TFPM, GDX, GDXJ). Every position has a trailing stop set to limit losses while locking in gains as they grow.",

  "tier_guide": {
    "overview": "Your portfolio uses a 3-tier system that balances high-growth ambition with real protection. Think of it as organizing your investments by how much excitement (and turbulence) you're willing to accept.",
    "tier_1_description": "TIER 1: MOMENTUM (39% of portfolio, ~$588K) — These are your race horses. Stocks rated 95+ by IBD with the strongest price momentum. This tier holds NEM, GOOGL, FIX, PLTR and more. Higher expected returns (25-30%) come with wider stops (22%) because these stocks can be volatile. Best for: money you won't need for 5+ years.",
    "tier_2_description": "TIER 2: QUALITY GROWTH (37% of portfolio, ~$563K) — These are your thoroughbreds. Strong companies (rated 85+) with proven track records. AVGO, MSFT, LLY, CRM lead this tier. Good returns (18-22%) with moderate volatility and 18% stops. Best for: your core growth allocation.",
    "tier_3_description": "TIER 3: DEFENSIVE (22% of portfolio, ~$329K) — These are your safety net. Quality companies and ETFs designed to hold up in tough markets. UNH, ISRG, V plus defensive ETFs like QUAL, SPLV. Lower returns (12-15%) but tighter stops (12%) mean you don't give back much in downturns. Best for: the money that lets you sleep at night.",
    "choosing_advice": "If your portfolio dropped 25% tomorrow, would you (a) buy more, (b) hold steady, or (c) feel sick? If mostly (a), lean toward T1. If (b), T2 is your sweet spot. If (c), lean toward T3. Your current 39/37/22 split is a Moderate profile — balanced growth with meaningful protection.",
    "current_allocation": "39% Tier 1 / 37% Tier 2 / 22% Tier 3 / 2.8% Cash",
    "analogy": "Tier 1 is a sports car, Tier 2 is a luxury sedan, Tier 3 is an armored SUV. Your portfolio is a 3-vehicle garage — you pick which car based on the road conditions (market environment)."
  },

  "glossary": {
    "Composite Rating": "IBD's master score (1-99) combining earnings, relative strength, fundamentals, and institutional interest. 99 = top 1% of all stocks. Your elite threshold: ≥85.",
    "RS Rating": "Relative Strength — how the stock's price compares to all others over 12 months. RS 90 = outperformed 90% of stocks. Elite threshold: ≥75.",
    "EPS Rating": "Earnings Per Share rating (1-99) — measures profit growth vs all stocks. Compares last 2 quarters AND 3-year trend. Elite threshold: ≥85.",
    "SMR Rating": "Sales, Margins, Return on equity — grades A through E measuring business quality. Elite: A, B, or B-.",
    "Acc/Dis Rating": "Accumulation/Distribution — shows whether big institutional investors are buying (accumulating) or selling (distributing). Elite: A, B, or B-.",
    "CAN SLIM": "IBD's 7-factor checklist: Current earnings, Annual growth, New products, Supply/demand, Leader, Institutional sponsorship, Market direction.",
    "Tier 1 Momentum": "Highest-growth portfolio tier. Composite ≥95, RS ≥90. Stocks like NVDA, NEM, GOOGL. 22% trailing stop. Target return: 25-30%.",
    "Tier 2 Quality Growth": "Balanced growth tier. Composite ≥85, RS ≥80. Stocks like AVGO, MSFT, LLY. 18% trailing stop. Target return: 18-22%.",
    "Tier 3 Defensive": "Capital preservation tier. Composite ≥80, RS ≥75. Stocks like UNH, ISRG, V. 12% trailing stop. Target return: 12-15%.",
    "Trailing Stop": "Automatic sell order that rises with the stock price but never falls. Protects gains while allowing upside. Set at 22%, 18%, or 12% depending on tier.",
    "Stop Tightening": "As a stock gains 10% or 20%, the trailing stop gets tighter (closer to current price) to lock in more profit. Like raising the floor as you climb higher.",
    "Sector Rotation": "The pattern of different market sectors taking turns leading. Detected by our Rotation Detector using 5 independent signals.",
    "Multi-Source Validated": "A stock recommended by 2+ independent sources with total validation score ≥5. Like getting multiple job references from people who don't know each other.",
    "IBD Keep Threshold": "Composite ≥93 AND RS ≥90. These elite stocks are kept unless specific overriding factors exist. 9 stocks meet this threshold in your portfolio.",
    "Sleep Well Score": "Risk Officer's rating (1-10) of how comfortable you should feel. 10 = sleep like a baby. Based on stops, diversification, concentration, and regime alignment."
  }
}
```

---

## 8. Testing Plan

### Level 1: Schema Tests (No LLM — instant)

| Test | What It Validates |
|---|---|
| `test_valid_output_parses` | Known-good output parses |
| `test_executive_summary_length` | ≥ 200 chars |
| `test_minimum_stock_explanations` | ≥ 15 stocks explained |
| `test_minimum_concept_lessons` | ≥ 5 concepts taught |
| `test_minimum_glossary_entries` | ≥ 15 terms defined |
| `test_minimum_action_items` | ≥ 3 action items |
| `test_rotation_explanation_length` | ≥ 100 chars |
| `test_risk_explainer_length` | ≥ 100 chars |
| `test_tier_guide_complete` | All 3 tiers described |
| `test_keep_explanations_complete` | All 2 categories present |
| `test_keep_total_is_14` | total_keeps == 14 |
| `test_stock_explanation_has_tier` | Every stock has tier 1/2/3 |
| `test_concept_has_example` | Every concept has example_from_analysis |

### Level 2: LLM Output Tests

| Test | What It Validates |
|---|---|
| `test_output_is_valid_json` | Parseable JSON |
| `test_output_matches_schema` | Conforms to EducatorOutput |
| `test_no_unexplained_jargon` | Financial terms appear in glossary |
| `test_analogies_are_relatable` | No financial jargon inside analogies |
| `test_action_items_are_specific` | Actionable, not vague |
| `test_ratings_explained_correctly` | Composite/RS/EPS definitions accurate |
| `test_tier_thresholds_correct` | T1≥95/90, T2≥85/80, T3≥80/75 |

### Level 3: Behavioral Tests

| Test | What It Validates |
|---|---|
| `test_does_not_make_decisions` | No "you should buy/sell" language |
| `test_does_not_change_analysis` | No modified tiers/scores/positions |
| `test_does_not_add_stocks` | Only explains existing analysis |
| `test_does_not_give_personal_advice` | No "based on your situation" |
| `test_references_actual_stocks` | Symbols match pipeline output |
| `test_framework_references_accurate` | Elite criteria, stop levels correct |

### Level 4: Integration (All Agents → Educator)

| Test | What It Validates |
|---|---|
| `test_educator_accepts_all_outputs` | Can consume full pipeline output |
| `test_explained_stocks_match_pipeline` | Symbols exist in PM output |
| `test_risk_explanation_reflects_findings` | Matches Risk Officer's actual vetoes/warnings |
| `test_rotation_explanation_matches_verdict` | Matches Detector's actual status |
| `test_tier_allocations_match_pm` | 39/37/22 referenced correctly |
| `test_keep_counts_match_pm` | 5+9=14 referenced correctly |

---

## 9. Integration Points

### Receives From (ALL 7 prior agents)
- **Research Agent:** `ResearchOutput` (stock universe, sector patterns, sources)
- **Analyst Agent:** `AnalystOutput` (top 50, tiers, convictions, strengths/weaknesses)
- **Rotation Detector:** `RotationDetectionOutput` (verdict, strength, sectors, signals)
- **Sector Strategist:** `SectorStrategyOutput` (rotation response, allocation model)
- **Portfolio Manager:** `PortfolioOutput` (3-tier portfolio, positions, keeps, stops)
- **Risk Officer:** `RiskAssessment` (checks, vetoes, warnings, Sleep Well scores)
- **Portfolio Reconciler:** `ReconciliationOutput` (current vs recommended diff, money flow, 4-week implementation plan, keep verification, transformation metrics)

### Passes To
- **Final Report** — this is the end of the pipeline
- Output is the investor-facing deliverable

### Contract Rules
- Educator **MUST NOT** modify any analysis from other agents
- Educator **MUST** explain decisions using other agents' reasoning
- Educator **MUST** define every financial term in glossary
- Educator **MUST** correctly reference Framework v4.0 thresholds
- Educator **CAN** choose which stocks to explain in depth (top 15-25)
- Educator **CAN** choose creative analogies and teaching approaches

---

## 10. Files to Create

```
1. src/ibd_agents/schemas/educator_output.py                ← Output contract (FIRST)
2. tests/unit/test_educator_schema.py                        ← Schema tests (13+)
3. src/ibd_agents/agents/educator_agent.py                   ← Agent + Task
4. config/agents.yaml                                        ← Add educator entry
5. config/tasks.yaml                                         ← Add educator task
6. tests/unit/test_educator_agent.py                         ← LLM output tests
7. tests/unit/test_educator_behavior.py                      ← Behavioral tests
8. tests/integration/test_full_pipeline_to_educator.py       ← Full pipeline test
9. golden_datasets/educator_golden.json                      ← Golden baseline
```