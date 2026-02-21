# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IBD Momentum Investment Framework v4.0 — a CrewAI-based multi-agent system that processes Investor's Business Daily (IBD) data files, Schwab thematic investment PDFs, and Motley Fool recommendations to discover, rate, and organize growth stock opportunities. The system **never makes buy/sell recommendations** — it only discovers, scores, and organizes.

## Commands

### Install dependencies
```bash
pip install -e .
# or within venv:
source venv/bin/activate && pip install -e .
```

### Run tests
```bash
# All tests
pytest

# Unit tests only (no LLM, fast)
pytest tests/unit/

# Single test file
pytest tests/unit/test_research_schema.py

# Single test class
pytest tests/unit/test_research_schema.py::TestIBDKeepLogic

# By marker
pytest -m schema      # Schema validation tests (no LLM)
pytest -m behavior    # Behavioral boundary tests
pytest -m integration # Cross-agent integration tests
pytest -m llm         # Tests that call real LLM agent
```

### Run the deterministic pipeline (no LLM)
```python
from ibd_agents.agents.research_agent import run_research_pipeline
from ibd_agents.agents.analyst_agent import run_analyst_pipeline
from ibd_agents.agents.rotation_detector import run_rotation_pipeline
from ibd_agents.agents.sector_strategist import run_strategist_pipeline
from ibd_agents.agents.portfolio_manager import run_portfolio_pipeline
from ibd_agents.agents.risk_officer import run_risk_pipeline
from ibd_agents.agents.returns_projector import run_returns_projector_pipeline
from ibd_agents.agents.portfolio_reconciler import run_reconciler_pipeline
from ibd_agents.agents.educator_agent import run_educator_pipeline

research = run_research_pipeline("data")
analyst = run_analyst_pipeline(research)
rotation = run_rotation_pipeline(analyst, research)
strategy = run_strategist_pipeline(rotation, analyst)
portfolio = run_portfolio_pipeline(strategy, analyst)
risk = run_risk_pipeline(portfolio, strategy)
returns = run_returns_projector_pipeline(portfolio, rotation, risk, analyst)
reconciliation = run_reconciler_pipeline(portfolio, analyst, returns_output=returns, rotation_output=rotation, strategy_output=strategy, risk_output=risk)
educator = run_educator_pipeline(portfolio, analyst, rotation, strategy, risk, reconciliation, returns_output=returns)

from ibd_agents.agents.executive_synthesizer import run_synthesizer_pipeline
synthesis = run_synthesizer_pipeline(research, analyst, rotation, strategy, portfolio, risk, returns, reconciliation, educator)
```

### Run the full pipeline with Excel output
```bash
python run_pipeline.py
# or with custom data/output dirs:
python run_pipeline.py data output
```

### Run the agentic pipeline (requires CrewAI + LLM)
```python
from ibd_agents.agents.research_agent import build_research_agent, build_research_task
agent = build_research_agent()
task = build_research_task(agent, data_dir="data")
```

## Architecture

### Agent Pipeline

The system runs ten agents in sequence:
1. **Agent 01: Research Agent** — discovers stocks from data files, extracts ratings, computes validation scores
2. **Agent 02: Analyst Agent** — applies elite screening, tier assignment, sector ranking, produces top 50
3. **Agent 03: Rotation Detector** — detects sector rotation using 5-signal framework, classifies type/stage
4. **Agent 04: Sector Strategist** — translates rotation + analyst data into sector allocations, theme ETF recommendations, and rotation action signals
5. **Agent 05: Portfolio Manager** — constructs 3-tier portfolio with position sizing, 14 keeps, trailing stops, and max loss limits
6. **Agent 06: Risk Officer** — runs 10-check risk assessment, stress tests, Sleep Well scores, keep validation
7. **Agent 07: Returns Projector** — projects portfolio returns across bull/base/bear scenarios, benchmarks against SPY/DIA/QQQ, computes alpha, risk metrics, and confidence intervals
8. **Agent 08: Portfolio Reconciler** — diffs current vs recommended holdings, generates KEEP/SELL/BUY/ADD/TRIM actions with 4-week implementation plan
9. **Agent 09: Educator** — explains every decision from agents 01-08 in plain language with analogies and glossary
10. **Agent 10: Executive Summary Synthesizer** — reads all 9 agent outputs, produces unified investment thesis with cross-agent connections, contradiction detection, key numbers dashboard, and action items; uses LLM (anthropic SDK) for narrative with deterministic template fallback
16. **Agent 16: Regime Detector** — classifies market into 5 regimes (CONFIRMED_UPTREND, UPTREND_UNDER_PRESSURE, FOLLOW_THROUGH_DAY, RALLY_ATTEMPT, CORRECTION) using distribution days, market breadth, leading stock health, sector rotation, and follow-through day analysis. Runs after Agent 02. Backward-compatible via `to_legacy_regime()` → bull/neutral/bear.

Each agent has two execution paths:
1. **Deterministic pipeline** (`run_*_pipeline()`) — no LLM, used by tests
2. **Agentic pipeline** (`build_*_agent()` + `build_*_task()`) — CrewAI agent with tools and LLM reasoning

All paths produce validated Pydantic schemas (`ResearchOutput` / `AnalystOutput` / `RotationDetectionOutput` / `SectorStrategyOutput` / `PortfolioOutput` / `RiskAssessment` / `ReturnsProjectionOutput` / `ReconciliationOutput` / `EducatorOutput` / `SynthesisOutput` / `RegimeDetectorOutput`).

### Source processing hierarchy (strict order)

1. **IBD XLS/CSV** (`data/ibd_xls/`) — highest weight, all 5 ratings (Composite, RS, EPS, SMR, Acc/Dis). Supports `.xls`, `.xlsx`, and `.csv` files with the same column format.
2. **IBD PDFs** (`data/ibd_pdf/`) — Smart Table, Tech Leaders, Top 200, etc.
3. **Motley Fool PDFs** (`data/fool_pdf/`) — supplementary validation
4. **Schwab Theme PDFs** (`data/schwab_themes/`) — thematic tagging only, not primary screening

### Key modules

- `src/ibd_agents/schemas/research_output.py` — **Research output contract**. Framework constants (IBD sectors, validation points, tier thresholds), Pydantic models (`ResearchStock`, `ResearchETF` (with RS, Acc/Dis, YTD, Vol%, score, rank, tier), `SectorPattern`, `ResearchOutput`), and helper functions.
- `src/ibd_agents/schemas/analyst_output.py` — **Analyst output contract**. Pydantic models (`RatedStock`, `RatedETF` (with tier, RS, Acc/Dis, YTD, score, rank, conviction 1-10, focus group, screening gates, strengths/weaknesses), `UnratedStock`, `SectorRank`, `IBDKeep`, `EliteFilterSummary`, `TierDistribution`, `ETFTierDistribution`, `AnalystOutput`).
- `src/ibd_agents/schemas/rotation_output.py` — **Rotation output contract**. Sector clusters, signal thresholds, enums (`RotationStatus`, `RotationType`, `RotationStage`), Pydantic models (`MarketRegime`, `SignalReading` (with `etf_confirmation`), `RotationSignals`, `SectorFlow`, `RotationDetectionOutput` (with `etf_flow_summary`, `rotation_narrative`, `narrative_source`)).
- `src/ibd_agents/schemas/strategy_output.py` — **Strategy output contract**. Portfolio architecture targets, sector limits, regime actions, 16 Schwab theme-to-ETF mappings, Pydantic models (`SectorAllocationPlan`, `ThemeRecommendation` (with `etf_rankings`), `RotationActionSignal`, `SectorStrategyOutput`).
- `src/ibd_agents/schemas/portfolio_output.py` — **Portfolio output contract**. Position sizing limits, trailing stops, max loss, 14 pre-committed keeps (5 fundamental + 9 IBD), Pydantic models (`PortfolioPosition` (with `volatility_adjustment`, `sizing_source`), `TierPortfolio`, `KeepsPlacement`, `OrderAction`, `PortfolioOutput`).
- `src/ibd_agents/schemas/risk_output.py` — **Risk output contract**. 10-check risk framework, stress scenarios, Sleep Well scores, Pydantic models (`RiskCheck`, `Veto`, `RiskWarning`, `StopLossRecommendation`, `StressScenario`, `SleepWellScores`, `RiskAssessment` (with `stop_loss_recommendations`, `stop_loss_source`)).
- `src/ibd_agents/schemas/returns_projection_output.py` — **Returns Projector output contract**. 3-scenario return projections (bull/base/bear), benchmark comparisons (SPY/DIA/QQQ), alpha analysis, risk metrics, tier mix alternatives, confidence intervals, ETF/stock decomposition, Pydantic models (`TierAllocation`, `ScenarioProjection`, `ExpectedReturn`, `BenchmarkComparison`, `AlphaSource`, `AlphaAnalysis`, `RiskMetrics`, `TierMixScenario`, `PercentileRange`, `ConfidenceIntervals`, `AssetTypeDecomposition`, `ReturnsProjectionOutput`).
- `src/ibd_agents/schemas/reconciliation_output.py` — **Reconciliation output contract**. KEEP/SELL/BUY/ADD/TRIM actions, money flow, 4-week implementation plan, ETF implementation notes, Pydantic models (`CurrentHolding`, `HoldingsSummary`, `PositionAction`, `MoneyFlow`, `ImplementationWeek`, `ReconciliationOutput` (with `rationale_source`)).
- `src/ibd_agents/schemas/educator_output.py` — **Educator output contract**. Educational content with analogies, glossary, concept lessons, ETF explanations, Pydantic models (`TierGuide`, `KeepExplanations`, `StockExplanation`, `ETFExplanation`, `ConceptLesson`, `TransitionGuide`, `EducatorOutput`).
- `src/ibd_agents/schemas/synthesis_output.py` — **Synthesis output contract**. Executive summary with cross-agent connections and contradiction detection, Pydantic models (`KeyNumbersDashboard`, `CrossAgentConnection`, `Contradiction`, `SynthesisOutput`).
- `src/ibd_agents/agents/research_agent.py` — Research Agent builder and deterministic pipeline (`run_research_pipeline()`).
- `src/ibd_agents/agents/analyst_agent.py` — Analyst Agent builder and deterministic pipeline (`run_analyst_pipeline()`).
- `src/ibd_agents/agents/rotation_detector.py` — Rotation Detector builder and deterministic pipeline (`run_rotation_pipeline()`).
- `src/ibd_agents/agents/sector_strategist.py` — Sector Strategist builder and deterministic pipeline (`run_strategist_pipeline()`).
- `src/ibd_agents/agents/portfolio_manager.py` — Portfolio Manager builder and deterministic pipeline (`run_portfolio_pipeline()`).
- `src/ibd_agents/agents/risk_officer.py` — Risk Officer builder and deterministic pipeline (`run_risk_pipeline()`).
- `src/ibd_agents/agents/returns_projector.py` — Returns Projector builder and deterministic pipeline (`run_returns_projector_pipeline()`).
- `src/ibd_agents/agents/portfolio_reconciler.py` — Portfolio Reconciler builder and deterministic pipeline (`run_reconciler_pipeline()`).
- `src/ibd_agents/agents/educator_agent.py` — Educator Agent builder and deterministic pipeline (`run_educator_pipeline()`).
- `src/ibd_agents/agents/executive_synthesizer.py` — Executive Summary Synthesizer builder and deterministic pipeline (`run_synthesizer_pipeline()`). LLM narrative via anthropic SDK with template fallback.
- `src/ibd_agents/schemas/regime_detector_output.py` — **Regime Detector output contract**. 5-state market regime classification. Constants (distribution day thresholds, FTD thresholds, exposure ranges, health score ranges), Enums (`MarketRegime5`, `PreviousRegime`, `Confidence`, `SignalDirection`, `Trend`, `LeaderHealth`, `SectorCharacter`, `FTDQuality`), Pydantic models (`DistributionDayAssessment`, `BreadthAssessment`, `LeaderAssessment`, `SectorAssessment`, `FollowThroughDayAssessment`, `TransitionCondition`, `RegimeChange`, `RegimeDetectorOutput`). Validators enforce exposure/health ranges per regime, signal counts sum to 5, FTD consistency, distribution day caps. `to_legacy_regime()` maps 5-state → 3-state.
- `src/ibd_agents/agents/regime_detector.py` — Regime Detector builder and deterministic pipeline (`run_regime_detector_pipeline()`). Runs after Agent 02.
- `src/ibd_agents/tools/regime_data_fetcher.py` — 5 data-fetching tools (distribution days, market breadth, leading stocks health, sector rankings, index price history). Mock implementations for 5 scenarios + real yfinance implementations. 5 CrewAI BaseTool wrappers.
- `src/ibd_agents/tools/regime_classifier.py` — Pure classification functions: `classify_regime()` decision tree, assessment builders, FTD detection, exposure/health/confidence computation, transition conditions, executive summary generation.
- `src/ibd_agents/tools/` — CrewAI tools, each with a pure function + BaseTool wrapper:
  - `xls_reader.py` — reads IBD Excel files via pandas (xlrd/openpyxl) and CSV files
  - `pdf_reader.py` — extracts from IBD and Motley Fool PDFs via pdfplumber
  - `theme_reader.py` — extracts tickers from Schwab theme PDFs, classifies stock vs ETF
  - `validation_scorer.py` — `StockUniverse` aggregates data across sources, computes validation scores
  - `sector_classifier.py` — static + LLM sector classification for sectorless stocks
  - `cap_classifier.py` — static + LLM market cap classification (large/mid/small)
  - `elite_screener.py` — 4-gate elite screening, tier assignment, conviction scoring (stocks + ETFs), ETF screening gates (RS>=70, Acc/Dis A/B/B-)
  - `sector_scorer.py` — sector score formula and ranking
  - `valuation_metrics.py` — estimated P/E, PEG, Beta, Sharpe, Alpha, Volatility, Return, Risk Rating from sector baselines
  - `catalyst_enrichment.py` — LLM catalyst lookup (earnings dates, FDA approvals, product launches) and conviction adjustment computation
  - `rotation_narrative.py` — LLM rotation historical context narrative (single call, not batched)
  - `stop_loss_tuning.py` — LLM per-position stop-loss recommendations with tier-range clamping
  - `dynamic_sizing.py` — LLM volatility-based position size adjustments with tier-max clamping
  - `reconciliation_reasoning.py` — LLM context-aware action rationale enrichment (rotation/risk/regime)
  - `rotation_signals.py` — 5-signal rotation detection: RS Divergence, Leadership Change, Breadth Shift, Elite Concentration Shift, IBD Keep Migration; ETF sector evidence annotations
  - `market_regime.py` — market regime detection (bull/bear/neutral), rotation type/stage/velocity classification, confidence scoring, sector flow detection
  - `sector_allocator.py` — regime-adjusted tier targets, sector scoring with rotation bias, per-tier and overall allocation computation, concentration limits
  - `theme_mapper.py` — Schwab theme-to-ETF recommendations based on sector alignment, competitive ETF ranking by analyst score, rotation action signal generation with confirmation/invalidation criteria
  - `position_sizer.py` — position sizing, trailing stop computation, max loss calculation, tier assignment
  - `keeps_manager.py` — 21-keep placement across tiers by category (fundamental/user/IBD)
  - `risk_analyzer.py` — 10 pure check functions (position sizing, stops, sectors, correlation + ETF theme overlap, regime, keeps), stress tests, Sleep Well scores
  - `tier_return_calculator.py` — computes per-tier returns by scenario with sector momentum adjustment, blended portfolio return, horizon scaling, tier contributions
  - `benchmark_fetcher.py` — retrieves benchmark (SPY/DIA/QQQ) return distributions by scenario, computes probability-weighted expected returns
  - `scenario_weighter.py` — provides scenario probability weights by market regime (bull/neutral/bear), computes weighted expected values
  - `alpha_calculator.py` — computes alpha vs benchmarks, expected alpha, alpha decomposition by source, outperform probability estimation
  - `drawdown_estimator.py` — estimates max drawdown with/without trailing stops, stop trigger probability and count
  - `confidence_builder.py` — builds percentile ranges (p10/p25/p50/p75/p90) from normal distribution, confidence intervals across time horizons
  - `portfolio_reader.py` — builds mock current holdings for deterministic pipeline (seed=42 for reproducibility)
  - `position_differ.py` — diffs current vs recommended positions, computes money flow, builds 4-week implementation plan
  - `file_lister.py` — discovers/classifies files in the data directory

### Data aggregation

`StockUniverse` (in `validation_scorer.py`) is the central aggregator. `StockAggregation` instances merge data per symbol across all sources, tracking validation labels for scoring. Numeric IBD ratings use "keep highest" merge strategy.

### Validation scoring system

Each source label maps to points (defined in `VALIDATION_POINTS`). Multi-source validated = total score >= 5 from 2+ distinct providers. Provider groups: IBD, CFRA, ARGUS, Morningstar, Motley Fool, Schwab.

### Key business rules (from schema validators)

- **IBD Keep candidate**: Composite >= 93 AND RS >= 90
- **Preliminary tiers**: T1 (Comp>=95, RS>=90, EPS>=80), T2 (>=85, >=80, >=75), T3 (>=80, >=75, >=70)
- **Elite screening**: All 4 must pass — EPS>=85, RS>=75, SMR A/B/B-, Acc/Dis A/B/B-
- **Sector score**: (avg_comp*0.25) + (avg_rs*0.25) + (elite_pct*0.30) + (multi_list_pct*0.20)
- **Conviction**: deterministic 1-10 (tier base + elite/validation/list/theme bonuses + catalyst timing adjustment)
- **Catalyst conviction adjustment**: LLM-enriched catalyst timing — ≤14 days: +2, ≤30 days: +1, >30 days: 0; catalyst_type must be one of: earnings/fda/product_launch/conference/guidance/dividend/split/other
- **Cap size categories**: large (>$10B), mid ($2B-$10B), small (<$2B) — static map + LLM fallback
- **Valid sectors**: exactly 28 IBD sector categories (defined in `IBD_SECTORS`)
- **SMR/Acc-Dis ratings**: only A/B/B-/C/D/E
- **ResearchOutput**: requires 1+ stocks across 1+ sectors
- **AnalystOutput**: all tiered rated stocks, sector rankings, IBD keeps, valuation summary
- **Rotation verdict**: ACTIVE (3+ signals, confidence>=50), EMERGING (2 signals), NONE (0-1 signals)
- **Rotation signals**: RS Divergence (gap>15), Leadership Change (2+ misaligned), Breadth Shift (cross-cluster <40%/>60%), Elite Concentration (cluster >50%/<20%), IBD Keep Migration (dominant cluster != growth or <40%)
- **7 sector clusters**: growth, commodity, defensive, financial, industrial, consumer_cyclical, services — mapping all 28 IBD sectors
- **Sector allocation**: 3-tier architecture (T1 Momentum=39%, T2 Quality=37%, T3 Defensive=22%, Cash=2%), regime-adjusted (bull: +5%T1/-5%T3, bear: -10%T1/+10%T3)
- **Concentration limits**: max 40% per sector, minimum 8 sectors in overall allocation
- **Theme ETFs**: 16 Schwab themes mapped to ETFs; growth themes → T1/T2 tier fit, defensive themes → T2/T3
- **Rotation action signals**: confirmation/invalidation criteria per signal (ACTIVE: overweight destination, EMERGING: prepare contingency, NONE: maintain)
- **Consistency validators**: keep candidates list must match flagged stocks; same for multi-source validated
- **Portfolio construction**: 3-tier portfolio — T1 Momentum (35-45%), T2 Quality Growth (33-40%), T3 Defensive (20-30%), Cash (2-15%)
- **Position limits**: stocks max 5%/4%/3% per tier, ETFs max 8%/8%/6% per tier
- **Trailing stops**: T1=22%, T2=18%, T3=12%; tighten after 10%/20% gains
- **Max loss**: stock {1:1.10%, 2:0.72%, 3:0.36%}, ETF {1:1.76%, 2:1.44%, 3:0.72%}
- **14 keeps**: 5 Fundamental (UNH,MU,AMZN,MRK,COP), 9 IBD (CLS,AGI,TFPM,GMED,SCCO,HWM,KLAC,TPR,GS)
- **Stock/ETF split**: 50/50 target with ±5% tolerance
- **Risk assessment**: 10 checks (PASS/WARNING/VETO), overall APPROVED/CONDITIONAL/REJECTED
- **Sleep Well scores**: 1-10 per tier and overall
- **Stress tests**: market crash (-20%), sector correction (-30%), rate hike (+50bp)
- **Returns projection**: exactly 3 scenarios (bull/base/bear), probabilities sum to 1.0; exactly 3 benchmarks (SPY/DIA/QQQ)
- **Scenario weights by regime**: bull={0.55,0.35,0.10}, neutral={0.25,0.50,0.25}, bear={0.10,0.35,0.55}
- **Tier historical returns**: T1 bull=35%/base=18%/bear=-8%, T2 bull=25%/base=14%/bear=-5%, T3 bull=15%/base=10%/bear=2%
- **Sector momentum multiplier**: leading=1.15, improving=1.05, lagging=0.95, declining=0.85
- **Trailing stop losses**: T1=22%, T2=18%, T3=12%; max portfolio loss with stops=17.88%
- **Confidence intervals**: p10/p25/p50/p75/p90 from normal distribution z-scores (-1.28/-0.67/0/0.67/1.28)
- **Alpha**: portfolio return - benchmark return; Sharpe = (return - 4.5%) / volatility
- **Drawdown ordering**: max_drawdown_with_stops <= max_drawdown_without_stops
- **Reconciliation actions**: KEEP (diff<0.3%), SELL (not recommended), BUY (not held), ADD (below target), TRIM (above target)
- **Implementation phases**: Week 1 Liquidation, Week 2 T1 Momentum, Week 3 T2 Quality Growth, Week 4 T3 Defensive
- **Money flow**: net_cash_change = (sell_proceeds + trim_proceeds) - (buy_cost + add_cost)
- **ETF scoring**: Composite score = RS×0.35 + Acc/Dis_numeric×0.25 + YTD_normalized×0.25 + Vol%_normalized×0.15 (all 0-100 scale)
- **ETF tiering**: Theme-based — Growth themes→T1, Defensive themes→T3, else→T2
- **ETF data from IBD PDFs**: RS Rating, Acc/Dis Rating, YTD Change, Close Price, Price Change, Volume % Change, Div Yield
- **ETF screening gates**: RS >= 70 (looser than stock's 75), Acc/Dis A/B/B-; both must pass
- **ETF conviction**: Base from tier (T1=7, T2=5, T3=3) + RS>=85 (+1) + Acc/Dis A (+1) + YTD>10% (+1) + has themes (+1); cap 10
- **ETF focus ranking**: ETFs grouped by focus area, sorted by etf_score DESC, ranked within group
- **ETF tier distribution**: per-tier ETF counts tracked in AnalystOutput
- **ETF rotation evidence**: ETF RS/volume trends annotate rotation signals 1 (RS Divergence) and 3 (Breadth Shift)
- **ETF competitive ranking**: Theme recommendations include ranked ETFs by analyst score
- **ETF theme overlap check**: Risk warning if any sector >15% from ETFs alone
- **ETF/stock return decomposition**: Per-tier stock vs ETF allocation %, count, volatility (ETF dampening 0.70x)
- **ETF implementation notes**: Execute ETFs before stocks (higher liquidity), limit orders for thematic ETFs, stagger if 4+ ETF orders
- **Educator**: min 15 stock/ETF explanations, 5 concept lessons (including 2 ETF concepts), 15 glossary entries, 3-7 action items, dedicated ETF explanations
- **Synthesis**: investment_thesis ≥200 chars, portfolio_narrative/risk_reward/market_context ≥100 chars, ≥3 cross-agent connections, 3-5 action items, synthesis_source "llm" or "template"
- **Contradiction detection**: sector concentration vs strategy allocation, bear regime vs offensive T1 weight >40%, risk not APPROVED vs optimistic returns >15%, fast rotation vs low turnover <30%
- **Cross-agent connections**: rotation→strategy, strategy→portfolio, portfolio→risk, risk→returns, rotation→portfolio (5 mandatory connections)
- **Regime Detector 5 regimes**: CONFIRMED_UPTREND (80-100% exposure), UPTREND_UNDER_PRESSURE (40-60%), FOLLOW_THROUGH_DAY (20-40%), RALLY_ATTEMPT (0-20%), CORRECTION (0%)
- **Distribution days**: index down ≥0.2% on higher volume; expire after 25 sessions; power-up days (≥1% up on higher volume) remove oldest; 5+ caps at UPTREND_UNDER_PRESSURE; 6+ forces CORRECTION
- **Follow-Through Day**: Day 4+ of rally, ≥1.25% gain, volume > prior day; quality STRONG (≥1.7%, Day 4-7, vol≥1.5x), MODERATE (Day 4-10), WEAK (after Day 10)
- **Regime hard rules**: no FTD after correction → no CONFIRMED_UPTREND; 5+ dist days → cannot be CONFIRMED_UPTREND; 6+ dist days → CORRECTION; conflicts → conservative default
- **Regime backward compat**: CONFIRMED_UPTREND→"bull", UPTREND_UNDER_PRESSURE→"neutral", FOLLOW_THROUGH_DAY→"neutral", RALLY_ATTEMPT→"bear", CORRECTION→"bear"
- **Regime 5 data tools**: distribution days, market breadth (% above 200MA/50MA, highs/lows, A/D line), leading stock health (RS≥90 above 50MA), sector rankings (growth vs defensive), index price history (FTD detection)

### Valuation & risk metrics (estimated, no external API)

All metrics in `valuation_metrics.py` are derived from hardcoded sector baselines + IBD ratings:

- **P/E**: estimated from sector P/E range (low/mid/high per 28 sectors) × growth factor from Comp/EPS/RS
- **P/E Category**: Deep Value / Value / Reasonable / Growth Premium / Speculative (relative to sector)
- **PEG**: P/E ÷ (EPS Rating × 0.5) — EPS×0.5 approximates growth rate
- **Beta**: sector baseline + momentum adjustment from RS/Composite deviation, clamped [0.2, 2.5]
- **Est. Return %**: stepped from RS Rating (RS99=60%, RS90=50%, RS80=30%, RS50=10%)
- **Volatility %**: sector baseline + momentum factor, clamped [10%, 60%]
- **Sharpe Ratio**: (Return - 4.5%) / Volatility
- **Alpha %**: Return - (4.5% + Beta × (12% - 4.5%)) — CAPM excess return
- **Risk Rating**: Excellent/Good/Moderate/Below Average/Poor (combines Sharpe + Beta)

Market constants: Risk-Free Rate=4.5%, Market Return=12%, S&P 500 Forward P/E=23.3x.

Excel output is color-coded: 6 category columns with PatternFill (Dark Green/Light Green/Yellow/Orange/Red).

### LLM-enriched valuation (optional, requires ANTHROPIC_API_KEY)

`enrich_valuation_llm()` in `valuation_metrics.py` uses Claude Haiku to look up real financial data per stock:
- **Metrics**: trailing P/E, forward P/E, beta, 1-year return, volatility, dividend yield, market cap
- **Grades**: valuation grade (same categories as P/E Category), risk grade
- **Guidance**: 2-3 sentence per-stock investment analysis
- **Batching**: 30 stocks per LLM call (~20 calls for 584 stocks)
- **Fallback**: if SDK missing or API error, all stocks keep deterministic estimates (zero breakage)
- **Schema fields**: `llm_pe_ratio`, `llm_forward_pe`, `llm_beta`, `llm_guidance`, etc. on `RatedStock`
- **`valuation_source`**: "llm" or "estimated" — tracks which data source was used
- **Data freshness**: LLM uses training knowledge (early 2025), not real-time data

When LLM data is available, it overrides the deterministic estimates for P/E, Beta, Return, Volatility. Derived metrics (Sharpe, Alpha, Risk Rating) are recomputed from LLM data.

### LLM-enriched catalyst timing (optional, requires ANTHROPIC_API_KEY)


`enrich_catalyst_llm()` in `catalyst_enrichment.py` uses Claude Haiku to identify upcoming corporate catalysts per stock:
- **Fields**: `catalyst_date` (YYYY-MM-DD), `catalyst_type` (earnings/fda/product_launch/conference/guidance/dividend/split/other), `catalyst_description`
- **Conviction adjustment**: ≤14 days: +2, ≤30 days: +1, >30 days or unknown: 0 — applied after base conviction, capped at 10
- **Schema fields**: `catalyst_date`, `catalyst_type`, `catalyst_conviction_adjustment`, `catalyst_source` on `RatedStock`
- **`catalyst_source`**: "llm" or "template" — tracks which catalyst path was used
- **Fallback**: if SDK missing or API error, all stocks keep template catalysts with adjustment=0

### LLM-enriched rotation narrative (optional, requires ANTHROPIC_API_KEY)

`generate_rotation_narrative_llm()` in `rotation_narrative.py` uses Claude Haiku to provide historical context for detected rotation patterns:
- **Single LLM call** (not batched) — one rotation analysis per pipeline run
- **Input**: rotation verdict, type, stage, source/destination clusters, regime, signals, velocity
- **Output**: 2-4 paragraph narrative (100-500 words) with historical parallels and implications
- **Schema field**: `rotation_narrative` (Optional[str], min_length=100) on `RotationDetectionOutput`
- **`narrative_source`**: "llm" or "template" — tracks which path was used
- **Fallback**: if SDK missing, API error, or verdict is "none", narrative stays None
- **Skips**: when verdict="none" (no rotation to narrate)

### LLM-enriched stop-loss tuning (optional, requires ANTHROPIC_API_KEY)

`enrich_stop_loss_llm()` in `stop_loss_tuning.py` uses Claude Haiku to recommend per-position trailing stop adjustments:
- **Batching**: 30 positions per LLM call
- **Input**: symbol, sector, tier, current stop %, conviction, asset type
- **Output**: `{symbol: {recommended_stop_pct, reason, volatility_flag}}`
- **Stop range clamping**: T1 (10-25%), T2 (8-20%), T3 (5-15%) via `compute_stop_recommendation()`
- **Volatility flags**: "high", "normal", "low"
- **Schema**: `StopLossRecommendation` model + `stop_loss_recommendations` list + `stop_loss_source` on `RiskAssessment`
- **`stop_loss_source`**: "llm" or "deterministic"
- **Fallback**: if SDK missing or API error, no recommendations added (empty list)

### LLM-enriched dynamic position sizing (optional, requires ANTHROPIC_API_KEY)

`enrich_sizing_llm()` in `dynamic_sizing.py` uses Claude Haiku for volatility-based position size adjustments:
- **Batching**: 30 positions per LLM call
- **Input**: symbol, sector, tier, conviction, target %, asset type, catalyst date
- **Output**: `{symbol: {volatility_score (1-10), size_adjustment_pct, reason}}`
- **Adjustment clamping**: T1 ±1.5%, T2 ±1.0%, T3 ±0.5% via `compute_size_adjustment()`
- **Schema fields**: `volatility_adjustment` (float, -2.0 to 2.0) and `sizing_source` on `PortfolioPosition`
- **`sizing_source`**: "llm" or "deterministic"
- **Fallback**: if SDK missing or API error, all positions keep 0.0 adjustment with "deterministic" source

### LLM-enriched reconciliation reasoning (optional, requires ANTHROPIC_API_KEY)

`enrich_rationale_llm()` in `reconciliation_reasoning.py` uses Claude Haiku for context-aware action rationales:
- **Batching**: 30 actions per LLM call
- **Input**: actions with type/pct + context (rotation verdict, rotation type, risk status, market regime)
- **Output**: `{symbol: enhanced_rationale}` — 1-2 sentence explanations incorporating rotation/risk/regime context
- **Schema field**: `rationale_source` on `ReconciliationOutput` — "llm" or "deterministic"
- **Pipeline**: `run_reconciler_pipeline()` accepts optional `rotation_output`, `strategy_output`, `risk_output` params
- **Fallback**: if SDK missing or API error, actions keep deterministic rationales

### Test structure

- `tests/unit/test_research_schema.py` — Level 1: pure Pydantic validation, no LLM, no file I/O
- `tests/unit/test_research_agent.py` — Level 2: deterministic pipeline tests with mock XLS data; Level 3: behavioral boundary tests
- `tests/unit/test_analyst_schema.py` — Level 1: elite screening, tier, conviction, sector score tests
- `tests/unit/test_analyst_agent.py` — Level 2: analyst pipeline tests, golden dataset verification, valuation integration tests, end-to-end Research→Analyst
- `tests/unit/test_valuation_metrics.py` — Pure function tests for all estimation formulas (P/E, PEG, Beta, Sharpe, Alpha, etc.)
- `tests/unit/test_valuation_llm.py` — LLM enrichment tests with mocked anthropic client (parsing, batching, error handling, pipeline integration)
- `tests/unit/test_catalyst_enrichment.py` — Catalyst enrichment tests (conviction adjustment logic, response parsing, date/type validation)
- `tests/unit/test_rotation_narrative.py` — Rotation narrative tool tests (generate_rotation_narrative_llm, graceful fallback)
- `tests/unit/test_stop_loss_tuning.py` — Stop-loss tuning tool tests (compute_stop_recommendation, _parse_stop_loss_response, constants)
- `tests/unit/test_dynamic_sizing.py` — Dynamic sizing tool tests (compute_size_adjustment, _parse_sizing_response, constants)
- `tests/unit/test_reconciliation_reasoning.py` — Reconciliation reasoning tool tests (_parse_reasoning_response, enrich_rationale_llm)
- `tests/unit/test_rotation_schema.py` — Level 1: rotation schema validation (enums, signals, flows, verdicts, narrative fields)
- `tests/unit/test_rotation_agent.py` — Level 2: rotation pipeline tests, individual signal detectors, narrative pipeline tests, behavioral boundaries, end-to-end chain, golden dataset
- `tests/unit/test_strategy_schema.py` — Level 1: strategy schema validation (allocation plans, theme recommendations, rotation signals, constants)
- `tests/unit/test_strategist_agent.py` — Level 2: strategist pipeline tests, tool function tests, behavioral boundaries, end-to-end Research→Analyst→Rotation→Strategy chain, golden dataset
- `tests/unit/test_portfolio_schema.py` — Level 1: portfolio schema validation (position limits, tier ranges, keeps count, stock/ETF split, dynamic sizing fields)
- `tests/unit/test_portfolio_agent.py` — Level 2: portfolio pipeline tests, tool functions, dynamic sizing pipeline tests, behavioral boundaries, end-to-end chain, golden dataset
- `tests/unit/test_risk_schema.py` — Level 1: risk schema validation (10 checks, veto rules, sleep well range, status consistency, StopLossRecommendation, stop-loss source)
- `tests/unit/test_risk_agent.py` — Level 2: risk pipeline tests, check functions, stop-loss pipeline tests, behavioral boundaries, end-to-end chain, golden dataset
- `tests/unit/test_returns_projection_schema.py` — Level 1: returns projection schema validation (scenarios, benchmarks, percentile ordering, drawdown ordering, constants)
- `tests/unit/test_returns_projector.py` — Level 2: returns projector pipeline tests, tool function tests, behavioral boundaries, end-to-end chain, golden dataset
- `tests/unit/test_reconciler_schema.py` — Level 1: reconciliation schema validation (money flow, 4 weeks, action types, rationale source)
- `tests/unit/test_reconciler_agent.py` — Level 2: reconciler pipeline tests, diff functions, rationale source pipeline tests, behavioral boundaries, end-to-end chain, golden dataset
- `tests/unit/test_educator_schema.py` — Level 1: educator schema validation (min counts, glossary size, summary lengths)
- `tests/unit/test_educator_agent.py` — Level 2: educator pipeline tests, behavioral boundaries, end-to-end chain, golden dataset
- `tests/unit/test_synthesis_schema.py` — Level 1: synthesis schema validation (key numbers, connections, contradictions, narrative lengths)
- `tests/unit/test_synthesizer_agent.py` — Level 2: synthesizer pipeline tests, key number extraction, contradiction detection, behavioral boundaries
- `tests/unit/test_regime_detector_schema.py` — Level 1: regime detector schema validation (enums, assessment models, validators, constants, backward-compat mapping)
- `tests/unit/test_regime_detector_agent.py` — Level 2: regime detector pipeline tests, classifier functions, decision tree, safety rules (MUST/MUST_NOT), mock data, golden dataset, end-to-end chain
- `tests/fixtures/conftest.py` — shared fixtures: `SAMPLE_IBD_STOCKS`, `PADDING_STOCKS`, `create_mock_xls()`, `create_mock_csv()`
- `golden_datasets/research_golden.json` — expected tier assignments and keep candidates for sample data
- `golden_datasets/analyst_golden.json` — expected tier assignments, elite results, IBD keeps, top sector
- `golden_datasets/rotation_golden.json` — expected rotation verdict, signals, type, regime for sample data
- `golden_datasets/strategy_golden.json` — expected min sectors, cash range, regime, tier target ranges for sample data
- `golden_datasets/portfolio_golden.json` — expected keeps count, sector/cash ranges, tier totals
- `golden_datasets/risk_golden.json` — expected check count, stress scenario count, sleep well range
- `golden_datasets/returns_projection_golden.json` — expected scenario count, benchmark count, caveats min, drawdown max, tier mix count
- `golden_datasets/reconciler_golden.json` — expected 4 weeks, week 1 liquidation, 14 keeps
- `golden_datasets/educator_golden.json` — expected min stock explanations, concepts, glossary, action items range
- `golden_datasets/synthesis_golden.json` — expected key numbers fields, min connections, action items range, synthesis source
- `golden_datasets/regime_detector_golden.json` — expected regime, confidence, exposure/health ranges, signal counts, legacy mapping for all 5 scenarios

### CrewAI compatibility

All tool imports use `try/except` for CrewAI — if crewai is not installed, tools fall back to inheriting from `pydantic.BaseModel`. This allows schema/pipeline tests to run without CrewAI installed.

## Python version

Requires Python >= 3.12 (uses `pyproject.toml` with modern type hints like `list[str]`). The venv is Python 3.9 but the project targets 3.12+.
