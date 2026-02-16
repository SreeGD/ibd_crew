# IBD Momentum Investment Framework v4.0

A multi-agent system that processes Investor's Business Daily (IBD) data files, Schwab thematic investment PDFs, and Motley Fool recommendations to discover, rate, and organize growth stock opportunities.

**This framework never makes buy or sell recommendations.** It only discovers, scores, and organizes.

## Architecture

The system runs 13 agents in a deterministic pipeline, each producing a validated Pydantic schema:

| Agent | Role | Output |
|-------|------|--------|
| 01 | **Research Agent** | Discovers stocks from IBD XLS/CSV, IBD PDFs, Motley Fool PDFs, Schwab theme PDFs |
| 02 | **Analyst Agent** | Elite screening, tier assignment (T1/T2/T3), conviction scoring, sector ranking |
| 03 | **Rotation Detector** | 5-signal sector rotation detection, market regime classification |
| 04 | **Sector Strategist** | Regime-adjusted sector allocations, theme ETF recommendations |
| 05 | **Portfolio Manager** | 3-tier portfolio construction with position sizing, keeps, trailing stops |
| 06 | **Risk Officer** | 10-check risk assessment, stress tests, Sleep Well scores |
| 07 | **Returns Projector** | Bull/base/bear scenario projections, benchmark comparison (SPY/DIA/QQQ) |
| 08 | **Portfolio Reconciler** | Current vs recommended diff, KEEP/SELL/BUY/ADD/TRIM actions, 4-week plan |
| 09 | **Educator** | Plain-language explanations of every decision with analogies and glossary |
| 10 | **Executive Synthesizer** | Unified investment thesis, cross-agent connections, contradiction detection |
| 11 | **Value Investor** | Deep value screening using Morningstar data (moat, fair value, margin of safety) |
| 12 | **PatternAlpha** | Technical pattern recognition and enhanced scoring |
| 13 | **Historical Analyst** | Historical context via ChromaDB vector store (optional) |

### Pipeline Flow

```
01 Research → 02 Analyst → 03 Rotation → 04 Strategy
                                              ↓
                              11 Value ← 02 Analyst → 12 Pattern
                                    ↓                      ↓
                              05 Portfolio Manager (integrates all)
                                    ↓
                              06 Risk → 07 Returns → 08 Reconciler
                                                          ↓
                              09 Educator ← ─ ─ ─ ─ ─ ─ ─┘
                                    ↓
                              10 Synthesizer → 13 Historical (optional)
```

### Dual Execution Paths

Each agent supports two modes:
- **Deterministic pipeline** (`run_*_pipeline()`) — no LLM, used by tests and default runs
- **Agentic pipeline** (`build_*_agent()` + `build_*_task()`) — CrewAI agent with tools and LLM reasoning

### LLM-Enhanced Features (Optional)

When `ANTHROPIC_API_KEY` is set, several tools use Claude Haiku for enrichment:
- Valuation metrics (real P/E, beta, guidance)
- Catalyst timing (earnings dates, FDA approvals)
- Rotation narrative (historical parallels)
- Stop-loss tuning (per-position recommendations)
- Dynamic position sizing (volatility adjustments)
- Reconciliation reasoning (context-aware rationales)
- Executive synthesis narrative

All LLM features have deterministic fallbacks — the pipeline runs fully without any API key.

## Data Sources

Place data files in the `data/` directory:

```
data/
├── ibd_xls/         # IBD Excel/CSV exports (highest priority)
├── ibd_pdf/         # IBD PDF reports (Smart Table, Tech Leaders, Top 200)
├── fool_pdf/        # Motley Fool PDF recommendations
├── schwab_themes/   # Schwab thematic investment PDFs
├── Morningstar/     # Morningstar stock data (for value analysis)
├── portfolios/      # Brokerage PDF statements (for reconciliation)
└── ibd_history/     # Historical IBD data (for Agent 13)
```

## Setup

### Requirements

- Python >= 3.12
- Dependencies listed in `requirements.txt`

### Installation

```bash
# Clone and install
git clone <repo-url>
cd ibd_crew
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Environment Variables (Optional)

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

## Usage

### Full Pipeline

```bash
# Run all 13 agents with default data/ and output/ directories
python run_pipeline.py

# Custom directories
python run_pipeline.py --data mydata --output results

# Include historical analysis (requires ChromaDB)
python run_pipeline.py --historical
```

### Incremental Runs

Resume from any agent using `--from`:

```bash
# Resume from Agent 05 (loads 01-04, 11, 12 from snapshots)
python run_pipeline.py --from 5

# Resume from Agent 08 (reconciler only)
python run_pipeline.py --from 8

# Resume from Agent 10 with historical
python run_pipeline.py --from 10 --historical
```

Snapshots are saved in `output/snapshots/` after each agent completes.

### Programmatic Usage

```python
from ibd_agents.agents.research_agent import run_research_pipeline
from ibd_agents.agents.analyst_agent import run_analyst_pipeline
from ibd_agents.agents.rotation_detector import run_rotation_pipeline
from ibd_agents.agents.sector_strategist import run_strategist_pipeline
from ibd_agents.agents.portfolio_manager import run_portfolio_pipeline
from ibd_agents.agents.risk_officer import run_risk_pipeline
from ibd_agents.agents.returns_projector import run_returns_projector_pipeline
from ibd_agents.agents.portfolio_reconciler import run_reconciler_pipeline

research = run_research_pipeline("data")
analyst = run_analyst_pipeline(research)
rotation = run_rotation_pipeline(analyst, research)
strategy = run_strategist_pipeline(rotation, analyst)
portfolio = run_portfolio_pipeline(strategy, analyst)
risk = run_risk_pipeline(portfolio, strategy)
returns = run_returns_projector_pipeline(portfolio, rotation, risk, analyst)
reconciliation = run_reconciler_pipeline(
    portfolio, analyst, returns_output=returns,
    rotation_output=rotation, strategy_output=strategy, risk_output=risk,
)
```

## Output

The pipeline generates the following in the `output/` directory:

| File | Description |
|------|-------------|
| `framework_report_YYYY-MM-DD.pdf` | Comprehensive PDF report (30+ pages) |
| `agent01_research_YYYY-MM-DD.xlsx` | Stock universe with validation scores |
| `agent02_analyst_YYYY-MM-DD.xlsx` | Rated stocks with tiers, conviction, valuation |
| `agent03_rotation_YYYY-MM-DD.xlsx` | Rotation signals and sector flows |
| `agent04_strategy_YYYY-MM-DD.xlsx` | Sector allocations and theme recommendations |
| `agent05_portfolio_YYYY-MM-DD.xlsx` | Portfolio positions with sizing and stops |
| `agent06_risk_YYYY-MM-DD.xlsx` | Risk checks, stress tests, Sleep Well scores |
| `agent07_returns_YYYY-MM-DD.xlsx` | Scenario projections and benchmark comparisons |
| `agent08_reconciler_YYYY-MM-DD.xlsx` | Position actions with implementation plan |
| `agent09_educator_YYYY-MM-DD.xlsx` | Educational explanations and glossary |
| `agent10_executive_summary_YYYY-MM-DD.xlsx` | Synthesis with cross-agent connections |
| `agent11_value_investor_YYYY-MM-DD.xlsx` | Value screening results |
| `agent12_pattern_alpha_YYYY-MM-DD.xlsx` | Pattern recognition scores |
| `snapshots/` | JSON snapshots for incremental runs |

## Testing

```bash
# All tests (1107 tests, no LLM required)
pytest

# Unit tests only
pytest tests/unit/

# By marker
pytest -m schema      # Schema validation (no LLM)
pytest -m behavior    # Behavioral boundary tests
pytest -m integration # Cross-agent integration tests
pytest -m llm         # Tests that call real LLM (requires API key)

# Single test file
pytest tests/unit/test_portfolio_agent.py
```

## Key Concepts

### Three-Tier Portfolio

| Tier | Label | Target % | Criteria |
|------|-------|----------|----------|
| T1 | Momentum | 35-45% | Comp >= 95, RS >= 90, EPS >= 80 |
| T2 | Quality Growth | 33-40% | Comp >= 85, RS >= 80, EPS >= 75 |
| T3 | Defensive | 20-30% | Comp >= 80, RS >= 75, EPS >= 70 |
| Cash | — | 2-15% | — |

### Position Selection Sources

Positions are selected from four pools:
- **Keep** (~14 positions): Pre-committed fundamental and IBD picks
- **Value** (~4 positions): Top deep-value picks from Agent 11
- **Pattern** (~4 positions): Top pattern leaders from Agent 12
- **Momentum** (~14 positions): Highest-conviction stocks from Agent 02

### Elite Screening (4 Gates)

All four must pass for a stock to be "elite":
1. EPS Rating >= 85
2. RS Rating >= 75
3. SMR Rating: A, B, or B-
4. Acc/Dis Rating: A, B, or B-

### Risk Metrics

Each position includes:
- **Sharpe Ratio**: (Return - 4.5%) / Volatility
- **Alpha**: CAPM excess return vs S&P 500
- **Trailing Stop**: T1=22%, T2=18%, T3=12%
- **Max Loss**: Per-position max loss limit by tier and asset type

## Project Structure

```
ibd_crew/
├── src/ibd_agents/
│   ├── agents/           # 13 agent modules (pipeline + CrewAI)
│   ├── schemas/           # Pydantic output contracts
│   ├── tools/             # Pure function tools + CrewAI wrappers
│   └── reports/           # PDF report generator
├── tests/
│   ├── unit/              # Unit tests (schema, agent, tool)
│   └── fixtures/          # Shared test fixtures
├── golden_datasets/       # Expected outputs for golden tests
├── data/                  # Input data files
├── output/                # Generated reports and snapshots
├── run_pipeline.py        # Main pipeline entry point
├── pyproject.toml         # Project metadata and dependencies
└── requirements.txt       # Pip requirements
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, coding standards, and the PR process.

## License

MIT License. See [LICENSE](LICENSE) for details.
