# Contributing to IBD Momentum Investment Framework

Thanks for your interest in contributing! This guide covers setup, standards, and the PR process.

## Getting Started

### Prerequisites

- Python >= 3.12
- Git

### Dev Setup

```bash
# Fork and clone
git clone https://github.com/<your-username>/ibd_crew.git
cd ibd_crew

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in editable mode
pip install -e .

# Verify setup
pytest tests/unit/ -q
```

All 1100+ tests should pass without any API key or data files.

### Optional: LLM features

Copy `.env.example` to `.env` and add your Anthropic API key to enable LLM-enhanced features. The pipeline works fully without it.

## Project Structure

```
src/ibd_agents/
├── agents/     # Pipeline orchestration (one file per agent)
├── schemas/    # Pydantic output contracts (the source of truth)
├── tools/      # Pure functions + CrewAI wrappers
└── reports/    # PDF/Excel report generators

tests/
├── unit/       # Unit tests (schema, agent, tool, behavioral)
└── fixtures/   # Shared test data and conftest.py

golden_datasets/ # Expected outputs for golden tests
```

## Making Changes

### 1. Create a branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Follow existing patterns

- **Schemas first**: If adding a new output field, add it to the Pydantic schema in `schemas/` first.
- **Pure functions**: Business logic goes in `tools/` as pure functions (no LLM, no I/O). The CrewAI `BaseTool` wrapper calls the pure function.
- **Deterministic path**: Every agent must have a `run_*_pipeline()` that works without LLM. LLM enrichment is always optional.
- **LLM fallback**: Any LLM call must have a `try/except` fallback that returns deterministic results.

### 3. Write tests

Every change needs tests. We use pytest with markers:

| Marker | Description | Requires LLM? |
|--------|-------------|----------------|
| `@pytest.mark.schema` | Pydantic validation, pure functions | No |
| `@pytest.mark.behavior` | Behavioral boundaries (what the system must NOT do) | No |
| `@pytest.mark.integration` | Cross-agent chain tests | No |
| `@pytest.mark.llm` | Tests that call a real LLM | Yes |

Most contributions should only need `schema` and `behavior` tests.

### 4. Run the test suite

```bash
# Run all unit tests (must pass)
pytest tests/unit/ -v

# Run only your test file
pytest tests/unit/test_your_file.py -v

# Run by marker
pytest -m schema -v
```

### 5. Check the golden datasets

If your change affects agent output structure, verify golden tests still pass:

```bash
pytest -k golden -v
```

If a golden test legitimately needs updating, update the JSON file in `golden_datasets/` and explain why in your PR.

## Submitting a Pull Request

1. Push your branch and open a PR against `master`.
2. Fill out the PR template.
3. Ensure CI passes (all tests green).
4. Describe **what** changed and **why** in the PR description.

### PR Checklist

- [ ] Tests added/updated for the change
- [ ] All existing tests pass (`pytest tests/unit/`)
- [ ] No secrets or API keys committed
- [ ] Golden datasets updated if schema changed
- [ ] README/CLAUDE.md updated if adding a new agent, tool, or schema field

## Code Style

- Use type hints (Python 3.12+ style: `list[str]` not `List[str]`).
- Docstrings on public functions.
- Keep functions small and focused.
- No external API calls in pure functions.

## What to Contribute

Good first issues:
- Add tests for edge cases in existing tools
- Improve error messages in schema validators
- Add new data source parsers in `tools/`
- Improve PDF report formatting in `reports/`

Larger contributions:
- New screening agents (follow the existing agent pattern)
- Additional risk checks
- New output formats (HTML reports, dashboards)

## Questions?

Open an issue on GitHub and we'll help you get started.
