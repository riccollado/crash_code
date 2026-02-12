# CODEX.md

## Purpose

This file defines working conventions for Codex in the `crash_code` repository.
Follow these instructions when making code changes.

## Project Snapshot

- Domain: stochastic branch-and-bound optimization for project crashing under uncertainty
- Language: Python `>=3.12,<3.13`
- Package manager: `uv`
- Build backend: `hatchling`
- Main source root: `src/`

## Repository Layout

- `src/db_manager`: SQLAlchemy models and DB I/O
- `src/gen_manager`: network/distribution/scenario data generation
- `src/matrix_manager`: correlation/covariance matrix utilities
- `src/opt_manager`: optimization algorithms and bounds
- `src/run_manager`: end-to-end run entry points
- `tests`: test and validation scripts
- `sql`: SQL scripts used by experiments
- `scripts`: notebooks for analysis and visualization

## Environment Setup

```bash
uv python install 3.12
uv sync
```

Optional dependency groups:

```bash
uv sync --group open_opt
uv sync --group closed_opt
uv sync --group doc
uv sync --all-groups
```

## Common Commands

```bash
# Run one end-to-end optimization experiment
uv run python -m run_manager.single_run

# Lint and format
uv run ruff check src tests --fix
uv run ruff format src tests

# Type checking
uv run mypy src

# Full repository checks
uv run pre-commit run --all-files

# Additional static checks used by this repo
uv run vulture src --min-confidence 70
uv run bandit -r src --skip=B101,B301,B403,B605,B607
```

## Coding Standards

- Prefer small, focused changes and preserve existing module boundaries.
- Add type hints to all new/changed function signatures.
- Keep imports grouped and ordered (`stdlib`, third-party, local).
- Use clear `snake_case` names for functions/variables and `UPPER_CASE` for constants.
- Keep docstrings in NumPy style where docstrings are present.
- Avoid introducing notebook outputs in committed files.

## Validation Expectations

Before finishing substantial changes:

1. Run `uv run ruff check src tests --fix`.
2. Run `uv run ruff format src tests`.
3. Run `uv run mypy src`.
4. Run `uv run pre-commit run --all-files`.

If a step cannot run due to unavailable solver dependencies (for example, commercial
optimizer backends), document that limitation in your handoff.

## Config and Secrets

- Runtime configuration may depend on values in `.env` (database and related settings).
- Never hardcode credentials or connection strings in source files.

## Commit Message Convention

Use this format:

```text
<2-4 lines summarizing what changed and why>

1. **path/to/file.py**: concise change description
2. **path/to/other_file.py**: concise change description
```

- Use present tense.
- List each staged file exactly once.
- Do not include hashes, command transcripts, or extra sections.
