# CLAUDE.md

## Project overview

**Learning to Crash** is a stochastic branch-and-bound optimization system for the
project crashing problem under uncertainty. It finds optimal activity crashing plans
for project networks with PERT-distributed activity times and deadline penalty
functions.

## Tech stack

- **Language:** Python 3.12 (strict: `>=3.12,<3.13`)
- **Package manager:** `uv` (lock file: `uv.lock`)
- **Build backend:** hatchling
- **Solvers:** Gurobi (primary), Pyomo/GLPK, HiGHS, PuLP, AMPL, CPLEX
- **Core libraries:** numpy, scipy, pandas, networkx, sqlalchemy, matplotlib

## Project structure

```
src/
  db_manager/       # SQLAlchemy models and database I/O (PostgreSQL)
  gen_manager/      # Problem instance generation (networks, distributions, costs)
  matrix_manager/   # Correlation/covariance matrix utilities (Higham algorithm)
  opt_manager/      # Core optimization: stochastic B&B, knowledge gradient, bounds
  run_manager/      # End-to-end experiment orchestration
tests/              # Test suite
scripts/            # Jupyter notebooks for analysis and visualization
docs/               # Project documentation
```

## Common commands

```bash
# Install dependencies
uv sync                          # Default (includes dev group)
uv sync --group open_opt         # Open-source solvers
uv sync --group closed_opt       # Commercial solvers (Gurobi, CPLEX)
uv sync --all-groups             # Everything

# Run the optimizer
uv run python -m run_manager.single_run

# Linting and formatting
ruff check src/ --fix            # Lint (replaces flake8, isort, pydocstyle)
ruff format src/                 # Format (replaces black)

# Type checking
mypy src/

# Pre-commit (runs ruff, mypy, vulture, bandit, nbstripout, nbqa)
pre-commit run --all-files

# Other quality checks
vulture src/ --min-confidence 70 # Dead code detection
bandit -r src/ --skip=B101,B301,B403,B605,B607  # Security linting
```

## Code conventions

- **Formatting:** 88-character line length (ruff/black compatible)
- **Type hints:** Required on all function signatures; use `typing` module types and
  `numpy.typing.NDArray`
- **Docstrings:** NumPy-style with Parameters/Returns sections
- **Imports:** stdlib, then third-party, then local; sorted by isort (black profile);
  no wildcard imports
- **Naming:** `snake_case` for functions/variables, `UPPER_CASE` for constants
- **Security:** Uses `secrets.SystemRandom()` for randomness; DB credentials via
  environment variables

## Commit message format

Start with 2-4 plain-text lines describing intent/context, then a blank line, then a
numbered list with every staged file exactly once:

```
<summary of what and why>

1. **path/to/file.ext**: <concise change description>
2. **path/to/other.ext**: <concise change description>
```

Use present tense. No commands, hashes, or extra sections.

## Key details

- Pre-commit hooks are configured and enforced; always run `pre-commit run --all-files`
  before committing
- The `.env` file contains database connection parameters
- Notebook cell outputs must be stripped before commit (nbstripout hook)
- Bandit skips: B101 (assert), B301 (pickle), B403 (pickle import), B605/B607
  (subprocess)
