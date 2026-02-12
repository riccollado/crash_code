Learning to Crash
==========

A Reinforcement Learning Approach to Project Scheduling
-------------------------------------------------------

This project provides a command-line tool that applies a stochastic branch-and-bound optimization method to the classic project crashing problem under uncertainty. In this setting, task durations are uncertain (PERT-style) and missing deadlines incurs threshold penalties. The tool explores the space of crash decisions and uses principled sampling and selection rules to efficiently search for high-quality solutions.

Overview
--------

The goal is to determine an optimal crashing plan for a project network with uncertain activity times. We build on:

- A stochastic branch-and-bound approach to activity crashing ([paper](https://pubsonline.informs.org/doi/pdf/10.1287/ijoc.12.2.125.11894?casa_token=PHCfqHAG120AAAAA:BsTfR2bDQEtx3tlkzJKbYcMAoSdDEcr65TkYU49hMCOUfULXn32p-9Li6bhKLWL-UpttA4DecBhA "A Stochastic Branch-and-Bound Approach to ActivityCrashing in Project Management")).
- A general framework for optimal allocation under uncertainty ([paper](https://pubsonline.informs.org/doi/pdf/10.1287/opre.46.3.381?casa_token=QsdLQM3thP0AAAAA:INj4Dv_NYAD48aM_odTL9AKv4dJHsbIguQSgHucoBmkDhPjoM5j8Z1kM16sZTXuANemOHEcp9kYT "On Optimal Allocation of Indivisibles Under Uncertainty")).
- The knowledge-gradient policy for correlated normal beliefs ([paper](https://pubsonline.informs.org/doi/pdf/10.1287/ijoc.1080.0314?casa_token=mADfuyTiLiMAAAAA:_NP3QhLLq_8ghTjK31heitjBhxa_YbEcEy0ng9QfaQlcGGtpusX7YrCMbfIarnGTNNQHHx76PJ9n "The Knowledge-Gradient Policy for COrrelated Normal Beliefs")).

Inputs:

- A project network in GraphML (XML).
- PERT three-point estimates (optimistic, most likely, pessimistic) for each activity.
- A covariance matrix for correlated activity times.
- Thresholds describing time-overrun penalties.
- A branching decision method: "KG", "Random", "Uniform", "Distance", "Pareto_Inverse", or "Pareto_Boltzman".
- The desired number of samples.

Output:

- An XML file containing the solution found by the stochastic branch-and-bound method for the selected branching strategy.
- Optionally, run metadata and intermediate solutions recorded in a SQL database for later analysis.

Branching strategies:

1. KG: Knowledge gradient with correlated normal beliefs ([paper](https://pubsonline.informs.org/doi/pdf/10.1287/ijoc.1080.0314?casa_token=mADfuyTiLiMAAAAA:_NP3QhLLq_8ghTjK31heitjBhxa_YbEcEy0ng9QfaQlcGGtpusX7YrCMbfIarnGTNNQHHx76PJ9n "The Knowledge-Gradient Policy for COrrelated Normal Beliefs")).
2. Random: Random branching selection ([paper](https://pubsonline.informs.org/doi/pdf/10.1287/ijoc.12.2.125.11894?casa_token=PHCfqHAG120AAAAA:BsTfR2bDQEtx3tlkzJKbYcMAoSdDEcr65TkYU49hMCOUfULXn32p-9Li6bhKLWL-UpttA4DecBhA "A Stochastic Branch-and-Bound Approach to ActivityCrashing in Project Management")).
3. Uniform: Sample uniformly from all available B&B leaves at each main iteration.
4. Distance: Non-dominated sorting of mean-variance pairs by vector distance ([paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=996017&casa_token=RX5FX8Ctu38AAAAA:BymQiux3DQammBgBVQANxxHhwDx5fhxT3FqRNB8nCvyND4WSajGqwvjyKNpISKO5aJj2akki&tag=1 "A Fast and Elitist Multiobjective Genetic Algorithm:")).
5. Pareto_Inverse: Non-dominated sorting with inverse-proportional selection pressure:
   p_i = ((1/i)^β) / Σ_k (1/k)^β, with fronts sized s_i and Σ s_i p_i = 1.
6. Pareto_Boltzman: Non-dominated sorting with Boltzmann-style selection:
   p_i = e^{-β i} / Σ_k s_k e^{-β k}, with fronts sized s_i and Σ s_i p_i = 1.

Getting Started
---------------

Prerequisites (system)

- Graphviz: required for graph visualization.
  - macOS: brew install graphviz
  - Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y graphviz
  - Windows: <https://graphviz.org/download/>
- Gurobi (optional, required if you use Gurobi-backed solvers):
  - Download and install: <https://www.gurobi.com/downloads/>
  - Obtain and activate a license: <https://www.gurobi.com/documentation/quickstart.html>

Install uv

- Docs: <https://docs.astral.sh/uv/>
- macOS / Linux: curl -LsSf <https://astral.sh/uv/install.sh> | sh
- Windows: powershell -ExecutionPolicy ByPass -c "irm <https://astral.sh/uv/install.ps1> | iex"

Clone the repository

- git clone <https://github.com/><your-username>/crash_code.git
- cd crash_code

Install Python and dependencies

uv manages both the Python distribution and all project packages. The required
Python version (`>=3.12,<3.13`) is declared in `pyproject.toml` and pinned in
`.python-version`.

```bash
# Install the required Python version (one-time)
uv python install 3.12

# Install project dependencies (includes the dev group by default)
uv sync
```

Install pre-commit hooks

- pre-commit is included in the dev dependency group.
- Install hooks defined in [.pre-commit-config.yaml](.pre-commit-config.yaml):
  - uv run pre-commit install
- Run on all files once:
  - uv run pre-commit run --all-files

Quick verification

- Check the managed Python version: uv run python -V
- Import the package modules to verify path resolution:
  - uv run python -c "import run_manager, opt_manager, gen_manager; print('OK')"

Dependency groups
-----------------

Dependencies are organized into groups in `pyproject.toml`:

| group | purpose | enable | disable |
| ----- | ------- | ------ | ------- |
| `dev` | Linting, typing, notebooks, pre-commit. | default | `uv sync --no-group dev` |
| `doc` | MkDocs docs build stack. | `uv sync --group doc` | `uv sync --no-group doc` |
| `open_opt` | Open-source optimization backends. | `uv sync --group open_opt` | `uv sync --no-group open_opt` |
| `closed_opt` | Commercial optimization solvers. | `uv sync --group closed_opt` | `uv sync --no-group closed_opt` |
| `package` | Packaging/publishing helpers. | `uv sync --group package` | `uv sync --no-group package` |

Useful patterns:

```bash
# Runtime-only install (no default groups)
uv sync --no-default-groups

# Install only docs environment
uv sync --only-group doc

# Install all environments
uv sync --all-groups
```

Usage
-----

- Single run with generated data:
  - uv run python -m run_manager.single_run
  - See [src/run_manager/single_run.py](src/run_manager/single_run.py) for available arguments and defaults.
- Core optimization entry points:
  - Orchestrator: [src/opt_manager/optimize.py](src/opt_manager/optimize.py)
  - Main stochastic B&B algorithm: [src/opt_manager/stochastic.py](src/opt_manager/stochastic.py)
- Database (optional):
  - Schema files live under [sql/](sql/). Create your database and apply these scripts as needed.
  - Connection details and how the project reads them are implemented in [src/db_manager/driver.py](src/db_manager/driver.py). Configure your environment (e.g., via [.env](.env)) accordingly.
- Outputs:
  - Figures and artifacts are placed in [output/](output/).

File Description
----------------

Top-level

- [README.md](README.md): Project overview and instructions.
- [.env](.env): Environment variable definitions (local development).
- [.flake8](.flake8), [.pylintrc](.pylintrc), [mypy.ini](mypy.ini): Linting and typing configuration.
- [.pre-commit-config.yaml](.pre-commit-config.yaml): Pre-commit hook configuration.
- [.python-version](.python-version): The Python version used by uv.
- [pyproject.toml](pyproject.toml): Build system (hatchling), dependency groups, and tool configuration.
- [uv.lock](uv.lock): Locked dependency versions for reproducible installs.
- [.markdownlint.json](.markdownlint.json): Markdown style rules.
- [copilot_commit_instructions.md](copilot_commit_instructions.md): Commit message format guidelines.
- [.vscode/settings.json](.vscode/settings.json): Editor defaults for VS Code.
- [sql/](sql): SQL scripts for database schema and support.
- [tests/](tests): Test scripts (e.g., `kg_test.py` for knowledge-gradient validation).
- [output/](output): Generated figures and outputs.
- [presentation/crash_learning.pdf](presentation/crash_learning.pdf): Project presentation.

Scripts (Jupyter)

- [scripts/histogram_learning_sequence.ipynb](scripts/histogram_learning_sequence.ipynb): Notebook to build comparison histograms from results data for learning quality assessment.
- [scripts/network_figure.ipynb](scripts/network_figure.ipynb): Notebook to generate the project network figures used in documentation/output.
- [scripts/scratch.ipynb](scripts/scratch.ipynb): Scratchpad for experiments and quick prototyping.

Source code (src)

- Database manager:
  - [src/db_manager/__init__.py](src/db_manager/__init__.py): Package initializer.
  - [src/db_manager/driver.py](src/db_manager/driver.py): Initializes a PostgreSQL database via SQLAlchemy and returns callable helpers to push experiments, iterations, solutions, update timing, and close the session.

- Data generation (inputs and scenarios):
  - [src/gen_manager/__init__.py](src/gen_manager/__init__.py): Package initializer.
  - [src/gen_manager/covariance.py](src/gen_manager/covariance.py): Generates random correlation matrices via the nearest-correlation algorithm.
  - [src/gen_manager/crash.py](src/gen_manager/crash.py): Generates crash alternatives (time reductions and associated costs) per activity.
  - [src/gen_manager/distribution.py](src/gen_manager/distribution.py): Generates PERT beta distributions for activity durations and geometric probabilities used to parameterize them.
  - [src/gen_manager/network.py](src/gen_manager/network.py): Generates connected layered DAGs with configurable density and renders PDF network figures.
  - [src/gen_manager/penalty.py](src/gen_manager/penalty.py): Produces linear/exponential penalty values and computes penalty bounds by solving uncrashed scheduling problems on most-likely and pessimistic scenarios.
  - [src/gen_manager/scenario.py](src/gen_manager/scenario.py): Generates correlated activity-duration scenarios via a Gaussian Copula over PERT beta distributions.

- Matrix utilities:
  - [src/matrix_manager/__init__.py](src/matrix_manager/__init__.py): Package initializer.
  - [src/matrix_manager/nearest_correlation.py](src/matrix_manager/nearest_correlation.py): Higham's nearest-correlation-matrix algorithm (projects a symmetric matrix onto the positive-semidefinite cone).
  - [src/matrix_manager/utilities.py](src/matrix_manager/utilities.py): Positive-definiteness checks and covariance/correlation matrix conversions.

- Optimization core:
  - [src/opt_manager/__init__.py](src/opt_manager/__init__.py): Package initializer.
  - [src/opt_manager/generator_subproblem.py](src/opt_manager/generator_subproblem.py): Builds and solves Gurobi MIP subproblems with crashing variables, activity scheduling, and penalty constraints.
  - [src/opt_manager/knowledge_gradient.py](src/opt_manager/knowledge_gradient.py): Knowledge-gradient computation with correlated normal beliefs, including parallelized multi-alternative KG and Bayesian mean/covariance updates.
  - [src/opt_manager/optimize.py](src/opt_manager/optimize.py): High-level orchestration: prepares inputs, commits problems to the database, and invokes the branch-and-bound loop.
  - [src/opt_manager/stochastic.py](src/opt_manager/stochastic.py): Core stochastic branch-and-bound loop with record-set partitioning, scenario assignment (Random, Distance, Pareto, KG), bound estimation, and bootstrap statistics.
  - [src/opt_manager/uncrashed_bounds.py](src/opt_manager/uncrashed_bounds.py): Solves the uncrashed scheduling problem on a single scenario via Gurobi to find minimum project completion time.
  - [src/opt_manager/uncrashed_bounds_pyomo.py](src/opt_manager/uncrashed_bounds_pyomo.py): Same uncrashed scheduling model solved via Pyomo/GLPK instead of Gurobi.

- Run manager:
  - [src/run_manager/__init__.py](src/run_manager/__init__.py): Package initializer.
  - [src/run_manager/single_run.py](src/run_manager/single_run.py): Main executable that generates a random problem instance, configures the SB&B method, and runs the optimization pipeline.

Development
-----------

- Run tests (if present under tests/):
  - uv run pytest -q
- Linting and formatting (ruff replaces black, isort, flake8, pydocstyle):
  - uv run ruff check src/ --fix
  - uv run ruff format src/
- Type checks:
  - uv run mypy src/
- Dead code detection:
  - uv run vulture src/ --min-confidence 70
- Security linting:
  - uv run bandit -r src/ --skip=B101,B301,B403,B605,B607
- Pre-commit (runs all of the above plus notebook stripping):
  - uv run pre-commit run --all-files

Database
--------

- Initialize your database using the scripts under [sql/](sql).
- Connection configuration and usage reside in [src/db_manager/driver.py](src/db_manager/driver.py). Adjust your [.env](.env) or environment variables to match the expected settings in that module (e.g., engine URL, credentials).
- Ensure your DB server is reachable before running long experiments.

Troubleshooting
---------------

- Graphviz not found:
  - Install the system package and ensure the binaries are on PATH (see "Prerequisites" above).
- Gurobi license errors:
  - Verify that GUROBI_HOME is set and grbgetkey has been executed for your license.
- Wrong Python version:
  - Run `uv python install 3.12` and ensure `.python-version` contains `3.12`.
- Import errors when running modules:
  - Use uv to run Python: uv run python -m run_manager.single_run
- Stale lock file:
  - Run `uv lock` to regenerate `uv.lock` after editing dependencies in `pyproject.toml`.

Contributing
------------

- Fork and create a feature branch.
- Install dependencies: uv sync
- Enable pre-commit hooks: uv run pre-commit install
- Keep changes typed and linted (mypy, ruff).
- Add/adjust tests where applicable.
- Submit a PR with a clear description and rationale.

References
----------

- Activity crashing under uncertainty (stochastic B&B): <https://pubsonline.informs.org/doi/pdf/10.1287/ijoc.12.2.125.11894>
- Optimal allocation under uncertainty: <https://pubsonline.informs.org/doi/pdf/10.1287/opre.46.3.381>
- Knowledge gradient for correlated beliefs: <https://pubsonline.informs.org/doi/pdf/10.1287/ijoc.1080.0314>
- NSGA-II (non-dominated sorting): <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=996017>
