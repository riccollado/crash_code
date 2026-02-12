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

Getting Started (pyenv + Poetry)
--------------------------------

Prerequisites (system)

- Graphviz: required for graph visualization.
  - macOS: brew install graphviz
  - Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y graphviz
  - Windows: <https://graphviz.org/download/>
- Gurobi (optional, required if you use Gurobi-backed solvers):
  - Download and install: <https://www.gurobi.com/downloads/>
  - Obtain and activate a license: <https://www.gurobi.com/documentation/quickstart.html>

Install pyenv

- Docs: <https://github.com/pyenv/pyenv>
- macOS (Homebrew): brew update && brew install pyenv
- Ubuntu/Debian:
  - sudo apt-get update && sudo apt-get install -y build-essential curl git zlib1g-dev libssl-dev libreadline-dev libbz2-dev libsqlite3-dev
  - curl <https://pyenv.run> | bash
- Add to shell (bash example):
  - echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
  - echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
  - echo 'eval "$(pyenv init -)"' >> ~/.bashrc
  - source ~/.bashrc

Install Poetry

- Docs: <https://python-poetry.org/docs/#installation>
- Recommended:
  - curl -sSL <https://install.python-poetry.org> | python3 -
  - Ensure Poetry is on your PATH (e.g., export PATH="$HOME/.local/bin:$PATH")

Clone the repository

- git clone <https://github.com/><your-username>/crash_code.git
- cd crash_code

Set the Python version with pyenv

- If the file .python-version exists (it does in this repo), install that exact version:
  - pyenv install -v "$(cat .python-version)"    # one-time
  - pyenv local "$(cat .python-version)"
- Alternatively, install the known compatible version (e.g., 3.8 series):
  - pyenv install -v 3.8.12
  - pyenv local 3.8.12

Create and use a Poetry virtual environment

- Point Poetry to pyenv's Python:
  - poetry env use "$(pyenv which python)"
- Install project dependencies:
  - poetry install

Install pre-commit hooks

- Ensure pre-commit is available (it's typically installed as a dev dependency by Poetry; otherwise: poetry add -D pre-commit)
- Install hooks defined in [.pre-commit-config.yaml](.pre-commit-config.yaml):
  - poetry run pre-commit install
- Run on all files once:
  - poetry run pre-commit run --all-files

Quick verification

- Print Poetry environment info: poetry env info
- Launch a Python REPL in the env: poetry run python -V
- Import the package modules to verify the path resolution:
  - poetry run python -c "import run_manager, opt_manager, gen_manager; print('OK')"

Usage
-----

- Single run with generated data:
  - poetry run python -m run_manager.single_run
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
- [.python-version](.python-version): The Python version used by pyenv.
- [pyproject.toml](pyproject.toml): Build system and dependency specification (Poetry).
- [mkdocs.yml](mkdocs.yml): MkDocs configuration (if using documentation site).
- [.vscode/settings.json](.vscode/settings.json): Editor defaults for VS Code.
- [sql/](sql): SQL scripts for database schema and support.
- [output/](output): Generated figures and outputs.
- [presentation/crash_learning.pdf](presentation/crash_learning.pdf): Project presentation.

Scripts (Jupyter)

- [scripts/histogram_learning_sequence.ipynb](scripts/histogram_learning_sequence.ipynb): Notebook to build comparison histograms from results data for learning quality assessment.
- [scripts/network_figure.ipynb](scripts/network_figure.ipynb): Notebook to generate the project network figures used in documentation/output.
- [scripts/scratch.ipynb](scripts/scratch.ipynb): Scratchpad for experiments and quick prototyping.

Source code (src)

- Database manager:
  - [src/db_manager/__init__.py](src/db_manager/__init__.py): Package initializer.
  - [src/db_manager/driver.py](src/db_manager/driver.py): Database driver and session/engine management. Encapsulates persistence of runs, solutions, and iterations (see [sql/](sql) for schema).

- Data generation (inputs and scenarios):
  - [src/gen_manager/__init__.py](src/gen_manager/__init__.py): Package initializer.
  - [src/gen_manager/covariance.py](src/gen_manager/covariance.py): Creates/validates covariance matrices for correlated activity durations.
  - [src/gen_manager/crash.py](src/gen_manager/crash.py): Generates crash alternatives (time reductions and associated costs) per activity.
  - [src/gen_manager/distribution.py](src/gen_manager/distribution.py): Builds PERT distributions from three-point estimates; utilities for sampling.
  - [src/gen_manager/network.py](src/gen_manager/network.py): Generates connected project networks (GraphML I/O and utilities).
  - [src/gen_manager/penalty.py](src/gen_manager/penalty.py): Produces linear/exponential penalty functions for tardiness thresholds.
  - [src/gen_manager/scenario.py](src/gen_manager/scenario.py): Samples scenarios from PERT distributions under correlation.

- Matrix utilities:
  - [src/matrix_manager/__init__.py](src/matrix_manager/__init__.py): Package initializer.
  - [src/matrix_manager/nearest_correlation.py](src/matrix_manager/nearest_correlation.py): Higham's nearest correlation algorithm (conversion/repair of covariance to valid correlation).
  - [src/matrix_manager/utilities.py](src/matrix_manager/utilities.py): Numerical helpers (matrix ops, stability fixes, transformations).

- Optimization core:
  - [src/opt_manager/__init__.py](src/opt_manager/__init__.py): Package initializer.
  - [src/opt_manager/generator_subproblem.py](src/opt_manager/generator_subproblem.py): Defines and solves subproblems with partially fixed variables (e.g., Gurobi-backed intermediate models).
  - [src/opt_manager/knowledge_gradient.py](src/opt_manager/knowledge_gradient.py): Knowledge-gradient acquisition with correlated normal beliefs for branching decisions.
  - [src/opt_manager/optimize.py](src/opt_manager/optimize.py): High-level orchestration: prepares inputs, commits problems to the database, and invokes the branch-and-bound loop.
  - [src/opt_manager/stochastic.py](src/opt_manager/stochastic.py): Stochastic branch-and-bound implementation; includes bootstrap and Pareto-based branching logic.
  - [src/opt_manager/uncrashed_bounds.py](src/opt_manager/uncrashed_bounds.py): Computes bounds from the baseline (uncrashed, unpenalized) schedule on a single scenario.
  - [src/opt_manager/uncrashed_bounds_pyomo.py](src/opt_manager/uncrashed_bounds_pyomo.py): Alternative bounds computation using Pyomo.

- Run manager:
  - [src/run_manager/__init__.py](src/run_manager/__init__.py): Package initializer.
  - [src/run_manager/single_run.py](src/run_manager/single_run.py): Convenience runner for a single experiment with generated inputs and configured method.

Development
-----------

- Run tests (if present under tests/):
  - poetry run pytest -q
- Type checks:
  - poetry run mypy src
- Linting:
  - poetry run flake8
  - poetry run pylint src
- Pre-commit:
  - poetry run pre-commit run --all-files

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
- Poetry cannot find Python:
  - poetry env use "$(pyenv which python)" after setting pyenv local.
- Import errors when running modules:
  - Use Poetry to run Python: poetry run python -m run_manager.single_run

Contributing
------------

- Fork and create a feature branch.
- Enable pre-commit hooks: poetry run pre-commit install
- Keep changes typed and linted (mypy, flake8, pylint).
- Add/adjust tests where applicable.
- Submit a PR with a clear description and rationale.

References
----------

- Activity crashing under uncertainty (stochastic B&B): <https://pubsonline.informs.org/doi/pdf/10.1287/ijoc.12.2.125.11894>
- Optimal allocation under uncertainty: <https://pubsonline.informs.org/doi/pdf/10.1287/opre.46.3.381>
- Knowledge gradient for correlated beliefs: <https://pubsonline.informs.org/doi/pdf/10.1287/ijoc.1080.0314>
- NSGA-II (non-dominated sorting): <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=996017>

License
-------

This repository does not declare a license in the root. If you intend to use or redistribute the code, please contact the repository owner or add a
