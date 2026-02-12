Learning to Crash
==========

A Reinforcement Learning Approach to Project Scheduling
-------------------------------------------------------

This repository contains a CLI tool to implement a stochastic branch & bound
optimization method for the problem of crashing a project activity network subject to
uncertain activity times and threshold penalties.

Overview
--------

This application approximates a solution to the project management problem of finding an optimal crashing plan with uncertain activity times. The basic problem description can be found [here](https://pubsonline.informs.org/doi/pdf/10.1287/ijoc.12.2.125.11894?casa_token=PHCfqHAG120AAAAA:BsTfR2bDQEtx3tlkzJKbYcMAoSdDEcr65TkYU49hMCOUfULXn32p-9Li6bhKLWL-UpttA4DecBhA "A Stochastic Branch-and-Bound Approach to ActivityCrashing in Project Management").
The solution approach follows the stochastic branch & bound method presented in [here](https://pubsonline.informs.org/doi/pdf/10.1287/opre.46.3.381?casa_token=QsdLQM3thP0AAAAA:INj4Dv_NYAD48aM_odTL9AKv4dJHsbIguQSgHucoBmkDhPjoM5j8Z1kM16sZTXuANemOHEcp9kYT "On Optimal Allocation of Indivisibles Under Uncertainty") and applies the knowledge gradient technique introduced [here](https://pubsonline.informs.org/doi/pdf/10.1287/ijoc.1080.0314?casa_token=mADfuyTiLiMAAAAA:_NP3QhLLq_8ghTjK31heitjBhxa_YbEcEy0ng9QfaQlcGGtpusX7YrCMbfIarnGTNNQHHx76PJ9n "The Knowledge-Gradient Policy for COrrelated Normal Beliefs").

The problem requires the following input data:

1. A project network graph in the GraphML format (supplied as an XML file).

2. Lists of pessimistic, most likely, and optimistic, values to describing the PERT
   activity time distributions.

3. A covariance matrix describing the linear relationships among activity times.

4. Threshold values describing penalties due to time overrun.

5. A selection of branching decision method to apply from: "KG", "Random", "Uniform",
   "Distance", "Pareto\_Inverse", and "Pareto\_Boltzman".

6. Number of samples that the user wants to consider.

The output of the CLI is an XML file with the solution obtained after applying the
stochastic branch & bound method with the selected branching decision.

The branching strategies are described below:

 1. **KG**: Branching selection via the knowledge gradient with correlated normal beliefs method discussed [here](https://pubsonline.informs.org/doi/pdf/10.1287/ijoc.1080.0314?casa_token=mADfuyTiLiMAAAAA:_NP3QhLLq_8ghTjK31heitjBhxa_YbEcEy0ng9QfaQlcGGtpusX7YrCMbfIarnGTNNQHHx76PJ9n "The Knowledge-Gradient Policy for COrrelated Normal Beliefs").

 2. **Random**: Random branching selection as discussed [here](https://pubsonline.informs.org/doi/pdf/10.1287/ijoc.12.2.125.11894?casa_token=PHCfqHAG120AAAAA:BsTfR2bDQEtx3tlkzJKbYcMAoSdDEcr65TkYU49hMCOUfULXn32p-9Li6bhKLWL-UpttA4DecBhA "A Stochastic Branch-and-Bound Approach to ActivityCrashing in Project Management").

 3. **Uniform**: Branching decision rule based on sampling on every available B&B leaf
    at every main iteration step.

 4. **Distance**: Branching decision obtained by forming the Pareto frontier of mean-variance pairs of each leaf and performing non-dominated sorting based on vector distance. The non-dominated sorting method is discussed [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=996017&casa_token=RX5FX8Ctu38AAAAA:BymQiux3DQammBgBVQANxxHhwDx5fhxT3FqRNB8nCvyND4WSajGqwvjyKNpISKO5aJj2akki&tag=1 "A Fast and Elitist Multiobjective Genetic Algorithm:").

 5. **Pareto\_Inverse**: Branching decision obtained by forming the Pareto frontier of
    mean-variance pairs of each leaf and performing non-dominated sorting based on
    inverse proportional probability:

<p>
<img src="https://latex.codecogs.com/svg.image?\mathbf{p_i&space;=&space;\frac{\left(1/i\right)^\beta}{\sum_{k=1}^{n}\left(1/k\right)^\beta}}" title="p_i = \frac{\left(1/i\right)^\beta}{\sum_{k=1}^{n}\left(1/k\right)^\beta}", />
</p>

<p>
where <img src="https://latex.codecogs.com/svg.image?\mathbf{\beta}" title="\beta" /> is the Pareto front selection pressure, <img src="https://latex.codecogs.com/svg.image?\mathbf{n}" title="n" /> is the total number of fronts, <img src="https://latex.codecogs.com/svg.image?\mathbf{s_i}" title="s_i" /> is the total number of solutions in front of rank <img src="https://latex.codecogs.com/svg.image?\mathbf{i}" title="i" />, and <img src="https://latex.codecogs.com/svg.image?\inline&space;\mathbf{&space;\sum_{i=1}^n&space;s_ip_i&space;=&space;1&space;}" title="\sum_{i=1}^n s_ip_i = 1" />.

6. **Pareto\_Boltzman**: Branching decision obtained by forming the Pareto frontier of
   mean-variance pairs of each leaf and performing non-dominated sorting based on
   inverse Boltzman probability:

<p>
<img src="https://latex.codecogs.com/svg.image?\mathbf{&space;&space;p_i&space;=&space;\frac{e^{-\beta&space;i}}{\sum_{k=1}^n&space;s_k&space;e^{-\beta&space;k}}&space;&space;&space;}" title="p_i = \frac{e^{-\beta i}}{\sum_{k=1}^n s_k e^{-\beta k}}", />
</p>

where <img src="https://latex.codecogs.com/svg.image?\mathbf{\beta}" title="\beta" /> is the Pareto front selection pressure, <img src="https://latex.codecogs.com/svg.image?\mathbf{n}" title="n" /> is the total number of fronts, <img src="https://latex.codecogs.com/svg.image?\mathbf{s_i}" title="s_i" /> is the total number of solutions in front of rank <img src="https://latex.codecogs.com/svg.image?\mathbf{i}" title="i" />, and <img src="https://latex.codecogs.com/svg.image?\inline\mathbf{&space;&space;\sum_{i=1}^n&space;s_ip_i&space;=&space;1&space;&space;&space;}" title="\sum_{i=1}^n s_ip_i = 1" />.

The application also stores every solution and iteration in a SQL database for later
retrieval and statistical analysis.

Dependencies
------------

This is a `uv`-managed project. Python and packages are resolved from
`pyproject.toml` (`requires-python = ">=3.12,<3.13"`).

Setup and run (`uv`)
--------------------

```bash
# Install uv first: https://docs.astral.sh/uv/
uv python install 3.12
uv sync
uv run python -m run_manager.single_run
```

Main packages
-------------

Core runtime:
`numpy`, `scipy`, `pandas`, `networkx`, `matplotlib`, `pydot`, `sqlalchemy`,
`pyfiglet`, `bootstrapped`, `jellyfish`, `statsmodels`, `ace-tools-open`.

Optimization/tooling (optional by environment):
`pyomo`, `glpk`, `highspy`, `pulp`, `amplpy`, `gurobipy`, `cplex`, `pygmo`.

Environment groups (`pyproject.toml`)
-------------------------------------

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

Project files
-------------

Scripts:

| path | use |
| ---- | --- |
| `scripts/histogram_learning_sequence.ipynb` | Build comparison histograms from experiment outputs. |
| `scripts/network_figure.ipynb` | Visualize generated project networks. |
| `scripts/scratch.ipynb` | Ad-hoc exploration and experiments. |

Source modules:

| path | use |
| ---- | --- |
| `src/db_manager/__init__.py` | Package marker. |
| `src/db_manager/driver.py` | SQLAlchemy models and DB I/O helpers for experiments, iterations, and solutions. |
| `src/gen_manager/__init__.py` | Package marker. |
| `src/gen_manager/covariance.py` | Generate random correlation/covariance structures. |
| `src/gen_manager/crash.py` | Generate crash-time and crash-cost vectors. |
| `src/gen_manager/distribution.py` | Generate PERT activity-time distributions and geometric probabilities. |
| `src/gen_manager/network.py` | Generate layered DAG project networks and figures. |
| `src/gen_manager/penalty.py` | Build penalty functions and penalty bounds. |
| `src/gen_manager/scenario.py` | Sample correlated activity-time scenarios. |
| `src/matrix_manager/__init__.py` | Package marker. |
| `src/matrix_manager/nearest_correlation.py` | Nearest-correlation-matrix implementation (Higham-based). |
| `src/matrix_manager/utilities.py` | Matrix and covariance/correlation utility helpers. |
| `src/opt_manager/__init__.py` | Package marker. |
| `src/opt_manager/generator_subproblem.py` | Build and solve subproblems for fixed branching decisions. |
| `src/opt_manager/knowledge_gradient.py` | Knowledge-gradient routines for branch selection. |
| `src/opt_manager/optimize.py` | Orchestrate DB initialization and optimization execution. |
| `src/opt_manager/stochastic.py` | Core stochastic branch-and-bound algorithm and selection logic. |
| `src/opt_manager/uncrashed_bounds.py` | Compute uncrashed baseline bounds with Gurobi. |
| `src/opt_manager/uncrashed_bounds_pyomo.py` | Alternative uncrashed-bounds model via Pyomo/GLPK. |
| `src/run_manager/__init__.py` | Package marker. |
| `src/run_manager/single_run.py` | End-to-end single experiment runner with random data generation and output. |
