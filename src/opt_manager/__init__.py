"""Optimization core for stochastic branch-and-bound project crashing.

Implements the full SB&B pipeline: subproblem construction and solving,
knowledge-gradient branch selection, bound estimation, and the main
branch-and-bound loop with multiple branching strategies.

Modules
-------
optimize
    High-level entry point that initializes the database, prepares
    attributes, and launches the branch-and-bound algorithm (``optimize``).
stochastic
    Core SB&B loop with record-set partitioning, scenario assignment
    (Random, Distance, Pareto, KG), bound estimation, and bootstrap
    statistics (``branch_bound_algorithm``, ``initialize_attributes``).
knowledge_gradient
    Knowledge-gradient computation with correlated normal beliefs,
    including parallelized multi-alternative KG and Bayesian
    mean/covariance updates (``kg_alg``, ``kg_multi``, ``update_mu_s``).
generator_subproblem
    Builds and solves Gurobi MIP subproblems with crashing variables,
    activity scheduling, and penalty constraints (``optimize_subproblem``).
    Requires ``gurobipy``.
uncrashed_bounds
    Solves the uncrashed scheduling problem on a single scenario via Gurobi
    to find minimum project completion time (``uncrashed_project_time``).
    Requires ``gurobipy``.
uncrashed_bounds_pyomo
    Same uncrashed scheduling model solved via Pyomo/GLPK instead of Gurobi
    (``uncrashed_project_time``).  Requires ``pyomo``.
"""

__all__ = [
    "optimize",
    "stochastic",
    "knowledge_gradient",
    "generator_subproblem",
    "uncrashed_bounds",
    "uncrashed_bounds_pyomo",
]
