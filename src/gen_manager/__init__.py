"""Problem-instance generation for project crashing experiments.

Builds every component of a random crashing problem: the project network,
PERT activity-time distributions, correlation structures, crash
alternatives, penalty functions, and correlated duration scenarios.

Modules
-------
covariance
    Random correlation matrix generation via the nearest-correlation
    algorithm (``generate_cov_mat``).
crash
    Random crash-time and crash-cost vectors
    (``generate_crash_times``, ``generate_crash_cost``).
distribution
    PERT beta distributions from geometric-probability parameters
    (``generate_pert_distributions``, ``generate_geometric``).
network
    Layered DAG construction with configurable density and PDF rendering
    (``generate_network``).
penalty
    Linear/exponential penalty values and penalty-bound estimation
    (``generate_penalty_vals_linear``, ``generate_penalty_vals_exponential``,
    ``generate_penalty_bounds``).
scenario
    Correlated activity-duration scenarios via Gaussian Copula
    (``dynamic_scenarios``).
"""

__all__ = [
    "covariance",
    "crash",
    "distribution",
    "network",
    "penalty",
    "scenario",
]
