# API Overview

This page provides a direct module index for the codebase.

## Database manager

- `src/db_manager/driver.py`: SQLAlchemy models and database I/O helpers for
  experiments, iterations, and solutions.

## Data generation

- `src/gen_manager/network.py`: network skeleton and graph generation.
- `src/gen_manager/distribution.py`: PERT-Beta and geometric-distribution
  generators.
- `src/gen_manager/scenario.py`: Gaussian-copula scenario generation.
- `src/gen_manager/penalty.py`: penalty function values and bounds.

## Optimization

- `src/opt_manager/knowledge_gradient.py`: knowledge-gradient scoring and
  updates.
- `src/opt_manager/generator_subproblem.py`: subproblem model creation/solve
  routines.
- `src/opt_manager/stochastic.py`: stochastic branch-and-bound core loop.
- `src/opt_manager/optimize.py`: orchestration entrypoint.

## Matrix utilities

- `src/matrix_manager/utilities.py`: covariance/correlation transformations and
  PSD checks.
- `src/matrix_manager/nearest_correlation.py`: nearest-correlation matrix
  routine.

## Run entrypoint

- `src/run_manager/single_run.py`: single experiment execution workflow.
