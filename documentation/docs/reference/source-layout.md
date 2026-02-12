# Source Layout

## Top-level runtime packages (`src/`)

- `db_manager`: SQLAlchemy models and persistence helpers for experiments,
  iterations, and outputs.
- `gen_manager`: generation of networks, distributions, covariance structures,
  penalties, and scenarios.
- `matrix_manager`: nearest-correlation algorithm and matrix utility functions.
- `opt_manager`: stochastic branch-and-bound core, knowledge-gradient logic,
  subproblem optimization, and orchestration.
- `run_manager`: executable run entrypoints.

## Notebooks (`scripts/`)

- `histogram_learning_sequence.ipynb`: histogram-based result comparison.
- `network_figure.ipynb`: network-graph visualization.
- `scratch.ipynb`: ad-hoc prototyping and exploratory analysis.

## Operational files

- `sql/`: schema and SQL support scripts.
- `output/`: generated plots and artifacts.
- `.env` / `.env_sample`: environment-based runtime configuration.
- `pyproject.toml`: package metadata, dependency groups, and tooling config.

## Documentation reference assets

- `documentation/docs/reference/*.md`: technical notes, API pages, and research-library
  summaries used by MkDocs.
- `documentation/docs/reference/papers/*.pdf`: supporting papers and presentations linked
  directly from the documentation navigation.
