# Running the Project

## Main execution path

Run a full single experiment with generated input data:

```bash
uv run python -m run_manager.single_run
```

This entrypoint:

1. Generates a random project network and correlated activity-time inputs.
2. Configures the selected branching strategy and optimization settings.
3. Executes the stochastic branch-and-bound loop.
4. Prints summary metrics (experiment ID, elapsed time, expected cost, standard deviation).

## Quick environment checks

```bash
uv run python -V
uv run python -c "import run_manager, opt_manager, gen_manager; print('OK')"
```

## Common development commands

```bash
uv run ruff check src/ --fix
uv run ruff format src/
uv run mypy src/
uv run pre-commit run --all-files
```

## Documentation commands

This MkDocs project is configured under `documentation/mkdocs.yml`.

```bash
# live preview
uv run --group doc mkdocs serve -f documentation/mkdocs.yml

# build static docs
uv run --group doc mkdocs build -f documentation/mkdocs.yml --strict
```
