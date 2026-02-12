# Installation

This project is fully managed with `uv` (Python runtime + package resolution).

## System prerequisites

- **Graphviz** (required for network visualization)
  - macOS: `brew install graphviz`
  - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y graphviz`
  - Windows: <https://graphviz.org/download/>
- **Gurobi** (optional; required only for Gurobi-backed optimization)
  - Download: <https://www.gurobi.com/downloads/>
  - License quickstart: <https://www.gurobi.com/documentation/quickstart.html>

## Install uv

- Documentation: <https://docs.astral.sh/uv/>
- macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Install Python and dependencies

The required Python range is defined in `pyproject.toml` as `>=3.12,<3.13`.

```bash
# one-time interpreter install
uv python install 3.12

# sync project dependencies (includes default group(s))
uv sync
```

## Dependency groups

| group | purpose | enable | disable |
| ----- | ------- | ------ | ------- |
| `dev` | Linting, typing, notebooks, pre-commit | default | `uv sync --no-group dev` |
| `doc` | MkDocs documentation toolchain | `uv sync --group doc` | `uv sync --no-group doc` |
| `open_opt` | Open-source optimization stack | `uv sync --group open_opt` | `uv sync --no-group open_opt` |
| `closed_opt` | Commercial solver stack | `uv sync --group closed_opt` | `uv sync --no-group closed_opt` |
| `package` | Packaging/publishing helpers | `uv sync --group package` | `uv sync --no-group package` |

Useful patterns:

```bash
# runtime-only (no default groups)
uv sync --no-default-groups

# docs-only environment
uv sync --only-group doc

# all groups
uv sync --all-groups
```

## Optional development hooks

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```
