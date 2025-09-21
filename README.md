# GTO Trainer

Heads-up no-limit hold’em trainer delivered through a FastAPI web UI. The engine plays out full hands, evaluates every decision against the best available action, and reports EV loss so you can review mistakes.

## Requirements

- Python 3.12.11 (matches CI; managed via `pyenv` or similar).
- [uv](https://docs.astral.sh/uv/) for dependency management.

## Quick start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
uv sync --no-config --locked --extra dev          # create the virtualenv with dev extras
uv run --no-config --locked --extra dev -- scripts/run_ci_tests.sh
```

Common one-offs:

- `uv run --no-config --locked --extra dev -- pytest -q`
- `uv run --no-config --locked --extra dev -- ruff check .`
- `uv run --no-config --locked --extra dev -- ruff format .`

All commands include `--no-config` so local uv configuration (e.g. private indices) cannot diverge from the project lockfile.

## Tooling

### Git hooks

Enable the repo-managed hooks so commits run the same checks as CI:

```bash
git config core.hooksPath .githooks
```

`pre-commit` formats staged Python files, then runs Ruff format, Ruff lint, and pytest via `uv run --no-config --locked --extra dev -- …`. On success it drops `.git/.precommit_passed`. `pre-push` refuses to push if the marker is missing, which catches skipped or failed local checks.

### Make targets

```bash
make install-dev   # uv sync with dev extras
make check         # Ruff + pytest (CI equivalent)
make test          # pytest -q
make lint          # Ruff lint only
make format        # Ruff formatter
make render-smoke  # build the Render image and hit /healthz
```

## Running the trainer

### Web UI

The legacy Rich-based CLI has been removed. Use the web interface (or build against the HTTP API) for interactive sessions.

Local development server:

```bash
uv sync --no-config --locked --extra dev
uv run --no-config --locked --extra dev -- uvicorn gto_trainer.web.app:app --reload
```

Environment variables `HANDS` and `MC` control the default session size and Monte Carlo sample count.

Live demo: [gto-trainer.onrender.com](https://gto-trainer.onrender.com/)

### HTTP API

All HTTP routes are versioned under `/api/v1`.

- `POST /api/v1/session` — create a session (`{"session": "..."}` response).
- `GET /api/v1/session/{id}/node` — fetch the current node payload with options.
- `POST /api/v1/session/{id}/choose` — submit a choice and receive the next payload.
- `GET /api/v1/session/{id}/summary` — retrieve the aggregated session summary.

Legacy `/api/session/...` paths remain available for now but will be removed after downstream clients migrate.

Send the `HX-Request: true` header to receive HTML partials (node panel, feedback, or summary) that can be swapped directly into the UI via HTMX. JSON remains the default response shape when the header is absent.

## Tests and CI parity

CI runs `uv sync --no-config --locked --extra dev`, installs Playwright browsers, then executes Ruff format, Ruff lint, and `pytest -q`. Run `uv run --no-config --locked --extra dev -- python -m playwright install --with-deps chromium` once locally before the browser tests. After that, mirror CI with `uv run --no-config --locked --extra dev -- scripts/run_ci_tests.sh` or `make check`.

## Architecture overview

- **Episode generation** – `dynamic.generator` assembles preflop-to-river node trees using sampled seat assignments and rival styles defined in `_STYLE_LIBRARY`.
- **Range & equity modelling** – `dynamic.range_model`, `dynamic.hand_strength`, and `dynamic.preflop_mix` build playable ranges; `dynamic.equity` performs adaptive Monte Carlo and board runouts to price options.
- **Decision policy** – `dynamic.policy` exposes `options_for` and `resolve_for`, combining range samples, equity results, and rival profile updates to score each action and track hand state.
- **Session management** – `features.session.service.SessionManager` coordinates hand loops, formats options (`core.formatting`), aggregates results with `core.scoring`, and powers the session API routers used by the web/UI layers.
- **Solver data** – `solver.oracle.CSVStrategyOracle` loads optional preflop charts and falls back to the dynamic policy when a lookup misses; `CompositeOptionProvider` swaps between them per node.

## License

Proprietary (see `pyproject.toml`).

The standalone CLI was removed; use the web UI or HTTP API endpoints for all interactive or automated workflows.
