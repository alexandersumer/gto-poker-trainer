# gtotrainer

Heads-up no-limit hold’em training built on FastAPI with a responsive web UI. The engine simulates full hands, evaluates every decision, and reports EV loss so you can drill and review.

Hosted app: [gto.alexandersumer.com](https://gto.alexandersumer.com/)

## Contents

- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Local Development](#local-development)
  - [Web UI](#web-ui)
  - [HTTP API](#http-api)
- [Tooling](#tooling)
  - [Git Hooks](#git-hooks)
  - [Make Targets](#make-targets)
- [Tests](#tests)
- [Architecture](#architecture)
- [License](#license)

## Requirements

- Python **3.13.7** (matches CI and container images; manage with `pyenv` or similar).
- [uv](https://docs.astral.sh/uv/) for dependency and tool management.

## Quick Start

1. Install uv if it is not already available.
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Sync dependencies with the locked dev environment.
   ```bash
   uv sync --no-config --locked --extra dev --no-build-isolation-package eval7
   ```
3. Run the trainer locally.
   ```bash
   uv run --no-config --locked --extra dev -- uvicorn gtotrainer.web.app:app --reload
   ```

Common follow-ups:

- Run tests: `uv run --no-config --locked --extra dev -- pytest -q`
- Lint: `uv run --no-config --locked --extra dev -- ruff check .`
- Format: `uv run --no-config --locked --extra dev -- ruff format .`

Every `uv` command intentionally includes `--no-config` so local configuration cannot override the locked resolver settings.

## Local Development

### Web UI

`uvicorn` serves the full web experience at `http://127.0.0.1:8000`. Set optional environment variables to seed sessions:

- `HANDS` — default number of scenarios in a session.
- `MC` — Monte Carlo sample count per evaluation.

### HTTP API

All routes are versioned under `/api/v1`.

| Method & Path | Purpose |
| --- | --- |
| `POST /api/v1/session` | Create a session (`{"session": "..."}`). |
| `GET /api/v1/session/{id}/node` | Fetch the current node payload and available options. |
| `POST /api/v1/session/{id}/choose` | Submit a decision and receive the next node. |
| `GET /api/v1/session/{id}/summary` | Retrieve the aggregated session summary. |

Include `HX-Request: true` to receive HTML partials (node panel, feedback, summary) for HTMX swaps. JSON is returned when the header is absent.

## Tooling

### Git Hooks

Point Git to the repo-managed hooks so local commits mirror CI checks:

```bash
git config core.hooksPath .githooks
```

The `pre-commit` hook formats staged Python files, runs Ruff format and lint, then executes pytest via uv. On success it writes `.git/.precommit_passed`, which the `pre-push` hook requires before allowing pushes.

### Make Targets

| Target | Description |
| --- | --- |
| `make install-dev` | Run the uv sync command with dev extras. |
| `make check` | Run Ruff lint and pytest (CI parity). |
| `make test` | Run `pytest -q`. |
| `make lint` | Run Ruff lint. |
| `make format` | Run the Ruff formatter. |
| `make render-smoke` | Build the Render image and call `/healthz`. |

## Tests

CI installs Playwright, runs Ruff format/lint, and executes the full pytest suite (unit, solver, calibration, browser). To mirror that flow:

```bash
uv run --no-config --locked --extra dev -- python -m playwright install --with-deps chromium
uv run --no-config --locked --extra dev -- scripts/run_ci_tests.sh
```

Helpful shortcuts:

- `make check` – Ruff + full pytest (CI parity).
- `pytest -m "not browser"` – skip the Playwright runs when iterating quickly.
- `pytest tests/calibration` – lock regression output against saved solver snapshots.

## Architecture

- **Episode generation** – `dynamic.generator` assembles preflop-to-river node trees using sampled seat assignments and rival profiles from `_STYLE_LIBRARY`.
- **Ranges & equities** – `dynamic.range_model`, `dynamic.hand_strength`, and `dynamic.preflop_mix` maintain weighted combo ranges street-to-street; `dynamic.equity` batches Monte Carlo trials through `eval7`, NumPy, and Numba with deterministic caching.
- **Decision policy** – `dynamic.policy` exposes `options_for` / `resolve_for`, carrying range weights into postflop, building river subgames (fold/call/jam branches), and feeding them into the CFR backend (`dynamic.cfr.LocalCFRBackend`) with multi-action payoffs.
- **Rival modelling** – `dynamic.rival_strategy` consumes the stored weights to bias fold/call decisions, sample jams, and adapt to repeated hero aggression without peeking at hero cards.
- **Session flow** – `features.session.service.SessionManager` drives the hand loop, formats options, aggregates results, and serves both JSON and HTML fragments via FastAPI routers. Blocking work runs in an async-aware worker pool (`features.session.concurrency`).
- **Solver data** – `solver.oracle.CSVStrategyOracle` loads optional preflop charts and falls back to the dynamic policy when no chart applies; `CompositeOptionProvider` swaps between them per node.
- **Calibration guardrails** – `tests/calibration` locks select solver outputs so regressions in EV, fold rates, or rival mixes are caught automatically.

## License

Distributed under the terms of the [MIT License](LICENSE).
