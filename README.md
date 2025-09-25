# gtotrainer

Heads-up no-limit hold'em training built with FastAPI and a responsive web UI. Full hands are simulated, every decision is scored, and EV deltas make it easy to drill and review.

Hosted app: [gto.alexandersumer.com](https://gto.alexandersumer.com/)

## Prerequisites

- Python **3.13.7** (matches CI and the published container images).
- [uv](https://docs.astral.sh/uv/) for dependency and tool management.

## Set Up

1. Install uv (if not already available).
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Sync the locked development environment.
   ```bash
   uv sync --no-config --locked --extra dev --no-build-isolation-package eval7
   ```
   `--no-config` keeps local uv settings from drifting off the locked resolver.

## Run The Trainer Locally

Start the FastAPI app with live reload:
```bash
uv run --no-config --locked --extra dev -- uvicorn gtotrainer.web.app:app --reload
```
The app serves the full web UI at `http://127.0.0.1:8000`.

Optional environment variables:

- `HANDS` - default number of scenarios to queue in a session.
- `MC` - Monte Carlo sample count per evaluation.

## Tests & Quality Gates

- Unit + integration tests: `uv run --no-config --locked --extra dev -- pytest -q`
- Lint: `uv run --no-config --locked --extra dev -- ruff check .`
- Format: `uv run --no-config --locked --extra dev -- ruff format .`
- CI parity (Playwright + full suite):
  ```bash
  uv run --no-config --locked --extra dev -- python -m playwright install --with-deps chromium
  uv run --no-config --locked --extra dev -- scripts/run_ci_tests.sh
  ```

## HTTP API

All routes live under `/api/v1`:

| Method & Path | Purpose |
| --- | --- |
| `POST /api/v1/session` | Start a session (`{"session": "..."}`). |
| `GET /api/v1/session/{id}/node` | Retrieve the current node payload and available options. |
| `POST /api/v1/session/{id}/choose` | Submit a decision and fetch the next node. |
| `GET /api/v1/session/{id}/summary` | Return aggregated session results. |

Add `HX-Request: true` to get HTML partials (for HTMX swaps); omit it for JSON.

## Developer Tooling

- Git hooks live in `.githooks`. Enable them with `git config core.hooksPath .githooks`.
- Helpful Make targets:

  | Target | Description |
  | --- | --- |
  | `make install-dev` | Runs the uv sync command above. |
  | `make check` | Runs Ruff lint and pytest (CI parity). |
  | `make lint` / `make format` | Ruff lint and formatter. |
  | `make test` | `pytest -q`. |
  | `make render-smoke` | Build the Render image and hit `/healthz`. |

## Architecture Highlights

- **Episode generation** - `dynamic.generator` builds preflop-to-river trees with sampled seats and rival profiles.
- **Ranges & equities** - `dynamic.range_model`, `dynamic.hand_strength`, and `dynamic.preflop_mix` keep combo weights updated; `dynamic.equity` batches Monte Carlo trials through `eval7`, NumPy, and Numba.
- **Decision policy** - `dynamic.policy` resolves options via the CFR backend (`dynamic.cfr.LocalCFRBackend`) and returns multi-action payoffs.
- **Session flow** - `features.session.service.SessionManager` orchestrates the loop, formats responses, and serves both JSON and HTML fragments.

## License

Distributed under the terms of the [MIT License](LICENSE).
