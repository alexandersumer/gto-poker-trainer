# GTO Trainer

GTO Trainer is a heads-up no-limit hold’em practice environment with both a Textual CLI and a FastAPI web UI. Each scenario deals a full rival hand, lets the opponent react street by street, and reports the EV delta for every action you take.

## Project status

- **Live demo** – [gto-trainer.onrender.com](https://gto-trainer.onrender.com/)

## Requirements

- Python 3.12.11 exactly (`pyenv` or another version manager is recommended to match CI).

## Quick start (uv)

1. Install [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
2. Sync dependencies (app + dev extras): `uv sync --no-config --locked --extra dev`.
3. Run everything in one go: `uv run --no-config --locked --extra dev -- scripts/run_ci_tests.sh`.

Common commands:

- `uv run --no-config --locked --extra dev -- pytest -q`
- `uv run --no-config --locked --extra dev -- ruff check .`
- `uv run --no-config --locked --extra dev -- ruff format .`

## Git hooks

Configure Git to use the repo-managed hooks so staged Python files are auto-formatted before every commit and pushes are blocked when pre-commit fails:

```bash
git config core.hooksPath .githooks
```

The pre-commit hook formats staged Python files and then runs the same Ruff + pytest trio as CI via `uv run --no-config --locked --extra dev -- …`, marking success in `.git/.precommit_passed`. The pre-push hook checks that marker, so pushes only proceed if the most recent pre-commit run passed.

## Install & run (CLI)

```bash
uv sync --no-config --locked
uv run --no-config --locked -- gto-trainer --hands 5
```

Run in-place without installing:

```bash
uv run --no-config --locked -- python -m gto_trainer
```

### CLI options

```
gto-trainer [--hands N] [--seed N] [--mc N] [--no-color] [--solver-csv PATH]
```

- `--hands N` — number of hands to play (default `1`).
- `--seed N` — RNG seed (omit for randomness).
- `--mc N` — Monte Carlo samples per node (default `200`).
- `--solver-csv PATH` — optional preflop CSV to seed opening ranges.
- `--no-color` — disable ANSI colors if your terminal strips them.

Controls inside the CLI: `1–9` choose an action, `h` opens contextual help, `?` shows pot + SPR, `q` quits.

## Web UI

- **Live demo** – [gto-trainer.onrender.com](https://gto-trainer.onrender.com/)
- **Local** – Install dev extras and launch FastAPI with reload enabled:

  ```bash
  uv sync --no-config --locked --extra dev
  uv run --no-config --locked --extra dev -- uvicorn gto_trainer.web.app:app --reload
  ```

Environment overrides: `HANDS` and `MC` mirror the CLI flags when exported before launch.

## How it works

- **Simulation loop** – Each hand is generated from sampled preflop ranges, then walked street by street with Monte Carlo rollouts (`--mc`) to stabilise EV estimates.
- **Solver logic** – Post-flop options blend heuristics with lookup data; when a CSV is supplied, the trainer wraps it in a composite provider that falls back to dynamic sizing rules.
- **EV math** – For every action we store `best_ev`, `chosen_ev`, and compute `ev_loss = best_ev - chosen_ev`, rolling those numbers into session-level accuracy and EV summaries.
- **Rival model** – Rival decisions come from range tightening plus fold / continue sampling, so the opponent profile updates as stacks and pot sizes change.

## Local development

If you prefer Make targets:

```bash
make install-dev   # uv sync with dev extras
make check         # lint + tests (matches CI)
make test          # pytest -q
make lint          # Ruff lint
make format        # Ruff formatter
make render-smoke  # build Docker image & hit /healthz (Render parity)
```

CI runs the same trio as `uv run --no-config --locked --extra dev -- scripts/run_ci_tests.sh` / `make check` (`ruff check`, `pytest -q`).

## Solver architecture (quick tour)

- **Episode generator** – `src/gto_trainer/dynamic/generator.py` creates preflop→river node trees, alternating blinds via `SeatRotation` so training covers both positions.
- **Trainer loop** – `SessionManager` (and CLI/web adapters) request actions from `dynamic.policy`, cache option lists defensively, and record outcomes for scoring.
- **Solver logic** – `dynamic.policy` samples rival ranges, runs equity Monte Carlo with adaptive precision, and emits `Option` objects carrying EVs, justifications, and metadata for resolution.
- **Rival model** – `dynamic.rival_strategy` consumes cached range profiles to decide folds/calls/raises so postflop play mirrors solver frequencies instead of perfect clairvoyance.
- **Resolution & scoring** – `dynamic.policy.resolve_for` applies actions, updates stacks/pot state, and `core.scoring` aggregates EV loss into session summaries.

## License

Proprietary (see `pyproject.toml`).
