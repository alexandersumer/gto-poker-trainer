# GTO Trainer (Multi‑Street Only)

Interactive CLI trainer that generates fresh hands on the fly and quizzes you from preflop through river in a single hand. Each decision is graded by EV (in big blinds), with step‑by‑step feedback and an end‑of‑session summary of top leaks.

## Quick start

- Install editable (recommended) then run the entry point:
  - `pip install -e .`
  - `gto-poker-trainer-cli`
- Or run in-place without installing:
  - `PYTHONPATH=src python -m gto_poker_trainer_cli` (from the project root), or
  - `cd src && python -m gto_poker_trainer_cli`

## Development (Python 3.12.11)

- This project requires exactly Python 3.12.11 (see `.python-version`).
- Ensure Python 3.12.11 is active (pyenv users: `.python-version` is provided).
- Install dev deps and run tests:
  - `make install-dev`
  - `make test`

### Linting and formatting

- Ruff is configured for linting, import sorting, and formatting.
- Common tasks:
  - `make lint` – check lint
  - `make fix` – auto-fix lint (incl. unused imports, sort imports)
  - `make format` – apply code formatting

The test suite includes end-to-end CLI tests that feed interactive input, validate per-step EV math, final summaries, leak ordering, color/TTY behavior, and optional preflop solver CSV integration.

## CLI usage

Single mode only (multi‑street play). The optional `play` word is accepted for compatibility but not required.

- `gto-poker-trainer-cli [--hands N] [--seed N] [--mc N] [--no-color|--force-color] [--solver-csv PATH]`

## Notes

- Per-step scoring: `ev_loss = best_ev - chosen_ev`.
- Session stats: total/average EV lost, hit rate, and a normalized 0–100 score.
