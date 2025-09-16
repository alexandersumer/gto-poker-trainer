# GTO Poker Trainer CLI

A command-line trainer for heads-up no‑limit hold’em decisions across all streets.

It deals a random hand, shows your hole cards on every street, and asks you to
pick an action. After each choice, you see the EV comparison and a short
explanation. A session summary at the end totals results and highlights the
largest EV losses.

## Requirements

- Python 3.12.11 exactly (see `.python-version`).

## Install and run

Editable install (recommended):

```
pip install -e .
gto-poker-trainer-cli
```

Run in place without installing (from the project root):

```
PYTHONPATH=src python -m gto_poker_trainer_cli
# or
cd src && python -m gto_poker_trainer_cli
```

## Usage

Single mode: multi‑street play. The optional word `play` is accepted but not required.

```
gto-poker-trainer-cli [--hands N] [--seed N] [--mc N] [--no-color|--force-color] [--solver-csv PATH]
```

Options:

- `--hands N` Number of random hands per session (default: 1).
- `--seed N` RNG seed for reproducible sessions (omit for randomness).
- `--mc N` Monte Carlo trials per node for EV estimates (default: 200).
- `--no-color` Disable color output.
- `--force-color` Force color output even if not a TTY.
- `--solver-csv PATH` Use a preflop strategy CSV before falling back to heuristics.

Controls during play:

- `1–9` choose an action
- `h` help
- `?` show pot and SPR info
- `q` quit the current session

## Development

- Ensure Python 3.12.11 is active (pyenv users: `.python-version` is provided).
- Install dev dependencies and run tests:

```
make install-dev
make test
```

Linting and formatting (Ruff):

```
make lint     # check
make fix      # auto-fix
make format   # apply formatting
```

## Notes

- Per-step scoring: `ev_loss = best_ev - chosen_ev`.
- Session stats include total/average EV lost, hit rate, and a 0–100 score.

## License

Proprietary (see `pyproject.toml`).
