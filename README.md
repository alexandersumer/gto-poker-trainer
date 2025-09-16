# GTO Poker Trainer CLI

Train heads‑up no‑limit hold’em decisions with instant EV feedback.

- Deals random hands across all streets.
- After each action, shows EV vs. best action plus a short note.
- Session summary totals EV, averages, hit rate, and biggest leaks.

## Requirements

- Python 3.12.11 exactly.

## Quick Start

Install and run:

```
pip install -e .
gto-poker-trainer-cli --hands 10
```

Run without installing (from project root):

```
PYTHONPATH=src python -m gto_poker_trainer_cli
```

## Usage

Single mode only; optional `play` subcommand is accepted.

```
gto-poker-trainer-cli [--hands N] [--seed N] [--mc N] [--no-color|--force-color] [--solver-csv PATH]
```

Options:

- `--hands N` Hands to play (default: 1).
- `--seed N` RNG seed (omit for randomness).
- `--mc N` Monte Carlo trials per node (default: 200).
- `--solver-csv PATH` Preflop strategy CSV (falls back to heuristics).
- `--no-color` Disable color.
- `--force-color` Force color even if not a TTY.

Controls:

- `1–9` choose an action
- `h` help
- `?` pot + SPR
- `q` quit session

## Development

Ensure Python 3.12.11, then:

```
make install-dev
make test
make lint
make fix
make format
```

## Notes

- Per step: `ev_loss = best_ev - chosen_ev`.
- Score uses total/avg EV lost and hit rate.

## License

Proprietary (see `pyproject.toml`).
