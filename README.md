# GTO Poker Trainer CLI

Command‑line trainer for heads‑up no‑limit hold’em. Each scenario now includes a fully dealt villain hand (never overlapping with yours or the board). Actions are resolved against that hand: villains can fold, call, raise, or take you to showdown, and the feedback panel explains both EV and their response.

## Requirements

- Python 3.12.11 exactly.

## Quick Start

Install and run:

```
pip install -e .
gto-poker-trainer-cli
```

Run without installing (from project root):

```
PYTHONPATH=src python -m gto_poker_trainer_cli
```

## Usage

Single mode only; optional `play` subcommand is accepted.

```
gto-poker-trainer-cli [--hands N] [--seed N] [--mc N] [--no-color] [--solver-csv PATH]
```

Options:

- `--hands N` Hands to play (default: 1).
- `--seed N` RNG seed (omit for randomness).
- `--mc N` Monte Carlo trials per node (default: 200).
- `--solver-csv PATH` Preflop strategy CSV (falls back to dynamic villain logic when entries are missing).
- `--no-color` Disable colored output (color is ON by default).

Controls:

- `1–9` choose an action
- `h` help
- `?` pot + SPR
- `q` quit session

During feedback you’ll also see a summary line such as “Villain folds…” or “Villain calls with …” that reflects the simulated counterplay.

## Development

Ensure Python 3.12.11, then:

```
make install-dev
make test
make lint
make fix
make format
make check        # lint + tests (same combo used in CI)
make render-smoke # build the Docker image and hit /healthz like Render does

Run the deep regression suite directly when iterating on strategy logic:

```
pytest -q
```
```

## Notes

- Per step: `ev_loss = best_ev - chosen_ev`.
- Score uses total/avg EV lost and hit rate.

## License

Proprietary (see `pyproject.toml`).
