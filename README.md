# GTO Poker Trainer CLI

Heads-up no-limit hold’em trainer with both a terminal flow and a minimal browser UI. Every scenario deals a full villain hand (never overlapping with yours or the board), lets them react street by street, and shows the EV swing behind each choice.

**Live demo:** https://gto-poker-trainer.onrender.com/

## Requirements

- Python 3.12.11 exactly (use `pyenv` or your system Python).

## Install & Run (CLI)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# Play a handful of random hands
gto-poker-trainer-cli --hands 5
```

Running in-place without installing:

```bash
PYTHONPATH=src python -m gto_poker_trainer_cli
```

### CLI options

```
gto-poker-trainer-cli [--hands N] [--seed N] [--mc N] [--no-color] [--solver-csv PATH]
```

- `--hands N` hands to play (default 1).
- `--seed N` RNG seed (omit for randomness).
- `--mc N` Monte Carlo trials per decision (default 200).
- `--solver-csv PATH` optional preflop strategy CSV before heuristics kick in.
- `--no-color` disable ANSI colors.

Controls: `1–9` choose an action, `h` help, `?` pot + SPR, `q` quit. Feedback includes a summary such as “Villain calls with…” with EV breakdowns.

## Web UI

- Cloud: the Render deployment mirrors the CLI defaults — try it at https://gto-poker-trainer.onrender.com/.
- Local: install `.[dev]`, then start FastAPI with reload support.

```bash
pip install -e .[dev]
uvicorn gto_poker_trainer_cli.web.app:app --reload
```

Environment overrides: `HANDS` and `MC` mirror the CLI flags when set before launch.

## Local Development

```bash
make install-dev   # editable install with dev extras
make check         # lint + tests (same combo as CI)
make test          # pytest -q
make lint          # Ruff lint
make format        # Ruff formatter
make render-smoke  # smoke test Render Docker image + /healthz
```

`pytest -q` runs the deeper regression suite during strategy work.

## Notes

- Per step: `ev_loss = best_ev - chosen_ev`.
- Score aggregates total/average EV lost plus hit rate.

## License

Proprietary (see `pyproject.toml`).
