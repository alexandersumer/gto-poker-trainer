# GTO Trainer

GTO Trainer is a heads-up no-limit hold’em practice environment with both a Textual CLI and a FastAPI web UI. Each scenario deals a full rival hand, lets the opponent react street by street, and reports the EV delta for every action you take.

## Project status

- **Live demo** – [gto-trainer.onrender.com](https://gto-trainer.onrender.com/)
- **Branding** – The codebase, CLI, and deployment all present as **GTO Trainer**.
- **Runtime target** – Python **3.12.11** across CLI, tests, and CI keeps solver output deterministic.

## Requirements

- Python 3.12.11 exactly (`pyenv` or another version manager is recommended).

## Install & run (CLI)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# Play a handful of random hands
gto-trainer --hands 5
```

Run in-place without installing:

```bash
PYTHONPATH=src python -m gto_trainer
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
  pip install -e .[dev]
  uvicorn gto_trainer.web.app:app --reload
  ```

Environment overrides: `HANDS` and `MC` mirror the CLI flags when exported before launch.

## Local development

Common helper targets:

```bash
make install-dev   # editable install with dev extras
make check         # lint + tests (matches CI)
make test          # pytest -q
make lint          # Ruff lint
make format        # Ruff formatter
make render-smoke  # build Docker image & hit /healthz (Render parity)
```

CI runs the same trio as `make check` (`ruff format --check`, `ruff check`, `pytest -q`).

## Notes

- Per decision we report `ev_loss = best_ev - chosen_ev`.
- Score aggregates EV loss (as pot %) across decisions; larger mistakes carry exponentially higher penalties.

## License

Proprietary (see `pyproject.toml`).
