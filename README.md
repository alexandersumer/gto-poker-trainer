# GTO Trainer (Rust Edition)

GTO Trainer is a heads-up no-limit hold'em practice environment rebuilt end-to-end in Rust. It provides a terminal experience and a lightweight web UI that serve multi-street scenarios, Monte Carlo EV estimates, and rival style presets. The project replaces the previous Python implementation while maintaining deep testing coverage and CI parity.

## Features

- **Rust native engine** &mdash; session management, EV sampling, and rival heuristics implemented in safe Rust.
- **Interactive CLI** &mdash; play through preflop, flop, turn, and river decisions with contextual descriptions and EV feedback. Auto-play mode available for quick simulations.
- **Web API + UI** &mdash; Axum-based API and static frontend that mirrors the CLI flow for browser-based training.
- **Monte Carlo equity** &mdash; configurable sampling depth per decision with deterministic seeding support.
- **Profiles** &mdash; balanced, aggressive, and passive opponent presets that influence fold frequencies and aggression.
- **Robust tests** &mdash; unit, integration, CLI, and HTTP smoke tests covering core behaviour and toolchain expectations.

## Getting started

### Prerequisites

- Rust toolchain (1.90 or newer). Install via [`rustup`](https://rustup.rs/).

### CLI quick start

```bash
cargo run -- --hands 3 --mc 400 --rival-style aggressive
```

Auto-play without manual input (useful for smoke tests):

```bash
cargo run -- --hands 5 --auto --no-color
```

### Web UI

```bash
# Run the web server on http://localhost:8080
cargo run -- serve --addr 127.0.0.1:8080
```

Open `http://127.0.0.1:8080` in your browser to play through the session.

### Testing & linting

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test --all --all-features
```

## Development workflow

- **Pre-commit hooks** &mdash; Configure git hooks to run formatting, clippy, and tests before each commit:
  ```bash
  git config core.hooksPath .githooks
  ```
- **CI** &mdash; GitHub Actions runs `cargo fmt --check`, `cargo clippy`, and `cargo test` on every push and pull request.
- **Docker** &mdash; Build a production image with:
  ```bash
  docker build -t gto-trainer .
  docker run --rm -p 8080:8080 gto-trainer
  ```

## Project structure

```
Cargo.toml
src/
  cards.rs        # Card representations and deck utilities
  equity.rs       # Hand evaluation + Monte Carlo equity sampling
  game.rs         # Shared structures for actions and nodes
  rival.rs        # Rival style presets and heuristics
  session.rs      # Session lifecycle, EV aggregation, and scoring
  trainer.rs      # CLI facade and interactive runner
  web/            # Axum web server + API handlers
public/           # Static assets for the web UI
tests/            # Integration tests (CLI, web, session)
```

## License

Proprietary (see `LICENSE` if provided). Contact the maintainers for usage guidance.
