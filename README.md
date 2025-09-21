# GTO Trainer (Rust)

GTO Trainer is a heads-up no-limit hold'em practice environment implemented entirely in Rust. It includes a CLI, REST API, and browser UI for running multi-street scenarios with Monte Carlo EV estimates and configurable rival profiles.

## Requirements

- Rust toolchain 1.90+ (`rustup` recommended)
- Node.js 20+ (for Playwright browser checks)
- Python 3.12 (for API smoke tests)

## Run the CLI

```bash
cargo run -- --hands 3 --mc 400 --rival-style aggressive
```

Autoplay without prompts:

```bash
cargo run -- --hands 5 --auto --no-color
```

## Run the web server

```bash
cargo run -- serve --addr 127.0.0.1:8080
# open http://127.0.0.1:8080
```

## Test matrix

```bash
# Rust lint + tests
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test --all --all-features

# Python API smokes (requires release build)
cargo build --release
GTO_TRAINER_BIN=./target/release/gto-trainer python -m pip install -r qa/requirements.txt
GTO_TRAINER_BIN=./target/release/gto-trainer python -m pytest qa/tests

# Browser smoke tests (Playwright)
npm ci
npx playwright install --with-deps chromium
GTO_TRAINER_HOST=127.0.0.1 GTO_TRAINER_PORT=8082 npm test
```

## Development notes

- Configure git hooks to run format/lint/test before each commit:
  ```bash
  git config core.hooksPath .githooks
  ```
- GitHub Actions runs the same Rust, Python, and Playwright suites on every push and pull request.
- Build a Docker image if needed:
  ```bash
  docker build -t gto-trainer .
  docker run --rm -p 8080:8080 gto-trainer
  ```

## Layout

```
Cargo.toml
src/
  cards.rs        # Card primitives and deck utilities
  equity.rs       # Hand evaluation + Monte Carlo equity sampling
  game.rs         # Shared action/node types
  rival.rs        # Rival style presets and heuristics
  session.rs      # Session lifecycle and EV bookkeeping
  trainer.rs      # CLI runner
  web/            # Axum server + static hosting
public/           # Static assets for the browser UI
qa/               # Python smoke tests
ui-tests/         # Playwright tests
tests/            # Rust integration tests (CLI, API, session)
```

## License

Proprietary (see `LICENSE` if provided).
