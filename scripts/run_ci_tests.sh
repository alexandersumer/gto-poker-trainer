#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  PYTEST_CMD=(uv run --no-config --locked --extra dev -- pytest)
else
  PYTEST_CMD=(python -m pytest)
fi

if [[ ${PYTEST:-} ]]; then
  # Allow callers to override the command completely via $PYTEST
  read -r -a PYTEST_CMD <<<"$PYTEST"
fi

run() {
  echo "[CI] $1"
  shift
  "${PYTEST_CMD[@]}" "$@"
}

run "Unit & smoke" tests/test_scoring.py tests/test_textual_cards_unit.py tests/test_web_smoke.py
run "Browser flow" tests/test_web_ui_browser.py
