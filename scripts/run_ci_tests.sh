#!/usr/bin/env bash
set -euo pipefail

PYTEST=${PYTEST:-python -m pytest}

run() {
  echo "[CI] $1"
  shift
  $PYTEST "$@"
}

run "Unit & smoke" tests/test_scoring.py tests/test_textual_cards_unit.py tests/test_play_e2e.py tests/test_web_smoke.py
run "Browser flow" tests/test_web_ui_browser.py
