from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def run_cli(args: list[str], input_text: str | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    cmd = [sys.executable, "-m", "gto_poker_trainer", "play", *args]
    return subprocess.run(
        cmd,
        input=(input_text or "").encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )


def test_help_and_pot_hotkeys_are_handled_gracefully():
    # Send 'h' and '?' then choose a valid option and quit.
    cp = run_cli(["--hands", "1", "--seed", "999", "--mc", "10", "--no-color"], input_text="h\n?\n1\nq\n")
    out = cp.stdout.decode()
    assert cp.returncode == 0, out
    assert "Controls" in out
    assert "Pot:" in out
