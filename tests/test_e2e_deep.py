from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def run_cli(args: list[str], input_text: str | None = None, cwd: Path | None = None) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    cmd = [sys.executable, "-m", "gto_trainer", *args]
    proc = subprocess.run(
        cmd,
        input=(input_text or "").encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd or PROJECT_ROOT,
        env=env,
        check=True,
    )
    return proc.stdout.decode()


def run_cli_play(args: list[str], input_text: str | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    cmd = [sys.executable, "-m", "gto_trainer", "play", *args]
    return subprocess.run(
        cmd,
        input=(input_text or "").encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )


def test_default_runs_play_and_shows_summary():
    # No subcommand: default executes multi-street play. Answer 4 prompts.
    out = run_cli(["--hands", "1", "--seed", "123", "--mc", "40", "--no-color"], input_text="2\n2\n2\n2\n")
    assert "GTO Trainer – Live" in out
    assert "PREFLOP" in out and "FLOP" in out and "TURN" in out and "RIVER" in out
    assert "Session Summary" in out


def test_play_with_solver_csv_runs_and_summarizes(tmp_path: Path):
    # Provide a minimal solver CSV with one entry; the engine will fall back to the
    # dynamic provider when keys don't match, but the presence of the CSV should not break.
    csv_path = tmp_path / "solver.csv"
    rows = [
        "street,hero_position,context_action,context_size,hero_hand,option_key,option_ev,option_why,gto_freq\n",
        "preflop,BB,open,2.0bb,AA,Fold,-1.0,Give up BB,0.00\n",
        "preflop,BB,open,2.0bb,AA,Call,0.10,Speculative,0.40\n",
        "preflop,BB,open,2.0bb,AA,3-bet to 9bb,0.20,Value,0.60\n",
    ]
    csv_path.write_text("".join(rows), encoding="utf-8")

    cp = run_cli_play(
        ["--hands", "1", "--seed", "321", "--mc", "10", "--solver-csv", str(csv_path)],
        input_text="2\nq\n",
    )
    out = cp.stdout.decode()
    assert cp.returncode == 0, out
    assert "GTO Trainer – Live" in out
    assert ("Session Summary" in out) or ("No hands answered." in out)
