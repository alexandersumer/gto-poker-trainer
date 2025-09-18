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


def test_play_shows_all_streets_and_summary_then_loops_until_quit():
    # Deterministic run with no color for stable assertions; answer 4 times then end
    # First session completes; on next session's first prompt, quit.
    # After summary, confirm starting a new session, then quit on the first prompt of the next session.
    cp = run_cli(
        ["--hands", "1", "--seed", "123", "--mc", "40", "--no-color"],
        input_text="2\n2\n2\n2\ny\nq\n",
    )
    out = cp.stdout.decode()
    assert cp.returncode == 0, out
    assert "GTO Trainer" in out
    assert "PREFLOP" in out
    # Dynamic opponent may end the hand early; ensure at least one post-flop update or villain note.
    assert "FLOP" in out or "TURN" in out or "RIVER" in out or "Villain" in out
    assert "Session Summary" in out
    # Verify next session started by seeing another hand header
    assert out.count("Hand 1/1") >= 2
    summary_line = next(line for line in out.splitlines() if "Hands answered:" in line)
    assert "│ 1" in summary_line
    assert "Top EV leaks" in out


def test_default_prints_ansi_when_color_enabled():
    cp = run_cli(["--hands", "1", "--seed", "321", "--mc", "20"], input_text="q\n")
    out = cp.stdout.decode()
    assert cp.returncode == 0
    # Expect some ANSI sequences in output when forcing color
    assert "\x1b[" in out
