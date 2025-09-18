from __future__ import annotations

import argparse
import sys

from .engine_play import run_play


def _add_play_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hands", type=int, default=1, help="Number of random hands to play")
    # If omitted, runs with a random seed for variety. Pass an int to reproduce.
    p.add_argument("--seed", type=int, default=None, help="RNG seed (random if omitted)")
    p.add_argument(
        "--mc",
        type=int,
        default=200,
        metavar="SAMPLES",
        help="Monte Carlo simulation samples per decision (higher = steadier guidance, slower)",
    )
    p.add_argument("--no-color", action="store_true", help="Disable colored output (default is colored)")
    p.add_argument("--solver-csv", type=str, default=None, help="Use preflop solver CSV before heuristics")


def main() -> None:
    """Single-mode CLI: multi-street play is the default and only mode.

    For backwards compatibility, an optional "play" subcommand is accepted,
    but omitted args run the same behavior.
    """
    argv = sys.argv[1:]

    # If the first non-flag token is "play", drop it and parse the rest.
    first_non_flag = next((t for t in argv if not t.startswith("-")), None)
    if first_non_flag == "play":
        argv = [t for t in argv if t != "play"]

    parser = argparse.ArgumentParser(prog="gto-trainer", description="Heads-up multi-street EV trainer (CLI)")
    _add_play_args(parser)
    args = parser.parse_args(argv)

    run_play(
        seed=args.seed,
        hands=args.hands,
        mc_trials=args.mc,
        no_color=args.no_color,
        solver_csv=args.solver_csv,
    )


if __name__ == "__main__":
    main()
