from __future__ import annotations

import argparse

from .textual_app import run_textual


def main() -> None:
    p = argparse.ArgumentParser(prog="gto-poker-trainer-textual", description="Textual UI for GTO Poker Trainer")
    p.add_argument("--hands", type=int, default=1, help="Hands per session")
    p.add_argument("--mc", type=int, default=200, help="Monte Carlo trials per decision")
    p.add_argument("--solver-csv", type=str, default=None, help="Optional preflop solver CSV")
    args = p.parse_args()
    run_textual(hands=args.hands, mc_trials=args.mc, solver_csv=args.solver_csv)


if __name__ == "__main__":
    main()

