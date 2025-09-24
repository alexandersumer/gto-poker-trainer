#!/usr/bin/env python3

"""Run the lightweight trainer benchmark from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from gtotrainer.analysis.benchmark import BenchmarkConfig, run_benchmark


def _parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in raw.split(",") if item.strip())
    except ValueError as exc:  # pragma: no cover - validation path
        raise argparse.ArgumentTypeError(f"invalid seed list '{raw}'") from exc


def _parse_flags(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(flag.strip() for flag in raw.split(",") if flag.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic trainer benchmark")
    parser.add_argument("--hands", type=int, default=50, help="Number of hands per run")
    parser.add_argument("--mc-trials", type=int, default=96, help="Monte Carlo trials per node")
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=(101,),
        help="Comma-separated list of integer seeds (default: 101)",
    )
    parser.add_argument(
        "--rival-style",
        type=str,
        default="balanced",
        help="Rival style key (balanced/aggressive/passive)",
    )
    parser.add_argument(
        "--hero-policy",
        type=str,
        default="gto",
        choices=("gto", "best"),
        help="Hero policy used during the benchmark",
    )
    parser.add_argument(
        "--enable",
        type=_parse_flags,
        default=(),
        help="Comma-separated feature flags to enable during the run",
    )
    parser.add_argument(
        "--disable",
        type=_parse_flags,
        default=(),
        help="Comma-separated feature flags to disable during the run",
    )

    args = parser.parse_args(argv)

    config = BenchmarkConfig(
        hands=args.hands,
        seeds=args.seeds,
        mc_trials=args.mc_trials,
        rival_style=args.rival_style,
        hero_policy=args.hero_policy,
        enable_features=args.enable,
        disable_features=args.disable,
    )

    result = run_benchmark(config)
    payload = {
        "combined": {
            "decisions": result.combined.decisions,
            "hands": result.combined.hands,
            "avg_ev_lost": result.combined.avg_ev_lost,
            "avg_loss_pct": result.combined.avg_loss_pct,
            "score_pct": result.combined.score_pct,
            "accuracy_pct": result.accuracy_pct,
            "exploitability_bb": result.exploitability_bb,
        },
        "runs": [
            {
                "seed": run.seed,
                "decisions": run.stats.decisions,
                "avg_ev_lost": run.stats.avg_ev_lost,
                "score_pct": run.stats.score_pct,
                "accuracy_pct": run.accuracy_pct,
            }
            for run in result.runs
        ],
    }

    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
