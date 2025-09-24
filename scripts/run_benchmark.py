#!/usr/bin/env python3

"""Run the lightweight trainer benchmark from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import cycle
from pathlib import Path

if __package__ is None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from gtotrainer.analysis.benchmark import BenchmarkConfig, BenchmarkScenario, run_benchmark


def _parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in raw.split(",") if item.strip())
    except ValueError as exc:  # pragma: no cover - validation path
        raise argparse.ArgumentTypeError(f"invalid seed list '{raw}'") from exc


def _resolve_scenarios(
    pack: str,
    seeds: tuple[int, ...],
    hands: int,
    rival_style: str,
    hero_policy: str,
) -> list[BenchmarkScenario]:
    if not seeds:
        raise ValueError("at least one seed must be supplied")

    pack_key = pack.strip().lower()
    if pack_key == "seeded":
        return [
            BenchmarkScenario(
                name=f"seed_{idx}_{seed}",
                seed=seed,
                rival_style=rival_style,
                hero_policy=hero_policy,
                hands=hands,
            )
            for idx, seed in enumerate(seeds)
        ]

    # Standard trio approximates SRP, probe, and 3-bet situations by mixing styles/policies.
    presets = [
        ("srp_btn_vs_bb", "balanced", "gto"),
        ("turn_probe_passive", "passive", "gto"),
        ("three_bet_defence", "aggressive", "best"),
    ]
    seed_cycle = cycle(seeds)
    scenarios: list[BenchmarkScenario] = []
    for name, style, policy in presets:
        scenarios.append(
            BenchmarkScenario(
                name=name,
                seed=next(seed_cycle),
                rival_style=style,
                hero_policy=policy,
                hands=hands,
            )
        )
    return scenarios


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
        "--scenario-pack",
        type=str,
        default="standard",
        choices=("standard", "seeded"),
        help="Scenario pack to run (default: standard trio)",
    )
    args = parser.parse_args(argv)

    config = BenchmarkConfig(
        hands=args.hands,
        seeds=args.seeds,
        mc_trials=args.mc_trials,
        rival_style=args.rival_style,
        hero_policy=args.hero_policy,
        scenarios=tuple(
            _resolve_scenarios(args.scenario_pack, args.seeds, args.hands, args.rival_style, args.hero_policy)
        ),
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
                "scenario": run.scenario.name,
                "seed": run.scenario.seed,
                "rival_style": run.scenario.rival_style,
                "hero_policy": run.scenario.hero_policy,
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
