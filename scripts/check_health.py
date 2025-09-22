#!/usr/bin/env python3
"""Compute aggregate NashConv estimates for canonical training spots.

Usage:
    uv run -- python scripts/check_health.py
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass

from gtotrainer.dynamic.policy import options_for
from gtotrainer.dynamic.seating import SeatRotation
from gtotrainer.features.session.analysis import estimate_nashconv
from gtotrainer.features.session.engine import SessionEngine


@dataclass(frozen=True)
class SpotSpec:
    name: str
    stacks_bb: float
    hands: int
    mc_trials: int


CANONICAL_SPOTS: tuple[SpotSpec, ...] = (
    SpotSpec(name="BTN_vs_BB_preflop", stacks_bb=93.0, hands=1, mc_trials=150),
    SpotSpec(name="BTN_vs_BB_flop", stacks_bb=93.0, hands=2, mc_trials=120),
)

SEEDS = (101, 202, 303)


def evaluate_spot(spec: SpotSpec) -> tuple[float, list[float]]:
    scores: list[float] = []
    for seed in SEEDS:
        engine = SessionEngine(rng=random.Random(seed), rotation=SeatRotation())
        episode = engine.build_episode(0, stacks_bb=spec.stacks_bb)
        node = episode.nodes[min(spec.hands - 1, len(episode.nodes) - 1)]
        options = options_for(node, random.Random(seed * 17 + 3), spec.mc_trials)
        scores.append(estimate_nashconv(options))
    mean = statistics.fmean(scores)
    return mean, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate blueprint health metrics.")
    parser.add_argument("--verbose", action="store_true", help="Print per-seed scores")
    args = parser.parse_args()

    for spec in CANONICAL_SPOTS:
        mean, samples = evaluate_spot(spec)
        deviation = statistics.pstdev(samples)
        print(f"{spec.name}: NashConv {mean:.3f} (σ={deviation:.3f})")
        if args.verbose:
            for seed, score in zip(SEEDS, samples, strict=False):
                print(f"  seed {seed}: {score:.3f}")


if __name__ == "__main__":
    main()
