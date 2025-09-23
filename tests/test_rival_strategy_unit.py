from __future__ import annotations

import random

from gtotrainer.dynamic import rival_strategy as vs
from gtotrainer.dynamic.cards import str_to_int


def test_build_profile_returns_expected_structure() -> None:
    combos = [(0, 1), (10, 11), (20, 21), (30, 31)]
    profile = vs.build_profile(combos, fold_probability=0.3, continue_ratio=0.5)
    assert set(profile) >= {
        "fold_probability",
        "continue_ratio",
        "total",
        "continue_count",
        "ranked",
        "strengths",
        "ranks",
    }
    assert profile["total"] == len(combos)
    assert isinstance(profile["ranked"], list)


def test_decide_action_biases_by_strength() -> None:
    combos = [(0, 1), (8, 9), (24, 25), (40, 41)]
    profile = vs.build_profile(combos, fold_probability=0.5, continue_ratio=0.5)
    ranked = profile["ranked"]
    strong_combo = tuple(ranked[0])
    weak_combo = tuple(ranked[-1])
    seeds = range(200)
    strong_folds = sum(
        vs.decide_action({"rival_profile": profile}, strong_combo, random.Random(s)).folds for s in seeds
    )
    weak_folds = sum(vs.decide_action({"rival_profile": profile}, weak_combo, random.Random(s)).folds for s in seeds)
    assert strong_folds < weak_folds


def test_decide_action_defaults_to_continue_without_profile() -> None:
    decision = vs.decide_action({}, (0, 1), random.Random(0))
    assert not decision.folds


def test_board_texture_adjusts_fold_tendency() -> None:
    combos = [(0, 1), (12, 13), (24, 25), (36, 37), (40, 41)]
    profile = vs.build_profile(combos, fold_probability=0.45, continue_ratio=0.55)

    base_meta = {
        "rival_profile": profile,
        "pot_before": 6.0,
        "bet": 3.0,
    }

    dry_board = [str_to_int("Ah"), str_to_int("7d"), str_to_int("2c")]
    wet_board = [str_to_int("Th"), str_to_int("Jh"), str_to_int("Qh")]

    dry_meta = {**base_meta, "board_cards": dry_board}
    wet_meta = {**base_meta, "board_cards": wet_board}

    seeds = [random.Random(seed) for seed in range(120)]
    dry_folds = sum(vs.decide_action(dry_meta, None, rng).folds for rng in seeds)
    wet_folds = sum(vs.decide_action(wet_meta, None, random.Random(10_000 + i)).folds for i in range(120))

    assert wet_folds < dry_folds
