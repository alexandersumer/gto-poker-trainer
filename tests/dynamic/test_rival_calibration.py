from __future__ import annotations

import random

from gtotrainer.dynamic.rival_strategy import build_profile, decide_action


SAMPLED_RANGE = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
STRENGTHS = [
    ((0, 1), 0.92),
    ((2, 3), 0.85),
    ((4, 5), 0.7),
    ((6, 7), 0.55),
    ((8, 9), 0.4),
    ((10, 11), 0.25),
]


def _simulate_fold_frequency(style: str) -> float:
    profile = build_profile(
        sampled_range=SAMPLED_RANGE,
        fold_probability=0.38,
        continue_ratio=0.62,
        strengths=STRENGTHS,
    )
    meta = {
        "rival_profile": profile,
        "rival_style": style,
        "pot_before": 6.0,
        "bet": 3.9,
        "board_cards": [12, 16, 20],
    }
    rng = random.Random(2024)
    draws = 6000
    folds = sum(decide_action(meta, None, rng).folds for _ in range(draws))
    return folds / draws


def test_persona_hierarchy_affects_fold_rate() -> None:
    aggressive = _simulate_fold_frequency("aggressive")
    balanced = _simulate_fold_frequency("balanced")
    passive = _simulate_fold_frequency("passive")

    assert aggressive < balanced < passive
    assert passive - aggressive > 0.05


def test_fold_frequency_tracks_profile_target() -> None:
    balanced = _simulate_fold_frequency("balanced")
    expected = 0.38
    assert abs(balanced - expected) < 0.08
