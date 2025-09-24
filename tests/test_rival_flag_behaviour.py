from __future__ import annotations

import copy
import random

from gtotrainer.dynamic import rival_strategy


def _card(rank: int, suit: int) -> int:
    return rank * 4 + suit


def _profile() -> dict:
    combos = [
        (_card(12, 0), _card(12, 3)),  # AA suited
        (_card(11, 1), _card(9, 2)),
        (_card(9, 0), _card(9, 3)),
        (_card(4, 0), _card(4, 1)),
        (_card(2, 2), _card(7, 0)),
        (_card(1, 3), _card(5, 2)),
    ]
    return rival_strategy.build_profile(
        combos,
        fold_probability=0.4,
        continue_ratio=0.6,
    )


def _meta(profile: dict) -> dict:
    return {
        "rival_profile": copy.deepcopy(profile),
        "bet": 6.0,
        "pot_before": 9.0,
        "board_cards": [_card(10, 0), _card(9, 1), _card(5, 2)],
        "rival_adapt": {"aggr": 2, "passive": 1},
    }


def _fold_rate(combo: tuple[int, int]) -> float:
    profile = _profile()
    rng = random.Random(321)
    folds = 0
    trials = 800
    for _ in range(trials):
        decision = rival_strategy.decide_action(_meta(profile), combo, rng)
        if decision.folds:
            folds += 1
    return folds / trials


def test_rival_decisions_respect_strength_ordering() -> None:
    strong = (_card(12, 0), _card(12, 3))
    weak = (_card(1, 3), _card(5, 2))

    strong_rate = _fold_rate(strong)
    weak_rate = _fold_rate(weak)

    assert strong_rate < 0.25
    assert weak_rate > 0.85
    assert weak_rate - strong_rate > 0.6
