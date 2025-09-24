from __future__ import annotations

import copy
import random

from gtotrainer.core import feature_flags
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
    meta = {
        "rival_profile": copy.deepcopy(profile),
        "bet": 6.0,
        "pot_before": 9.0,
        "board_cards": [_card(10, 0), _card(9, 1), _card(5, 2)],
        "rival_adapt": {"aggr": 2, "passive": 1},
    }
    return meta


def _fold_rate(combo: tuple[int, int], *, enable_flag: bool) -> float:
    profile = _profile()
    rng = random.Random(321)
    trials = 800
    enabled = {"rival.texture_v2"} if enable_flag else set()
    with feature_flags.override(enable=enabled):
        folds = 0
        for _ in range(trials):
            meta = _meta(profile)
            decision = rival_strategy.decide_action(meta, combo, rng)
            if decision.folds:
                folds += 1
    return folds / trials


def test_texture_flag_strengthens_strength_separation() -> None:
    strong = (_card(12, 0), _card(12, 3))
    weak = (_card(1, 3), _card(5, 2))

    base_gap = _fold_rate(weak, enable_flag=False) - _fold_rate(strong, enable_flag=False)
    flagged_gap = _fold_rate(weak, enable_flag=True) - _fold_rate(strong, enable_flag=True)

    assert flagged_gap > base_gap + 0.05
