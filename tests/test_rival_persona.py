from __future__ import annotations

import random

from gtotrainer.dynamic.rival_strategy import build_profile, decide_action
from gtotrainer.dynamic.cards import str_to_int


def _combo(card1: str, card2: str) -> tuple[int, int]:
    return tuple(sorted((str_to_int(card1), str_to_int(card2))))


def _profile() -> dict:
    strong = _combo("As", "Ks")
    weak = _combo("7h", "4c")
    combos = [strong, weak]
    return build_profile(
        combos,
        fold_probability=0.5,
        continue_ratio=0.6,
        strengths=[(strong, 0.9), (weak, 0.2)],
    )


def _fold_rate(persona: str) -> float:
    profile = _profile()
    rng = random.Random(123)
    folds = 0
    trials = 600
    meta = {
        "rival_profile": profile,
        "rival_style": persona,
        "bet": 4.0,
        "pot_before": 6.0,
    }
    weak = _combo("7h", "4c")
    for _ in range(trials):
        if decide_action(meta, weak, rng).folds:
            folds += 1
    return folds / trials


def test_aggressive_persona_folds_less_than_passive() -> None:
    aggressive = _fold_rate("aggressive")
    passive = _fold_rate("passive")
    balanced = _fold_rate("balanced")

    assert aggressive < balanced < passive
    assert passive - aggressive > 0.08
