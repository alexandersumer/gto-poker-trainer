from __future__ import annotations

import random

from gtotrainer.dynamic.cards import str_to_int
from gtotrainer.dynamic.rival_strategy import build_profile, decide_action


class ConstantRandom(random.Random):
    def __init__(self, value: float) -> None:
        super().__init__()
        self._value = value

    def random(self) -> float:
        return self._value


def _combo(card1: str, card2: str) -> tuple[int, int]:
    a = str_to_int(card1)
    b = str_to_int(card2)
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def test_build_profile_uses_strengths() -> None:
    strong = _combo("As", "Ks")
    medium = _combo("Qs", "Jd")
    weak = _combo("7h", "4c")
    combos = [strong, medium, weak]

    profile = build_profile(
        combos,
        fold_probability=0.4,
        continue_ratio=0.5,
        strengths=[
            (strong, 0.9),
            (medium, 0.6),
            (weak, 0.1),
        ],
    )

    ranked = profile["ranked"]
    top_combo = tuple(sorted((int(ranked[0][0]), int(ranked[0][1]))))
    assert top_combo == strong
    assert profile["continue_count"] >= 1


def test_decide_action_biases_toward_stronger_holdings() -> None:
    strong = _combo("As", "Ks")
    weak = _combo("7h", "4c")
    combos = [strong, weak]

    profile = build_profile(
        combos,
        fold_probability=0.5,
        continue_ratio=0.5,
        strengths=[
            (strong, 0.92),
            (weak, 0.15),
        ],
    )
    meta = {"rival_profile": profile}

    rng = ConstantRandom(0.5)
    strong_decision = decide_action(meta, strong, rng)
    weak_decision = decide_action(meta, weak, rng)

    assert strong_decision.folds is False
    assert weak_decision.folds is True


def test_decide_action_adapts_to_hero_aggression() -> None:
    strong = _combo("As", "Ks")
    weak = _combo("7h", "4c")
    combos = [strong, weak]

    profile = build_profile(
        combos,
        fold_probability=0.65,
        continue_ratio=0.8,
        strengths=[
            (strong, 0.92),
            (weak, 0.2),
        ],
    )

    passive_meta = {"rival_profile": profile, "villain_adapt": {"aggr": 0, "passive": 6}}
    aggressive_meta = {"rival_profile": profile, "villain_adapt": {"aggr": 6, "passive": 0}}

    rng_value = 0.7
    passive_decision = decide_action(passive_meta, weak, ConstantRandom(rng_value))
    aggressive_decision = decide_action(aggressive_meta, weak, ConstantRandom(rng_value))

    assert passive_decision.folds is True
    assert aggressive_decision.folds is False
