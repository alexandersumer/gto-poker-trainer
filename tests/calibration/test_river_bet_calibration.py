from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from gtotrainer.dynamic.cards import str_to_int
from gtotrainer.dynamic.episode import Node
from gtotrainer.dynamic.policy import options_for

REFERENCE_PATH = Path(__file__).with_name("reference_river_bet.json")


def _combo(card1: str, card2: str) -> tuple[int, int]:
    return tuple(sorted((str_to_int(card1), str_to_int(card2))))


def _hand_state() -> dict:
    combos = [
        _combo("Qc", "Jc"),
        _combo("9h", "8h"),
        _combo("7d", "6d"),
        _combo("5c", "4c"),
    ]
    weights = [0.45, 0.3, 0.2, 0.05]
    total = sum(weights)
    norm = [w / total for w in weights]
    return {
        "pot": 5.5,
        "hero_contrib": 1.0,
        "rival_contrib": 3.5,
        "hero_stack": 99.0,
        "rival_stack": 96.5,
        "effective_stack": 96.5,
        "rival_continue_range": combos,
        "rival_continue_weights": [[c[0], c[1], w] for c, w in zip(combos, norm, strict=False)],
        "nodes": {},
        "rival_range": "bb_defend",
    }


@pytest.mark.parametrize("reference", json.loads(REFERENCE_PATH.read_text()))
def test_river_half_pot_calibration(reference: dict[str, object]) -> None:
    hand_state = _hand_state()
    node = Node(
        street="river",
        description="calibration",
        pot_bb=hand_state["pot"],
        effective_bb=hand_state["effective_stack"],
        hero_cards=[str_to_int("Ah"), str_to_int("Kd")],
        board=[
            str_to_int("9c"),
            str_to_int("6s"),
            str_to_int("2d"),
            str_to_int("Ts"),
            str_to_int("4h"),
        ],
        actor="BB",
        context={
            "open_size": 2.5,
            "rival_range": "bb_defend",
            "hand_state": hand_state,
        },
    )

    options = options_for(node, random.Random(0), mc_trials=96)
    target_bet = float(reference["bet"])
    option = next(opt for opt in options if math.isclose(float(opt.meta.get("bet", 0.0)), target_bet, rel_tol=1e-6))

    assert math.isclose(option.ev, float(reference["ev"]), rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(option.meta["rival_fe"], float(reference["rival_fe"]), rel_tol=1e-4)
    assert math.isclose(option.meta["rival_continue_ratio"], float(reference["rival_continue_ratio"]), rel_tol=1e-4)
    assert math.isclose(option.meta["rival_raise_ratio"], float(reference["rival_raise_ratio"]), rel_tol=1e-4)

    expected_mix: dict[str, float] = reference["cfr_mix"]  # type: ignore[assignment]
    actual_mix = option.meta.get("cfr_rival_mix")
    assert isinstance(actual_mix, dict)
    for key, value in expected_mix.items():
        assert math.isclose(actual_mix[key], float(value), rel_tol=1e-4, abs_tol=1e-4)
