from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from gtotrainer.dynamic.cards import str_to_int
from gtotrainer.dynamic.generator import Node
from gtotrainer.dynamic.policy import flop_options, preflop_options, reset_bet_sizing_state

REFERENCE_PATH = Path(__file__).with_name("policy_reference.json")
_EV_TOLERANCE = 0.15
_FREQ_TOLERANCE = 0.05


def _hero_cards() -> list[int]:
    return [str_to_int("As"), str_to_int("Kd")]


def _preflop_node() -> Node:
    hero = _hero_cards()
    state = {
        "pot": 3.5,
        "hero_cards": tuple(hero),
        "rival_cards": (str_to_int("Qh"), str_to_int("Js")),
        "full_board": (0, 0, 0, 0, 0),
        "street": "preflop",
        "board_index": 0,
        "history": [],
        "nodes": {},
        "hero_contrib": 1.0,
        "rival_contrib": 2.5,
        "hero_stack": 99.0,
        "rival_stack": 97.5,
        "effective_stack": 97.5,
        "rival_style": "balanced",
    }
    return Node(
        street="preflop",
        description="calibration preflop",
        pot_bb=3.5,
        effective_bb=97.5,
        hero_cards=hero,
        board=[],
        actor="BB",
        context={"open_size": 2.5, "hand_state": state},
    )


def _flop_node() -> Node:
    hero = _hero_cards()
    board = [str_to_int(card) for card in ("2c", "7d", "Jh")]
    state = {
        "pot": 5.0,
        "hero_cards": tuple(hero),
        "rival_cards": (str_to_int("Qh"), str_to_int("Js")),
        "full_board": tuple(board + [0, 0]),
        "street": "flop",
        "board_index": 3,
        "history": [],
        "nodes": {},
        "hero_contrib": 2.5,
        "rival_contrib": 2.5,
        "hero_stack": 97.5,
        "rival_stack": 97.5,
        "effective_stack": 97.5,
        "rival_style": "balanced",
    }
    return Node(
        street="flop",
        description="calibration flop",
        pot_bb=5.0,
        effective_bb=97.5,
        hero_cards=hero,
        board=board,
        actor="BB",
        context={"hand_state": state, "open_size": 2.5},
    )


def _compare_options(option, expected) -> None:
    warnings = option.meta.get("warnings", []) if isinstance(option.meta, dict) else []
    assert option.key == expected["key"]
    assert warnings == expected["warnings"]
    assert option.ev == pytest.approx(expected["ev"], abs=_EV_TOLERANCE)
    expected_freq = expected["gto_freq"]
    if expected_freq is None:
        assert option.gto_freq is None
    else:
        assert option.gto_freq == pytest.approx(expected_freq, abs=_FREQ_TOLERANCE)


def test_policy_reference_regression() -> None:
    reference = json.loads(REFERENCE_PATH.read_text())

    preflop_cfg = reference["preflop"]
    rng_pre = random.Random(int(preflop_cfg["rng_seed"]))
    reset_bet_sizing_state()
    pre_opts = preflop_options(_preflop_node(), rng_pre, mc_trials=int(preflop_cfg["mc_trials"]))
    assert len(pre_opts) == len(preflop_cfg["options"])
    for option, expected in zip(pre_opts, preflop_cfg["options"], strict=False):
        _compare_options(option, expected)

    flop_cfg = reference["flop"]
    rng_flop = random.Random(int(flop_cfg["rng_seed"]))
    reset_bet_sizing_state()
    flop_opts = flop_options(_flop_node(), rng_flop, mc_trials=int(flop_cfg["mc_trials"]))
    assert len(flop_opts) == len(flop_cfg["options"])
    for option, expected in zip(flop_opts, flop_cfg["options"], strict=False):
        _compare_options(option, expected)
