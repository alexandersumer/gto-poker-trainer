from __future__ import annotations

import random
from collections.abc import Iterable

import pytest

from gtotrainer.dynamic.cards import str_to_int
from gtotrainer.dynamic.episode import Node
from gtotrainer.dynamic.policy import options_for, resolve_for
from gtotrainer.dynamic.rival_strategy import RivalDecision


def _combo(card1: str, card2: str) -> tuple[int, int]:
    first = str_to_int(card1)
    second = str_to_int(card2)
    return tuple(sorted((first, second)))  # type: ignore[return-value]


def _build_hand_state(
    combos: Iterable[tuple[int, int]],
    weights: Iterable[float],
) -> dict:
    combo_list = list(combos)
    weight_list = list(weights)
    if len(combo_list) != len(weight_list):
        raise ValueError("combos and weights must align")
    total = sum(weight_list)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    norm_weights = [value / total for value in weight_list]

    return {
        "pot": 5.5,
        "hero_contrib": 1.0,
        "rival_contrib": 3.5,
        "hero_stack": 99.0,
        "rival_stack": 96.5,
        "effective_stack": 96.5,
        "rival_continue_range": combo_list,
        "rival_continue_weights": [
            [combo[0], combo[1], weight] for combo, weight in zip(combo_list, norm_weights, strict=False)
        ],
        "nodes": {},
        "rival_range": "bb_defend",
    }


def _make_test_node(hand_state: dict) -> Node:
    hero_cards = [str_to_int("Ah"), str_to_int("Kd")]
    board = [str_to_int("9c"), str_to_int("6s"), str_to_int("2d")]
    context = {
        "open_size": 2.5,
        "rival_range": "bb_defend",
        "hand_state": hand_state,
    }
    return Node(
        street="flop",
        description="unit-test flop node",
        pot_bb=hand_state["pot"],
        effective_bb=hand_state["effective_stack"],
        hero_cards=hero_cards,
        board=board,
        actor="BB",
        context=context,
    )


def test_options_carry_rival_continue_weights() -> None:
    combos = [
        _combo("Qc", "Jc"),
        _combo("9h", "8h"),
        _combo("7d", "6d"),
        _combo("5c", "4c"),
    ]
    weights = [0.45, 0.3, 0.2, 0.05]
    hand_state = _build_hand_state(combos, weights)
    node = _make_test_node(hand_state)

    rng = random.Random(0)
    options = options_for(node, rng, mc_trials=96)
    bet_option = next(opt for opt in options if opt.meta.get("action") == "bet")

    weight_entries = bet_option.meta.get("rival_continue_weights")
    assert isinstance(weight_entries, list) and weight_entries

    meta_weights = {tuple(sorted((int(entry[0]), int(entry[1])))): float(entry[2]) for entry in weight_entries}
    sum_weights = sum(meta_weights.values())
    assert 0.99 <= sum_weights <= 1.01

    assert meta_weights[combos[0]] > meta_weights[combos[1]] > meta_weights[combos[2]] > meta_weights[combos[3]]

    cfr_data = bet_option.meta.get("cfr_payoffs")
    assert isinstance(cfr_data, dict)
    rival_actions = [str(label) for label in cfr_data.get("rival_actions", [])]
    raise_ratio = float(bet_option.meta.get("rival_raise_ratio", 0.0))
    if "jam" in rival_actions:
        assert raise_ratio > 0
    else:
        assert raise_ratio == 0


def test_resolve_for_persists_continue_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    combos = [
        _combo("Qc", "Jc"),
        _combo("9h", "8h"),
        _combo("7d", "6d"),
    ]
    weights = [0.6, 0.3, 0.1]
    hand_state = _build_hand_state(combos, weights)
    node = _make_test_node(hand_state)

    rng = random.Random(0)
    options = options_for(node, rng, mc_trials=96)
    bet_option = next(opt for opt in options if opt.meta.get("action") == "bet")

    def _always_continue(_meta: dict | None, _cards, _rng) -> RivalDecision:
        return RivalDecision(folds=False)

    monkeypatch.setattr("gtotrainer.dynamic.rival_strategy.decide_action", _always_continue)

    resolve_for(node, bet_option, random.Random(0))

    stored = hand_state.get("rival_continue_weights")
    assert isinstance(stored, list) and stored
    total = sum(float(entry[2]) for entry in stored)
    assert 0.99 <= total <= 1.01

    normalized_meta = {tuple(sorted((int(entry[0]), int(entry[1])))): float(entry[2]) for entry in stored}
    assert set(normalized_meta.keys()) == set(combos)
