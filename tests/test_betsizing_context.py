from __future__ import annotations

import random

from gtotrainer.dynamic.cards import str_to_int
from gtotrainer.dynamic.episode import Node
from gtotrainer.dynamic.policy import options_for, reset_bet_sizing_state


def _hand_state(hero, rival, board, pot=6.0, street="flop"):
    return {
        "pot": pot,
        "hero_cards": tuple(hero),
        "rival_cards": tuple(rival),
        "full_board": tuple(board),
        "street": street,
        "nodes": {},
        "hero_contrib": pot / 2,
        "rival_contrib": pot / 2,
        "hero_stack": 97.0,
        "rival_stack": 97.0,
        "effective_stack": 97.0,
        "rival_style": "balanced",
    }


def _build_node(street: str, board_cards: list[int], hand_state: dict) -> Node:
    return Node(
        street=street,
        description=f"Test node {street}",
        pot_bb=float(hand_state["pot"]),
        effective_bb=100.0,
        hero_cards=list(hand_state["hero_cards"]),
        board=board_cards,
        actor="BB",
        context={"facing": "check", "open_size": 2.5, "hand_state": hand_state},
    )


def test_flop_sizing_includes_overbet_and_small():
    hero = [str_to_int("Ah"), str_to_int("Kd")]
    board = [str_to_int("Ac"), str_to_int("7d"), str_to_int("2s")]
    rival = [str_to_int("9c"), str_to_int("8c")]
    state = _hand_state(hero, rival, board)
    node = _build_node("flop", board, state)

    reset_bet_sizing_state()
    options = options_for(node, random.Random(5), mc_trials=96)
    bet_fracs = [opt.meta.get("sizing_fraction") for opt in options if opt.meta and opt.meta.get("action") == "bet"]
    assert any(abs(float(frac) - 0.25) < 1e-6 for frac in bet_fracs)
    assert any(float(frac) >= 1.0 for frac in bet_fracs)
    assert any(abs(float(frac) - 0.66) < 1e-6 or abs(float(frac) - 0.75) < 1e-6 for frac in bet_fracs)
    assert all(opt.meta.get("rival_style") == "balanced" for opt in options if opt.meta)


def test_flop_wet_board_prefers_medium_sizes():
    hero = [str_to_int("Qs"), str_to_int("Jc")]
    board = [str_to_int("Js"), str_to_int("Ts"), str_to_int("9s")]
    rival = [str_to_int("7h"), str_to_int("6h")]
    state = _hand_state(hero, rival, board)
    node = _build_node("flop", board, state)

    reset_bet_sizing_state()
    options = options_for(node, random.Random(7), mc_trials=96)
    bet_fracs = [
        float(opt.meta.get("sizing_fraction")) for opt in options if opt.meta and opt.meta.get("action") == "bet"
    ]
    assert any(abs(frac - 0.5) < 1e-6 for frac in bet_fracs)
    assert any(abs(frac - 0.66) < 1e-6 or abs(frac - 1.0) < 1e-6 for frac in bet_fracs)


def test_turn_probe_sizes_depend_on_texture():
    hero = [str_to_int("Ah"), str_to_int("Qs")]
    board = [str_to_int("Ac"), str_to_int("7d"), str_to_int("2s"), str_to_int("9h")]
    rival = [str_to_int("Kc"), str_to_int("Jc")]
    state = _hand_state(hero, rival, board, pot=8.0, street="turn")
    node = _build_node("turn", board, state)

    reset_bet_sizing_state()
    options = options_for(node, random.Random(9), mc_trials=96)
    bet_fracs = [
        float(opt.meta.get("sizing_fraction")) for opt in options if opt.meta and opt.meta.get("action") == "bet"
    ]
    assert any(frac <= 0.5 for frac in bet_fracs)
    assert any(frac >= 0.75 for frac in bet_fracs)


def test_river_fold_keep_style_metadata():
    hero = [str_to_int("Ah"), str_to_int("Qs")]
    board = [str_to_int("Ac"), str_to_int("7d"), str_to_int("2s"), str_to_int("9h"), str_to_int("4c")]
    rival = [str_to_int("Kc"), str_to_int("Jc")]
    state = _hand_state(hero, rival, board, pot=10.0, street="river")
    node = _build_node("river", board, state)

    reset_bet_sizing_state()
    options = options_for(node, random.Random(11), mc_trials=96)
    for opt in options:
        if opt.meta and opt.meta.get("action") in {"fold", "call", "bet", "raise"}:
            assert opt.meta.get("rival_style") == "balanced"
