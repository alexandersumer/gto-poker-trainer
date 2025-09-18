from __future__ import annotations

import random

from gto_trainer.dynamic.cards import str_to_int
from gto_trainer.dynamic.generator import Node
from gto_trainer.dynamic.policy import preflop_options, resolve_for, river_options, turn_options


def _make_hand_state(
    hero: list[int],
    villain: tuple[int, int],
    board: list[int],
    *,
    pot: float,
    street: str,
    hero_contrib: float | None = None,
    villain_contrib: float | None = None,
    hero_stack: float | None = None,
    villain_stack: float | None = None,
) -> dict:
    if hero_contrib is None and villain_contrib is None:
        hero_contrib = pot / 2
        villain_contrib = pot - hero_contrib
    elif hero_contrib is None:
        hero_contrib = pot - (villain_contrib or 0.0)
    elif villain_contrib is None:
        villain_contrib = pot - hero_contrib

    hero_stack_val = hero_stack if hero_stack is not None else max(0.0, 100.0 - hero_contrib)
    villain_stack_val = villain_stack if villain_stack is not None else max(0.0, 100.0 - villain_contrib)
    effective_stack = min(hero_stack_val, villain_stack_val)

    return {
        "pot": pot,
        "hero_cards": tuple(hero),
        "villain_cards": villain,
        "full_board": tuple(board + [0] * (5 - len(board))),
        "street": street,
        "board_index": len(board),
        "history": [],
        "nodes": {},
        "hero_contrib": hero_contrib,
        "villain_contrib": villain_contrib,
        "hero_stack": hero_stack_val,
        "villain_stack": villain_stack_val,
        "effective_stack": effective_stack,
    }


def _str_cards(cards: list[str]) -> list[int]:
    return [str_to_int(c) for c in cards]


def _villain_tuple(c1: str, c2: str) -> tuple[int, int]:
    return (str_to_int(c1), str_to_int(c2))


def test_preflop_call_updates_state_and_pot():
    hero = _str_cards(["As", "Kd"])
    villain = _villain_tuple("Qh", "Js")
    board = _str_cards(["2c", "7d", "Jh", "9s", "3d"])
    state = _make_hand_state(
        hero,
        villain,
        board[:0],
        pot=3.5,
        street="preflop",
        hero_contrib=1.0,
        villain_contrib=2.5,
    )
    node = Node(
        street="preflop",
        description="test",
        pot_bb=3.5,
        effective_bb=100.0,
        hero_cards=hero,
        board=[],
        actor="BB",
        context={"open_size": 2.5, "hand_state": state},
    )

    opts = preflop_options(node, random.Random(1), mc_trials=120)
    call_opt = next(o for o in opts if o.key == "Call")
    res = resolve_for(node, call_opt, random.Random(1))

    assert not res.hand_ended
    assert state["street"] == "flop"
    assert state["board_index"] == 3
    # Pot should now include the BB call (2.5 - 1.0 = 1.5) → 3.5 + 1.5 = 5.0
    assert abs(state["pot"] - 5.0) < 1e-9
    assert "call" in (res.note or "").lower()


def test_preflop_three_bet_folds_weak_villain():
    hero = _str_cards(["As", "Ad"])
    villain = _villain_tuple("7c", "2d")
    board = _str_cards(["2h", "9d", "Qh", "5s", "Jc"])
    initial_pot = 3.5
    state = _make_hand_state(
        hero,
        villain,
        board[:0],
        pot=initial_pot,
        street="preflop",
        hero_contrib=1.0,
        villain_contrib=initial_pot - 1.0,
    )
    node = Node(
        street="preflop",
        description="test",
        pot_bb=3.5,
        effective_bb=100.0,
        hero_cards=hero,
        board=[],
        actor="BB",
        context={"open_size": 2.5, "hand_state": state},
    )

    opts = preflop_options(node, random.Random(2), mc_trials=160)
    three_bet = next(o for o in opts if o.key.startswith("3-bet"))
    res = resolve_for(node, three_bet, random.Random(2))

    assert res.hand_ended
    assert state.get("hand_over", False)
    assert "fold" in (res.note or "").lower()
    assert "net" in (res.note or "")


def test_preflop_three_bet_strong_villain_continues():
    hero = _str_cards(["7c", "2d"])
    villain = _villain_tuple("As", "Ad")
    board = _str_cards(["2h", "9d", "Qh", "5s", "Jc"])
    initial_pot = 3.5
    state = _make_hand_state(
        hero,
        villain,
        board[:0],
        pot=initial_pot,
        street="preflop",
        hero_contrib=1.0,
        villain_contrib=initial_pot - 1.0,
    )
    node = Node(
        street="preflop",
        description="test",
        pot_bb=initial_pot,
        effective_bb=100.0,
        hero_cards=hero,
        board=[],
        actor="BB",
        context={"open_size": 2.5, "hand_state": state},
    )

    opts = preflop_options(node, random.Random(3), mc_trials=160)
    three_bet = next(o for o in opts if o.key.startswith("3-bet"))
    res = resolve_for(node, three_bet, random.Random(3))

    assert not res.hand_ended
    raise_to = float(three_bet.meta["raise_to"])
    expected = initial_pot + (raise_to - 1.0) + (raise_to - 2.5)
    assert abs(state["pot"] - expected) < 1e-9
    assert "calls" in (res.note or "").lower()


def test_preflop_jam_resolves_and_folds_weak_villain():
    hero = _str_cards(["As", "Ah"])
    villain = _villain_tuple("7c", "2d")
    board = _str_cards(["2h", "9d", "Qh", "5s", "Jc"])
    state = _make_hand_state(
        hero,
        villain,
        board[:0],
        pot=3.5,
        street="preflop",
        hero_contrib=1.0,
        villain_contrib=2.5,
    )
    node = Node(
        street="preflop",
        description="test",
        pot_bb=3.5,
        effective_bb=100.0,
        hero_cards=hero,
        board=[],
        actor="BB",
        context={"open_size": 2.5, "hand_state": state},
    )

    opts = preflop_options(node, random.Random(7), mc_trials=200)
    jam_opt = next(o for o in opts if "All-in" in o.key)
    res = resolve_for(node, jam_opt, random.Random(7))

    assert res.hand_ended
    assert "fold" in (res.note or "").lower()
    assert state.get("hand_over", False)
    assert "net" in (res.note or "")


def test_turn_raise_can_force_fold():
    hero = _str_cards(["As", "Kd"])
    villain = _villain_tuple("7c", "2d")
    board = _str_cards(["Ad", "Kh", "Qc", "2s"])
    state = _make_hand_state(
        hero,
        villain,
        board[:4],
        pot=10.0,
        street="turn",
        hero_contrib=5.0,
        villain_contrib=5.0,
    )
    node = Node(
        street="turn",
        description="turn",
        pot_bb=10.0,
        effective_bb=100.0,
        hero_cards=hero,
        board=board[:4],
        actor="BB",
        context={"facing": "bet", "bet": 5.0, "open_size": 2.5, "hand_state": state},
    )

    opts = turn_options(node, random.Random(4), mc_trials=120)
    raise_opt = next(o for o in opts if o.key.startswith("Raise"))
    res = resolve_for(node, raise_opt, random.Random(4))

    assert res.hand_ended
    assert state.get("hand_over", False)
    assert "fold" in (res.note or "").lower()
    assert "net" in (res.note or "")


def test_turn_call_moves_to_river_when_villain_strong():
    hero = _str_cards(["7c", "2d"])
    villain = _villain_tuple("As", "Ad")
    board = _str_cards(["Ad", "Kh", "Qc", "2s"])
    state = _make_hand_state(hero, villain, board[:4], pot=10.0, street="turn")
    node = Node(
        street="turn",
        description="turn",
        pot_bb=10.0,
        effective_bb=100.0,
        hero_cards=hero,
        board=board[:4],
        actor="BB",
        context={"facing": "bet", "bet": 5.0, "open_size": 2.5, "hand_state": state},
    )

    opts = turn_options(node, random.Random(5), mc_trials=120)
    call_opt = next(o for o in opts if o.key == "Call")
    res = resolve_for(node, call_opt, random.Random(5))

    assert not res.hand_ended
    assert state["street"] == "river"
    assert state["board_index"] == 5
    assert abs(state["pot"] - 20.0) < 1e-9
    assert "call" in (res.note or "").lower()


def test_river_bet_call_reveals_villain():
    hero = _str_cards(["7c", "2d"])
    villain = _villain_tuple("As", "Ad")
    board = _str_cards(["Ad", "Kh", "Qc", "2s", "9d"])
    state = _make_hand_state(hero, villain, board, pot=20.0, street="river")
    node = Node(
        street="river",
        description="river",
        pot_bb=20.0,
        effective_bb=100.0,
        hero_cards=hero,
        board=board,
        actor="BB",
        context={"facing": "oop-check", "open_size": 2.5, "hand_state": state},
    )

    opts = river_options(node, random.Random(6), mc_trials=120)
    bet_opt = next(o for o in opts if o.key.startswith("Bet 100"))
    res = resolve_for(node, bet_opt, random.Random(6))

    assert res.hand_ended
    assert res.reveal_villain
    assert "calls" in (res.note or "").lower()
    assert state.get("hand_over", False)
