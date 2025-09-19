from __future__ import annotations

import random

import pytest

from gto_trainer.core.models import Option
from gto_trainer.dynamic.cards import str_to_int
from gto_trainer.dynamic.generator import Node, generate_episode
from gto_trainer.dynamic.policy import (
    flop_options,
    options_for,
    preflop_options,
    resolve_for,
    river_options,
    turn_options,
)


def test_generate_episode_structure_and_contexts_bb_defense():
    ep = generate_episode(random.Random(123), hero_seat="BB")
    assert len(ep.nodes) == 4
    streets = [n.street for n in ep.nodes]
    assert streets == ["preflop", "flop", "turn", "river"]
    assert ep.hero_seat == "BB"
    assert ep.villain_seat == "SB"
    hand_states = []
    for node in ep.nodes:
        assert "hand_state" in node.context
        hs = node.context["hand_state"]
        assert isinstance(hs, dict)
        hand_states.append(hs)
        assert node.actor == ep.hero_seat
        assert node.context.get("villain_range") == "sb_open"
    # Basic sanity of contexts
    assert "open_size" in ep.nodes[0].context
    assert ep.nodes[1].context.get("facing") == "check"
    assert ep.nodes[2].context.get("facing") in {"bet", "check"}
    assert ep.nodes[3].context.get("facing") in {"oop-check", "bet"}

    # Pot arithmetic consistency across streets
    n_pf, n_flop, n_turn, n_river = ep.nodes
    # Extract open size from description, formatted like "SB opens X.Xbb"
    import re

    m = re.search(r"opens\s+([0-9]+\.[0-9])bb", n_pf.description)
    assert m, f"could not parse open size from: {n_pf.description}"
    open_sz = float(m.group(1))
    blind_hero = 1.0
    call_cost = open_sz - blind_hero

    # Before any hero action, all streets share the same initial pot
    assert n_pf.pot_bb == pytest.approx(n_flop.pot_bb)
    assert n_flop.pot_bb == pytest.approx(n_turn.pot_bb)
    assert n_turn.pot_bb == pytest.approx(n_river.pot_bb)

    # Simulate the default line: call preflop, check flop, call turn
    sim_rng = random.Random(999)
    pf_opts = options_for(n_pf, sim_rng, 120)
    call_opt = next(o for o in pf_opts if o.meta and o.meta.get("action") == "call")
    resolve_for(n_pf, call_opt, sim_rng)
    assert n_flop.pot_bb == pytest.approx(n_pf.pot_bb + call_cost)

    flop_opts = options_for(n_flop, sim_rng, 120)
    check_opt = next(o for o in flop_opts if o.meta and o.meta.get("action") == "check")
    resolve_for(n_flop, check_opt, sim_rng)
    assert n_turn.pot_bb == pytest.approx(n_flop.pot_bb)

    turn_opts = options_for(n_turn, sim_rng, 120)
    facing_turn = str(n_turn.context.get("facing"))
    if facing_turn == "bet":
        call_turn = next(o for o in turn_opts if o.meta and o.meta.get("action") == "call")
        resolve_for(n_turn, call_turn, sim_rng)
        expected_turn_bet = float(n_turn.context["bet"])
        assert n_river.pot_bb == pytest.approx(n_turn.pot_bb + expected_turn_bet)
    else:
        check_turn = next(o for o in turn_opts if o.meta and o.meta.get("action") == "check")
        resolve_for(n_turn, check_turn, sim_rng)
        assert n_river.pot_bb == pytest.approx(n_turn.pot_bb)

    # Rival hole cards are unique and never duplicated on board or hero hand.
    villain_cards = hand_states[0]["villain_cards"]
    assert isinstance(villain_cards, tuple) and len(villain_cards) == 2
    hero_cards = ep.nodes[0].hero_cards
    board_cards = ep.nodes[-1].board
    all_cards = set(hero_cards) | set(villain_cards) | set(board_cards)
    assert len(all_cards) == len(hero_cards) + len(villain_cards) + len(board_cards)


def test_generate_episode_structure_and_contexts_sb_ip():
    ep = generate_episode(random.Random(321), hero_seat="SB")
    assert len(ep.nodes) == 4
    streets = [n.street for n in ep.nodes]
    assert streets == ["preflop", "flop", "turn", "river"]
    assert ep.hero_seat == "SB"
    assert ep.villain_seat == "BB"

    hand_states = []
    for node in ep.nodes:
        assert "hand_state" in node.context
        hs = node.context["hand_state"]
        assert isinstance(hs, dict)
        hand_states.append(hs)
        assert node.actor == ep.hero_seat
        assert node.context.get("villain_range") == "sb_open"

    n_pf, n_flop, n_turn, n_river = ep.nodes
    import re

    match = re.search(r"opens\s+([0-9]+\.[0-9])bb", n_pf.description)
    assert match, f"could not parse open size from: {n_pf.description}"
    open_sz = float(match.group(1))
    blind_hero = 0.5
    call_cost = open_sz - blind_hero

    assert n_pf.pot_bb == pytest.approx(n_flop.pot_bb)
    assert n_flop.pot_bb == pytest.approx(n_turn.pot_bb)
    assert n_turn.pot_bb == pytest.approx(n_river.pot_bb)

    sim_rng = random.Random(314)
    pf_opts = options_for(n_pf, sim_rng, 120)
    call_opt = next(o for o in pf_opts if o.meta and o.meta.get("action") == "call")
    resolve_for(n_pf, call_opt, sim_rng)
    assert n_flop.pot_bb == pytest.approx(n_pf.pot_bb + call_cost)

    flop_opts = options_for(n_flop, sim_rng, 120)
    check_opt = next(o for o in flop_opts if o.meta and o.meta.get("action") == "check")
    resolve_for(n_flop, check_opt, sim_rng)
    assert n_turn.pot_bb == pytest.approx(n_flop.pot_bb)

    turn_opts = options_for(n_turn, sim_rng, 120)
    facing_turn = str(n_turn.context.get("facing"))
    if facing_turn == "bet":
        call_turn = next(o for o in turn_opts if o.meta and o.meta.get("action") == "call")
        resolve_for(n_turn, call_turn, sim_rng)
        bet_turn = float(n_turn.context["bet"])
        assert n_river.pot_bb == pytest.approx(n_turn.pot_bb + bet_turn)
    else:
        check_turn = next(o for o in turn_opts if o.meta and o.meta.get("action") == "check")
        resolve_for(n_turn, check_turn, sim_rng)
        assert n_river.pot_bb == pytest.approx(n_turn.pot_bb)

    hand_state = hand_states[0]
    villain_cards = hand_state["villain_cards"]
    assert isinstance(villain_cards, tuple)
    hero_cards = ep.nodes[0].hero_cards
    board_cards = ep.nodes[-1].board
    all_cards = set(hero_cards) | set(villain_cards) | set(board_cards)
    assert len(all_cards) == len(hero_cards) + len(villain_cards) + len(board_cards)


def test_generate_episode_always_starts_preflop():
    for seat in ("BB", "SB"):
        rng = random.Random(777)
        episode = generate_episode(rng, hero_seat=seat)
        assert episode.nodes, "episode should contain nodes"
        first = episode.nodes[0]
        assert first.street == "preflop"
        assert first.actor == episode.hero_seat
        assert "opens" in first.description.lower()


def _assert_options_signature(opts: list[Option]):
    assert opts, "options list should not be empty"
    for o in opts:
        assert isinstance(o.key, str)
        assert isinstance(o.ev, float)
        assert isinstance(o.why, str)


def test_policy_preflop_flop_turn_river_option_shapes():
    rng = random.Random(7)
    hero = [str_to_int("As"), str_to_int("5s")]
    board = [str_to_int(x) for x in ["2c", "7d", "Jh", "Qc", "Kd"]]

    n_pf = Node(
        street="preflop",
        description="test pf",
        pot_bb=1.5 + (2.5 - 0.5),
        effective_bb=100,
        hero_cards=hero,
        board=[],
        actor="BB",
        context={"open_size": 2.5},
    )
    pf = preflop_options(n_pf, rng, mc_trials=20)
    keys = [o.key for o in pf]
    assert keys[:2] == ["Fold", "Call"]
    assert any(k.startswith("3-bet to ") for k in keys[2:])
    assert any("All-in" in k for k in keys)
    _assert_options_signature(pf)

    n_flop = Node(
        street="flop",
        description="test flop",
        pot_bb=5.0,
        effective_bb=100,
        hero_cards=hero,
        board=board[:3],
        actor="BB",
        context={"facing": "check"},
    )
    fl = flop_options(n_flop, rng, mc_trials=15)
    assert [o.key for o in fl][0] == "Check"
    assert any("All-in" in o.key for o in fl)
    _assert_options_signature(fl)

    n_turn = Node(
        street="turn",
        description="test turn",
        pot_bb=10.0,
        effective_bb=100,
        hero_cards=hero,
        board=board[:4],
        actor="BB",
        context={"facing": "bet", "bet": 5.0},
    )
    tr = turn_options(n_turn, rng, mc_trials=15)
    assert [o.key for o in tr][:2] == ["Fold", "Call"]
    _assert_options_signature(tr)

    n_river = Node(
        street="river",
        description="test river",
        pot_bb=20.0,
        effective_bb=100,
        hero_cards=hero,
        board=board,
        actor="BB",
        context={"facing": "oop-check"},
    )
    rv = river_options(n_river, rng, mc_trials=10)
    assert [o.key for o in rv][0] == "Check"
    assert any("All-in" in o.key for o in rv)
    _assert_options_signature(rv)


def test_options_for_dispatches_all_streets():
    rng = random.Random(9)
    hero = [str_to_int("Ac"), str_to_int("Kd")]
    board = [str_to_int(x) for x in ["2c", "7d", "Jh", "Qc", "Kd"]]
    for street, ctx in (
        ("preflop", {"open_size": 2.0}),
        ("flop", {"facing": "check"}),
        ("turn", {"facing": "bet", "bet": 3.0}),
        ("river", {"facing": "oop-check"}),
    ):
        node = Node(
            street=street,
            description=street,
            pot_bb=5.0,
            effective_bb=100.0,
            hero_cards=hero,
            board=board if street == "river" else board[: (3 if street == "flop" else 4 if street == "turn" else 0)],
            actor="BB",
            context=ctx,
        )
        opts = options_for(node, rng, mc_trials=10)
        assert isinstance(opts, list) and opts


def test_flop_resolution_folds_weak_villain_combo():
    rng = random.Random(21)
    hero = [str_to_int("As"), str_to_int("Ad")]
    villain = (str_to_int("7c"), str_to_int("2d"))
    full_board = [str_to_int(c) for c in ["Ks", "Qc", "Th", "3s", "8d"]]
    hand_state = {
        "pot": 6.0,
        "hero_cards": tuple(hero),
        "villain_cards": villain,
        "full_board": tuple(full_board),
        "street": "flop",
        "nodes": {},
        "hero_contrib": 3.0,
        "villain_contrib": 3.0,
        "hero_stack": 97.0,
        "villain_stack": 97.0,
        "effective_stack": 97.0,
    }
    flop_board = full_board[:3]
    node = Node(
        street="flop",
        description="Board K Q T. SB checks.",
        pot_bb=6.0,
        effective_bb=100.0,
        hero_cards=hero,
        board=flop_board,
        actor="BB",
        context={"facing": "check", "open_size": 2.5, "hand_state": hand_state},
    )
    opts = flop_options(node, rng, mc_trials=40)
    bet_opt = next(
        o
        for o in opts
        if o.meta and o.meta.get("action") == "bet" and abs(o.meta["bet"] - round(node.pot_bb * 0.75, 2)) < 1e-6
    )
    res = resolve_for(node, bet_opt, rng)
    assert res.hand_ended
    assert "fold" in res.note.lower()
    assert hand_state.get("hand_over", False)


def test_flop_resolution_continues_when_villain_strong():
    rng = random.Random(22)
    hero = [str_to_int("Jh"), str_to_int("Td")]
    villain = (str_to_int("Qh"), str_to_int("9h"))
    full_board = [str_to_int(c) for c in ["Kh", "8h", "7c", "2s", "4d"]]
    hand_state = {
        "pot": 5.0,
        "hero_cards": tuple(hero),
        "villain_cards": villain,
        "full_board": tuple(full_board),
        "street": "flop",
        "nodes": {},
        "hero_contrib": 2.5,
        "villain_contrib": 2.5,
        "hero_stack": 97.5,
        "villain_stack": 97.5,
        "effective_stack": 97.5,
    }
    flop_board = full_board[:3]
    node = Node(
        street="flop",
        description="Board K 8 7. SB checks.",
        pot_bb=5.0,
        effective_bb=100.0,
        hero_cards=hero,
        board=flop_board,
        actor="BB",
        context={"facing": "check", "open_size": 2.5, "hand_state": hand_state},
    )
    opts = flop_options(node, rng, mc_trials=40)
    bet_opt = next(
        o
        for o in opts
        if o.meta and o.meta.get("action") == "bet" and abs(o.meta["bet"] - round(node.pot_bb * 0.33, 2)) < 1e-6
    )
    prev_pot = hand_state["pot"]
    res = resolve_for(node, bet_opt, rng)
    assert not res.hand_ended
    assert "call" in res.note.lower()
    assert hand_state["pot"] > prev_pot
