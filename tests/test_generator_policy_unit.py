from __future__ import annotations

import random

from gto_trainer.core.models import Option
from gto_trainer.dynamic.cards import str_to_int
from gto_trainer.dynamic.generator import Node, generate_episode
from gto_trainer.dynamic.policy import (
    flop_options,
    options_for,
    preflop_options,
    river_options,
    turn_options,
)


def test_generate_episode_structure_and_contexts():
    ep = generate_episode(random.Random(123))
    assert len(ep.nodes) == 4
    streets = [n.street for n in ep.nodes]
    assert streets == ["preflop", "flop", "turn", "river"]
    # Basic sanity of contexts
    assert "open_size" in ep.nodes[0].context
    assert ep.nodes[1].context.get("facing") == "check"
    assert ep.nodes[2].context.get("facing") == "bet"
    assert ep.nodes[3].context.get("facing") == "oop-check"


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

