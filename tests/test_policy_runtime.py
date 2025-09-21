from __future__ import annotations

import random
from time import perf_counter

from gtotrainer.dynamic.cards import str_to_int
from gtotrainer.dynamic.generator import Node
from gtotrainer.dynamic.policy import preflop_options


def _basic_node() -> Node:
    hero = [str_to_int("As"), str_to_int("Kd")]
    return Node(
        street="preflop",
        description="runtime check",
        pot_bb=3.5,
        effective_bb=100.0,
        hero_cards=hero,
        board=[],
        actor="BB",
        context={"open_size": 2.5},
    )


def test_preflop_options_completes_quickly():
    node = _basic_node()
    start = perf_counter()
    preflop_options(node, random.Random(0), mc_trials=120)
    elapsed = perf_counter() - start
    assert elapsed < 0.9
