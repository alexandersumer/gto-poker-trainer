from __future__ import annotations

import random

from gto_trainer.dynamic.cards import str_to_int
from gto_trainer.dynamic.generator import Node
from gto_trainer.dynamic.policy import options_for


def _sample_node() -> Node:
    return Node(
        street="preflop",
        description="integration",
        pot_bb=3.5,
        effective_bb=100.0,
        hero_cards=[str_to_int("As"), str_to_int("Kd")],
        board=[],
        actor="BB",
        context={"open_size": 2.5},
    )


def test_options_for_attaches_cfr_metadata() -> None:
    node = _sample_node()
    options = options_for(node, random.Random(0), mc_trials=120)

    cfr_options = [opt for opt in options if opt.meta and opt.meta.get("supports_cfr")]
    assert cfr_options, "expected at least one option to participate in CFR refinement"

    for opt in cfr_options:
        assert opt.gto_freq is not None
        assert opt.meta is not None
        assert opt.meta.get("cfr_backend") == "local_cfr_v1"
        assert "hero_ev_fold" in opt.meta
        assert "hero_ev_continue" in opt.meta
        assert "villain_ev_fold" in opt.meta
        assert "villain_ev_continue" in opt.meta

    passive = [opt for opt in options if not (opt.meta and opt.meta.get("supports_cfr"))]
    for opt in passive:
        assert opt.gto_freq is None
