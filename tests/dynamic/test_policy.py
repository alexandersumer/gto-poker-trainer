from __future__ import annotations

import random

from gtotrainer.dynamic.episode import Node
from gtotrainer.dynamic.policy import preflop_options


def _hand_state() -> dict[str, float]:
    return {
        "pot": 3.5,
        "hero_contrib": 1.0,
        "rival_contrib": 2.5,
        "hero_stack": 99.0,
        "rival_stack": 97.5,
        "effective_stack": 97.5,
    }


def test_preflop_options_surface_solver_frequencies() -> None:
    node = Node(
        street="preflop",
        description="SB opens",
        pot_bb=3.5,
        effective_bb=97.5,
        hero_cards=[0, 12],  # random offsuit combo
        board=[],
        actor="BB",
        context={
            "open_size": 2.5,
            "hand_state": _hand_state(),
            "hero_seat": "BB",
            "rival_seat": "SB",
        },
    )

    options = preflop_options(node, random.Random(1), mc_trials=80)
    assert options, "expected at least one option"

    fold = options[0]
    assert fold.gto_freq is not None
    assert fold.meta and "solver_mix" in fold.meta

    threebets = [opt for opt in options if "3-bet" in opt.key]
    if threebets:
        assert all(opt.gto_freq is not None for opt in threebets)

    jams = [opt for opt in options if opt.key.lower() == "all-in"]
    if jams:
        assert jams[0].gto_freq is not None
