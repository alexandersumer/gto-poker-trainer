from __future__ import annotations

import random

from gtotrainer.core import feature_flags
from gtotrainer.dynamic.generator import EpisodeBuilder, SeatAssignment
from gtotrainer.dynamic.policy import options_for, reset_bet_sizing_state
from gtotrainer.dynamic.seating import BB, SB


def _preflop_node() -> object:
    rng = random.Random(24)
    builder = EpisodeBuilder(rng, SeatAssignment(hero=BB, rival=SB))
    episode = builder.build()
    return episode.nodes[0]


def _avg_iterations(options) -> float:
    values = [opt.meta.get("cfr_iterations") for opt in options if opt.meta and opt.meta.get("supports_cfr")]
    if not values:
        raise AssertionError("expected CFR-enabled options in test fixture")
    return sum(float(v) for v in values) / len(values)


def test_solver_high_precision_flag_increases_iterations() -> None:
    reset_bet_sizing_state()
    node = _preflop_node()
    base_options = options_for(node, random.Random(5), mc_trials=120)
    base_avg = _avg_iterations(base_options)

    reset_bet_sizing_state()
    node_hp = _preflop_node()
    with feature_flags.override(enable={"solver.high_precision_cfr"}):
        precise_options = options_for(node_hp, random.Random(5), mc_trials=120)

    precise_avg = _avg_iterations(precise_options)
    assert precise_avg > base_avg
