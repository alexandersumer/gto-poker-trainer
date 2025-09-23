from __future__ import annotations

import random

from gtotrainer.dynamic.generator import EpisodeBuilder, SeatAssignment
from gtotrainer.dynamic.policy import options_for, reset_bet_sizing_state
from gtotrainer.dynamic.seating import BB, SB


def _episode() -> tuple:
    rng = random.Random(42)
    builder = EpisodeBuilder(rng, SeatAssignment(hero=BB, rival=SB))
    episode = builder.build()
    return episode.nodes[0], episode.nodes[1]


def test_preflop_options_include_cfr_metrics() -> None:
    reset_bet_sizing_state()
    preflop_node, _ = _episode()
    opts = options_for(preflop_node, random.Random(7), mc_trials=160)
    cfr_opts = [opt for opt in opts if opt.meta and opt.meta.get("supports_cfr")]
    assert cfr_opts, "expected CFR-participating options"
    assert all(opt.meta.get("cfr_max_regret") is not None for opt in cfr_opts)
    assert all(opt.meta.get("cfr_iterations") for opt in cfr_opts)


def test_flop_options_attach_bet_context_and_fraction() -> None:
    reset_bet_sizing_state()
    _, flop_node = _episode()
    opts = options_for(flop_node, random.Random(11), mc_trials=160)
    bet_opts = [opt for opt in opts if opt.meta and opt.meta.get("action") == "bet"]
    assert bet_opts, "expected flop betting options"
    assert any(opt.meta.get("sizing_fraction") is not None for opt in bet_opts)
    assert any(opt.meta.get("bet_context") for opt in bet_opts)
