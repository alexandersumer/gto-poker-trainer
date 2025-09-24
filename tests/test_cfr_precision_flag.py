from __future__ import annotations

import random

import pytest

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


def test_cfr_backend_uses_high_iteration_budget() -> None:
    reset_bet_sizing_state()
    node = _preflop_node()
    options = options_for(node, random.Random(5), mc_trials=120)
    avg_iters = _avg_iterations(options)
    assert avg_iters >= 1500


def test_cfr_baseline_ev_is_tracked() -> None:
    reset_bet_sizing_state()
    node = _preflop_node()
    options = options_for(node, random.Random(7), mc_trials=96)

    for opt in options:
        if not opt.meta or not opt.meta.get("supports_cfr"):
            continue
        baseline = opt.meta.get("baseline_ev")
        avg_ev = opt.meta.get("cfr_avg_ev")
        assert baseline is not None
        assert avg_ev is not None
        assert opt.ev == pytest.approx(float(avg_ev), rel=1e-9)
