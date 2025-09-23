from __future__ import annotations

import random

from gtotrainer.dynamic.generator import generate_episode
from gtotrainer.dynamic.policy import options_for


def _aggressive_actions(options: list) -> list:
    aggressive_labels = {"bet", "raise", "3bet"}
    return [opt for opt in options if opt.meta and opt.meta.get("action") in aggressive_labels]


def _jam_actions(options: list) -> list:
    return [opt for opt in options if opt.meta and opt.meta.get("action") == "jam"]


def test_bet_option_cap_per_street() -> None:
    episode = generate_episode(random.Random(2024))
    mc_trials = 120

    for idx, node in enumerate(episode.nodes):
        rng = random.Random(1000 + idx)
        options = options_for(node, rng, mc_trials)
        aggressive = _aggressive_actions(options)
        jams = _jam_actions(options)

        assert len(aggressive) <= 3
        assert len(jams) <= 1

        if node.street != "preflop":
            assert node.context.get("board_key")
