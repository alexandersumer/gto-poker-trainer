from __future__ import annotations

import random

import pytest

from gtotrainer.dynamic.generator import EpisodeBuilder, SeatAssignment
from gtotrainer.dynamic.policy import options_for, reset_bet_sizing_state, resolve_for
from gtotrainer.dynamic.rival_strategy import RivalDecision
from gtotrainer.dynamic.seating import BB, SB


def _preflop_node(seed: int = 91):
    rng = random.Random(seed)
    builder = EpisodeBuilder(rng, SeatAssignment(hero=BB, rival=SB))
    episode = builder.build()
    node = episode.nodes[0]
    hand_state = node.context["hand_state"]
    return node, hand_state


def _three_bet_option(node, rng: random.Random):
    reset_bet_sizing_state()
    options = options_for(node, rng, mc_trials=96)
    for opt in options:
        if opt.meta and opt.meta.get("action") == "3bet":
            return opt
    raise AssertionError("no 3-bet option available")


def test_resolve_for_continue_keeps_range_and_tracks_aggression(monkeypatch: pytest.MonkeyPatch) -> None:
    node, hand_state = _preflop_node(101)
    three_bet = _three_bet_option(node, random.Random(2))
    assert three_bet.meta is not None and three_bet.meta.get("rival_continue_range")

    def _always_continue(_meta, _cards, _rng):  # noqa: D401
        return RivalDecision(folds=False)

    monkeypatch.setattr("gtotrainer.dynamic.rival_strategy.decide_action", _always_continue)

    prior_aggr = hand_state.get("rival_adapt", {}).get("aggr", 0)
    resolve_for(node, three_bet, random.Random(5))

    adapt = hand_state.get("rival_adapt", {})
    assert adapt.get("aggr") == prior_aggr + 1
    assert "rival_continue_range" in hand_state
    assert hand_state["rival_continue_range"]
    assert hand_state["pot"] == pytest.approx(hand_state["hero_contrib"] + hand_state["rival_contrib"])
    assert hand_state["effective_stack"] == pytest.approx(min(hand_state["hero_stack"], hand_state["rival_stack"]))


def test_resolve_for_fold_clears_range(monkeypatch: pytest.MonkeyPatch) -> None:
    node, hand_state = _preflop_node(202)
    three_bet = _three_bet_option(node, random.Random(3))

    def _always_fold(_meta, _cards, _rng):  # noqa: D401
        return RivalDecision(folds=True)

    monkeypatch.setattr("gtotrainer.dynamic.rival_strategy.decide_action", _always_fold)

    resolve_for(node, three_bet, random.Random(8))
    assert hand_state.get("hand_over") is True
    assert "rival_continue_range" not in hand_state
    assert "rival_continue_weights" not in hand_state
    assert hand_state["pot"] == pytest.approx(
        hand_state.get("hero_contrib", 0.0) + hand_state.get("rival_contrib", 0.0)
    )
    assert hand_state["effective_stack"] == pytest.approx(
        min(hand_state.get("hero_stack", 0.0), hand_state.get("rival_stack", 0.0))
    )
