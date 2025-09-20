from __future__ import annotations

import random

import pytest

from gto_trainer.dynamic.generator import EpisodeBuilder
from gto_trainer.dynamic.policy import resolve_for, river_options, turn_options
from gto_trainer.dynamic.seating import BB, SB, SeatAssignment


def _build_episode(seed: int, style: str = "balanced") -> tuple[EpisodeBuilder, object]:
    rng = random.Random(seed)
    builder = EpisodeBuilder(rng, seats=SeatAssignment(hero=BB, rival=SB), rival_style=style)
    return builder, builder.build()


def test_episode_builder_produces_both_turn_modes():
    modes: set[str] = set()
    for seed in range(20):
        _, episode = _build_episode(seed)
        turn_node = next(node for node in episode.nodes if node.street == "turn")
        modes.add(str(turn_node.context.get("facing")))
        if {"bet", "check"}.issubset(modes):
            break
    assert {"bet", "check"}.issubset(modes)


def test_episode_builder_produces_river_leads():
    modes: set[str] = set()
    for seed in range(30):
        _, episode = _build_episode(seed)
        river_node = next(node for node in episode.nodes if node.street == "river")
        modes.add(str(river_node.context.get("facing")))
        if {"oop-check", "bet"}.issubset(modes):
            break
    assert {"oop-check", "bet"}.issubset(modes)


def test_turn_options_include_check_when_checked_to():
    for seed in range(50):
        builder, episode = _build_episode(seed)
        turn_node = next(node for node in episode.nodes if node.street == "turn")
        if str(turn_node.context.get("facing")) != "check":
            continue
        options = turn_options(turn_node, random.Random(seed), mc_trials=80)
        assert any(opt.key == "Check" for opt in options)
        break
    else:
        pytest.fail("unable to generate turn check scenario within 50 seeds")


def test_river_options_vs_bet_offer_call():
    for seed in range(60):
        builder, episode = _build_episode(seed)
        river_node = next(node for node in episode.nodes if node.street == "river")
        if str(river_node.context.get("facing")) != "bet":
            continue
        options = river_options(river_node, random.Random(seed), mc_trials=80)
        assert any(opt.key == "Call" for opt in options)
        break
    else:
        pytest.fail("unable to generate river lead scenario within 60 seeds")


def test_resolve_turn_check_back_transitions_to_river():
    for seed in range(70):
        builder, episode = _build_episode(seed)
        turn_node = next(node for node in episode.nodes if node.street == "turn")
        if str(turn_node.context.get("facing")) != "check":
            continue
        options = turn_options(turn_node, random.Random(seed), mc_trials=80)
        check_option = next(opt for opt in options if opt.key == "Check")
        result = resolve_for(turn_node, check_option, random.Random(seed))
        assert "check back" in (result.note or "")
        break
    else:
        pytest.fail("unable to evaluate turn check scenario within 70 seeds")


def test_resolve_river_fold_finishes_hand():
    for seed in range(80):
        builder, episode = _build_episode(seed)
        river_node = next(node for node in episode.nodes if node.street == "river")
        if str(river_node.context.get("facing")) != "bet":
            continue
        options = river_options(river_node, random.Random(seed), mc_trials=100)
        fold_option = next(opt for opt in options if opt.key == "Fold")
        result = resolve_for(river_node, fold_option, random.Random(seed))
        assert result.hand_ended
        assert "fold river" in (result.note or "").lower()
        break
    else:
        pytest.fail("unable to find river lead scenario within 80 seeds")


def test_style_aggressive_checks_less_than_passive():
    aggressive_bets = 0
    passive_bets = 0
    for seed in range(120):
        _, aggressive_episode = _build_episode(seed, "aggressive")
        _, passive_episode = _build_episode(seed, "passive")
        agg_turn = next(node for node in aggressive_episode.nodes if node.street == "turn")
        pas_turn = next(node for node in passive_episode.nodes if node.street == "turn")
        if str(agg_turn.context.get("facing")) == "bet":
            aggressive_bets += 1
        if str(pas_turn.context.get("facing")) == "bet":
            passive_bets += 1
    assert aggressive_bets > passive_bets


def test_style_aggressive_river_leads_superset_passive():
    aggressive_leads = 0
    passive_leads = 0
    for seed in range(120):
        _, aggressive_episode = _build_episode(seed, "aggressive")
        _, passive_episode = _build_episode(seed, "passive")
        agg_river = next(node for node in aggressive_episode.nodes if node.street == "river")
        pas_river = next(node for node in passive_episode.nodes if node.street == "river")
        if str(agg_river.context.get("facing")) == "bet":
            aggressive_leads += 1
        if str(pas_river.context.get("facing")) == "bet":
            passive_leads += 1
    assert aggressive_leads > passive_leads
