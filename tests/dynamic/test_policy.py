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


def test_preflop_option_coaching_notes_cover_key_numbers() -> None:
    """Ensure rationale strings expose actionable odds and equity cues."""

    node = Node(
        street="preflop",
        description="SB opens",
        pot_bb=3.5,
        effective_bb=97.5,
        hero_cards=[0, 12],
        board=[],
        actor="BB",
        context={
            "open_size": 2.5,
            "hand_state": _hand_state(),
            "hero_seat": "BB",
            "rival_seat": "SB",
        },
    )

    options = preflop_options(node, random.Random(7), mc_trials=80)
    lookup = {opt.key.lower(): opt for opt in options}

    call = lookup.get("call")
    assert call is not None and call.why
    call_text = call.why.lower()
    assert "bb" in call_text
    assert "equity" in call_text
    assert "%" in call_text or "percent" in call_text

    three_bets = [opt for opt in options if "3-bet" in opt.key.lower()]
    if three_bets:
        three_bet_text = three_bets[0].why.lower()
        assert three_bet_text
        assert "fold" in three_bet_text
        assert "equity" in three_bet_text
        assert "ev" in three_bet_text

    jam = lookup.get("all-in")
    if jam:
        jam_text = jam.why.lower()
        assert jam_text
        assert "fold" in jam_text
        assert "equity" in jam_text
