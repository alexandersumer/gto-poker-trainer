from __future__ import annotations

from gtotrainer.dynamic.hand_state import (
    apply_contribution,
    recalculate_pot,
    state_value,
    update_effective_stack,
)


def test_recalculate_pot_uses_contributions() -> None:
    hand_state = {"hero_contrib": 1.5, "rival_contrib": 2.5}

    pot = recalculate_pot(hand_state)

    assert pot == 4.0
    assert hand_state["pot"] == 4.0


def test_apply_contribution_caps_at_stack_and_updates_effective() -> None:
    hand_state = {
        "hero_contrib": 1.0,
        "rival_contrib": 2.0,
        "hero_stack": 2.5,
        "rival_stack": 7.5,
    }
    update_effective_stack(hand_state)
    assert hand_state["effective_stack"] == 2.5

    applied = apply_contribution(hand_state, "hero", 5.0)

    assert applied == 2.5
    assert hand_state["hero_contrib"] == 3.5
    assert hand_state["hero_stack"] == 0.0
    assert hand_state["pot"] == 5.5
    assert hand_state["effective_stack"] == 0.0


def test_state_value_returns_default_for_bad_data() -> None:
    hand_state = {"hero_stack": "not-a-number"}

    value = state_value(hand_state, "hero_stack", default=3.0)

    assert value == 3.0
