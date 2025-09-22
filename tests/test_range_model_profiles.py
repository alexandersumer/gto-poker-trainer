from __future__ import annotations

import math

import pytest

from gtotrainer.dynamic import cards, hand_strength, range_model


def _combo(hand: str) -> tuple[int, int]:
    return cards.str_to_int(hand[:2]), cards.str_to_int(hand[2:])


def _ratio(r: list[tuple[int, int]]) -> float:
    total = len(range_model.combos_without_blockers())
    return len(r) / total


@pytest.mark.parametrize(
    ("open_size", "target", "tolerance"),
    [
        (2.0, 0.9, 0.03),
        (2.5, 0.82, 0.04),
        (3.0, 0.7, 0.05),
    ],
)
def test_sb_open_range_percentages(open_size: float, target: float, tolerance: float) -> None:
    ratio = _ratio(range_model.rival_sb_open_range(open_size))
    assert math.isclose(ratio, target, rel_tol=tolerance, abs_tol=tolerance)


@pytest.mark.parametrize(
    ("open_size", "target", "tolerance"),
    [
        (2.0, 0.66, 0.04),
        (2.5, 0.54, 0.05),
        (3.0, 0.36, 0.06),
    ],
)
def test_bb_defend_range_percentages(open_size: float, target: float, tolerance: float) -> None:
    ratio = _ratio(range_model.rival_bb_defend_range(open_size))
    assert math.isclose(ratio, target, rel_tol=tolerance, abs_tol=tolerance)


def test_stack_specific_range_tightens_profiles() -> None:
    default_ratio = _ratio(range_model.rival_sb_open_range(2.5))
    stacked_ratio = _ratio(range_model.rival_sb_open_range(2.5, stack_depth=93))
    assert stacked_ratio < default_ratio


def test_unknown_stack_uses_default_profiles() -> None:
    default_ratio = _ratio(range_model.rival_bb_defend_range(2.3))
    unknown_ratio = _ratio(range_model.rival_bb_defend_range(2.3, stack_depth=180))
    assert math.isclose(default_ratio, unknown_ratio, rel_tol=1e-6, abs_tol=1e-6)


def test_playability_score_orders_connectivity() -> None:
    q4o = _combo("Qs4h")
    q9s = _combo("Qh9h")
    score_q4o = hand_strength.combo_playability_score(q4o)
    score_q9s = hand_strength.combo_playability_score(q9s)
    assert score_q9s > score_q4o
