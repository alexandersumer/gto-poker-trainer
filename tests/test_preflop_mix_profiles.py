from __future__ import annotations

from collections import defaultdict

import pytest

from gtotrainer.dynamic import cards, preflop_mix


def _combo(hand: str) -> tuple[int, int]:
    assert len(hand) == 4, "Hand must be like 'Qs4h'"
    return cards.str_to_int(hand[:2]), cards.str_to_int(hand[2:])


def _aggregate_mix(open_size: float) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    combos = preflop_mix._sorted_combos()
    for combo in combos:
        mix = preflop_mix.normalise_mix(preflop_mix.action_mix_for_combo(combo, open_size=open_size))
        for action, freq in mix.items():
            totals[action] += freq
    n = len(combos)
    return {action: freq / n for action, freq in totals.items()}


@pytest.mark.parametrize(
    ("open_size", "target_defend", "tolerance"),
    [
        (2.0, 0.68, 0.035),
        (2.5, 0.62, 0.04),
        (3.0, 0.38, 0.05),
    ],
)
def test_bb_defence_rates_track_solver_guidance(open_size: float, target_defend: float, tolerance: float) -> None:
    agg = _aggregate_mix(open_size)
    defend = 1.0 - agg.get("fold", 0.0)
    assert abs(defend - target_defend) <= tolerance


def test_weak_offsuit_hands_fold_more_as_opens_grow() -> None:
    q4o = _combo("Qs4h")
    mix_small = preflop_mix.normalise_mix(preflop_mix.action_mix_for_combo(q4o, open_size=2.0))
    mix_mid = preflop_mix.normalise_mix(preflop_mix.action_mix_for_combo(q4o, open_size=2.5))
    mix_large = preflop_mix.normalise_mix(preflop_mix.action_mix_for_combo(q4o, open_size=3.0))

    assert mix_small.get("fold", 0.0) == 0.0
    assert mix_mid.get("fold", 0.0) >= 0.55
    assert mix_large.get("fold", 0.0) == 1.0


def test_value_hands_prefer_aggressive_actions() -> None:
    aa = _combo("AcAd")
    mix = preflop_mix.normalise_mix(preflop_mix.action_mix_for_combo(aa, open_size=2.5))
    assert mix.get("threebet", 0.0) + mix.get("jam", 0.0) >= 0.95

    kjo = _combo("KdJh")
    mix_small = preflop_mix.normalise_mix(preflop_mix.action_mix_for_combo(kjo, open_size=2.0))
    mix_large = preflop_mix.normalise_mix(preflop_mix.action_mix_for_combo(kjo, open_size=3.0))
    assert mix_small.get("threebet", 0.0) > 0.3
    assert mix_large.get("threebet", 0.0) >= mix_small.get("threebet", 0.0)


def test_cfr_profile_sum_and_monotonicity() -> None:
    aa = _combo("AsAh")
    profile = preflop_mix.action_profile_for_combo(aa, open_size=2.5)
    total = profile.get("fold", 0.0) + profile.get("call", 0.0) + profile.get("threebet", 0.0) + profile.get("jam", 0.0)
    assert pytest.approx(total, rel=1e-6, abs=1e-6) == 1.0
    assert profile.get("threebet", 0.0) + profile.get("jam", 0.0) >= 0.3


def test_cfr_profile_adjusts_to_open_size() -> None:
    q4o = _combo("Qs4h")
    profile_small = preflop_mix.action_profile_for_combo(q4o, open_size=2.0)
    profile_large = preflop_mix.action_profile_for_combo(q4o, open_size=3.0)
    assert profile_small.get("fold", 0.0) <= 0.4
    assert profile_large.get("fold", 0.0) >= 0.3
