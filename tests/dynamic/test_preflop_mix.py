from __future__ import annotations

from gtotrainer.dynamic import preflop_mix


def test_action_profile_defend_share_consistent() -> None:
    combo = (0, 1)
    profile = preflop_mix.action_profile_for_combo(combo, open_size=2.5)
    defend = profile["defend"]
    call = profile.get("call", 0.0)
    threebet = profile.get("threebet", 0.0)
    jam = profile.get("jam", 0.0)
    assert 0.0 <= defend <= 1.0
    assert abs(defend - (call + threebet + jam)) < 1e-6


def test_continue_combos_respects_blockers() -> None:
    blocked = [0, 1]
    combos = preflop_mix.continue_combos(open_size=2.5, blocked=blocked, minimum_defend=0.02)
    assert combos  # ensure we still have content
    for a, b in combos:
        assert a not in blocked
        assert b not in blocked
