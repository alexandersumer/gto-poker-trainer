from __future__ import annotations

import random

import pytest

from gto_trainer.dynamic import villain_strategy as vs


def _sample_profile() -> dict:
    combos = [(0, 12), (4, 16), (8, 20), (24, 28)]
    return vs.build_profile(combos, fold_probability=0.35, continue_ratio=0.5)


def test_sample_profile_combo_uses_continue_and_tail_segments():
    profile = _sample_profile()
    rng = random.Random(1)

    first = vs._sample_profile_combo(profile, rng)
    second = vs._sample_profile_combo(profile, rng)

    assert first in {tuple(map(int, combo)) for combo in profile["ranked"]}
    assert second in {tuple(map(int, combo)) for combo in profile["ranked"]}
    assert first != second  # rng seeded to hit different segments


def test_decide_action_without_villain_cards_samples_profile(monkeypatch: pytest.MonkeyPatch):
    profile = _sample_profile()
    meta = {"villain_profile": profile}
    captured: list[tuple[int, int]] = []

    def capture_percentile(_profile, combo):
        captured.append(tuple(combo))
        return 0.5

    monkeypatch.setattr(vs, "_percentile_for_combo", capture_percentile)
    rng = random.Random(2)

    vs.decide_action(meta, None, rng)

    assert captured, "expected percentile helper to be invoked"
    assert captured[0] in {tuple(map(int, combo)) for combo in profile["ranked"]}
