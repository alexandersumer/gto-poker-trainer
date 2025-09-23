from __future__ import annotations

from gtotrainer.data.range_loader import get_repository
from gtotrainer.dynamic.cards import str_to_int


def test_sb_open_range_weights_and_blockers() -> None:
    repo = get_repository()
    combos, weights = repo.range_for("sb_open", 2.5, blocked_cards=[str_to_int("As")])
    assert combos, "expected combos for sb_open"
    assert weights is not None
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 1e-6
    for combo in combos:
        assert combo[0] != str_to_int("As") and combo[1] != str_to_int("As")


def test_bb_defend_interpolates_between_sizings() -> None:
    repo = get_repository()
    combos_low, weights_low = repo.range_for("bb_defend", 2.3)
    combos_high, weights_high = repo.range_for("bb_defend", 2.5)
    combos_mid, weights_mid = repo.range_for("bb_defend", 2.4)

    assert combos_low and combos_high and combos_mid
    assert weights_low is not None and weights_high is not None and weights_mid is not None

    # Interpolated distribution should include hands from both neighbours.
    low_ordered = sorted(weights_low, key=weights_low.get, reverse=True)
    high_ordered = sorted(weights_high, key=weights_high.get, reverse=True)
    mid_ordered = sorted(weights_mid, key=weights_mid.get, reverse=True)
    low_top = set(low_ordered[:20])
    high_top = set(high_ordered[:20])
    mid_top = set(mid_ordered[:40])
    assert low_top.intersection(mid_top)
    assert high_top.intersection(mid_top)
