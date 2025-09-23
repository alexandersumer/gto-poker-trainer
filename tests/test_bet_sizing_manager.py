from __future__ import annotations

from gtotrainer.dynamic.bet_sizing import BetSizingManager


def test_preflop_expands_when_regret_high() -> None:
    manager = BetSizingManager()
    base = manager.preflop_raise_sizes(
        open_size=2.5,
        hero_contrib=1.0,
        hero_stack=99.0,
        rival_stack=99.0,
    )
    assert len(base) >= 2

    high_regret = max(base)
    manager.observe_preflop(
        open_size=2.5,
        hero_contrib=1.0,
        hero_stack=99.0,
        rival_stack=99.0,
        observations=[(high_regret, 0.7, 1.2)],
    )
    expanded = manager.preflop_raise_sizes(
        open_size=2.5,
        hero_contrib=1.0,
        hero_stack=99.0,
        rival_stack=99.0,
    )
    assert len(expanded) >= len(base)


def test_postflop_fraction_adjustment() -> None:
    manager = BetSizingManager()
    base = manager.postflop_bet_fractions(street="flop", context="test", base_fractions=(0.33, 0.5, 0.75))
    assert base == (0.33, 0.5, 0.75)

    manager.observe_postflop(
        street="flop",
        context="test",
        observations=[(0.75, 0.6, 0.9)],
    )
    adjusted = manager.postflop_bet_fractions(street="flop", context="test", base_fractions=(0.33, 0.5, 0.75))
    assert len(adjusted) >= len(base)
