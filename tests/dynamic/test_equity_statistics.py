from __future__ import annotations

import random

import gtotrainer.dynamic.equity as eq
from gtotrainer.dynamic.cards import str_to_int
from gtotrainer.dynamic.equity import (
    EquityEstimate,
    estimate_equity_with_stats,
    hero_equity_vs_combo,
    hero_equity_vs_combo_stats,
)


def test_estimate_equity_with_stats_hits_target_error() -> None:
    hero = [str_to_int("As"), str_to_int("Kd")]
    rng = random.Random(1234)
    estimate = estimate_equity_with_stats(hero, [], None, rng, trials=600, target_std_error=0.015)

    assert isinstance(estimate, EquityEstimate)
    assert 0.0 <= estimate.equity <= 1.0
    assert estimate.trials >= 600
    assert estimate.std_error <= 0.015 + 1e-6
    assert estimate.std_error == eq._LAST_MONTE_STD_ERROR


def test_hero_equity_vs_combo_stats_matches_scalar_equity() -> None:
    hero = [str_to_int("Ah"), str_to_int("Kh")]
    board = [str_to_int(x) for x in ["Qh", "Jh", "2c"]]
    rival = (str_to_int("9s"), str_to_int("9d"))

    scalar = hero_equity_vs_combo(hero, board, rival, trials=200)
    stats = hero_equity_vs_combo_stats(hero, board, rival, trials=200)

    assert isinstance(stats, EquityEstimate)
    assert abs(stats.equity - scalar) < 1e-9
    assert stats.std_error == 0.0
    assert stats.trials > 0
