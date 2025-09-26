from __future__ import annotations

import random

from gtotrainer.dynamic.cards import str_to_int
from gtotrainer.dynamic.equity import estimate_equity_with_stats, hero_equity_vs_combo


def _cards(hand: str) -> list[int]:
    return [str_to_int(hand[:2]), str_to_int(hand[2:])]


def test_adaptive_monte_carlo_hits_requested_precision() -> None:
    hero = _cards("AhKd")
    board = [str_to_int(card) for card in ("Qs", "7c", "3d")]
    rival = (str_to_int("Qc"), str_to_int("Qd"))

    rng = random.Random(1337)
    target = 0.01
    stats = estimate_equity_with_stats(hero, board, list(rival), rng, trials=64, target_std_error=target)

    assert stats.trials >= 64
    assert stats.std_error <= target + 1e-6


def test_monte_carlo_tracks_enumerated_equity() -> None:
    hero = _cards("AhKd")
    board = [str_to_int(card) for card in ("Qs", "7c", "3d")]
    rival = (str_to_int("Qc"), str_to_int("Qd"))

    rng = random.Random(42)
    stats = estimate_equity_with_stats(hero, board, list(rival), rng, trials=96, target_std_error=0.006)
    exact = hero_equity_vs_combo(hero, board, rival, 1)

    assert abs(stats.equity - exact) <= 3 * stats.std_error
