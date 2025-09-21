from __future__ import annotations

import random

from gtotrainer.dynamic import equity as eq
from gtotrainer.dynamic.cards import str_to_int
from gtotrainer.dynamic.equity import estimate_equity, hero_equity_vs_combo


def test_equity_royal_flush_with_hero_cards_is_certain_win():
    # Hero: As Ks; Board: Qs Js Ts 2c 3d -> hero holds the nut royal flush
    hero = [str_to_int("As"), str_to_int("Ks")]
    board = [str_to_int(x) for x in ["Qs", "Js", "Ts", "2c", "3d"]]
    rng = random.Random(1)
    eq = estimate_equity(hero, board, None, rng, trials=40)
    assert eq == 1.0


def test_equity_royal_flush_on_board_is_always_tie():
    # Hero has irrelevant cards; board is a royal flush using only board cards
    hero = [str_to_int("2c"), str_to_int("3d")]
    board = [str_to_int(x) for x in ["Ts", "Js", "Qs", "Ks", "As"]]
    rng = random.Random(2)
    eq = estimate_equity(hero, board, None, rng, trials=50)
    assert eq == 0.5


def test_equity_exact_enumeration_flop_matches_manual_probability():
    hero = [str_to_int("Ah"), str_to_int("Kh")]
    board = [str_to_int(x) for x in ["Qh", "Jh", "2c"]]
    rival_combo = (str_to_int("9s"), str_to_int("9d"))

    eq = hero_equity_vs_combo(hero, board, rival_combo, trials=200)
    # Hero has nut flush draw + straight draw vs middle pair; exact equity ≈ 0.63535.
    assert abs(eq - 0.63535) < 0.002


def test_equity_exact_enumeration_turn_is_deterministic():
    hero = [str_to_int("As"), str_to_int("Kd")]
    board = [str_to_int(x) for x in ["Ah", "Ac", "7d", "2s"]]
    rival_combo = (str_to_int("Qc"), str_to_int("Qd"))

    first = hero_equity_vs_combo(hero, board, rival_combo, trials=40)
    second = hero_equity_vs_combo(hero, board, rival_combo, trials=80)
    assert abs(first - second) < 1e-9


def test_adaptive_monte_carlo_respects_minimum(monkeypatch):
    monkeypatch.setattr(eq, "_MIN_MONTE_TRIALS", 20, raising=False)
    monkeypatch.setattr(eq, "_MAX_MONTE_TRIALS", 60, raising=False)
    monkeypatch.setattr(eq, "_MONTE_CHUNK", 5, raising=False)
    monkeypatch.setattr(eq, "_TARGET_STD_ERROR", 1e-6, raising=False)
    eq._cached_equity.cache_clear()

    hero = [str_to_int("Ah"), str_to_int("7d")]
    rival = (str_to_int("Kc"), str_to_int("Qd"))

    hero_equity_vs_combo(hero, [], rival, trials=5)
    assert eq._LAST_MONTE_TRIALS >= eq._MIN_MONTE_TRIALS


def test_adaptive_monte_carlo_is_deterministic():
    hero = [str_to_int("9h"), str_to_int("8d")]
    rival = (str_to_int("Jc"), str_to_int("Td"))
    eq._cached_equity.cache_clear()
    first = hero_equity_vs_combo(hero, [], rival, trials=150)
    second = hero_equity_vs_combo(hero, [], rival, trials=150)
    assert first == second


def test_canonical_equity_cache_hits_on_isomorphic_inputs():
    hero_a = [str_to_int("As"), str_to_int("Kd")]
    board_a = [str_to_int(x) for x in ["Qs", "Jh", "2d", "7c"]]
    rival_a = (str_to_int("9c"), str_to_int("8s"))

    suit_rot = {"s": "h", "h": "d", "d": "c", "c": "s"}

    def _map(card: str) -> str:
        return card[0] + suit_rot[card[1]]

    hero_b = [str_to_int(_map("As")), str_to_int(_map("Kd"))]
    board_b = [str_to_int(_map(x)) for x in ["Qs", "Jh", "2d", "7c"]]
    rival_b = (str_to_int(_map("9c")), str_to_int(_map("8s")))

    eq._cached_equity.cache_clear()
    first = hero_equity_vs_combo(hero_a, board_a, rival_a, trials=120)
    info_after_first = eq._cached_equity.cache_info()

    second = hero_equity_vs_combo(hero_b, board_b, rival_b, trials=120)
    info_after_second = eq._cached_equity.cache_info()

    assert abs(first - second) < 1e-9
    assert info_after_second.hits >= info_after_first.hits + 1
