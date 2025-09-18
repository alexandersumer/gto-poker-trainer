from __future__ import annotations

import random

from gto_trainer.dynamic import equity as eq
from gto_trainer.dynamic.cards import str_to_int
from gto_trainer.dynamic.equity import estimate_equity, hero_equity_vs_combo


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
    villain_combo = (str_to_int("9s"), str_to_int("9d"))

    eq = hero_equity_vs_combo(hero, board, villain_combo, trials=200)
    # Hero has nut flush draw + straight draw vs middle pair; exact equity ≈ 0.63535.
    assert abs(eq - 0.63535) < 0.002


def test_equity_exact_enumeration_turn_is_deterministic():
    hero = [str_to_int("As"), str_to_int("Kd")]
    board = [str_to_int(x) for x in ["Ah", "Ac", "7d", "2s"]]
    villain_combo = (str_to_int("Qc"), str_to_int("Qd"))

    first = hero_equity_vs_combo(hero, board, villain_combo, trials=40)
    second = hero_equity_vs_combo(hero, board, villain_combo, trials=80)
    assert abs(first - second) < 1e-9


def test_adaptive_monte_carlo_respects_minimum(monkeypatch):
    monkeypatch.setattr(eq, "_MIN_MONTE_TRIALS", 20, raising=False)
    monkeypatch.setattr(eq, "_MAX_MONTE_TRIALS", 60, raising=False)
    monkeypatch.setattr(eq, "_MONTE_CHUNK", 5, raising=False)
    monkeypatch.setattr(eq, "_TARGET_STD_ERROR", 1e-6, raising=False)
    eq._cached_equity.cache_clear()

    hero = [str_to_int("Ah"), str_to_int("7d")]
    villain = (str_to_int("Kc"), str_to_int("Qd"))

    hero_equity_vs_combo(hero, [], villain, trials=5)
    assert eq._LAST_MONTE_TRIALS >= eq._MIN_MONTE_TRIALS


def test_adaptive_monte_carlo_is_deterministic():
    hero = [str_to_int("9h"), str_to_int("8d")]
    villain = (str_to_int("Jc"), str_to_int("Td"))
    eq._cached_equity.cache_clear()
    first = hero_equity_vs_combo(hero, [], villain, trials=150)
    second = hero_equity_vs_combo(hero, [], villain, trials=150)
    assert first == second
