from __future__ import annotations

import random

from gto_poker_trainer_cli.dynamic.cards import str_to_int
from gto_poker_trainer_cli.dynamic.equity import estimate_equity, hero_equity_vs_combo


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
