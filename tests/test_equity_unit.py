from __future__ import annotations

import random

from gto_trainer.dynamic.cards import str_to_int
from gto_trainer.dynamic.equity import estimate_equity


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

