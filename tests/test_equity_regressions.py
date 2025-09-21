from __future__ import annotations

from itertools import combinations

import eval7
import pytest

from gtotrainer.dynamic.cards import card_int_to_str, str_to_int
from gtotrainer.dynamic.equity import hero_equity_vs_combo, hero_equity_vs_range


def _eval7_exact(hero: list[int], board: list[int], rival: tuple[int, int]) -> float:
    hero_cards = [eval7.Card(card_int_to_str(c)) for c in hero]
    board_cards = [eval7.Card(card_int_to_str(c)) for c in board]
    rival_cards = [eval7.Card(card_int_to_str(c)) for c in rival]

    need = 5 - len(board_cards)
    assert need >= 0

    if need == 0:
        hero_rank = eval7.evaluate(hero_cards + board_cards)
        rival_rank = eval7.evaluate(rival_cards + board_cards)
        if hero_rank > rival_rank:
            return 1.0
        if hero_rank == rival_rank:
            return 0.5
        return 0.0

    deck = [eval7.Card(card_int_to_str(c)) for c in range(52) if c not in hero and c not in board and c not in rival]
    wins = ties = total = 0

    for fill in combinations(deck, need):
        total += 1
        combined_board = board_cards + list(fill)
        hero_rank = eval7.evaluate(hero_cards + combined_board)
        rival_rank = eval7.evaluate(rival_cards + combined_board)
        if hero_rank > rival_rank:
            wins += 1
        elif hero_rank == rival_rank:
            ties += 1

    return (wins + 0.5 * ties) / total if total else 0.0


@pytest.mark.parametrize(
    "hero,board,rival",
    [
        (
            [str_to_int("As"), str_to_int("Kd")],
            [str_to_int("Qc"), str_to_int("Jh"), str_to_int("2d")],
            (str_to_int("9c"), str_to_int("8s")),
        ),
        (
            [str_to_int("Ah"), str_to_int("Ad")],
            [str_to_int("Ks"), str_to_int("Ts"), str_to_int("2c"), str_to_int("7h")],
            (str_to_int("Kc"), str_to_int("Th")),
        ),
    ],
)
def test_equity_vs_combo_matches_eval7_exact(hero: list[int], board: list[int], rival: tuple[int, int]) -> None:
    expected = _eval7_exact(hero, board, rival)
    result = hero_equity_vs_combo(hero, board, rival, trials=120)
    assert result == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_equity_vs_range_averages_exact_combo_equities() -> None:
    hero = [str_to_int("Ah"), str_to_int("Kd")]
    board = [str_to_int("Qs"), str_to_int("Jh"), str_to_int("3d")]
    rival_combos = [
        (str_to_int("Qd"), str_to_int("Qc")),
        (str_to_int("As"), str_to_int("Th")),
        (str_to_int("9h"), str_to_int("8h")),
        (str_to_int("7s"), str_to_int("7c")),
    ]

    ours = hero_equity_vs_range(hero, board, rival_combos, trials=300)
    expected = sum(_eval7_exact(hero, board, combo) for combo in rival_combos) / len(rival_combos)

    assert ours == pytest.approx(expected, rel=1e-9, abs=1e-9)
