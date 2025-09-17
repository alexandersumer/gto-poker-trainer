from __future__ import annotations

import random
from collections.abc import Iterable
from functools import lru_cache

from .cards import card_int_to_str, str_to_treys


def estimate_equity(
    hero: list[int],
    board: list[int],
    known_villain: list[int] | None,
    rng: random.Random,
    trials: int = 400,
) -> float:
    """Monte Carlo equity estimate vs random villain hand (or a known one).

    - hero, board are 0..51 int-coded cards; board may be 0..5 cards.
    - If known_villain is None, sample random opponent hand without collision each trial.
    - Returns equity in [0,1].
    """
    from treys import Evaluator

    evaluator = Evaluator()
    wins = 0
    ties = 0
    seen = set(hero + board + (known_villain or []))

    # Convert hero ledger once
    hero_cards = [str_to_treys(card_int_to_str(c)) for c in hero]
    preset_board = [str_to_treys(card_int_to_str(c)) for c in board]

    for _ in range(trials):
        remaining = [c for c in range(52) if c not in seen]
        v = rng.sample(remaining, 2) if known_villain is None else known_villain

        # Board completion
        need = 5 - len(board)
        remaining2 = [c for c in remaining if c not in v]
        fill = rng.sample(remaining2, need)

        board_now = preset_board + [str_to_treys(card_int_to_str(c)) for c in fill]
        villain_cards = [str_to_treys(card_int_to_str(c)) for c in v]

        hr = evaluator.evaluate(hero_cards, board_now)
        vr = evaluator.evaluate(villain_cards, board_now)
        if hr < vr:
            wins += 1
        elif hr == vr:
            ties += 1

    return (wins + 0.5 * ties) / max(1, trials)


def _sorted_tuple(cards: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(cards))


@lru_cache(maxsize=50000)
def _cached_equity(hero: tuple[int, ...], board: tuple[int, ...], villain: tuple[int, ...], trials: int) -> float:
    seed = hash((hero, board, villain, trials)) & 0xFFFFFFFF
    rng = random.Random(seed)
    hero_list = list(hero)
    board_list = list(board)
    villain_list = list(villain) if villain else None
    return estimate_equity(hero_list, board_list, villain_list, rng, trials)


def hero_equity_vs_combo(hero: list[int], board: list[int], combo: tuple[int, int], trials: int) -> float:
    hero_t = _sorted_tuple(hero)
    board_t = _sorted_tuple(board)
    villain_t = _sorted_tuple(combo)
    return _cached_equity(hero_t, board_t, villain_t, trials)


def hero_equity_vs_range(
    hero: list[int],
    board: list[int],
    combos: Iterable[tuple[int, int]],
    trials: int,
) -> float:
    combos_list = list(combos)
    if not combos_list:
        return 0.0
    total = 0.0
    for combo in combos_list:
        total += hero_equity_vs_combo(hero, board, combo, trials)
    return total / len(combos_list)
