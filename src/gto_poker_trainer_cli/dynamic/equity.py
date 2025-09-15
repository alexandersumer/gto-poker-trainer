from __future__ import annotations

import random

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
