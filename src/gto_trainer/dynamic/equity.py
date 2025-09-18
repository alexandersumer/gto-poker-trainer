from __future__ import annotations

import math
import random
from collections.abc import Iterable
from functools import lru_cache
from itertools import combinations

from treys import Card, Evaluator

from .cards import card_int_to_str


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
    wins = 0
    ties = 0

    hero_cards = _treys_cards(hero)
    preset_board = _treys_cards(board)

    seen = set(hero) | set(board)
    deck = [c for c in range(52) if c not in seen]

    need = 5 - len(board)
    if need < 0:
        raise ValueError("Board cannot have more than 5 cards")

    villain_fixed = None
    remaining_deck = deck
    if known_villain is not None:
        villain_fixed = _treys_cards(known_villain)
        remaining_deck = [c for c in deck if c not in known_villain]

    for _ in range(trials):
        if villain_fixed is None:
            picks = rng.sample(deck, 2 + need)
            villain_raw = picks[:2]
            fill_raw = picks[2:]
            villain_cards = _treys_cards(villain_raw)
        else:
            villain_cards = villain_fixed
            fill_raw = rng.sample(remaining_deck, need) if need else ()

        board_now = preset_board + [_TREYS_CACHE[c] for c in fill_raw]

        hr = _EVALUATOR.evaluate(hero_cards, board_now)
        vr = _EVALUATOR.evaluate(villain_cards, board_now)
        if hr < vr:
            wins += 1
        elif hr == vr:
            ties += 1

    return (wins + 0.5 * ties) / max(1, trials)


def _sorted_tuple(cards: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(cards))


_EVALUATOR = Evaluator()
_TREYS_CACHE = [Card.new(card_int_to_str(c)) for c in range(52)]


_MIN_MONTE_TRIALS = 0
_MAX_MONTE_TRIALS = 1000
_MONTE_CHUNK = 150
_TARGET_STD_ERROR = 0.025

# Exposed for tests to introspect how many trials were used in the most recent
# adaptive Monte Carlo run. Not relied upon by runtime logic.
_LAST_MONTE_TRIALS = 0


def _treys_cards(cards: Iterable[int]) -> list[int]:
    return [_TREYS_CACHE[c] for c in cards]


def _enumerate_remaining(hero: tuple[int, ...], board: tuple[int, ...], villain: tuple[int, ...]) -> float:
    """Enumerate all remaining board fillings for len(board) >= 3 for precise equity."""

    # hero/villain are 2-card combos; board is current board cards.
    need = 5 - len(board)
    if need < 0:
        raise ValueError("Board cannot have more than 5 cards")

    hero_cards = _treys_cards(hero)
    villain_cards = _treys_cards(villain)

    if need == 0:
        board_treys = _treys_cards(board)
        hero_rank = _EVALUATOR.evaluate(hero_cards, board_treys)
        villain_rank = _EVALUATOR.evaluate(villain_cards, board_treys)
        if hero_rank < villain_rank:
            return 1.0
        if hero_rank == villain_rank:
            return 0.5
        return 0.0

    known = set(hero) | set(board) | set(villain)
    deck = [c for c in range(52) if c not in known]
    wins = 0
    ties = 0
    total = 0
    board_prefix = list(board)
    hero_cards_eval = hero_cards
    villain_cards_eval = villain_cards

    for fill in combinations(deck, need):
        total += 1
        board_cards = _treys_cards(board_prefix + list(fill))
        hero_rank = _EVALUATOR.evaluate(hero_cards_eval, board_cards)
        villain_rank = _EVALUATOR.evaluate(villain_cards_eval, board_cards)
        if hero_rank < villain_rank:
            wins += 1
        elif hero_rank == villain_rank:
            ties += 1

    return (wins + 0.5 * ties) / total if total else 0.0


@lru_cache(maxsize=50000)
def _cached_equity(
    hero: tuple[int, ...],
    board: tuple[int, ...],
    villain: tuple[int, ...],
    trials: int,
) -> float:
    if len(board) >= 3:
        return _enumerate_remaining(hero, board, villain)
    seed = hash((hero, board, villain, trials)) & 0xFFFFFFFF
    rng = random.Random(seed)
    hero_list = list(hero)
    board_list = list(board)
    villain_list = list(villain) if villain else None
    return _adaptive_monte_carlo(
        hero_list,
        board_list,
        villain_list,
        base_trials=trials,
        rng=rng,
    )


def _adaptive_monte_carlo(
    hero: list[int],
    board: list[int],
    villain: list[int] | None,
    *,
    base_trials: int,
    rng: random.Random,
    min_trials: int | None = None,
    max_trials: int | None = None,
    target_std_error: float | None = None,
) -> float:
    """Return a Monte Carlo equity estimate with adaptive precision.

    The sampler keeps drawing until either the estimated standard error drops
    below ``target_std_error`` or ``max_trials`` is reached. Deterministic
    seeding is maintained by the caller.
    """

    global _LAST_MONTE_TRIALS

    min_trials = max(base_trials, _MIN_MONTE_TRIALS, min_trials or 0)
    max_trials = max(_MAX_MONTE_TRIALS, min_trials, max_trials or 0)
    target = target_std_error if target_std_error is not None else _TARGET_STD_ERROR
    chunk = max(1, min(_MONTE_CHUNK, max_trials))

    hero_cards = _treys_cards(hero)
    preset_board = _treys_cards(board)

    seen = set(hero) | set(board)
    deck = [c for c in range(52) if c not in seen]

    need = 5 - len(board)
    if need < 0:
        raise ValueError("Board cannot have more than 5 cards")

    villain_fixed = None
    remaining_deck = deck
    if villain is not None:
        villain_fixed = _treys_cards(villain)
        remaining_deck = [c for c in deck if c not in villain]

    wins = 0
    ties = 0
    trials = 0

    def _sample_once() -> None:
        nonlocal wins, ties, trials
        if villain_fixed is None:
            picks = rng.sample(deck, 2 + need)
            villain_raw = picks[:2]
            fill_raw = picks[2:]
            villain_cards = _treys_cards(villain_raw)
        else:
            villain_cards = villain_fixed
            fill_raw = rng.sample(remaining_deck, need) if need else ()

        board_now = preset_board + [_TREYS_CACHE[c] for c in fill_raw]

        hr = _EVALUATOR.evaluate(hero_cards, board_now)
        vr = _EVALUATOR.evaluate(villain_cards, board_now)
        if hr < vr:
            wins += 1
        elif hr == vr:
            ties += 1
        trials += 1

    while trials < max_trials:
        remaining = max_trials - trials
        current_chunk = min(chunk, remaining)
        for _ in range(current_chunk):
            _sample_once()

        equity = (wins + 0.5 * ties) / trials if trials else 0.0
        variance = max(equity * (1 - equity), 0.0)
        std_error = math.sqrt(variance / trials) if trials else float("inf")

        if trials >= min_trials and std_error <= target:
            break

    _LAST_MONTE_TRIALS = trials
    return (wins + 0.5 * ties) / max(1, trials)


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
