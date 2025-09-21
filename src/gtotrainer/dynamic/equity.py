from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from functools import lru_cache
from itertools import combinations
from typing import Final

import eval7
import numpy as np

from .cards import canonicalize_cards, card_int_to_str

__all__ = [
    "estimate_equity",
    "hero_equity_vs_combo",
    "hero_equity_vs_range",
]


class _Eval7MonteCarlo:
    """Vectorised Monte Carlo equity evaluation backed by eval7."""

    def __init__(self) -> None:
        # Cache eval7.Card objects for 0..51 to avoid repeated allocations in hot loops.
        self._card_cache: np.ndarray = np.array([eval7.Card(card_int_to_str(idx)) for idx in range(52)], dtype=object)

    def _cards_from_ints(self, cards: Sequence[int]) -> list[eval7.Card]:
        if not cards:
            return []
        indices = np.fromiter((int(c) for c in cards), dtype=np.int16)
        return self._card_cache[indices].tolist()

    @staticmethod
    def _evaluate(hand: list[eval7.Card]) -> int:
        # eval7.evaluate returns a higher number for stronger hands.
        return eval7.evaluate(hand)

    @staticmethod
    def _sample_unique(
        deck_idx: np.ndarray,
        draws: int,
        trials: int,
        generator: np.random.Generator,
    ) -> np.ndarray:
        if trials <= 0:
            return np.empty((0, draws), dtype=np.int16)
        if draws <= 0:
            return np.empty((trials, 0), dtype=np.int16)
        if draws > deck_idx.size:
            raise ValueError("Cannot draw more unique cards than remain in the deck")
        # Assign random priorities to each card per trial and take the lowest `draws` priorities.
        priorities = generator.random((trials, deck_idx.size))
        order = np.argsort(priorities, axis=1)
        selection = deck_idx[order[:, :draws]]
        return selection.astype(np.int16)

    def run_trials(
        self,
        hero: Sequence[int],
        board: Sequence[int],
        rival: Sequence[int] | None,
        *,
        trials: int,
        rng: random.Random,
    ) -> tuple[int, int, int]:
        if trials <= 0:
            return 0, 0, 0

        hero_cards = self._cards_from_ints(hero)
        board_cards = self._cards_from_ints(board)
        need = 5 - len(board_cards)
        if need < 0:
            raise ValueError("Board cannot have more than 5 cards")

        seen = set(hero) | set(board)
        seed = rng.getrandbits(63)
        generator = np.random.default_rng(seed)

        wins = 0
        ties = 0

        if rival is not None:
            rival_cards = self._cards_from_ints(rival)
            seen.update(rival)
            remaining_idx = np.array([c for c in range(52) if c not in seen], dtype=np.int16)
            if need <= 0 or remaining_idx.size == 0:
                combined_board = board_cards
                hero_rank = self._evaluate(hero_cards + combined_board)
                rival_rank = self._evaluate(rival_cards + combined_board)
                if hero_rank > rival_rank:
                    wins += 1
                elif hero_rank == rival_rank:
                    ties += 1
                return wins, ties, 1

            samples = self._sample_unique(remaining_idx, need, trials, generator)
            for row in samples:
                fill_cards = self._cards_from_ints(row.tolist())
                combined_board = board_cards + fill_cards
                hero_rank = self._evaluate(hero_cards + combined_board)
                rival_rank = self._evaluate(rival_cards + combined_board)
                if hero_rank > rival_rank:
                    wins += 1
                elif hero_rank == rival_rank:
                    ties += 1
            return wins, ties, samples.shape[0]

        deck_idx = np.array([c for c in range(52) if c not in seen], dtype=np.int16)
        draws = need + 2
        if draws <= 0:
            raise ValueError("Monte Carlo requires draws >= 1 when rival is random")
        samples = self._sample_unique(deck_idx, draws, trials, generator)
        for row in samples:
            rival_idx = row[:2].tolist()
            fill_idx = row[2:].tolist()
            rival_cards = self._cards_from_ints(rival_idx)
            combined_board = board_cards + self._cards_from_ints(fill_idx)
            hero_rank = self._evaluate(hero_cards + combined_board)
            rival_rank = self._evaluate(rival_cards + combined_board)
            if hero_rank > rival_rank:
                wins += 1
            elif hero_rank == rival_rank:
                ties += 1
        return wins, ties, samples.shape[0]


_ENGINE: Final[_Eval7MonteCarlo] = _Eval7MonteCarlo()

_MIN_MONTE_TRIALS = 0
_MAX_MONTE_TRIALS = 1000
_MONTE_CHUNK = 150
_TARGET_STD_ERROR = 0.025
_LAST_MONTE_TRIALS = 0


def estimate_equity(
    hero: list[int],
    board: list[int],
    known_rival: list[int] | None,
    rng: random.Random,
    trials: int = 400,
) -> float:
    wins, ties, total = _ENGINE.run_trials(hero, board, known_rival, trials=trials, rng=rng)
    total = max(1, total)
    return (wins + 0.5 * ties) / total


@lru_cache(maxsize=50000)
def _cached_equity(
    hero: tuple[int, ...],
    board: tuple[int, ...],
    rival: tuple[int, ...],
    trials: int,
    target_std_error: float | None,
) -> float:
    if len(board) >= 3:
        return _enumerate_remaining(hero, board, rival)
    seed = hash((hero, board, rival, trials, round(target_std_error or 0.0, 4))) & 0xFFFFFFFF
    rng = random.Random(seed)
    hero_list = list(hero)
    board_list = list(board)
    rival_list = list(rival) if rival else None
    return _adaptive_monte_carlo(
        hero_list,
        board_list,
        rival_list,
        base_trials=trials,
        rng=rng,
        target_std_error=target_std_error,
    )


def _enumerate_remaining(hero: tuple[int, ...], board: tuple[int, ...], rival: tuple[int, ...]) -> float:
    hero_cards = _ENGINE._cards_from_ints(hero)
    board_cards = _ENGINE._cards_from_ints(board)
    rival_cards = _ENGINE._cards_from_ints(rival)

    need = 5 - len(board_cards)
    if need < 0:
        raise ValueError("Board cannot have more than 5 cards")

    if need == 0:
        hero_rank = _ENGINE._evaluate(hero_cards + board_cards)
        rival_rank = _ENGINE._evaluate(rival_cards + board_cards)
        if hero_rank > rival_rank:
            return 1.0
        if hero_rank == rival_rank:
            return 0.5
        return 0.0

    seen = set(hero) | set(board) | set(rival)
    deck = [c for c in range(52) if c not in seen]

    wins = 0
    ties = 0
    total = 0

    for fill in combinations(deck, need):
        fill_cards = _ENGINE._cards_from_ints(fill)
        combined_board = board_cards + fill_cards
        hero_rank = _ENGINE._evaluate(hero_cards + combined_board)
        rival_rank = _ENGINE._evaluate(rival_cards + combined_board)
        total += 1
        if hero_rank > rival_rank:
            wins += 1
        elif hero_rank == rival_rank:
            ties += 1

    return (wins + 0.5 * ties) / total if total else 0.0


def _adaptive_monte_carlo(
    hero: list[int],
    board: list[int],
    rival: list[int] | None,
    *,
    base_trials: int,
    rng: random.Random,
    min_trials: int | None = None,
    max_trials: int | None = None,
    target_std_error: float | None = None,
) -> float:
    """Return a Monte Carlo equity estimate with adaptive precision."""

    global _LAST_MONTE_TRIALS

    min_trials = max(base_trials, _MIN_MONTE_TRIALS, min_trials or 0)
    max_trials = max(_MAX_MONTE_TRIALS, min_trials, max_trials or 0)
    target = target_std_error if target_std_error is not None else _TARGET_STD_ERROR
    chunk = max(1, min(_MONTE_CHUNK, max_trials))

    wins = 0
    ties = 0
    total_trials = 0

    while total_trials < max_trials:
        remaining = max_trials - total_trials
        current_chunk = min(chunk, remaining)
        chunk_wins, chunk_ties, chunk_total = _ENGINE.run_trials(
            hero,
            board,
            rival,
            trials=current_chunk,
            rng=rng,
        )
        if chunk_total == 0:
            break
        wins += chunk_wins
        ties += chunk_ties
        total_trials += chunk_total

        equity = (wins + 0.5 * ties) / total_trials if total_trials else 0.0
        variance = max(equity * (1 - equity), 0.0)
        std_error = math.sqrt(variance / total_trials) if total_trials else float("inf")

        if total_trials >= min_trials and std_error <= target:
            break

    _LAST_MONTE_TRIALS = total_trials
    return (wins + 0.5 * ties) / max(1, total_trials)


def hero_equity_vs_combo(
    hero: list[int],
    board: list[int],
    combo: tuple[int, int],
    trials: int,
    *,
    target_std_error: float | None = None,
) -> float:
    hero_canon, board_canon, rival_canon = canonicalize_cards(hero, board, combo)
    target = target_std_error if target_std_error and target_std_error > 0 else None
    return _cached_equity(hero_canon, board_canon, rival_canon, trials, target)


def hero_equity_vs_range(
    hero: list[int],
    board: list[int],
    combos: Iterable[tuple[int, int]],
    trials: int,
    *,
    target_std_error: float | None = None,
) -> float:
    combos_list = list(combos)
    if not combos_list:
        return 0.0
    total = 0.0
    for combo in combos_list:
        total += hero_equity_vs_combo(hero, board, combo, trials, target_std_error=target_std_error)
    return total / len(combos_list)
