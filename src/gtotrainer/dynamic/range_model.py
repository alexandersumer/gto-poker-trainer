"""Rival range modelling helpers.

The heuristics here aim for consistency and transparency rather than perfectly
replicating commercial solvers. We rank every possible two-card holding via a
lightweight strength metric so the ordering is deterministic and inexpensive,
keeping the training loop fast while still producing reasonable ranges.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

from ..data.range_loader import get_repository
from .cards import fresh_deck
from .hand_strength import combo_playability_score

# Pre-compute and cache the full deck once; card ints are 0..51.
_DECK = fresh_deck()


def _sorted_combo(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


@lru_cache(maxsize=2000)
def _combo_strength(combo: tuple[int, int]) -> float:
    """Return a deterministic heuristic score for a preflop holding."""

    return combo_playability_score(combo)


@lru_cache(maxsize=1)
def _all_combos_sorted() -> list[tuple[int, int]]:
    combos: list[tuple[int, int]] = []
    for i in range(len(_DECK)):
        for j in range(i + 1, len(_DECK)):
            combo = _sorted_combo(_DECK[i], _DECK[j])
            combos.append(combo)
    combos.sort(key=_combo_strength, reverse=True)
    return combos


def _filter_blocked(combos: Iterable[tuple[int, int]], blocked: set[int]) -> list[tuple[int, int]]:
    return [c for c in combos if c[0] not in blocked and c[1] not in blocked]


def top_percent(percent: float, blocked_cards: Iterable[int] | None = None) -> list[tuple[int, int]]:
    """Return the top `percent` of combos excluding any blocked cards."""

    blocked = set(blocked_cards or [])
    all_combos = _filter_blocked(_all_combos_sorted(), blocked)
    count = max(1, int(round(len(all_combos) * max(0.0, min(1.0, percent)))))
    return all_combos[:count]


def rival_sb_open_range(open_size: float, blocked_cards: Iterable[int] | None = None) -> list[tuple[int, int]]:
    """Solver-aligned SB open-raise model by sizing."""

    combos, _ = _range_with_weights("sb_open", open_size, blocked_cards)
    if combos:
        return combos
    profile = RangeProfile(percent=0.8)
    return top_percent(profile.percent, blocked_cards)


def rival_bb_defend_range(open_size: float, blocked_cards: Iterable[int] | None = None) -> list[tuple[int, int]]:
    """Solver-aligned BB defend range versus SB open sizing."""

    combos, _ = _range_with_weights("bb_defend", open_size, blocked_cards)
    if combos:
        return combos
    profile = RangeProfile(percent=0.5)
    return top_percent(profile.percent, blocked_cards)


def tighten_range(combos: Iterable[tuple[int, int]], fraction: float) -> list[tuple[int, int]]:
    """Return the strongest subset of an existing range."""

    combos_list = list(combos)
    combos_list.sort(key=_combo_strength, reverse=True)
    count = max(1, int(round(len(combos_list) * max(0.0, min(1.0, fraction)))))
    return combos_list[:count]


def combos_without_blockers(blocked_cards: Iterable[int] | None = None) -> list[tuple[int, int]]:
    return _filter_blocked(_all_combos_sorted(), set(blocked_cards or []))


@dataclass(frozen=True)
class RangeProfile:
    percent: float


def _range_with_weights(
    range_id: str,
    open_size: float,
    blocked_cards: Iterable[int] | None,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], float] | None]:
    repo = get_repository()
    combos, weights = repo.range_for(range_id, open_size, blocked_cards)
    return combos, weights


def load_range_with_weights(
    range_id: str,
    open_size: float,
    blocked_cards: Iterable[int] | None = None,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], float] | None]:
    """Expose weighted range data for downstream strategy modules."""

    return _range_with_weights(range_id, open_size, blocked_cards)
