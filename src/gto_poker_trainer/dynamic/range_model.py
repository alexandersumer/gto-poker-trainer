"""Villain range modelling helpers.

The heuristics here aim for consistency and transparency rather than perfectly
replicating commercial solvers. We rank every possible two-card holding via a
lightweight strength metric so the ordering is deterministic and inexpensive,
keeping the training loop fast while still producing reasonable ranges.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

from .cards import fresh_deck

# Pre-compute and cache the full deck once; card ints are 0..51.
_DECK = fresh_deck()


def _sorted_combo(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


@lru_cache(maxsize=2000)
def _combo_strength(combo: tuple[int, int]) -> float:
    """Return a deterministic heuristic score for a preflop holding."""

    a, b = combo
    ra, rb = a // 4, b // 4
    suited = (a % 4) == (b % 4)
    high, low = (ra, rb) if ra >= rb else (rb, ra)

    score = high * 10 + low
    if high == low:
        score += 80 + high * 5
    if suited:
        score += 5
    gap = high - low - 1
    if high != low:
        if gap <= 0:
            score += 4
        elif gap == 1:
            score += 3
        elif gap == 2:
            score += 1
        elif gap >= 4:
            score -= gap
    return float(score)


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


def villain_sb_open_range(open_size: float, blocked_cards: Iterable[int] | None = None) -> list[tuple[int, int]]:
    """Simple SB open-raise model by sizing.

    Smaller opens incentivise wider ranges; larger opens tighten up. The
    percentages below align with contemporary HU solver guidance that opens
    ≈85–90% of hands when min-raising and ≈70–80% when using larger 2.5–3.0bb
    sizes at 100bb effective.
    """

    if open_size <= 2.0:
        percent = 0.88
    elif open_size <= 2.3:
        percent = 0.85
    elif open_size <= 2.7:
        percent = 0.78
    else:
        percent = 0.72
    return top_percent(percent, blocked_cards)


def villain_bb_defend_range(open_size: float, blocked_cards: Iterable[int] | None = None) -> list[tuple[int, int]]:
    """Approximate BB defend range versus SB open sizing.

    The thresholds roughly match contemporary HU recommendations where the BB
    continues with ~70–75% versus a 2.0x open and gradually tightens as the SB
    chooses larger sizes. This keep-logic mirrors the open range helper so both
    positions share a consistent strength ordering.
    """

    if open_size <= 2.0:
        percent = 0.74
    elif open_size <= 2.3:
        percent = 0.7
    elif open_size <= 2.7:
        percent = 0.62
    else:
        percent = 0.56
    return top_percent(percent, blocked_cards)


def tighten_range(combos: Iterable[tuple[int, int]], fraction: float) -> list[tuple[int, int]]:
    """Return the strongest subset of an existing range."""

    combos_list = list(combos)
    combos_list.sort(key=_combo_strength, reverse=True)
    count = max(1, int(round(len(combos_list) * max(0.0, min(1.0, fraction)))))
    return combos_list[:count]


def combos_without_blockers(blocked_cards: Iterable[int] | None = None) -> list[tuple[int, int]]:
    return _filter_blocked(_all_combos_sorted(), set(blocked_cards or []))
