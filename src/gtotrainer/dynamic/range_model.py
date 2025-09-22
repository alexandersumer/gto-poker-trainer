"""Rival range modelling helpers.

The heuristics here aim for consistency and transparency rather than perfectly
replicating commercial solvers. We rank every possible two-card holding via a
lightweight strength metric so the ordering is deterministic and inexpensive,
keeping the training loop fast while still producing reasonable ranges.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

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


@dataclass(frozen=True)
class RangeProfile:
    percent: float


@dataclass(frozen=True)
class RangeTable:
    sb_open: list[tuple[float, RangeProfile]]
    bb_defend: list[tuple[float, RangeProfile]]


_SB_OPEN_PROFILES: list[tuple[float, RangeProfile]] = [
    (2.0, RangeProfile(percent=0.9)),
    (2.2, RangeProfile(percent=0.87)),
    (2.5, RangeProfile(percent=0.82)),
    (2.8, RangeProfile(percent=0.75)),
    (3.2, RangeProfile(percent=0.68)),
]


_BB_DEFEND_PROFILES: list[tuple[float, RangeProfile]] = [
    (2.0, RangeProfile(percent=0.66)),
    (2.3, RangeProfile(percent=0.58)),
    (2.5, RangeProfile(percent=0.54)),
    (2.8, RangeProfile(percent=0.45)),
    (3.2, RangeProfile(percent=0.36)),
]


_DEFAULT_TABLE = RangeTable(sb_open=_SB_OPEN_PROFILES, bb_defend=_BB_DEFEND_PROFILES)


def _parse_entries(
    entries: Iterable[dict[str, float]] | None,
    fallback: list[tuple[float, RangeProfile]],
) -> list[tuple[float, RangeProfile]]:
    parsed: list[tuple[float, RangeProfile]] = []
    if entries:
        for item in entries:
            try:
                size = float(item["size"])
                percent = float(item["percent"])
            except (KeyError, TypeError, ValueError):
                continue
            parsed.append((size, RangeProfile(percent=percent)))
    if not parsed:
        return fallback
    parsed.sort(key=lambda pair: pair[0])
    return parsed


@lru_cache(maxsize=1)
def _load_range_tables() -> dict[str, RangeTable]:
    try:
        config_path = resources.files("gtotrainer.data").joinpath("ranges", "config.json")
        raw_text = config_path.read_text(encoding="utf-8")
        loaded = json.loads(raw_text)
    except Exception:
        return {"default": _DEFAULT_TABLE}

    default_raw = loaded.get("default", {})
    default_table = RangeTable(
        sb_open=_parse_entries(default_raw.get("sb_open"), _SB_OPEN_PROFILES),
        bb_defend=_parse_entries(default_raw.get("bb_defend"), _BB_DEFEND_PROFILES),
    )
    tables: dict[str, RangeTable] = {"default": default_table}
    stacks = loaded.get("stacks", {})
    if isinstance(stacks, dict):
        for key, value in stacks.items():
            if not isinstance(value, dict):
                continue
            table = RangeTable(
                sb_open=_parse_entries(value.get("sb_open"), default_table.sb_open),
                bb_defend=_parse_entries(value.get("bb_defend"), default_table.bb_defend),
            )
            tables[str(key)] = table
    return tables


def _table_for_stack(stack_depth: float | None) -> RangeTable:
    tables = _load_range_tables()
    if stack_depth is None or not math.isfinite(stack_depth):
        return tables["default"]
    key = str(int(round(stack_depth)))
    return tables.get(key, tables["default"])


def _interpolate_profile(value: float, profiles: list[tuple[float, RangeProfile]]) -> RangeProfile:
    if not profiles:
        return RangeProfile(percent=0.5)
    if value <= profiles[0][0]:
        return profiles[0][1]
    for (lo_x, lo_prof), (hi_x, hi_prof) in zip(
        profiles,
        profiles[1:],
        strict=False,
    ):
        if value <= hi_x:
            span = hi_x - lo_x
            if span <= 0:
                return hi_prof
            t = (value - lo_x) / span
            percent = lo_prof.percent * (1 - t) + hi_prof.percent * t
            return RangeProfile(percent=percent)
    return profiles[-1][1]


def rival_sb_open_range(
    open_size: float,
    blocked_cards: Iterable[int] | None = None,
    *,
    stack_depth: float | None = None,
) -> list[tuple[int, int]]:
    """Solver-aligned SB open-raise model by sizing."""

    table = _table_for_stack(stack_depth)
    profile = _interpolate_profile(open_size, table.sb_open)
    return top_percent(profile.percent, blocked_cards)


def rival_bb_defend_range(
    open_size: float,
    blocked_cards: Iterable[int] | None = None,
    *,
    stack_depth: float | None = None,
) -> list[tuple[int, int]]:
    """Solver-aligned BB defend range versus SB open sizing."""

    table = _table_for_stack(stack_depth)
    profile = _interpolate_profile(open_size, table.bb_defend)
    return top_percent(profile.percent, blocked_cards)


def tighten_range(combos: Iterable[tuple[int, int]], fraction: float) -> list[tuple[int, int]]:
    """Return the strongest subset of an existing range."""

    combos_list = list(combos)
    combos_list.sort(key=_combo_strength, reverse=True)
    count = max(1, int(round(len(combos_list) * max(0.0, min(1.0, fraction)))))
    return combos_list[:count]


def combos_without_blockers(blocked_cards: Iterable[int] | None = None) -> list[tuple[int, int]]:
    return _filter_blocked(_all_combos_sorted(), set(blocked_cards or []))
