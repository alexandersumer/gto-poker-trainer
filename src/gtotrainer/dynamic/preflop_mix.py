"""Preflop action frequency heuristics inspired by solver outputs.

This module provides light-weight approximations of solver-calibrated
frequencies for big blind defence against small blind opens in heads-up play.
The goal is to keep the trainer's behaviour directionally aligned with
professional-grade charts without requiring bundled proprietary solves.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache

from .cards import fresh_deck
from .hand_strength import combo_playability_score

# Heads-up uses the same ranking heuristic as range_model for determinism.


@lru_cache(maxsize=1)
def _sorted_combos() -> list[tuple[int, int]]:
    deck = fresh_deck()
    combos: list[tuple[int, int]] = []
    for i in range(len(deck)):
        for j in range(i + 1, len(deck)):
            combos.append((deck[i], deck[j]))
    combos.sort(key=combo_playability_score, reverse=True)
    return combos


def _combos_without_blockers(blocked: Iterable[int]) -> list[tuple[int, int]]:
    blocked_set = set(blocked)
    return [c for c in _sorted_combos() if c[0] not in blocked_set and c[1] not in blocked_set]


def _percentile(combo: tuple[int, int], blocked: Iterable[int]) -> float:
    combos = _combos_without_blockers(blocked)
    total = len(combos)
    for idx, candidate in enumerate(combos):
        if candidate == combo or candidate == (combo[1], combo[0]):
            return 1.0 - (idx / max(1, total - 1)) if total > 1 else 1.0
    # If the combo is blocked, fall back to average percentile.
    return 0.5


@dataclass(frozen=True)
class DefenseProfile:
    defend: float  # total continue share (call + 3bet + jam)
    threebet: float  # total 3bet (excluding jams)
    jam: float  # shove share (rare at 100bb but non-zero for top combos)
    marginal_band: float  # width of call/fold smoothing band near bottom
    threebet_smooth: float  # portion of the 3bet band used for call/3bet mixing


_PROFILE_ANCHORS: list[tuple[float, DefenseProfile]] = [
    (
        2.0,
        DefenseProfile(
            defend=0.72,
            threebet=0.13,
            jam=0.012,
            marginal_band=0.08,
            threebet_smooth=0.042,
        ),
    ),
    (
        2.3,
        DefenseProfile(
            defend=0.68,
            threebet=0.13,
            jam=0.012,
            marginal_band=0.15,
            threebet_smooth=0.038,
        ),
    ),
    (
        2.5,
        DefenseProfile(
            defend=0.7,
            threebet=0.13,
            jam=0.012,
            marginal_band=0.2,
            threebet_smooth=0.033,
        ),
    ),
    (
        2.8,
        DefenseProfile(
            defend=0.48,
            threebet=0.115,
            jam=0.015,
            marginal_band=0.058,
            threebet_smooth=0.028,
        ),
    ),
    (
        3.2,
        DefenseProfile(
            defend=0.34,
            threebet=0.17,
            jam=0.015,
            marginal_band=0.05,
            threebet_smooth=0.015,
        ),
    ),
]


def _blend_profiles(low: DefenseProfile, high: DefenseProfile, t: float) -> DefenseProfile:
    inv = 1.0 - t
    return DefenseProfile(
        defend=low.defend * inv + high.defend * t,
        threebet=low.threebet * inv + high.threebet * t,
        jam=low.jam * inv + high.jam * t,
        marginal_band=low.marginal_band * inv + high.marginal_band * t,
        threebet_smooth=low.threebet_smooth * inv + high.threebet_smooth * t,
    )


def _profile_for_open(open_size: float) -> DefenseProfile:
    if open_size <= _PROFILE_ANCHORS[0][0]:
        return _PROFILE_ANCHORS[0][1]
    for (lo_x, lo_prof), (hi_x, hi_prof) in zip(
        _PROFILE_ANCHORS,
        _PROFILE_ANCHORS[1:],
        strict=False,
    ):
        if open_size <= hi_x:
            span = hi_x - lo_x
            t = 0.0 if span <= 0 else (open_size - lo_x) / span
            return _blend_profiles(lo_prof, hi_prof, t)
    return _PROFILE_ANCHORS[-1][1]


def action_mix_for_combo(
    combo: tuple[int, int],
    *,
    open_size: float,
    blocked: Iterable[int] | None = None,
) -> Mapping[str, float]:
    """Return recommended fold/call/3bet/jam frequencies for the combo.

    The distribution is an approximation of solver behaviour:
    - Weakest hands fold entirely.
    - Marginal holdings mix folds and calls.
    - Strong holdings call; premium hands mix between 3-bet and jam.
    """

    blocked_cards = blocked or []
    percentile = _percentile(combo, blocked_cards)
    profile = _profile_for_open(open_size)

    fold_cut = max(0.0, 1.0 - profile.defend)
    marginal_end = min(1.0, fold_cut + profile.marginal_band)

    if percentile <= fold_cut:
        return {"fold": 1.0}

    if percentile <= marginal_end:
        band = max(profile.marginal_band, 1e-6)
        progress = (percentile - fold_cut) / band
        call_freq = max(0.0, min(1.0, progress))
        return {"fold": 1.0 - call_freq, "call": call_freq}

    jam_start = max(marginal_end, 1.0 - profile.jam) if profile.jam > 0 else 1.0
    threebet_start = max(marginal_end, jam_start - profile.threebet)

    if percentile >= jam_start:
        if profile.jam <= 0:
            return {"threebet": 1.0}
        span = max(1e-6, 1.0 - jam_start)
        weight = (percentile - jam_start) / span
        jam_freq = 0.55 + 0.45 * weight
        return {"jam": jam_freq, "threebet": 1.0 - jam_freq}

    if percentile >= threebet_start:
        threebet_span = max(profile.threebet, 1e-6)
        smooth = min(profile.threebet_smooth, threebet_span)
        if percentile <= threebet_start + smooth:
            local = (percentile - threebet_start) / max(smooth, 1e-6)
            threebet_freq = 0.45 + 0.45 * local
            return {"threebet": threebet_freq, "call": 1.0 - threebet_freq}
        return {"threebet": 0.92, "call": 0.08}

    return {"call": 1.0}


def normalise_mix(mix: Mapping[str, float]) -> Mapping[str, float]:
    total = sum(mix.values())
    if total <= 0:
        return mix
    return {k: v / total for k, v in mix.items()}
