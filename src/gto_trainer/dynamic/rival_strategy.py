"""Villain decision modelling helpers.

This module centralises the logic that decides whether a simulated villain
continues or folds after the hero raises/bets. The previous implementation
made the villain clairvoyant by comparing against the hero's exact hand. We now
bias decisions using the villain's own holding strength and target fold
frequencies inferred from solver-style heuristics.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# The profile dictionary stored on Option.meta uses only standard Python
# container types (lists/dicts) so it can be copied or serialised in tests.


@dataclass(frozen=True)
class VillainDecision:
    """Outcome of a villain response sample."""

    folds: bool


def _combo_strength(combo: Sequence[int]) -> float:
    """Replicate the heuristic ranking used by range_model without importing private helpers."""

    a, b = int(combo[0]), int(combo[1])
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


def _encode_combo(combo: Sequence[int]) -> str:
    a, b = int(combo[0]), int(combo[1])
    if a > b:
        a, b = b, a
    return f"{a}-{b}"


def build_profile(
    sampled_range: Iterable[Sequence[int]],
    *,
    fold_probability: float,
    continue_ratio: float,
) -> dict:
    """Create a lightweight metadata profile for villain response sampling.

    - ``sampled_range`` is the subset of villain combos considered when the
      option was evaluated (typically <= 200 entries).
    - ``fold_probability`` represents the aggregate fold frequency implied by
      the EV calculation.
    - ``continue_ratio`` is the share of holdings that should continue.
    """

    combos = {_encode_combo(combo): [int(combo[0]), int(combo[1])] for combo in sampled_range}
    ranked = sorted(combos.values(), key=_combo_strength, reverse=True)
    total = len(ranked)
    fold_probability = max(0.0, min(1.0, float(fold_probability)))
    continue_ratio = max(0.0, min(1.0, float(continue_ratio)))
    continue_count = min(total, max(0, round(total * continue_ratio)))
    if continue_ratio > 0 and continue_count == 0:
        continue_count = 1

    strengths = [_combo_strength(c) for c in ranked]
    ranks = {_encode_combo(c): idx for idx, c in enumerate(ranked)}

    return {
        "fold_probability": fold_probability,
        "continue_ratio": continue_ratio,
        "total": total,
        "continue_count": continue_count,
        "ranked": ranked,
        "strengths": strengths,
        "ranks": ranks,
    }


def _percentile_for_combo(profile: Mapping[str, object], combo: Sequence[int]) -> float:
    ranked = profile.get("ranked")
    strengths = profile.get("strengths")
    if not ranked or not strengths:
        return 0.5
    ranked_list = list(ranked)  # type: ignore[arg-type]
    strengths_list = list(strengths)  # type: ignore[arg-type]
    total = len(ranked_list) or 1
    key = _encode_combo(combo)
    ranks = profile.get("ranks")
    if isinstance(ranks, Mapping) and key in ranks:
        idx = int(ranks[key])
    else:
        target = _combo_strength(combo)
        idx = total - 1
        for i, strength in enumerate(strengths_list):
            if target >= strength:
                idx = i
                break
    # Convert to percentile where 1.0 -> strongest, 0.0 -> weakest.
    return 1.0 - (idx / max(1, total - 1)) if total > 1 else 1.0


def _sample_profile_combo(profile: Mapping[str, object], rng: random.Random) -> tuple[int, int] | None:
    ranked = profile.get("ranked")
    if not isinstance(ranked, list) or not ranked:
        return None
    combos: list[tuple[int, int]] = []
    for combo in ranked:
        try:
            a, b = int(combo[0]), int(combo[1])  # type: ignore[index]
        except (TypeError, ValueError, IndexError):
            continue
        combos.append((a, b))
    if not combos:
        return None

    continue_ratio = float(profile.get("continue_ratio", 0.0))
    continue_count = int(profile.get("continue_count", 0))
    total = len(combos)

    if continue_count <= 0 or continue_count >= total:
        idx = int(rng.random() * total)
        return combos[idx]

    if rng.random() < continue_ratio:
        idx = int(rng.random() * continue_count)
        return combos[idx]

    tail_size = total - continue_count
    idx = continue_count + int(rng.random() * tail_size)
    return combos[idx]


def decide_action(
    meta: Mapping[str, object] | None,
    villain_cards: Sequence[int] | None,
    rng: random.Random,
) -> VillainDecision:
    """Sample whether the villain folds or continues.

    ``meta`` is the Option.meta mapping. If the stored profile is missing we
    default to always continuing to preserve backwards compatibility.
    """

    if not meta:
        return VillainDecision(folds=False)
    profile = meta.get("villain_profile") if isinstance(meta, Mapping) else None
    if not isinstance(profile, Mapping):
        return VillainDecision(folds=False)

    fold_prob = float(profile.get("fold_probability", 0.0))
    percentile = 0.5
    if villain_cards is not None:
        percentile = _percentile_for_combo(profile, villain_cards)
    else:
        sampled = _sample_profile_combo(profile, rng)
        if sampled is not None:
            percentile = _percentile_for_combo(profile, sampled)

    bias_scale = min(0.6, max(0.2, fold_prob + 0.2))
    fold_prob = max(0.0, min(1.0, fold_prob + (0.5 - percentile) * bias_scale))
    draw = rng.random()
    return VillainDecision(folds=draw < fold_prob)
