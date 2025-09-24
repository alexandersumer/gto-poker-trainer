"""Rival decision modelling helpers.

This module centralises the logic that decides whether a simulated rival
continues or folds after the hero raises/bets. The previous implementation
made the rival clairvoyant by comparing against the hero's exact hand. We now
bias decisions using the rival's own holding strength and target fold
frequencies inferred from solver-style heuristics.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# The profile dictionary stored on Option.meta uses only standard Python
# container types (lists/dicts) so it can be copied or serialised in tests.
from .hand_strength import combo_playability_score


@dataclass(frozen=True)
class RivalDecision:
    """Outcome of a rival response sample."""

    folds: bool


@dataclass(frozen=True)
class PersonaTuning:
    name: str
    fold_bias: float = 0.0
    threshold_delta: float = 0.0
    strength_scale: float = 1.0
    aggression_scale: float = 1.0
    call_bias: float = 0.0
    noise_scale: float = 0.6


def _combo_strength(combo: Sequence[int]) -> float:
    """Replicate the heuristic ranking used by range_model without importing private helpers."""

    a, b = int(combo[0]), int(combo[1])
    return combo_playability_score((a, b))


def _encode_combo(combo: Sequence[int]) -> str:
    a, b = int(combo[0]), int(combo[1])
    if a > b:
        a, b = b, a
    return f"{a}-{b}"


def _board_draw_intensity(cards: Sequence[int] | None) -> float:
    if not cards:
        return 0.5
    ranks = sorted(card // 4 for card in cards)
    suits = [card % 4 for card in cards]
    unique_ranks = len(set(ranks))
    span = ranks[-1] - ranks[0] if ranks else 0
    suit_counts: dict[int, int] = {}
    for suit in suits:
        suit_counts[suit] = suit_counts.get(suit, 0) + 1
    max_suit = max(suit_counts.values(), default=0)
    flush_factor = 0.0 if len(cards) < 3 else max(0.0, (max_suit - 1) / (len(cards) - 1))
    paired_factor = 1.0 if unique_ranks < len(ranks) else 0.0
    straight_factor = max(0.0, 1.0 - min(4, span) / 4)
    texture = 0.25 + 0.45 * flush_factor + 0.2 * straight_factor + 0.1 * paired_factor
    return max(0.0, min(1.0, texture))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bet_size_ratio(meta: Mapping[str, object]) -> float:
    pot = _safe_float(
        meta.get("pot_before")
        or meta.get("pot_before_action")
        or meta.get("pot_before_bet")
        or meta.get("pot_before_raise"),
        default=0.0,
    )
    risk_candidates = (
        meta.get("bet"),
        meta.get("hero_add"),
        meta.get("risk"),
        meta.get("call_cost"),
    )
    risk = 0.0
    for candidate in risk_candidates:
        risk = max(risk, _safe_float(candidate))
    if risk <= 0.0 and "raise_to" in meta:
        risk = _safe_float(meta["raise_to"]) - _safe_float(meta.get("pot_before"))
        if risk <= 0.0:
            risk = _safe_float(meta["raise_to"])
    if pot <= 0.0:
        pot = 1.0
    return max(0.0, min(2.5, risk / pot))


PERSONA_LIBRARY: dict[str, PersonaTuning] = {
    "balanced": PersonaTuning("balanced"),
    "aggressive": PersonaTuning(
        name="aggressive",
        fold_bias=-0.18,
        threshold_delta=-0.07,
        strength_scale=1.18,
        aggression_scale=1.25,
        call_bias=0.07,
        noise_scale=0.5,
    ),
    "passive": PersonaTuning(
        name="passive",
        fold_bias=0.2,
        threshold_delta=0.1,
        strength_scale=0.8,
        aggression_scale=0.68,
        call_bias=-0.12,
        noise_scale=0.78,
    ),
}


def _blend_probability(base: float, adjusted: float, weight: float) -> float:
    mix = max(0.0, min(1.0, weight))
    return (1.0 - mix) * base + mix * adjusted


def _persona_for_meta(meta: Mapping[str, object] | None) -> PersonaTuning:
    if not isinstance(meta, Mapping):
        return PERSONA_LIBRARY["balanced"]
    style = str(meta.get("rival_style") or meta.get("style") or "balanced").strip().lower()
    return PERSONA_LIBRARY.get(style, PERSONA_LIBRARY["balanced"])


def _calibrated_fold_probability(
    base: float,
    *,
    strength_norm: float | None,
    threshold_norm: float,
    texture: float,
    size_ratio: float,
    adapt_scale: float,
    continue_ratio: float,
) -> float:
    clamped = max(1e-4, min(1.0 - 1e-4, base))
    logit = math.log(clamped / (1.0 - clamped))

    texture_adj = (texture - 0.5) * 0.9
    size_deviation = size_ratio - 0.85
    size_scale = 1.2 + 0.3 * (1.0 - continue_ratio)
    size_adj = math.tanh(size_deviation * 1.35) * size_scale
    logit += size_adj - texture_adj

    if strength_norm is not None:
        delta = strength_norm - threshold_norm
        precision = 1.8 + 0.5 * (1.0 - continue_ratio)
        logit -= delta * precision

    if adapt_scale:
        logit -= 1.5 * adapt_scale

    logit = max(-12.0, min(12.0, logit))
    adjusted = 1.0 / (1.0 + math.exp(-logit))
    shift = adjusted - clamped
    size_pressure = max(-0.75, min(1.75, size_ratio - 0.7))
    max_shift = 0.12 + 0.12 * (1.0 - continue_ratio)
    if size_pressure > 0:
        max_shift += 0.14 * min(size_pressure, 1.0)
    else:
        max_shift *= 1.0 + 0.5 * size_pressure
    max_shift = max(0.05, min(0.5, max_shift))
    adjusted = clamped + max(-max_shift, min(max_shift, shift))
    adjusted = max(1e-4, min(1.0 - 1e-4, adjusted))
    # Blend back towards the base value to keep aggregate frequencies stable.
    blend_weight = 0.45 + 0.25 * (1.0 - continue_ratio)
    if size_ratio > 1.0:
        blend_weight += 0.12 * min(size_ratio - 1.0, 1.0)
    blend_weight = max(0.25, min(0.95, blend_weight))
    return _blend_probability(clamped, adjusted, blend_weight)


def build_profile(
    sampled_range: Iterable[Sequence[int]],
    *,
    fold_probability: float,
    continue_ratio: float,
    strengths: Iterable[tuple[tuple[int, int], float]] | None = None,
    weights: Iterable[tuple[tuple[int, int], float]] | Mapping[str, float] | None = None,
) -> dict:
    """Create a lightweight metadata profile for rival response sampling.

    - ``sampled_range`` is the subset of rival combos considered when the
      option was evaluated (typically <= 200 entries).
    - ``fold_probability`` represents the aggregate fold frequency implied by
      the EV calculation.
    - ``continue_ratio`` is the share of holdings that should continue.
    """

    strength_lookup: dict[str, float] = {}
    if strengths:
        for combo_pair, score in strengths:
            key = _encode_combo(combo_pair)
            strength_lookup[key] = float(score)

    weight_lookup: dict[str, float] = {}
    if weights:
        if isinstance(weights, Mapping):
            for key, value in weights.items():
                if isinstance(key, str):
                    weight_lookup[key] = max(0.0, float(value))
                else:
                    try:
                        a, b = int(key[0]), int(key[1])
                    except (TypeError, ValueError, IndexError):
                        continue
                    weight_lookup[_encode_combo((a, b))] = max(0.0, float(value))
        else:
            for combo_pair, value in weights:
                key = _encode_combo(combo_pair)
                weight_lookup[key] = max(0.0, float(value))

    scored: list[tuple[float, list[int]]] = []
    for combo in sampled_range:
        a, b = int(combo[0]), int(combo[1])
        if a > b:
            a, b = b, a
        ordered = [a, b]
        score = strength_lookup.get(_encode_combo(ordered), _combo_strength(ordered))
        scored.append((score, ordered))

    sorted_scored = sorted(scored, key=lambda item: item[0], reverse=True)
    ranked: list[list[int]] = [entry for _, entry in sorted_scored]
    strengths_sorted: list[float] = [weight for weight, _ in sorted_scored]
    total = len(ranked)
    fold_probability = max(0.0, min(1.0, float(fold_probability)))
    continue_ratio = max(0.0, min(1.0, float(continue_ratio)))
    continue_count = min(total, max(0, round(total * continue_ratio)))
    if continue_ratio > 0 and continue_count == 0:
        continue_count = 1

    ranks = {_encode_combo(c): idx for idx, c in enumerate(ranked)}
    min_strength = min(strengths_sorted) if strengths_sorted else 0.0
    max_strength = max(strengths_sorted) if strengths_sorted else 1.0
    threshold_strength = strengths_sorted[continue_count - 1] if continue_count > 0 else min_strength
    temperature = max(0.05, 0.2 * (1.0 - continue_ratio))

    # Prepare deterministic noise seed contribution: sorted key list hashed as float.
    hash_seed = float(total)

    base_weights: list[float] = []
    for idx, combo in enumerate(ranked):
        key = _encode_combo(combo)
        if key in weight_lookup:
            base_weights.append(weight_lookup[key])
            continue
        score = strengths_sorted[idx]
        scaled = (score - threshold_strength) / max(temperature, 1e-6)
        # Clamp exponent to avoid overflow while preserving ordering.
        scaled = max(-40.0, min(40.0, scaled))
        base_weights.append(math.exp(scaled))

    positive_total = sum(value for value in base_weights if value > 0)
    if positive_total <= 0 and continue_count > 0:
        base_weights = [1.0 if idx < continue_count else 0.0 for idx in range(total)]
        positive_total = float(continue_count)

    continue_weights: list[list[float | int]] = []
    if continue_ratio > 0 and positive_total > 0:
        # Convert to conditional distribution among continuing combos.
        conditional_scale = sum(base_weights)
        if conditional_scale <= 0:
            conditional_scale = positive_total
        for combo, raw_weight in zip(ranked, base_weights, strict=False):
            if raw_weight <= 0:
                continue
            continue_weights.append([int(combo[0]), int(combo[1]), float(raw_weight / conditional_scale)])

    return {
        "fold_probability": fold_probability,
        "continue_ratio": continue_ratio,
        "total": total,
        "continue_count": continue_count,
        "ranked": ranked,
        "strengths": strengths_sorted,
        "ranks": ranks,
        "strength_bounds": (min_strength, max_strength),
        "threshold_strength": threshold_strength,
        "temperature": temperature,
        "noise_seed": hash_seed,
        "continue_weights": continue_weights,
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


def _strength_for_combo(profile: Mapping[str, object], combo: Sequence[int]) -> float:
    ranked = profile.get("ranked")
    strengths = profile.get("strengths")
    if not ranked or not strengths:
        return _combo_strength(combo)
    ranks = profile.get("ranks")
    key = _encode_combo(combo)
    if isinstance(ranks, Mapping) and key in ranks:
        idx = int(ranks[key])
        try:
            return float(strengths[idx])  # type: ignore[index]
        except (IndexError, TypeError, ValueError):
            return _combo_strength(combo)
    return _combo_strength(combo)


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
        weighted_entries = profile.get("continue_weights")
        distribution: list[tuple[tuple[int, int], float]] = []
        if isinstance(weighted_entries, list):
            for entry in weighted_entries:
                try:
                    card_a = int(entry[0])
                    card_b = int(entry[1])
                    weight = float(entry[2])
                except (TypeError, ValueError, IndexError):
                    continue
                if weight <= 0:
                    continue
                distribution.append(((card_a, card_b), weight))

        if distribution:
            total_weight = sum(weight for _, weight in distribution)
            if total_weight > 0:
                draw = rng.random() * total_weight
                cumulative = 0.0
                for combo, weight in distribution:
                    cumulative += weight
                    if draw <= cumulative:
                        return combo

        idx = int(rng.random() * continue_count)
        return combos[idx]

    tail_size = total - continue_count
    idx = continue_count + int(rng.random() * tail_size)
    return combos[idx]


def decide_action(
    meta: Mapping[str, object] | None,
    rival_cards: Sequence[int] | None,
    rng: random.Random,
) -> RivalDecision:
    """Sample whether the rival folds or continues.

    ``meta`` is the Option.meta mapping. If the stored profile is missing we
    default to always continuing to preserve backwards compatibility.
    """

    if not meta:
        return RivalDecision(folds=False)
    profile = meta.get("rival_profile") if isinstance(meta, Mapping) else None
    if not isinstance(profile, Mapping):
        return RivalDecision(folds=False)

    fold_prob = float(profile.get("fold_probability", 0.0))
    persona = _persona_for_meta(meta)
    fold_prob += persona.fold_bias
    threshold_strength = float(profile.get("threshold_strength", 0.0))
    bounds = profile.get("strength_bounds", (0.0, 1.0))
    if isinstance(bounds, Sequence) and len(bounds) == 2:
        min_strength = float(bounds[0])
        max_strength = float(bounds[1])
    else:
        min_strength = 0.0
        max_strength = 1.0
    spread = max(1e-6, max_strength - min_strength)
    continue_ratio = float(profile.get("continue_ratio", 0.0))
    temperature = float(profile.get("temperature", 0.12))
    board_meta = meta.get("board_cards") if isinstance(meta, Mapping) else None
    board_cards: Sequence[int] | None
    if isinstance(board_meta, (list, tuple)):
        board_cards = tuple(int(c) for c in board_meta)
    else:
        board_cards = None
    texture = _board_draw_intensity(board_cards)
    temperature = max(0.035, temperature * (0.75 + 0.4 * texture))
    noise = min(0.12, max(0.0, 0.12 * (1.0 - continue_ratio) + 0.05 * texture))
    size_ratio = _bet_size_ratio(meta)
    fold_prob -= 0.14 * (texture - 0.5)
    fold_prob += 0.1 * (size_ratio - 0.7)
    fold_prob -= persona.call_bias

    adapt = meta.get("rival_adapt") if isinstance(meta, Mapping) else None
    adapt_scale = 0.0
    if isinstance(adapt, Mapping):
        try:
            observed_aggr = float(adapt.get("aggr", 0.0))
        except (TypeError, ValueError):
            observed_aggr = 0.0
        try:
            observed_passive = float(adapt.get("passive", 0.0))
        except (TypeError, ValueError):
            observed_passive = 0.0
        deviation = math.log((observed_aggr + 1.0) / (observed_passive + 1.0))
        sample_total = observed_aggr + observed_passive
        sample_weight = min(1.0, sample_total / 5.0)
        adapt_scale = max(-0.25, min(0.25, 0.09 * deviation * sample_weight))

    strength = None
    strength_norm = None
    threshold_norm = (threshold_strength - min_strength) / spread if spread > 0 else 0.5
    threshold_norm = max(0.0, min(1.0, threshold_norm + persona.threshold_delta))

    if rival_cards is not None:
        strength = _strength_for_combo(profile, rival_cards)
    else:
        sampled = _sample_profile_combo(profile, rng)
        if sampled is not None:
            strength = _strength_for_combo(profile, sampled)

    if strength is not None:
        strength_norm = (strength - min_strength) / spread if spread > 0 else 0.5
        delta = (strength_norm - threshold_norm) * persona.strength_scale
        bias_scale = min(0.45, max(0.18, (1.0 - fold_prob) * 0.5 + 0.18)) * persona.aggression_scale
        slope = max(0.02, temperature)
        shift = math.tanh(delta / slope)
        fold_prob -= shift * bias_scale

    if adapt_scale:
        fold_prob -= 0.6 * adapt_scale * persona.aggression_scale

    fold_prob = _calibrated_fold_probability(
        fold_prob,
        strength_norm=strength_norm,
        threshold_norm=threshold_norm,
        texture=texture,
        size_ratio=size_ratio,
        adapt_scale=adapt_scale * persona.aggression_scale,
        continue_ratio=continue_ratio,
    )

    noise *= persona.noise_scale

    if noise > 0:
        fold_prob += (rng.random() - 0.5) * 2.0 * noise

    fold_prob = max(0.0, min(1.0, fold_prob))
    draw = rng.random()
    return RivalDecision(folds=draw < fold_prob)
