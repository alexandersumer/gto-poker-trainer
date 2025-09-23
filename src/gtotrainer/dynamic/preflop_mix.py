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

from ..core.models import Option
from .cards import fresh_deck
from .cfr import LocalCFRBackend, LocalCFRConfig
from .equity import hero_equity_vs_combo
from .hand_strength import combo_playability_score
from .range_model import rival_sb_open_range

# Heads-up uses the same ranking heuristic as range_model for determinism.

_PRELOP_STACK = 100.0
_SB_CONTRIBUTION = 0.5
_BB_CONTRIBUTION = 1.0
_PRELOP_MC_TRIALS = 150
_PRELOP_SAMPLE_LIMIT = 80
_PRELOP_SOLVER = LocalCFRBackend(LocalCFRConfig(iterations=160, extra_iterations_per_action=80))


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


def _villain_open_range(open_size: float, blocked_cards: Iterable[int]) -> list[tuple[int, int]]:
    return rival_sb_open_range(open_size, blocked_cards)


def _fold_continue_stats(
    hero_equities: Iterable[tuple[float, float]] | Iterable[float],
    rival_threshold: float,
) -> tuple[float, float, float]:
    fold_weight = 0.0
    continue_weight = 0.0
    continue_eq = 0.0
    total_weight = 0.0
    entries = list(hero_equities)
    if not entries:
        return 0.0, 0.0, 0.0
    for entry in entries:
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            eq = float(entry[0])
            weight = max(0.0, float(entry[1]))
        else:
            eq = float(entry)
            weight = 1.0
        if weight <= 0:
            continue
        total_weight += weight
        if 1.0 - eq < rival_threshold:
            fold_weight += weight
        else:
            continue_weight += weight
            continue_eq += eq * weight
    if total_weight <= 0:
        return 0.0, 0.0, 0.0
    fe = fold_weight / total_weight
    continue_ratio = continue_weight / total_weight
    avg_eq = (continue_eq / continue_weight) if continue_weight > 0 else 0.0
    return fe, avg_eq, continue_ratio


def _equity_profiles(
    hero_combo: tuple[int, int],
    villain_combos: Iterable[tuple[int, int]],
    trials: int,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], list[tuple[float, float]], float]:
    combos = list(villain_combos)
    equities: dict[tuple[int, int], float] = {}
    errors: dict[tuple[int, int], float] = {}
    if not combos:
        return equities, errors, [], 0.0
    hero_cards = [int(card) for card in hero_combo]
    weight = 1.0 / len(combos)
    avg_eq = 0.0
    pairs: list[tuple[float, float]] = []
    for combo in combos:
        equity = hero_equity_vs_combo(hero_cards, [], combo, trials)
        equities[combo] = equity
        errors[combo] = 0.0
        pairs.append((equity, weight))
        avg_eq += equity * weight
    return equities, errors, pairs, avg_eq


def _weighted_error(
    errors: Mapping[tuple[int, int], float],
    weights: Mapping[tuple[int, int], float],
) -> float:
    total = 0.0
    weighted_sum = 0.0
    for combo, error in errors.items():
        weight = float(weights.get(combo, 0.0))
        if weight <= 0:
            continue
        total += weight
        weighted_sum += error * weight
    if total <= 0:
        return 0.0
    return weighted_sum / total


def _continue_error(
    equities: Mapping[tuple[int, int], float],
    errors: Mapping[tuple[int, int], float],
    weights: Mapping[tuple[int, int], float],
    threshold: float,
) -> float:
    total = 0.0
    weighted_sum = 0.0
    for combo, eq in equities.items():
        weight = float(weights.get(combo, 0.0))
        if weight <= 0:
            continue
        if 1.0 - eq < threshold:
            continue
        total += weight
        weighted_sum += errors.get(combo, 0.0) * weight
    if total <= 0:
        return 0.0
    return weighted_sum / total


def _weights_for_combos(
    combos: Iterable[tuple[int, int]],
    blocked: Iterable[int],
) -> dict[tuple[int, int], float]:
    combo_list = [tuple(int(c) for c in combo) for combo in combos]
    if not combo_list:
        return {}
    blocked_set = set(blocked)
    filtered = [combo for combo in combo_list if combo[0] not in blocked_set and combo[1] not in blocked_set]
    target = filtered if filtered else combo_list
    weight = 1.0 / len(target)
    return {combo: weight for combo in target}


def _sample_combos(
    combos: Iterable[tuple[int, int]],
    limit: int,
) -> list[tuple[int, int]]:
    combo_list = list(combos)
    if limit <= 0 or len(combo_list) <= limit:
        return combo_list

    buckets: dict[str, list[tuple[float, tuple[int, int]]]] = {
        "pair": [],
        "suited": [],
        "offsuit": [],
    }
    for combo in combo_list:
        score = combo_playability_score(combo)
        if combo[0] // 4 == combo[1] // 4:
            buckets["pair"].append((score, combo))
        elif combo[0] % 4 == combo[1] % 4:
            buckets["suited"].append((score, combo))
        else:
            buckets["offsuit"].append((score, combo))

    allocations = {k: 0 for k in buckets}
    totals = {k: len(v) for k, v in buckets.items()}
    assigned = 0
    for bucket, count in totals.items():
        if count == 0:
            continue
        alloc = min(count, int(limit * (count / len(combo_list))))
        allocations[bucket] = alloc
        assigned += alloc

    remaining = max(0, limit - assigned)
    if remaining:
        remainders = sorted(
            (
                (totals[bucket] - allocations[bucket], bucket)
                for bucket in buckets
                if totals[bucket] > allocations[bucket]
            ),
            reverse=True,
        )
        for _, bucket in remainders:
            if remaining == 0:
                break
            allocations[bucket] += 1
            remaining -= 1

    sampled: list[tuple[int, int]] = []
    for bucket, entries in buckets.items():
        if not entries:
            continue
        entries.sort(key=lambda item: item[0], reverse=True)
        take = min(allocations[bucket], len(entries))
        sampled.extend([combo for _, combo in entries[:take]])

    if not sampled:
        entries = sorted(((combo_playability_score(combo), combo) for combo in combo_list), reverse=True)
        sampled = [combo for _, combo in entries[:limit]]
    return sampled


@lru_cache(maxsize=8192)
def _solve_combo_profile(
    hero_combo: tuple[int, int],
    open_size: float,
) -> Mapping[str, float]:
    hero_combo = tuple(sorted(int(card) for card in hero_combo))
    blocked = list(hero_combo)
    villain_range = _villain_open_range(open_size, blocked)
    if not villain_range:
        return {"fold": 1.0, "call": 0.0, "threebet": 0.0, "jam": 0.0, "defend": 0.0}

    sampled_villain = _sample_combos(villain_range, _PRELOP_SAMPLE_LIMIT)
    equities, errors, equity_pairs, avg_eq = _equity_profiles(hero_combo, sampled_villain, _PRELOP_MC_TRIALS)
    if not equity_pairs:
        return {"fold": 1.0, "call": 0.0, "threebet": 0.0, "jam": 0.0, "defend": 0.0}

    pot = _BB_CONTRIBUTION + open_size
    hero_stack = max(0.0, _PRELOP_STACK - _BB_CONTRIBUTION)
    rival_stack = max(0.0, _PRELOP_STACK - open_size)

    weights = _weights_for_combos(sampled_villain, blocked)
    range_error = _weighted_error(errors, weights)

    options: list[Option] = []

    options.append(
        Option(
            key="Fold",
            ev=0.0,
            why="Fold preflop.",
            ends_hand=True,
            meta={
                "supports_cfr": True,
                "hero_ev_fold": 0.0,
                "hero_ev_continue": 0.0,
                "rival_fe": 0.0,
                "rival_continue_ratio": 1.0,
                "equity_std_error": range_error,
            },
        )
    )

    call_cost = max(0.0, open_size - _BB_CONTRIBUTION)
    final_pot_call = pot + call_cost
    call_ev = avg_eq * final_pot_call - call_cost
    options.append(
        Option(
            key="Call",
            ev=call_ev,
            why="Call the open raise.",
            meta={
                "supports_cfr": True,
                "hero_ev_fold": call_ev,
                "hero_ev_continue": call_ev,
                "rival_fe": 0.0,
                "rival_continue_ratio": 1.0,
                "equity_std_error": range_error,
            },
        )
    )

    raise_targets: list[float] = []
    for mult in (2.8, 3.5, 5.0):
        raise_to = round(open_size * mult, 2)
        hero_add = raise_to - _BB_CONTRIBUTION
        if hero_add <= 0 or hero_add > hero_stack:
            continue
        raise_targets.append(raise_to)

    for raise_to in raise_targets:
        hero_add = raise_to - _BB_CONTRIBUTION
        rival_call_to = max(0.0, raise_to - open_size)
        rival_call = min(rival_call_to, rival_stack)
        final_pot = pot + hero_add + rival_call
        if final_pot <= 0:
            continue
        be_threshold = rival_call / final_pot if final_pot > 0 else 1.0
        fe, avg_eq_called, continue_ratio = _fold_continue_stats(equity_pairs, be_threshold)
        ev_called = avg_eq_called * final_pot - hero_add if continue_ratio else -hero_add
        hero_ev_fold = pot
        hero_ev_continue = ev_called
        continue_error = _continue_error(equities, errors, weights, be_threshold)
        ev = fe * pot + (1 - fe) * hero_ev_continue
        options.append(
            Option(
                key=f"3-bet to {raise_to:.2f}bb",
                ev=ev,
                why=f"Three-bet to {raise_to:.2f}bb.",
                meta={
                    "supports_cfr": True,
                    "hero_ev_fold": hero_ev_fold,
                    "hero_ev_continue": hero_ev_continue,
                    "rival_fe": fe,
                    "rival_continue_ratio": continue_ratio,
                    "equity_std_error": range_error,
                    "continue_std_error": continue_error,
                },
            )
        )

    if hero_stack > 0.0:
        jam_to = _BB_CONTRIBUTION + hero_stack
        rival_call_to = max(0.0, min(jam_to, _PRELOP_STACK) - open_size)
        rival_call = min(rival_call_to, rival_stack)
        final_pot = pot + hero_stack + rival_call
        if final_pot > 0:
            be_threshold = rival_call / final_pot if final_pot > 0 else 1.0
            fe, avg_eq_called, continue_ratio = _fold_continue_stats(equity_pairs, be_threshold)
            ev_called = avg_eq_called * final_pot - hero_stack if continue_ratio else -hero_stack
            hero_ev_fold = pot
            hero_ev_continue = ev_called
            continue_error = _continue_error(equities, errors, weights, be_threshold)
            ev = fe * pot + (1 - fe) * hero_ev_continue
            options.append(
                Option(
                    key="All-in",
                    ev=ev,
                    why="Jam all-in.",
                    ends_hand=True,
                    meta={
                        "supports_cfr": True,
                        "hero_ev_fold": hero_ev_fold,
                        "hero_ev_continue": hero_ev_continue,
                        "rival_fe": fe,
                        "rival_continue_ratio": continue_ratio,
                        "equity_std_error": range_error,
                        "continue_std_error": continue_error,
                    },
                )
            )

    refined = _PRELOP_SOLVER.refine(None, options)
    profile = {"fold": 0.0, "call": 0.0, "threebet": 0.0, "jam": 0.0}
    for opt in refined:
        freq = float(getattr(opt, "gto_freq", 0.0))
        key = opt.key.lower()
        if key.startswith("fold"):
            profile["fold"] += freq
        elif key.startswith("call"):
            profile["call"] += freq
        elif key.startswith("3-bet"):
            profile["threebet"] += freq
        elif key.startswith("all-in") or key.startswith("jam"):
            profile["jam"] += freq

    total = sum(profile.values())
    if total <= 1e-9:
        profile["fold"] = 1.0
        total = 1.0
    normalised = {k: v / total for k, v in profile.items()}
    normalised["defend"] = normalised["call"] + normalised["threebet"] + normalised["jam"]
    base_mix = normalise_mix(action_mix_for_combo(hero_combo, open_size=open_size))
    base_profile = {
        "fold": base_mix.get("fold", 0.0),
        "call": base_mix.get("call", 0.0),
        "threebet": base_mix.get("threebet", 0.0),
        "jam": base_mix.get("jam", 0.0),
    }
    base_profile["defend"] = base_profile["call"] + base_profile["threebet"] + base_profile["jam"]

    blend = 0.35
    blended = {}
    for key in {"fold", "call", "threebet", "jam"}:
        blended[key] = (1 - blend) * normalised.get(key, 0.0) + blend * base_profile.get(key, 0.0)
    total_blended = sum(blended.values())
    if total_blended <= 0:
        blended["fold"] = 1.0
        total_blended = 1.0
    for key in blended:
        blended[key] /= total_blended
    blended["defend"] = blended["call"] + blended["threebet"] + blended["jam"]
    return blended


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


def action_profile_for_combo(
    combo: tuple[int, int],
    *,
    open_size: float,
    blocked: Iterable[int] | None = None,
) -> Mapping[str, float]:
    """Return a normalised defensive profile for the combo.

    The helper exposes a stable mapping so other modules can reference the
    aggregate defend share or specific action buckets without re-implementing
    ``action_mix_for_combo`` logic.
    """
    try:
        solved = _solve_combo_profile(combo, float(open_size))
        if solved:
            return solved
    except Exception:
        pass

    base_mix = action_mix_for_combo(combo, open_size=open_size, blocked=blocked or [])
    normalised = normalise_mix(base_mix)
    defend = normalised.get("call", 0.0) + normalised.get("threebet", 0.0) + normalised.get("jam", 0.0)
    profile = dict(normalised)
    profile["defend"] = defend
    return profile


def continue_combos(
    *,
    open_size: float,
    blocked: Iterable[int] | None = None,
    minimum_defend: float = 0.05,
) -> list[tuple[int, int]]:
    """Return combos that defend at least ``minimum_defend`` share vs the open.

    This gives the policy and rival modelling layers a deterministic way to
    approximate solver continue ranges while still honouring blockers.
    """

    blocked_cards = blocked or []
    threshold = max(0.0, min(1.0, minimum_defend))
    combos = _combos_without_blockers(blocked_cards)
    result: list[tuple[int, int]] = []
    for combo in combos:
        profile = action_profile_for_combo(combo, open_size=open_size, blocked=blocked_cards)
        if profile.get("defend", 0.0) >= threshold:
            result.append(combo)
    return result
