from __future__ import annotations

import copy
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

from ..core.models import Option, OptionResolution
from . import rival_strategy
from .bet_sizing import BetSizingManager
from .cards import format_card_ascii, format_cards_spaced
from .cfr import refine_options as _refine_with_cfr
from .episode import Node
from .equity import hero_equity_vs_combo, hero_equity_vs_range as _hero_equity_vs_range
from .preflop_mix import action_profile_for_combo, continue_combos
from .range_model import load_range_with_weights, rival_bb_defend_range, rival_sb_open_range, tighten_range

MAX_BET_OPTIONS = 4


def _fmt_pct(x: float, decimals: int = 0) -> str:
    return f"{100.0 * x:.{decimals}f}%"


def _select_fractions(fractions: Iterable[float], limit: int) -> list[float]:
    unique = sorted({float(max(0.0, frac)) for frac in fractions if float(frac) > 0})
    if not unique or limit <= 0:
        return []
    if len(unique) <= limit:
        return unique

    selected: list[float] = []
    selected.append(unique[0])
    used = {unique[0]}

    if limit > 1:
        selected.append(unique[-1])
        used.add(unique[-1])

    targets = [0.33, 0.5, 0.66, 0.75, 1.0, 1.25]
    while len(selected) < limit:
        candidate = None
        best_distance = float("inf")
        for value in unique:
            if value in used:
                continue
            for target in targets:
                distance = abs(value - target)
                if distance < best_distance - 1e-6:
                    best_distance = distance
                    candidate = value
            if candidate is None and value not in used:
                candidate = value
        if candidate is None:
            break
        selected.append(candidate)
        used.add(candidate)

    selected = sorted(selected[:limit])
    if len(selected) > limit:
        selected = selected[:limit]
    return selected


def _board_texture_score(board: Sequence[int]) -> float:
    try:
        return float(rival_strategy._board_draw_intensity(board))  # type: ignore[attr-defined]
    except AttributeError:
        # Fallback: treat missing helper as neutral texture.
        return 0.5


def _effective_spr(hand_state: Mapping[str, Any], pot: float) -> float:
    stack = _state_value(hand_state, "effective_stack", pot)
    return stack / max(pot, 1e-6)


def _flop_fraction_candidates(board: Sequence[int], spr: float) -> Iterable[float]:
    texture = _board_texture_score(board)
    candidates: set[float] = set()

    if texture < 0.45:
        candidates.add(0.25)
    candidates.add(0.33)
    if texture > 0.6:
        candidates.add(0.5)
    medium = 0.66 if spr > 2.2 else 0.5
    candidates.add(medium)
    candidates.add(0.75)
    if spr > 3.2:
        candidates.add(1.0)
    if spr > 4.5 and texture < 0.35:
        candidates.add(1.15)
    return candidates


def _turn_probe_candidates(texture: float, spr: float) -> Iterable[float]:
    candidates: set[float] = {0.5}
    if texture < 0.55:
        candidates.add(0.4)
    candidates.add(0.75 if spr > 2.0 else 0.6)
    if spr > 3.5:
        candidates.add(1.0)
    return candidates


def _river_lead_candidates(texture: float, spr: float) -> Iterable[float]:
    candidates: set[float] = {0.5, 0.85}
    if spr > 1.6:
        candidates.add(1.0)
    if spr > 2.8:
        candidates.add(1.35)
    if texture < 0.4 and spr > 3.5:
        candidates.add(1.6)
    return candidates


def _board_texture_key(cards: Iterable[int]) -> str:
    board = [format_card_ascii(card, upper=True) for card in cards]
    if not board:
        return ""
    return "|".join(sorted(board))


def _blocked_cards(hero: Iterable[int], board: Iterable[int]) -> set[int]:
    return set(hero) | set(board)


def _board_metadata(node: Node) -> dict[str, Any]:
    data: dict[str, Any] = {}
    board_key = node.context.get("board_key")
    if isinstance(board_key, str) and board_key:
        data["board_key"] = board_key
    if node.board:
        data["board_cards"] = list(node.board)
    return data


def _state_value(hand_state: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not hand_state:
        return default
    value = hand_state.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _recalc_pot(hand_state: dict[str, Any]) -> float:
    if "hero_contrib" in hand_state and "rival_contrib" in hand_state:
        pot = _state_value(hand_state, "hero_contrib") + _state_value(hand_state, "rival_contrib")
    else:
        pot = _state_value(hand_state, "pot")
    hand_state["pot"] = pot
    return pot


def _update_effective_stack(hand_state: dict[str, Any]) -> float:
    hero_stack = _state_value(hand_state, "hero_stack")
    rival_stack = _state_value(hand_state, "rival_stack")
    effective = min(hero_stack, rival_stack)
    hand_state["effective_stack"] = effective
    nodes = hand_state.get("nodes")
    if isinstance(nodes, dict):
        for node in nodes.values():
            if isinstance(node, Node):
                node.effective_bb = effective
    return effective


def _apply_contribution(hand_state: dict[str, Any], role: str, amount: float) -> float:
    if not hand_state or amount <= 0:
        return 0.0
    stack_key = f"{role}_stack"
    contrib_key = f"{role}_contrib"
    default_stack = float(hand_state.get("effective_stack", 100.0))
    stack = _state_value(hand_state, stack_key, default_stack)
    if stack <= 0:
        return 0.0
    applied = min(amount, stack)
    current_contrib = _state_value(hand_state, contrib_key)
    hand_state[contrib_key] = current_contrib + applied
    hand_state[stack_key] = max(0.0, stack - applied)
    _recalc_pot(hand_state)
    _update_effective_stack(hand_state)
    return applied


def _fold_continue_stats(
    hero_equities: Iterable[float | tuple[float, float]], rival_threshold: float
) -> tuple[float, float, float]:
    entries = list(hero_equities)
    if not entries:
        return 0.0, 0.0, 0.0
    fold_weight = 0.0
    continue_weight = 0.0
    continue_eq = 0.0
    total_weight = 0.0
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


def _sample_cap_preflop(mc_trials: int) -> int:
    return max(50, min(200, int(mc_trials * 1.2)))


def _sample_cap_postflop(mc_trials: int) -> int:
    return max(30, min(120, int(mc_trials * 0.6)))


def _rival_adapt_state(hand_state: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(hand_state, dict):
        return {"aggr": 0, "passive": 0}
    adapt = hand_state.setdefault("rival_adapt", {"aggr": 0, "passive": 0})
    if "aggr" not in adapt:
        adapt["aggr"] = 0
    if "passive" not in adapt:
        adapt["passive"] = 0
    return adapt


def _record_rival_adapt(hand_state: dict[str, Any] | None, aggressive: bool) -> None:
    if not isinstance(hand_state, dict):
        return
    adapt = _rival_adapt_state(hand_state)
    key = "aggr" if aggressive else "passive"
    adapt[key] = int(adapt.get(key, 0)) + 1


def _decision_meta(base_meta: dict[str, Any] | None, hand_state: dict[str, Any] | None) -> dict[str, Any]:
    meta_copy: dict[str, Any] = dict(base_meta or {})
    adapt = _rival_adapt_state(hand_state)
    meta_copy["rival_adapt"] = {
        "aggr": int(adapt.get("aggr", 0)),
        "passive": int(adapt.get("passive", 0)),
    }
    return meta_copy


def _combo_category(combo: tuple[int, int]) -> str:
    a, b = combo
    if a // 4 == b // 4:
        return "pair"
    if a % 4 == b % 4:
        return "suited"
    return "offsuit"


def _normalize_combo(combo: Iterable[int] | tuple[int, int]) -> tuple[int, int]:
    try:
        a, b = int(combo[0]), int(combo[1])  # type: ignore[index]
    except (TypeError, ValueError, IndexError):
        raise ValueError("combo must contain two card indices")
    if a > b:
        a, b = b, a
    return a, b


def _evenly_sample_indexed(entries: list[tuple[int, tuple[int, int]]], count: int) -> list[tuple[int, tuple[int, int]]]:
    if count <= 0 or not entries:
        return []
    if len(entries) <= count:
        return entries.copy()

    step = len(entries) / count
    sampled: list[tuple[int, tuple[int, int]]] = []
    for i in range(count):
        idx = int(i * step)
        if idx >= len(entries):
            idx = len(entries) - 1
        sampled.append(entries[idx])
    return sampled


def _weighted_sample(
    entries: list[tuple[int, tuple[int, int]]],
    count: int,
    weights: Mapping[tuple[int, int], float] | None,
    rng: random.Random,
) -> list[tuple[int, tuple[int, int]]]:
    if count <= 0 or not entries:
        return []
    pool = entries.copy()
    result: list[tuple[int, tuple[int, int]]] = []

    def entry_weight(entry: tuple[int, tuple[int, int]]) -> float:
        if not weights:
            return 1.0
        combo = _normalize_combo(entry[1])
        return max(0.0, float(weights.get(combo, 0.0)))

    for _ in range(min(count, len(pool))):
        totals = [entry_weight(entry) for entry in pool]
        weight_sum = sum(totals)
        if weight_sum <= 0.0:
            chosen = rng.randrange(len(pool))
        else:
            target = rng.random() * weight_sum
            cumulative = 0.0
            chosen = len(pool) - 1
            for idx, weight in enumerate(totals):
                cumulative += weight
                if cumulative >= target:
                    chosen = idx
                    break
        result.append(pool.pop(chosen))

    return result


def _sample_range(
    combos: Iterable[tuple[int, int]],
    limit: int,
    weights: Mapping[tuple[int, int], float] | None,
    rng: random.Random | None,
) -> list[tuple[int, int]]:
    combos_list = list(combos)
    total = len(combos_list)
    if limit <= 0 or total <= limit:
        return combos_list

    local_rng = rng or random.Random()

    buckets: dict[str, list[tuple[int, tuple[int, int]]]] = {
        "pair": [],
        "suited": [],
        "offsuit": [],
    }
    for idx, combo in enumerate(combos_list):
        buckets[_combo_category(combo)].append((idx, combo))

    allocations: dict[str, int] = {"pair": 0, "suited": 0, "offsuit": 0}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for cat, entries in buckets.items():
        count = len(entries)
        if count == 0:
            continue
        exact = limit * (count / total)
        alloc = min(count, int(exact))
        allocations[cat] = alloc
        assigned += alloc
        remainders.append((exact - alloc, cat))

    remaining = limit - assigned
    if remaining > 0:
        remainders.sort(reverse=True)
        for _, cat in remainders:
            if remaining <= 0:
                break
            available = len(buckets[cat])
            current = allocations[cat]
            if current >= available:
                continue
            allocations[cat] += 1
            remaining -= 1

    if remaining > 0:
        for cat in ("pair", "suited", "offsuit"):
            if remaining <= 0:
                break
            available = len(buckets[cat])
            if available == 0:
                continue
            alloc = allocations[cat]
            extra = min(available - alloc, remaining)
            if extra <= 0:
                continue
            allocations[cat] += extra
            remaining -= extra

    selected: list[tuple[int, tuple[int, int]]] = []
    for cat in ("pair", "suited", "offsuit"):
        entries = buckets[cat]
        if not entries:
            continue
        take = allocations[cat]
        if take <= 0:
            continue
        selected.extend(_weighted_sample(entries, take, weights, local_rng))

    selected.sort(key=lambda item: item[0])
    if len(selected) > limit:
        selected = selected[:limit]

    return [combo for _, combo in selected]


def _subset_weights(
    weights: Mapping[tuple[int, int], float] | None,
    combos: Iterable[tuple[int, int]],
) -> dict[tuple[int, int], float] | None:
    if not weights:
        return None
    subset: dict[tuple[int, int], float] = {}
    for combo in combos:
        normalized = _normalize_combo(combo)
        weight = weights.get(normalized, 0.0)
        if weight > 0:
            subset[normalized] = weight
    if not subset:
        return None
    total = sum(subset.values())
    if total <= 0:
        return None
    scale = 1.0 / total
    return {combo: weight * scale for combo, weight in subset.items()}


def _weighted_average(
    values: Mapping[tuple[int, int], float],
    weights: Mapping[tuple[int, int], float] | None,
) -> float:
    if not values:
        return 0.0
    if not weights:
        return float(sum(values.values()) / len(values))
    total_weight = 0.0
    weighted_sum = 0.0
    for combo, value in values.items():
        weight = float(weights.get(combo, 0.0))
        if weight <= 0:
            continue
        total_weight += weight
        weighted_sum += weight * value
    if total_weight <= 0:
        return float(sum(values.values()) / len(values))
    return float(weighted_sum / total_weight)


def _equity_with_weights(
    values: Mapping[tuple[int, int], float],
    weights: Mapping[tuple[int, int], float] | None,
) -> list[float | tuple[float, float]]:
    if not weights:
        return list(values.values())
    return [(values[combo], weights.get(combo, 0.0)) for combo in values]


def _top_weight_fraction(
    weights: Mapping[tuple[int, int], float] | None,
    fraction: float,
) -> tuple[dict[tuple[int, int], float] | None, float]:
    if not weights:
        return None, 0.0
    fraction = max(0.0, min(1.0, fraction))
    if fraction <= 0.0:
        return None, 0.0
    sorted_weights = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return None, 0.0
    target = total_weight * fraction
    selected: dict[tuple[int, int], float] = {}
    cumulative = 0.0
    for combo, weight in sorted_weights:
        if weight <= 0:
            continue
        selected[combo] = weight
        cumulative += weight
        if cumulative >= target:
            break
    if not selected:
        return None, 0.0
    selected_total = sum(selected.values())
    if selected_total <= 0:
        return None, 0.0
    scale = 1.0 / selected_total
    normalized = {combo: weight * scale for combo, weight in selected.items()}
    return normalized, min(1.0, selected_total)


def _weighted_equity(
    equities: Mapping[tuple[int, int], float],
    weights: Mapping[tuple[int, int], float] | None,
) -> float:
    if not equities:
        return 0.0
    if not weights:
        return sum(equities.values()) / len(equities)
    numerator = 0.0
    denominator = 0.0
    for combo, weight in weights.items():
        equity = equities.get(combo)
        if equity is None:
            continue
        numerator += equity * weight
        denominator += weight
    if denominator <= 0:
        return sum(equities.values()) / len(equities)
    return numerator / denominator


def _default_open_size(node: Node) -> float:
    return float(node.context.get("open_size") or 2.5)


# Public alias preserved for tests that monkeypatch hero_equity_vs_range.
hero_equity_vs_range = _hero_equity_vs_range


def reset_bet_sizing_state() -> None:
    """Reset the global bet sizing manager (used in tests)."""

    global BET_SIZING
    BET_SIZING = BetSizingManager()


@dataclass(frozen=True)
class MonteCarloPrecision:
    trials: int
    target_std_error: float | None = None

    def to_meta(self) -> dict[str, float | int]:
        data: dict[str, float | int] = {"combo_trials": self.trials}
        if self.target_std_error is not None:
            data["target_std_error"] = self.target_std_error
        return data


def _precision_for_street(mc_trials: int, street: str) -> MonteCarloPrecision:
    street_key = street.lower()
    base_trials = max(40, int(mc_trials * 0.55))

    if street_key == "river":
        return MonteCarloPrecision(trials=max(base_trials, int(mc_trials * 0.95)), target_std_error=0.025)
    if street_key == "turn":
        return MonteCarloPrecision(trials=max(base_trials, int(mc_trials * 0.8)), target_std_error=0.03)
    if street_key == "flop":
        return MonteCarloPrecision(trials=max(base_trials, int(mc_trials * 0.65)), target_std_error=0.04)
    return MonteCarloPrecision(trials=base_trials, target_std_error=0.05)


def _precision_from_meta(meta: dict[str, Any] | None, street: str) -> MonteCarloPrecision:
    default = _precision_for_street(80, street)
    if not isinstance(meta, dict):
        return default
    trials_raw = meta.get("combo_trials", default.trials)
    try:
        trials = int(trials_raw)
    except (TypeError, ValueError):
        trials = default.trials
    target_raw = meta.get("target_std_error", default.target_std_error)
    target: float | None
    try:
        target = float(target_raw) if target_raw is not None else default.target_std_error
    except (TypeError, ValueError):
        target = default.target_std_error
    if target is not None and (not math.isfinite(target) or target <= 0):
        target = default.target_std_error
    return MonteCarloPrecision(trials=max(1, trials), target_std_error=target)


@lru_cache(maxsize=2048)
def _cached_profile(
    _tag: str,
    combos_key: tuple[tuple[int, int], ...],
    fold_probability: float,
    continue_ratio: float,
    strengths_key: tuple[tuple[int, float], ...] | None,
    weights_key: tuple[tuple[int, float], ...] | None,
) -> dict | None:
    try:
        return rival_strategy.build_profile(
            list(combos_key),
            fold_probability=fold_probability,
            continue_ratio=continue_ratio,
            strengths=strengths_key,
            weights=weights_key,
        )
    except Exception:
        return None


def _rival_profile(
    combos: Iterable[tuple[int, int]],
    *,
    tag: str,
    fold_probability: float,
    continue_ratio: float,
    strengths: dict[tuple[int, int], float] | None = None,
    weights: Mapping[tuple[int, int], float] | None = None,
) -> tuple[dict | None, tuple[tuple[int, int], ...] | None]:
    combo_list = list(combos)
    if not combo_list:
        return None, None
    combos_key = tuple(tuple(map(int, combo)) for combo in combo_list)
    strengths_key: tuple[tuple[int, float], ...] | None = None
    if strengths:
        keyed: list[tuple[tuple[int, int], float]] = []
        for combo in combo_list:
            score = strengths.get(combo)
            normalized = tuple(sorted((int(combo[0]), int(combo[1]))))
            if score is None:
                score = strengths.get(normalized)
            if score is None:
                continue
            keyed.append((normalized, float(score)))
        if keyed:
            keyed.sort(key=lambda item: item[0])
            strengths_key = tuple(((pair[0], pair[1]), round(score, 4)) for pair, score in keyed)
    weights_key: tuple[tuple[int, float], ...] | None = None
    if weights:
        weighted_entries: list[tuple[tuple[int, int], float]] = []
        for combo in combo_list:
            normalized = tuple(sorted((int(combo[0]), int(combo[1]))))
            if normalized in weights:
                weight = float(weights[normalized])
                if weight > 0:
                    weighted_entries.append((normalized, weight))
        if weighted_entries:
            weighted_entries.sort(key=lambda item: item[0])
            weights_key = tuple(((c[0], c[1]), round(w, 6)) for c, w in weighted_entries)
    profile = _cached_profile(
        tag,
        combos_key,
        round(float(fold_probability), 4),
        round(float(continue_ratio), 4),
        strengths_key,
        weights_key,
    )
    if not profile:
        return None, None
    ranked = profile.get("ranked")
    continue_count = int(profile.get("continue_count", 0))
    continue_range: tuple[tuple[int, int], ...] | None = None
    if isinstance(ranked, list) and continue_count > 0:
        slice_end = min(len(ranked), continue_count)
        continue_range = tuple(tuple(map(int, combo)) for combo in ranked[:slice_end])
    # Return defensive copy so callers can mutate metadata without affecting the cache.
    return copy.deepcopy(profile), continue_range


def _apply_profile_meta(
    meta: dict[str, Any],
    profile: dict | None,
    continue_range: tuple[tuple[int, int], ...] | None,
) -> None:
    if profile:
        meta["rival_profile"] = profile
        weights = profile.get("continue_weights") if isinstance(profile, dict) else None
        if isinstance(weights, list):
            normalized: list[list[float | int]] = []
            for entry in weights:
                try:
                    a = int(entry[0])
                    b = int(entry[1])
                    w = float(entry[2])
                except (TypeError, ValueError, IndexError):
                    continue
                if w <= 0:
                    continue
                normalized.append([a, b, w])
            if normalized:
                meta["rival_continue_weights"] = normalized
    if continue_range:
        meta["rival_continue_range"] = continue_range


def _ensure_board_metadata(node: Node, options: Iterable[Option]) -> None:
    info = _board_metadata(node)
    if not info:
        return
    for opt in options:
        existing = opt.meta or {}
        for key, value in info.items():
            if key not in existing:
                existing[key] = value[:] if isinstance(value, list) else value
        opt.meta = existing


def _attach_cfr_meta(
    meta: dict[str, Any],
    *,
    fold_ev: float,
    continue_evs: Mapping[str, float] | float,
) -> None:
    if isinstance(continue_evs, Mapping):
        cont_map = {str(key): float(value) for key, value in continue_evs.items()}
    else:
        cont_map = {"continue": float(continue_evs)}

    meta["supports_cfr"] = True
    meta["hero_ev_fold"] = float(fold_ev)
    meta["rival_ev_fold"] = -float(fold_ev)

    # Backwards compatibility: prefer the explicit "continue" label when present.
    if "continue" in cont_map:
        primary_label = "continue"
    else:
        primary_label = next(iter(cont_map))
    primary_value = cont_map[primary_label]
    meta["hero_ev_continue"] = float(primary_value)
    meta["rival_ev_continue"] = -float(primary_value)

    rival_actions = ["fold", *cont_map.keys()]
    hero_payoffs = [float(fold_ev), *cont_map.values()]
    rival_payoffs = [-value for value in hero_payoffs]
    meta["cfr_payoffs"] = {
        "rival_actions": rival_actions,
        "hero": hero_payoffs,
        "rival": rival_payoffs,
    }


def _update_rival_range(hand_state: dict[str, Any], meta: dict[str, Any] | None, rival_folds: bool) -> None:
    if not hand_state:
        return
    if rival_folds:
        hand_state.pop("rival_continue_range", None)
        hand_state.pop("rival_continue_weights", None)
        return
    if not isinstance(meta, dict):
        return
    cont_range = meta.get("rival_continue_range")
    if not isinstance(cont_range, (list, tuple)):
        return
    normalized: list[tuple[int, int]] = [
        tuple(int(c) for c in combo) for combo in cont_range if isinstance(combo, (list, tuple)) and len(combo) == 2
    ]
    if normalized:
        hand_state["rival_continue_range"] = normalized
    weights_meta = meta.get("rival_continue_weights")
    if isinstance(weights_meta, list):
        normalized_weights: list[list[float | int]] = []
        total_weight = 0.0
        for entry in weights_meta:
            try:
                a = int(entry[0])
                b = int(entry[1])
                weight = float(entry[2])
            except (TypeError, ValueError, IndexError):
                continue
            if weight <= 0:
                continue
            normalized_weights.append([a, b, weight])
            total_weight += weight
        if total_weight > 0:
            scale = 1.0 / total_weight
            for value in normalized_weights:
                value[2] = float(value[2] * scale)
            hand_state["rival_continue_weights"] = normalized_weights
        else:
            hand_state.pop("rival_continue_weights", None)
    else:
        hand_state.pop("rival_continue_weights", None)


def _combo_equity(
    hero: list[int],
    board: list[int],
    combo: tuple[int, int],
    precision: MonteCarloPrecision,
) -> float:
    target = precision.target_std_error
    if target is not None:
        try:
            return hero_equity_vs_combo(
                hero,
                board,
                combo,
                precision.trials,
                target_std_error=target,
            )
        except TypeError:
            pass
    return hero_equity_vs_combo(hero, board, combo, precision.trials)


def _hand_state(node: Node) -> dict[str, Any] | None:
    hand_state = node.context.get("hand_state")
    if isinstance(hand_state, dict):
        return hand_state
    return None


def _rival_range_tag(node: Node, default: str = "sb_open") -> str:
    ctx_val = node.context.get("rival_range")
    if isinstance(ctx_val, str) and ctx_val:
        return ctx_val
    hand_state = _hand_state(node)
    if hand_state:
        hs_val = hand_state.get("rival_range")
        if isinstance(hs_val, str) and hs_val:
            return hs_val
    return default


def _rival_base_range(
    node: Node,
    blocked: Iterable[int],
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], float] | None]:
    hand_state = _hand_state(node)
    if hand_state:
        stored = hand_state.get("rival_continue_range")
        if isinstance(stored, (list, tuple)):
            stored_combos = [tuple(int(c) for c in combo) for combo in stored if len(combo) == 2]
            filtered = [combo for combo in stored_combos if combo[0] not in blocked and combo[1] not in blocked]
            weights_meta = hand_state.get("rival_continue_weights")
            weights: dict[tuple[int, int], float] | None = None
            if isinstance(weights_meta, list):
                temp_weights: dict[tuple[int, int], float] = {}
                for item in weights_meta:
                    try:
                        a = int(item[0])
                        b = int(item[1])
                        w = float(item[2])
                    except (TypeError, ValueError, IndexError):
                        continue
                    combo = tuple(sorted((a, b)))
                    if combo[0] in blocked or combo[1] in blocked or w <= 0:
                        continue
                    temp_weights[combo] = w
                if temp_weights:
                    total = sum(temp_weights.values())
                    if total > 0:
                        scale = 1.0 / total
                        weights = {combo: weight * scale for combo, weight in temp_weights.items() if weight > 0}
            if filtered:
                if weights:
                    # Ensure weights only cover filtered combos.
                    weights = {combo: weights.get(combo, 0.0) for combo in filtered}
                    total = sum(weights.values())
                    if total > 0:
                        weights = {combo: weight / total for combo, weight in weights.items() if weight > 0}
                    else:
                        weights = None
                return filtered, weights

    tag = _rival_range_tag(node)
    open_size = _default_open_size(node)
    if tag == "bb_defend":
        combos, weights = load_range_with_weights("bb_defend", open_size, blocked)
        if combos:
            return combos, weights
        improved = continue_combos(open_size=open_size, blocked=blocked, minimum_defend=0.08)
        if improved:
            return improved, None
        return rival_bb_defend_range(open_size, blocked), None
    combos, weights = load_range_with_weights("sb_open", open_size, blocked)
    if combos:
        return combos, weights
    return rival_sb_open_range(open_size, blocked), None


def _set_node_pot_from_state(node: Node, hand_state: dict[str, Any] | None) -> float:
    if not hand_state:
        return float(node.pot_bb)
    pot = _recalc_pot(hand_state)
    node.pot_bb = pot
    node.effective_bb = _state_value(hand_state, "effective_stack", node.effective_bb)
    return pot


def _rival_cards(hand_state: dict[str, Any] | None) -> tuple[int, int] | None:
    if not hand_state:
        return None
    cards = hand_state.get("rival_cards")
    if isinstance(cards, tuple) and len(cards) == 2:
        return cards  # type: ignore[return-value]
    if isinstance(cards, list) and len(cards) == 2:
        return tuple(cards)  # type: ignore[return-value]
    return None


def _rival_str(hand_state: dict[str, Any] | None, reveal: bool) -> str:
    cards = _rival_cards(hand_state)
    if reveal and cards:
        return format_cards_spaced(list(cards))
    return "(hidden)"


def _bet_context_tag(node: Node, suffix: str) -> str:
    ctx = node.context if isinstance(node.context, dict) else {}
    facing = str(ctx.get("facing", "neutral"))
    style = str(ctx.get("rival_style", ctx.get("style", "")))
    texture = str(ctx.get("texture", ctx.get("board_key", "")))
    return f"{suffix}|{facing}|{style}|{texture}"


def _set_street_pot(hand_state: dict[str, Any], street: str, pot: float) -> None:
    nodes = hand_state.get("nodes")
    if not isinstance(nodes, dict):
        return
    node = nodes.get(street)
    if isinstance(node, Node):
        node.pot_bb = pot
        node.effective_bb = _state_value(hand_state, "effective_stack", node.effective_bb)


def _rebuild_turn_node(hand_state: dict[str, Any], pot: float) -> None:
    nodes = hand_state.get("nodes")
    if not isinstance(nodes, dict):
        return
    turn_node = nodes.get("turn")
    if not isinstance(turn_node, Node):
        return
    turn_node.pot_bb = pot
    turn_node.effective_bb = _state_value(hand_state, "effective_stack", turn_node.effective_bb)
    bet_turn = round(0.5 * pot, 2)
    turn_node.context["facing"] = "bet"
    turn_node.context["bet"] = bet_turn
    board_turn = turn_node.board
    turn_node.context["board_key"] = _board_texture_key(board_turn)
    board_str = " ".join(format_card_ascii(c, upper=True) for c in board_turn)
    rival_seat = str(hand_state.get("rival_seat", "SB"))
    turn_node.description = f"{board_str}; Rival ({rival_seat}) bets {bet_turn:.2f}bb into {pot:.2f}bb."


def _rebuild_river_node(hand_state: dict[str, Any], pot: float) -> None:
    nodes = hand_state.get("nodes")
    if not isinstance(nodes, dict):
        return
    river_node = nodes.get("river")
    if not isinstance(river_node, Node):
        return
    river_node.pot_bb = pot
    river_node.effective_bb = _state_value(hand_state, "effective_stack", river_node.effective_bb)
    board_river = river_node.board
    board_str = " ".join(format_card_ascii(c, upper=True) for c in board_river)
    river_node.description = f"{board_str}; choose your bet."
    river_node.context["facing"] = "oop-check"
    river_node.context["board_key"] = _board_texture_key(board_river)
    river_node.context.pop("bet", None)


def _showdown_outcome(hero: list[int], board: list[int], rival: tuple[int, int]) -> float:
    eq = hero_equity_vs_combo(hero, board, rival, 1)
    if eq >= 0.999:
        return 1.0
    if eq <= 0.001:
        return 0.0
    return 0.5


def preflop_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    hero = node.hero_cards
    open_size = float(node.context.get("open_size") or 2.5)
    hero_combo = tuple(sorted(hero))
    hand_state = _hand_state(node)
    pot = _set_node_pot_from_state(node, hand_state)
    hero_contrib = _state_value(hand_state, "hero_contrib", 1.0)
    rival_contrib = _state_value(hand_state, "rival_contrib", open_size)
    hero_stack = _state_value(hand_state, "hero_stack", node.effective_bb)
    rival_stack = _state_value(hand_state, "rival_stack", node.effective_bb)
    hero_total = hero_contrib + hero_stack
    rival_total = rival_contrib + rival_stack

    call_cost = max(0.0, min(hero_stack, rival_contrib - hero_contrib))

    blocked = _blocked_cards(hero, [])
    solver_profile = action_profile_for_combo(hero_combo, open_size=open_size, blocked=blocked)
    fold_freq = max(0.0, 1.0 - float(solver_profile.get("defend", 0.0)))
    call_freq = float(solver_profile.get("call", 0.0))
    threebet_freq = float(solver_profile.get("threebet", 0.0))
    jam_freq = float(solver_profile.get("jam", 0.0))
    open_range, range_weights = _rival_base_range(node, blocked)
    sampled_range = _sample_range(open_range, _sample_cap_preflop(mc_trials), range_weights, rng)
    if not sampled_range:
        sampled_range = open_range
    precision = _precision_for_street(mc_trials, "preflop")
    equities: dict[tuple[int, int], float] = {}
    for combo in sampled_range:
        normalized_combo = _normalize_combo(combo)
        equities[normalized_combo] = _combo_equity(hero, [], normalized_combo, precision)
    sample_weights = _subset_weights(range_weights, sampled_range)
    avg_range_eq = _weighted_average(equities, sample_weights)

    options: list[Option] = [
        Option(
            "Fold",
            0.0,
            f"Fold now and lose nothing extra. Recommended {fold_freq:.0%} of the time versus this open.",
            gto_freq=fold_freq,
            ends_hand=True,
            meta={"street": "preflop", "action": "fold", "solver_mix": {"fold": fold_freq}},
        ),
    ]

    if call_cost > 0:
        final_pot_call = pot + call_cost
        be_call_eq = call_cost / final_pot_call if final_pot_call > 0 else 1.0
        options.append(
            Option(
                "Call",
                avg_range_eq * final_pot_call - call_cost,
                (
                    f"Pot odds: call {call_cost:.2f} bb to win {final_pot_call:.2f} bb. "
                    f"Need ≈{_fmt_pct(be_call_eq, 1)} equity, hand has {_fmt_pct(avg_range_eq, 1)}. "
                    f"Solver mix suggests calling {call_freq:.0%}."
                ),
                gto_freq=call_freq,
                meta={
                    "street": "preflop",
                    "action": "call",
                    "call_cost": call_cost,
                    "solver_mix": {"call": call_freq},
                },
            )
        )

    sorted_raises = BET_SIZING.preflop_raise_sizes(
        open_size=open_size,
        hero_contrib=hero_contrib,
        hero_stack=hero_stack,
        rival_stack=rival_stack,
    )
    selected_raises = sorted_raises[:MAX_BET_OPTIONS]
    raise_share = threebet_freq / len(selected_raises) if selected_raises else 0.0
    for raise_to in selected_raises:
        hero_add = raise_to - hero_contrib
        if hero_add <= 0 or hero_add > hero_stack + 1e-6:
            continue
        rival_call_to = max(0.0, raise_to - rival_contrib)
        rival_call = min(rival_call_to, rival_stack)
        final_pot = pot + hero_add + rival_call
        if final_pot <= 0:
            continue
        be_threshold = rival_call / final_pot if final_pot > 0 else 1.0
        fe, avg_eq_when_called, continue_ratio = _fold_continue_stats(
            _equity_with_weights(equities, sample_weights),
            be_threshold,
        )
        ev_called = avg_eq_when_called * final_pot - hero_add if continue_ratio else -hero_add
        ev = fe * pot + (1 - fe) * ev_called
        why = (
            f"Rival folds {_fmt_pct(fe)} needing eq {_fmt_pct(be_threshold, 1)}. "
            f"When called (~{_fmt_pct(continue_ratio)}) you have {_fmt_pct(avg_eq_when_called, 1)} equity "
            f"→ EV {ev_called:.2f} bb."
        )
        profile, continue_range = _rival_profile(
            sampled_range,
            tag=_rival_range_tag(node),
            fold_probability=fe,
            continue_ratio=continue_ratio,
            strengths=equities,
            weights=sample_weights,
        )
        meta = {
            "street": "preflop",
            "action": "3bet",
            "raise_to": raise_to,
            "rival_threshold": be_threshold,
            "pot_before": pot,
            "hero_add": hero_add,
            "rival_call": rival_call,
            "rival_fe": fe,
            "rival_continue_ratio": continue_ratio,
            "solver_mix": {"threebet": threebet_freq},
            "sizing_key": round(raise_to, 2),
        }
        meta.update(precision.to_meta())
        _apply_profile_meta(meta, profile, continue_range)
        _attach_cfr_meta(meta, fold_ev=pot, continue_evs={"call": ev_called})
        options.append(
            Option(
                f"3-bet to {raise_to:.2f}bb",
                ev,
                why + f" Solver mix uses this sizing {raise_share:.0%} of the time.",
                gto_freq=raise_share,
                meta=meta,
            )
        )

    hero_add = hero_stack
    if hero_add > 0.0:
        jam_to = hero_contrib + hero_add
        rival_call_to = max(0.0, min(jam_to, rival_total) - rival_contrib)
        rival_call = min(rival_call_to, rival_stack)
        final_pot = pot + hero_add + rival_call
        be_threshold = rival_call / final_pot if final_pot > 0 else 1.0
        fe, avg_eq_when_called, continue_ratio = _fold_continue_stats(
            _equity_with_weights(equities, sample_weights),
            be_threshold,
        )
        ev_called = avg_eq_when_called * final_pot - hero_add if continue_ratio else -hero_add
        ev = fe * pot + (1 - fe) * ev_called
        why_jam = (
            f"Rival folds {_fmt_pct(fe)} needing eq {_fmt_pct(be_threshold, 1)}. "
            f"When called (~{_fmt_pct(continue_ratio)}) you have {_fmt_pct(avg_eq_when_called, 1)} equity "
            f"→ EV {ev_called:.2f} bb."
        )
        profile, continue_range = _rival_profile(
            sampled_range,
            tag=_rival_range_tag(node),
            fold_probability=fe,
            continue_ratio=continue_ratio,
            strengths=equities,
            weights=sample_weights,
        )
        meta = {
            "street": "preflop",
            "action": "jam",
            "raise_to": jam_to,
            "risk": hero_add,
            "rival_threshold": be_threshold,
            "pot_before": pot,
            "rival_call_cost": rival_call_to,
            "rival_fe": fe,
            "rival_continue_ratio": continue_ratio,
            "solver_mix": {"jam": jam_freq},
            "sizing_key": round(jam_to, 2),
        }
        meta.update(precision.to_meta())
        _apply_profile_meta(meta, profile, continue_range)
        _attach_cfr_meta(meta, fold_ev=pot, continue_evs={"call": ev_called})
        meta["supports_cfr"] = False
        options.append(
            Option(
                "All-in",
                ev,
                why_jam + f" Solver mix jams {jam_freq:.0%}.",
                gto_freq=jam_freq,
                ends_hand=True,
                meta=meta,
            )
        )

    return options


def _turn_probe_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    hero = node.hero_cards
    board = node.board
    hand_state = _hand_state(node) or {}
    pot = _set_node_pot_from_state(node, hand_state)
    blocked = _blocked_cards(hero, board)
    base_range, base_weights = _rival_base_range(node, blocked)
    probe_tighten = float(hand_state.get("style_turn_probe_tighten", 0.7))
    probe_range = tighten_range(base_range, probe_tighten)
    probe_weights = _subset_weights(base_weights, probe_range)
    sampled_range = _sample_range(probe_range, _sample_cap_postflop(mc_trials), probe_weights, rng)
    if not sampled_range:
        sampled_range = probe_range
    precision = _precision_for_street(mc_trials, "turn")
    equities: dict[tuple[int, int], float] = {}
    for combo in sampled_range:
        normalized_combo = _normalize_combo(combo)
        equities[normalized_combo] = _combo_equity(hero, board, normalized_combo, precision)
    sample_weights = _subset_weights(probe_weights, sampled_range)
    avg_eq = _weighted_average(equities, sample_weights)

    options: list[Option] = [
        Option(
            "Check",
            avg_eq * pot,
            f"Take the free card with {_fmt_pct(avg_eq, 1)} equity.",
            meta={"street": "turn", "action": "check"},
        )
    ]

    base_probe_sizes = tuple(hand_state.get("style_turn_probe_sizes", (0.45, 0.75, 1.1)))
    probe_context = _bet_context_tag(node, "turn_probe")
    texture = _board_texture_score(board)
    spr = _effective_spr(hand_state, pot)
    probe_candidates = set(_turn_probe_candidates(texture, spr))
    if base_probe_sizes:
        probe_candidates.update(float(size) for size in base_probe_sizes)
    probe_sizes = BET_SIZING.postflop_bet_fractions(
        street="turn",
        context=probe_context,
        base_fractions=tuple(sorted(probe_candidates)),
    )
    for pct in _select_fractions(probe_sizes, MAX_BET_OPTIONS):
        bet = round(pot * pct, 2)
        if bet <= 0:
            continue
        final_pot = pot + 2 * bet
        be_threshold = bet / final_pot if final_pot > 0 else 1.0
        fe, eq_call, continue_ratio = _fold_continue_stats(
            _equity_with_weights(equities, sample_weights),
            be_threshold,
        )
        ev_called = eq_call * final_pot - bet if continue_ratio else -bet
        ev = fe * pot + (1 - fe) * ev_called
        why = (
            f"Bet {int(pct * 100)}%: rival folds {_fmt_pct(fe)} (needs eq {_fmt_pct(be_threshold, 1)}). "
            f"Calls (~{_fmt_pct(continue_ratio)}) give {_fmt_pct(eq_call, 1)} equity → EV {ev_called:.2f} bb."
        )
        profile, continue_range = _rival_profile(
            sampled_range,
            tag=_rival_range_tag(node),
            fold_probability=fe,
            continue_ratio=continue_ratio,
            strengths=equities,
            weights=sample_weights,
        )
        meta = {
            "street": "turn",
            "action": "bet",
            "bet": bet,
            "pot_before": pot,
            "rival_threshold": be_threshold,
            "rival_fe": fe,
            "rival_continue_ratio": continue_ratio,
            "sizing_fraction": float(pct),
            "bet_context": probe_context,
        }
        meta.update(precision.to_meta())
        _apply_profile_meta(meta, profile, continue_range)
        _attach_cfr_meta(meta, fold_ev=pot, continue_evs={"continue": ev_called})
        options.append(Option(f"Bet {int(pct * 100)}% pot", ev, why, meta=meta))

    risk = round(node.effective_bb, 2)
    if risk > 0:
        final_pot = pot + 2 * risk
        be_threshold = risk / final_pot if final_pot > 0 else 1.0
        fe, eq_call, continue_ratio = _fold_continue_stats(
            _equity_with_weights(equities, sample_weights),
            be_threshold,
        )
        ev_called = eq_call * final_pot - risk if continue_ratio else -risk
        ev = fe * pot + (1 - fe) * ev_called
        profile, continue_range = _rival_profile(
            sampled_range,
            tag=_rival_range_tag(node),
            fold_probability=fe,
            continue_ratio=continue_ratio,
            strengths=equities,
            weights=sample_weights,
        )
        meta = {
            "street": "turn",
            "action": "jam",
            "risk": risk,
            "rival_threshold": be_threshold,
            "pot_before": pot,
            "rival_fe": fe,
            "rival_continue_ratio": continue_ratio,
        }
        meta.update(precision.to_meta())
        _apply_profile_meta(meta, profile, continue_range)
        _attach_cfr_meta(meta, fold_ev=pot, continue_evs={"call": ev_called})
        meta["supports_cfr"] = False
        options.append(
            Option(
                "All-in",
                ev,
                (
                    f"All-in: rival folds {_fmt_pct(fe)} needing eq {_fmt_pct(be_threshold, 1)}. "
                    f"Calls (~{_fmt_pct(continue_ratio)}) you have {_fmt_pct(eq_call, 1)} → EV {ev_called:.2f} bb."
                ),
                ends_hand=True,
                meta=meta,
            )
        )

    return options


def flop_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    hero = node.hero_cards
    board = node.board
    hand_state = _hand_state(node) or {}
    pot = _set_node_pot_from_state(node, hand_state)
    blocked = _blocked_cards(hero, board)
    open_range, range_weights = _rival_base_range(node, blocked)
    sampled_range = _sample_range(open_range, _sample_cap_postflop(mc_trials), range_weights, rng)
    if not sampled_range:
        sampled_range = open_range
    precision = _precision_for_street(mc_trials, "flop")
    equities: dict[tuple[int, int], float] = {}
    for combo in sampled_range:
        normalized_combo = _normalize_combo(combo)
        equities[normalized_combo] = _combo_equity(hero, board, normalized_combo, precision)
    sample_weights = _subset_weights(range_weights, sampled_range)
    avg_eq = _weighted_average(equities, sample_weights)

    options: list[Option] = [
        Option(
            "Check",
            avg_eq * pot,
            f"Realize equity {_fmt_pct(avg_eq, 1)} in-position.",
            meta={"street": "flop", "action": "check"},
        ),
    ]

    cbet_context = _bet_context_tag(node, "flop_cbet")
    spr = _effective_spr(hand_state, pot)
    base_candidates = _flop_fraction_candidates(board, spr)
    cbet_fractions = BET_SIZING.postflop_bet_fractions(
        street="flop",
        context=cbet_context,
        base_fractions=tuple(sorted(base_candidates)),
    )
    for pct in _select_fractions(cbet_fractions, MAX_BET_OPTIONS):
        bet = round(pot * pct, 2)
        if bet <= 0:
            continue
        final_pot = pot + 2 * bet
        be_threshold = bet / final_pot
        fe, eq_call, continue_ratio = _fold_continue_stats(
            _equity_with_weights(equities, sample_weights),
            be_threshold,
        )
        ev_called = eq_call * final_pot - bet if continue_ratio else -bet
        ev = fe * pot + (1 - fe) * ev_called
        why = (
            f"{int(pct * 100)}% pot: rival folds {_fmt_pct(fe)} (needs eq {_fmt_pct(be_threshold, 1)}). "
            f"Continuing range (~{_fmt_pct(continue_ratio)}) gives you {_fmt_pct(eq_call, 1)} equity "
            f"→ EV {ev_called:.2f} bb."
        )
        why += f" Additional sizing detail: {bet:.2f} bb (equals {int(pct * 100)}% pot)."
        profile, continue_range = _rival_profile(
            sampled_range,
            tag=_rival_range_tag(node),
            fold_probability=fe,
            continue_ratio=continue_ratio,
            strengths=equities,
            weights=sample_weights,
        )
        meta = {
            "street": "flop",
            "action": "bet",
            "bet": bet,
            "pot_before": pot,
            "rival_threshold": be_threshold,
            "rival_fe": fe,
            "rival_continue_ratio": continue_ratio,
            "sizing_fraction": float(pct),
            "bet_context": cbet_context,
        }
        meta.update(precision.to_meta())
        _apply_profile_meta(meta, profile, continue_range)
        _attach_cfr_meta(meta, fold_ev=pot, continue_evs={"continue": ev_called})
        options.append(Option(f"Bet {int(pct * 100)}% pot", ev, why, meta=meta))

    # All-in shove option for maximum pressure
    risk = round(node.effective_bb, 2)
    if risk > 0:
        final_pot = pot + 2 * risk
        be_threshold = risk / final_pot if final_pot > 0 else 1.0
        fe, eq_call, continue_ratio = _fold_continue_stats(
            _equity_with_weights(equities, sample_weights),
            be_threshold,
        )
        ev_called = eq_call * final_pot - risk if continue_ratio else -risk
        ev = fe * pot + (1 - fe) * ev_called
        profile, continue_range = _rival_profile(
            sampled_range,
            tag=_rival_range_tag(node),
            fold_probability=fe,
            continue_ratio=continue_ratio,
            strengths=equities,
            weights=sample_weights,
        )
        meta = {
            "street": "flop",
            "action": "jam",
            "risk": risk,
            "rival_threshold": be_threshold,
            "pot_before": pot,
            "rival_fe": fe,
            "rival_continue_ratio": continue_ratio,
        }
        meta.update(precision.to_meta())
        _apply_profile_meta(meta, profile, continue_range)
        _attach_cfr_meta(meta, fold_ev=pot, continue_evs={"call": ev_called})
        meta["supports_cfr"] = False
        options.append(
            Option(
                "All-in",
                ev,
                (
                    f"Full stack shove: rival folds {_fmt_pct(fe)} needing eq {_fmt_pct(be_threshold, 1)}. "
                    f"When called you have {_fmt_pct(eq_call, 1)} → EV {ev_called:.2f} bb."
                ),
                ends_hand=True,
                meta=meta,
            )
        )

    return options


def _river_vs_bet_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    hero = node.hero_cards
    board = node.board
    hand_state = _hand_state(node) or {}
    pot_start = _set_node_pot_from_state(node, hand_state)
    rival_bet = float(node.context.get("bet") or round(0.75 * pot_start, 2))
    node.context["bet"] = rival_bet
    pot_after_bet = pot_start + rival_bet
    blocked = _blocked_cards(hero, board)
    base_range, base_weights = _rival_base_range(node, blocked)
    lead_tighten = float(hand_state.get("style_river_lead_tighten", 0.5))
    lead_range = tighten_range(base_range, lead_tighten)
    lead_weights = _subset_weights(base_weights, lead_range)
    sampled_range = _sample_range(lead_range, _sample_cap_postflop(mc_trials), lead_weights, rng)
    if not sampled_range:
        sampled_range = lead_range
    precision = _precision_for_street(mc_trials, "river")
    equities: dict[tuple[int, int], float] = {}
    for combo in sampled_range:
        normalized_combo = _normalize_combo(combo)
        equities[normalized_combo] = _combo_equity(hero, board, normalized_combo, precision)
    sample_weights = _subset_weights(lead_weights, sampled_range)
    avg_eq = _weighted_average(equities, sample_weights)

    options: list[Option] = [
        Option(
            "Fold",
            0.0,
            "Cut losses and fold river.",
            ends_hand=True,
            meta={"street": "river", "action": "fold", "rival_bet": rival_bet},
        )
    ]

    call_meta = {
        "street": "river",
        "action": "call",
        "rival_bet": rival_bet,
        **precision.to_meta(),
    }
    call_ev = avg_eq * pot_after_bet - (1 - avg_eq) * rival_bet
    options.append(
        Option(
            "Call",
            call_ev,
            f"Calling invests {rival_bet:.2f}bb with {_fmt_pct(avg_eq, 1)} showdown equity.",
            meta=call_meta,
        )
    )

    raise_to = round(max(rival_bet * 2.5, rival_bet + pot_start), 2)
    risk = raise_to - _state_value(hand_state, "hero_contrib")
    risk = max(risk, rival_bet + 0.5)
    rival_call_cost = max(0.0, raise_to - _state_value(hand_state, "rival_contrib") - rival_bet)
    final_pot = pot_after_bet + raise_to
    be_threshold = rival_call_cost / final_pot if final_pot > 0 else 1.0
    fe, eq_call, continue_ratio = _fold_continue_stats(
        _equity_with_weights(equities, sample_weights),
        be_threshold,
    )
    ev_called = eq_call * final_pot - risk if continue_ratio else -risk
    ev = fe * pot_after_bet + (1 - fe) * ev_called
    raise_profile, continue_range = _rival_profile(
        sampled_range,
        tag=_rival_range_tag(node),
        fold_probability=fe,
        continue_ratio=continue_ratio,
        strengths=equities,
        weights=sample_weights,
    )
    raise_meta = {
        "street": "river",
        "action": "raise",
        "raise_to": raise_to,
        "rival_threshold": be_threshold,
        "rival_bet": rival_bet,
        "pot_before": pot_after_bet,
        "rival_fe": fe,
        "rival_continue_ratio": continue_ratio,
    }
    raise_meta.update(precision.to_meta())
    _apply_profile_meta(raise_meta, raise_profile, continue_range)
    _attach_cfr_meta(raise_meta, fold_ev=pot_after_bet, continue_evs={"continue": ev_called})
    options.append(
        Option(
            f"Raise to {raise_to:.2f} bb",
            ev,
            f"Raise: FE {_fmt_pct(fe)}; EV {ev:.2f}bb.",
            meta=raise_meta,
        )
    )

    risk_allin = round(node.effective_bb, 2)
    if risk_allin > 0:
        final_pot_allin = pot_after_bet + 2 * risk_allin
        be_threshold_allin = risk_allin / final_pot_allin if final_pot_allin > 0 else 1.0
        fe_ai, eq_call_ai, continue_ratio_ai = _fold_continue_stats(
            _equity_with_weights(equities, sample_weights),
            be_threshold_allin,
        )
        ev_called_ai = eq_call_ai * final_pot_allin - risk_allin if continue_ratio_ai else -risk_allin
        ev_ai = fe_ai * pot_after_bet + (1 - fe_ai) * ev_called_ai
        profile_ai, continue_range_ai = _rival_profile(
            sampled_range,
            tag=_rival_range_tag(node),
            fold_probability=fe_ai,
            continue_ratio=continue_ratio_ai,
            strengths=equities,
            weights=sample_weights,
        )
        jam_meta = {
            "street": "river",
            "action": "jam",
            "risk": risk_allin,
            "rival_threshold": be_threshold_allin,
            "pot_before": pot_after_bet,
            "rival_fe": fe_ai,
            "rival_continue_ratio": continue_ratio_ai,
        }
        jam_meta.update(precision.to_meta())
        _apply_profile_meta(jam_meta, profile_ai, continue_range_ai)
        _attach_cfr_meta(jam_meta, fold_ev=pot_after_bet, continue_evs={"continue": ev_called_ai})
        options.append(
            Option(
                "All-in",
                ev_ai,
                (
                    f"Jam: rival folds {_fmt_pct(fe_ai)} needing eq {_fmt_pct(be_threshold_allin, 1)}. "
                    f"Calls (~{_fmt_pct(continue_ratio_ai)}) give {_fmt_pct(eq_call_ai, 1)} equity "
                    f"→ EV {ev_called_ai:.2f} bb."
                ),
                ends_hand=True,
                meta=jam_meta,
            )
        )

    return options


def turn_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    facing = str(node.context.get("facing") or "bet").lower()
    if facing == "check":
        return _turn_probe_options(node, rng, mc_trials)

    hero = node.hero_cards
    board = node.board
    hand_state = _hand_state(node) or {}
    pot_start = _set_node_pot_from_state(node, hand_state)
    rival_bet = float(node.context.get("bet") or round(0.5 * pot_start, 2))
    node.context["bet"] = rival_bet
    pot_before_action = pot_start + rival_bet
    blocked = _blocked_cards(hero, board)
    base_range, base_weights = _rival_base_range(node, blocked)
    tighten = float(hand_state.get("style_turn_bet_tighten", 0.55))
    bet_range = tighten_range(base_range, tighten)
    bet_weights = _subset_weights(base_weights, bet_range)
    sampled_range = _sample_range(bet_range, _sample_cap_postflop(mc_trials), bet_weights, rng)
    if not sampled_range:
        sampled_range = bet_range
    precision = _precision_for_street(mc_trials, "turn")
    equities: dict[tuple[int, int], float] = {}
    for combo in sampled_range:
        normalized_combo = _normalize_combo(combo)
        equities[normalized_combo] = _combo_equity(hero, board, normalized_combo, precision)
    sample_weights = _subset_weights(bet_weights, sampled_range)
    avg_eq = _weighted_average(equities, sample_weights)

    options = [
        Option(
            "Fold",
            0.0,
            "Fold and wait for a better spot.",
            ends_hand=True,
            meta={"street": "turn", "action": "fold", "rival_bet": rival_bet},
        )
    ]

    final_pot_call = pot_start + 2 * rival_bet
    be_call_eq = rival_bet / final_pot_call if final_pot_call > 0 else 1.0
    call_meta = {
        "street": "turn",
        "action": "call",
        "rival_bet": rival_bet,
    }
    call_meta.update(precision.to_meta())
    options.append(
        Option(
            "Call",
            avg_eq * final_pot_call - rival_bet,
            (
                f"Pot odds: call {rival_bet:.2f} bb to win {final_pot_call:.2f} bb. "
                f"Need {_fmt_pct(be_call_eq, 1)} equity, hand has {_fmt_pct(avg_eq, 1)}."
            ),
            meta=call_meta,
        )
    )

    raise_to = round(max(rival_bet * 2.5, rival_bet + 0.5), 2)
    risk = raise_to
    final_pot = pot_start + 2 * raise_to
    rival_call_cost = raise_to - rival_bet
    be_threshold = rival_call_cost / final_pot if final_pot > 0 else 1.0
    fe, eq_call, continue_ratio = _fold_continue_stats(
        _equity_with_weights(equities, sample_weights),
        be_threshold,
    )
    ev_called = eq_call * final_pot - risk if continue_ratio else -risk
    ev = fe * pot_before_action + (1 - fe) * ev_called
    fe_break_even = risk / (risk + pot_before_action) if (risk + pot_before_action) > 0 else 1.0
    why_raise = (
        f"Rival folds {_fmt_pct(fe)} (needs eq {_fmt_pct(be_threshold, 1)}). "
        f"break-even FE {_fmt_pct(fe_break_even)}. "
        f"Continuing (~{_fmt_pct(continue_ratio)}) you have {_fmt_pct(eq_call, 1)} equity → "
        f"EV {ev_called:.2f} bb."
    )
    profile, continue_range = _rival_profile(
        sampled_range,
        tag=_rival_range_tag(node),
        fold_probability=fe,
        continue_ratio=continue_ratio,
        strengths=equities,
        weights=sample_weights,
    )
    raise_meta = {
        "street": "turn",
        "action": "raise",
        "raise_to": raise_to,
        "rival_threshold": be_threshold,
        "rival_bet": rival_bet,
        "pot_before": pot_start,
        "rival_fe": fe,
        "rival_continue_ratio": continue_ratio,
    }
    raise_meta.update(precision.to_meta())
    _apply_profile_meta(raise_meta, profile, continue_range)
    _attach_cfr_meta(raise_meta, fold_ev=pot_before_action, continue_evs={"continue": ev_called})
    options.append(Option(f"Raise to {raise_to:.2f} bb", ev, why_raise, meta=raise_meta))

    return options


def river_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    facing = str(node.context.get("facing") or "oop-check").lower()
    if facing == "bet":
        return _river_vs_bet_options(node, rng, mc_trials)

    hero = node.hero_cards
    board = node.board
    hand_state = _hand_state(node) or {}
    pot = _set_node_pot_from_state(node, hand_state)
    blocked = _blocked_cards(hero, board)
    base_range, base_weights = _rival_base_range(node, blocked)
    check_tighten = float(hand_state.get("style_river_check_tighten", 0.65))
    check_range = tighten_range(base_range, check_tighten)
    check_weights = _subset_weights(base_weights, check_range)
    sampled_range = _sample_range(check_range, _sample_cap_postflop(mc_trials), check_weights, rng)
    if not sampled_range:
        sampled_range = check_range
    precision = _precision_for_street(mc_trials, "river")
    equities: dict[tuple[int, int], float] = {}
    for combo in sampled_range:
        normalized_combo = _normalize_combo(combo)
        equities[normalized_combo] = _combo_equity(hero, board, normalized_combo, precision)
    sample_weights = _subset_weights(check_weights, sampled_range)
    avg_eq = _weighted_average(equities, sample_weights)

    options: list[Option] = [
        Option(
            "Check",
            avg_eq * pot,
            f"Showdown equity {_fmt_pct(avg_eq, 1)} vs check range.",
            meta={"street": "river", "action": "check", **precision.to_meta()},
        )
    ]

    river_context = _bet_context_tag(node, "river_lead")
    texture = _board_texture_score(board)
    spr = _effective_spr(hand_state, pot)
    lead_candidates = set(_river_lead_candidates(texture, spr))
    river_fractions = BET_SIZING.postflop_bet_fractions(
        street="river",
        context=river_context,
        base_fractions=tuple(sorted(lead_candidates)),
    )
    for pct in _select_fractions(river_fractions, MAX_BET_OPTIONS):
        bet = round(pot * pct, 2)
        if bet <= 0:
            continue
        final_pot = pot + 2 * bet
        be_threshold = bet / final_pot if final_pot > 0 else 1.0
        fe, eq_call, continue_ratio = _fold_continue_stats(
            _equity_with_weights(equities, sample_weights),
            be_threshold,
        )
        ev_showdown = eq_call * (pot + bet) - (1 - eq_call) * bet
        ev_called = ev_showdown if continue_ratio else -bet
        jam_weights, jam_mass = _top_weight_fraction(check_weights, 0.35)
        if jam_mass > continue_ratio:
            jam_mass = continue_ratio
        call_share = max(0.0, continue_ratio - jam_mass)

        jam_total = max(bet, node.effective_bb)
        final_pot_jam = pot + bet + jam_total
        eq_jam = _weighted_equity(equities, jam_weights)
        hero_call_ev = eq_jam * final_pot_jam - jam_total
        hero_best_vs_jam = max(-bet, hero_call_ev)

        ev = fe * pot + call_share * ev_called + jam_mass * hero_best_vs_jam
        why = (
            f"Bet {int(pct * 100)}%: rival folds {_fmt_pct(fe)} (needs eq {_fmt_pct(be_threshold, 1)}). "
            f"Calls (~{_fmt_pct(continue_ratio)}) give you {_fmt_pct(eq_call, 1)} equity → EV {ev_called:.2f} bb."
        )
        if jam_mass > 0:
            why += (
                f" Check-raise jams (~{_fmt_pct(jam_mass)}) best response EV {hero_best_vs_jam:.2f} bb"
                f" (call EV {hero_call_ev:.2f} bb)."
            )
        why += f" Additional sizing detail: {bet:.2f} bb (equals {int(pct * 100)}% pot)."
        profile, continue_range = _rival_profile(
            sampled_range,
            tag=_rival_range_tag(node),
            fold_probability=fe,
            continue_ratio=continue_ratio,
            strengths=equities,
            weights=check_weights,
        )
        meta = {
            "street": "river",
            "action": "bet",
            "bet": bet,
            "pot_before": pot,
            "rival_threshold": be_threshold,
            "rival_fe": fe,
            "rival_continue_ratio": continue_ratio,
            "rival_raise_ratio": jam_mass,
            "hero_ev_raise": hero_best_vs_jam,
            "hero_call_vs_raise": hero_call_ev,
            "sizing_fraction": float(pct),
            "bet_context": river_context,
        }
        meta.update(precision.to_meta())
        _apply_profile_meta(meta, profile, continue_range)
        continuation_evs: dict[str, float] = {"call": ev_called}
        if jam_mass > 0:
            continuation_evs["jam"] = hero_best_vs_jam
        _attach_cfr_meta(meta, fold_ev=pot, continue_evs=continuation_evs)
        options.append(Option(f"Bet {int(pct * 100)}% pot", ev, why, meta=meta))

    risk = round(node.effective_bb, 2)
    if risk > 0:
        final_pot = pot + 2 * risk
        be_threshold = risk / final_pot if final_pot > 0 else 1.0
        fe, eq_call, continue_ratio = _fold_continue_stats(
            _equity_with_weights(equities, sample_weights),
            be_threshold,
        )
        ev_called = eq_call * final_pot - risk if continue_ratio else -risk
        ev = fe * pot + (1 - fe) * ev_called
        profile, continue_range = _rival_profile(
            sampled_range,
            tag=_rival_range_tag(node),
            fold_probability=fe,
            continue_ratio=continue_ratio,
            strengths=equities,
            weights=sample_weights,
        )
        meta = {
            "street": "river",
            "action": "jam",
            "risk": risk,
            "rival_threshold": be_threshold,
            "pot_before": pot,
            "rival_fe": fe,
            "rival_continue_ratio": continue_ratio,
        }
        meta.update(precision.to_meta())
        _apply_profile_meta(meta, profile, continue_range)
        _attach_cfr_meta(meta, fold_ev=pot, continue_evs={"call": ev_called})
        meta["supports_cfr"] = False
        options.append(
            Option(
                "All-in",
                ev,
                (
                    f"All-in: rival folds {_fmt_pct(fe)} needing eq {_fmt_pct(be_threshold, 1)}. "
                    f"Calls (~{_fmt_pct(continue_ratio)}) give you {_fmt_pct(eq_call, 1)} → EV {ev_called:.2f} bb."
                ),
                ends_hand=True,
                meta=meta,
            )
        )

    return options


def options_for(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    if node.street == "preflop":
        options = preflop_options(node, rng, mc_trials)
    elif node.street == "flop":
        options = flop_options(node, rng, mc_trials)
    elif node.street == "turn":
        options = turn_options(node, rng, mc_trials)
    elif node.street == "river":
        options = river_options(node, rng, mc_trials)
    else:
        raise ValueError(f"Unknown street: {node.street}")
    refined = _refine_with_cfr(node, options)
    _record_bet_sizing_feedback(node, refined)
    _ensure_board_metadata(node, refined)
    return refined


def _record_bet_sizing_feedback(node: Node, options: list[Option]) -> None:
    hand_state = _hand_state(node)
    if not hand_state:
        return
    if node.street == "preflop":
        open_size = float(node.context.get("open_size", 2.5))
        hero_contrib = _state_value(hand_state, "hero_contrib", 1.0)
        hero_stack = _state_value(hand_state, "hero_stack", node.effective_bb)
        rival_stack = _state_value(hand_state, "rival_stack", node.effective_bb)
        observations: list[tuple[float, float, float]] = []
        for opt in options:
            meta = opt.meta or {}
            if meta.get("action") != "3bet":
                continue
            size = float(meta.get("raise_to", hero_contrib))
            freq = float(getattr(opt, "gto_freq", 0.0))
            if freq <= 0:
                continue
            regret = float(meta.get("cfr_regret", 0.0))
            observations.append((size, freq, regret))
        if observations:
            BET_SIZING.observe_preflop(
                open_size=open_size,
                hero_contrib=hero_contrib,
                hero_stack=hero_stack,
                rival_stack=rival_stack,
                observations=observations,
            )
        return

    grouped: dict[str, list[tuple[float, float, float]]] = {}
    for opt in options:
        meta = opt.meta or {}
        fraction = meta.get("sizing_fraction")
        context = meta.get("bet_context")
        if fraction is None or context is None:
            continue
        freq = float(getattr(opt, "gto_freq", 0.0))
        if freq <= 0:
            continue
        regret = float(meta.get("cfr_regret", 0.0))
        grouped.setdefault(str(context), []).append((float(fraction), freq, regret))

    for context, obs in grouped.items():
        BET_SIZING.observe_postflop(
            street=node.street,
            context=context,
            observations=obs,
        )


def resolve_for(node: Node, option: Option, rng: random.Random) -> OptionResolution:
    meta = option.meta or {}
    hand_state = _hand_state(node)
    if not hand_state:
        return OptionResolution(hand_ended=getattr(option, "ends_hand", False))

    street = str(meta.get("street") or node.street)
    if street == "preflop":
        return _resolve_preflop(node, option, hand_state, rng)
    if street == "flop":
        return _resolve_flop(node, option, hand_state, rng)
    if street == "turn":
        return _resolve_turn(node, option, hand_state, rng)
    if street == "river":
        return _resolve_river(node, option, hand_state, rng)
    return OptionResolution(hand_ended=getattr(option, "ends_hand", False))


def _resolve_preflop(
    node: Node,
    option: Option,
    hand_state: dict[str, Any],
    rng: random.Random,
) -> OptionResolution:
    action = option.meta.get("action") if option.meta else None
    pot = _state_value(hand_state, "pot", node.pot_bb)
    open_size = float(node.context.get("open_size", 2.5))
    rival_cards = _rival_cards(hand_state)
    hero_cards = node.hero_cards
    hero_contrib = _state_value(hand_state, "hero_contrib")
    rival_contrib = _state_value(hand_state, "rival_contrib", open_size)

    if action == "fold":
        hand_state["hand_over"] = True
        rival_seat = str(hand_state.get("rival_seat", "SB"))
        _update_rival_range(hand_state, option.meta, True)
        return OptionResolution(hand_ended=True, note=f"You fold. {rival_seat} keeps {pot:.2f}bb.")

    if action == "call":
        call_cost = float(option.meta.get("call_cost", max(0.0, rival_contrib - hero_contrib)))
        _record_rival_adapt(hand_state, aggressive=False)
        invested = _apply_contribution(hand_state, "hero", call_cost)
        hand_state["street"] = "flop"
        hand_state["board_index"] = 3
        _set_street_pot(hand_state, "flop", _state_value(hand_state, "pot"))
        return OptionResolution(note=f"You call {invested:.2f}bb. Pot now {hand_state['pot']:.2f}bb.")

    if action == "3bet":
        raise_to = float(option.meta.get("raise_to", hero_contrib))
        hero_add = max(0.0, raise_to - hero_contrib)
        _apply_contribution(hand_state, "hero", hero_add)
        precision = _precision_from_meta(option.meta, "preflop")
        be_threshold = float(option.meta.get("rival_threshold", 1.0))
        board = node.board if node.board else []
        _record_rival_adapt(hand_state, aggressive=True)
        decision_meta = _decision_meta(option.meta, hand_state)
        decision = rival_strategy.decide_action(decision_meta, rival_cards, rng)
        if rival_cards is not None:
            hero_vs_known = _combo_equity(hero_cards, board, rival_cards, precision)
            if 1.0 - hero_vs_known < be_threshold:
                decision = rival_strategy.RivalDecision(folds=True)
        if decision.folds:
            _update_rival_range(hand_state, option.meta, True)
            hand_state["hand_over"] = True
            total_pot = _state_value(hand_state, "pot")
            net_gain = total_pot - hero_add
            if rival_cards is None:
                note = f"Rival folds to your 3-bet. Pot {total_pot:.2f}bb (rival hand hidden)."
            else:
                note = f"Rival folds to your 3-bet. Pot {total_pot:.2f}bb awarded (net +{net_gain:.2f}bb)."
            return OptionResolution(hand_ended=True, note=note)
        call_amount = max(0.0, raise_to - rival_contrib)
        _apply_contribution(hand_state, "rival", call_amount)
        hand_state["street"] = "flop"
        hand_state["board_index"] = 3
        _set_street_pot(hand_state, "flop", _state_value(hand_state, "pot"))
        _update_rival_range(hand_state, option.meta, False)
        if rival_cards is None:
            return OptionResolution(note=f"3-bet to {raise_to:.1f}bb. Pot now {hand_state['pot']:.2f}bb.")
        hero_eq = _combo_equity(hero_cards, [], rival_cards, precision)
        equity_note = _fmt_pct(hero_eq, 1)
        return OptionResolution(note=f"Rival calls the 3-bet. Your equity {equity_note}.")

    if action in {"jam", "allin", "all-in"}:
        jam_to = float(option.meta.get("raise_to", hero_contrib + _state_value(hand_state, "hero_stack")))
        hero_add = max(0.0, jam_to - hero_contrib)
        _apply_contribution(hand_state, "hero", hero_add)
        precision = _precision_from_meta(option.meta, "preflop")
        hand_state["hand_over"] = True
        _record_rival_adapt(hand_state, aggressive=True)
        decision_meta = _decision_meta(option.meta, hand_state)
        decision = rival_strategy.decide_action(decision_meta, rival_cards, rng)
        if decision.folds:
            _update_rival_range(hand_state, option.meta, True)
            scoop = _state_value(hand_state, "pot")
            net_gain = scoop - hero_add
            if rival_cards is None:
                note = f"You jam to {jam_to:.2f}bb. Rival folds (hand hidden)."
                return OptionResolution(hand_ended=True, note=note)
            note = f"Rival folds to your jam. Pot {scoop:.2f}bb awarded (net +{net_gain:.2f}bb)."
            return OptionResolution(hand_ended=True, note=note, reveal_rival=True)
        call_amount = max(0.0, min(jam_to, rival_contrib + _state_value(hand_state, "rival_stack")) - rival_contrib)
        _apply_contribution(hand_state, "rival", call_amount)
        _update_rival_range(hand_state, option.meta, False)
        if rival_cards is None:
            return OptionResolution(hand_ended=True, note=f"You jam to {jam_to:.2f}bb. Rival action hidden.")
        hero_eq = _combo_equity(hero_cards, [], rival_cards, precision)
        rival_text = format_cards_spaced(list(rival_cards))
        equity_note = _fmt_pct(hero_eq, 1)
        return OptionResolution(
            hand_ended=True,
            note=f"Rival calls jam with {rival_text}. Your equity {equity_note}.",
            reveal_rival=True,
        )

    return OptionResolution(hand_ended=getattr(option, "ends_hand", False))


def _resolve_flop(
    node: Node,
    option: Option,
    hand_state: dict[str, Any],
    rng: random.Random,
) -> OptionResolution:
    action = option.meta.get("action") if option.meta else None
    pot = _state_value(hand_state, "pot", node.pot_bb)
    hand_state["street"] = "flop"
    rival_cards = _rival_cards(hand_state)
    hero_cards = node.hero_cards
    board = node.board

    if action == "check":
        _record_rival_adapt(hand_state, aggressive=False)
        hand_state["board_index"] = 4
        _set_street_pot(hand_state, "turn", pot)
        _rebuild_turn_node(hand_state, pot)
        return OptionResolution(note=f"You check back. Pot stays {pot:.2f}bb.")

    if action == "bet":
        bet_size = float(option.meta.get("bet", 0.0))
        precision = _precision_from_meta(option.meta, "flop")
        _apply_contribution(hand_state, "hero", bet_size)
        _record_rival_adapt(hand_state, aggressive=True)
        decision_meta = _decision_meta(option.meta, hand_state)
        decision = rival_strategy.decide_action(decision_meta, rival_cards, rng)
        if rival_cards is not None:
            hero_vs_known = _combo_equity(hero_cards, board, rival_cards, precision)
            threshold = float(option.meta.get("rival_threshold", 1.0))
            if 1.0 - hero_vs_known < threshold:
                decision = rival_strategy.RivalDecision(folds=True)
        if decision.folds:
            _update_rival_range(hand_state, option.meta, True)
            hand_state["hand_over"] = True
            total_pot = _state_value(hand_state, "pot")
            net_gain = total_pot - bet_size
            if rival_cards is None:
                note = f"You bet {bet_size:.2f}bb. Rival folds (hand hidden)."
            else:
                note = f"Rival folds flop. Pot {total_pot:.2f}bb awarded (net +{net_gain:.2f}bb)."
            return OptionResolution(hand_ended=True, note=note)
        call_amount = min(bet_size, _state_value(hand_state, "rival_stack"))
        _apply_contribution(hand_state, "rival", call_amount)
        hand_state["board_index"] = 4
        _set_street_pot(hand_state, "turn", _state_value(hand_state, "pot"))
        _rebuild_turn_node(hand_state, _state_value(hand_state, "pot"))
        _update_rival_range(hand_state, option.meta, False)
        if rival_cards is None:
            return OptionResolution(note=f"You bet {bet_size:.2f}bb. (Rival response hidden)")
        hero_eq = _combo_equity(hero_cards, board, rival_cards, precision)
        equity_note = _fmt_pct(hero_eq, 1)
        return OptionResolution(
            note=f"Rival calls. Pot {_state_value(hand_state, 'pot'):.2f}bb. Your equity {equity_note}."
        )

    if action in {"jam", "allin", "all-in"}:
        risk = float(option.meta.get("risk", _state_value(hand_state, "hero_stack", node.effective_bb)))
        precision = _precision_from_meta(option.meta, "flop")
        _apply_contribution(hand_state, "hero", risk)
        hand_state["hand_over"] = True
        _record_rival_adapt(hand_state, aggressive=True)
        decision_meta = _decision_meta(option.meta, hand_state)
        decision = rival_strategy.decide_action(decision_meta, rival_cards, rng)
        if decision.folds:
            _update_rival_range(hand_state, option.meta, True)
            total_pot = _state_value(hand_state, "pot")
            net_gain = total_pot - risk
            if rival_cards is None:
                note = f"You jam for {risk:.2f}bb. Rival folds (hand hidden)."
                return OptionResolution(hand_ended=True, note=note)
            note = f"Rival folds to your jam. Pot {total_pot:.2f}bb awarded (net +{net_gain:.2f}bb)."
            return OptionResolution(hand_ended=True, note=note)
        call_amount = min(risk, _state_value(hand_state, "rival_stack"))
        _apply_contribution(hand_state, "rival", call_amount)
        _update_rival_range(hand_state, option.meta, False)
        if rival_cards is None:
            return OptionResolution(hand_ended=True, note=f"You jam for {risk:.2f}bb. Rival action hidden.")
        hero_eq = _combo_equity(hero_cards, board, rival_cards, precision)
        rival_text = format_cards_spaced(list(rival_cards))
        equity_note = _fmt_pct(hero_eq, 1)
        return OptionResolution(
            hand_ended=True,
            note=f"Rival calls jam with {rival_text}. Your equity {equity_note}.",
            reveal_rival=True,
        )

    return OptionResolution(hand_ended=getattr(option, "ends_hand", False))


def _resolve_turn(
    node: Node,
    option: Option,
    hand_state: dict[str, Any],
    rng: random.Random,
) -> OptionResolution:
    action = option.meta.get("action") if option.meta else None
    rival_bet = float(option.meta.get("rival_bet", node.context.get("bet", 0.0)))
    if rival_bet > 0:
        _apply_contribution(hand_state, "rival", rival_bet)
        _set_street_pot(hand_state, "turn", _state_value(hand_state, "pot"))
    pot_after_bet = _state_value(hand_state, "pot", node.pot_bb)
    rival_cards = _rival_cards(hand_state)
    hero_cards = node.hero_cards
    board = node.board
    precision = _precision_from_meta(option.meta, "turn")

    if action == "fold":
        hand_state["hand_over"] = True
        note = f"You fold turn. Rival collects {pot_after_bet:.2f}bb."
        _update_rival_range(hand_state, option.meta, True)
        return OptionResolution(hand_ended=True, note=note)

    if action == "call":
        call_amount = min(rival_bet, _state_value(hand_state, "hero_stack"))
        _apply_contribution(hand_state, "hero", call_amount)
        hand_state["street"] = "river"
        hand_state["board_index"] = 5
        _set_street_pot(hand_state, "river", _state_value(hand_state, "pot"))
        _rebuild_river_node(hand_state, _state_value(hand_state, "pot"))
        _update_rival_range(hand_state, option.meta, False)
        _record_rival_adapt(hand_state, aggressive=False)
        return OptionResolution(
            note=f"You call {call_amount:.2f}bb. Pot {_state_value(hand_state, 'pot'):.2f}bb on river."
        )

    if action == "check":
        hand_state["street"] = "river"
        hand_state["board_index"] = 5
        _set_street_pot(hand_state, "river", _state_value(hand_state, "pot"))
        _rebuild_river_node(hand_state, _state_value(hand_state, "pot"))
        _record_rival_adapt(hand_state, aggressive=False)
        return OptionResolution(note=f"You check back. Pot {_state_value(hand_state, 'pot'):.2f}bb.")

    if action == "bet":
        bet_size = float(option.meta.get("bet", 0.0))
        _apply_contribution(hand_state, "hero", bet_size)
        _record_rival_adapt(hand_state, aggressive=True)
        decision_meta = _decision_meta(option.meta, hand_state)
        decision = rival_strategy.decide_action(decision_meta, rival_cards, rng)
        if decision.folds:
            _update_rival_range(hand_state, option.meta, True)
            hand_state["hand_over"] = True
            total_pot = _state_value(hand_state, "pot")
            net_gain = total_pot - bet_size
            if rival_cards is None:
                note = f"You bet {bet_size:.2f}bb. Rival folds (hand hidden)."
            else:
                note = f"Rival folds turn. Pot {total_pot:.2f}bb awarded (net +{net_gain:.2f}bb)."
            return OptionResolution(hand_ended=True, note=note)

        call_amount = min(bet_size, _state_value(hand_state, "rival_stack"))
        _apply_contribution(hand_state, "rival", call_amount)
        hand_state["street"] = "river"
        hand_state["board_index"] = 5
        _set_street_pot(hand_state, "river", _state_value(hand_state, "pot"))
        _rebuild_river_node(hand_state, _state_value(hand_state, "pot"))
        _update_rival_range(hand_state, option.meta, False)
        if rival_cards is None:
            return OptionResolution(note=f"You bet {bet_size:.2f}bb. Rival action hidden.")
        hero_eq = _combo_equity(hero_cards, board, rival_cards, precision)
        equity_note = _fmt_pct(hero_eq, 1)
        return OptionResolution(
            note=f"Rival calls. Pot {_state_value(hand_state, 'pot'):.2f}bb. Your equity {equity_note}."
        )

    if action == "raise":
        raise_to = float(option.meta.get("raise_to", rival_bet * 2.5))
        hero_contrib = _state_value(hand_state, "hero_contrib")
        hero_add = max(0.0, raise_to - hero_contrib)
        _apply_contribution(hand_state, "hero", hero_add)
        _record_rival_adapt(hand_state, aggressive=True)
        decision_meta = _decision_meta(option.meta, hand_state)
        decision = rival_strategy.decide_action(decision_meta, rival_cards, rng)
        if decision.folds:
            _update_rival_range(hand_state, option.meta, True)
            hand_state["hand_over"] = True
            total_pot = _state_value(hand_state, "pot")
            net_gain = total_pot - hero_add
            if rival_cards is None:
                note = f"You raise to {raise_to:.2f}bb. Rival folds (hand hidden)."
            else:
                note = f"Rival folds turn. Pot {total_pot:.2f}bb awarded (net +{net_gain:.2f}bb)."
            return OptionResolution(hand_ended=True, note=note)
        call_amount = max(0.0, raise_to - _state_value(hand_state, "rival_contrib"))
        _apply_contribution(hand_state, "rival", call_amount)
        hand_state["street"] = "river"
        hand_state["board_index"] = 5
        _set_street_pot(hand_state, "river", _state_value(hand_state, "pot"))
        _rebuild_river_node(hand_state, _state_value(hand_state, "pot"))
        _update_rival_range(hand_state, option.meta, False)
        if rival_cards is None:
            return OptionResolution(
                note=f"You raise to {raise_to:.2f}bb. Pot now {_state_value(hand_state, 'pot'):.2f}bb."
            )
        hero_eq = _combo_equity(hero_cards, board, rival_cards, precision)
        equity_note = _fmt_pct(hero_eq, 1)
        return OptionResolution(
            note=f"Rival calls raise. Pot {_state_value(hand_state, 'pot'):.2f}bb. Your equity {equity_note}."
        )

    return OptionResolution(hand_ended=getattr(option, "ends_hand", False))


def _resolve_river(
    node: Node,
    option: Option,
    hand_state: dict[str, Any],
    rng: random.Random,
) -> OptionResolution:
    action = option.meta.get("action") if option.meta else None
    pot = _state_value(hand_state, "pot", node.pot_bb)
    hero_cards = node.hero_cards
    rival_cards = _rival_cards(hand_state)
    board = node.board
    rival_bet = float(option.meta.get("rival_bet", node.context.get("bet", 0.0)))
    if action in {"fold", "call", "raise"} and rival_bet > 0:
        _apply_contribution(hand_state, "rival", rival_bet)
        _set_street_pot(hand_state, "river", _state_value(hand_state, "pot"))

    if action == "fold":
        hand_state["hand_over"] = True
        _update_rival_range(hand_state, option.meta, True)
        total = _state_value(hand_state, "pot")
        return OptionResolution(hand_ended=True, note=f"You fold river. Rival collects {total:.2f}bb.")

    if action == "check":
        hand_state["hand_over"] = True
        hand_state.pop("rival_continue_range", None)
        _record_rival_adapt(hand_state, aggressive=False)
        if rival_cards is None:
            return OptionResolution(hand_ended=True, note=f"Hand checks down. Pot {pot:.2f}bb.")
        outcome = _showdown_outcome(hero_cards, board, rival_cards)
        rival_text = format_cards_spaced(list(rival_cards))
        win_note = f"Showdown win vs {rival_text}. You take {pot:.2f}bb."
        lose_note = f"Showdown loss vs {rival_text}."
        chop_note = f"Showdown chop vs {rival_text}. Pot split."
        if outcome > 0.5:
            return OptionResolution(hand_ended=True, note=win_note, reveal_rival=True)
        if outcome < 0.5:
            return OptionResolution(hand_ended=True, note=lose_note, reveal_rival=True)
        return OptionResolution(hand_ended=True, note=chop_note, reveal_rival=True)

    if action == "call":
        call_amount = min(rival_bet, _state_value(hand_state, "hero_stack"))
        _apply_contribution(hand_state, "hero", call_amount)
        hand_state["hand_over"] = True
        hand_state.pop("rival_continue_range", None)
        _record_rival_adapt(hand_state, aggressive=False)
        if rival_cards is None:
            return OptionResolution(hand_ended=True, note=f"You call {call_amount:.2f}bb. Rival hand hidden.")
        outcome = _showdown_outcome(hero_cards, board, rival_cards)
        rival_text = format_cards_spaced(list(rival_cards))
        total_pot = _state_value(hand_state, "pot")
        if outcome > 0.5:
            return OptionResolution(
                hand_ended=True,
                note=f"You call. Win vs {rival_text} for {total_pot:.2f}bb.",
                reveal_rival=True,
            )
        if outcome < 0.5:
            return OptionResolution(
                hand_ended=True,
                note=f"You call. Lose vs {rival_text}.",
                reveal_rival=True,
            )
        return OptionResolution(
            hand_ended=True,
            note=f"You call. Chop with {rival_text}.",
            reveal_rival=True,
        )

    if action == "raise":
        raise_to = float(option.meta.get("raise_to", rival_bet * 2.5))
        hero_contrib = _state_value(hand_state, "hero_contrib")
        hero_add = max(0.0, raise_to - hero_contrib)
        _apply_contribution(hand_state, "hero", hero_add)
        _record_rival_adapt(hand_state, aggressive=True)
        decision_meta = _decision_meta(option.meta, hand_state)
        decision = rival_strategy.decide_action(decision_meta, rival_cards, rng)
        if decision.folds:
            _update_rival_range(hand_state, option.meta, True)
            hand_state["hand_over"] = True
            total_pot = _state_value(hand_state, "pot")
            net_gain = total_pot - hero_add
            if rival_cards is None:
                note = f"You raise to {raise_to:.2f}bb. Rival folds (hand hidden)."
            else:
                note = f"Rival folds river raise. Pot {total_pot:.2f}bb awarded (net +{net_gain:.2f}bb)."
            return OptionResolution(hand_ended=True, note=note)
        call_amount = max(0.0, raise_to - _state_value(hand_state, "rival_contrib"))
        _apply_contribution(hand_state, "rival", call_amount)
        hand_state["hand_over"] = True
        hand_state.pop("rival_continue_range", None)
        if rival_cards is None:
            return OptionResolution(hand_ended=True, note=f"You raise to {raise_to:.2f}bb. Rival action hidden.")
        rival_text = format_cards_spaced(list(rival_cards))
        total_pot = _state_value(hand_state, "pot")
        outcome = _showdown_outcome(hero_cards, board, rival_cards)
        if outcome > 0.5:
            return OptionResolution(
                hand_ended=True,
                note=f"Rival calls raise with {rival_text}. You win {total_pot:.2f}bb.",
                reveal_rival=True,
            )
        if outcome < 0.5:
            return OptionResolution(
                hand_ended=True,
                note=f"Rival calls raise with {rival_text}. You lose.",
                reveal_rival=True,
            )
        return OptionResolution(
            hand_ended=True,
            note=f"Rival calls raise with {rival_text}. Pot split.",
            reveal_rival=True,
        )

    if action == "bet":
        bet_size = float(option.meta.get("bet", 0.0))
        _apply_contribution(hand_state, "hero", bet_size)
        _record_rival_adapt(hand_state, aggressive=True)
        decision_meta = _decision_meta(option.meta, hand_state)
        decision = rival_strategy.decide_action(decision_meta, rival_cards, rng)
        if decision.folds:
            _update_rival_range(hand_state, option.meta, True)
            hand_state["hand_over"] = True
            total_pot = _state_value(hand_state, "pot")
            net_gain = total_pot - bet_size
            if rival_cards is None:
                note = f"You bet {bet_size:.2f}bb. Rival folds (hand hidden)."
            else:
                note = f"Rival folds river. Pot {total_pot:.2f}bb awarded (net +{net_gain:.2f}bb)."
            return OptionResolution(hand_ended=True, note=note)
        call_amount = min(bet_size, _state_value(hand_state, "rival_stack"))
        _apply_contribution(hand_state, "rival", call_amount)
        hand_state["hand_over"] = True
        hand_state.pop("rival_continue_range", None)
        if rival_cards is None:
            return OptionResolution(hand_ended=True, note=f"You bet {bet_size:.2f}bb. Rival action hidden.")
        outcome = _showdown_outcome(hero_cards, board, rival_cards)
        rival_text = format_cards_spaced(list(rival_cards))
        total_pot = _state_value(hand_state, "pot")
        win_note = f"Rival calls with {rival_text}. You win {total_pot:.2f}bb."
        lose_note = f"Rival calls with {rival_text}. You lose."
        chop_note = f"Rival calls with {rival_text}. Pot split."
        if outcome > 0.5:
            return OptionResolution(hand_ended=True, note=win_note, reveal_rival=True)
        if outcome < 0.5:
            return OptionResolution(hand_ended=True, note=lose_note, reveal_rival=True)
        return OptionResolution(hand_ended=True, note=chop_note, reveal_rival=True)

    if action in {"jam", "allin", "all-in"}:
        risk = float(option.meta.get("risk", _state_value(hand_state, "hero_stack", node.effective_bb)))
        _apply_contribution(hand_state, "hero", risk)
        hand_state["hand_over"] = True
        _record_rival_adapt(hand_state, aggressive=True)
        decision_meta = _decision_meta(option.meta, hand_state)
        decision = rival_strategy.decide_action(decision_meta, rival_cards, rng)
        if decision.folds:
            _update_rival_range(hand_state, option.meta, True)
            total_pot = _state_value(hand_state, "pot")
            net_gain = total_pot - risk
            if rival_cards is None:
                note = f"You jam river for {risk:.2f}bb. Rival folds (hand hidden)."
                return OptionResolution(hand_ended=True, note=note)
            note = f"Rival folds river jam. Pot {total_pot:.2f}bb awarded (net +{net_gain:.2f}bb)."
            return OptionResolution(hand_ended=True, note=note, reveal_rival=True)
        call_amount = min(risk, _state_value(hand_state, "rival_stack"))
        _apply_contribution(hand_state, "rival", call_amount)
        hand_state.pop("rival_continue_range", None)
        if rival_cards is None:
            return OptionResolution(hand_ended=True, note=f"You jam river for {risk:.2f}bb. Rival action hidden.")
        rival_text = format_cards_spaced(list(rival_cards))
        outcome = _showdown_outcome(hero_cards, board, rival_cards)
        total_pot = _state_value(hand_state, "pot")
        if outcome > 0.5:
            return OptionResolution(
                hand_ended=True,
                note=f"Rival calls jam with {rival_text}. You win {total_pot:.2f}bb.",
                reveal_rival=True,
            )
        if outcome < 0.5:
            return OptionResolution(
                hand_ended=True,
                note=f"Rival calls jam with {rival_text}. You lose.",
                reveal_rival=True,
            )
        return OptionResolution(
            hand_ended=True,
            note=f"Rival calls jam with {rival_text}. Pot split.",
            reveal_rival=True,
        )

    return OptionResolution(hand_ended=getattr(option, "ends_hand", False))


BET_SIZING = BetSizingManager()
