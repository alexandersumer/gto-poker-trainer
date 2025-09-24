"""Range sampling helpers shared across policy generation routines.

These utilities centralise combo categorisation and weighted sampling so the
policy module and future callers can share consistent behaviour.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "combo_category",
    "normalize_combo",
    "evenly_sample_indexed",
    "weighted_sample",
    "sample_range",
    "subset_weights",
    "weighted_average",
    "equity_with_weights",
    "top_weight_fraction",
    "weighted_equity",
]

logger = logging.getLogger(__name__)


def combo_category(combo: Sequence[int]) -> str:
    """Return the combo classification used in solver outputs."""

    a, b = int(combo[0]), int(combo[1])
    if a // 4 == b // 4:
        return "pair"
    if a % 4 == b % 4:
        return "suited"
    return "offsuit"


def normalize_combo(combo: Iterable[int] | Sequence[int]) -> tuple[int, int]:
    try:
        a, b = int(combo[0]), int(combo[1])  # type: ignore[index]
    except (TypeError, ValueError, IndexError) as exc:  # pragma: no cover - defensive
        raise ValueError("combo must contain two card indices") from exc
    if a > b:
        a, b = b, a
    return a, b


def evenly_sample_indexed(entries: list[tuple[int, tuple[int, int]]], count: int) -> list[tuple[int, tuple[int, int]]]:
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


def weighted_sample(
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
        combo = normalize_combo(entry[1])
        return max(0.0, float(weights.get(combo, 0.0)))

    for _ in range(min(count, len(pool))):
        totals = [entry_weight(entry) for entry in pool]
        weight_sum = sum(totals)
        if weight_sum <= 0.0:
            logger.debug("Weight sum <= 0 detected; falling back to uniform sample.")
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


def sample_range(
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
        buckets[combo_category(combo)].append((idx, combo))

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
        selected.extend(weighted_sample(entries, take, weights, local_rng))

    selected.sort(key=lambda item: item[0])
    if len(selected) > limit:
        selected = selected[:limit]

    return [combo for _, combo in selected]


def subset_weights(
    weights: Mapping[tuple[int, int], float] | None,
    combos: Iterable[tuple[int, int]],
) -> dict[tuple[int, int], float] | None:
    if not weights:
        return None
    subset: dict[tuple[int, int], float] = {}
    for combo in combos:
        normalized = normalize_combo(combo)
        weight = weights.get(normalized, 0.0)
        if weight > 0:
            subset[normalized] = weight
    if not subset:
        return None
    total = sum(subset.values())
    if total <= 0:
        logger.debug("Subset weights sum to zero; ignoring weight map.")
        return None
    scale = 1.0 / total
    return {combo: weight * scale for combo, weight in subset.items()}


def weighted_average(
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
        logger.debug("Total weight zero; falling back to simple average.")
        return float(sum(values.values()) / len(values))
    return float(weighted_sum / total_weight)


def equity_with_weights(
    values: Mapping[tuple[int, int], float],
    weights: Mapping[tuple[int, int], float] | None,
) -> list[float | tuple[float, float]]:
    if not weights:
        return list(values.values())
    return [(values[combo], weights.get(combo, 0.0)) for combo in values]


def top_weight_fraction(
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
        logger.debug("Total weight <= 0 when extracting top fraction.")
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
        logger.debug("Selected weight <= 0 after filtering top fraction.")
        return None, 0.0
    scale = 1.0 / selected_total
    normalized = {combo: weight * scale for combo, weight in selected.items()}
    return normalized, min(1.0, selected_total)


def weighted_equity(
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
        logger.debug("Denominator zero in weighted_equity; falling back to simple average.")
        return sum(equities.values()) / len(equities)
    return numerator / denominator
