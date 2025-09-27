from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import math

from ..dynamic.cards import str_to_int


@dataclass(slots=True)
class RangeLoaderConfig:
    """Configuration for precomputed range interpolation."""

    resource: Path


class RangeRepository:
    """Load and interpolate precomputed hand ranges for the trainer."""

    def __init__(self, config: RangeLoaderConfig | None = None) -> None:
        resource = config.resource if config else Path(__file__).with_name("ranges") / "heads_up_ranges.json"
        self._config = RangeLoaderConfig(resource=resource)
        self._payload = self._load_resource(resource)

    @staticmethod
    def _load_resource(path: Path) -> dict[str, dict[str, dict[str, float]]]:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("Invalid range data payload")
        return data

    def range_for(
        self,
        range_id: str,
        sizing: float,
        blocked_cards: Iterable[int] | None = None,
    ) -> tuple[list[tuple[int, int]], dict[tuple[int, int], float] | None]:
        profiles = self._payload.get(range_id)
        if not profiles:
            return [], None
        combos, weights = self._interpolate_profiles(profiles, sizing)
        if not combos:
            return [], None
        blocked = set(blocked_cards or [])
        filtered: list[tuple[int, int]] = []
        filtered_weights: dict[tuple[int, int], float] = {}
        for combo, weight in zip(combos, weights, strict=False):
            if combo[0] in blocked or combo[1] in blocked or weight <= 0:
                continue
            filtered.append(combo)
            filtered_weights[combo] = weight
        if not filtered:
            return [], None
        total = sum(filtered_weights.values())
        if total > 0:
            filtered_weights = {combo: value / total for combo, value in filtered_weights.items() if value > 0}
        else:
            filtered_weights = {}
        filtered.sort(key=lambda combo: filtered_weights.get(combo, 0.0), reverse=True)
        return filtered, filtered_weights or None

    def raw_profiles(self, range_id: str) -> dict[str, dict[str, float]]:
        data = self._payload.get(range_id)
        if not data:
            return {}
        return {key: dict(value) for key, value in data.items()}

    @staticmethod
    def _interpolate_profiles(
        profiles: dict[str, dict[str, float]],
        sizing: float,
    ) -> tuple[list[tuple[int, int]], list[float]]:
        available = sorted(float(key) for key in profiles.keys())
        if not available:
            return [], []
        if sizing <= available[0]:
            mapping = profiles[f"{available[0]:.1f}"]
            return _decode_range(mapping)
        if sizing >= available[-1]:
            mapping = profiles[f"{available[-1]:.1f}"]
            return _decode_range(mapping)
        low = max(value for value in available if value <= sizing)
        high = min(value for value in available if value >= sizing)
        if math.isclose(low, high):
            mapping = profiles[f"{low:.1f}"]
            return _decode_range(mapping)
        lower_map = profiles[f"{low:.1f}"]
        upper_map = profiles[f"{high:.1f}"]
        t = (sizing - low) / (high - low)
        blended: dict[str, float] = {}
        all_keys = set(lower_map.keys()) | set(upper_map.keys())
        for key in all_keys:
            lower = float(lower_map.get(key, 0.0))
            upper = float(upper_map.get(key, 0.0))
            weight = (1 - t) * lower + t * upper
            if weight > 0:
                blended[key] = weight
        _boost_endpoints(blended, lower_map)
        _boost_endpoints(blended, upper_map)
        return _decode_range(blended)


def _boost_endpoints(blended: dict[str, float], endpoint: dict[str, float]) -> None:
    if not endpoint:
        return
    ordered = sorted(endpoint.items(), key=lambda item: float(item[1]), reverse=True)[:20]
    boost = 1e-3
    length = len(ordered)
    if length == 0:
        return
    for rank, (key, _value) in enumerate(ordered):
        blended[key] = blended.get(key, 0.0) + boost * (length - rank)


def _decode_range(payload: dict[str, float]) -> tuple[list[tuple[int, int]], list[float]]:
    combos: list[tuple[int, int]] = []
    weights: list[float] = []
    for key, value in payload.items():
        if len(key) != 4:
            continue
        try:
            a = str_to_int(key[:2])
            b = str_to_int(key[2:])
        except ValueError:
            continue
        combo = (a, b) if a < b else (b, a)
        combos.append(combo)
        weights.append(float(value))
    ordered = sorted(zip(combos, weights, strict=False), key=lambda item: item[1], reverse=True)
    sorted_combos = [combo for combo, _ in ordered]
    sorted_weights = [weight for _, weight in ordered]
    return sorted_combos, sorted_weights


_REPOSITORY: Optional[RangeRepository] = None
_REPOSITORY_STAMP: Optional[float] = None


def get_repository() -> RangeRepository:
    """Return a repository instance using the default payload, reloading on change."""

    global _REPOSITORY, _REPOSITORY_STAMP
    default_path = Path(__file__).with_name("ranges") / "heads_up_ranges.json"
    stamp = default_path.stat().st_mtime
    if _REPOSITORY is None or _REPOSITORY_STAMP != stamp:
        _REPOSITORY = RangeRepository(RangeLoaderConfig(resource=default_path))
        _REPOSITORY_STAMP = stamp
    return _REPOSITORY
