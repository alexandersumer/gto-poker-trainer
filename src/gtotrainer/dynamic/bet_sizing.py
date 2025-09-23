from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(slots=True)
class BetSizingConfig:
    """Configuration knobs for dynamic bet sizing refinement."""

    min_increment: float = 0.25
    usage_decay: float = 0.82
    usage_floor: float = 0.01
    usage_drop_threshold: float = 0.012
    regret_expand_threshold: float = 0.45
    postflop_regret_expand_threshold: float = 0.3
    preflop_base_multipliers: tuple[float, ...] = (2.8, 3.5, 5.0)
    preflop_min_count: int = 2
    preflop_max_count: int = 6
    postflop_max_count: int = 5


@dataclass(slots=True)
class _PreflopState:
    sizes: list[float]
    protected: set[float]
    usage: dict[float, float] = field(default_factory=dict)
    regrets: dict[float, float] = field(default_factory=dict)

    def normalised_sizes(self, hero_contrib: float, hero_stack: float, config: BetSizingConfig) -> list[float]:
        cap = hero_contrib + hero_stack
        trimmed = []
        for size in sorted(set(self.sizes)):
            if size <= hero_contrib + config.min_increment:
                continue
            if size > cap + 1e-6:
                continue
            trimmed.append(round(size, 2))
        if not trimmed:
            fallback = min(cap, hero_contrib + max(config.min_increment, hero_stack * 0.65))
            trimmed.append(round(fallback, 2))
        return sorted(dict.fromkeys(trimmed))

    def observe(
        self,
        *,
        hero_contrib: float,
        hero_stack: float,
        observations: list[tuple[float, float, float]],
        config: BetSizingConfig,
    ) -> None:
        if not observations:
            return
        cap = hero_contrib + hero_stack
        max_regret = float("-inf")
        max_size = None

        for size, freq, regret in observations:
            size = round(min(max(size, hero_contrib + config.min_increment), cap), 2)
            self.usage[size] = self.usage.get(size, config.usage_floor) * config.usage_decay + max(freq, 0.0)
            self.regrets[size] = regret
            if regret > max_regret:
                max_regret = regret
                max_size = size
            if size not in self.sizes:
                self.sizes.append(size)

        if (
            max_size is not None
            and max_regret > config.regret_expand_threshold
            and len(self.sizes) < config.preflop_max_count
        ):
            candidate = _interpolate_preflop_size(max_size, self.sizes, hero_contrib, cap, config)
            if candidate is not None:
                self.sizes.append(candidate)
                self.usage.setdefault(candidate, config.usage_floor)
                self.regrets.setdefault(candidate, 0.0)

        # Drop rarely used, high-stack sizes (but keep protected baselines).
        if len(self.sizes) > config.preflop_min_count:
            removable = [
                size
                for size in self.sizes
                if size not in self.protected and self.usage.get(size, 0.0) < config.usage_drop_threshold
            ]
            removable.sort(key=lambda size: self.usage.get(size, 0.0))
            for size in removable:
                if len(self.sizes) <= config.preflop_min_count:
                    break
                self.sizes.remove(size)
                self.usage.pop(size, None)
                self.regrets.pop(size, None)


@dataclass(slots=True)
class _PostflopState:
    fractions: list[float]
    protected: set[float]
    usage: dict[float, float] = field(default_factory=dict)
    regrets: dict[float, float] = field(default_factory=dict)

    def fractions_for(self, config: BetSizingConfig) -> list[float]:
        trimmed = [fraction for fraction in sorted(set(self.fractions)) if fraction > 0]
        if not trimmed:
            trimmed = [0.5]
        return trimmed[: config.postflop_max_count]

    def observe(self, observations: list[tuple[float, float, float]], config: BetSizingConfig) -> None:
        if not observations:
            return
        max_regret = float("-inf")
        max_fraction = None
        for fraction, freq, regret in observations:
            fraction = max(0.05, min(3.0, round(fraction, 3)))
            self.usage[fraction] = self.usage.get(fraction, config.usage_floor) * config.usage_decay + max(freq, 0.0)
            self.regrets[fraction] = regret
            if regret > max_regret:
                max_regret = regret
                max_fraction = fraction
            if fraction not in self.fractions:
                self.fractions.append(fraction)
        if (
            max_fraction is not None
            and max_regret > config.postflop_regret_expand_threshold
            and len(self.fractions) < config.postflop_max_count
        ):
            candidate = _interpolate_fraction(max_fraction, self.fractions)
            if candidate is not None:
                self.fractions.append(candidate)
                self.usage.setdefault(candidate, config.usage_floor)
                self.regrets.setdefault(candidate, 0.0)

        if len(self.fractions) > 1:
            removable = [
                fraction
                for fraction in self.fractions
                if fraction not in self.protected and self.usage.get(fraction, 0.0) < config.usage_drop_threshold
            ]
            for fraction in removable:
                if len(self.fractions) <= 1:
                    break
                self.fractions.remove(fraction)
                self.usage.pop(fraction, None)
                self.regrets.pop(fraction, None)


class BetSizingManager:
    """Track CFR feedback to dynamically expand or collapse bet sizing branches."""

    def __init__(self, config: BetSizingConfig | None = None) -> None:
        self.config = config or BetSizingConfig()
        self._preflop_states: dict[tuple[float, float], _PreflopState] = {}
        self._postflop_states: dict[tuple[str, str], _PostflopState] = {}

    # ------------------------------------------------------------------
    # Preflop

    def preflop_raise_sizes(
        self,
        *,
        open_size: float,
        hero_contrib: float,
        hero_stack: float,
        rival_stack: float,
    ) -> list[float]:
        key = (_bucket(open_size, 0.1), _bucket(min(hero_stack, rival_stack), 5.0))
        state = self._preflop_states.get(key)
        if state is None:
            sizes = _initial_preflop_sizes(
                open_size=open_size,
                hero_contrib=hero_contrib,
                hero_stack=hero_stack,
                rival_stack=rival_stack,
                config=self.config,
            )
            state = _PreflopState(sizes=sizes, protected=set(sizes))
            self._preflop_states[key] = state
        return state.normalised_sizes(hero_contrib, hero_stack, self.config)

    def observe_preflop(
        self,
        *,
        open_size: float,
        hero_contrib: float,
        hero_stack: float,
        rival_stack: float,
        observations: list[tuple[float, float, float]],
    ) -> None:
        if not observations:
            return
        key = (_bucket(open_size, 0.1), _bucket(min(hero_stack, rival_stack), 5.0))
        state = self._preflop_states.get(key)
        if state is None:
            return
        state.observe(
            hero_contrib=hero_contrib,
            hero_stack=hero_stack,
            observations=observations,
            config=self.config,
        )

    # ------------------------------------------------------------------
    # Postflop

    def postflop_bet_fractions(
        self,
        *,
        street: str,
        context: str,
        base_fractions: Iterable[float],
    ) -> tuple[float, ...]:
        key = (street, context)
        state = self._postflop_states.get(key)
        if state is None:
            base = [max(0.05, float(frac)) for frac in base_fractions if float(frac) > 0]
            if not base:
                base = [0.5]
            state = _PostflopState(fractions=base[: self.config.postflop_max_count], protected=set(base))
            self._postflop_states[key] = state
        return tuple(state.fractions_for(self.config))

    def observe_postflop(
        self,
        *,
        street: str,
        context: str,
        observations: list[tuple[float, float, float]],
    ) -> None:
        if not observations:
            return
        key = (street, context)
        state = self._postflop_states.get(key)
        if state is None:
            return
        state.observe(observations, self.config)


# ----------------------------------------------------------------------
# Helpers


def _initial_preflop_sizes(
    *,
    open_size: float,
    hero_contrib: float,
    hero_stack: float,
    rival_stack: float,
    config: BetSizingConfig,
) -> list[float]:
    cap = hero_contrib + min(hero_stack, rival_stack)
    sizes: list[float] = []
    for mult in config.preflop_base_multipliers:
        size = round(open_size * mult, 2)
        if size <= hero_contrib + config.min_increment:
            continue
        size = min(size, cap)
        if size > hero_contrib + config.min_increment:
            sizes.append(size)
    if not sizes and cap > hero_contrib + config.min_increment:
        sizes.append(round(min(cap, hero_contrib + max(config.min_increment, hero_stack * 0.6)), 2))
    return sorted(dict.fromkeys(sizes))


def _interpolate_preflop_size(
    anchor: float,
    sizes: list[float],
    hero_contrib: float,
    cap: float,
    config: BetSizingConfig,
) -> float | None:
    ordered = sorted(set(sizes))
    try:
        idx = ordered.index(anchor)
    except ValueError:
        return None
    lower_bound = hero_contrib + config.min_increment
    upper_bound = cap
    if idx > 0:
        lower_bound = max(lower_bound, ordered[idx - 1] + config.min_increment)
    if idx + 1 < len(ordered):
        upper_bound = min(upper_bound, ordered[idx + 1] - config.min_increment)
    if upper_bound - lower_bound <= config.min_increment:
        return None
    midpoint = (upper_bound + lower_bound) / 2
    candidate = round(midpoint, 2)
    if candidate <= hero_contrib + config.min_increment or candidate >= cap - config.min_increment / 2:
        return None
    if candidate in ordered:
        candidate += config.min_increment
    return round(min(max(candidate, hero_contrib + config.min_increment), cap), 2)


def _interpolate_fraction(anchor: float, fractions: list[float]) -> float | None:
    ordered = sorted(set(fractions))
    try:
        idx = ordered.index(anchor)
    except ValueError:
        return None
    lower = ordered[idx - 1] if idx > 0 else max(0.05, anchor * 0.5)
    upper = ordered[idx + 1] if idx + 1 < len(ordered) else min(3.0, anchor * 1.5)
    if upper - lower <= 0.05:
        return None
    candidate = round((upper + lower) / 2, 3)
    if candidate in ordered:
        candidate += 0.05
    return round(min(max(candidate, 0.05), 3.0), 3)


def _bucket(value: float, step: float) -> float:
    if step <= 0:
        return round(value, 2)
    return round(round(value / step) * step, 2)
