from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..core.models import Option

if TYPE_CHECKING:  # pragma: no cover
    from .episode import Node


@dataclass(slots=True)
class LinearCFRConfig:
    """Configuration for discounted/linear CFR on normal-form subgames."""

    iterations: int = 600
    minimum_actions: int = 2
    extra_iterations_per_action: int = 220
    linear_weight_pow: float = 1.5
    regret_floor: float = 1e-9


@dataclass(slots=True)
class _SubgamePayload:
    hero_payoff: np.ndarray
    rival_payoff: np.ndarray
    rival_labels: tuple[str, ...]


class LinearCFRBackend:
    """Run discounted CFR with linear weighting on local betting subgames."""

    def __init__(self, config: LinearCFRConfig | None = None) -> None:
        self.config = config or LinearCFRConfig()
        self.name = "linear_cfr_v1"

    def refine(self, node: Node | None, options: list[Option]) -> list[Option]:
        del node  # reserved for future tree-aware refinements
        eligible = [(idx, opt) for idx, opt in enumerate(options) if _supports_cfr(opt)]
        if len(eligible) < self.config.minimum_actions:
            return options

        payload = _build_payload([opt for _, opt in eligible])
        if payload is None:
            return options

        iterations = _iteration_budget(self.config, len(eligible), len(payload.rival_labels))
        solver = _LinearCFR(payload.hero_payoff, payload.rival_payoff, config=self.config)
        stats = solver.solve(iterations)

        adjusted_values = payload.hero_payoff @ stats.rival_avg

        for (idx, option), action_value, hero_prob, regret in zip(
            eligible,
            adjusted_values,
            stats.hero_avg,
            stats.hero_regret,
            strict=False,
        ):
            meta = option.meta or {}
            meta.setdefault("baseline_ev", option.ev)
            meta["cfr_backend"] = self.name
            meta["cfr_probability"] = float(hero_prob)
            meta["cfr_iterations"] = iterations
            meta["cfr_avg_ev"] = float(action_value)
            meta["cfr_regret"] = float(regret)
            meta["cfr_max_regret"] = float(np.max(stats.hero_regret))
            meta["cfr_max_rival_regret"] = float(np.max(stats.rival_regret))
            meta["cfr_rival_mix"] = {
                str(action): float(prob) for action, prob in zip(payload.rival_labels, stats.rival_avg, strict=False)
            }
            option.meta = meta
            option.gto_freq = float(hero_prob)
            option.ev = float(action_value)
            options[idx] = option

        return options


def refine_options(node: Node | None, options: list[Option]) -> list[Option]:
    return _SOLVER.refine(node, options)


_SOLVER = LinearCFRBackend()


@dataclass(slots=True)
class _LinearCFRStats:
    hero_avg: np.ndarray
    rival_avg: np.ndarray
    hero_regret: np.ndarray
    rival_regret: np.ndarray


class _LinearCFR:
    """Linear CFR on a normal-form, zero-sum payoff matrix."""

    def __init__(
        self,
        hero_payoff: np.ndarray,
        rival_payoff: np.ndarray,
        *,
        config: LinearCFRConfig,
    ) -> None:
        self.hero_payoff = hero_payoff
        self.rival_payoff = rival_payoff.T
        self.config = config

        num_hero_actions = hero_payoff.shape[0]
        num_rival_actions = hero_payoff.shape[1]

        self._hero_regret = np.zeros(num_hero_actions, dtype=np.float64)
        self._rival_regret = np.zeros(num_rival_actions, dtype=np.float64)
        self._hero_strategy_sum = np.zeros(num_hero_actions, dtype=np.float64)
        self._rival_strategy_sum = np.zeros(num_rival_actions, dtype=np.float64)

    def solve(self, iterations: int) -> _LinearCFRStats:
        payoff = self.hero_payoff
        rival_payoff = self.rival_payoff
        cfg = self.config

        for t in range(1, iterations + 1):
            weight = float(t) ** cfg.linear_weight_pow

            hero_strategy = _regret_matching_plus(self._hero_regret, cfg.regret_floor)
            rival_strategy = _regret_matching_plus(self._rival_regret, cfg.regret_floor)

            self._hero_strategy_sum += weight * hero_strategy
            self._rival_strategy_sum += weight * rival_strategy

            hero_util = payoff @ rival_strategy
            hero_ev = float(hero_strategy @ hero_util)
            hero_regret_delta = hero_util - hero_ev
            self._hero_regret += weight * hero_regret_delta

            rival_util = rival_payoff @ hero_strategy
            rival_ev = float(rival_strategy @ rival_util)
            rival_regret_delta = rival_util - rival_ev
            self._rival_regret += weight * rival_regret_delta

        hero_avg = _normalise_strategy(self._hero_strategy_sum, cfg.regret_floor)
        rival_avg = _normalise_strategy(self._rival_strategy_sum, cfg.regret_floor)

        return _LinearCFRStats(
            hero_avg=hero_avg,
            rival_avg=rival_avg,
            hero_regret=self._hero_regret.copy(),
            rival_regret=self._rival_regret.copy(),
        )


def _supports_cfr(option: Option) -> bool:
    meta = option.meta or {}
    if not meta.get("supports_cfr"):
        return False
    if "cfr_payoffs" in meta:
        return True
    return {"hero_ev_fold", "hero_ev_continue"}.issubset(meta.keys())


def _iteration_budget(
    config: LinearCFRConfig,
    hero_actions: int,
    rival_actions: int,
) -> int:
    extra_actions = max(0, hero_actions - config.minimum_actions)
    extra_rival = max(0, rival_actions - config.minimum_actions)
    base = config.iterations + extra_actions * config.extra_iterations_per_action
    base += extra_rival * (config.extra_iterations_per_action // 2)
    return max(base, 1)


def _build_payload(options: list[Option]) -> _SubgamePayload | None:
    hero_matrix: list[list[float]] = []
    rival_matrix: list[list[float]] = []
    labels_order: list[str] = []

    for option in options:
        meta = option.meta or {}
        payoffs = meta.get("cfr_payoffs")
        if isinstance(payoffs, dict):
            hero_row, rival_row, labels = _row_from_payoff_dict(payoffs)
        else:
            hero_row, rival_row, labels = _row_from_fallback(meta)
        if hero_row is None or rival_row is None or labels is None:
            return None
        hero_matrix.append(hero_row)
        rival_matrix.append(rival_row)
        for label in labels:
            if label not in labels_order:
                labels_order.append(label)

    if not hero_matrix or not labels_order:
        return None

    reordered_hero: list[list[float]] = []
    reordered_rival: list[list[float]] = []
    for hero_row, rival_row in zip(hero_matrix, rival_matrix, strict=False):
        hero_map = dict(zip(labels_order, hero_row))
        rival_map = dict(zip(labels_order, rival_row))
        reordered_hero.append([hero_map[label] for label in labels_order])
        reordered_rival.append([rival_map[label] for label in labels_order])

    hero_array = np.array(reordered_hero, dtype=np.float64)
    rival_array = np.array(reordered_rival, dtype=np.float64)

    return _SubgamePayload(
        hero_payoff=hero_array,
        rival_payoff=rival_array,
        rival_labels=tuple(labels_order),
    )


def _row_from_payoff_dict(payoffs: dict) -> tuple[list[float] | None, list[float] | None, list[str] | None]:
    raw_labels = payoffs.get("rival_actions")
    hero_values = payoffs.get("hero")
    rival_values = payoffs.get("rival")
    if not isinstance(raw_labels, (list, tuple)):
        return None, None, None
    if not isinstance(hero_values, (list, tuple)):
        return None, None, None
    if rival_values is None:
        rival_values = [-float(value) for value in hero_values]
    if not isinstance(rival_values, (list, tuple)):
        return None, None, None
    labels = [str(label) for label in raw_labels]
    try:
        hero_row = [float(value) for value in hero_values]
        rival_row = [float(value) for value in rival_values]
    except (TypeError, ValueError):
        return None, None, None
    if len(hero_row) != len(labels) or len(rival_row) != len(labels):
        return None, None, None
    if any(not math.isfinite(v) for v in hero_row + rival_row):
        return None, None, None
    return hero_row, rival_row, labels


def _row_from_fallback(meta: dict) -> tuple[list[float] | None, list[float] | None, list[str] | None]:
    try:
        hero_fold = float(meta["hero_ev_fold"])
        hero_continue = float(meta["hero_ev_continue"])
        rival_fold = float(meta.get("rival_ev_fold", -hero_fold))
        rival_continue = float(meta.get("rival_ev_continue", -hero_continue))
    except (KeyError, TypeError, ValueError):
        return None, None, None
    if not all(math.isfinite(value) for value in (hero_fold, hero_continue, rival_fold, rival_continue)):
        return None, None, None
    hero_row = [hero_fold, hero_continue]
    rival_row = [rival_fold, rival_continue]
    labels = ["fold", "continue"]
    return hero_row, rival_row, labels


def _regret_matching_plus(regrets: np.ndarray, floor: float) -> np.ndarray:
    positives = np.maximum(regrets, 0.0)
    total = positives.sum()
    if total <= floor:
        return np.full(positives.shape, 1.0 / positives.size, dtype=np.float64)
    return positives / total


def _normalise_strategy(strategy_sum: np.ndarray, floor: float) -> np.ndarray:
    total = strategy_sum.sum()
    if total <= floor:
        return np.full(strategy_sum.shape, 1.0 / strategy_sum.size, dtype=np.float64)
    return strategy_sum / total
