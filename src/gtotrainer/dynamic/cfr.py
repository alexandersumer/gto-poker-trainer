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
class LocalCFRConfig:
    iterations: int = 200
    minimum_actions: int = 2
    extra_iterations_per_action: int = 120


class LocalCFRBackend:
    """Run small-form counterfactual regret minimisation on local betting spots."""

    def __init__(self, config: LocalCFRConfig | None = None) -> None:
        self.config = config or LocalCFRConfig()
        self.name = "local_cfr_v1"

    def refine(self, node: Node | None, options: list[Option]) -> list[Option]:
        del node  # reserved for future tree-aware refinements
        eligible = [(idx, opt) for idx, opt in enumerate(options) if _supports_cfr(opt)]
        if len(eligible) < self.config.minimum_actions:
            return options

        fold_continue_payoffs = _extract_payoffs([opt for _, opt in eligible])
        if fold_continue_payoffs is None:
            return options

        payoff_matrix = fold_continue_payoffs
        num_actions = payoff_matrix.shape[0]
        hero_regret = np.zeros(num_actions, dtype=np.float64)
        hero_strategy_sum = np.zeros(num_actions, dtype=np.float64)
        villain_regret = np.zeros(2, dtype=np.float64)
        villain_strategy_sum = np.zeros(2, dtype=np.float64)

        extra_actions = max(0, num_actions - self.config.minimum_actions)
        iterations = max(
            self.config.iterations + extra_actions * self.config.extra_iterations_per_action,
            1,
        )

        for _ in range(iterations):
            hero_strategy = _regret_matching(hero_regret)
            villain_strategy = _regret_matching(villain_regret)

            hero_strategy_sum += hero_strategy
            villain_strategy_sum += villain_strategy

            hero_util = payoff_matrix @ villain_strategy
            hero_expected = float(hero_strategy @ hero_util)
            hero_regret += hero_util - hero_expected

            villain_payoff = (-(payoff_matrix.T)) @ hero_strategy
            villain_expected = float(villain_strategy @ villain_payoff)
            villain_regret += villain_payoff - villain_expected

        hero_avg = _normalise_strategy(hero_strategy_sum)
        villain_avg = _normalise_strategy(villain_strategy_sum)
        adjusted_values = payoff_matrix @ villain_avg

        for (idx, option), action_value, hero_prob in zip(eligible, adjusted_values, hero_avg, strict=False):
            meta = option.meta or {}
            meta.setdefault("baseline_ev", option.ev)
            meta["cfr_backend"] = self.name
            meta["cfr_probability"] = float(hero_prob)
            meta["cfr_villain_mix"] = {
                "fold": float(villain_avg[0]),
                "continue": float(villain_avg[1]),
            }
            option.meta = meta
            option.gto_freq = float(hero_prob)
            option.ev = float(action_value)
            options[idx] = option

        return options


def refine_options(node: Node | None, options: list[Option]) -> list[Option]:
    return _SOLVER.refine(node, options)


_SOLVER = LocalCFRBackend()


def _supports_cfr(option: Option) -> bool:
    meta = option.meta or {}
    return bool(meta.get("supports_cfr")) and {
        "hero_ev_fold",
        "hero_ev_continue",
    }.issubset(meta.keys())


def _extract_payoffs(options: Iterable[Option]) -> np.ndarray | None:
    matrix_rows: list[list[float]] = []
    for option in options:
        meta = option.meta or {}
        try:
            fold_ev = float(meta["hero_ev_fold"])
            continue_ev = float(meta["hero_ev_continue"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(fold_ev) or not math.isfinite(continue_ev):
            return None
        matrix_rows.append([fold_ev, continue_ev])
    if not matrix_rows:
        return None
    return np.array(matrix_rows, dtype=np.float64)


def _regret_matching(regrets: np.ndarray) -> np.ndarray:
    positives = np.maximum(regrets, 0.0)
    total = positives.sum()
    if total <= 1e-12:
        return np.full_like(positives, 1.0 / positives.size)
    return positives / total


def _normalise_strategy(strategy_sum: np.ndarray) -> np.ndarray:
    total = strategy_sum.sum()
    if total <= 1e-12:
        return np.full_like(strategy_sum, 1.0 / strategy_sum.size)
    return strategy_sum / total
