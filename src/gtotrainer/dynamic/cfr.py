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

        payoff_matrix, rival_actions = _extract_payoffs([opt for _, opt in eligible])
        if payoff_matrix is None or rival_actions is None:
            return options

        num_actions = payoff_matrix.shape[0]
        num_rival_actions = payoff_matrix.shape[1]
        hero_regret = np.zeros(num_actions, dtype=np.float64)
        hero_strategy_sum = np.zeros(num_actions, dtype=np.float64)
        rival_regret = np.zeros(num_rival_actions, dtype=np.float64)
        rival_strategy_sum = np.zeros(num_rival_actions, dtype=np.float64)

        extra_actions = max(0, num_actions - self.config.minimum_actions)
        iterations = max(
            self.config.iterations + extra_actions * self.config.extra_iterations_per_action,
            1,
        )

        for _ in range(iterations):
            hero_strategy = _regret_matching(hero_regret)
            rival_strategy = _regret_matching(rival_regret)

            hero_strategy_sum += hero_strategy
            rival_strategy_sum += rival_strategy

            hero_util = payoff_matrix @ rival_strategy
            hero_expected = float(hero_strategy @ hero_util)
            hero_regret += hero_util - hero_expected

            rival_payoff = (-(payoff_matrix.T)) @ hero_strategy
            rival_expected = float(rival_strategy @ rival_payoff)
            rival_regret += rival_payoff - rival_expected

        hero_avg = _normalise_strategy(hero_strategy_sum)
        rival_avg = _normalise_strategy(rival_strategy_sum)
        adjusted_values = payoff_matrix @ rival_avg

        for (idx, option), action_value, hero_prob in zip(eligible, adjusted_values, hero_avg, strict=False):
            meta = option.meta or {}
            meta.setdefault("baseline_ev", option.ev)
            meta["cfr_backend"] = self.name
            meta["cfr_probability"] = float(hero_prob)
            meta["cfr_rival_mix"] = {
                str(action): float(prob) for action, prob in zip(rival_actions, rival_avg, strict=False)
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
    if not meta.get("supports_cfr"):
        return False
    if "cfr_payoffs" in meta:
        return True
    return {"hero_ev_fold", "hero_ev_continue"}.issubset(meta.keys())


def _extract_payoffs(options: Iterable[Option]) -> tuple[np.ndarray | None, tuple[str, ...] | None]:
    rows_info: list[tuple[list[str], list[float]]] = []
    labels_order: list[str] = []
    for option in options:
        meta = option.meta or {}
        payoffs = meta.get("cfr_payoffs")
        if isinstance(payoffs, dict):
            raw_labels = payoffs.get("rival_actions")
            hero_values = payoffs.get("hero")
            if not isinstance(raw_labels, (list, tuple)) or not isinstance(hero_values, (list, tuple)):
                return None, None
            try:
                hero_row = [float(value) for value in hero_values]
            except (TypeError, ValueError):
                return None, None
            labels = [str(label) for label in raw_labels]
        else:
            try:
                fold_ev = float(meta["hero_ev_fold"])
                continue_ev = float(meta["hero_ev_continue"])
            except (KeyError, TypeError, ValueError):
                return None, None
            if not math.isfinite(fold_ev) or not math.isfinite(continue_ev):
                return None, None
            labels = ["fold", "continue"]
            hero_row = [fold_ev, continue_ev]

        if any(not math.isfinite(value) for value in hero_row):
            return None, None
        rows_info.append((labels, hero_row))
        for label in labels:
            if label not in labels_order:
                labels_order.append(label)

    if not rows_info:
        return None, None

    if "fold" in labels_order:
        labels_order.remove("fold")
        labels_order.insert(0, "fold")

    matrix_rows: list[list[float]] = []
    for labels, hero_values in rows_info:
        value_map = dict(zip(labels, hero_values))
        row: list[float] = []
        for label in labels_order:
            if label in value_map:
                row.append(value_map[label])
            elif label == "jam" and "call" in value_map:
                row.append(value_map["call"])
            elif label == "continue" and "call" in value_map:
                row.append(value_map["call"])
            else:
                row.append(value_map.get("fold", 0.0))
        matrix_rows.append(row)

    return np.array(matrix_rows, dtype=np.float64), tuple(labels_order)


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
