from __future__ import annotations

import math

import numpy as np

from gtotrainer.core.models import Option
from gtotrainer.dynamic.cfr import LinearCFRBackend, LinearCFRConfig


def _make_option(key: str, hero_row: list[float], rival_row: list[float]) -> Option:
    meta = {
        "supports_cfr": True,
        "cfr_payoffs": {
            "rival_actions": ["A", "B"],
            "hero": hero_row,
            "rival": rival_row,
        },
    }
    return Option(key=key, ev=0.0, why="test", meta=meta)


def test_linear_cfr_validation_balanced_matrix() -> None:
    config = LinearCFRConfig(
        iterations=5000,
        minimum_actions=2,
        extra_iterations_per_action=0,
        linear_weight_pow=1.0,
        regret_floor=1e-12,
        hero_dropout_threshold=0.15,
        hero_min_freq=0.05,
        force_best_response=False,
    )
    backend = LinearCFRBackend(config)
    hero_payoffs = [[0.0, -1.0], [1.0, 0.0]]
    rival_payoffs = [[0.0, 1.0], [-1.0, 0.0]]
    options = [
        _make_option("h0", hero_payoffs[0], rival_payoffs[0]),
        _make_option("h1", hero_payoffs[1], rival_payoffs[1]),
    ]

    refined = backend.refine(None, options)
    hero_freqs = np.array([opt.gto_freq for opt in refined], dtype=float)

    assert np.isfinite(hero_freqs).all()
    assert math.isclose(float(hero_freqs.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-6)

    for opt in refined:
        diagnostics = opt.meta["cfr_validation"]
        assert diagnostics["hero_exploitability"] < 0.01
        assert diagnostics["rival_exploitability"] < 0.01
        assert diagnostics["zero_sum_deviation"] < 1e-9
        assert diagnostics["flags"] == []
        assert "warnings" not in opt.meta


def test_linear_cfr_flags_non_zero_sum_triggered() -> None:
    config = LinearCFRConfig(iterations=500, extra_iterations_per_action=0)
    backend = LinearCFRBackend(config)
    options = [
        _make_option("h0", [0.0, 1.0], [0.2, 0.8]),
        _make_option("h1", [1.0, -0.5], [-0.8, 0.6]),
    ]

    refined = backend.refine(None, options)
    flags_seen = set()
    for opt in refined:
        diagnostics = opt.meta["cfr_validation"]
        flags = diagnostics["flags"]
        flags_seen.update(flags)
        assert "cfr_non_zero_sum_payoffs" in flags
        assert opt.meta.get("warnings")

    assert "cfr_non_zero_sum_payoffs" in flags_seen
