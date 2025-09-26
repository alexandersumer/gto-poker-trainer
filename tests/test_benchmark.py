from __future__ import annotations

import random

import pytest

from gtotrainer.analysis.benchmark import (
    BenchmarkConfig,
    BenchmarkScenario,
    _HeroPolicy,
    run_benchmark,
)
from gtotrainer.core.models import Option


def test_benchmark_deterministic_small_run() -> None:
    scenario = BenchmarkScenario(name="tiny", seed=11, rival_style="balanced", hands=5)
    config = BenchmarkConfig(hands=5, seeds=(11,), mc_trials=32, scenarios=(scenario,))
    result_one = run_benchmark(config)
    result_two = run_benchmark(config)

    assert len(result_one.runs) == 1
    assert result_one.runs[0].scenario.name == "tiny"
    assert result_one.runs[0].street_stats
    assert result_one.combined.hands == 5
    assert result_two.combined.hands == 5
    assert result_one.combined.decisions == result_two.combined.decisions
    assert result_one.combined.decisions > 0
    assert result_one.combined.avg_ev_lost == pytest.approx(result_two.combined.avg_ev_lost, rel=1e-9)
    assert 0.0 <= result_one.exploitability_bb < 10.0


def test_default_seed_expansion_generates_runs() -> None:
    config = BenchmarkConfig(hands=3, seeds=(101, 202), mc_trials=24)
    result = run_benchmark(config)
    assert {run.scenario.name for run in result.runs} == {"seed_0_101", "seed_1_202"}


def test_hero_policy_best_uses_baseline_meta() -> None:
    policy = _HeroPolicy("best")
    options = [
        Option("call", 0.9, "", meta={"baseline_ev": 1.35}),
        Option("fold", 1.2, "", meta={"baseline_ev": 1.0}),
        Option("raise", 1.1, ""),
    ]

    idx = policy.select(options, random.Random(1))
    assert options[idx].key == "call"


def test_hero_policy_gto_ignores_zero_freq_options() -> None:
    policy = _HeroPolicy("gto")
    high = Option("high", 1.0, "", gto_freq=0.7)
    low = Option("low", 0.8, "", gto_freq=0.3)
    zero = Option("zero", 2.0, "", gto_freq=0.0)
    hit = set()
    rng = random.Random(7)

    for _ in range(200):
        choice = policy.select([high, low, zero], rng)
        selected = [high, low, zero][choice]
        assert selected is not zero
        hit.add(selected.key)

    assert hit == {"high", "low"}


def test_hero_policy_unknown_mode_errors() -> None:
    with pytest.raises(ValueError):
        _HeroPolicy("mystery")


def test_benchmark_config_rejects_unknown_styles() -> None:
    with pytest.raises(ValueError):
        BenchmarkConfig(hands=5, seeds=(101,), mc_trials=24, rival_style="loose")


def test_benchmark_scenario_validates_styles() -> None:
    scenario = BenchmarkScenario(name="bad", seed=11, rival_style="hyper")
    with pytest.raises(ValueError):
        BenchmarkConfig(hands=5, seeds=(101,), mc_trials=24, scenarios=(scenario,))
