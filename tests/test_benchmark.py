from __future__ import annotations

import pytest

from gtotrainer.analysis.benchmark import BenchmarkConfig, BenchmarkScenario, run_benchmark


def test_benchmark_deterministic_small_run() -> None:
    scenario = BenchmarkScenario(name="tiny", seed=11, rival_style="balanced", hands=5)
    config = BenchmarkConfig(hands=5, seeds=(11,), mc_trials=32, scenarios=(scenario,))
    result_one = run_benchmark(config)
    result_two = run_benchmark(config)

    assert len(result_one.runs) == 1
    assert result_one.runs[0].scenario.name == "tiny"
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
