from __future__ import annotations

import pytest

from gtotrainer.analysis.benchmark import BenchmarkConfig, run_benchmark


def test_benchmark_deterministic_small_run() -> None:
    config = BenchmarkConfig(hands=5, seeds=(11,), mc_trials=48)
    result_one = run_benchmark(config)
    result_two = run_benchmark(config)

    assert result_one.combined.hands == 5
    assert result_two.combined.hands == 5
    assert result_one.combined.decisions == result_two.combined.decisions
    assert result_one.combined.decisions > 0
    assert result_one.combined.avg_ev_lost == pytest.approx(result_two.combined.avg_ev_lost, rel=1e-9)
    assert 0.0 <= result_one.exploitability_bb < 10.0
