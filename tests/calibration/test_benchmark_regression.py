from __future__ import annotations

import json
from pathlib import Path

import pytest

from gtotrainer.analysis.benchmark import BenchmarkConfig, run_benchmark

REFERENCE_PATH = Path(__file__).with_name("reference_benchmark.json")


def _assert_close(stats, expected: dict[str, float], *, ev_tol: float = 0.6, acc_tol: float = 10.0) -> None:
    assert stats.avg_ev_lost == pytest.approx(float(expected["avg_ev_lost"]), abs=ev_tol)
    assert stats.accuracy_pct == pytest.approx(float(expected["accuracy_pct"]), abs=acc_tol)


def test_benchmark_regression_against_reference() -> None:
    reference = json.loads(REFERENCE_PATH.read_text())
    config_data = reference["config"]
    config = BenchmarkConfig(
        hands=int(config_data["hands"]),
        seeds=tuple(config_data["seeds"]),
        mc_trials=int(config_data["mc_trials"]),
        rival_style=str(config_data["rival_style"]),
    )

    result = run_benchmark(config)

    _assert_close(result.combined, reference["combined"])

    expected_streets: dict[str, dict[str, float]] = reference["street"]
    assert set(result.combined_by_street) == set(expected_streets)
    for street, expected_stats in expected_streets.items():
        _assert_close(result.combined_by_street[street], expected_stats, ev_tol=1.2)
