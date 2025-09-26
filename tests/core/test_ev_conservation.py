from __future__ import annotations

from gtotrainer.core.scoring import ev_conservation_diagnostics


def test_ev_conservation_within_tolerance() -> None:
    records = [
        {"best_ev": 1.2, "chosen_ev": 1.0},
        {"best_ev": -0.5, "chosen_ev": -0.8},
    ]
    diagnostics = ev_conservation_diagnostics(records)
    assert diagnostics["within_tolerance"] is True
    assert abs(diagnostics["delta"]) < 1e-9


def test_ev_conservation_detects_inconsistent_records() -> None:
    records = [
        {"best_ev": 0.6, "chosen_ev": 0.1},
        {"best_ev": -0.2, "chosen_ev": -0.1},
    ]
    diagnostics = ev_conservation_diagnostics(records, tolerance=1e-8)
    assert diagnostics["within_tolerance"] is False
    assert diagnostics["delta"] != 0.0
