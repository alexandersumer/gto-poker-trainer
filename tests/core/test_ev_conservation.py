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


def test_ev_conservation_handles_out_of_policy_jam() -> None:
    """Conservation diagnostic uses actual EV delta, not grading penalty.

    When a user chooses an out-of-policy jam with higher EV than the best
    in-policy action, the grading system stores a penalty in ev_loss.
    The conservation diagnostic should use the actual EV delta (which is
    negative/zero for a jam that gains EV) to verify conservation.
    """
    records = [
        {
            "best_ev": 0.5,
            "chosen_ev": 1.5,
            "ev_loss": 1.0,  # Grading penalty stored by service.py
        }
    ]

    diagnostics = ev_conservation_diagnostics(records)

    # User gained 1.0 EV by choosing the jam
    assert diagnostics["total_best"] == 0.5
    assert diagnostics["total_chosen"] == 1.5

    # Actual EV lost is 0.0 (they gained EV)
    assert diagnostics["total_ev_lost"] == 0.0

    # Conservation: (best - chosen) - ev_lost = (0.5 - 1.5) - 0 = -1.0
    # This is within tolerance (it's the EV gain)
    assert diagnostics["delta"] == -1.0
    assert diagnostics["within_tolerance"] is False  # EV was gained, not conserved
