from __future__ import annotations

import math

import pytest

from gto_trainer.core import scoring


def test_decision_score_respects_dynamic_noise_floor():
    record = {
        "best_ev": 3.0,
        "chosen_ev": 2.93,  # 0.07 bb loss
        "pot_bb": 30.0,
    }

    score = scoring.decision_score(record)

    assert math.isclose(score, 100.0, rel_tol=1e-6)


def test_summary_uses_pot_weighting():
    records = [
        {
            "best_ev": 1.5,
            "chosen_ev": 1.0,
            "pot_bb": 2.0,
            "hand_index": 0,
        },
        {
            "best_ev": 4.0,
            "chosen_ev": 3.9,
            "pot_bb": 10.0,
            "hand_index": 0,
        },
    ]

    summary = scoring.summarize_records(records)

    pots = [r["pot_bb"] for r in records]
    weights = [p if p > 0 else 1.0 for p in pots]
    expected_score = sum(scoring.decision_score(r) * w for r, w in zip(records, weights, strict=False)) / sum(weights)
    expected_loss_pct = 100.0 * (
        sum(scoring.decision_loss_ratio(r) * w for r, w in zip(records, weights, strict=False)) / sum(weights)
    )

    assert summary.score_pct == pytest.approx(expected_score, rel=1e-6)
    assert summary.avg_loss_pct == pytest.approx(expected_loss_pct, rel=1e-6)
