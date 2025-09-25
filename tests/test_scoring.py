from __future__ import annotations

import math

import pytest

from gtotrainer.core import scoring


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
            "best_key": "bet",
            "chosen_key": "call",
            "hand_index": 0,
        },
        {
            "best_ev": 4.0,
            "chosen_ev": 3.9,
            "pot_bb": 10.0,
            "best_key": "raise",
            "chosen_key": "raise",
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


def test_summary_tracks_hits_and_ev_loss_with_noise_floor():
    records = [
        {
            "best_ev": 1.0,
            "chosen_ev": 1.0,
            "pot_bb": 3.0,
            "best_key": "bet",
            "chosen_key": "bet",
            "hand_index": 0,
        },
        {
            "best_ev": 1.52,
            "chosen_ev": 1.50,
            "pot_bb": 4.0,
            "best_key": "bet",
            "chosen_key": "call",
            "hand_index": 0,
        },
        {
            "best_ev": 3.0,
            "chosen_ev": 2.0,
            "pot_bb": 6.0,
            "best_key": "raise",
            "chosen_key": "call",
            "hand_index": 1,
        },
    ]

    summary = scoring.summarize_records(records)

    assert summary.hits == 2
    assert summary.total_ev_lost == pytest.approx(1.02, rel=1e-6)
    assert summary.avg_ev_lost == pytest.approx(1.02 / 3, rel=1e-6)


def test_ev_band_accuracy_curve_respects_warning_band():
    def make_record(loss: float, *, pot: float = 20.0, idx: int = 0) -> dict:
        return {
            "best_ev": 1.0,
            "chosen_ev": 1.0 - loss,
            "pot_bb": pot,
            "best_key": f"best_{idx}",
            "chosen_key": f"choice_{idx}",
            "hand_index": idx,
        }

    def accuracy_for(loss: float) -> float:
        record = make_record(loss, idx=int(loss * 100))
        stats = scoring.summarize_records([record])
        return stats.accuracy_pct

    assert accuracy_for(0.0) == pytest.approx(100.0)
    assert accuracy_for(0.1) == pytest.approx(96.1941757068, rel=1e-9)
    assert accuracy_for(0.5) == pytest.approx(71.9644465063, rel=1e-9)
    assert accuracy_for(1.5) == pytest.approx(31.8814075811, rel=1e-9)
    assert accuracy_for(12.0) == pytest.approx(0.0)
