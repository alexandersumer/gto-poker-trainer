from pytest import approx

from gto_trainer.core.scoring import (
    decision_loss_ratio,
    decision_score,
    summarize_records,
)


def test_summarize_records_empty() -> None:
    stats = summarize_records([])
    assert stats.decisions == 0
    assert stats.score_pct == 0.0
    assert stats.total_ev_lost == 0.0


def test_noise_floor_keeps_perfect_score() -> None:
    record = {
        "best_ev": 1.00,
        "chosen_ev": 0.995,
        "pot_bb": 10.0,
        "best_key": "Call",
        "chosen_key": "Call",
        "hand_index": 0,
    }
    stats = summarize_records([record])
    assert stats.decisions == 1
    assert stats.score_pct > 99.0
    assert stats.avg_loss_pct < 0.1


def test_large_mistake_penalises_score() -> None:
    record = {
        "best_ev": 3.0,
        "chosen_ev": 2.0,
        "pot_bb": 10.0,
        "best_key": "Jam",
        "chosen_key": "Call",
        "hand_index": 0,
    }
    stats = summarize_records([record])
    assert stats.score_pct < 40.0
    assert stats.avg_loss_pct > 5.0


def test_average_score_across_records() -> None:
    records = [
        {
            "best_ev": 2.0,
            "chosen_ev": 2.0,
            "pot_bb": 10.0,
            "best_key": "Bet small",
            "chosen_key": "Bet small",
            "hand_index": 0,
        },
        {
            "best_ev": 2.0,
            "chosen_ev": 1.7,
            "pot_bb": 10.0,
            "best_key": "Jam",
            "chosen_key": "Call",
            "hand_index": 0,
        },
    ]
    stats = summarize_records(records)
    assert stats.decisions == 2
    assert 60.0 < stats.score_pct < 90.0
    assert stats.hands == 1
    # Ensure dedicated helper matches per-record score for the mistake entry
    mistake_score = decision_score(records[1])
    assert mistake_score == approx(stats.score_pct * 2 - 100.0, rel=1e-3)


def test_decision_loss_ratio_uses_room_when_pot_missing() -> None:
    record = {
        "best_ev": 5.0,
        "chosen_ev": 4.0,
        "room_ev": 2.0,
        "pot_bb": 0.0,
    }
    ratio = decision_loss_ratio(record)
    assert ratio == approx(0.5, rel=1e-6)


def test_decision_loss_ratio_falls_back_to_ev_magnitude() -> None:
    record = {
        "best_ev": -1.0,
        "chosen_ev": -2.0,
        "pot_bb": 0.0,
        "room_ev": 0.0,
    }
    ratio = decision_loss_ratio(record)
    assert ratio == approx(0.5, rel=1e-6)


def test_decision_loss_ratio_clamps_negative_ev_loss() -> None:
    record = {
        "best_ev": 1.0,
        "chosen_ev": 1.2,
        "pot_bb": 10.0,
    }
    ratio = decision_loss_ratio(record)
    assert ratio == 0.0
