from __future__ import annotations

import pytest

from gto_poker_trainer_cli.application import ChoiceResult, NodeResponse, SessionConfig, SessionManager


def test_session_manager_basic_flow():
    manager = SessionManager()
    session_id = manager.create_session(SessionConfig(hands=2, mc_trials=50, seed=1234))

    first = manager.get_node(session_id)
    assert isinstance(first, NodeResponse)
    data = first.to_dict()
    assert not data["done"]
    node = data["node"]
    assert node["hand_no"] == 1
    assert node["total_hands"] == 2
    assert len(node["hero_cards"]) == 2
    assert all(len(card) == 2 for card in node["hero_cards"])
    assert data["options"], "options should be available"
    first_option = data["options"][0]
    assert {"key", "label", "ev", "why", "ends_hand"}.issubset(first_option)

    # choose first option for a couple of nodes to ensure caching works
    choice = manager.choose(session_id, 0)
    assert isinstance(choice, ChoiceResult)
    choice_payload = choice.to_dict()
    assert "feedback" in choice_payload
    feedback = choice_payload["feedback"]
    assert "chosen" in feedback and "label" in feedback["chosen"]
    next_payload = choice_payload["next"]
    assert "done" in next_payload

    # Play until session completes to validate summary
    steps = 0
    current = next_payload
    while not current["done"]:
        current = manager.choose(session_id, 0).to_dict()["next"]
        steps += 1
        assert steps < 50, "session did not converge"

    summary = current["summary"]
    assert summary is not None
    assert summary["hands"] >= 1

    # Summary endpoint mirrors the same content
    summary_direct = manager.summary(session_id).to_dict()
    assert summary_direct == summary


def test_invalid_session_errors():
    manager = SessionManager()
    with pytest.raises(KeyError):
        manager.get_node("missing")
    sid = manager.create_session(SessionConfig(hands=1, mc_trials=20, seed=42))
    manager.choose(sid, 0)  # progress once to ensure valid session
    with pytest.raises(ValueError):
        manager.choose(sid, 999)


def test_summary_scoring_uses_full_room_between_best_and_worst():
    from gto_poker_trainer_cli.application.session_service import _summary_payload

    records = [
        {
            "street": "TURN",
            "chosen_key": "call",
            "chosen_ev": 3.8,
            "best_key": "raise",
            "best_ev": 4.2,
            "worst_ev": 3.0,
            "room_ev": 1.2,
            "ev_loss": 0.4,
            "hand_ended": False,
            "resolution_note": None,
            "hand_index": 0,
        },
        {
            "street": "RIVER",
            "chosen_key": "bet",
            "chosen_ev": 2.0,
            "best_key": "shove",
            "best_ev": 2.5,
            "worst_ev": 1.8,
            "room_ev": 0.7,
            "ev_loss": 0.5,
            "hand_ended": True,
            "resolution_note": "Villain folds",
            "hand_index": 0,
        },
    ]

    summary = _summary_payload(records)

    assert summary.hands == 1
    assert summary.decisions == 2
    assert summary.hits == 0
    assert summary.ev_lost == pytest.approx(0.9)
    assert summary.score == pytest.approx(52.63, rel=1e-3)


def test_summary_counts_unique_hands():
    from gto_poker_trainer_cli.application.session_service import _summary_payload

    records = [
        {
            "street": "PREFLOP",
            "chosen_key": "call",
            "chosen_ev": 1.0,
            "best_key": "3bet",
            "best_ev": 1.3,
            "worst_ev": 0.4,
            "room_ev": 0.9,
            "ev_loss": 0.3,
            "hand_ended": True,
            "resolution_note": "Villain folds to your 3-bet. Pot 4.00bb awarded (net +2.50bb).",
            "hand_index": 0,
        },
        {
            "street": "PREFLOP",
            "chosen_key": "fold",
            "chosen_ev": 0.0,
            "best_key": "call",
            "best_ev": 0.1,
            "worst_ev": -0.5,
            "room_ev": 0.6,
            "ev_loss": 0.1,
            "hand_ended": True,
            "resolution_note": "You fold. SB keeps 3.50bb.",
            "hand_index": 1,
        },
    ]

    summary = _summary_payload(records)

    assert summary.hands == 2
    assert summary.decisions == 2


def _play_session_with_policy(manager, sid, chooser):
    node_resp = manager.get_node(sid)
    guard = 0
    while not node_resp.done:
        assert node_resp.options, "expected options while session active"
        choice_index = chooser(node_resp.options)
        choice = manager.choose(sid, choice_index)
        node_resp = choice.next_payload
        guard += 1
        if guard > 32:
            raise AssertionError("session did not complete within expected steps")
    return manager.summary(sid)


def test_optimal_play_produces_zero_ev_loss():
    manager = SessionManager()
    sid = manager.create_session(SessionConfig(hands=1, mc_trials=60, seed=321))

    def take_best(options):
        return max(range(len(options)), key=lambda idx: options[idx].ev)

    summary = _play_session_with_policy(manager, sid, take_best)

    assert summary.hands == 1
    assert summary.decisions >= summary.hands
    assert summary.ev_lost == pytest.approx(0.0, abs=1e-9)
    assert summary.score == pytest.approx(100.0)
    assert summary.hits > 0


def test_poor_choices_accumulate_ev_loss():
    manager = SessionManager()
    sid = manager.create_session(SessionConfig(hands=1, mc_trials=60, seed=321))

    def take_worst(options):
        return min(range(len(options)), key=lambda idx: options[idx].ev)

    summary = _play_session_with_policy(manager, sid, take_worst)

    assert summary.hands == 1
    assert summary.decisions >= summary.hands
    assert summary.ev_lost > 0.0
    assert summary.score < 100.0
    assert summary.hits == 0
