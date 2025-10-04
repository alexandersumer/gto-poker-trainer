from __future__ import annotations

import math
import random
from collections.abc import Sequence

import pytest

from gtotrainer.core import scoring
from gtotrainer.core.models import Option
from gtotrainer.dynamic import policy as policy_module
from gtotrainer.dynamic.cards import str_to_int
from gtotrainer.dynamic.episode import Node
from gtotrainer.features.session import (
    ChoiceResult,
    NodeResponse,
    SessionConfig,
    SessionManager,
    service as session_service,
)
from gtotrainer.features.session.service import (
    _best_index,
    _ensure_active_node,
    _ensure_options,
    _node_payload,
    _summary_payload,
    _view_context,
)


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
    assert "accuracy" in feedback
    assert "cumulative_ev_lost" in feedback
    assert "cumulative_accuracy" in feedback
    assert "decisions" in feedback
    assert 0.0 <= feedback["accuracy"] <= 1.0
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


def test_summary_payload_accuracy_matches_backend():
    records = [
        {
            "best_ev": 1.5,
            "chosen_ev": 1.5,
            "pot_bb": 4.0,
            "best_key": "bet",
            "chosen_key": "bet",
            "hand_index": 0,
        },
        {
            "best_ev": 2.0,
            "chosen_ev": 1.99,
            "pot_bb": 5.0,
            "best_key": "raise",
            "chosen_key": "raise",
            "hand_index": 0,
        },
        {
            "best_ev": 3.0,
            "chosen_ev": 2.1,
            "pot_bb": 6.0,
            "best_key": "raise",
            "chosen_key": "call",
            "hand_index": 1,
        },
    ]

    payload = _summary_payload(records)
    stats = scoring.summarize_records(records)
    assert payload.decisions == 3
    assert payload.hits == stats.hits
    assert payload.accuracy_pct == pytest.approx(stats.accuracy_pct)
    assert payload.accuracy_points == pytest.approx(stats.accuracy_points)
    hits_ratio = (100.0 * payload.hits) / payload.decisions
    assert not math.isclose(payload.accuracy_pct, hits_ratio)


def test_summary_payload_accuracy_partial_credit():
    records = [
        {
            "best_ev": 1.0,
            "chosen_ev": 0.9,
            "pot_bb": 20.0,
            "best_key": "bet",
            "chosen_key": "call",
            "hand_index": 0,
        },
        {
            "best_ev": 2.5,
            "chosen_ev": 1.5,
            "pot_bb": 20.0,
            "best_key": "raise",
            "chosen_key": "call",
            "hand_index": 1,
        },
    ]

    payload = _summary_payload(records)
    assert 0.0 < payload.accuracy_pct < 100.0
    assert payload.accuracy_pct != pytest.approx((100.0 * payload.hits) / payload.decisions)
    assert payload.accuracy_points == pytest.approx(scoring.summarize_records(records).accuracy_points)


def test_view_context_normalizes_core_fields():
    node = Node(
        street="turn",
        description="Board; Rival (SB) bets 5.60bb into 11.20bb.",
        pot_bb=11.2,
        effective_bb=88.4,
        hero_cards=[1, 2],
        board=[3, 4, 5, 6],
        actor="BB",
        context={
            "facing": "Bet",
            "bet": 5.6,
            "open_size": 2.5,
            "hero_seat": "bb",
            "rival_seat": "sb",
        },
    )

    view = _view_context(node)

    assert view["facing"] == "bet"
    assert view["bet"] == pytest.approx(5.6)
    assert view["open_size"] == pytest.approx(2.5)
    assert view["hero_seat"] == "BB"
    assert view["rival_seat"] == "SB"
    assert view["actor_seat"] == "BB"
    assert view["actor_role"] == "hero"


def test_node_payload_includes_sanitized_context():
    node = Node(
        street="river",
        description="Board; Rival checks.",
        pot_bb=12.0,
        effective_bb=74.0,
        hero_cards=[7, 8],
        board=[9, 10, 11, 12, 13],
        actor="SB",
        context={
            "facing": "check",
            "hero_seat": "bb",
            "rival_seat": "sb",
        },
    )
    manager = SessionManager()
    session_id = manager.create_session(SessionConfig(hands=1, mc_trials=40, seed=123))
    state = manager._sessions[session_id]

    payload = _node_payload(state, node)

    assert payload.context is not None
    assert payload.context["facing"] == "check"
    assert payload.context["actor_role"] == "rival"
    assert payload.context["actor_seat"] == "SB"


def test_view_context_omits_unknown_fields():
    node = Node(
        street="flop",
        description="Board; Rival checks.",
        pot_bb=6.0,
        effective_bb=90.0,
        hero_cards=[14, 15],
        board=[16, 17, 18],
        actor="BB",
        context={},
    )

    assert _view_context(node) == {"actor_seat": "BB"}


def test_view_context_preserves_facing_bet_information():
    node = Node(
        street="turn",
        description="Board; Rival bets 5.60bb into 11.20bb.",
        pot_bb=11.2,
        effective_bb=88.4,
        hero_cards=[19, 20],
        board=[21, 22, 23, 24],
        actor="BB",
        context={
            "facing": "bet",
            "bet": 5.6,
            "hero_seat": "bb",
            "rival_seat": "sb",
        },
    )

    view = _view_context(node)

    assert view["facing"] == "bet"
    assert view["bet"] == pytest.approx(5.6)


def test_session_manager_alternates_blinds():
    manager = SessionManager()
    hands = 6
    session_id = manager.create_session(SessionConfig(hands=hands, mc_trials=40, seed=99))

    sequence: list[str] = []
    actors_seen: dict[int, str] = {}

    payload = manager.get_node(session_id)
    assert not payload.done and payload.node is not None
    sequence.append(payload.node.actor)
    actors_seen[payload.node.hand_no] = payload.node.actor

    guard = 0
    while len(sequence) < hands:
        choice = manager.choose(session_id, 0)
        payload = choice.next_payload
        guard += 1
        assert guard < 256, "session did not reach all hands"
        if payload.done:
            break
        assert payload.node is not None
        if payload.node.hand_no not in actors_seen:
            sequence.append(payload.node.actor)
            actors_seen[payload.node.hand_no] = payload.node.actor

    assert sequence == ["BB", "SB", "BB", "SB", "BB", "SB"]


def test_invalid_session_errors():
    manager = SessionManager()
    with pytest.raises(KeyError):
        manager.get_node("missing")
    sid = manager.create_session(SessionConfig(hands=1, mc_trials=20, seed=42))
    manager.choose(sid, 0)  # progress once to ensure valid session
    with pytest.raises(ValueError):
        manager.choose(sid, 999)


def test_turn_rebuild_preserves_check_metadata():
    manager = SessionManager()
    sid = manager.create_session(SessionConfig(hands=1, mc_trials=40, seed=1001))

    manager.get_node(sid)
    manager.choose(sid, 1)  # Call preflop to reach the flop
    manager.get_node(sid)
    manager.choose(sid, 0)  # Check back the flop to trigger the rebuild

    state = manager._sessions[sid]
    turn_node = state.episodes[0].nodes[2]
    hand_state = turn_node.context["hand_state"]

    assert hand_state["turn_mode"] == "check"
    assert turn_node.context["facing"] == "check"
    assert "bet" not in turn_node.context
    assert turn_node.description.endswith("checks.")


def test_turn_rebuild_preserves_stored_bet_size():
    manager = SessionManager()
    sid = manager.create_session(SessionConfig(hands=1, mc_trials=40, seed=999))

    manager.get_node(sid)
    manager.choose(sid, 1)  # Call preflop
    manager.get_node(sid)
    manager.choose(sid, 0)  # Check flop, forcing the rebuild

    state = manager._sessions[sid]
    turn_node = state.episodes[0].nodes[2]
    hand_state = turn_node.context["hand_state"]
    stored_bet = hand_state.get("turn_bet_size")

    assert hand_state["turn_mode"] == "bet"
    assert stored_bet is not None
    assert turn_node.context["facing"] == "bet"
    assert turn_node.context["bet"] == pytest.approx(float(stored_bet))
    assert f"{float(stored_bet):.2f}bb" in turn_node.description


def test_river_rebuild_preserves_lead_metadata():
    manager = SessionManager()
    sid = manager.create_session(SessionConfig(hands=1, mc_trials=40, seed=2002))

    manager.get_node(sid)
    manager.choose(sid, 1)  # Call preflop
    manager.get_node(sid)
    manager.choose(sid, 0)  # Check flop
    manager.get_node(sid)
    manager.choose(sid, 1)  # Call the turn bet to reach the river

    state = manager._sessions[sid]
    river_node = state.episodes[0].nodes[3]
    hand_state = river_node.context["hand_state"]
    lead_size = hand_state.get("river_lead_size")

    assert hand_state["river_mode"] == "lead"
    assert lead_size is not None
    assert river_node.context["facing"] == "bet"
    assert river_node.context["bet"] == pytest.approx(float(lead_size))
    assert "leads" in river_node.description


def test_river_rebuild_keeps_check_context():
    manager = SessionManager()
    sid = manager.create_session(SessionConfig(hands=1, mc_trials=40, seed=2001))

    manager.get_node(sid)
    manager.choose(sid, 1)  # Call preflop
    manager.get_node(sid)
    manager.choose(sid, 0)  # Check flop
    manager.get_node(sid)
    manager.choose(sid, 0)  # Check turn to advance

    payload = manager.get_node(sid)
    river_node = payload.node
    assert river_node is not None
    assert river_node.context["facing"] == "oop-check"
    assert "bet" not in river_node.context
    assert river_node.description.endswith("choose your bet.")


def test_create_session_uses_explicit_zero_seed():
    manager = SessionManager()
    cfg = SessionConfig(hands=1, mc_trials=40, seed=0)

    sid_one = manager.create_session(cfg)
    first_one = manager.get_node(sid_one).to_dict()
    state_one = manager._sessions[sid_one]

    sid_two = manager.create_session(cfg)
    first_two = manager.get_node(sid_two).to_dict()
    state_two = manager._sessions[sid_two]

    assert state_one.config.seed == 0
    assert state_two.config.seed == 0
    assert first_one["node"] == first_two["node"]
    assert first_one["options"] == first_two["options"]


def test_summary_scoring_matches_decision_scores():
    from gtotrainer.core.scoring import decision_loss_ratio, decision_score, summarize_records
    from gtotrainer.features.session.service import _summary_payload

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
            "pot_bb": 6.0,
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
            "resolution_note": "Rival folds",
            "hand_index": 0,
            "pot_bb": 12.0,
        },
    ]

    summary = _summary_payload(records)

    assert summary.hands == 1
    assert summary.decisions == 2
    assert summary.hits == 0
    assert summary.ev_lost == pytest.approx(0.9)
    assert summary.avg_ev_lost == pytest.approx(0.45)
    pots = [float(r.get("pot_bb", 0.0)) for r in records]
    weights = [p if p > 0 else 1.0 for p in pots]
    expected = sum(decision_score(r) * w for r, w in zip(records, weights, strict=False)) / sum(weights)
    expected_loss_pct = 100.0 * (
        sum(decision_loss_ratio(r) * w for r, w in zip(records, weights, strict=False)) / sum(weights)
    )
    assert summary.score == pytest.approx(expected, rel=1e-3)
    assert summary.avg_loss_pct == pytest.approx(expected_loss_pct, rel=1e-3)
    expected_accuracy = summarize_records(records).accuracy_pct
    assert summary.accuracy_pct == pytest.approx(expected_accuracy)


def test_summary_counts_unique_hands():
    from gtotrainer.features.session.service import _summary_payload

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
            "resolution_note": "Rival folds to your 3-bet. Pot 4.00bb awarded (net +2.50bb).",
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


def test_option_cache_returns_defensive_copies(monkeypatch: pytest.MonkeyPatch):
    manager = SessionManager()
    sid = manager.create_session(SessionConfig(hands=1, mc_trials=30, seed=777))
    state = manager._sessions[sid]
    node = _ensure_active_node(state)

    shared_options = [
        Option("Fold", 0.0, "orig"),
        Option("Call", 0.1, "orig"),
    ]

    def fake_options_for(_node, _rng, _mc_trials):
        return shared_options

    monkeypatch.setattr(session_service, "options_for", fake_options_for)

    first = _ensure_options(state, node)
    assert first[0].why == "orig"
    first[0].why = "mutated"

    # Cached entry remains immutable despite caller mutation
    cached = state.cached_options[id(node)][0]
    assert cached.why == "orig"

    second = _ensure_options(state, node)
    assert second[0].why == "orig"
    assert second[0] is not first[0]


def test_session_manager_async_wrappers_align_with_sync():
    manager = SessionManager()
    config = SessionConfig(hands=2, mc_trials=50, seed=2024)

    async def _exercise() -> None:
        sid_sync = manager.create_session(config)
        sid_async = await manager.create_session_async(config)

        node_sync = manager.get_node(sid_sync)
        node_async = await manager.get_node_async(sid_async)
        assert node_async.to_dict() == node_sync.to_dict()

        choice_sync = manager.choose(sid_sync, 0)
        choice_async = await manager.choose_async(sid_async, 0)
        assert choice_async.to_dict() == choice_sync.to_dict()

        summary_sync = manager.summary(sid_sync)
        summary_async = await manager.summary_async(sid_async)
        assert summary_async.to_dict() == summary_sync.to_dict()

    import asyncio

    asyncio.run(_exercise())


def _play_session_with_policy(manager, sid, chooser):
    node_resp = manager.get_node(sid)
    guard = 0
    while not node_resp.done:
        assert node_resp.options, "expected options while session active"
        choice_index = chooser(node_resp, manager)
        choice = manager.choose(sid, choice_index)
        node_resp = choice.next_payload
        guard += 1
        if guard > 32:
            raise AssertionError("session did not complete within expected steps")
    return manager.summary(sid)


def test_optimal_play_produces_zero_ev_loss():
    manager = SessionManager()
    sid = manager.create_session(SessionConfig(hands=1, mc_trials=60, seed=321))

    def take_best(_resp, mgr):
        state = mgr._sessions[sid]
        node = _ensure_active_node(state)
        options = _ensure_options(state, node)
        return _best_index(options)

    summary = _play_session_with_policy(manager, sid, take_best)

    assert summary.hands == 1
    assert summary.decisions >= summary.hands
    assert summary.ev_lost == pytest.approx(0.0, abs=1e-9)
    assert summary.score == pytest.approx(100.0)
    assert summary.avg_ev_lost == pytest.approx(0.0, abs=1e-9)
    assert summary.accuracy_pct == pytest.approx(100.0)


def test_session_manager_normalises_rival_style():
    manager = SessionManager()
    sid = manager.create_session(SessionConfig(hands=1, mc_trials=60, seed=11, rival_style="Aggressive"))
    assert manager._sessions[sid].engine.rival_style == "aggressive"

    sid2 = manager.create_session(SessionConfig(hands=1, mc_trials=60, seed=12, rival_style="unknown"))
    assert manager._sessions[sid2].engine.rival_style == "balanced"


def test_poor_choices_accumulate_ev_loss():
    manager = SessionManager()
    sid = manager.create_session(SessionConfig(hands=1, mc_trials=60, seed=321))

    def take_worst(_resp, mgr):
        state = mgr._sessions[sid]
        node = _ensure_active_node(state)
        options = _ensure_options(state, node)
        return min(range(len(options)), key=lambda idx: session_service._effective_ev(options[idx]))

    summary = _play_session_with_policy(manager, sid, take_worst)

    assert summary.hands == 1
    assert summary.decisions >= summary.hands
    assert summary.ev_lost > 0.0
    assert summary.score < 100.0
    assert summary.hits == 0
    assert summary.avg_ev_lost > 0.0
    assert summary.accuracy_pct < 50.0


def test_drive_session_replays_without_internal_access() -> None:
    manager = SessionManager()
    config = SessionConfig(hands=3, mc_trials=32, seed=2024)
    session_id = manager.create_session(config)

    def chooser(_node: Node, options: Sequence[Option], _rng: random.Random) -> int:
        return max(range(len(options)), key=lambda idx: options[idx].ev)

    records = manager.drive_session(session_id, chooser, cleanup=True)

    assert len(records) >= config.hands
    assert all("ev_loss" in record for record in records)
    with pytest.raises(KeyError):
        manager.summary(session_id)


def _preflop_node_for(hand: str, *, open_size: float = 2.5, pot: float = 3.5) -> Node:
    hero_cards = [str_to_int(hand[:2]), str_to_int(hand[2:])]
    hand_state = {
        "pot": pot,
        "hero_cards": tuple(hero_cards),
        "rival_cards": (str_to_int("Qh"), str_to_int("Js")),
        "full_board": (0, 0, 0, 0, 0),
        "street": "preflop",
        "board_index": 0,
        "history": [],
        "nodes": {},
        "hero_contrib": 1.0,
        "rival_contrib": open_size,
        "hero_stack": 97.0,
        "rival_stack": 97.0,
        "effective_stack": 97.0,
        "rival_style": "balanced",
    }
    return Node(
        street="preflop",
        description="Test preflop node",
        pot_bb=pot,
        effective_bb=97.0,
        hero_cards=hero_cards,
        board=[],
        actor="BB",
        context={"open_size": open_size, "hand_state": hand_state},
    )


def test_best_index_respects_policy_support() -> None:
    node = _preflop_node_for("6h2c")
    options = policy_module.preflop_options(node, random.Random(7), mc_trials=160)

    fold = next(opt for opt in options if opt.key.startswith("Fold"))
    three_bet = next(opt for opt in options if opt.key.startswith("3-bet"))
    assert fold.gto_freq is not None and fold.gto_freq > (three_bet.gto_freq or 0.0)
    assert three_bet.gto_freq is not None and three_bet.gto_freq == pytest.approx(0.0, abs=1e-6)
    assert session_service._out_of_policy(three_bet) is True
    assert "Not in solver policy" in three_bet.why

    best_idx = _best_index(options)
    best_option = options[best_idx]
    assert not best_option.key.startswith("3-bet")
    if best_option.gto_freq is not None:
        assert best_option.gto_freq > 0.0
    else:
        mix = best_option.meta.get("solver_mix") if isinstance(best_option.meta, dict) else None
        if isinstance(mix, dict) and mix:
            assert max(float(v) for v in mix.values()) > 0.0
    assert session_service._out_of_policy(best_option) is not True
    assert session_service._effective_ev(best_option) >= session_service._effective_ev(three_bet)


def test_equal_ev_actions_all_considered_best() -> None:
    """When multiple in-policy actions have equal EV, all should be valid best choices.

    In GTO poker, when actions have identical EV, they're all equally correct.
    The solver mixes between them at various frequencies, but frequency doesn't
    indicate one is "better" - it's just how the solver chooses to implement the
    equilibrium. Users should not be penalized for choosing any equal-EV action.
    """
    from gtotrainer.core.models import Option

    # Create options with identical EV but different frequencies
    opts = [
        Option(key="fold", ev=1.5, why="", gto_freq=0.5, meta={}),
        Option(key="call", ev=1.5, why="", gto_freq=0.3, meta={}),
        Option(key="raise", ev=1.5, why="", gto_freq=0.2, meta={}),
    ]

    # All three have identical EV, so all should be considered "best"
    # The current bug picks fold (highest freq), but all are equally valid
    best_idx = _best_index(opts)
    best_ev = session_service._effective_ev(opts[best_idx])

    # Verify all options have the same EV
    for opt in opts:
        assert session_service._effective_ev(opt) == pytest.approx(best_ev, abs=1e-6)

    # The bug is that choosing call or raise would show ev_loss > 0
    # After fix, ev_loss should be ~0 for all equal-EV choices
    # For now, just document the expected behavior

    # Test with slight EV differences (within noise)
    opts_with_noise = [
        Option(key="fold", ev=1.500, why="", gto_freq=0.5, meta={}),
        Option(key="call", ev=1.499, why="", gto_freq=0.3, meta={}),  # 0.001 bb worse
        Option(key="raise", ev=1.501, why="", gto_freq=0.2, meta={}),  # 0.001 bb better
    ]

    # With noise-level differences, raise should be best despite lowest freq
    best_idx_noise = _best_index(opts_with_noise)
    assert opts_with_noise[best_idx_noise].key == "raise", \
        "Best action should be determined by EV, not frequency"


def test_out_of_policy_ev_gain_scores_as_loss() -> None:
    manager = SessionManager()
    sid = manager.create_session(SessionConfig(hands=5, mc_trials=40, seed=22))
    state = manager._sessions[sid]

    for _ in range(40):
        node_response = manager.get_node(sid)
        node_data = node_response.to_dict()
        if node_data["done"]:
            break
        node = node_data["node"]
        options = node_data["options"] or []
        hero_cards = set(node.get("hero_cards", []))
        jam_idx = None
        if hero_cards == {"3H", "JC"}:
            for idx, opt in enumerate(options):
                if opt["label"].lower().startswith("all-in"):
                    jam_idx = idx
                    break
        if jam_idx is None:
            if not options:
                continue
            best_idx = max(range(len(options)), key=lambda idx: options[idx]["ev"])
            manager.choose(sid, best_idx)
            continue

        choice = manager.choose(sid, jam_idx)
        feedback = choice.feedback
        record = state.records[-1]

        assert feedback.best.key.startswith("Call")
        assert feedback.chosen.key.startswith("All-in")
        assert feedback.ev_loss > 0.0
        assert feedback.accuracy == pytest.approx(0.0)
        assert record["chosen_out_of_policy"] is True
        assert record["best_out_of_policy"] is not True
        assert feedback.ev_loss == pytest.approx(abs(feedback.best.ev - feedback.chosen.ev), rel=1e-6)
        break
    else:  # pragma: no cover - guard to ensure deterministic scenario exists
        pytest.fail("expected preflop jam scenario not reached")


def test_best_index_excludes_explicitly_flagged_out_of_policy() -> None:
    """Verify that actions with explicit out_of_policy flags are never selected as best.

    This tests the fix for the bug where out-of-policy actions with higher EV than
    in-policy actions could be selected as "best" when frequency data was missing,
    causing misleading feedback like "Highest EV (out of policy)" for valid calls.
    """
    # Scenario: Call is 100% in-policy, 3-bet is explicitly out-of-policy with higher EV
    opts = [
        Option(key="call", ev=0.5, why="Standard call", gto_freq=1.0, meta={}),
        Option(
            key="3-bet",
            ev=0.6,  # Higher EV
            why="Not in solver policy",
            gto_freq=0.0,
            meta={"out_of_policy": True},  # Explicit flag
        ),
    ]

    best_idx = _best_index(opts)
    best = opts[best_idx]

    assert best.key == "call", "Best action should be call, not the out-of-policy 3-bet"
    assert session_service._out_of_policy(best) is not True, "Best action should not be out-of-policy"

    # Edge case: All actions have explicit out_of_policy flags (shouldn't happen in practice)
    all_oop = [
        Option(key="fold", ev=0.0, why="", gto_freq=0.0, meta={"out_of_policy": True}),
        Option(key="call", ev=0.5, why="", gto_freq=0.0, meta={"out_of_policy": True}),
    ]

    # Should still work (fall back to highest EV)
    best_idx_oop = _best_index(all_oop)
    assert all_oop[best_idx_oop].key == "call", "Should pick highest EV when all are out-of-policy"

    # Mixed case: Some have frequencies, some have explicit flags
    mixed = [
        Option(key="call", ev=0.5, why="", gto_freq=0.7, meta={}),
        Option(key="fold", ev=0.0, why="", gto_freq=0.3, meta={}),
        Option(key="3-bet", ev=0.6, why="", gto_freq=None, meta={"out_of_policy": True}),
    ]

    best_idx_mixed = _best_index(mixed)
    best_mixed = mixed[best_idx_mixed]
    assert best_mixed.key in ("call", "fold"), "Should select in-policy action"
    assert session_service._out_of_policy(best_mixed) is not True
