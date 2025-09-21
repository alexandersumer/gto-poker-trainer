from __future__ import annotations

from gtotrainer.dynamic import policy as pol


def test_record_rival_adapt_and_meta_copy_isolated() -> None:
    hand_state: dict[str, object] = {}

    pol._record_rival_adapt(hand_state, aggressive=True)
    pol._record_rival_adapt(hand_state, aggressive=False)

    adapt = hand_state.get("rival_adapt")
    assert isinstance(adapt, dict)
    assert adapt["aggr"] == 1
    assert adapt["passive"] == 1

    meta = pol._decision_meta({}, hand_state)
    assert meta["rival_adapt"] == {"aggr": 1, "passive": 1}

    meta["rival_adapt"]["aggr"] = 99
    # Ensure original hand state not mutated by consumer changes
    assert adapt["aggr"] == 1
