from __future__ import annotations

import random

from gto_trainer.application.session_engine import SessionEngine
from gto_trainer.dynamic.seating import SeatRotation


def test_session_engine_alternates_seats():
    engine = SessionEngine(rng=random.Random(123), rotation=SeatRotation())
    first = engine.build_episode(0)
    second = engine.build_episode(1)

    assert first.hero_seat == "BB"
    assert first.villain_seat == "SB"
    assert second.hero_seat == "SB"
    assert second.villain_seat == "BB"


def test_session_engine_current_seats_matches_build():
    rotation = SeatRotation()
    engine = SessionEngine(rng=random.Random(321), rotation=rotation)

    for index in range(4):
        seats = engine.current_seats(index)
        episode = engine.build_episode(index)
        assert episode.hero_seat == seats.hero
        assert episode.villain_seat == seats.villain


def test_session_engine_cycle_repeats_bbsb_sequence():
    engine = SessionEngine(rng=random.Random(456), rotation=SeatRotation())
    expected = ["BB", "SB", "BB", "SB", "BB", "SB"]
    actual = [engine.build_episode(i).hero_seat for i in range(len(expected))]
    assert actual == expected


def test_session_engine_always_produces_preflop_node():
    engine = SessionEngine(rng=random.Random(999), rotation=SeatRotation())
    for index in range(6):
        episode = engine.build_episode(index)
        assert episode.nodes, "episode should contain nodes"
        first = episode.nodes[0]
        assert first.street == "preflop"
        assert first.actor == episode.hero_seat
