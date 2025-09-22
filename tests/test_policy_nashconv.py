from __future__ import annotations

import random

from gtotrainer.dynamic.policy import options_for
from gtotrainer.dynamic.seating import SeatRotation
from gtotrainer.features.session.analysis import estimate_nashconv
from gtotrainer.features.session.engine import SessionEngine


def test_preflop_canonical_nashconv_below_threshold() -> None:
    rng = random.Random(1234)
    engine = SessionEngine(rng=rng, rotation=SeatRotation())
    episode = engine.build_episode(0, stacks_bb=93.0)
    node = episode.nodes[0]
    options = options_for(node, random.Random(4321), 120)
    conv = estimate_nashconv(options)
    assert conv < 0.75


def test_nashconv_zero_without_cfr_metadata() -> None:
    dummy_options = [
        type("Opt", (), {"ev": 1.0, "meta": {}})(),
        type("Opt", (), {"ev": 0.8, "meta": {}})(),
    ]
    assert estimate_nashconv(dummy_options) == 0.0
