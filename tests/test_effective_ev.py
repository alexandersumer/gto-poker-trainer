from __future__ import annotations

import math

import pytest

from gtotrainer.core.ev import effective_ev, effective_option_ev
from gtotrainer.core.models import Option


def _option(ev: float, baseline: float | None = None) -> Option:
    meta = {} if baseline is None else {"baseline_ev": baseline}
    return Option(key="test", ev=ev, why="", meta=meta)


@pytest.mark.parametrize(
    ("ev", "baseline", "expected"),
    [
        (1.2, None, 1.2),
        (1.2, 0.8, 1.2),
        (0.5, 1.1, 1.1),
        (0.0, 0.0, 0.0),
    ],
)
def test_effective_option_ev_clamps_to_baseline(ev: float, baseline: float | None, expected: float) -> None:
    option = _option(ev, baseline)
    assert effective_option_ev(option) == pytest.approx(expected)


def test_effective_option_ev_handles_invalid_baseline() -> None:
    option = Option(key="bad", ev=1.3, why="", meta={"baseline_ev": "oops"})
    assert effective_option_ev(option) == pytest.approx(1.3)


def test_effective_ev_scalar_helper_matches_option_variant() -> None:
    option = _option(-0.25, 0.1)
    as_option = effective_option_ev(option)
    as_scalar = effective_ev(option.ev, baseline=option.meta["baseline_ev"])
    assert math.isclose(as_option, as_scalar)
