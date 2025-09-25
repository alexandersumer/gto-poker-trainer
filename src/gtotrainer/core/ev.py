"""Utilities for working with EV values shared across subsystems."""

from __future__ import annotations

from typing import Any

from .models import Option


def effective_option_ev(option: Option) -> float:
    """Return the EV for an option after applying baseline guards.

    The dynamic solver can nudge EV values below pre-computed baselines when it
    refines strategies.  Downstream consumers (scoring, UI feedback, summary
    stats) expect EV deltas to never exceed the original room that was
    available.  This helper mirrors the logic that was previously embedded in
    multiple places: it falls back to ``option.meta['baseline_ev']`` when the
    refined EV dips below the baseline, and gracefully handles malformed meta
    payloads.
    """

    meta: dict[str, Any] | None = option.meta
    baseline = None
    if meta is not None:
        baseline = meta.get("baseline_ev")

    value = float(option.ev)

    if baseline is None:
        return value

    try:
        return max(value, float(baseline))
    except (TypeError, ValueError):
        return value


def effective_ev(raw_ev: float, *, baseline: float | None) -> float:
    """Clamp ``raw_ev`` to ``baseline`` when the latter is provided.

    This is the scalar variant of :func:`effective_option_ev`.  It exists to
    support record transformations that operate on primitive floats rather than
    :class:`Option` instances.
    """

    if baseline is None:
        return raw_ev

    try:
        return max(raw_ev, float(baseline))
    except (TypeError, ValueError):
        return raw_ev
