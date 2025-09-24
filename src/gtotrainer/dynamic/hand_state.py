"""Shared helpers for manipulating per-hand mutable state.

The policy layer and session generator both update the mutable ``hand_state``
map while an episode progresses. Consolidating these helpers keeps the side
-effects consistent and makes future refactors easier to test.
"""

from __future__ import annotations

import logging
from typing import Any

from .episode import Node

__all__ = [
    "state_value",
    "recalculate_pot",
    "update_effective_stack",
    "apply_contribution",
    "set_street_pot",
]

logger = logging.getLogger(__name__)


def state_value(hand_state: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    """Return a numeric view of ``hand_state[key]`` with a fallback."""

    if not hand_state:
        return default
    value = hand_state.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.debug("Failed to coerce %s from hand_state; falling back to %s", key, default)
        return default


def recalculate_pot(hand_state: dict[str, Any]) -> float:
    """Update and return the pot using the tracked contributions."""

    if "hero_contrib" in hand_state and "rival_contrib" in hand_state:
        pot = state_value(hand_state, "hero_contrib") + state_value(hand_state, "rival_contrib")
    else:
        pot = state_value(hand_state, "pot")
    hand_state["pot"] = pot
    return pot


def update_effective_stack(hand_state: dict[str, Any]) -> float:
    """Refresh the effective stack and propagate it to cached nodes."""

    hero_stack = state_value(hand_state, "hero_stack")
    rival_stack = state_value(hand_state, "rival_stack")
    effective = min(hero_stack, rival_stack)
    hand_state["effective_stack"] = effective

    nodes = hand_state.get("nodes")
    if isinstance(nodes, dict):
        for node in nodes.values():
            if isinstance(node, Node):
                node.effective_bb = effective
    return effective


def apply_contribution(hand_state: dict[str, Any], role: str, amount: float) -> float:
    """Apply a bet/call to the relevant stack and contributions.

    Returns the amount that was actually applied (stack capped by availability).
    """

    if not hand_state or amount <= 0:
        return 0.0
    stack_key = f"{role}_stack"
    contrib_key = f"{role}_contrib"
    default_stack = float(hand_state.get("effective_stack", 100.0))
    stack = state_value(hand_state, stack_key, default_stack)
    if stack <= 0:
        logger.debug("Skipping contribution for %s; stack depleted", role)
        return 0.0
    applied = min(amount, stack)
    current_contrib = state_value(hand_state, contrib_key)
    hand_state[contrib_key] = current_contrib + applied
    hand_state[stack_key] = max(0.0, stack - applied)
    if applied < amount:
        logger.debug("Contribution for %s truncated from %.2f to %.2f due to stack limit", role, amount, applied)
    recalculate_pot(hand_state)
    update_effective_stack(hand_state)
    return applied


def set_street_pot(hand_state: dict[str, Any], street: str, pot: float) -> None:
    """Synchronise the cached node for ``street`` with the latest pot/stack."""

    nodes = hand_state.get("nodes")
    if not isinstance(nodes, dict):
        return
    node = nodes.get(street)
    if isinstance(node, Node):
        node.pot_bb = pot
        node.effective_bb = state_value(hand_state, "effective_stack", node.effective_bb)
