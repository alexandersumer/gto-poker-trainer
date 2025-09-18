"""Episode domain models used across the dynamic engine.

Keeping the node and episode dataclasses in a dedicated module makes them
available to both the generator logic and the policy layer without creating a
large web of imports.  The types here intentionally stay lightweight so they
can be serialised and passed between the various adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """Represents a single decision point within an episode."""

    street: str
    description: str
    pot_bb: float
    effective_bb: float
    hero_cards: list[int]
    board: list[int]
    actor: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """Collection of sequential nodes plus seat metadata."""

    nodes: list[Node]
    hero_seat: str
    villain_seat: str

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("Episode must contain at least one node")
