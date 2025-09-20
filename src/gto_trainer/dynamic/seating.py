"""Seat rotation helpers shared across the trainer.

The trainer always alternates hero positions between the small blind (SB) and
big blind (BB).  Previously that logic lived in the session service module,
which made it harder to reuse and reason about.  Consolidating it here keeps
the seating rules close to the rest of the dynamic engine code and allows
other components (e.g. tests, future simulators) to reuse the same rotation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

SB = "SB"
BB = "BB"


@dataclass(frozen=True)
class SeatAssignment:
    """Hero/rival seat mapping for a single hand."""

    hero: str
    rival: str

    def swap(self) -> SeatAssignment:
        return SeatAssignment(hero=self.rival, rival=self.hero)


class SeatRotation:
    """Deterministic two-player seat rotation."""

    def __init__(self, order: Iterable[str] | None = None) -> None:
        seats: tuple[str, ...] = tuple(order or (BB, SB))
        if len(seats) != 2 or set(seats) != {SB, BB}:
            raise ValueError("Seat rotation must contain SB and BB exactly once")
        self._order = seats

    def assignment_for(self, hand_index: int) -> SeatAssignment:
        hero = self._order[hand_index % len(self._order)]
        rival = SB if hero == BB else BB
        return SeatAssignment(hero=hero, rival=rival)


DEFAULT_ROTATION = SeatRotation()
