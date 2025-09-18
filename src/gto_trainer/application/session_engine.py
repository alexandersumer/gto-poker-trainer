"""Session engine primitives.

A compact orchestration layer keeps random number generation, seat rotation and
episode construction together.  This makes the broader session service easier
to test while avoiding repetition of the episode-generation rules.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..dynamic.generator import Episode, generate_episode
from ..dynamic.seating import SeatAssignment, SeatRotation


@dataclass
class SessionEngine:
    """Wraps RNG and seat rotation for deterministic episode generation."""

    rng: random.Random
    rotation: SeatRotation

    def build_episode(
        self,
        index: int,
        *,
        stacks_bb: float = 100.0,
        sb: float = 0.5,
        bb: float = 1.0,
    ) -> Episode:
        seats = self.rotation.assignment_for(index)
        return generate_episode(
            self.rng,
            seat_assignment=seats,
            stacks_bb=stacks_bb,
            sb=sb,
            bb=bb,
        )

    def current_seats(self, index: int) -> SeatAssignment:
        return self.rotation.assignment_for(index)
