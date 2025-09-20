from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Option:
    key: str
    ev: float
    why: str
    # Optional: solver frequency signal for grading/UX
    gto_freq: float | None = None
    # If True, selecting this option ends the current hand immediately
    ends_hand: bool = False
    # Internal metadata used by the engine to resolve downstream state changes.
    meta: dict[str, Any] | None = None
    # Optional runtime note describing what happened after the action resolved.
    resolution_note: str | None = None


@dataclass
class OptionResolution:
    """Structured outcome returned after an option is applied in the engine."""

    hand_ended: bool = False
    note: str | None = None
    reveal_rival: bool = False
