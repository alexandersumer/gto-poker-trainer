from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Option:
    key: str
    ev: float
    why: str
    # Optional: solver frequency signal for grading/UX
    gto_freq: float | None = None
    # If True, selecting this option ends the current hand immediately
    ends_hand: bool = False
