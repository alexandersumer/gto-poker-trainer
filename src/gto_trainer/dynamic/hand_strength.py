"""Shared heuristics for ranking two-card holdings by playability.

The score intentionally mirrors trends from contemporary heads-up solvers:
- Suited and connected holdings gain value for their board coverage.
- Offsuit, gappy hands — particularly with weak kickers — lose weight.

This keeps preflop heuristics (open ranges, defence mixes, rival modelling)
aligned so that every subsystem reasons about combo quality the same way.
"""

from __future__ import annotations

from typing import Tuple


def combo_playability_score(combo: Tuple[int, int]) -> float:
    """Return a deterministic strength metric for two hole cards.

    The base formulation follows classic rank ordering used across the
    (range_model, rival_strategy, preflop_mix) modules. We then add playability
    penalties calibrated against solver charts so weak offsuit hands fall back
    when open sizes grow. This keeps frequencies realistic without hard-coded
    fold overrides.
    """

    a, b = combo
    ra, rb = a // 4, b // 4
    suited = (a % 4) == (b % 4)
    high, low = (ra, rb) if ra >= rb else (rb, ra)

    score = high * 10 + low
    if high == low:
        score += 80 + high * 5
    if suited:
        score += 5
    gap = high - low - 1
    if high != low:
        if gap <= 0:
            score += 4
        elif gap == 1:
            score += 3
        elif gap == 2:
            score += 1
        elif gap >= 4:
            score -= gap

        # Extra weighting tuned to preflop solver guidance.
        offsuit_penalty = 0.0 if suited else 1.0
        if gap >= 2:
            score -= offsuit_penalty * (gap - 1.5)
        if not suited and low <= 4:
            score -= 3.4 - 0.35 * low
        if not suited and gap >= 2 and low <= 5:
            score -= 0.6 * (6 - low)
        if gap >= 5:
            score -= 0.5 * gap
        if not suited and high <= 9:
            score -= 0.5
        if not suited and gap >= 3:
            score -= 1.2 * (gap - 2)
        if not suited and high >= 10 and low <= 5 and gap >= 3:
            score -= 6.0

    return float(score)
