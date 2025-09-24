from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MIN_POT = 1e-6
NOISE_EPSILON = 1e-9

# Ratio scoring parameters (dimensionless).
RATIO_NOISE_FLOOR_BASE = 0.003  # 0.3% baseline solver noise
RATIO_NOISE_FLOOR_PCT = 0.00075  # additional tolerance per bb of pot
RATIO_DECAY = 20.0  # Controls how aggressively large mistakes are penalised

# EV-based scoring tuned around practical solver error margins.
EV_NOISE_FLOOR_BASE = 0.02  # Ignore < 0.02 bb diffs as solver noise
EV_NOISE_FLOOR_PCT = 0.0025  # Additional tolerance (0.25% of pot in bb)
EV_DECAY = 2.0  # Higher = harsher punishment for bigger EV mistakes


@dataclass(frozen=True)
class SummaryStats:
    hands: int
    decisions: int
    hits: int
    total_ev_chosen: float
    total_ev_best: float
    total_ev_lost: float
    avg_ev_lost: float
    avg_loss_pct: float
    score_pct: float


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _ev_loss(record: Mapping[str, Any]) -> float:
    """Return the non-negative EV gap between the best and chosen actions."""

    best = _as_float(record.get("best_ev", 0.0))
    chosen = _as_float(record.get("chosen_ev", 0.0))
    return max(0.0, best - chosen)


def _extract_pot(record: Mapping[str, Any]) -> float:
    pot = _as_float(record.get("pot_bb", 0.0))
    if pot > MIN_POT:
        return pot
    room = _as_float(record.get("room_ev", 0.0))
    if room > MIN_POT:
        return room
    best = abs(_as_float(record.get("best_ev", 0.0)))
    chosen = abs(_as_float(record.get("chosen_ev", 0.0)))
    fallback = max(best, chosen)
    return fallback if fallback > MIN_POT else 1.0


def decision_loss_ratio(record: Mapping[str, Any]) -> float:
    pot = _extract_pot(record)
    ev_loss = _ev_loss(record)
    return ev_loss / max(pot, MIN_POT)


def decision_score(record: Mapping[str, Any]) -> float:
    pot = _extract_pot(record)
    ev_loss = _ev_loss(record)
    score_ev = _score_for_ev_loss(ev_loss, noise_floor=_ev_noise_floor(pot))
    score_ratio = _score_for_ratio(decision_loss_ratio(record), noise_floor=_ratio_noise_floor(pot))
    return min(score_ev, score_ratio)


def _score_for_ratio(ratio: float, *, noise_floor: float, decay: float = RATIO_DECAY) -> float:
    adjusted = max(0.0, ratio - noise_floor)
    raw = 100.0 * math.exp(-decay * adjusted)
    if raw < 0.001:
        return 0.0
    if raw > 100.0:
        return 100.0
    return raw


def _score_for_ev_loss(ev_loss: float, *, noise_floor: float, decay: float = EV_DECAY) -> float:
    adjusted = max(0.0, ev_loss - noise_floor)
    raw = 100.0 * math.exp(-decay * adjusted)
    if raw < 0.001:
        return 0.0
    return min(raw, 100.0)


def _within_noise(record: Mapping[str, Any], *, score: float | None = None) -> bool:
    """Return True when the chosen action is within solver noise tolerances."""

    if record.get("chosen_key") == record.get("best_key"):
        return True

    if score is not None and score >= 99.999:
        return True

    pot = _extract_pot(record)
    ev_loss = _ev_loss(record)
    if ev_loss == 0.0:
        return True

    denom = max(pot, MIN_POT)
    ratio = ev_loss / denom
    ev_tolerance = _ev_noise_floor(pot) + NOISE_EPSILON
    ratio_tolerance = _ratio_noise_floor(pot) + NOISE_EPSILON
    return ev_loss <= ev_tolerance and ratio <= ratio_tolerance


def summarize_records(records: Sequence[Mapping[str, Any]]) -> SummaryStats:
    if not records:
        return SummaryStats(
            hands=0,
            decisions=0,
            hits=0,
            total_ev_chosen=0.0,
            total_ev_best=0.0,
            total_ev_lost=0.0,
            avg_ev_lost=0.0,
            avg_loss_pct=0.0,
            score_pct=0.0,
        )

    decisions = len(records)
    total_ev_best = sum(_as_float(r.get("best_ev", 0.0)) for r in records)
    total_ev_chosen = sum(_as_float(r.get("chosen_ev", 0.0)) for r in records)
    total_ev_lost = 0.0
    hits = 0
    hand_ids = {r.get("hand_index", idx) for idx, r in enumerate(records)}
    hands = len(hand_ids) if hand_ids else decisions

    weights: list[float] = []
    loss_ratios: list[float] = []
    decision_scores: list[float] = []

    for record in records:
        pot = _extract_pot(record)
        weight = max(pot, MIN_POT)
        weights.append(weight)

        ev_loss = _ev_loss(record)
        total_ev_lost += ev_loss

        ratio = ev_loss / weight
        loss_ratios.append(ratio)

        score_ev = _score_for_ev_loss(ev_loss, noise_floor=_ev_noise_floor(pot))
        score_ratio = _score_for_ratio(ratio, noise_floor=_ratio_noise_floor(pot))
        score = min(score_ev, score_ratio)
        decision_scores.append(score)

        if _within_noise(record, score=score):
            hits += 1

    total_weight = sum(weights)
    weighted_loss_ratio = sum(ratio * weight for ratio, weight in zip(loss_ratios, weights, strict=False))
    avg_loss_ratio = (weighted_loss_ratio / total_weight) if total_weight > 0 else 0.0
    avg_loss_pct = 100.0 * avg_loss_ratio

    weighted_score = sum(score * weight for score, weight in zip(decision_scores, weights, strict=False))
    score_pct = (weighted_score / total_weight) if total_weight > 0 else 0.0

    avg_ev_lost = total_ev_lost / decisions

    return SummaryStats(
        hands=hands,
        decisions=decisions,
        hits=hits,
        total_ev_chosen=total_ev_chosen,
        total_ev_best=total_ev_best,
        total_ev_lost=total_ev_lost,
        avg_ev_lost=avg_ev_lost,
        avg_loss_pct=avg_loss_pct,
        score_pct=score_pct,
    )


def _ev_noise_floor(pot: float) -> float:
    pot_scaled = max(pot, MIN_POT)
    return EV_NOISE_FLOOR_BASE + EV_NOISE_FLOOR_PCT * pot_scaled


def _ratio_noise_floor(pot: float) -> float:
    pot_scaled = max(pot, MIN_POT)
    floor = RATIO_NOISE_FLOOR_BASE + RATIO_NOISE_FLOOR_PCT * pot_scaled
    return min(floor, 0.99)
