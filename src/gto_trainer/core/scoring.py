from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MIN_POT = 1e-6

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
    ev_loss = max(0.0, _as_float(record.get("best_ev", 0.0)) - _as_float(record.get("chosen_ev", 0.0)))
    return ev_loss / max(pot, MIN_POT)


def decision_score(record: Mapping[str, Any]) -> float:
    pot = _extract_pot(record)
    ev_loss = max(0.0, _as_float(record.get("best_ev", 0.0)) - _as_float(record.get("chosen_ev", 0.0)))
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
    total_ev_lost = total_ev_best - total_ev_chosen
    hits = sum(1 for r in records if r.get("chosen_key") == r.get("best_key"))
    hand_ids = {r.get("hand_index", idx) for idx, r in enumerate(records)}
    hands = len(hand_ids) if hand_ids else decisions

    pots = [_extract_pot(r) for r in records]
    weights = [max(p, MIN_POT) for p in pots]
    total_weight = sum(weights)

    loss_ratios = [decision_loss_ratio(r) for r in records]
    weighted_loss_ratio = sum(
        ratio * weight for ratio, weight in zip(loss_ratios, weights, strict=False)
    )
    avg_loss_ratio = (weighted_loss_ratio / total_weight) if total_weight > 0 else 0.0
    avg_loss_pct = 100.0 * avg_loss_ratio

    decision_scores = [decision_score(r) for r in records]
    weighted_score = sum(
        score * weight for score, weight in zip(decision_scores, weights, strict=False)
    )
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
    dynamic = EV_NOISE_FLOOR_PCT * max(pot, MIN_POT)
    return max(EV_NOISE_FLOOR_BASE, dynamic)


def _ratio_noise_floor(pot: float) -> float:
    dynamic = RATIO_NOISE_FLOOR_PCT * max(pot, MIN_POT)
    return max(RATIO_NOISE_FLOOR_BASE, dynamic)
