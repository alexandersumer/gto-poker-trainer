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

# Accuracy weighting curve parameters.
YELLOW_POT_MULTIPLIER = 0.05
YELLOW_FALLBACK = 0.35
MIN_YELLOW_BAND = 0.08
YELLOW_GAMMA = 0.75
RED_DECAY = 18.0
HARD_MISTAKE_RATIO = 0.5


@dataclass(frozen=True)
class SummaryStats:
    hands: int
    decisions: int
    hits: int
    accuracy_points: float
    accuracy_pct: float
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


def _policy_mismatch(record: Mapping[str, Any]) -> bool:
    chosen_flag = record.get("chosen_out_of_policy")
    best_flag = record.get("best_out_of_policy")
    return bool(chosen_flag) and best_flag is not True


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


def decision_accuracy(record: Mapping[str, Any]) -> float:
    pot = _extract_pot(record)
    ev_loss = _ev_loss(record)
    score = decision_score(record)
    if _policy_mismatch(record):
        return 0.0
    within_noise = _within_noise(record, score=score)
    if within_noise:
        return 1.0
    return _ev_band_credit(ev_loss=ev_loss, pot=pot)


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

    if _policy_mismatch(record):
        return False

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
            accuracy_points=0.0,
            accuracy_pct=0.0,
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
    accuracy_points = 0.0
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

        within_noise = _within_noise(record, score=score)
        if within_noise:
            hits += 1

        accuracy_points += decision_accuracy(record)

    total_weight = sum(weights)
    weighted_loss_ratio = sum(ratio * weight for ratio, weight in zip(loss_ratios, weights, strict=False))
    avg_loss_ratio = (weighted_loss_ratio / total_weight) if total_weight > 0 else 0.0
    avg_loss_pct = 100.0 * avg_loss_ratio

    weighted_score = sum(score * weight for score, weight in zip(decision_scores, weights, strict=False))
    score_pct = (weighted_score / total_weight) if total_weight > 0 else 0.0

    avg_ev_lost = total_ev_lost / decisions
    accuracy_pct = (100.0 * accuracy_points / decisions) if decisions else 0.0

    return SummaryStats(
        hands=hands,
        decisions=decisions,
        hits=hits,
        accuracy_points=accuracy_points,
        accuracy_pct=accuracy_pct,
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


def _ev_band_credit(*, ev_loss: float, pot: float) -> float:
    pot = max(pot, MIN_POT)
    if ev_loss <= 0.0:
        return 1.0

    noise_floor = _ev_noise_floor(pot)
    if ev_loss <= noise_floor:
        return 1.0

    yellow_upper = max(YELLOW_POT_MULTIPLIER * pot, YELLOW_FALLBACK)
    if yellow_upper <= noise_floor:
        yellow_upper = noise_floor + MIN_YELLOW_BAND

    if ev_loss <= yellow_upper:
        span = max(yellow_upper - noise_floor, MIN_POT)
        t = (ev_loss - noise_floor) / span
        return max(0.0, min(1.0, 1.0 - 0.5 * (t**YELLOW_GAMMA)))

    over = ev_loss - yellow_upper
    ratio = over / pot
    if ratio >= HARD_MISTAKE_RATIO:
        return 0.0

    credit = 0.5 * math.exp(-RED_DECAY * ratio)
    return max(0.0, min(1.0, credit))


def ev_conservation_diagnostics(
    records: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 1e-6,
) -> dict[str, float | bool]:
    """Return aggregate EV conservation metrics for audit/CI hooks.

    ``records`` should contain ``best_ev``/``chosen_ev`` pairs. Optional keys
    ``best_baseline_ev`` and ``chosen_baseline_ev`` mirror the behaviour of
    :func:`effective_option_ev` by clamping downward revisions to their stored
    baselines so downstream totals remain comparable.
    """

    if not records:
        return {
            "total_best": 0.0,
            "total_chosen": 0.0,
            "total_ev_lost": 0.0,
            "delta": 0.0,
            "within_tolerance": True,
        }

    def _clamp(value: float, baseline: float | None) -> float:
        if baseline is None:
            return value
        try:
            return max(value, float(baseline))
        except (TypeError, ValueError):
            return value

    total_best = 0.0
    total_chosen = 0.0
    total_ev_lost = 0.0

    for record in records:
        best = _as_float(record.get("best_ev"))
        chosen = _as_float(record.get("chosen_ev"))
        best = _clamp(best, record.get("best_baseline_ev"))
        chosen = _clamp(chosen, record.get("chosen_baseline_ev"))
        total_best += best
        total_chosen += chosen
        total_ev_lost += max(0.0, best - chosen)

    delta = (total_best - total_chosen) - total_ev_lost
    return {
        "total_best": total_best,
        "total_chosen": total_chosen,
        "total_ev_lost": total_ev_lost,
        "delta": delta,
        "within_tolerance": math.isclose(0.0, delta, abs_tol=tolerance, rel_tol=0.0),
    }
