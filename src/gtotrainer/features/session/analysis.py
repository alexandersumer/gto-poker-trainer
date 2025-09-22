from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ...core.formatting import format_option_label
from ...core.models import Option
from ...dynamic.cards import format_card_ascii
from ...dynamic.generator import Node
from .schemas import (
    ActionEVBreakdown,
    NodeAnalysisPayload,
    OptionAnalysisPayload,
    VillainRangeEntry,
)

Percent = float


def _safe_float(meta: Mapping[str, object] | None, key: str) -> float | None:
    if not meta:
        return None
    value = meta.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_probability(value: float | None) -> float | None:
    if value is None:
        return None
    if value != value:
        return None
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _combo_to_text(combo: Sequence[int]) -> str:
    cards = [format_card_ascii(int(card), upper=True) for card in combo]
    return "".join(cards)


@dataclass(frozen=True)
class _BreakdownInputs:
    fold_pct: Percent | None
    continue_pct: Percent | None
    hero_ev_fold: float | None
    hero_ev_continue: float | None
    hero_invest: float | None
    villain_invest: float | None
    pot_before: float | None
    pot_if_called: float | None
    hero_eq_continue: float | None
    villain_continue_range: list[Sequence[int]]
    villain_continue_total: int | None


def _prepare_breakdown(meta: Mapping[str, object] | None) -> _BreakdownInputs:
    fold_pct = _clamp_probability(_safe_float(meta, "rival_fe"))
    continue_pct = _clamp_probability(_safe_float(meta, "rival_continue_ratio"))
    if fold_pct is not None and continue_pct is None:
        continue_pct = max(0.0, 1.0 - fold_pct)
    hero_ev_fold = _safe_float(meta, "hero_ev_fold")
    hero_ev_continue = _safe_float(meta, "hero_ev_continue")
    hero_invest = _safe_float(meta, "hero_invest")
    villain_invest = _safe_float(meta, "villain_invest")
    pot_before = _safe_float(meta, "pot_before")
    pot_if_called = _safe_float(meta, "pot_if_called")
    hero_eq_continue = _safe_float(meta, "hero_eq_continue")
    if hero_eq_continue is None and hero_ev_continue is not None and pot_if_called and hero_invest is not None:
        denom = pot_if_called
        if denom > 0:
            hero_eq_continue = (hero_ev_continue + hero_invest) / denom
    continue_range: list[Sequence[int]] = []
    if isinstance(meta, Mapping):
        raw_range = meta.get("rival_continue_range")
        if isinstance(raw_range, (list, tuple)):
            for entry in raw_range:
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    try:
                        a = int(entry[0])
                        b = int(entry[1])
                    except (TypeError, ValueError):
                        continue
                    continue_range.append((a, b))
    continue_total: int | None = None
    if isinstance(meta, Mapping):
        profile = meta.get("rival_profile")
        if isinstance(profile, Mapping):
            total_raw = profile.get("continue_count")
            try:
                continue_total = int(total_raw)
            except (TypeError, ValueError):
                continue_total = None
    if continue_total is None:
        continue_total = len(continue_range) or None
    return _BreakdownInputs(
        fold_pct=fold_pct,
        continue_pct=continue_pct,
        hero_ev_fold=hero_ev_fold,
        hero_ev_continue=hero_ev_continue,
        hero_invest=hero_invest,
        villain_invest=villain_invest,
        pot_before=pot_before,
        pot_if_called=pot_if_called,
        hero_eq_continue=hero_eq_continue,
        villain_continue_range=continue_range,
        villain_continue_total=continue_total,
    )


def _build_breakdown(meta: Mapping[str, object] | None) -> ActionEVBreakdown | None:
    inputs = _prepare_breakdown(meta)
    fold_term = None
    continue_term = None
    if inputs.fold_pct is not None and inputs.hero_ev_fold is not None:
        fold_term = inputs.fold_pct * inputs.hero_ev_fold
    continue_prob: float | None = inputs.continue_pct
    if continue_prob is None and inputs.fold_pct is not None:
        continue_prob = max(0.0, 1.0 - inputs.fold_pct)
    if continue_prob is not None and inputs.hero_ev_continue is not None:
        continue_term = continue_prob * inputs.hero_ev_continue
    villain_sample: list[VillainRangeEntry] | None = None
    if inputs.villain_continue_range:
        sample_limit = 12
        villains = [_combo_to_text(combo) for combo in inputs.villain_continue_range[:sample_limit]]
        villain_sample = [VillainRangeEntry(combo=combo) for combo in villains]
    if (
        inputs.fold_pct is None
        and inputs.continue_pct is None
        and fold_term is None
        and continue_term is None
        and inputs.hero_eq_continue is None
        and inputs.pot_before is None
        and inputs.pot_if_called is None
        and inputs.hero_invest is None
        and inputs.villain_invest is None
        and not villain_sample
    ):
        return None
    return ActionEVBreakdown(
        fold_pct=inputs.fold_pct,
        continue_pct=inputs.continue_pct,
        fold_term=fold_term,
        continue_term=continue_term,
        hero_equity_vs_continue=inputs.hero_eq_continue,
        pot_before=inputs.pot_before,
        pot_if_called=inputs.pot_if_called,
        hero_invest=inputs.hero_invest,
        villain_invest=inputs.villain_invest,
        villain_continue_total=inputs.villain_continue_total,
        villain_continue_sample=villain_sample,
    )


def build_node_analysis(node: Node, options: Sequence[Option]) -> NodeAnalysisPayload | None:
    option_list = list(options)
    if not option_list:
        return None
    best = max(option_list, key=lambda opt: opt.ev)
    best_ev = best.ev
    best_key = best.key
    analyses: list[OptionAnalysisPayload] = []
    for opt in option_list:
        label = format_option_label(node, opt)
        ev_delta = max(0.0, best_ev - opt.ev)
        is_best = abs(opt.ev - best_ev) <= 1e-9
        breakdown = _build_breakdown(opt.meta if hasattr(opt, "meta") else None)
        analyses.append(
            OptionAnalysisPayload(
                key=opt.key,
                label=label,
                ev=opt.ev,
                ev_delta=ev_delta,
                is_best=is_best,
                breakdown=breakdown,
            )
        )
    return NodeAnalysisPayload(best_key=best_key, options=analyses)


def estimate_nashconv(options: Sequence[Option]) -> float:
    eligible: list[Option] = []
    hero_probs: list[float] = []
    for opt in options:
        meta = opt.meta or {}
        prob = meta.get("cfr_probability")
        if prob is None:
            continue
        try:
            hero_probs.append(max(0.0, float(prob)))
            eligible.append(opt)
        except (TypeError, ValueError):
            continue
    if not eligible:
        return 0.0
    total = sum(hero_probs)
    if total <= 1e-12:
        hero_probs = [1.0 / len(eligible)] * len(eligible)
    else:
        hero_probs = [p / total for p in hero_probs]
    hero_eq_ev = sum(prob * opt.ev for prob, opt in zip(hero_probs, eligible, strict=False))
    hero_best_ev = max(opt.ev for opt in options)
    hero_gap = max(0.0, hero_best_ev - hero_eq_ev)

    villain_fold_ev = 0.0
    villain_continue_ev = 0.0
    have_villain = True
    for prob, opt in zip(hero_probs, eligible, strict=False):
        meta = opt.meta or {}
        try:
            fold_ev = float(meta["rival_ev_fold"])
            continue_ev = float(meta["rival_ev_continue"])
        except (KeyError, TypeError, ValueError):
            have_villain = False
            break
        villain_fold_ev += prob * fold_ev
        villain_continue_ev += prob * continue_ev
    villain_gap = 0.0
    if have_villain:
        mix_meta = eligible[0].meta if isinstance(eligible[0].meta, Mapping) else None
        mix = mix_meta.get("cfr_rival_mix") if isinstance(mix_meta, Mapping) else None
        if isinstance(mix, Mapping):
            try:
                mix_fold = float(mix.get("fold", 0.0))
                mix_continue = float(mix.get("continue", 0.0))
            except (TypeError, ValueError):
                mix_fold = 1.0
                mix_continue = 0.0
        else:
            mix_fold = 1.0
            mix_continue = 0.0
        total_mix = mix_fold + mix_continue
        if total_mix <= 1e-12:
            mix_fold, mix_continue = 1.0, 0.0
        else:
            mix_fold /= total_mix
            mix_continue /= total_mix
        villain_eq = mix_fold * villain_fold_ev + mix_continue * villain_continue_ev
        villain_best = max(villain_fold_ev, villain_continue_ev)
        villain_gap = max(0.0, villain_best - villain_eq)
    return hero_gap + villain_gap


__all__ = ["build_node_analysis", "estimate_nashconv"]
