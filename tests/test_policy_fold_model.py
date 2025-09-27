from __future__ import annotations

import gtotrainer.dynamic.policy as pol


def _entries(*equities: float) -> list[tuple[float, float]]:
    return [(eq, 1.0) for eq in equities]


def test_fold_model_size_pressure_monotonic() -> None:
    entries = _entries(0.68, 0.61, 0.54)
    base = pol._fold_params({}, pot=4.0, bet=1.0, board=[])
    small = pol._fold_continue_stats(entries, 0.3, params=base)[0]
    larger = pol._fold_params({}, pot=4.0, bet=2.0, board=[])
    big = pol._fold_continue_stats(entries, 0.3, params=larger)[0]
    assert big > small


def test_fold_model_texture_reduces_folds_on_wet_boards() -> None:
    entries = _entries(0.64, 0.58, 0.5)
    dry_params = pol.FoldModelParams(size_ratio=0.75, texture=0.3, spr=3.0)
    wet_params = pol.FoldModelParams(size_ratio=0.75, texture=0.8, spr=3.0)
    dry_fe = pol._fold_continue_stats(entries, 0.33, params=dry_params)[0]
    wet_fe = pol._fold_continue_stats(entries, 0.33, params=wet_params)[0]
    assert wet_fe < dry_fe


def test_fold_model_adaptation_adjusts_thresh() -> None:
    entries = _entries(0.68, 0.63, 0.59)
    neutral = pol.FoldModelParams(size_ratio=0.8, texture=0.5, spr=2.5)
    passive = pol.FoldModelParams(size_ratio=0.8, texture=0.5, spr=2.5, adapt_aggr=0, adapt_passive=5)
    aggressive = pol.FoldModelParams(size_ratio=0.8, texture=0.5, spr=2.5, adapt_aggr=5, adapt_passive=0)
    neutral_fe = pol._fold_continue_stats(entries, 0.32, params=neutral)[0]
    passive_fe = pol._fold_continue_stats(entries, 0.32, params=passive)[0]
    aggressive_fe = pol._fold_continue_stats(entries, 0.32, params=aggressive)[0]
    assert aggressive_fe < neutral_fe < passive_fe
