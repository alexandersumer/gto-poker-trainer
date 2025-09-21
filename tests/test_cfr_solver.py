from __future__ import annotations

from gtotrainer.core.models import Option
from gtotrainer.dynamic.cfr import LocalCFRBackend, LocalCFRConfig


def _option(key: str, fold_ev: float, continue_ev: float) -> Option:
    meta = {
        "supports_cfr": True,
        "hero_ev_fold": fold_ev,
        "hero_ev_continue": continue_ev,
        "rival_ev_fold": -fold_ev,
        "rival_ev_continue": -continue_ev,
    }
    return Option(key=key, ev=fold_ev, why="", meta=meta)


def test_local_cfr_backend_updates_strategy_profile() -> None:
    opts = [_option("Small bet", 1.0, -0.5), _option("Big bet", 0.5, 1.5)]
    baseline = [opt.ev for opt in opts]
    backend = LocalCFRBackend(LocalCFRConfig(iterations=100))
    refined = backend.refine(None, opts)
    hero_mix = [opt.gto_freq for opt in refined]
    assert all(freq is not None for freq in hero_mix)
    assert abs(sum(freq for freq in hero_mix if freq is not None) - 1.0) < 1e-6
    assert refined[0].ev != baseline[0]  # CFR adjusts baseline EV
    assert refined[1].ev != baseline[1]
