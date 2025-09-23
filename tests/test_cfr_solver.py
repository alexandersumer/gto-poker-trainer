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
        "cfr_payoffs": {
            "rival_actions": ["fold", "continue"],
            "hero": [fold_ev, continue_ev],
            "rival": [-fold_ev, -continue_ev],
        },
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


def test_local_cfr_backend_handles_multiple_rival_responses() -> None:
    options = [
        Option(
            key="Small bet",
            ev=0.0,
            why="",
            meta={
                "supports_cfr": True,
                "cfr_payoffs": {
                    "rival_actions": ["fold", "call", "raise"],
                    "hero": [0.8, -0.2, -1.0],
                    "rival": [-0.8, 0.2, 1.0],
                },
            },
        ),
        Option(
            key="Large bet",
            ev=0.0,
            why="",
            meta={
                "supports_cfr": True,
                "cfr_payoffs": {
                    "rival_actions": ["fold", "call", "raise"],
                    "hero": [0.5, 0.6, -0.4],
                    "rival": [-0.5, -0.6, 0.4],
                },
            },
        ),
    ]

    backend = LocalCFRBackend(LocalCFRConfig(iterations=150))
    refined = backend.refine(None, options)

    hero_mix = [opt.gto_freq for opt in refined]
    assert all(freq is not None for freq in hero_mix)
    assert abs(sum(freq for freq in hero_mix if freq is not None) - 1.0) < 1e-6

    rival_mix_sets = [opt.meta.get("cfr_rival_mix") for opt in refined if opt.meta]
    assert all(isinstance(mix, dict) and {"fold", "call", "raise"} <= set(mix.keys()) for mix in rival_mix_sets)
