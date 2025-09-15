from __future__ import annotations

import secrets

from .core.engine_core import run_core
from .core.interfaces import EpisodeGenerator, OptionProvider
from .dynamic.generator import generate_episode
from .dynamic.policy import options_for
from .solver.oracle import CompositeOptionProvider, CSVStrategyOracle
from .ui.presenters import RichPresenter


class _DynamicGenerator(EpisodeGenerator):
    def generate(self, rng):
        return generate_episode(rng)


class _DynamicOptions(OptionProvider):
    def options(self, node, rng, mc_trials):
        return options_for(node, rng, mc_trials)


def run_play(
    seed: int | None = None,
    hands: int = 1,
    mc_trials: int = 200,
    no_color: bool = False,
    force_color: bool = False,
    solver_csv: str | None = None,
    _input_fn=input,
) -> None:
    presenter = RichPresenter(no_color=no_color, force_color=force_color)
    base_provider = _DynamicOptions()
    option_provider: OptionProvider
    if solver_csv:
        solver = CSVStrategyOracle(solver_csv)
        option_provider = CompositeOptionProvider(primary=solver, fallback=base_provider)
    else:
        option_provider = base_provider
    # Choose a random seed when none is provided for varied sessions
    actual_seed = seed if seed is not None else secrets.randbits(32)
    run_core(
        generator=_DynamicGenerator(),
        option_provider=option_provider,
        presenter=presenter,
        seed=actual_seed,
        hands=hands,
        mc_trials=mc_trials,
    )
