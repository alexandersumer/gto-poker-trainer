from __future__ import annotations

import random
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
    solver_csv: str | None = None,
    _input_fn=input,
) -> None:
    presenter = RichPresenter(no_color=no_color)
    base_provider = _DynamicOptions()
    option_provider: OptionProvider
    if solver_csv:
        solver = CSVStrategyOracle(solver_csv)
        option_provider = CompositeOptionProvider(primary=solver, fallback=base_provider)
    else:
        option_provider = base_provider

    # Session seed strategy: deterministic progression if a seed is provided;
    # otherwise use secure randomness per session.
    session_rng = None if seed is None else random.Random(seed)

    while True:
        presenter.quit_requested = False
        actual_seed = session_rng.getrandbits(32) if session_rng is not None else secrets.randbits(32)
        run_core(
            generator=_DynamicGenerator(),
            option_provider=option_provider,
            presenter=presenter,
            seed=actual_seed,
            hands=hands,
            mc_trials=mc_trials,
        )
        # If the user quit mid-session, exit without prompting for a new session.
        if presenter.quit_requested:
            break

        # Smooth UX: ask whether to start a new session rather than auto-starting.
        # Accept Enter/"y"/"yes" as affirmative; "n"/"no"/"q"/"quit" will exit.
        while True:
            reply = _input_fn("Start a new session? [Y/n]: ").strip().lower()
            if reply in {"", "y", "yes"}:
                break  # start another loop iteration (new session)
            if reply in {"n", "no", "q", "quit", "exit"}:
                return
            # Invalid input; reprompt briefly.
            print("Please answer 'y' to continue or 'n' to quit.")
