from __future__ import annotations

import random
from time import perf_counter

from gto_trainer.core.engine_core import run_core
from gto_trainer.core.interfaces import EpisodeGenerator, OptionProvider, Presenter
from gto_trainer.core.models import Option, OptionResolution
from gto_trainer.dynamic.generator import Episode, Node, generate_episode
from gto_trainer.dynamic.policy import options_for, resolve_for


class _QuickGenerator(EpisodeGenerator):
    def generate(self, rng: random.Random) -> Episode:  # type: ignore[override]
        return generate_episode(rng)


class _QuickProvider(OptionProvider):
    def options(self, node: Node, rng: random.Random, mc_trials: int) -> list[Option]:  # type: ignore[override]
        return options_for(node, rng, mc_trials)

    def resolve(self, node: Node, chosen: Option, rng: random.Random) -> OptionResolution:  # type: ignore[override]
        return resolve_for(node, chosen, rng)


class _TestPresenter(Presenter):
    def __init__(self) -> None:
        self._choices: list[int] = []

    def start_session(self, _total_hands: int) -> None:  # noqa: D401
        return None

    def start_hand(self, _hand_index: int, _total_hands: int) -> None:  # noqa: D401
        return None

    def show_node(self, _node: Node, _options: list[str]) -> None:  # noqa: D401
        self._choices.append(0)

    def prompt_choice(self, _n: int) -> int:  # noqa: D401
        return 0

    def step_feedback(self, _node: Node, _chosen: Option, _best: Option) -> None:  # noqa: D401
        return None

    def summary(self, _records: list[dict]) -> None:  # noqa: D401
        return None


def test_run_core_finishes_quickly():
    presenter = _TestPresenter()
    start = perf_counter()
    run_core(
        generator=_QuickGenerator(),
        option_provider=_QuickProvider(),
        presenter=presenter,
        seed=42,
        hands=1,
        mc_trials=120,
    )
    elapsed = perf_counter() - start
    assert elapsed < 2.0
