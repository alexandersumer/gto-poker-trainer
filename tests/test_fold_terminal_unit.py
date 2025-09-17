from __future__ import annotations

import random
from dataclasses import dataclass

from gto_poker_trainer_cli.core.engine_core import run_core
from gto_poker_trainer_cli.core.interfaces import EpisodeGenerator, OptionProvider, Presenter
from gto_poker_trainer_cli.core.models import Option, OptionResolution
from gto_poker_trainer_cli.dynamic.generator import Episode, Node


@dataclass
class _StaticGen(EpisodeGenerator):
    nodes: list[Node]

    def generate(self, _rng: random.Random) -> Episode:  # type: ignore[override]
        return Episode(nodes=self.nodes)


class _ProviderWithFold(OptionProvider):
    def options(self, _node, _rng, _mc_trials):  # type: ignore[override]
        return [
            Option("Fold", 0.0, "end hand", ends_hand=True),
            Option("Call", -0.1, "", ends_hand=False),
        ]

    def resolve(self, _node, chosen: Option, _rng) -> OptionResolution:  # type: ignore[override]
        return OptionResolution(hand_ended=getattr(chosen, "ends_hand", False))


class _RecorderPresenter(Presenter):
    def __init__(self):
        self.shown = 0
        self.records: list[dict] = []

    def start_session(self, total_hands: int) -> None:  # noqa: ARG002
        pass

    def start_hand(self, hand_index: int, total_hands: int) -> None:  # noqa: ARG002
        pass

    def show_node(self, node: Node, options: list[str]) -> None:  # noqa: ARG002
        self.shown += 1

    def prompt_choice(self, n: int) -> int:  # noqa: ARG002
        return 0  # always choose Fold

    def step_feedback(self, node: Node, chosen: Option, best: Option) -> None:  # noqa: ARG002
        pass

    def summary(self, records: list[dict]) -> None:
        self.records = list(records)


def test_fold_ends_hand_early_and_summarizes_single_hand():
    # Two-node episode but fold on the first node should end the hand.
    nodes = [
        Node("preflop", "n1", 2.0, 100.0, [0, 1], [], "BB", {"open_size": 2.0}),
        Node("flop", "n2", 4.0, 100.0, [0, 1], [2, 3, 4], "BB", {"facing": "check"}),
    ]
    gen = _StaticGen(nodes)
    provider = _ProviderWithFold()
    presenter = _RecorderPresenter()

    records = run_core(
        generator=gen,
        option_provider=provider,
        presenter=presenter,
        seed=0,
        hands=1,
        mc_trials=1,
    )

    # Only one decision recorded and only first node shown
    assert len(records) == 1
    assert presenter.shown == 1
    assert records[0]["chosen_key"] == "Fold"
