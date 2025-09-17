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


class _OneOptionProvider(OptionProvider):
    def __init__(self, options: list[Option]):
        self._opts = options

    def options(self, _node, _rng, _mc_trials):  # type: ignore[override]
        return list(self._opts)

    def resolve(self, _node, chosen: Option, _rng) -> OptionResolution:  # type: ignore[override]
        return OptionResolution(hand_ended=getattr(chosen, "ends_hand", False))


class _ScriptedPresenter(Presenter):
    def __init__(self, choices: list[int]):
        self._choices = choices
        self._i = 0
        self.shown_nodes: list[Node] = []
        self.records: list[dict] = []

    def start_session(self, total_hands: int) -> None:  # noqa: ARG002 - defined by protocol
        pass

    def start_hand(self, hand_index: int, total_hands: int) -> None:  # noqa: ARG002
        pass

    def show_node(self, node: Node, options: list[str]) -> None:  # noqa: ARG002
        self.shown_nodes.append(node)

    def prompt_choice(self, n: int) -> int:  # noqa: ARG002
        # Return next scripted choice (0-based); -1 means quit
        v = self._choices[self._i]
        self._i += 1
        return v

    def step_feedback(self, node: Node, chosen: Option, best: Option) -> None:  # noqa: ARG002
        pass

    def summary(self, records: list[dict]) -> None:
        self.records = list(records)


def test_run_core_records_and_quit_mid_episode():
    node1 = Node(
        street="preflop",
        description="n1",
        pot_bb=2.0,
        effective_bb=100,
        hero_cards=[0, 1],
        board=[],
        actor="BB",
        context={"open_size": 2.0},
    )
    node2 = Node(
        street="flop",
        description="n2",
        pot_bb=4.0,
        effective_bb=100,
        hero_cards=[0, 1],
        board=[2, 3, 4],
        actor="BB",
        context={"facing": "check"},
    )
    gen = _StaticGen([node1, node2])
    opts = [Option("A", 0.2, ""), Option("B", 0.5, "")]
    provider = _OneOptionProvider(opts)
    presenter = _ScriptedPresenter(choices=[0, -1])  # answer first node, quit on second

    records = run_core(
        generator=gen,
        option_provider=provider,
        presenter=presenter,
        seed=0,
        hands=1,
        mc_trials=1,
    )
    assert len(records) == 1
    assert records[0]["chosen_key"] in {"A", "B"}
    assert "best_key" in records[0]
