from __future__ import annotations

from dataclasses import dataclass

import pytest

from gtotrainer.core.engine_core import run_core
from gtotrainer.core.models import Option, OptionResolution
from gtotrainer.dynamic.episode import Episode, Node


class _StaticGenerator:
    def __init__(self, episode: Episode) -> None:
        self._episode = episode

    def generate(self, _rng):  # pragma: no cover - signature matches protocol
        return self._episode


class _StaticProvider:
    def __init__(self, options: list[Option]) -> None:
        self._options = options

    def options(self, _node, _rng, _mc_trials):  # pragma: no cover - protocol hook
        return [
            Option(opt.key, opt.ev, opt.why, gto_freq=opt.gto_freq, meta=dict(opt.meta or {})) for opt in self._options
        ]

    def resolve(self, _node, _chosen, _rng):  # pragma: no cover - protocol hook
        return OptionResolution()


@dataclass
class _CapturingPresenter:
    choice_index: int
    last_feedback: tuple[Option, Option] | None = None
    summary_records: list[dict] | None = None

    def start_session(self, total_hands: int) -> None:  # pragma: no cover - noop for tests
        pass

    def start_hand(self, hand_index: int, total_hands: int) -> None:  # pragma: no cover - noop
        pass

    def show_node(self, _node, _options):  # pragma: no cover - noop
        pass

    def prompt_choice(self, _n: int) -> int:  # noqa: D401
        return self.choice_index

    def step_feedback(self, _node, chosen: Option, best: Option) -> None:  # pragma: no cover - capture
        self.last_feedback = (chosen, best)

    def summary(self, records: list[dict]) -> None:  # pragma: no cover - capture
        self.summary_records = records


def test_run_core_uses_effective_ev_for_feedback_and_records() -> None:
    node = Node(
        street="flop",
        description="Test",
        pot_bb=10.0,
        effective_bb=100.0,
        hero_cards=[1, 2],
        board=[3, 4, 5],
        actor="BTN",
    )
    episode = Episode(nodes=[node], hero_seat="BTN", rival_seat="BB")

    baseline_best = 1.5
    best = Option("raise", 0.9, "", meta={"baseline_ev": baseline_best})
    chosen = Option("call", 1.0, "", meta={"baseline_ev": 1.0})
    provider = _StaticProvider([best, chosen])
    presenter = _CapturingPresenter(choice_index=1)

    records = run_core(
        generator=_StaticGenerator(episode),
        option_provider=provider,
        presenter=presenter,
        seed=7,
        hands=1,
        mc_trials=10,
    )

    assert presenter.last_feedback is not None
    chosen_feedback, best_feedback = presenter.last_feedback
    assert best_feedback.ev == pytest.approx(baseline_best)
    assert chosen_feedback.ev == pytest.approx(1.0)

    assert len(records) == 1
    record = records[0]
    assert record["best_key"] == "raise"
    assert record["best_ev"] == pytest.approx(baseline_best)
    assert record["ev_loss"] == pytest.approx(baseline_best - 1.0)
    assert record["room_ev"] == pytest.approx(baseline_best - 1.0)
