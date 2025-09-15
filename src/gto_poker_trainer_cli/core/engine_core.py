from __future__ import annotations

import random
from typing import Any

from .interfaces import EpisodeGenerator, OptionProvider, Presenter


def run_core(
    generator: EpisodeGenerator,
    option_provider: OptionProvider,
    presenter: Presenter,
    *,
    seed: int,
    hands: int,
    mc_trials: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    presenter.start_session(hands)
    records: list[dict[str, Any]] = []

    for h in range(hands):
        presenter.start_hand(h + 1, hands)
        ep = generator.generate(rng)
        for node in ep.nodes:
            opts = option_provider.options(node, rng, mc_trials)
            best_idx = max(range(len(opts)), key=lambda i: opts[i].ev)
            presenter.show_node(node, [o.key for o in opts])
            choice = presenter.prompt_choice(len(opts))
            if choice == -1:
                presenter.summary(records)
                return records
            chosen = opts[choice]
            best = opts[best_idx]
            presenter.step_feedback(node, chosen, best)
            records.append(
                {
                    "street": node.street,
                    "chosen_key": chosen.key,
                    "chosen_ev": chosen.ev,
                    "best_key": best.key,
                    "best_ev": best.ev,
                    "ev_loss": best.ev - chosen.ev,
                }
            )

    presenter.summary(records)
    return records
