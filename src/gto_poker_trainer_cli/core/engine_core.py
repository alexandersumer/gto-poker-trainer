from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from .formatting import format_option_label
from .interfaces import EpisodeGenerator, OptionProvider, Presenter
from .models import OptionResolution


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
        hand_ended = False
        for node in ep.nodes:
            opts = option_provider.options(node, rng, mc_trials)
            best_idx = max(range(len(opts)), key=lambda i: opts[i].ev)
            presenter.show_node(node, [format_option_label(node, o) for o in opts])
            choice = presenter.prompt_choice(len(opts))
            if choice == -1:
                presenter.summary(records)
                return records
            chosen = opts[choice]
            best = opts[best_idx]
            resolution: OptionResolution = option_provider.resolve(node, chosen, rng)
            chosen_feedback = replace(chosen)
            if resolution.note:
                chosen_feedback.resolution_note = resolution.note
            if resolution.hand_ended:
                chosen_feedback.ends_hand = True
            presenter.step_feedback(node, chosen_feedback, best)
            records.append(
                {
                    "street": node.street,
                    "chosen_key": chosen.key,
                    "chosen_ev": chosen.ev,
                    "best_key": best.key,
                    "best_ev": best.ev,
                    "ev_loss": best.ev - chosen.ev,
                    "hand_ended": chosen_feedback.ends_hand,
                    "resolution_note": chosen_feedback.resolution_note,
                }
            )
            # If the chosen action ends the hand (e.g., Fold), stop traversing further nodes.
            if chosen_feedback.ends_hand:
                hand_ended = True
                break
        if hand_ended:
            # Proceed to next hand (or summary if this was the last hand)
            continue

    presenter.summary(records)
    return records
