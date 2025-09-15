from __future__ import annotations

import csv
from pathlib import Path

from gto_trainer.core.models import Option
from gto_trainer.solver.oracle import CSVStrategyOracle, CompositeOptionProvider
from gto_trainer.dynamic.cards import str_to_int
from gto_trainer.dynamic.generator import Node


class _FallbackProvider:
    def options(self, node, rng, mc_trials):  # noqa: D401 - protocol-compatible stub
        return [Option("Fallback", 0.0, "used fallback")]


def _node(street: str) -> Node:
    return Node(
        street=street,
        description="",
        pot_bb=2.0,
        effective_bb=100,
        hero_cards=[str_to_int("As"), str_to_int("Kd")],
        board=[],
        actor="BB",
        context={"open_size": 2.0},
    )


def test_composite_falls_back_when_csv_does_not_support_street(tmp_path: Path):
    # Minimal CSV with a single preflop row
    p = tmp_path / "solver.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "street",
            "hero_position",
            "context_action",
            "context_size",
            "hero_hand",
            "option_key",
            "option_ev",
            "option_why",
            "gto_freq",
        ])
        w.writerow(["preflop", "BB", "open", "2.0bb", "AKo", "Fold", -1.0, "", 0.0])

    primary = CSVStrategyOracle(str(p))
    provider = CompositeOptionProvider(primary=primary, fallback=_FallbackProvider())

    # For flop street, CSVStrategyOracle raises; composite should use fallback
    opts = provider.options(_node("flop"), None, 0)
    assert [o.key for o in opts] == ["Fallback"]

