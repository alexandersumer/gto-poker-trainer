from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gto_poker_trainer.dynamic.cards import str_to_int
from gto_poker_trainer.dynamic.generator import Node
from gto_poker_trainer.solver.oracle import CSVStrategyOracle


def _node_preflop(hero_hand: str, open_size: float) -> Node:
    h1 = str_to_int(hero_hand[:2])
    h2 = str_to_int(hero_hand[2:])
    return Node(
        street="preflop",
        description="test",
        pot_bb=1.5 + (open_size - 0.5),
        effective_bb=100,
        hero_cards=[h1, h2],
        board=[],
        actor="BB",
        context={"open_size": open_size},
    )


def test_csv_strategy_oracle_matches_preflop_hand(tmp_path: Path):
    csv_path = tmp_path / "solver.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "street",
                "hero_position",
                "context_action",
                "context_size",
                "hero_hand",
                "option_key",
                "option_ev",
                "option_why",
                "gto_freq",
            ]
        )
        # Preflop: BB vs SB open 2.5bb, hero A5s; options: Fold, Call, 3-bet
        w.writerow(["preflop", "BB", "open", "2.5bb", "A5S", "Fold", -1.0, "Give up BB", 0.00])
        w.writerow(["preflop", "BB", "open", "2.5bb", "A5S", "Call", 0.28, "Playable", 0.40])
        w.writerow(["preflop", "BB", "open", "2.5bb", "A5S", "3-bet to 9bb", 0.36, "Blockers", 0.60])

    oracle = CSVStrategyOracle(str(csv_path))
    node = _node_preflop("As5s", 2.5)
    opts = oracle.options(node, None, 0)
    keys = [o.key for o in opts]
    assert keys == ["Fold", "Call", "3-bet to 9bb"]
    best = max(opts, key=lambda o: o.ev)
    assert best.key == "3-bet to 9bb"
    assert best.gto_freq is not None and best.gto_freq >= 0.5
