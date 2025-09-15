from __future__ import annotations

import random

from gto_poker_trainer_cli.dynamic.cards import str_to_int
from gto_poker_trainer_cli.dynamic.generator import Node
from gto_poker_trainer_cli.dynamic.policy import turn_options


def test_turn_raise_break_even_fe_and_call_pot_odds_in_why_strings():
    # Construct a turn node with clear numbers: P=4.0, B=3.0 → R=7.5 (2.5x)
    rng = random.Random(42)
    hero = [str_to_int("8h"), str_to_int("5h")]
    board = [str_to_int(x) for x in ["Ks", "Qd", "Th", "3s"]]
    node = Node(
        street="turn",
        description="test turn math",
        pot_bb=4.0,
        effective_bb=100.0,
        hero_cards=hero,
        board=board,
        actor="BB",
        context={"facing": "bet", "bet": 3.0},
    )
    opts = turn_options(node, rng, mc_trials=10)
    keys = [o.key for o in opts]
    # Ensure options order begins with Fold, Call, Raise... and strings mention math hints
    assert keys[:2] == ["Fold", "Call"]
    why_call = opts[1].why
    assert "Pot odds" in why_call and "BE eq" in why_call
    # For R = 2.5 * 3 = 7.5, breakeven FE (ignoring equity) is R / (R + (P+B)) = 7.5 / 14.5 ≈ 0.517
    fe_be = 7.5 / (7.5 + (4.0 + 3.0))
    why_raise = opts[2].why
    # Expect percentage rounded to 0 decimals in string (≈52%)
    assert f"FE≥{int(round(100 * fe_be, 0))}%" in why_raise
