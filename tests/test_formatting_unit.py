from __future__ import annotations

from gto_poker_trainer_cli.core.formatting import format_option_label
from gto_poker_trainer_cli.core.models import Option
from gto_poker_trainer_cli.dynamic.generator import Node


def _node(*, pot: float = 8.0, street: str = "flop") -> Node:
    return Node(
        street=street,
        description="",
        pot_bb=pot,
        effective_bb=100.0,
        hero_cards=[0, 1],
        board=[2, 3, 4] if street != "preflop" else [],
        actor="BB",
        context={"open_size": 2.5} if street == "preflop" else {},
    )


def test_format_option_label_uses_meta_percentage_for_bet():
    opt = Option(
        key="Bet 50% pot",
        ev=0.0,
        why="",
        meta={"action": "bet", "bet": 3.0},
    )
    label = format_option_label(_node(pot=6.0), opt)
    assert label == "Bet 50%"


def test_format_option_label_converts_bb_to_percentage():
    opt = Option("Bet 4.00 bb", 0.0, "")
    label = format_option_label(_node(pot=8.0), opt)
    assert label == "Bet 50%"


def test_format_option_label_convert_raise_from_text():
    opt = Option("Raise to 6.00 bb", 0.0, "")
    label = format_option_label(_node(pot=9.0), opt)
    assert label == "Raise 66.7%"


def test_format_option_label_leave_fold_literal():
    opt = Option("Fold", 0.0, "")
    label = format_option_label(_node(pot=7.0), opt)
    assert label == "Fold"


def test_format_option_label_call_without_cost_is_literal():
    opt = Option("Call", 0.0, "", meta={"action": "call"})
    label = format_option_label(_node(pot=5.0), opt)
    assert label == "Call"


def test_format_option_label_uppercase_bb_text():
    opt = Option("3-bet to 10 BB", 0.0, "")
    label = format_option_label(_node(pot=12.0), opt)
    assert label.startswith("3-bet")
    assert "%" in label


def test_format_option_label_all_in_text():
    opt = Option("All-in for 100bb", 0.0, "")
    label = format_option_label(_node(pot=15.0), opt)
    assert label == "All-in"


def _preflop_node(pot: float = 3.5) -> Node:
    return Node(
        street="preflop",
        description="preflop",
        pot_bb=pot,
        effective_bb=100.0,
        hero_cards=[0, 1],
        board=[],
        actor="BB",
        context={"open_size": 2.5},
    )


def test_format_option_label_preflop_call_uses_bb():
    opt = Option("Call", 0.0, "", meta={"action": "call", "call_cost": 1.5})
    label = format_option_label(_preflop_node(), opt)
    assert label == "Call 1.50bb"


def test_format_option_label_preflop_3bet_uses_bb():
    opt = Option(
        "3-bet",
        0.0,
        "",
        meta={"action": "3bet", "raise_to": 12.0},
    )
    label = format_option_label(_preflop_node(), opt)
    assert label == "3-bet to 12.00bb"


def test_format_option_label_preflop_jam_literal():
    opt = Option("Jam option", 0.0, "", meta={"action": "jam"})
    label = format_option_label(_preflop_node(), opt)
    assert label == "All-in"
