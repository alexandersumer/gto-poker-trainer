from __future__ import annotations

import random

import gto_poker_trainer_cli.dynamic.policy as pol
from gto_poker_trainer_cli.dynamic.cards import str_to_int
from gto_poker_trainer_cli.dynamic.generator import Node


def test_preflop_3bet_ev_uses_2x_total_and_cost_minus_blind(monkeypatch):
    # Force deterministic equity
    def _eq(_hero, _board, _vill, _rng, trials=400):  # noqa: ARG001
        return 0.50

    monkeypatch.setattr(pol, "estimate_equity", _eq)
    rng = random.Random(0)

    hero = [str_to_int("As"), str_to_int("5s")]
    sz = 2.5  # SB opens to 2.5
    # pot after open = 1.5 + (sz - 0.5) = sz + 1.0 = 3.5
    node = Node(
        street="preflop",
        description="pf",
        pot_bb=1.5 + (sz - 0.5),
        effective_bb=100.0,
        hero_cards=hero,
        board=[],
        actor="BB",
        context={"open_size": sz},
    )

    opts = pol.preflop_options(node, rng, mc_trials=1)
    # Find 3-bet to 9bb entry
    opt = next(o for o in opts if o.key.startswith("3-bet to 9"))

    # Expected EV
    total_to = 9.0
    P = node.pot_bb
    risk = total_to - 1.0  # additional beyond posted 1bb
    pot_called = 2.0 * total_to
    eq3 = 0.50 * 0.96
    fe = max(0.05, min(0.55, 0.20 + 0.05 * (total_to - 8)))
    expected = fe * P + (1 - fe) * (eq3 * pot_called - risk)
    assert abs(opt.ev - expected) < 1e-9


def test_flop_bet_ev_uses_p_plus_2b_when_called(monkeypatch):
    # Force deterministic equity
    def _eq(_hero, _board, _vill, _rng, trials=400):  # noqa: ARG001
        return 0.50

    monkeypatch.setattr(pol, "estimate_equity", _eq)
    rng = random.Random(1)

    hero = [str_to_int("Ac"), str_to_int("Kd")]
    board = [str_to_int(x) for x in ["2c", "7d", "Jh"]]
    P = 4.0
    node = Node(
        street="flop",
        description="flop",
        pot_bb=P,
        effective_bb=100.0,
        hero_cards=hero,
        board=board,
        actor="BB",
        context={"facing": "check"},
    )

    opts = pol.flop_options(node, rng, mc_trials=1)
    # Grab the 50% pot sizing
    fifty = next(o for o in opts if o.key == "Bet 50% pot")

    pct = 0.5
    bet = round(P * pct, 2)
    fe = min(0.60, max(0.05, 0.10 + 0.25 * (pct / 0.75) + 0.15 * max(0.0, 0.50 - 0.4)))
    pot_called = P + 2 * bet
    eq_post = min(1.0, 0.50 + 0.02)
    expected = fe * P + (1 - fe) * (eq_post * pot_called - bet)
    assert abs(fifty.ev - expected) < 1e-9
