from __future__ import annotations

import random

import pytest

import gtotrainer.dynamic.policy as pol
from gtotrainer.dynamic.cards import str_to_int
from gtotrainer.dynamic.generator import Node


def _simple_range(*pairs: tuple[int, int]) -> list[tuple[int, int]]:
    return [tuple(sorted(p)) for p in pairs]


@pytest.fixture(autouse=True)
def _disable_range_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pol, "load_range_with_weights", lambda *_args, **_kwargs: ([], None))


def test_preflop_threebet_uses_fold_equity_threshold(monkeypatch):
    hero = [str_to_int("As"), str_to_int("Kh")]
    node = Node(
        street="preflop",
        description="pf",
        pot_bb=3.5,
        effective_bb=100.0,
        hero_cards=hero,
        board=[],
        actor="BB",
        context={"open_size": 2.5},
    )

    combos = _simple_range((str_to_int("Qc"), str_to_int("Qd")), (str_to_int("7s"), str_to_int("2d")))

    def fake_range(_open_size, _blocked):  # noqa: ARG001
        return combos

    eq_map = {
        combos[0]: 0.62,  # rival equity 0.38 → continues vs threshold 0.35
        combos[1]: 0.70,  # rival equity 0.30 → folds
    }

    def fake_combo_eq(_hero, _board, combo, _trials):
        return eq_map[tuple(sorted(combo))]

    def fake_range_eq(_hero, _board, _range, _trials):
        return sum(eq_map.values()) / len(eq_map)

    monkeypatch.setattr(pol, "rival_sb_open_range", fake_range)
    monkeypatch.setattr(pol, "hero_equity_vs_combo", fake_combo_eq)
    monkeypatch.setattr(pol, "hero_equity_vs_range", fake_range_eq)

    opts = pol.preflop_options(node, random.Random(0), mc_trials=100)
    three_bet = next(o for o in opts if o.key.startswith("3-bet"))

    pot = node.pot_bb
    raise_to = float(three_bet.key.split()[2][:-2])
    hero_add = raise_to - 1.0
    risk = hero_add
    call_cost = raise_to - 2.5
    final_pot = pot + risk + call_cost
    be_threshold = call_cost / final_pot
    entries = [(eq_map[combo], 1.0) for combo in combos]
    params = pol._fold_params({}, pot=pot, bet=hero_add, board=node.board)
    fe_calc, avg_eq_when_called, continue_ratio = pol._fold_continue_stats(entries, be_threshold, params=params)
    ev_called = avg_eq_when_called * final_pot - hero_add if continue_ratio else -hero_add
    expected = fe_calc * pot + (1 - fe_calc) * ev_called

    assert three_bet.meta["rival_fe"] == pytest.approx(fe_calc, rel=1e-6)
    assert three_bet.meta["rival_continue_ratio"] == pytest.approx(continue_ratio, rel=1e-6)
    hero_payoffs = three_bet.meta["cfr_payoffs"]["hero"]
    assert hero_payoffs[0] == pytest.approx(pot)
    assert hero_payoffs[1] == pytest.approx(ev_called)
    assert three_bet.ev == pytest.approx(expected)
    assert f"{be_threshold * 100:.1f}%" in three_bet.why


def test_flop_half_pot_bet_uses_p_plus_2b_when_called(monkeypatch):
    hero = [str_to_int("Ah"), str_to_int("Qh")]
    board = [str_to_int(x) for x in ["Qd", "7c", "3s"]]
    node = Node(
        street="flop",
        description="flop",
        pot_bb=4.0,
        effective_bb=100.0,
        hero_cards=hero,
        board=board,
        actor="BB",
        context={"facing": "check", "open_size": 2.5},
    )

    combos = _simple_range((str_to_int("Kd"), str_to_int("Jh")), (str_to_int("8c"), str_to_int("4c")))

    def fake_range(_open_size, _blocked):
        return combos

    eq_map = {
        combos[0]: 0.40,  # rival equity 0.60 → continues
        combos[1]: 0.80,  # rival equity 0.20 → folds
    }

    def fake_combo_eq(_hero, _board, combo, _trials):
        return eq_map[tuple(sorted(combo))]

    def fake_range_eq(_hero, _board, _range, _trials):
        return sum(eq_map.values()) / len(eq_map)

    monkeypatch.setattr(pol, "rival_sb_open_range", fake_range)
    monkeypatch.setattr(pol, "hero_equity_vs_combo", fake_combo_eq)
    monkeypatch.setattr(pol, "hero_equity_vs_range", fake_range_eq)

    opts = pol.flop_options(node, random.Random(1), mc_trials=100)
    bet_half = next(o for o in opts if o.key.startswith("Bet 50% pot"))

    bet = round(node.pot_bb * 0.5, 2)
    final_pot = node.pot_bb + 2 * bet
    be_threshold = bet / final_pot if final_pot > 0 else 1.0
    entries = [(eq_map[combo], 1.0) for combo in combos]
    params = pol._fold_params({}, pot=node.pot_bb, bet=bet, board=node.board)
    fe_calc, avg_eq_when_called, continue_ratio = pol._fold_continue_stats(entries, be_threshold, params=params)
    ev_called = avg_eq_when_called * final_pot - bet if continue_ratio else -bet
    expected = fe_calc * node.pot_bb + (1 - fe_calc) * ev_called

    assert bet_half.meta["rival_fe"] == pytest.approx(fe_calc, rel=1e-6)
    assert bet_half.meta["rival_continue_ratio"] == pytest.approx(continue_ratio, rel=1e-6)
    hero_payoffs = bet_half.meta["cfr_payoffs"]["hero"]
    assert hero_payoffs[0] == pytest.approx(node.pot_bb)
    assert hero_payoffs[1] == pytest.approx(ev_called)
    assert bet_half.ev == pytest.approx(expected)
    assert "needs" in bet_half.why and "equity" in bet_half.why


def test_river_bet_uses_showdown_payout_formula(monkeypatch):
    hero = [str_to_int("Ah"), str_to_int("Kd")]
    board = [str_to_int(x) for x in ["6h", "6d", "6c", "2s", "9h"]]
    node = Node(
        street="river",
        description="river",
        pot_bb=12.0,
        effective_bb=100.0,
        hero_cards=hero,
        board=board,
        actor="BB",
        context={"facing": "oop-check", "open_size": 2.5},
    )

    combos = _simple_range((str_to_int("Qc"), str_to_int("Qd")), (str_to_int("7s"), str_to_int("2d")))

    def fake_range(_open_size, _blocked):  # noqa: ARG001
        return combos

    def fake_tighten_range(_combos, _fraction):  # noqa: ARG001
        return combos

    eq_map = {
        combos[0]: 0.40,  # rival equity 0.60 → continues vs threshold 0.25
        combos[1]: 0.90,  # rival equity 0.10 → folds
    }

    def fake_combo_eq(_hero, _board, combo, _trials):
        return eq_map[tuple(sorted(combo))]

    monkeypatch.setattr(pol, "rival_sb_open_range", fake_range)
    monkeypatch.setattr(pol, "tighten_range", fake_tighten_range)
    monkeypatch.setattr(pol, "hero_equity_vs_combo", fake_combo_eq)

    opts = pol.river_options(node, random.Random(5), mc_trials=80)
    bet_half = next(o for o in opts if o.key.startswith("Bet 50% pot"))

    pot = node.pot_bb
    bet = round(pot * 0.5, 2)
    final_pot = pot + 2 * bet
    entries = [(eq_map[combo], 1.0) for combo in combos]
    params = pol._fold_params({}, pot=pot, bet=bet, board=node.board)
    fe_calc, avg_eq_when_called, continue_ratio = pol._fold_continue_stats(entries, bet / final_pot, params=params)
    showdown_ev = avg_eq_when_called * (pot + bet) - (1 - avg_eq_when_called) * bet
    expected = fe_calc * pot + (1 - fe_calc) * showdown_ev

    assert bet_half.meta["rival_fe"] == pytest.approx(fe_calc, rel=1e-6)
    assert bet_half.meta["rival_continue_ratio"] == pytest.approx(continue_ratio, rel=1e-6)
    hero_payoffs = bet_half.meta["cfr_payoffs"]["hero"]
    assert hero_payoffs[0] == pytest.approx(pot)
    assert hero_payoffs[1] == pytest.approx(showdown_ev)
    assert bet_half.ev == pytest.approx(expected)
    assert "EV" in bet_half.why and "equity" in bet_half.why
