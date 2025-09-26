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
    risk = raise_to - 1.0
    call_cost = raise_to - 2.5
    final_pot = pot + risk + call_cost
    be_threshold = call_cost / final_pot
    hero_eq_continue = eq_map[combos[0]]
    fe_expected = 1 / len(eq_map)
    ev_called = hero_eq_continue * final_pot - risk
    expected = fe_expected * pot + (1 - fe_expected) * ev_called

    assert abs(three_bet.ev - expected) < 1e-9
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
    fe_expected = 0.5
    eq_call = eq_map[combos[0]]
    ev_called = eq_call * final_pot - bet
    expected = fe_expected * node.pot_bb + (1 - fe_expected) * ev_called

    assert abs(bet_half.ev - expected) < 1e-9
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
    fe_expected = 0.5  # one of two combos folds
    eq_call = eq_map[combos[0]]
    showdown_ev = eq_call * (pot + bet) - (1 - eq_call) * bet
    expected = fe_expected * pot + (1 - fe_expected) * showdown_ev

    assert abs(bet_half.ev - expected) < 1e-9
    assert "EV" in bet_half.why and "equity" in bet_half.why
