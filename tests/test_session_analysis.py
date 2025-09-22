from __future__ import annotations

import pytest

from gtotrainer.core.models import Option
from gtotrainer.dynamic.episode import Node
from gtotrainer.features.session.analysis import build_node_analysis


def _make_node() -> Node:
    return Node(
        street="flop",
        description="Hero holds a suited connector",
        pot_bb=6.0,
        effective_bb=45.0,
        hero_cards=[0, 1],
        board=[12, 16, 24],
        actor="Hero",
        context={"bet": 2.0},
    )


def test_build_node_analysis_includes_breakdown():
    node = _make_node()
    combos = [(i, i + 1) for i in range(0, 24, 2)]  # 12 combos
    option_raise = Option(
        "Bet 66%",
        ev=1.85,
        why="Pressure range with fold equity.",
        meta={
            "rival_fe": 0.46,
            "rival_continue_ratio": 0.54,
            "hero_ev_fold": 6.0,
            "hero_ev_continue": -0.4,
            "hero_invest": 4.0,
            "villain_invest": 4.0,
            "pot_before": 6.0,
            "pot_if_called": 14.0,
            "hero_eq_continue": 0.41,
            "rival_continue_range": combos,
        },
    )
    option_check = Option("Check", ev=1.32, why="Take the free card.")
    analysis = build_node_analysis(node, [option_raise, option_check])
    assert analysis is not None
    assert analysis.best_key == option_raise.key
    assert len(analysis.options) == 2
    first = analysis.options[0]
    assert first.key == option_raise.key
    assert first.breakdown is not None
    assert first.breakdown.fold_pct == pytest.approx(0.46)
    assert first.breakdown.continue_pct == pytest.approx(0.54)
    assert first.breakdown.hero_equity_vs_continue == pytest.approx(0.41)
    assert first.breakdown.villain_continue_sample is not None
    assert len(first.breakdown.villain_continue_sample) == len(combos)
    second = analysis.options[1]
    assert second.key == option_check.key
    assert second.breakdown is None


def test_build_node_analysis_truncates_large_ranges():
    node = _make_node()
    combos = [(i, i + 1) for i in range(0, 60, 2)]  # 30 combos -> should truncate to 12
    option = Option(
        "All-in",
        ev=0.5,
        why="Jam against capped range.",
        meta={
            "rival_fe": 0.35,
            "hero_ev_fold": 6.0,
            "hero_ev_continue": -1.2,
            "hero_invest": 10.0,
            "villain_invest": 10.0,
            "pot_before": 6.0,
            "pot_if_called": 26.0,
            "rival_continue_range": combos,
        },
    )
    analysis = build_node_analysis(node, [option])
    assert analysis is not None
    card = analysis.options[0]
    breakdown = card.breakdown
    assert breakdown is not None
    assert breakdown.villain_continue_sample is not None
    assert len(breakdown.villain_continue_sample) == 12
    assert breakdown.villain_continue_total == len(combos)


def test_breakdown_terms_match_meta_inputs():
    node = _make_node()
    combos = [(0, 1), (2, 3)]
    option = Option(
        "Bet",
        ev=2.0,
        why="",
        meta={
            "rival_fe": 0.4,
            "rival_continue_ratio": 0.6,
            "hero_ev_fold": 5.0,
            "hero_ev_continue": -1.5,
            "hero_invest": 3.0,
            "villain_invest": 3.0,
            "pot_before": 6.0,
            "pot_if_called": 12.0,
            "hero_eq_continue": 0.42,
            "rival_continue_range": combos,
        },
    )
    analysis = build_node_analysis(node, [option])
    breakdown = analysis.options[0].breakdown
    assert breakdown is not None
    assert breakdown.fold_term == pytest.approx(0.4 * 5.0)
    assert breakdown.continue_term == pytest.approx(0.6 * -1.5)
    assert breakdown.hero_invest == pytest.approx(3.0)
    assert breakdown.pot_if_called == pytest.approx(12.0)
