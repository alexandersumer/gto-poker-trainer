from __future__ import annotations

import pytest

from gto_trainer.dynamic import policy as pol


def test_precision_for_street_scales_trials():
    preflop = pol._precision_for_street(100, "preflop")
    flop = pol._precision_for_street(100, "flop")
    turn = pol._precision_for_street(100, "turn")
    river = pol._precision_for_street(100, "river")

    assert preflop.trials == 55 and preflop.target_std_error == 0.05
    assert flop.trials == 65 and flop.target_std_error == 0.04
    assert turn.trials == 80 and turn.target_std_error == 0.03
    assert river.trials == 95 and river.target_std_error == 0.025


def test_precision_from_meta_handles_invalid_values():
    meta = {"combo_trials": "", "target_std_error": "nan"}
    precision = pol._precision_from_meta(meta, "flop")

    # falls back to street defaults
    assert precision.trials == pol._precision_for_street(80, "flop").trials
    assert precision.target_std_error == pol._precision_for_street(80, "flop").target_std_error


def test_combo_equity_falls_back_when_target_kw_unsupported(monkeypatch: pytest.MonkeyPatch):
    call_log: list[tuple] = []

    def fake_equity(hero, board, combo, trials):
        call_log.append((tuple(hero), tuple(board), combo, trials))
        return 0.42

    monkeypatch.setattr(pol, "hero_equity_vs_combo", fake_equity)

    precision = pol.MonteCarloPrecision(trials=70, target_std_error=0.02)

    result = pol._combo_equity([1, 2], [3], (4, 5), precision)

    assert result == pytest.approx(0.42)
    # fallback call (without target keyword) should be logged exactly once
    assert call_log == [((1, 2), (3,), (4, 5), 70)]


def test_precision_to_meta_roundtrips_optional_target():
    precision = pol.MonteCarloPrecision(trials=80, target_std_error=0.03)
    meta = precision.to_meta()

    assert meta["combo_trials"] == 80
    assert meta["target_std_error"] == pytest.approx(0.03)
