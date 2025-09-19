from __future__ import annotations

import inspect

import pytest

pytest.importorskip("textual")

from gto_trainer.ui.textual_app import TrainerApp
from gto_trainer.dynamic.episode import Node


def _split_rows(board_markup: str) -> tuple[str, str]:
    top, bottom = board_markup.splitlines()
    return top, bottom


def _count_tokens(row: str) -> int:
    return row.count("[/]")


def test_format_board_rows_preflop_has_fixed_slots():
    app = TrainerApp()
    top_row, bottom_row = _split_rows(app._format_board_rows([]))
    assert _count_tokens(top_row) == 3
    assert _count_tokens(bottom_row) == 2
    assert top_row.count("--") == 3
    assert bottom_row.count("--") == 2


def test_format_board_rows_flop_keeps_placeholders():
    app = TrainerApp()
    flop_board = [0, 4, 8]  # arbitrary integers representing cards
    top_row, bottom_row = _split_rows(app._format_board_rows(flop_board))
    assert _count_tokens(top_row) == 3
    assert _count_tokens(bottom_row) == 2
    # Turn and river slots should still be placeholders until revealed.
    assert "--" in bottom_row


def test_format_board_rows_river_full_board_has_no_placeholders():
    app = TrainerApp()
    river_board = [0, 4, 8, 12, 16]
    top_row, bottom_row = _split_rows(app._format_board_rows(river_board))
    assert _count_tokens(top_row) == 3
    assert _count_tokens(bottom_row) == 2
    assert "--" not in top_row + bottom_row


def test_options_css_uses_grid_layout():
    css = TrainerApp.CSS
    assert "#options" in css
    assert "layout: grid" in css
    assert "grid-columns: 1fr" in css


def test_trainer_app_has_end_session_control():
    source = inspect.getsource(TrainerApp.compose)
    assert "btn-end" in source


def test_trainer_app_bindings_include_end_session():
    bindings = {}
    for binding in TrainerApp.BINDINGS:
        if hasattr(binding, "key"):
            bindings[binding.key] = binding.action
        else:
            key, action, *_ = binding
            bindings[key] = action
    assert bindings.get("escape") == "end_session"


def test_hand_progress_fragment_shows_remaining_hands():
    app = TrainerApp()
    app._total_hands = 4
    app._current_hand_index = 2
    fragment = app._hand_progress_fragment()
    assert "Hand 2/4" in fragment
    assert "left" not in fragment


def test_build_headline_marks_final_hand():
    app = TrainerApp()
    app._total_hands = 3
    app._current_hand_index = 3
    node = Node(
        street="river",
        description="Villain checks.",
        pot_bb=12.0,
        effective_bb=88.0,
        hero_cards=[],
        board=[],
        actor="BB",
        context={"facing": "oop-check"},
    )
    headline = app._build_headline(node)
    assert "Hand 3/3" in headline
    assert "River" in headline
    assert "Facing" not in headline
    assert "Range" not in headline
    assert "Check to hero" not in headline


def test_session_perf_fragment_reports_accuracy_and_ev():
    app = TrainerApp()
    app._decisions_played = 4
    app._best_hits = 3
    app._total_ev_lost = 1.25
    fragment = app._session_perf_fragment()
    assert fragment is not None
    assert "ΔEV -1.25 bb" in fragment
    assert "75%" in fragment


def test_preparing_text_includes_hint():
    app = TrainerApp()
    app._preparing_hint = "[dim]Custom hint[/]"
    text = app._format_preparing_text()
    assert "Custom hint" in text


def test_apply_preparing_placeholders_sets_defaults():
    class Dummy:
        def __init__(self) -> None:
            self.payload = None

        def update(self, value: str) -> None:
            self.payload = value

    app = TrainerApp()
    hand = Dummy()
    board = Dummy()
    meta = Dummy()
    app._hand_panel = hand  # type: ignore[assignment]
    app._board_panel = board  # type: ignore[assignment]
    app._meta_panel = meta  # type: ignore[assignment]
    app._apply_preparing_placeholders()
    assert hand.payload is not None and "-- --" in hand.payload
    assert board.payload is not None and board.payload.count("--") == 5
    assert meta.payload is not None and "Crunching equities" in meta.payload
