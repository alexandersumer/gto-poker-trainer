from __future__ import annotations

import inspect

import pytest

pytest.importorskip("textual")

from gto_poker_trainer_cli.ui.textual_app import TrainerApp


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
