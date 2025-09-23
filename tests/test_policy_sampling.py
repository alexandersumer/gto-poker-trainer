from __future__ import annotations

from gtotrainer.dynamic.cards import str_to_int
from gtotrainer.dynamic.policy import _combo_category, _sample_range


def _make_combo(card1: str, card2: str) -> tuple[int, int]:
    cards = sorted((str_to_int(card1), str_to_int(card2)))
    return tuple(cards)  # type: ignore[return-value]


def _build_test_combos() -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for rank in "AKQJT98765432":
        first = f"{rank}s"
        second = f"{rank}h"
        pairs.append(_make_combo(first, second))
        if len(pairs) >= 12:
            break

    suited: list[tuple[int, int]] = []
    suited_rank_pairs = [
        ("A", "K"),
        ("A", "Q"),
        ("K", "Q"),
        ("Q", "J"),
        ("J", "T"),
        ("T", "9"),
        ("9", "8"),
        ("8", "7"),
        ("7", "6"),
        ("6", "5"),
        ("5", "4"),
        ("4", "3"),
        ("3", "2"),
    ]
    for high, low in suited_rank_pairs:
        suited.append(_make_combo(f"{high}s", f"{low}s"))

    offsuit: list[tuple[int, int]] = []
    offsuit_rank_pairs = [
        ("A", "K"),
        ("A", "Q"),
        ("A", "J"),
        ("K", "Q"),
        ("K", "J"),
        ("Q", "J"),
        ("J", "T"),
        ("T", "9"),
        ("9", "7"),
        ("8", "6"),
        ("7", "5"),
        ("6", "4"),
        ("5", "3"),
        ("4", "2"),
    ]
    for high, low in offsuit_rank_pairs:
        offsuit.append(_make_combo(f"{high}s", f"{low}d"))

    return pairs + suited + offsuit


def test_sample_range_respects_limit_and_is_deterministic() -> None:
    combos = _build_test_combos()
    limit = 20

    first = _sample_range(combos, limit)
    second = _sample_range(combos, limit)

    assert len(first) == limit
    assert first == second
    assert all(combo in combos for combo in first)


def test_sample_range_covers_available_categories_when_possible() -> None:
    combos = _build_test_combos()
    limit = 24

    sampled = _sample_range(combos, limit)
    categories = {_combo_category(combo) for combo in sampled}

    assert categories.issuperset({"pair", "suited", "offsuit"})
    assert len(sampled) == limit


def test_sample_range_prioritizes_weighted_combos() -> None:
    suited_only = [
        _make_combo("As", "Qs"),
        _make_combo("Ks", "Js"),
        _make_combo("Qs", "Ts"),
        _make_combo("Js", "9s"),
        _make_combo("Ts", "8s"),
    ]
    weights = {
        suited_only[3]: 0.8,
        suited_only[4]: 0.7,
    }

    sampled = _sample_range(suited_only, 3, weights)

    assert suited_only[3] in sampled
    assert suited_only[4] in sampled
    assert len(sampled) == 3
