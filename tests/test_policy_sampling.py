from __future__ import annotations

import random

from gtotrainer.dynamic import policy as policy_module
from gtotrainer.dynamic.cards import str_to_int
from gtotrainer.dynamic.range_sampling import combo_category, sample_range


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

    first = sample_range(combos, limit, None, random.Random(42))
    second = sample_range(combos, limit, None, random.Random(42))

    assert len(first) == limit
    assert first == second
    assert all(combo in combos for combo in first)

    different = sample_range(combos, limit, None, random.Random(7))
    assert different != first


def test_sample_range_covers_available_categories_when_possible() -> None:
    combos = _build_test_combos()
    limit = 24

    sampled = sample_range(combos, limit, None, random.Random(99))
    categories = {combo_category(combo) for combo in sampled}

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

    sampled = sample_range(suited_only, 3, weights, random.Random(7))

    assert suited_only[3] in sampled
    assert suited_only[4] in sampled
    assert len(sampled) == 3


def test_sample_range_handles_zero_weight_maps() -> None:
    combos = _build_test_combos()
    limit = 10
    zero_weights = dict.fromkeys(combos[: limit + 5], 0.0)

    first = sample_range(combos, limit, zero_weights, random.Random(123))
    second = sample_range(combos, limit, zero_weights, random.Random(123))

    assert len(first) == limit
    assert first == second
    assert all(combo in combos for combo in first)


def test_flop_fraction_candidates_reflect_board_texture() -> None:
    dry_board = [str_to_int("Ah"), str_to_int("7d"), str_to_int("2c")]
    wet_board = [str_to_int("Th"), str_to_int("Jh"), str_to_int("Qh"), str_to_int("9h")]

    dry_candidates = policy_module._flop_fraction_candidates(dry_board, spr=3.0)
    wet_candidates = policy_module._flop_fraction_candidates(wet_board, spr=3.0)

    assert 0.25 in dry_candidates
    assert 0.25 not in wet_candidates
    assert 0.5 in wet_candidates
