from __future__ import annotations

import random

from gtotrainer.dynamic.cards import (
    RANKS,
    SUITS,
    canonical_hand_abbrev,
    canonicalize_cards,
    card_int_to_str,
    deal_hand_and_board,
    deal_unique,
    format_card_ascii,
    format_cards_spaced,
    fresh_deck,
    ints_to_str,
    str_to_int,
)


def test_str_to_int_and_back_roundtrip_all_cards():
    for r in RANKS:
        for s in SUITS:
            ci = str_to_int(r + s)
            assert 0 <= ci < 52
            rs = card_int_to_str(ci)
            assert rs == r + s


def test_ints_to_str_and_formatters_sorting_and_symbols():
    # Use an unsorted set of ranks to verify sorting in format_cards_spaced
    cards = [str_to_int(x) for x in ["2c", "As", "Td"]]
    asc = ints_to_str(cards)
    assert asc.split() == ["2c", "As", "Td"]
    # Sorting places A first, then T, then 2
    spaced = format_cards_spaced(cards)
    assert spaced.split() == ["AS", "TD", "2C"]
    # Single-card helpers
    c_as = str_to_int("As")
    assert format_card_ascii(c_as) == "AS"


def test_canonical_hand_abbrev_pairs_suited_offsuit():
    a5s = [str_to_int("As"), str_to_int("5s")]
    assert canonical_hand_abbrev(a5s) == "A5s"
    kqo = [str_to_int("Kd"), str_to_int("Qh")]
    assert canonical_hand_abbrev(kqo) == "KQo"
    sevens = [str_to_int("7h"), str_to_int("7c")]
    assert canonical_hand_abbrev(sevens) == "77"


def test_deal_unique_and_deal_hand_and_board_no_collisions():
    rng = random.Random(123)
    deck = fresh_deck()
    dealt = deal_unique(rng, deck, 5)
    assert len(dealt) == 5
    assert len(set(dealt)) == 5
    # Deck should shrink accordingly
    assert len(deck) == 52 - 5

    # deal_hand_and_board coherence: hero+rival+board all distinct
    rng2 = random.Random(456)
    d = deal_hand_and_board(rng2)
    all_cards = d.hero + d.rival + d.board
    assert len(all_cards) == 2 + 2 + 5
    assert len(set(all_cards)) == len(all_cards)


def test_canonicalize_cards_matches_suit_isomorphisms():
    hero_a = [str_to_int("As"), str_to_int("Kd")]
    board_a = [str_to_int(x) for x in ["Qs", "Jh", "2d"]]

    suit_rot = {"s": "h", "h": "d", "d": "c", "c": "s"}

    def _map(card: str) -> str:
        return card[0] + suit_rot[card[1]]

    hero_b = [str_to_int(_map("As")), str_to_int(_map("Kd"))]
    board_b = [str_to_int(_map(x)) for x in ["Qs", "Jh", "2d"]]

    canon_a = canonicalize_cards(hero_a, board_a)
    canon_b = canonicalize_cards(hero_b, board_b)

    assert canon_a[0] == canon_b[0]
    assert canon_a[1] == canon_b[1]


def test_canonicalize_cards_rejects_duplicates():
    hero = [str_to_int("As"), str_to_int("Kd")]
    board = [str_to_int("Qs"), str_to_int("Jh"), str_to_int("2d")]

    # Construct extra that would collide if canonicalisation reused suits incorrectly.
    extra = [str_to_int("Ad"), str_to_int("Kc")]

    hero_c, board_c, extra_c = canonicalize_cards(hero, board, extra)
    combined = set(hero_c + board_c + extra_c)
    assert len(combined) == len(hero_c) + len(board_c) + len(extra_c)
