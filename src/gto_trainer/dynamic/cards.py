from __future__ import annotations

import random
from dataclasses import dataclass

RANKS = "23456789TJQKA"
SUITS = "shdc"  # spades, hearts, diamonds, clubs


def card_int_to_str(c: int) -> str:
    r = RANKS[c // 4]
    s = SUITS[c % 4]
    return r + s


def str_to_treys(card: str) -> int:
    from treys import Card

    r, s = card[0], card[1]
    return Card.new(r + s)


def str_to_int(card: str) -> int:
    r, s = card[0].upper(), card[1].lower()
    return RANKS.index(r) * 4 + SUITS.index(s)


def ints_to_str(cards: list[int]) -> str:
    return " ".join(card_int_to_str(c) for c in cards)


# --- Formatting helpers for consistent UI (letters only) ---


def format_card_ascii(c: int, upper: bool = True) -> str:
    rs = card_int_to_str(c)
    return rs.upper() if upper else rs


def format_cards_spaced(cards: list[int]) -> str:
    # Sort by rank descending for a professional, stable look
    def rank_key(ci: int) -> int:
        return ci // 4

    ordered = sorted(cards, key=rank_key, reverse=True)
    return " ".join(format_card_ascii(c, upper=True) for c in ordered)


def deal_unique(rng: random.Random, deck: list[int], n: int) -> list[int]:
    out = []
    for _ in range(n):
        idx = rng.randrange(len(deck))
        out.append(deck.pop(idx))
    return out


def fresh_deck() -> list[int]:
    return list(range(52))


@dataclass
class Dealt:
    hero: list[int]
    rival: list[int]
    board: list[int]  # full 5-card board to keep a coherent storyline


def deal_hand_and_board(rng: random.Random) -> Dealt:
    deck = fresh_deck()
    hero = deal_unique(rng, deck, 2)
    rival = deal_unique(rng, deck, 2)
    board = deal_unique(rng, deck, 5)
    return Dealt(hero=hero, rival=rival, board=board)


def canonical_hand_abbrev(cards: list[int]) -> str:
    # Return like 'A5s', 'KQo', or '55'
    assert len(cards) == 2
    r1 = RANKS[cards[0] // 4]
    r2 = RANKS[cards[1] // 4]
    s1 = cards[0] % 4
    s2 = cards[1] % 4
    # order ranks high-first
    if RANKS.index(r2) > RANKS.index(r1):
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return r1 + r2
    suited = s1 == s2
    return f"{r1}{r2}{'s' if suited else 'o'}"
