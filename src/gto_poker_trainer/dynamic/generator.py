from __future__ import annotations

import random
from dataclasses import dataclass

from .cards import Dealt, deal_hand_and_board, format_card_ascii

_SB = "SB"
_BB = "BB"


def _postflop_nodes(
    *,
    hero_pos: str,
    villain_pos: str,
    board: list[int],
    pot_flop: float,
    effective_bb: float,
    hero_cards: list[int],
    hand_state: dict[str, object],
    open_size: float,
    villain_range: str,
) -> list[Node]:
    flop_cards = board[:3]
    flop_str = " ".join(format_card_ascii(c, upper=True) for c in flop_cards)
    desc_flop = f"{flop_str}; {villain_pos} checks."
    n_flop = Node(
        street="flop",
        description=desc_flop,
        pot_bb=pot_flop,
        effective_bb=effective_bb,
        hero_cards=hero_cards,
        board=flop_cards,
        actor=hero_pos,
        context={
            "facing": "check",
            "open_size": open_size,
            "hand_state": hand_state,
            "hero_seat": hero_pos,
            "villain_seat": villain_pos,
            "villain_range": villain_range,
        },
    )

    pot_turn = pot_flop
    bet_turn = round(0.5 * pot_turn, 2)
    board_turn = board[:4]
    board_turn_str = " ".join(format_card_ascii(c, upper=True) for c in board_turn)
    desc_turn = f"{board_turn_str}; {villain_pos} bets {bet_turn:.2f}bb into {pot_turn:.2f}bb."
    n_turn = Node(
        street="turn",
        description=desc_turn,
        pot_bb=pot_turn,
        effective_bb=effective_bb,
        hero_cards=hero_cards,
        board=board_turn,
        actor=hero_pos,
        context={
            "facing": "bet",
            "bet": bet_turn,
            "open_size": open_size,
            "hand_state": hand_state,
            "hero_seat": hero_pos,
            "villain_seat": villain_pos,
            "villain_range": villain_range,
        },
    )

    pot_river = pot_turn + 2 * bet_turn
    river_cards = board
    river_str = " ".join(format_card_ascii(c, upper=True) for c in river_cards)
    desc_river = f"{river_str}; choose your bet."
    n_river = Node(
        street="river",
        description=desc_river,
        pot_bb=pot_river,
        effective_bb=effective_bb,
        hero_cards=hero_cards,
        board=river_cards,
        actor=hero_pos,
        context={
            "facing": "oop-check",
            "open_size": open_size,
            "hand_state": hand_state,
            "hero_seat": hero_pos,
            "villain_seat": villain_pos,
            "villain_range": villain_range,
        },
    )

    return [n_flop, n_turn, n_river]


@dataclass
class Node:
    street: str  # preflop, flop, turn, river
    description: str
    pot_bb: float
    effective_bb: float
    hero_cards: list[int]
    board: list[int]
    actor: str  # SB or BB (hero is always one seat; actor indicates who's to act)
    context: dict


@dataclass
class Episode:
    nodes: list[Node]
    hero_seat: str
    villain_seat: str


def _episode_bb_defense(
    rng: random.Random,
    *,
    stacks_bb: float,
    sb: float,
    bb: float,
) -> Episode:
    hero_pos = _BB
    villain_pos = _SB
    dealt: Dealt = deal_hand_and_board(rng)
    hero_cards = dealt.hero
    villain_cards = dealt.villain
    board = dealt.board

    # Preflop pot starts at 1.5bb (SB+BB)
    pot = sb + bb
    eff = stacks_bb

    # Preflop node: Villain opens to s in {2.0, 2.5, 3.0}; hero defends from the opposite seat.
    open_sizes = [2.0, 2.5, 3.0]
    sz = rng.choice(open_sizes)
    eff_stacks = int(stacks_bb)
    desc_pf = f"{villain_pos} opens {sz:.1f}bb. You're {hero_pos} with {eff_stacks}bb behind."
    # Pot after SB opens to sz: add only the incremental chips beyond the posted SB
    # Example: pot=1.5 (0.5 SB + 1 BB); SB opens to 2.0 → adds 1.5; pot becomes 3.0
    pot_after_open = pot + (sz - sb)
    hand_state: dict[str, object] = {
        "pot": pot_after_open,
        "hero_cards": tuple(hero_cards),
        "villain_cards": tuple(villain_cards),
        "full_board": tuple(board),
        "street": "preflop",
        "history": [],
        "board_index": 0,
        "hero_seat": hero_pos,
        "villain_seat": villain_pos,
    }

    n_preflop = Node(
        street="preflop",
        description=desc_pf,
        pot_bb=pot_after_open,
        effective_bb=eff,
        hero_cards=hero_cards,
        board=[],
        actor=hero_pos,
        context={
            "open_size": sz,
            "hand_state": hand_state,
            "hero_seat": hero_pos,
            "villain_seat": villain_pos,
            "villain_range": "sb_open",
        },
    )

    pot_flop = pot_after_open + (sz - 1.0)
    n_flop, n_turn, n_river = _postflop_nodes(
        hero_pos=hero_pos,
        villain_pos=villain_pos,
        board=board,
        pot_flop=pot_flop,
        effective_bb=eff,
        hero_cards=hero_cards,
        hand_state=hand_state,
        open_size=sz,
        villain_range="sb_open",
    )

    hand_state["nodes"] = {
        "preflop": n_preflop,
        "flop": n_flop,
        "turn": n_turn,
        "river": n_river,
    }

    return Episode(
        nodes=[n_preflop, n_flop, n_turn, n_river],
        hero_seat=hero_pos,
        villain_seat=villain_pos,
    )


def _episode_sb_ip(
    rng: random.Random,
    *,
    stacks_bb: float,
    sb: float,
    bb: float,
) -> Episode:
    hero_pos = _SB
    villain_pos = _BB
    dealt: Dealt = deal_hand_and_board(rng)
    hero_cards = dealt.hero
    villain_cards = dealt.villain
    board = dealt.board

    open_sizes = [2.0, 2.5, 3.0]
    sz = rng.choice(open_sizes)

    # Starting pot 0.5 + 1.0 = 1.5; hero tops up to sz and villain calls.
    pot_after_open = sb + bb + (sz - sb)
    pot_flop = pot_after_open + (sz - 1.0)

    hand_state: dict[str, object] = {
        "pot": pot_flop,
        "hero_cards": tuple(hero_cards),
        "villain_cards": tuple(villain_cards),
        "full_board": tuple(board),
        "street": "flop",
        "history": [],
        "board_index": 3,
        "hero_seat": hero_pos,
        "villain_seat": villain_pos,
        "villain_range": "bb_defend",
    }

    nodes = _postflop_nodes(
        hero_pos=hero_pos,
        villain_pos=villain_pos,
        board=board,
        pot_flop=pot_flop,
        effective_bb=stacks_bb,
        hero_cards=hero_cards,
        hand_state=hand_state,
        open_size=sz,
        villain_range="bb_defend",
    )

    hand_state["nodes"] = {
        "flop": nodes[0],
        "turn": nodes[1],
        "river": nodes[2],
    }

    return Episode(
        nodes=nodes,
        hero_seat=hero_pos,
        villain_seat=villain_pos,
    )


def generate_episode(
    rng: random.Random,
    stacks_bb: float = 100.0,
    sb: float = 0.5,
    bb: float = 1.0,
    hero_seat: str | None = None,
) -> Episode:
    seat = (hero_seat or _BB).upper()
    if seat not in {_SB, _BB}:
        raise ValueError(f"Unsupported hero seat '{hero_seat}'")
    if seat == _BB:
        return _episode_bb_defense(rng, stacks_bb=stacks_bb, sb=sb, bb=bb)
    return _episode_sb_ip(rng, stacks_bb=stacks_bb, sb=sb, bb=bb)
