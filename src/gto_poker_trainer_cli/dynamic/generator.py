from __future__ import annotations

import random
from dataclasses import dataclass

from .cards import Dealt, deal_hand_and_board, format_card_ascii, format_cards_spaced


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


def generate_episode(rng: random.Random, stacks_bb: float = 100.0, sb: float = 0.5, bb: float = 1.0) -> Episode:
    # Heads-up only. Hero randomly SB or BB.
    rng.random()  # randomness placeholder; seating not yet used in storyline
    dealt: Dealt = deal_hand_and_board(rng)
    hero_cards = dealt.hero
    villain_cards = dealt.villain
    board = dealt.board

    # Preflop pot starts at 1.5bb (SB+BB)
    pot = sb + bb
    eff = stacks_bb

    # Preflop node: SB opens to s in {2.0, 2.5, 3.0}, hero is BB when villain opens
    open_sizes = [2.0, 2.5, 3.0]
    sz = rng.choice(open_sizes)
    desc_pf = f"SB opens {sz:.1f}bb. You're BB with {format_cards_spaced(hero_cards)}, {int(stacks_bb)}bb."
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
    }

    n_preflop = Node(
        street="preflop",
        description=desc_pf,
        pot_bb=pot_after_open,
        effective_bb=eff,
        hero_cards=hero_cards,
        board=[],
        actor="BB",
        context={"open_size": sz, "hand_state": hand_state},
    )

    # Flop node (hero BB facing check from SB)
    # After BB calls the open, pot increases by (sz - 1bb).
    # Example: sz=2.0 → pot_flop = 3.0 + (2.0 - 1.0) = 4.0
    pot_flop = pot_after_open + (sz - 1.0)
    flop_cards = board[:3]
    flop_str = " ".join(format_card_ascii(c, upper=True) for c in flop_cards)
    desc_flop = f"Board {flop_str}. SB checks."
    n_flop = Node(
        street="flop",
        description=desc_flop,
        pot_bb=pot_flop,
        effective_bb=eff,
        hero_cards=hero_cards,
        board=flop_cards,
        actor="BB",
        context={"facing": "check", "open_size": sz, "hand_state": hand_state},
    )

    # Turn node (villain bets half pot; hero to act)
    # If flop checks through, start-of-turn pot remains the same as end-of-flop pot
    pot_turn = pot_flop
    bet_turn = round(0.5 * pot_turn, 2)
    board_turn_str = " ".join(format_card_ascii(c, upper=True) for c in board[:4])
    desc_turn = f"Board {board_turn_str}. SB bets {bet_turn:.2f}bb into {pot_turn:.2f}bb."
    n_turn = Node(
        street="turn",
        description=desc_turn,
        pot_bb=pot_turn,
        effective_bb=eff,
        hero_cards=hero_cards,
        board=board[:4],
        actor="BB",
        context={"facing": "bet", "bet": bet_turn, "open_size": sz, "hand_state": hand_state},
    )

    # River node (hero in position, chooses bet size)
    # If hero calls the turn bet, both players contribute bet_turn: pot increases by 2× bet_turn
    pot_river = pot_turn + 2 * bet_turn
    river_str = " ".join(format_card_ascii(c, upper=True) for c in board)
    desc_river = f"Board {river_str}. Choose your bet size."
    n_river = Node(
        street="river",
        description=desc_river,
        pot_bb=pot_river,
        effective_bb=eff,
        hero_cards=hero_cards,
        board=board,
        actor="BB",
        context={"facing": "oop-check", "open_size": sz, "hand_state": hand_state},
    )

    hand_state["nodes"] = {
        "preflop": n_preflop,
        "flop": n_flop,
        "turn": n_turn,
        "river": n_river,
    }

    return Episode(nodes=[n_preflop, n_flop, n_turn, n_river])
