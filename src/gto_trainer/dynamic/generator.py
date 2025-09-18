"""Episode generation utilities.

This module previously mixed dataclass declarations, seat assignment logic and
all of the node construction code in one large function.  The new structure
introduces a small builder type that encapsulates the common rules used to
create nodes, while delegating seat rotation concerns to
``dynamic.seating``.  The result is easier to test and reason about without
changing the external behaviour of ``generate_episode``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .cards import Dealt, deal_hand_and_board, format_card_ascii
from .episode import Episode, Node
from .seating import BB, SB, SeatAssignment

_SB_VILLAIN_RANGE = "sb_open"
_BB_VILLAIN_RANGE = "bb_defend"
_DEFAULT_STACKS = 100.0
_DEFAULT_SB = 0.5
_DEFAULT_BB = 1.0
_OPEN_SIZES = (2.0, 2.5, 3.0)


@dataclass
class _HandContext:
    hero_cards: list[int]
    villain_cards: list[int]
    board: list[int]
    open_size: float
    villain_range: str


class EpisodeBuilder:
    """Builds an episode for a specific seat assignment."""

    def __init__(
        self,
        rng: random.Random,
        seats: SeatAssignment,
        stacks_bb: float = _DEFAULT_STACKS,
        sb: float = _DEFAULT_SB,
        bb: float = _DEFAULT_BB,
    ) -> None:
        self._rng = rng
        self._stacks = stacks_bb
        self._sb = sb
        self._bb = bb
        self._display_seats = seats
        self._tree_seats = seats if seats.hero == BB else SeatAssignment(hero=BB, villain=SB)
        self._rival_label = f"Rival ({self._display_seats.villain})"

    def build(self) -> Episode:
        return self._build_classic_tree()

    # ------------------------------------------------------------------
    # Seat specific builders

    def _build_classic_tree(self) -> Episode:
        dealt = self._deal()
        villain_range = _SB_VILLAIN_RANGE if self._tree_seats.villain == SB else _BB_VILLAIN_RANGE
        ctx = self._hand_context(villain_range=villain_range, dealt=dealt)

        hand_state = self._base_state(
            ctx,
            street="preflop",
            board_index=0,
        )

        preflop = Node(
            street="preflop",
            description=(f"{self._rival_label} opens {ctx.open_size:.1f}bb."),
            pot_bb=hand_state["pot"],
            effective_bb=hand_state["effective_stack"],
            hero_cards=ctx.hero_cards,
            board=[],
            actor=self._display_seats.hero,
            context=self._node_context(
                ctx,
                hand_state,
                extra={"villain_range": ctx.villain_range},
            ),
        )

        postflop = self._postflop_nodes(ctx, hand_state)
        hand_state["nodes"] = {
            "preflop": preflop,
            "flop": postflop[0],
            "turn": postflop[1],
            "river": postflop[2],
        }

        return Episode(
            nodes=[preflop, *postflop],
            hero_seat=self._display_seats.hero,
            villain_seat=self._display_seats.villain,
        )

    # ------------------------------------------------------------------
    # Helpers

    def _deal(self) -> Dealt:
        return deal_hand_and_board(self._rng)

    def _hand_context(self, *, villain_range: str, dealt: Dealt) -> _HandContext:
        open_size = self._rng.choice(_OPEN_SIZES)
        return _HandContext(
            hero_cards=list(dealt.hero),
            villain_cards=list(dealt.villain),
            board=list(dealt.board),
            open_size=open_size,
            villain_range=villain_range,
        )

    def _base_state(
        self,
        ctx: _HandContext,
        *,
        street: str,
        board_index: int,
    ) -> dict:
        hero_contrib, villain_contrib = self._initial_contributions(ctx)
        hero_stack = max(0.0, self._stacks - hero_contrib)
        villain_stack = max(0.0, self._stacks - villain_contrib)
        pot = hero_contrib + villain_contrib
        effective_stack = min(hero_stack, villain_stack)

        return {
            "pot": pot,
            "hero_cards": tuple(ctx.hero_cards),
            "villain_cards": tuple(ctx.villain_cards),
            "full_board": tuple(ctx.board),
            "street": street,
            "history": [],
            "board_index": board_index,
            "hero_seat": self._display_seats.hero,
            "villain_seat": self._display_seats.villain,
            "villain_range": ctx.villain_range,
            "hero_contrib": hero_contrib,
            "villain_contrib": villain_contrib,
            "hero_stack": hero_stack,
            "villain_stack": villain_stack,
            "effective_stack": effective_stack,
        }

    def _node_context(
        self,
        ctx: _HandContext,
        hand_state: dict,
        *,
        extra: dict | None = None,
    ) -> dict:
        base = {
            "open_size": ctx.open_size,
            "hand_state": hand_state,
            "hero_seat": self._display_seats.hero,
            "villain_seat": self._display_seats.villain,
        }
        if extra:
            base.update(extra)
        return base

    def _postflop_nodes(
        self,
        ctx: _HandContext,
        hand_state: dict,
    ) -> list[Node]:
        flop_cards = ctx.board[:3]
        flop_desc = " ".join(format_card_ascii(card, upper=True) for card in flop_cards)
        flop_node = Node(
            street="flop",
            description=f"{flop_desc}; {self._rival_label} checks.",
            pot_bb=hand_state["pot"],
            effective_bb=hand_state["effective_stack"],
            hero_cards=ctx.hero_cards,
            board=flop_cards,
            actor=self._display_seats.hero,
            context=self._node_context(
                ctx,
                hand_state,
                extra={"facing": "check", "villain_range": ctx.villain_range},
            ),
        )

        bet_turn = round(0.5 * hand_state["pot"], 2)
        turn_board = ctx.board[:4]
        turn_desc = " ".join(format_card_ascii(card, upper=True) for card in turn_board)
        turn_node = Node(
            street="turn",
            description=(f"{turn_desc}; {self._rival_label} bets {bet_turn:.2f}bb into {hand_state['pot']:.2f}bb."),
            pot_bb=hand_state["pot"],
            effective_bb=hand_state["effective_stack"],
            hero_cards=ctx.hero_cards,
            board=turn_board,
            actor=self._display_seats.hero,
            context=self._node_context(
                ctx,
                hand_state,
                extra={"facing": "bet", "bet": bet_turn, "villain_range": ctx.villain_range},
            ),
        )

        river_desc = " ".join(format_card_ascii(card, upper=True) for card in ctx.board)
        river_node = Node(
            street="river",
            description=f"{river_desc}; choose your bet.",
            pot_bb=hand_state["pot"],
            effective_bb=hand_state["effective_stack"],
            hero_cards=ctx.hero_cards,
            board=ctx.board,
            actor=self._display_seats.hero,
            context=self._node_context(
                ctx,
                hand_state,
                extra={"facing": "oop-check", "villain_range": ctx.villain_range},
            ),
        )

        return [flop_node, turn_node, river_node]

    def _initial_contributions(self, ctx: _HandContext) -> tuple[float, float]:
        hero_contrib = self._blind_for(self._display_seats.hero)
        villain_contrib = self._blind_for(self._display_seats.villain)

        opener_blind = self._blind_for(self._display_seats.villain)
        additional = max(0.0, ctx.open_size - opener_blind)
        villain_contrib += additional

        return hero_contrib, villain_contrib

    def _blind_for(self, seat: str) -> float:
        return self._sb if seat == SB else self._bb


def generate_episode(
    rng: random.Random,
    stacks_bb: float = _DEFAULT_STACKS,
    sb: float = _DEFAULT_SB,
    bb: float = _DEFAULT_BB,
    hero_seat: str | None = None,
    *,
    seat_assignment: SeatAssignment | None = None,
) -> Episode:
    """Generate a fresh episode for the provided seat assignment.

    The legacy ``hero_seat`` argument is still supported for backwards
    compatibility; it is validated and converted into a ``SeatAssignment``.
    """

    seats = _resolve_seat_assignment(hero_seat=hero_seat, seat_assignment=seat_assignment)
    builder = EpisodeBuilder(rng, seats=seats, stacks_bb=stacks_bb, sb=sb, bb=bb)
    return builder.build()


def _resolve_seat_assignment(*, hero_seat: str | None, seat_assignment: SeatAssignment | None) -> SeatAssignment:
    if seat_assignment and hero_seat:
        hero = hero_seat.upper()
        if hero not in {SB, BB}:
            raise ValueError(f"Unsupported hero seat '{hero_seat}'")
        expected = seat_assignment.hero
        if hero != expected:
            raise ValueError("hero_seat does not match seat_assignment; pass only one of these arguments")
        return seat_assignment

    if seat_assignment:
        return seat_assignment

    hero = (hero_seat or BB).upper()
    if hero not in {SB, BB}:
        raise ValueError(f"Unsupported hero seat '{hero_seat}'")
    villain = SB if hero == BB else BB
    return SeatAssignment(hero=hero, villain=villain)
