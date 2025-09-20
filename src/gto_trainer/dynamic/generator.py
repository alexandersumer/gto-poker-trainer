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


@dataclass(frozen=True)
class _VillainStyleConfig:
    name: str
    turn_bet_probability: float
    turn_bet_sizes: tuple[float, ...]
    turn_probe_sizes: tuple[float, ...]
    river_lead_probability: float
    river_lead_sizes: tuple[float, ...]
    turn_bet_tighten: float
    turn_probe_tighten: float
    river_check_tighten: float
    river_lead_tighten: float


_STYLE_LIBRARY: dict[str, _VillainStyleConfig] = {
    "balanced": _VillainStyleConfig(
        name="balanced",
        turn_bet_probability=0.65,
        turn_bet_sizes=(0.33, 0.5, 0.75, 1.0),
        turn_probe_sizes=(0.5, 0.8),
        river_lead_probability=0.35,
        river_lead_sizes=(0.5, 1.0, 1.25),
        turn_bet_tighten=0.55,
        turn_probe_tighten=0.6,
        river_check_tighten=0.65,
        river_lead_tighten=0.5,
    ),
    "aggressive": _VillainStyleConfig(
        name="aggressive",
        turn_bet_probability=0.8,
        turn_bet_sizes=(0.5, 0.75, 1.0),
        turn_probe_sizes=(0.66, 1.0),
        river_lead_probability=0.55,
        river_lead_sizes=(0.66, 1.0, 1.5),
        turn_bet_tighten=0.5,
        turn_probe_tighten=0.55,
        river_check_tighten=0.6,
        river_lead_tighten=0.45,
    ),
    "passive": _VillainStyleConfig(
        name="passive",
        turn_bet_probability=0.45,
        turn_bet_sizes=(0.33, 0.5),
        turn_probe_sizes=(0.4, 0.6),
        river_lead_probability=0.18,
        river_lead_sizes=(0.4, 0.75),
        turn_bet_tighten=0.6,
        turn_probe_tighten=0.68,
        river_check_tighten=0.7,
        river_lead_tighten=0.55,
    ),
}


def available_villain_styles() -> tuple[str, ...]:
    return tuple(_STYLE_LIBRARY.keys())


def _resolve_villain_style(style: str | None) -> _VillainStyleConfig:
    key = (style or "balanced").strip().lower()
    if key not in _STYLE_LIBRARY:
        raise ValueError(f"Unknown villain_style '{style}'. Options: {', '.join(available_villain_styles())}")
    return _STYLE_LIBRARY[key]


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
        villain_style: str = "balanced",
    ) -> None:
        self._rng = rng
        self._stacks = stacks_bb
        self._sb = sb
        self._bb = bb
        self._display_seats = seats
        self._tree_seats = seats if seats.hero == BB else SeatAssignment(hero=BB, villain=SB)
        self._rival_label = f"Rival ({self._display_seats.villain})"
        self._style = _resolve_villain_style(villain_style)

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
            "villain_style": self._style.name,
            "style_turn_bet_tighten": self._style.turn_bet_tighten,
            "style_turn_probe_tighten": self._style.turn_probe_tighten,
            "style_turn_probe_sizes": self._style.turn_probe_sizes,
            "style_river_check_tighten": self._style.river_check_tighten,
            "style_river_lead_tighten": self._style.river_lead_tighten,
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

        turn_board = ctx.board[:4]
        turn_desc = " ".join(format_card_ascii(card, upper=True) for card in turn_board)
        turn_mode = "bet" if self._rng.random() < self._style.turn_bet_probability else "check"
        hand_state["turn_mode"] = turn_mode
        if turn_mode == "bet":
            bet_multiplier = self._rng.choice(self._style.turn_bet_sizes)
            bet_turn = round(max(0.25, hand_state["pot"] * bet_multiplier), 2)
            hand_state["turn_bet_size"] = bet_turn
            turn_description = f"{turn_desc}; {self._rival_label} bets {bet_turn:.2f}bb into {hand_state['pot']:.2f}bb."
            turn_context = self._node_context(
                ctx,
                hand_state,
                extra={
                    "facing": "bet",
                    "bet": bet_turn,
                    "villain_range": ctx.villain_range,
                    "villain_style": self._style.name,
                },
            )
        else:
            bet_turn = 0.0
            turn_description = f"{turn_desc}; {self._rival_label} checks."
            turn_context = self._node_context(
                ctx,
                hand_state,
                extra={"facing": "check", "villain_range": ctx.villain_range, "villain_style": self._style.name},
            )

        turn_node = Node(
            street="turn",
            description=turn_description,
            pot_bb=hand_state["pot"],
            effective_bb=hand_state["effective_stack"],
            hero_cards=ctx.hero_cards,
            board=turn_board,
            actor=self._display_seats.hero,
            context=turn_context,
        )

        river_desc = " ".join(format_card_ascii(card, upper=True) for card in ctx.board)
        river_mode = "lead" if self._rng.random() < self._style.river_lead_probability else "check"
        hand_state["river_mode"] = river_mode
        if river_mode == "lead":
            lead_size = round(max(0.25, hand_state["pot"] * self._rng.choice(self._style.river_lead_sizes)), 2)
            hand_state["river_lead_size"] = lead_size
            river_description = (
                f"{river_desc}; {self._rival_label} leads {lead_size:.2f}bb into {hand_state['pot']:.2f}bb."
            )
            river_context = self._node_context(
                ctx,
                hand_state,
                extra={
                    "facing": "bet",
                    "bet": lead_size,
                    "villain_range": ctx.villain_range,
                    "villain_style": self._style.name,
                },
            )
        else:
            river_description = f"{river_desc}; choose your bet."
            river_context = self._node_context(
                ctx,
                hand_state,
                extra={"facing": "oop-check", "villain_range": ctx.villain_range, "villain_style": self._style.name},
            )

        river_node = Node(
            street="river",
            description=river_description,
            pot_bb=hand_state["pot"],
            effective_bb=hand_state["effective_stack"],
            hero_cards=ctx.hero_cards,
            board=ctx.board,
            actor=self._display_seats.hero,
            context=river_context,
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
    villain_style: str = "balanced",
) -> Episode:
    """Generate a fresh episode for the provided seat assignment.

    The legacy ``hero_seat`` argument is still supported for backwards
    compatibility; it is validated and converted into a ``SeatAssignment``.
    """

    seats = _resolve_seat_assignment(hero_seat=hero_seat, seat_assignment=seat_assignment)
    builder = EpisodeBuilder(
        rng,
        seats=seats,
        stacks_bb=stacks_bb,
        sb=sb,
        bb=bb,
        villain_style=villain_style,
    )
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
