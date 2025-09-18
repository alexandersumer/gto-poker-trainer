from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

from ..core.models import Option, OptionResolution
from .cards import format_card_ascii, format_cards_spaced
from .episode import Node
from .equity import hero_equity_vs_combo, hero_equity_vs_range as _hero_equity_vs_range
from .range_model import tighten_range, villain_bb_defend_range, villain_sb_open_range


def _fmt_pct(x: float, decimals: int = 0) -> str:
    return f"{100.0 * x:.{decimals}f}%"


def _blocked_cards(hero: Iterable[int], board: Iterable[int]) -> set[int]:
    return set(hero) | set(board)


def _state_value(hand_state: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not hand_state:
        return default
    value = hand_state.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _recalc_pot(hand_state: dict[str, Any]) -> float:
    if "hero_contrib" in hand_state and "villain_contrib" in hand_state:
        pot = _state_value(hand_state, "hero_contrib") + _state_value(hand_state, "villain_contrib")
    else:
        pot = _state_value(hand_state, "pot")
    hand_state["pot"] = pot
    return pot


def _update_effective_stack(hand_state: dict[str, Any]) -> float:
    hero_stack = _state_value(hand_state, "hero_stack")
    villain_stack = _state_value(hand_state, "villain_stack")
    effective = min(hero_stack, villain_stack)
    hand_state["effective_stack"] = effective
    nodes = hand_state.get("nodes")
    if isinstance(nodes, dict):
        for node in nodes.values():
            if isinstance(node, Node):
                node.effective_bb = effective
    return effective


def _apply_contribution(hand_state: dict[str, Any], role: str, amount: float) -> float:
    if not hand_state or amount <= 0:
        return 0.0
    stack_key = f"{role}_stack"
    contrib_key = f"{role}_contrib"
    default_stack = float(hand_state.get("effective_stack", 100.0))
    stack = _state_value(hand_state, stack_key, default_stack)
    if stack <= 0:
        return 0.0
    applied = min(amount, stack)
    current_contrib = _state_value(hand_state, contrib_key)
    hand_state[contrib_key] = current_contrib + applied
    hand_state[stack_key] = max(0.0, stack - applied)
    _recalc_pot(hand_state)
    _update_effective_stack(hand_state)
    return applied


def _fold_continue_stats(hero_equities: Iterable[float], villain_threshold: float) -> tuple[float, float, float]:
    hero_list = list(hero_equities)
    if not hero_list:
        return 0.0, 0.0, 0.0
    folds = 0
    continue_eq = 0.0
    for eq in hero_list:
        if 1.0 - eq < villain_threshold:
            folds += 1
        else:
            continue_eq += eq
    total = len(hero_list)
    continue_count = total - folds
    fe = folds / total
    avg_eq = continue_eq / continue_count if continue_count else 0.0
    continue_ratio = continue_count / total
    return fe, avg_eq, continue_ratio


def _combo_trials(mc_trials: int) -> int:
    return max(40, int(mc_trials * 0.55))


def _sample_cap_preflop(mc_trials: int) -> int:
    return max(50, min(200, int(mc_trials * 1.2)))


def _sample_cap_postflop(mc_trials: int) -> int:
    return max(30, min(120, int(mc_trials * 0.6)))


def _sample_range(
    combos: Iterable[tuple[int, int]],
    limit: int,
) -> list[tuple[int, int]]:
    combos_list = list(combos)
    if len(combos_list) <= limit:
        return combos_list
    # Take evenly spaced combos across the ordered range to keep a representative mix.
    step = len(combos_list) / limit
    sampled: list[tuple[int, int]] = []
    for i in range(limit):
        idx = int(i * step)
        if idx >= len(combos_list):
            idx = len(combos_list) - 1
        sampled.append(combos_list[idx])
    return sampled


def _default_open_size(node: Node) -> float:
    return float(node.context.get("open_size") or 2.5)


# Public alias preserved for tests that monkeypatch hero_equity_vs_range.
hero_equity_vs_range = _hero_equity_vs_range


def _hand_state(node: Node) -> dict[str, Any] | None:
    hand_state = node.context.get("hand_state")
    if isinstance(hand_state, dict):
        return hand_state
    return None


def _villain_range_tag(node: Node, default: str = "sb_open") -> str:
    ctx_val = node.context.get("villain_range")
    if isinstance(ctx_val, str) and ctx_val:
        return ctx_val
    hand_state = _hand_state(node)
    if hand_state:
        hs_val = hand_state.get("villain_range")
        if isinstance(hs_val, str) and hs_val:
            return hs_val
    return default


def _villain_base_range(node: Node, blocked: Iterable[int]) -> list[tuple[int, int]]:
    tag = _villain_range_tag(node)
    open_size = _default_open_size(node)
    if tag == "bb_defend":
        return villain_bb_defend_range(open_size, blocked)
    return villain_sb_open_range(open_size, blocked)


def _set_node_pot_from_state(node: Node, hand_state: dict[str, Any] | None) -> float:
    if not hand_state:
        return float(node.pot_bb)
    pot = _recalc_pot(hand_state)
    node.pot_bb = pot
    node.effective_bb = _state_value(hand_state, "effective_stack", node.effective_bb)
    return pot


def _villain_cards(hand_state: dict[str, Any] | None) -> tuple[int, int] | None:
    if not hand_state:
        return None
    cards = hand_state.get("villain_cards")
    if isinstance(cards, tuple) and len(cards) == 2:
        return cards  # type: ignore[return-value]
    if isinstance(cards, list) and len(cards) == 2:
        return tuple(cards)  # type: ignore[return-value]
    return None


def _villain_str(hand_state: dict[str, Any] | None, reveal: bool) -> str:
    cards = _villain_cards(hand_state)
    if reveal and cards:
        return format_cards_spaced(list(cards))
    return "(hidden)"


def _set_street_pot(hand_state: dict[str, Any], street: str, pot: float) -> None:
    nodes = hand_state.get("nodes")
    if not isinstance(nodes, dict):
        return
    node = nodes.get(street)
    if isinstance(node, Node):
        node.pot_bb = pot
        node.effective_bb = _state_value(hand_state, "effective_stack", node.effective_bb)


def _rebuild_turn_node(hand_state: dict[str, Any], pot: float) -> None:
    nodes = hand_state.get("nodes")
    if not isinstance(nodes, dict):
        return
    turn_node = nodes.get("turn")
    if not isinstance(turn_node, Node):
        return
    turn_node.pot_bb = pot
    turn_node.effective_bb = _state_value(hand_state, "effective_stack", turn_node.effective_bb)
    bet_turn = round(0.5 * pot, 2)
    turn_node.context["bet"] = bet_turn
    board_turn = turn_node.board
    board_str = " ".join(format_card_ascii(c, upper=True) for c in board_turn)
    villain_seat = str(hand_state.get("villain_seat", "SB"))
    turn_node.description = f"{board_str}; Rival ({villain_seat}) bets {bet_turn:.2f}bb into {pot:.2f}bb."


def _rebuild_river_node(hand_state: dict[str, Any], pot: float) -> None:
    nodes = hand_state.get("nodes")
    if not isinstance(nodes, dict):
        return
    river_node = nodes.get("river")
    if not isinstance(river_node, Node):
        return
    river_node.pot_bb = pot
    river_node.effective_bb = _state_value(hand_state, "effective_stack", river_node.effective_bb)
    board_river = river_node.board
    board_str = " ".join(format_card_ascii(c, upper=True) for c in board_river)
    river_node.description = f"{board_str}; choose your bet."


def _showdown_outcome(hero: list[int], board: list[int], villain: tuple[int, int]) -> float:
    eq = hero_equity_vs_combo(hero, board, villain, 1)
    if eq >= 0.999:
        return 1.0
    if eq <= 0.001:
        return 0.0
    return 0.5


def preflop_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    del rng
    hero = node.hero_cards
    open_size = float(node.context.get("open_size") or 2.5)
    hand_state = _hand_state(node)
    pot = _set_node_pot_from_state(node, hand_state)
    hero_contrib = _state_value(hand_state, "hero_contrib", 1.0)
    villain_contrib = _state_value(hand_state, "villain_contrib", open_size)
    hero_stack = _state_value(hand_state, "hero_stack", node.effective_bb)
    villain_stack = _state_value(hand_state, "villain_stack", node.effective_bb)
    hero_total = hero_contrib + hero_stack
    villain_total = villain_contrib + villain_stack

    call_cost = max(0.0, min(hero_stack, villain_contrib - hero_contrib))

    blocked = _blocked_cards(hero, [])
    open_range = _villain_base_range(node, blocked)
    sampled_range = _sample_range(open_range, _sample_cap_preflop(mc_trials)) or open_range
    combo_trials = _combo_trials(mc_trials)
    equities = {combo: hero_equity_vs_combo(hero, [], combo, combo_trials) for combo in sampled_range}
    avg_range_eq = sum(equities.values()) / len(equities) if equities else 0.0

    options: list[Option] = [
        Option(
            "Fold",
            0.0,
            "Fold now and lose nothing extra.",
            ends_hand=True,
            meta={"street": "preflop", "action": "fold"},
        ),
    ]

    if call_cost > 0:
        final_pot_call = pot + call_cost
        be_call_eq = call_cost / final_pot_call if final_pot_call > 0 else 1.0
        options.append(
            Option(
                "Call",
                avg_range_eq * final_pot_call - call_cost,
                (
                    f"Pot odds: call {call_cost:.2f} bb to win {final_pot_call:.2f} bb. "
                    f"Need ≈{_fmt_pct(be_call_eq, 1)} equity, hand has {_fmt_pct(avg_range_eq, 1)}."
                ),
                meta={
                    "street": "preflop",
                    "action": "call",
                    "call_cost": call_cost,
                },
            )
        )

    max_raise = hero_total
    raise_targets: set[float] = set()
    for mult in (2.8, 3.5, 5.0):
        target = round(open_size * mult, 2)
        if target <= villain_contrib + 0.25:
            continue
        target = min(target, max_raise)
        if target >= max_raise - 0.05:
            continue
        if target <= hero_contrib + 0.25:
            continue
        raise_targets.add(round(target, 2))

    for raise_to in sorted(raise_targets):
        hero_add = raise_to - hero_contrib
        if hero_add <= 0 or hero_add > hero_stack + 1e-6:
            continue
        villain_call_to = max(0.0, raise_to - villain_contrib)
        villain_call = min(villain_call_to, villain_stack)
        final_pot = pot + hero_add + villain_call
        if final_pot <= 0:
            continue
        be_threshold = villain_call / final_pot if final_pot > 0 else 1.0
        fe, avg_eq_when_called, continue_ratio = _fold_continue_stats(equities.values(), be_threshold)
        ev_called = avg_eq_when_called * final_pot - hero_add if continue_ratio else -hero_add
        ev = fe * pot + (1 - fe) * ev_called
        why = (
            f"Rival folds {_fmt_pct(fe)} needing eq {_fmt_pct(be_threshold, 1)}. "
            f"When called (~{_fmt_pct(continue_ratio)}) you have {_fmt_pct(avg_eq_when_called, 1)} equity "
            f"→ EV {ev_called:.2f} bb."
        )
        options.append(
            Option(
                f"3-bet to {raise_to:.2f}bb",
                ev,
                why,
                meta={
                    "street": "preflop",
                    "action": "3bet",
                    "raise_to": raise_to,
                    "villain_threshold": be_threshold,
                    "combo_trials": combo_trials,
                    "pot_before": pot,
                    "hero_add": hero_add,
                    "villain_call": villain_call,
                },
            )
        )

    hero_add = hero_stack
    if hero_add > 0.0:
        jam_to = hero_contrib + hero_add
        villain_call_to = max(0.0, min(jam_to, villain_total) - villain_contrib)
        villain_call = min(villain_call_to, villain_stack)
        final_pot = pot + hero_add + villain_call
        be_threshold = villain_call / final_pot if final_pot > 0 else 1.0
        fe, avg_eq_when_called, continue_ratio = _fold_continue_stats(equities.values(), be_threshold)
        ev_called = avg_eq_when_called * final_pot - hero_add if continue_ratio else -hero_add
        ev = fe * pot + (1 - fe) * ev_called
        why_jam = (
            f"Rival folds {_fmt_pct(fe)} needing eq {_fmt_pct(be_threshold, 1)}. "
            f"When called (~{_fmt_pct(continue_ratio)}) you have {_fmt_pct(avg_eq_when_called, 1)} equity "
            f"→ EV {ev_called:.2f} bb."
        )
        options.append(
            Option(
                "All-in",
                ev,
                why_jam,
                ends_hand=True,
                meta={
                    "street": "preflop",
                    "action": "jam",
                    "raise_to": jam_to,
                    "risk": hero_add,
                    "villain_threshold": be_threshold,
                    "combo_trials": combo_trials,
                    "pot_before": pot,
                    "villain_call_cost": villain_call_to,
                },
            )
        )

    return options


def flop_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    del rng
    hero = node.hero_cards
    board = node.board
    hand_state = _hand_state(node)
    pot = _set_node_pot_from_state(node, hand_state)
    blocked = _blocked_cards(hero, board)
    open_range = _villain_base_range(node, blocked)
    sampled_range = _sample_range(open_range, _sample_cap_postflop(mc_trials)) or open_range
    combo_trials = _combo_trials(mc_trials)
    equities = {combo: hero_equity_vs_combo(hero, board, combo, combo_trials) for combo in sampled_range}
    avg_eq = sum(equities.values()) / len(equities) if equities else 0.0

    options: list[Option] = [
        Option(
            "Check",
            avg_eq * pot,
            f"Realize equity {_fmt_pct(avg_eq, 1)} in-position.",
            meta={"street": "flop", "action": "check"},
        ),
    ]

    for pct in (0.33, 0.5, 0.75):
        bet = round(pot * pct, 2)
        if bet <= 0:
            continue
        final_pot = pot + 2 * bet
        be_threshold = bet / final_pot
        fe, eq_call, continue_ratio = _fold_continue_stats(equities.values(), be_threshold)
        ev_called = eq_call * final_pot - bet if continue_ratio else -bet
        ev = fe * pot + (1 - fe) * ev_called
        why = (
            f"{int(pct * 100)}% pot: rival folds {_fmt_pct(fe)} (needs eq {_fmt_pct(be_threshold, 1)}). "
            f"Continuing range (~{_fmt_pct(continue_ratio)}) gives you {_fmt_pct(eq_call, 1)} equity "
            f"→ EV {ev_called:.2f} bb."
        )
        why += f" Additional sizing detail: {bet:.2f} bb (equals {int(pct * 100)}% pot)."
        options.append(
            Option(
                f"Bet {int(pct * 100)}% pot",
                ev,
                why,
                meta={
                    "street": "flop",
                    "action": "bet",
                    "bet": bet,
                    "villain_threshold": be_threshold,
                    "combo_trials": combo_trials,
                },
            )
        )

    # All-in shove option for maximum pressure
    risk = round(node.effective_bb, 2)
    if risk > 0:
        final_pot = pot + 2 * risk
        be_threshold = risk / final_pot if final_pot > 0 else 1.0
        fe, eq_call, continue_ratio = _fold_continue_stats(equities.values(), be_threshold)
        ev_called = eq_call * final_pot - risk if continue_ratio else -risk
        ev = fe * pot + (1 - fe) * ev_called
        options.append(
            Option(
                "All-in",
                ev,
                (
                    f"Full stack shove: rival folds {_fmt_pct(fe)} needing eq {_fmt_pct(be_threshold, 1)}. "
                    f"When called you have {_fmt_pct(eq_call, 1)} → EV {ev_called:.2f} bb."
                ),
                ends_hand=True,
                meta={
                    "street": "flop",
                    "action": "jam",
                    "risk": risk,
                    "combo_trials": combo_trials,
                    "villain_threshold": be_threshold,
                    "pot_before": pot,
                },
            )
        )

    return options


def turn_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    del rng
    hero = node.hero_cards
    board = node.board
    hand_state = _hand_state(node)
    pot_start = _set_node_pot_from_state(node, hand_state)
    villain_bet = float(node.context.get("bet") or round(0.5 * pot_start, 2))
    node.context["bet"] = villain_bet
    pot_before_action = pot_start + villain_bet
    blocked = _blocked_cards(hero, board)
    base_range = _villain_base_range(node, blocked)
    # Rival betting range is tightened to the stronger half of their holdings.
    bet_range = tighten_range(base_range, 0.55)
    sampled_range = _sample_range(bet_range, _sample_cap_postflop(mc_trials)) or bet_range
    combo_trials = _combo_trials(mc_trials)
    equities = {combo: hero_equity_vs_combo(hero, board, combo, combo_trials) for combo in sampled_range}
    avg_eq = sum(equities.values()) / len(equities) if equities else 0.0

    options = [
        Option(
            "Fold",
            0.0,
            "Release the hand and wait for better spot.",
            ends_hand=True,
            meta={"street": "turn", "action": "fold", "villain_bet": villain_bet},
        )
    ]

    final_pot_call = pot_start + 2 * villain_bet
    be_call_eq = villain_bet / final_pot_call if final_pot_call > 0 else 1.0
    options.append(
        Option(
            "Call",
            avg_eq * final_pot_call - villain_bet,
            (
                f"Pot odds: call {villain_bet:.2f} bb to win {final_pot_call:.2f} bb. "
                f"Need {_fmt_pct(be_call_eq, 1)} equity, hand has {_fmt_pct(avg_eq, 1)}."
            ),
            meta={
                "street": "turn",
                "action": "call",
                "villain_bet": villain_bet,
                "combo_trials": combo_trials,
            },
        )
    )

    raise_to = round(villain_bet * 2.5, 2)
    risk = raise_to
    final_pot = pot_start + 2 * raise_to
    villain_call_cost = raise_to - villain_bet
    be_threshold = villain_call_cost / final_pot if final_pot > 0 else 1.0
    fe, eq_call, continue_ratio = _fold_continue_stats(equities.values(), be_threshold)
    ev_called = eq_call * final_pot - risk if continue_ratio else -risk
    ev = fe * pot_before_action + (1 - fe) * ev_called
    fe_break_even = risk / (risk + pot_before_action) if (risk + pot_before_action) > 0 else 1.0
    why_raise = (
        f"Rival folds {_fmt_pct(fe)} (needs eq {_fmt_pct(be_threshold, 1)}). "
        f"break-even FE {_fmt_pct(fe_break_even)}. "
        f"Continuing (~{_fmt_pct(continue_ratio)}) you have {_fmt_pct(eq_call, 1)} equity → "
        f"EV {ev_called:.2f} bb."
    )
    options.append(
        Option(
            f"Raise to {raise_to:.2f} bb",
            ev,
            why_raise,
            meta={
                "street": "turn",
                "action": "raise",
                "raise_to": raise_to,
                "villain_threshold": be_threshold,
                "villain_bet": villain_bet,
                "combo_trials": combo_trials,
                "pot_before": pot_start,
            },
        )
    )

    return options


def river_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    del rng
    hero = node.hero_cards
    board = node.board
    hand_state = _hand_state(node)
    pot = _set_node_pot_from_state(node, hand_state)
    blocked = _blocked_cards(hero, board)
    base_range = _villain_base_range(node, blocked)
    # After checking river, assume villain keeps medium-strength holdings.
    check_range = tighten_range(base_range, 0.65)
    sampled_range = _sample_range(check_range, _sample_cap_postflop(mc_trials)) or check_range
    combo_trials = _combo_trials(mc_trials)
    equities = {combo: hero_equity_vs_combo(hero, board, combo, combo_trials) for combo in sampled_range}
    avg_eq = sum(equities.values()) / len(equities) if equities else 0.0

    options: list[Option] = [
        Option(
            "Check",
            avg_eq * pot,
            f"Showdown equity {_fmt_pct(avg_eq, 1)} vs check range.",
            meta={"street": "river", "action": "check", "combo_trials": combo_trials},
        )
    ]

    for pct in (0.5, 1.0):
        bet = round(pot * pct, 2)
        if bet <= 0:
            continue
        final_pot = pot + 2 * bet
        be_threshold = bet / final_pot if final_pot > 0 else 1.0
        fe, eq_call, continue_ratio = _fold_continue_stats(equities.values(), be_threshold)
        # When called on the river there are no future cards; hero wins pot + villain's
        # call with probability eq_call and otherwise loses their bet.
        ev_showdown = eq_call * (pot + bet) - (1 - eq_call) * bet
        ev_called = ev_showdown if continue_ratio else -bet
        ev = fe * pot + (1 - fe) * ev_called
        why = (
            f"Bet {int(pct * 100)}%: rival folds {_fmt_pct(fe)} (needs eq {_fmt_pct(be_threshold, 1)}). "
            f"Calls (~{_fmt_pct(continue_ratio)}) give you {_fmt_pct(eq_call, 1)} equity → EV {ev_called:.2f} bb."
        )
        why += f" Additional sizing detail: {bet:.2f} bb (equals {int(pct * 100)}% pot)."
        options.append(
            Option(
                f"Bet {int(pct * 100)}% pot",
                ev,
                why,
                meta={
                    "street": "river",
                    "action": "bet",
                    "bet": bet,
                    "villain_threshold": be_threshold,
                    "combo_trials": combo_trials,
                },
            )
        )

    risk = round(node.effective_bb, 2)
    if risk > 0:
        final_pot = pot + 2 * risk
        be_threshold = risk / final_pot if final_pot > 0 else 1.0
        fe, eq_call, continue_ratio = _fold_continue_stats(equities.values(), be_threshold)
        ev_called = eq_call * final_pot - risk if continue_ratio else -risk
        ev = fe * pot + (1 - fe) * ev_called
        options.append(
            Option(
                "All-in",
                ev,
                (
                    f"All-in: rival folds {_fmt_pct(fe)} needing eq {_fmt_pct(be_threshold, 1)}. "
                    f"Calls (~{_fmt_pct(continue_ratio)}) give you {_fmt_pct(eq_call, 1)} → EV {ev_called:.2f} bb."
                ),
                ends_hand=True,
                meta={
                    "street": "river",
                    "action": "jam",
                    "risk": risk,
                    "combo_trials": combo_trials,
                    "villain_threshold": be_threshold,
                    "pot_before": pot,
                },
            )
        )

    return options


def options_for(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    if node.street == "preflop":
        return preflop_options(node, rng, mc_trials)
    if node.street == "flop":
        return flop_options(node, rng, mc_trials)
    if node.street == "turn":
        return turn_options(node, rng, mc_trials)
    if node.street == "river":
        return river_options(node, rng, mc_trials)
    raise ValueError(f"Unknown street: {node.street}")


def resolve_for(node: Node, option: Option, rng: random.Random) -> OptionResolution:
    del rng
    meta = option.meta or {}
    hand_state = _hand_state(node)
    if not hand_state:
        return OptionResolution(hand_ended=getattr(option, "ends_hand", False))

    street = str(meta.get("street") or node.street)
    if street == "preflop":
        return _resolve_preflop(node, option, hand_state)
    if street == "flop":
        return _resolve_flop(node, option, hand_state)
    if street == "turn":
        return _resolve_turn(node, option, hand_state)
    if street == "river":
        return _resolve_river(node, option, hand_state)
    return OptionResolution(hand_ended=getattr(option, "ends_hand", False))


def _resolve_preflop(node: Node, option: Option, hand_state: dict[str, Any]) -> OptionResolution:
    action = option.meta.get("action") if option.meta else None
    pot = _state_value(hand_state, "pot", node.pot_bb)
    open_size = float(node.context.get("open_size", 2.5))
    villain_cards = _villain_cards(hand_state)
    hero_cards = node.hero_cards
    hero_contrib = _state_value(hand_state, "hero_contrib")
    villain_contrib = _state_value(hand_state, "villain_contrib", open_size)

    if action == "fold":
        hand_state["hand_over"] = True
        villain_seat = str(hand_state.get("villain_seat", "SB"))
        return OptionResolution(hand_ended=True, note=f"You fold. {villain_seat} keeps {pot:.2f}bb.")

    if action == "call":
        call_cost = float(option.meta.get("call_cost", max(0.0, villain_contrib - hero_contrib)))
        invested = _apply_contribution(hand_state, "hero", call_cost)
        hand_state["street"] = "flop"
        hand_state["board_index"] = 3
        _set_street_pot(hand_state, "flop", _state_value(hand_state, "pot"))
        return OptionResolution(note=f"You call {invested:.2f}bb. Pot now {hand_state['pot']:.2f}bb.")

    if action == "3bet":
        raise_to = float(option.meta.get("raise_to", hero_contrib))
        hero_add = max(0.0, raise_to - hero_contrib)
        _apply_contribution(hand_state, "hero", hero_add)
        combo_trials = int(option.meta.get("combo_trials", _combo_trials(80)))
        threshold = float(option.meta.get("villain_threshold", 0.45))
        if villain_cards is None:
            call_amount = max(0.0, raise_to - villain_contrib)
            _apply_contribution(hand_state, "villain", call_amount)
            hand_state["street"] = "flop"
            hand_state["board_index"] = 3
            _set_street_pot(hand_state, "flop", _state_value(hand_state, "pot"))
            return OptionResolution(note=f"3-bet to {raise_to:.1f}bb. Pot now {hand_state['pot']:.2f}bb.")
        hero_eq = hero_equity_vs_combo(hero_cards, [], villain_cards, combo_trials)
        villain_eq = 1.0 - hero_eq
        if villain_eq < threshold:
            hand_state["hand_over"] = True
            total_pot = _state_value(hand_state, "pot")
            net_gain = total_pot - hero_add
            note = f"Rival folds to your 3-bet. Pot {total_pot:.2f}bb awarded (net +{net_gain:.2f}bb)."
            return OptionResolution(hand_ended=True, note=note)
        call_amount = max(0.0, raise_to - villain_contrib)
        _apply_contribution(hand_state, "villain", call_amount)
        hand_state["street"] = "flop"
        hand_state["board_index"] = 3
        _set_street_pot(hand_state, "flop", _state_value(hand_state, "pot"))
        return OptionResolution(note=f"Rival calls the 3-bet. Pot {hand_state['pot']:.2f}bb.")

    if action in {"jam", "allin", "all-in", "jam"}:
        jam_to = float(option.meta.get("raise_to", hero_contrib + _state_value(hand_state, "hero_stack")))
        hero_add = max(0.0, jam_to - hero_contrib)
        _apply_contribution(hand_state, "hero", hero_add)
        combo_trials = int(option.meta.get("combo_trials", _combo_trials(80)))
        threshold = float(option.meta.get("villain_threshold", 0.5))
        hand_state["hand_over"] = True
        if villain_cards is None:
            scoop = _state_value(hand_state, "pot")
            return OptionResolution(hand_ended=True, note=f"You jam to {jam_to:.2f}bb. Rival action hidden.")
        hero_eq = hero_equity_vs_combo(hero_cards, [], villain_cards, combo_trials)
        villain_eq = 1.0 - hero_eq
        villain_text = format_cards_spaced(list(villain_cards))
        if villain_eq < threshold:
            scoop = _state_value(hand_state, "pot")
            net_gain = scoop - hero_add
            note = f"Rival folds to your jam. Pot {scoop:.2f}bb awarded (net +{net_gain:.2f}bb)."
            return OptionResolution(hand_ended=True, note=note, reveal_villain=True)
        call_amount = max(0.0, min(jam_to, villain_contrib + _state_value(hand_state, "villain_stack")) - villain_contrib)
        _apply_contribution(hand_state, "villain", call_amount)
        equity_note = _fmt_pct(hero_eq, 1)
        return OptionResolution(
            hand_ended=True,
            note=f"Rival calls jam with {villain_text}. Your equity {equity_note}.",
            reveal_villain=True,
        )

    return OptionResolution(hand_ended=getattr(option, "ends_hand", False))


def _resolve_flop(node: Node, option: Option, hand_state: dict[str, Any]) -> OptionResolution:
    action = option.meta.get("action") if option.meta else None
    pot = _state_value(hand_state, "pot", node.pot_bb)
    hand_state["street"] = "flop"
    villain_cards = _villain_cards(hand_state)
    hero_cards = node.hero_cards
    board = node.board

    if action == "check":
        hand_state["board_index"] = 4
        _set_street_pot(hand_state, "turn", pot)
        _rebuild_turn_node(hand_state, pot)
        return OptionResolution(note=f"You check back. Pot stays {pot:.2f}bb.")

    if action == "bet":
        bet_size = float(option.meta.get("bet", 0.0))
        threshold = float(option.meta.get("villain_threshold", 0.5))
        combo_trials = int(option.meta.get("combo_trials", _combo_trials(60)))
        pot_before = _state_value(hand_state, "pot")
        _apply_contribution(hand_state, "hero", bet_size)
        if villain_cards is None:
            call_amount = min(bet_size, _state_value(hand_state, "villain_stack"))
            _apply_contribution(hand_state, "villain", call_amount)
            hand_state["board_index"] = 4
            _set_street_pot(hand_state, "turn", _state_value(hand_state, "pot"))
            _rebuild_turn_node(hand_state, _state_value(hand_state, "pot"))
            return OptionResolution(note=f"You bet {bet_size:.2f}bb. (Rival response hidden)")
        hero_eq = hero_equity_vs_combo(hero_cards, board, villain_cards, combo_trials)
        villain_eq = 1.0 - hero_eq
        if villain_eq < threshold:
            hand_state["hand_over"] = True
            total_pot = _state_value(hand_state, "pot")
            net_gain = total_pot - bet_size
            note = f"Rival folds flop. Pot {total_pot:.2f}bb awarded (net +{net_gain:.2f}bb)."
            return OptionResolution(hand_ended=True, note=note)
        call_amount = min(bet_size, _state_value(hand_state, "villain_stack"))
        _apply_contribution(hand_state, "villain", call_amount)
        hand_state["board_index"] = 4
        _set_street_pot(hand_state, "turn", _state_value(hand_state, "pot"))
        _rebuild_turn_node(hand_state, _state_value(hand_state, "pot"))
        return OptionResolution(note=f"Rival calls. Pot {_state_value(hand_state, 'pot'):.2f}bb going to turn.")

    if action in {"jam", "allin", "all-in", "jam"}:
        risk = float(option.meta.get("risk", _state_value(hand_state, "hero_stack", node.effective_bb)))
        threshold = float(option.meta.get("villain_threshold", 0.5))
        combo_trials = int(option.meta.get("combo_trials", _combo_trials(60)))
        _apply_contribution(hand_state, "hero", risk)
        hand_state["hand_over"] = True
        if villain_cards is None:
            return OptionResolution(hand_ended=True, note=f"You jam for {risk:.2f}bb. Rival action hidden.")
        hero_eq = hero_equity_vs_combo(hero_cards, board, villain_cards, combo_trials)
        villain_eq = 1.0 - hero_eq
        villain_text = format_cards_spaced(list(villain_cards))
        if villain_eq < threshold:
            total_pot = _state_value(hand_state, "pot")
            net_gain = total_pot - risk
            note = f"Rival folds to your jam. Pot {total_pot:.2f}bb awarded (net +{net_gain:.2f}bb)."
            return OptionResolution(hand_ended=True, note=note, reveal_villain=True)
        call_amount = min(risk, _state_value(hand_state, "villain_stack"))
        _apply_contribution(hand_state, "villain", call_amount)
        equity_note = _fmt_pct(hero_eq, 1)
        return OptionResolution(
            hand_ended=True,
            note=f"Rival calls jam with {villain_text}. Your equity {equity_note}.",
            reveal_villain=True,
        )

    return OptionResolution(hand_ended=getattr(option, "ends_hand", False))


def _resolve_turn(node: Node, option: Option, hand_state: dict[str, Any]) -> OptionResolution:
    action = option.meta.get("action") if option.meta else None
    villain_bet = float(option.meta.get("villain_bet", node.context.get("bet", 0.0)))
    if villain_bet > 0:
        _apply_contribution(hand_state, "villain", villain_bet)
        _set_street_pot(hand_state, "turn", _state_value(hand_state, "pot"))
    pot_after_bet = _state_value(hand_state, "pot", node.pot_bb)
    villain_cards = _villain_cards(hand_state)
    hero_cards = node.hero_cards
    board = node.board

    if action == "fold":
        hand_state["hand_over"] = True
        note = f"You fold turn. Rival collects {pot_after_bet:.2f}bb."
        return OptionResolution(hand_ended=True, note=note)

    if action == "call":
        call_amount = min(villain_bet, _state_value(hand_state, "hero_stack"))
        _apply_contribution(hand_state, "hero", call_amount)
        hand_state["street"] = "river"
        hand_state["board_index"] = 5
        _set_street_pot(hand_state, "river", _state_value(hand_state, "pot"))
        _rebuild_river_node(hand_state, _state_value(hand_state, "pot"))
        return OptionResolution(note=f"You call {call_amount:.2f}bb. Pot {_state_value(hand_state, 'pot'):.2f}bb on river.")

    if action == "raise":
        raise_to = float(option.meta.get("raise_to", villain_bet * 2.5))
        threshold = float(option.meta.get("villain_threshold", 0.45))
        combo_trials = int(option.meta.get("combo_trials", _combo_trials(60)))
        hero_contrib = _state_value(hand_state, "hero_contrib")
        hero_add = max(0.0, raise_to - hero_contrib)
        _apply_contribution(hand_state, "hero", hero_add)
        if villain_cards is None:
            call_amount = max(0.0, raise_to - _state_value(hand_state, "villain_contrib"))
            _apply_contribution(hand_state, "villain", call_amount)
            hand_state["street"] = "river"
            hand_state["board_index"] = 5
            _set_street_pot(hand_state, "river", _state_value(hand_state, "pot"))
            _rebuild_river_node(hand_state, _state_value(hand_state, "pot"))
            return OptionResolution(note=f"You raise to {raise_to:.2f}bb. Pot now {_state_value(hand_state, 'pot'):.2f}bb.")
        hero_eq = hero_equity_vs_combo(hero_cards, board, villain_cards, combo_trials)
        villain_eq = 1.0 - hero_eq
        if villain_eq < threshold:
            hand_state["hand_over"] = True
            total_pot = _state_value(hand_state, "pot")
            net_gain = total_pot - hero_add
            note = f"Rival folds turn. Pot {total_pot:.2f}bb awarded (net +{net_gain:.2f}bb)."
            return OptionResolution(hand_ended=True, note=note)
        call_amount = max(0.0, raise_to - _state_value(hand_state, "villain_contrib"))
        _apply_contribution(hand_state, "villain", call_amount)
        hand_state["street"] = "river"
        hand_state["board_index"] = 5
        _set_street_pot(hand_state, "river", _state_value(hand_state, "pot"))
        _rebuild_river_node(hand_state, _state_value(hand_state, "pot"))
        return OptionResolution(note=f"Rival calls raise. Pot {_state_value(hand_state, 'pot'):.2f}bb.")

    return OptionResolution(hand_ended=getattr(option, "ends_hand", False))


def _resolve_river(node: Node, option: Option, hand_state: dict[str, Any]) -> OptionResolution:
    action = option.meta.get("action") if option.meta else None
    pot = _state_value(hand_state, "pot", node.pot_bb)
    hero_cards = node.hero_cards
    villain_cards = _villain_cards(hand_state)
    board = node.board

    if action == "check":
        hand_state["hand_over"] = True
        if villain_cards is None:
            return OptionResolution(hand_ended=True, note=f"Hand checks down. Pot {pot:.2f}bb.")
        outcome = _showdown_outcome(hero_cards, board, villain_cards)
        villain_text = format_cards_spaced(list(villain_cards))
        win_note = f"Showdown win vs {villain_text}. You take {pot:.2f}bb."
        lose_note = f"Showdown loss vs {villain_text}."
        chop_note = f"Showdown chop vs {villain_text}. Pot split."
        if outcome > 0.5:
            return OptionResolution(hand_ended=True, note=win_note, reveal_villain=True)
        if outcome < 0.5:
            return OptionResolution(hand_ended=True, note=lose_note, reveal_villain=True)
        return OptionResolution(hand_ended=True, note=chop_note, reveal_villain=True)

    if action == "bet":
        bet_size = float(option.meta.get("bet", 0.0))
        threshold = float(option.meta.get("villain_threshold", 0.5))
        combo_trials = int(option.meta.get("combo_trials", _combo_trials(50)))
        _apply_contribution(hand_state, "hero", bet_size)
        if villain_cards is None:
            hand_state["hand_over"] = True
            return OptionResolution(hand_ended=True, note=f"You bet {bet_size:.2f}bb. Rival action hidden.")
        hero_eq = hero_equity_vs_combo(hero_cards, board, villain_cards, combo_trials)
        villain_eq = 1.0 - hero_eq
        if villain_eq < threshold:
            hand_state["hand_over"] = True
            total_pot = _state_value(hand_state, "pot")
            net_gain = total_pot - bet_size
            note = f"Rival folds river. Pot {total_pot:.2f}bb awarded (net +{net_gain:.2f}bb)."
            return OptionResolution(hand_ended=True, note=note)
        call_amount = min(bet_size, _state_value(hand_state, "villain_stack"))
        _apply_contribution(hand_state, "villain", call_amount)
        hand_state["hand_over"] = True
        outcome = _showdown_outcome(hero_cards, board, villain_cards)
        villain_text = format_cards_spaced(list(villain_cards))
        total_pot = _state_value(hand_state, "pot")
        win_note = f"Rival calls with {villain_text}. You win {total_pot:.2f}bb."
        lose_note = f"Rival calls with {villain_text}. You lose."
        chop_note = f"Rival calls with {villain_text}. Pot split."
        if outcome > 0.5:
            return OptionResolution(hand_ended=True, note=win_note, reveal_villain=True)
        if outcome < 0.5:
            return OptionResolution(hand_ended=True, note=lose_note, reveal_villain=True)
        return OptionResolution(hand_ended=True, note=chop_note, reveal_villain=True)

    if action in {"jam", "allin", "all-in", "jam"}:
        risk = float(option.meta.get("risk", _state_value(hand_state, "hero_stack", node.effective_bb)))
        threshold = float(option.meta.get("villain_threshold", 0.5))
        combo_trials = int(option.meta.get("combo_trials", _combo_trials(50)))
        _apply_contribution(hand_state, "hero", risk)
        hand_state["hand_over"] = True
        if villain_cards is None:
            return OptionResolution(hand_ended=True, note=f"You jam river for {risk:.2f}bb. Rival action hidden.")
        hero_eq = hero_equity_vs_combo(hero_cards, board, villain_cards, combo_trials)
        villain_eq = 1.0 - hero_eq
        villain_text = format_cards_spaced(list(villain_cards))
        if villain_eq < threshold:
            total_pot = _state_value(hand_state, "pot")
            net_gain = total_pot - risk
            note = f"Rival folds river jam. Pot {total_pot:.2f}bb awarded (net +{net_gain:.2f}bb)."
            return OptionResolution(hand_ended=True, note=note, reveal_villain=True)
        call_amount = min(risk, _state_value(hand_state, "villain_stack"))
        _apply_contribution(hand_state, "villain", call_amount)
        total_pot = _state_value(hand_state, "pot")
        outcome = _showdown_outcome(hero_cards, board, villain_cards)
        if outcome > 0.5:
            return OptionResolution(
                hand_ended=True,
                note=f"Rival calls jam with {villain_text}. You win {total_pot:.2f}bb.",
                reveal_villain=True,
            )
        if outcome < 0.5:
            return OptionResolution(
                hand_ended=True,
                note=f"Rival calls jam with {villain_text}. You lose.",
                reveal_villain=True,
            )
        return OptionResolution(
            hand_ended=True,
            note=f"Rival calls jam with {villain_text}. Pot split.",
            reveal_villain=True,
        )

    return OptionResolution(hand_ended=getattr(option, "ends_hand", False))
