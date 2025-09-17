from __future__ import annotations

import random
from collections.abc import Iterable

from ..core.models import Option
from .equity import hero_equity_vs_combo, hero_equity_vs_range as _hero_equity_vs_range
from .generator import Node
from .range_model import tighten_range, villain_sb_open_range


def _fmt_pct(x: float, decimals: int = 0) -> str:
    return f"{100.0 * x:.{decimals}f}%"


def _blocked_cards(hero: Iterable[int], board: Iterable[int]) -> set[int]:
    return set(hero) | set(board)


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
    return max(30, int(mc_trials * 0.4))


def _sample_cap_preflop(mc_trials: int) -> int:
    return max(35, min(90, int(mc_trials * 0.4)))


def _sample_cap_postflop(mc_trials: int) -> int:
    return max(25, min(70, int(mc_trials * 0.3)))


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


def preflop_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    del rng
    hero = node.hero_cards
    open_size = float(node.context["open_size"])
    pot_after_open = node.pot_bb
    call_cost = open_size - 1.0
    blocked = _blocked_cards(hero, [])
    open_range = villain_sb_open_range(open_size, blocked)
    sampled_range = _sample_range(open_range, _sample_cap_preflop(mc_trials)) or open_range
    combo_trials = _combo_trials(mc_trials)
    equities = {combo: hero_equity_vs_combo(hero, [], combo, combo_trials) for combo in sampled_range}
    avg_range_eq = sum(equities.values()) / len(equities) if equities else 0.0

    options: list[Option] = [
        Option("Fold", 0.0, "Fold now and lose nothing extra.", ends_hand=True),
    ]

    final_pot_call = pot_after_open + call_cost
    be_call_eq = call_cost / final_pot_call if final_pot_call > 0 else 1.0
    options.append(
        Option(
            "Call",
            avg_range_eq * final_pot_call - call_cost,
            (
                f"Pot odds: call {call_cost:.2f} bb to win {final_pot_call:.2f} bb. "
                f"Need ≈{_fmt_pct(be_call_eq, 1)} equity, hand has {_fmt_pct(avg_range_eq, 1)}."
            ),
        )
    )

    for raise_to in (8.0, 9.0, 10.0):
        risk = raise_to - 1.0
        pot_after_raise = pot_after_open + risk
        villain_call_cost = raise_to - open_size
        final_pot = pot_after_raise + villain_call_cost
        be_threshold = villain_call_cost / final_pot if final_pot > 0 else 1.0
        fe, avg_eq_when_called, continue_ratio = _fold_continue_stats(equities.values(), be_threshold)
        ev_called = avg_eq_when_called * final_pot - risk if continue_ratio else -risk
        ev = fe * pot_after_open + (1 - fe) * ev_called
        why = (
            f"Villain folds {_fmt_pct(fe)} needing eq {_fmt_pct(be_threshold, 1)}. "
            f"When called (≈{_fmt_pct(continue_ratio)}) you have {_fmt_pct(avg_eq_when_called, 1)} equity "
            f"→ EV {ev_called:.2f} bb."
        )
        options.append(Option(f"3-bet to {raise_to:.0f}bb", ev, why))

    return options


def flop_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    del rng
    hero = node.hero_cards
    board = node.board
    pot = node.pot_bb
    open_size = _default_open_size(node)
    blocked = _blocked_cards(hero, board)
    open_range = villain_sb_open_range(open_size, blocked)
    sampled_range = _sample_range(open_range, _sample_cap_postflop(mc_trials)) or open_range
    combo_trials = _combo_trials(mc_trials)
    equities = {combo: hero_equity_vs_combo(hero, board, combo, combo_trials) for combo in sampled_range}
    avg_eq = sum(equities.values()) / len(equities) if equities else 0.0

    options: list[Option] = [
        Option("Check", avg_eq * pot, f"Realize equity {_fmt_pct(avg_eq, 1)} in-position."),
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
            f"{int(pct * 100)}% pot: villain folds {_fmt_pct(fe)} (needs eq {_fmt_pct(be_threshold, 1)}). "
            f"Continuing range (~{_fmt_pct(continue_ratio)}) gives you {_fmt_pct(eq_call, 1)} equity "
            f"→ EV {ev_called:.2f} bb."
        )
        why += f" Additional sizing detail: {bet:.2f} bb (equals {int(pct * 100)}% pot)."
        options.append(Option(f"Bet {int(pct * 100)}% pot", ev, why))

    return options


def turn_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    del rng
    hero = node.hero_cards
    board = node.board
    pot_start = node.pot_bb
    villain_bet = float(node.context["bet"])
    pot_before_action = pot_start + villain_bet
    open_size = _default_open_size(node)
    blocked = _blocked_cards(hero, board)
    base_range = villain_sb_open_range(open_size, blocked)
    # Villain betting range is tightened to the stronger half of their holdings.
    bet_range = tighten_range(base_range, 0.55)
    sampled_range = _sample_range(bet_range, _sample_cap_postflop(mc_trials)) or bet_range
    combo_trials = _combo_trials(mc_trials)
    equities = {combo: hero_equity_vs_combo(hero, board, combo, combo_trials) for combo in sampled_range}
    avg_eq = sum(equities.values()) / len(equities) if equities else 0.0

    options = [Option("Fold", 0.0, "Release the hand and wait for better spot.", ends_hand=True)]

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
        f"Villain folds {_fmt_pct(fe)} (needs eq {_fmt_pct(be_threshold, 1)}). "
        f"break-even FE {_fmt_pct(fe_break_even)}. "
        f"Continuing (~{_fmt_pct(continue_ratio)}) you have {_fmt_pct(eq_call, 1)} equity → "
        f"EV {ev_called:.2f} bb."
    )
    options.append(Option(f"Raise to {raise_to:.2f} bb", ev, why_raise))

    return options


def river_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    del rng
    hero = node.hero_cards
    board = node.board
    pot = node.pot_bb
    open_size = _default_open_size(node)
    blocked = _blocked_cards(hero, board)
    base_range = villain_sb_open_range(open_size, blocked)
    # After checking river, assume villain keeps medium-strength holdings.
    check_range = tighten_range(base_range, 0.65)
    sampled_range = _sample_range(check_range, _sample_cap_postflop(mc_trials)) or check_range
    combo_trials = _combo_trials(mc_trials)
    equities = {combo: hero_equity_vs_combo(hero, board, combo, combo_trials) for combo in sampled_range}
    avg_eq = sum(equities.values()) / len(equities) if equities else 0.0

    options: list[Option] = [Option("Check", avg_eq * pot, f"Showdown equity {_fmt_pct(avg_eq, 1)} vs check range.")]

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
            f"Bet {int(pct * 100)}%: villain folds {_fmt_pct(fe)} (needs eq {_fmt_pct(be_threshold, 1)}). "
            f"Calls (~{_fmt_pct(continue_ratio)}) give you {_fmt_pct(eq_call, 1)} equity → EV {ev_called:.2f} bb."
        )
        why += f" Additional sizing detail: {bet:.2f} bb (equals {int(pct * 100)}% pot)."
        options.append(Option(f"Bet {int(pct * 100)}% pot", ev, why))

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
