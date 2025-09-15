from __future__ import annotations

import random

from ..core.models import Option
from .equity import estimate_equity
from .generator import Node


def preflop_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    # Actions: Fold, Call, 3-bet sizes
    sz = float(node.context["open_size"])  # villain opened to sz
    pot = node.pot_bb
    call_amt = sz - 1.0  # BB has posted 1bb

    eq = estimate_equity(node.hero_cards, [], None, rng, trials=mc_trials)
    # Fold EV is -1bb (BB already posted)
    ev_fold = -1.0
    # Call EV approx: equity * (pot + call) - call
    ev_call = eq * (pot + call_amt) - call_amt

    # 3-bet sizes and a simple fold equity model
    sizes = [8.0, 9.0, 10.0]
    opts = [
        Option("Fold", ev_fold, "Folding forfeits the big blind."),
        Option("Call", ev_call, f"Equity {eq:.2f} vs open; priced at {call_amt:.2f}bb."),
    ]
    for b in sizes:
        risk = b - 1.0  # BB adds from posted 1bb to 3-bet size
        fe = max(0.0, min(0.8, 0.25 + 0.10 * (b - 8)))  # crude: bigger size → more FE
        win_when_fold = pot
        # When called: assume pot ~ b + sz (ignoring 4-bets) and go to flop
        pot_when_called = b + sz
        eq3 = eq * 0.98  # slight realization discount OOP even if aggressor
        ev_bet = fe * win_when_fold + (1 - fe) * (eq3 * pot_when_called - risk)
        opts.append(Option(f"3-bet to {b:.0f}bb", ev_bet, f"FE≈{fe:.2f}, equity≈{eq3:.2f} vs continue."))
    return opts


def flop_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    # Facing check; options: check, bet 33/50/75% pot
    pot = node.pot_bb
    board = node.board
    eq = estimate_equity(node.hero_cards, board, None, rng, trials=mc_trials)
    opts: list[Option] = [Option("Check", eq * pot - 0.0, f"Realize equity {eq:.2f} to turn.")]
    for pct in (0.33, 0.5, 0.75):
        bet = round(pot * pct, 2)
        fe = min(0.65, max(0.05, 0.15 + 0.4 * (eq - 0.5) + 0.2 * (pct - 0.33)))
        win_when_fold = pot
        pot_when_called = pot + bet
        eq_post = max(0.0, min(1.0, eq + 0.02))
        ev = fe * win_when_fold + (1 - fe) * (eq_post * pot_when_called - bet)
        why = f"Bet {int(pct * 100)}%: FE≈{fe:.2f}, equity≈{eq_post:.2f} when called."
        opts.append(Option(f"Bet {int(pct * 100)}% pot", ev, why))
    return opts


def turn_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    # Facing a bet; options: fold, call, raise to 2.5x
    pot = node.pot_bb
    bet = float(node.context["bet"])
    board = node.board
    eq = estimate_equity(node.hero_cards, board, None, rng, trials=mc_trials)
    # Fold: 0
    ev_fold = 0.0
    # Call: equity on river with pot odds
    pot_after_call = pot + bet
    call_amt = bet
    ev_call = eq * pot_after_call - call_amt
    # Raise: to 2.5x
    raise_to = round(bet * 2.5, 2)
    risk = raise_to
    fe = min(0.7, max(0.1, 0.10 + 0.5 * (raise_to / max(1e-6, pot))))
    pot_when_called = pot + raise_to
    eq_post = max(0.0, min(1.0, eq + 0.03))
    ev_raise = fe * pot + (1 - fe) * (eq_post * pot_when_called - risk)
    return [
        Option("Fold", ev_fold, "Avoid marginal spots vs bet."),
        Option("Call", ev_call, f"Pot odds; equity≈{eq:.2f} to river."),
        Option(f"Raise to {raise_to:.2f}bb", ev_raise, f"FE≈{fe:.2f}, equity≈{eq_post:.2f} when called."),
    ]


def river_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    # In position, OOP checks; options: check, 50% pot, 100% pot
    pot = node.pot_bb
    board = node.board
    eq = estimate_equity(node.hero_cards, board, None, rng, trials=mc_trials)
    opts: list[Option] = [Option("Check", eq * pot, f"Showdown value with equity {eq:.2f}.")]
    for pct in (0.5, 1.0):
        bet = round(pot * pct, 2)
        # River FE: driven by board texture and sizing; approximate via equity and size
        fe = min(0.75, max(0.05, 0.2 + 0.5 * (pct - 0.5) + 0.3 * (eq - 0.5)))
        win_when_fold = pot
        # When called: either win full pot+bet or lose bet
        # EV approx: FE*pot + (1-FE)*(eq*(pot+bet) - (1-eq)*bet)
        ev = fe * win_when_fold + (1 - fe) * (eq * (pot + bet) - (1 - eq) * bet)
        why = f"Bet {int(pct * 100)}%: FE≈{fe:.2f}, equity≈{eq:.2f} if called."
        opts.append(Option(f"Bet {int(pct * 100)}% pot", ev, why))
    return opts


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
