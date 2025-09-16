from __future__ import annotations

import random

from ..core.models import Option
from .equity import estimate_equity
from .generator import Node


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.0f}%"


def preflop_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    # Actions: Fold, Call, 3-bet sizes (BB facing SB open)
    sz = float(node.context["open_size"])  # villain opened to sz
    P = node.pot_bb  # pot after open (e.g., 3.0 when SB opens to 2.0)
    call_amt = sz - 1.0  # BB has posted 1bb already

    # Equity vs random (heuristic). We do not assume a solver range here.
    eq = estimate_equity(node.hero_cards, [], None, rng, trials=mc_trials)

    # Baseline: EV measured from this node; folding EV = 0.0
    ev_fold = 0.0

    # Call EV (approx): eq * (P + call) - call
    ev_call = eq * (P + call_amt) - call_amt
    be_call_eq = call_amt / (P + call_amt) if (P + call_amt) > 0 else 1.0

    # 3-bet sizes with conservative FE model, and clear pot math
    sizes = [8.0, 9.0, 10.0]
    opts: list[Option] = [
        Option("Fold", ev_fold, "Give up preflop (baseline EV = 0.0)."),
        Option(
            "Call",
            ev_call,
            f"Pot odds: call {call_amt:.2f} to win {P:.2f} (BE eq≈{be_call_eq:.2f}).",
        ),
    ]
    for total_to in sizes:
        # Risk is total contribution on this node (raise 'to' size includes the call)
        risk = total_to - 0.0  # relative to folding baseline
        # Win when folds: we win P (pot after open)
        win_when_fold = P
        # When called: assume heads-up, final preflop pot ≈ total_to + sz
        pot_when_called = total_to + sz
        # Slight realization discount for 3-bet pots without a range model
        eq3 = max(0.0, min(1.0, eq * 0.96))
        # Conservative FE estimate increases mildly with size
        fe = max(0.05, min(0.55, 0.20 + 0.05 * (total_to - 8)))
        ev_bet = fe * win_when_fold + (1 - fe) * (eq3 * pot_when_called - risk)
        why = f"Wins {P:.2f} when folds; if called plays pot≈{pot_when_called:.1f}. Est. FE≈{_fmt_pct(fe)}."
        opts.append(Option(f"3-bet to {total_to:.0f}bb", ev_bet, why))
    return opts


def flop_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    # Facing check; options: check, bet 33/50/75% pot
    P = node.pot_bb
    board = node.board
    eq = estimate_equity(node.hero_cards, board, None, rng, trials=mc_trials)
    opts: list[Option] = [
        Option(
            "Check",
            eq * P,
            "Low equity or range disadvantage → check back often; realize equity.",
        )
    ]
    for pct in (0.33, 0.5, 0.75):
        bet = round(P * pct, 2)
        # Simple FE model: increases with size and with holding equity a touch
        fe = min(0.60, max(0.05, 0.10 + 0.25 * (pct / 0.75) + 0.15 * max(0.0, eq - 0.4)))
        win_when_fold = P
        pot_when_called = P + bet
        eq_post = max(0.0, min(1.0, eq + 0.02))
        ev = fe * win_when_fold + (1 - fe) * (eq_post * pot_when_called - bet)
        why = f"{int(pct * 100)}% pot: wins {P:.2f} when folds; est. FE≈{_fmt_pct(fe)}."
        opts.append(Option(f"Bet {int(pct * 100)}% pot", ev, why))
    return opts


def turn_options(node: Node, rng: random.Random, mc_trials: int) -> list[Option]:
    # Facing a bet; options: fold, call, raise-to 2.5x (all EVs from node baseline; fold=0)
    P = node.pot_bb  # pot at start of turn before facing bet
    B = float(node.context["bet"])  # villain bet amount
    board = node.board
    eq = estimate_equity(node.hero_cards, board, None, rng, trials=mc_trials)

    # Fold
    ev_fold = 0.0

    # Call: final pot if called is P + 2B; cost is B
    ev_call = eq * (P + 2 * B) - B
    be_call_eq = B / (P + 2 * B) if (P + 2 * B) > 0 else 1.0

    # Raise-to 2.5x the bet (total the IP puts in on turn)
    R = round(B * 2.5, 2)
    # Win when folds: villain forfeits the bet; pot award is P + B
    win_when_fold = P + B
    # When called: final pot P + 2R; our cost is R (relative to fold baseline)
    pot_when_called = P + 2 * R
    # Modest equity bump when called is unrealistic here; keep conservative
    eq_post = max(0.0, min(1.0, eq + 0.01))
    # Conservative FE estimate: depends on size relative to (P+B) and holding equity
    fe_est = max(0.05, min(0.55, 0.10 + 0.15 * (R / max(1e-9, P + B)) + 0.20 * max(0.0, eq - 0.35)))
    ev_raise = fe_est * win_when_fold + (1 - fe_est) * (eq_post * pot_when_called - R)
    # Break-even FE for raise ignoring equity
    fe_break_even = R / (R + (P + B)) if (R + (P + B)) > 0 else 1.0

    return [
        Option("Fold", ev_fold, "Weak hand/no blockers → fold vs turn stab."),
        Option(
            "Call",
            ev_call,
            f"Pot odds: call {B:.2f} to win {(P + 2 * B):.2f} (BE eq≈{be_call_eq:.2f}).",
        ),
        Option(
            f"Raise to {R:.2f}bb",
            ev_raise,
            f"Needs FE≥{_fmt_pct(fe_break_even)} to break even; est. FE≈{_fmt_pct(fe_est)}.",
        ),
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
