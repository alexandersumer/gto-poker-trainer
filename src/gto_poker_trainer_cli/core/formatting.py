from __future__ import annotations

from typing import Any

from ..dynamic.generator import Node
from .models import Option


def _safe_pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def _fmt_pct(value: float) -> str:
    if value >= 100 or value == 0:
        return f"{value:.0f}%"
    if value < 1:
        return f"{value:.2f}%"
    return f"{value:.1f}%" if (value * 10) % 10 else f"{value:.0f}%"


def format_option_label(node: Node, option: Option) -> str:
    meta: dict[str, Any] = option.meta or {}
    action = str(meta.get("action") or "").lower()

    key = option.key.strip()
    if not action:
        return key

    pot = float(node.pot_bb)

    if action == "bet":
        bet_size = float(meta.get("bet", 0.0))
        pct = _safe_pct(bet_size, pot)
        return f"Bet {_fmt_pct(pct)}"

    if action == "raise":
        raise_to = float(meta.get("raise_to", 0.0))
        pot_before = float(meta.get("pot_before", pot))
        pct = _safe_pct(raise_to, pot_before)
        return f"Raise {_fmt_pct(pct)}"

    if action == "3bet":
        raise_to = float(meta.get("raise_to", 0.0))
        pot_before = float(meta.get("pot_before", pot))
        pct = _safe_pct(raise_to, pot_before)
        return f"3-bet {_fmt_pct(pct)}"

    if action == "call":
        amount = float(meta.get("call_cost", meta.get("villain_bet", 0.0)))
        pct = _safe_pct(amount, pot)
        if amount <= 0:
            return "Call"
        return f"Call {_fmt_pct(pct)}"

    if action == "check":
        return "Check"

    if action == "fold":
        return "Fold"

    return key
