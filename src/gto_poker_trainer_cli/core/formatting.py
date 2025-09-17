from __future__ import annotations

import re
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
    if node.street == "preflop":
        return _format_preflop_label(node, key, action, meta)
    if not action:
        return _fallback_percent_label(key, node)

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

    if action in {"jam", "allin", "all-in"}:
        return "All-in"

    if action == "call":
        amount = float(meta.get("call_cost", meta.get("villain_bet", 0.0)))
        if amount > 0:
            return f"Call {amount:.2f}bb"
        return "Call"

    if action == "check":
        return "Check"

    if action == "fold":
        return "Fold"

    return _fallback_percent_label(key, node)


_BB_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*bb", re.IGNORECASE)


def _fallback_percent_label(key: str, node: Node) -> str:
    if node.street == "preflop":
        if "all-in" in key.lower() or "jam" in key.lower():
            return "All-in"
        return key
    key_lower = key.lower()
    if "all-in" in key_lower or "jam" in key_lower:
        return "All-in"
    if "%" in key_lower or "all-in" in key_lower:
        return key

    match = _BB_PATTERN.search(key_lower)
    if not match:
        return key

    amount = float(match.group(1))
    pot = float(node.pot_bb)
    pct = _safe_pct(amount, pot)

    if "3-bet" in key_lower:
        return f"3-bet {_fmt_pct(pct)}"
    if "raise" in key_lower:
        return f"Raise {_fmt_pct(pct)}"
    if "bet" in key_lower:
        return f"Bet {_fmt_pct(pct)}"
    if "call" in key_lower:
        return f"Call {_fmt_pct(pct)}"
    return key


def _format_preflop_label(node: Node, key: str, action: str, meta: dict[str, Any]) -> str:
    _ = node
    action = action or ""
    action = action.lower()
    if action in {"jam", "allin", "all-in"}:
        return "All-in"
    if action == "fold" or key.lower().startswith("fold"):
        return "Fold"
    if action == "check":
        return "Check"
    if action == "call":
        amount = float(meta.get("call_cost", 0.0))
        return f"Call {amount:.2f}bb" if amount > 0 else "Call"
    if action == "3bet":
        raise_to = float(meta.get("raise_to", 0.0))
        return f"3-bet to {raise_to:.2f}bb" if raise_to > 0 else key
    if action == "raise":
        raise_to = float(meta.get("raise_to", 0.0))
        return f"Raise to {raise_to:.2f}bb" if raise_to > 0 else key
    if "all-in" in key.lower() or "jam" in key.lower():
        return "All-in"
    return key
