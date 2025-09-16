from __future__ import annotations

import csv
from contextlib import suppress
from dataclasses import dataclass

from ..core.interfaces import OptionProvider
from ..core.models import Option
from ..dynamic.cards import canonical_hand_abbrev
from ..dynamic.generator import Node


@dataclass
class SolverEntry:
    key: str
    options: list[Option]


class CSVStrategyOracle(OptionProvider):
    """Preflop-focused CSV strategy oracle.

    CSV columns (long format):
      street (preflop/flop/turn/river), hero_position, context_action, context_size,
      hero_hand (e.g., A5s, KQo, 77), option_key, option_ev, option_why, gto_freq
    Only 'preflop' is used for now; others may be ignored/fallback.
    """

    def __init__(self, csv_path: str):
        self.by_key: dict[str, list[Option]] = {}
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {
                "street",
                "hero_position",
                "context_action",
                "context_size",
                "hero_hand",
                "option_key",
                "option_ev",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing required columns: {sorted(missing)}")
            for row in reader:
                if (row.get("street") or "").strip().lower() != "preflop":
                    continue
                k = self._key(
                    street=row["street"],
                    hero_position=row["hero_position"],
                    context_action=row["context_action"],
                    context_size=row["context_size"],
                    hero_hand=row["hero_hand"],
                )
                key = (row["option_key"] or "").strip()
                opt = Option(
                    key=key,
                    ev=float(row["option_ev"]),
                    why=(row.get("option_why") or "").strip(),
                    ends_hand=key.lower().startswith("fold"),
                )
                freq = row.get("gto_freq")
                if freq not in (None, ""):
                    with suppress(Exception):
                        opt.gto_freq = float(freq)  # type: ignore[attr-defined]
                self.by_key.setdefault(k, []).append(opt)

    def _key(
        self,
        *,
        street: str,
        hero_position: str,
        context_action: str,
        context_size: str,
        hero_hand: str,
    ) -> str:
        s = street.strip().lower()
        p = hero_position.strip().upper()
        a = (context_action or "").strip().lower()
        z = (context_size or "").strip().lower()
        h = hero_hand.strip().upper()
        return f"{s}|{p}|{a}|{z}|{h}"

    def options(self, node: Node, _rng, _mc_trials: int) -> list[Option]:  # type: ignore[override]
        if node.street != "preflop":
            raise LookupError("CSVStrategyOracle only supports preflop in this version")
        hero_pos = node.actor  # BB in our generator
        action = "open"
        size = f"{float(node.context['open_size']):.1f}bb"
        hand = canonical_hand_abbrev(node.hero_cards).upper()
        k = self._key(
            street="preflop",
            hero_position=hero_pos,
            context_action=action,
            context_size=size,
            hero_hand=hand,
        )
        if k not in self.by_key:
            raise LookupError("No solver entry for key")
        return list(self.by_key[k])


class CompositeOptionProvider(OptionProvider):
    def __init__(self, primary: OptionProvider, fallback: OptionProvider):
        self.primary = primary
        self.fallback = fallback

    def options(self, node: Node, rng, mc_trials: int) -> list[Option]:  # type: ignore[override]
        try:
            return self.primary.options(node, rng, mc_trials)
        except Exception:
            return self.fallback.options(node, rng, mc_trials)
