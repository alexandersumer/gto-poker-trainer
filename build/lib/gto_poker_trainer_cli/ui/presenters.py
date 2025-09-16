from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.interfaces import Presenter
from ..core.models import Option
from ..dynamic.cards import canonical_hand_abbrev, format_card_ascii
from ..dynamic.generator import Node


class RichPresenter(Presenter):
    def __init__(self, *, no_color: bool = False):
        # Default: color ON (forced), unless explicitly disabled via --no-color.
        if no_color:
            self.console = Console(force_terminal=False, color_system=None)
        else:
            self.console = Console(force_terminal=True, color_system="auto")
        self.quit_requested = False
        self._last_meta: str | None = None

    def start_session(self, total_hands: int) -> None:
        pass

    def start_hand(self, hand_index: int, total_hands: int) -> None:
        self.console.print(
            Panel.fit(
                f"Hand {hand_index}/{total_hands}",
                title="GTO Poker Trainer CLI",
                style="bold cyan",
            )
        )

    def show_node(self, node: Node, options: list[str]) -> None:
        # Headline (strip duplicated Board prefix from description if present)
        desc = node.description
        if node.board and desc.startswith("Board "):
            # keep text after the first period+space, e.g. "Board KQJ. SB checks." -> "SB checks."
            dot = desc.find(". ")
            desc = desc[dot + 2 :] if dot != -1 else ""

        headline = f"[bold magenta]{node.street.upper()}[/]"
        if desc:
            headline += f"  [dim]- {desc}[/]"
        self.console.print(headline)
        # Always show hero's hole cards on every street for continuity
        # Render hole cards with suit-aware colors (sorted by rank for readability)
        hand_sorted = self._sort_cards_by_rank(node.hero_cards)
        hand_str_colored = self._format_cards_colored(hand_sorted)
        hand_abbrev = canonical_hand_abbrev(node.hero_cards)
        self.console.print(f"Your hand: {hand_str_colored} [dim]({hand_abbrev})[/]")
        # Community board (with suit-aware colors) for quick scanning
        if node.board:
            board_str = self._format_cards_colored(node.board)
            self.console.print(f"Board: {board_str}")

        # Pot/SPR and sizing context for clarity
        P = float(node.pot_bb)
        spr = (node.effective_bb / P) if P > 0 else float("inf")
        meta = f"Pot: {P:.2f}bb | SPR: {spr:.1f}"
        # Bet details when facing a bet
        bet = node.context.get("bet")
        if isinstance(bet, (int, float)):
            pct = 100.0 * float(bet) / max(1e-9, P)
            meta += f" | OOP bet: {float(bet):.2f}bb ({pct:.0f}% pot)"
        self._last_meta = meta
        self.console.print(f"[dim]{meta}[/]")
        self.console.print(f"[dim]Controls: 1–{len(options)} to act • h=help • ?=pot • q=quit[/]")
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("#", justify="right")
        table.add_column("Action")
        table.add_column("Hint", style="dim")
        for i, k in enumerate(options, 1):
            table.add_row(str(i), k, "EV hidden")
        self.console.print(table)

    def prompt_choice(self, n: int) -> int:
        while True:
            raw = input(f"Your choice (1-{n}), or 'q' to quit: ").strip().lower()
            if raw == "q":
                self.quit_requested = True
                return -1
            if raw in {"h", "help"}:
                self._print_help(n)
                continue
            if raw in {"?"}:
                self.console.print(f"[dim]{self._last_meta or 'No pot info yet.'}[/]")
                continue
            if raw.isdigit():
                v = int(raw)
                if 1 <= v <= n:
                    return v - 1
            self.console.print(f"[red]Invalid input[/]. Please enter a number 1-{n} or 'q'.")

    def step_feedback(self, _node: Node, chosen: Option, best: Option) -> None:
        correct = chosen.key == best.key
        ev_loss = best.ev - chosen.ev
        self.console.print("\n[bold]Step feedback[/]")
        if correct:
            self.console.print(f"✓ [green]Best choice[/]: {best.key} (EV {best.ev:.2f} bb)")
        else:
            self.console.print(f"✗ [yellow]Better was[/]: {best.key} (EV {best.ev:.2f} bb)")
            self.console.print(f"You chose: {chosen.key} (EV {chosen.ev:.2f} bb) → EV lost: [red]{ev_loss:.2f} bb[/]")
        self.console.print(f"Why (your action): {chosen.why}")
        if not correct:
            self.console.print(f"Why (best action):  {best.why}")
        if getattr(chosen, "ends_hand", False):
            self.console.print("[dim]Hand ends on this action.[/]")
        self.console.print("[dim]—[/]\n")

    def summary(self, records: list[dict]) -> None:
        if not records:
            self.console.print("No hands answered.")
            return
        total_ev_best = sum(r["best_ev"] for r in records)
        total_ev_chosen = sum(r["chosen_ev"] for r in records)
        total_ev_lost = total_ev_best - total_ev_chosen
        avg_ev_lost = total_ev_lost / len(records)
        room = sum(max(1e-9, r["best_ev"] - min(0.0, r["chosen_ev"])) for r in records)
        score_pct = 100.0 * max(0.0, 1.0 - (total_ev_lost / room)) if room > 1e-9 else 100.0
        hits = sum(1 for r in records if r["chosen_key"] == r["best_key"])

        summary = Table(title="Session Summary", show_header=False)
        summary.add_row("Hands answered:", str(len(records)))
        summary.add_row("Best choices hit:", f"{hits} ({100.0 * hits / len(records):.0f}%)")
        summary.add_row("Total EV (chosen):", f"{total_ev_chosen:.2f} bb")
        summary.add_row("Total EV (best possible):", f"{total_ev_best:.2f} bb")
        summary.add_row("Total EV lost:", f"{total_ev_lost:.2f} bb")
        summary.add_row("Average EV lost/decision:", f"{avg_ev_lost:.2f} bb")
        summary.add_row("Score (0–100):", f"{score_pct:.0f}")
        self.console.print("\n")
        self.console.print(summary)

        # Leaks
        leaks = sorted(records, key=lambda r: r["ev_loss"], reverse=True)
        leak_table = Table(title="Top EV leaks", show_header=True, header_style="bold blue")
        leak_table.add_column("#")
        leak_table.add_column("Street")
        leak_table.add_column("Chosen")
        leak_table.add_column("Best")
        leak_table.add_column("EV lost (bb)", justify="right")
        for i, r in enumerate(leaks[:3], 1):
            leak_table.add_row(str(i), r["street"], r["chosen_key"], r["best_key"], f"{r['ev_loss']:.2f}")
        self.console.print(leak_table)

    # --- helpers ---
    def _print_help(self, n: int) -> None:
        table = Table(show_header=False)
        table.add_row("Choose action:", f"1–{n}")
        table.add_row("Show pot math:", "?")
        table.add_row("Help:", "h")
        table.add_row("Quit:", "q")
        self.console.print(Panel.fit(table, title="Controls", style="dim"))

    # --- card rendering helpers ---
    def _sort_cards_by_rank(self, cards: list[int]) -> list[int]:
        # Rank is encoded as integer division by 4; higher rank first
        return sorted(cards, key=lambda ci: ci // 4, reverse=True)

    def _format_cards_colored(self, cards: list[int]) -> str:
        # Suits: 0=spades, 1=hearts, 2=diamonds, 3=clubs
        colors = {
            0: "white",  # spades
            1: "bright_red",  # hearts
            2: "bright_cyan",  # diamonds
            3: "green",  # clubs
        }
        # Keep original order for streets (flop/turn/river)
        parts: list[str] = []
        for c in cards:
            suit = c % 4
            color = colors.get(suit, "white")
            txt = format_card_ascii(c, upper=True)
            parts.append(f"[bold {color}]{txt}[/]")
        return " ".join(parts)
