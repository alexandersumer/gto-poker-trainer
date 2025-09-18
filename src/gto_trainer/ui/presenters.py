from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.interfaces import Presenter
from ..core.models import Option
from ..core.scoring import summarize_records
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
        self._hand_index: int = 0
        self._hand_total: int = 0

    def start_session(self, total_hands: int) -> None:
        self._hand_total = total_hands
        guide = (
            "[bold]Welcome![/] This trainer deals one decision at a time.\n"
            "- Read the situation banner for board, pot, and action context.\n"
            "- Type the number next to your chosen play.\n"
            "- After each choice you'll see why it wins or loses EV.\n\n"
            "[bold]Controls[/]: numbers = act • h = help • ? = pot math • q = quit"
        )
        panel = Panel(guide, title="Session Guide", border_style="green")
        self.console.print(panel)
        self.console.print()

    def start_hand(self, hand_index: int, total_hands: int) -> None:
        self._hand_index = hand_index
        self._hand_total = total_hands
        remaining = total_hands - hand_index
        hand_header = f"Hand {hand_index}/{total_hands}\nRemaining: {remaining}"
        self.console.print(
            Panel(
                hand_header,
                title="GTO Trainer",
                border_style="bold cyan",
                expand=False,
            )
        )
        self.console.print()

    def show_node(self, node: Node, options: list[str]) -> None:
        # Headline (strip duplicated Board prefix from description if present)
        desc = node.description
        if node.board and desc.startswith("Board "):
            # keep text after the first period+space, e.g. "Board KQJ. SB checks." -> "SB checks."
            dot = desc.find(". ")
            desc = desc[dot + 2 :] if dot != -1 else ""

        headline = f"{node.street.upper()}"
        if desc:
            headline += f" — {desc}"
        self.console.rule(headline)

        # Always show hero's hole cards on every street for continuity
        # Render hole cards with suit-aware colors (sorted by rank for readability)
        hand_sorted = self._sort_cards_by_rank(node.hero_cards)
        hand_str_colored = self._format_cards_colored(hand_sorted)
        hand_abbrev = canonical_hand_abbrev(node.hero_cards)

        info = Table.grid(padding=(0, 1))
        info.add_column(style="bold cyan", justify="right")
        info.add_column(justify="left")
        info.add_row("Hand", f"{self._hand_index}/{self._hand_total}")
        info.add_row("Your hand", f"{hand_str_colored} [dim]({hand_abbrev})[/]")
        if node.board:
            board_str = self._format_cards_colored(node.board)
            info.add_row("Board", board_str)

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
        info.add_row("Pot / SPR", f"{P:.2f}bb (SPR {spr:.1f})")
        if isinstance(bet, (int, float)):
            info.add_row("Facing", f"Bet {float(bet):.2f}bb ({pct:.0f}% pot)")
        if desc:
            info.add_row("Situation", desc)
        self.console.print(Panel(info, title="Table Status", border_style="magenta", expand=False))

        self.console.print("Choose an action:")
        table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE_HEAVY)
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("Action", style="bold")
        table.add_column("What it means", overflow="fold")
        for i, k in enumerate(options, 1):
            hint = self._hint_for_action(k)
            table.add_row(str(i), k, hint)
        self.console.print(table)
        self.console.print(f"[dim]Controls: 1–{len(options)} to act • h=help • ?=pot • q=quit[/]")

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
        if getattr(chosen, "resolution_note", None):
            self.console.print(f"Rival response: {chosen.resolution_note}")
        if getattr(chosen, "ends_hand", False):
            self.console.print("[dim]Hand ends on this action.[/]")
        self.console.print("[dim]—[/]\n")

    def summary(self, records: list[dict]) -> None:
        if not records:
            self.console.print("No hands answered.")
            return
        stats = summarize_records(records)
        total_ev_chosen = stats.total_ev_chosen
        total_ev_best = stats.total_ev_best
        total_ev_lost = stats.total_ev_lost
        avg_ev_lost = stats.avg_ev_lost
        hands_answered = stats.hands
        score_pct = stats.score_pct
        hits = stats.hits
        avg_loss_pct = stats.avg_loss_pct

        summary = Table(title="Session Summary", show_header=False)
        summary.add_row("Hands answered:", str(hands_answered))
        summary.add_row("Best choices hit:", f"{hits} ({100.0 * hits / len(records):.0f}%)")
        summary.add_row("Total EV (chosen):", f"{total_ev_chosen:.2f} bb")
        summary.add_row("Total EV (best possible):", f"{total_ev_best:.2f} bb")
        summary.add_row("Total EV lost:", f"{total_ev_lost:.2f} bb")
        summary.add_row("Average EV lost/decision:", f"{avg_ev_lost:.2f} bb")
        summary.add_row("Average EV lost (% pot):", f"{avg_loss_pct:.2f}%")
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
    def _hint_for_action(self, action: str) -> str:
        text = action.lower()
        if text.startswith("fold"):
            return "Let the hand go and move to the next decision."
        if text.startswith("call"):
            return "Match the wager to continue in the hand."
        if text.startswith("check"):
            return "Stay in the hand without adding more chips."
        if text.startswith("bet"):
            tail = action.split(" ", 1)[1] if " " in action else ""
            if "%" in text and tail:
                return f"Bet {tail} to apply pressure."
            return "Lead out and put chips in the pot."
        if text.startswith("raise"):
            tail = action.split(" ", 1)[1] if " " in action else ""
            return f"Increase the bet {tail} to apply pressure.".strip()
        if "3-bet" in text:
            tail = action.split(" ", 2)[-1] if " " in action else ""
            return f"Re-raise preflop to {tail} and seize initiative.".strip()
        return "Select to see more feedback after the action."

    def _sort_cards_by_rank(self, cards: list[int]) -> list[int]:
        # Rank is encoded as integer division by 4; higher rank first
        return sorted(cards, key=lambda ci: ci // 4, reverse=True)

    def _format_cards_colored(self, cards: list[int]) -> str:
        # Suits: 0=spades, 1=hearts, 2=diamonds, 3=clubs
        # Tailored four-color deck palette tuned for readability on both light/dark terminals.
        colors = {
            0: "bold white",  # spades – bright neutral
            1: "bold #c14657",  # hearts – vivid crimson rose
            2: "bold #2f73d2",  # diamonds – bright sapphire blue
            3: "bold #2f8a5e",  # clubs – saturated evergreen
        }
        # Keep original order for streets (flop/turn/river)
        parts: list[str] = []
        for c in cards:
            suit = c % 4
            color = colors.get(suit, "bold #f9fafb")
            txt = format_card_ascii(c, upper=True)
            parts.append(f"[{color}]{txt}[/]")
        return " ".join(parts)
