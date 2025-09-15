from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.interfaces import Presenter
from ..core.models import Option
from ..dynamic.generator import Node


class RichPresenter(Presenter):
    def __init__(self, *, no_color: bool = False, force_color: bool = False):
        self.console = Console(force_terminal=force_color, color_system=None if no_color else "auto")

    def start_session(self, total_hands: int) -> None:
        pass

    def start_hand(self, hand_index: int, total_hands: int) -> None:
        self.console.print(Panel.fit(f"Hand {hand_index}/{total_hands}", title="GTO Trainer – Live", style="bold cyan"))

    def show_node(self, node: Node, options: list[str]) -> None:
        self.console.print(f"[bold magenta]{node.street.upper()}[/]  [dim]- {node.description}[/]")
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
                return -1
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
            self.console.print(f"✅ [green]Best choice[/]: {best.key} (EV {best.ev:.2f} bb)")
        else:
            self.console.print(f"❌ [yellow]Better was[/]: {best.key} (EV {best.ev:.2f} bb)")
            self.console.print(f"You chose: {chosen.key} (EV {chosen.ev:.2f} bb) → EV lost: [red]{ev_loss:.2f} bb[/]")
        self.console.print(f"Why (your action): {chosen.why}")
        if not correct:
            self.console.print(f"Why (best action):  {best.why}")
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
