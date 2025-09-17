from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Label, Static

from ..core.engine_core import run_core
from ..core.interfaces import EpisodeGenerator, OptionProvider, Presenter
from ..core.models import Option
from ..dynamic.cards import canonical_hand_abbrev, format_card_ascii
from ..dynamic.generator import Node, generate_episode
from ..dynamic.policy import options_for
from ..solver.oracle import CompositeOptionProvider, CSVStrategyOracle

# --- Small helpers / adapters ---


class _DynamicGenerator(EpisodeGenerator):
    def generate(self, rng):  # type: ignore[override]
        return generate_episode(rng)


class _DynamicOptions(OptionProvider):
    def options(self, node, rng, mc_trials):  # type: ignore[override]
        return options_for(node, rng, mc_trials)


@dataclass
class AppConfig:
    hands: int = 1
    mc_trials: int = 200
    solver_csv: str | None = None


class _TextualPresenter(Presenter):
    """Bridges the synchronous engine to the async Textual UI via thread events."""

    def __init__(self, app: TrainerApp) -> None:
        self.app = app
        self._choice_event = threading.Event()
        self._choice_index: int | None = None

    # --- Protocol impl ---
    def start_session(self, total_hands: int) -> None:  # noqa: D401
        self.app.call_from_thread(self.app.show_session_start, total_hands)

    def start_hand(self, hand_index: int, total_hands: int) -> None:  # noqa: D401
        self.app.call_from_thread(self.app.show_hand_start, hand_index, total_hands)

    def show_node(self, node: Node, options: list[str]) -> None:  # noqa: D401
        self._choice_index = None
        self._choice_event.clear()
        self.app.call_from_thread(self.app.show_node, node, options)

    def prompt_choice(self, n: int) -> int:  # noqa: D401, ARG002
        # Block engine thread until UI signals a choice
        self._choice_event.wait()
        if self._choice_index is None:
            return -1
        return self._choice_index

    def step_feedback(self, node: Node, chosen: Option, best: Option) -> None:  # noqa: D401
        self.app.call_from_thread(self.app.show_step_feedback, node, chosen, best)

    def summary(self, records: list[dict]) -> None:  # noqa: D401
        self.app.call_from_thread(self.app.show_summary, records)

    # --- UI callback ---
    def set_choice(self, idx: int) -> None:
        self._choice_index = idx
        self._choice_event.set()


class TrainerApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }
    # Title bar
    # Main content splits into info (top) and options/feedback (bottom)
    # Keep mobile-friendly spacing and readable buttons
    .section { padding: 1; }
    # Cards row
    # Options: vertical list of large buttons
    # Feedback: static text area
    # Footer shows key hints
    # Buttons wider on mobile
    Button { width: 100%; min-height: 3; margin: 0 0 1 0; }
    # Labels wrap
    Label { text-align: left; }
    """

    # Reactive state for headline / meta
    headline: reactive[str | None] = reactive(None)
    meta: reactive[str | None] = reactive(None)
    _presenter: _TextualPresenter
    _engine_thread: threading.Thread | None = None
    _config: AppConfig

    def __init__(self, *, hands: int = 1, mc_trials: int = 200, solver_csv: str | None = None) -> None:
        super().__init__()
        self._config = AppConfig(hands=hands, mc_trials=mc_trials, solver_csv=solver_csv)

    # --- Compose UI ---
    def compose(self) -> ComposeResult:  # type: ignore[override]
        yield Header(show_clock=False)
        with Container(classes="section"):
            yield Label("GTO Poker Trainer", id="title")
            yield Label("", id="headline")
            yield Label("", id="meta")
            yield Label("", id="hand")
            yield Label("", id="board")
        with Container(classes="section"):
            yield Label("Choose your action:")
            yield Vertical(id="options")
        with Container(classes="section"):
            yield Label("Feedback:")
            yield Static("", id="feedback")
        with Horizontal(classes="section"):
            yield Button("Start New Session", id="btn-new", variant="success")
            yield Button("Quit", id="btn-quit", variant="error")
        yield Footer()

    # --- Engine control ---
    def on_mount(self) -> None:  # type: ignore[override]
        self._presenter = _TextualPresenter(self)
        # Start first session automatically
        self._start_engine_session()

    def _start_engine_session(self) -> None:
        if self._engine_thread and self._engine_thread.is_alive():
            return
        self.query_one("#feedback", Static).update("")
        self.query_one("#options", Vertical).remove_children()
        self._engine_thread = threading.Thread(target=self._engine_run, daemon=True)
        self._engine_thread.start()

    def _engine_run(self) -> None:
        # Build provider chain; for now, always dynamic. CSV solver support can be added later.
        option_provider: OptionProvider = _DynamicOptions()
        if self._config.solver_csv:
            try:
                solver = CSVStrategyOracle(self._config.solver_csv)
                option_provider = CompositeOptionProvider(primary=solver, fallback=option_provider)
            except Exception as exc:  # pragma: no cover - optional path
                self.query_one("#feedback", Static).update(
                    f"[red]Failed to load solver CSV[/]: {exc}. Falling back to heuristics."
                )
        # Deterministic per-session seed
        seed = secrets.randbits(32)
        run_core(
            generator=_DynamicGenerator(),
            option_provider=option_provider,
            presenter=self._presenter,
            seed=seed,
            hands=self._config.hands,
            mc_trials=self._config.mc_trials,
        )

    # --- Presenter-driven UI updates ---
    def show_session_start(self, total_hands: int) -> None:
        self.query_one("#headline", Label).update(f"Session start — {total_hands} hand(s)")
        self.query_one("#meta", Label).update("")

    def show_hand_start(self, hand_index: int, total_hands: int) -> None:
        self.query_one("#headline", Label).update(f"Hand {hand_index}/{total_hands}")
        self.query_one("#feedback", Static).update("")
        self.query_one("#options", Vertical).remove_children()

    def _format_cards_colored(self, cards: list[int]) -> str:
        colors = {
            0: "#111827",  # spades (slate 900)
            1: "#ef4444",  # hearts (red 500)
            2: "#2563eb",  # diamonds (blue 600)
            3: "#10b981",  # clubs (emerald 500)
        }
        parts = []
        for c in cards:
            suit = c % 4
            txt = format_card_ascii(c, upper=True)
            style = colors.get(suit, "#f9fafb")
            parts.append(f"[bold {style}]" + txt + "[/]")
        return " ".join(parts)

    def show_node(self, node: Node, options: list[str]) -> None:
        # Headline and context
        desc = node.description
        if node.board and desc.startswith("Board "):
            dot = desc.find(". ")
            desc = desc[dot + 2 :] if dot != -1 else ""
        headline = f"[bold magenta]{node.street.upper()}[/]"
        if desc:
            headline += f"  [dim]- {desc}[/]"
        self.query_one("#headline", Label).update(headline)

        P = float(node.pot_bb)
        spr = (node.effective_bb / P) if P > 0 else float("inf")
        meta = f"Pot: {P:.2f}bb | SPR: {spr:.1f}"
        bet = node.context.get("bet")
        if isinstance(bet, (int, float)):
            pct = 100.0 * float(bet) / max(1e-9, P)
            meta += f" | OOP bet: {float(bet):.2f}bb ({pct:.0f}% pot)"
        self.query_one("#meta", Label).update(meta)

        hand_str = self._format_cards_colored(node.hero_cards)
        self.query_one("#hand", Label).update(
            f"Your hand: {hand_str} [dim]({canonical_hand_abbrev(node.hero_cards)})[/]"
        )
        if node.board:
            self.query_one("#board", Label).update(f"Board: {self._format_cards_colored(node.board)}")
        else:
            self.query_one("#board", Label).update("")

        # Render actions as buttons
        opt_container = self.query_one("#options", Vertical)
        opt_container.remove_children()
        for i, k in enumerate(options, 1):
            btn = Button(f"{i}. {k}", id=f"opt-{i - 1}")
            opt_container.mount(btn)

    def show_step_feedback(self, _node: Node, chosen: Option, best: Option) -> None:
        correct = chosen.key == best.key
        ev_loss = best.ev - chosen.ev
        lines = []
        if correct:
            lines.append(f"[green]✓ Best choice[/]: {best.key} (EV {best.ev:.2f} bb)")
        else:
            lines.append(f"[yellow]✗ Better was[/]: {best.key} (EV {best.ev:.2f} bb)")
            lines.append(f"You chose: {chosen.key} (EV {chosen.ev:.2f} bb) → EV lost: [red]{ev_loss:.2f} bb[/]")
        if chosen.why:
            lines.append(f"Why (yours): {chosen.why}")
        if not correct and best.why:
            lines.append(f"Why (best): {best.why}")
        if getattr(chosen, "ends_hand", False):
            lines.append("[dim]Hand ends on this action.[/]")
        self.query_one("#feedback", Static).update("\n".join(lines))

    def show_summary(self, records: list[dict[str, Any]]) -> None:
        if not records:
            self.query_one("#feedback", Static).update("No hands answered.")
            return
        total_ev_best = sum(r["best_ev"] for r in records)
        total_ev_chosen = sum(r["chosen_ev"] for r in records)
        total_ev_lost = total_ev_best - total_ev_chosen
        avg_ev_lost = total_ev_lost / len(records)
        hits = sum(1 for r in records if r["chosen_key"] == r["best_key"])
        score_pct = 0.0
        room = sum(max(1e-9, r["best_ev"] - min(0.0, r["chosen_ev"])) for r in records)
        if room > 1e-9:
            score_pct = 100.0 * max(0.0, 1.0 - (total_ev_lost / room))
        msg = (
            f"[b]Session Summary[/]\nHands answered: {len(records)}\n"
            f"Best choices hit: {hits} ({(100.0 * hits / len(records)):.0f}%)\n"
            f"Total EV (chosen): {total_ev_chosen:.2f} bb\n"
            f"Total EV (best): {total_ev_best:.2f} bb\n"
            f"Total EV lost: {total_ev_lost:.2f} bb\n"
            f"Avg EV lost/decision: {avg_ev_lost:.2f} bb\n"
            f"Score (0–100): {score_pct:.0f}"
        )
        self.query_one("#feedback", Static).update(msg)

    # --- UI events ---
    @on(Button.Pressed, "#btn-new")
    def _on_new(self) -> None:
        self._start_engine_session()

    @on(Button.Pressed, "#btn-quit")
    def _on_quit(self) -> None:
        # Signal a quit to any pending prompt; -1 means end session
        if hasattr(self, "_presenter"):
            self._presenter.set_choice(-1)
        self.exit()

    @on(Button.Pressed)
    def _on_option_pressed(self, event: Button.Pressed) -> None:
        if not event.button.id or not event.button.id.startswith("opt-"):
            return
        idx = int(event.button.id.split("-", 1)[1])
        self._presenter.set_choice(idx)


def run_textual(hands: int = 1, mc_trials: int = 200, solver_csv: str | None = None) -> None:
    app = TrainerApp(hands=hands, mc_trials=mc_trials, solver_csv=solver_csv)
    app.run()
