from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Grid, Horizontal
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Label, Static

from ..core.engine_core import run_core
from ..core.interfaces import EpisodeGenerator, OptionProvider, Presenter
from ..core.models import Option
from ..dynamic.cards import canonical_hand_abbrev, format_card_ascii
from ..dynamic.generator import Node, generate_episode
from ..dynamic.policy import options_for, resolve_for
from ..solver.oracle import CompositeOptionProvider, CSVStrategyOracle

# --- Small helpers / adapters ---


class _DynamicGenerator(EpisodeGenerator):
    def generate(self, rng):  # type: ignore[override]
        return generate_episode(rng)


class _DynamicOptions(OptionProvider):
    def options(self, node, rng, mc_trials):  # type: ignore[override]
        return options_for(node, rng, mc_trials)

    def resolve(self, node, chosen, rng):  # type: ignore[override]
        return resolve_for(node, chosen, rng)


@dataclass
class AppConfig:
    hands: int = 1
    mc_trials: int = 120
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

    def cancel_session(self) -> None:
        self.set_choice(-1)


class TrainerApp(App[None]):
    BINDINGS = [
        ("ctrl+n", "new_session", "Start Fresh Hand"),
        ("escape", "end_session", "End Session"),
        ("ctrl+q", "quit_app", "Quit"),
    ]
    CSS = """
    Screen {
        layout: vertical;
        background: #0b1220;
        color: #d5dde9;
    }
    .section {
        padding: 1;
        background: #101727;
        border: 1px solid #1c2940;
        margin: 0 0 1 0;
    }
    #info {
        width: 100%;
        background: #0f1625;
        border: 1px solid #1b273b;
    }
    #headline-row, #meta-row, #cards-row, #board-row { width: 100%; }
    .headline-col { width: 100%; }
    .meta-panel {
        width: 100%;
        background: #131d30;
        border: 1px solid #1f2c45;
        padding: 0 1;
    }
    .card-panel {
        width: 100%;
        padding: 0;
        font-family: monospace;
        background: #131d30;
        border: 1px solid #1f2c45;
    }
    #options { layout: grid; grid-columns: 1fr; grid-gutter: 0 1; width: 100%; max-width: 48; }
    #controls { column-gap: 1; }
    Button {
        width: 100%;
        min-height: 2;
        padding: 0 1;
        margin: 0 0 0.5 0;
        background: #1f2937;
        color: #f8fafc;
        border: 1px solid #334155;
        transition: background 0.15s ease, border 0.15s ease;
        text-align: left;
    }
    Button:hover { background: #273449; }
    Button:focus { border: 1px solid #94a3b8; }
    .option-button { background: #1f2937; border: 1px solid #334155; color: #f8fafc; }
    .option-button:hover { background: #273449; }
    .option-button:focus { border: 1px solid #64748b; }
    .option-fold { background: #3b1010; border: 1px solid #7f1d1d; }
    .option-fold:hover { background: #4c1515; }
    .option-call { background: #0f2f52; border: 1px solid #1d4ed8; }
    .option-call:hover { background: #153a63; }
    .option-check { background: #1f2937; border: 1px solid #475569; }
    .option-check:hover { background: #273447; }
    .option-value { background: #0f4c3a; border: 1px solid #0f766e; }
    .option-value:hover { background: #135b47; }
    #btn-new { background: #0f766e; border: 1px solid #0f766e; }
    #btn-new:hover { background: #0d5f58; }
    #btn-end { background: #b45309; border: 1px solid #b45309; }
    #btn-end:hover { background: #92400e; }
    #btn-quit { background: #7f1d1d; border: 1px solid #7f1d1d; }
    #btn-quit:hover { background: #641616; }
    Label { text-align: left; width: 100%; color: #e2e8f0; }
    Static { color: #d5dde9; }
    #board { white-space: pre-wrap; background: #131d30; border: 1px solid #1f2c45; }
    #feedback {
        background: #131d30;
        border: 1px solid #1f2c45;
        padding: 1;
    }
    #options { grid-size: 3; grid-gutter: 1 1; }
    """

    # Reactive state for headline / meta
    headline: reactive[str | None] = reactive(None)
    meta: reactive[str | None] = reactive(None)
    _presenter: _TextualPresenter
    _engine_thread: threading.Thread | None = None
    _config: AppConfig
    CARD_STYLES = {
        0: "white",  # spades – bright neutral
        1: "bright_red",  # hearts – vivid red
        2: "bright_cyan",  # diamonds – high-contrast cyan
        3: "bright_green",  # clubs – lively green
    }

    # Cached widget references populated on mount to avoid hot-path lookups
    _title_label: Label | None = None
    _headline_label: Label | None = None
    _meta_panel: Static | None = None
    _hand_panel: Static | None = None
    _board_panel: Static | None = None
    _options_container: Grid | None = None
    _feedback_panel: Static | None = None
    _pending_restart: bool = False
    _idle_after_stop: bool = False

    def __init__(self, *, hands: int = 1, mc_trials: int = 120, solver_csv: str | None = None) -> None:
        super().__init__()
        self._config = AppConfig(hands=hands, mc_trials=mc_trials, solver_csv=solver_csv)
        self._pending_restart = False

    # --- Compose UI ---
    def compose(self) -> ComposeResult:  # type: ignore[override]
        yield Header(show_clock=False)
        with Container(classes="section", id="info"):
            yield Label("GTO Poker Trainer", id="title")
            with Horizontal(id="headline-row"):
                yield Label("", id="headline", classes="headline-col")
            with Horizontal(id="meta-row"):
                yield Static("", id="meta", classes="meta-panel")
            with Horizontal(id="cards-row"):
                yield Static("", id="hand", classes="card-panel")
            with Horizontal(id="board-row"):
                yield Static("", id="board", classes="card-panel")
        with Container(classes="section"):
            yield Label("Choose your action:")
            yield Grid(id="options")
        with Container(classes="section"):
            yield Label("Feedback:")
            yield Static("", id="feedback")
        with Horizontal(classes="section", id="controls"):
            yield Button("Start Fresh Hand", id="btn-new", variant="success")
            yield Button("End Session", id="btn-end", variant="warning")
            yield Button("Quit", id="btn-quit", variant="error")
        yield Footer()

    # --- Engine control ---
    def on_mount(self) -> None:  # type: ignore[override]
        self._title_label = self.query_one("#title", Label)
        self._headline_label = self.query_one("#headline", Label)
        self._meta_panel = self.query_one("#meta", Static)
        self._hand_panel = self.query_one("#hand", Static)
        self._board_panel = self.query_one("#board", Static)
        self._options_container = self.query_one("#options", Grid)
        self._feedback_panel = self.query_one("#feedback", Static)

        self._presenter = _TextualPresenter(self)
        # Start first session automatically
        self._start_engine_session()

    def _start_engine_session(self) -> None:
        if self._engine_thread and self._engine_thread.is_alive():
            return
        if self._feedback_panel:
            self._feedback_panel.update("")
        if self._options_container:
            self._options_container.remove_children()
        self._engine_thread = threading.Thread(target=self._engine_run, daemon=True)
        self._engine_thread.start()

    def _engine_run(self) -> None:
        current = threading.current_thread()
        try:
            # Build provider chain; for now, always dynamic. CSV solver support can be added later.
            option_provider: OptionProvider = _DynamicOptions()
            if self._config.solver_csv:
                try:
                    solver = CSVStrategyOracle(self._config.solver_csv)
                    option_provider = CompositeOptionProvider(primary=solver, fallback=option_provider)
                except Exception as exc:  # pragma: no cover - optional path
                    if self._feedback_panel:
                        self._feedback_panel.update(
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
        finally:
            if self._engine_thread is current:
                self._engine_thread = None
            if self._idle_after_stop:
                self._idle_after_stop = False
                self.call_from_thread(self._show_idle_prompt)

    # --- Presenter-driven UI updates ---
    def show_session_start(self, total_hands: int) -> None:
        if self._headline_label:
            self._headline_label.update(f"Session start — {total_hands} hand(s)")
        if self._meta_panel:
            self._meta_panel.update("")

    def show_hand_start(self, hand_index: int, total_hands: int) -> None:
        if self._headline_label:
            self._headline_label.update(f"Hand {hand_index}/{total_hands}")
        if self._feedback_panel:
            self._feedback_panel.update("")
        if self._options_container:
            self._options_container.remove_children()

    def _render_card_token(self, card: int | None, *, placeholder: str = "--") -> str:
        if card is None:
            return f"[dim]{placeholder}[/]"
        suit = card % 4
        txt = format_card_ascii(card, upper=True)
        style = self.CARD_STYLES.get(suit, "white")
        return f"[bold {style}]" + txt + "[/]"

    def _format_cards_colored(self, cards: list[int]) -> str:
        return " ".join(self._render_card_token(c) for c in cards)

    def _format_board_rows(self, board: list[int]) -> str:
        slots: list[int | None] = list(board)
        while len(slots) < 5:
            slots.append(None)
        flop_row = " ".join(self._render_card_token(c) for c in slots[:3])
        turn_row = " ".join(self._render_card_token(c) for c in slots[3:])
        return f"{flop_row}\n{turn_row}"

    def show_node(self, node: Node, options: list[str]) -> None:
        # Headline and context
        desc = node.description
        if node.board and desc.startswith("Board "):
            dot = desc.find(". ")
            desc = desc[dot + 2 :] if dot != -1 else ""
        headline = f"[bold magenta]{node.street.upper()}[/]"
        if desc:
            headline += f"  [dim]- {desc}[/]"
        if self._headline_label:
            self._headline_label.update(headline)

        P = float(node.pot_bb)
        spr = (node.effective_bb / P) if P > 0 else float("inf")
        meta_lines = [f"Pot: {P:.2f} bb (SPR {spr:.1f})", f"Effective stack: {node.effective_bb:.1f} bb"]
        bet = node.context.get("bet")
        if isinstance(bet, (int, float)):
            pct = 100.0 * float(bet) / max(1e-9, P)
            meta_lines.append(f"Facing bet: {float(bet):.2f} bb ({pct:.0f}% pot)")
        if self._meta_panel:
            self._meta_panel.update("\n".join(meta_lines))

        hand_str = self._format_cards_colored(node.hero_cards)
        if self._hand_panel:
            self._hand_panel.update(
                f"Hero: {hand_str} [dim]({canonical_hand_abbrev(node.hero_cards)})[/]"
            )
        if self._board_panel:
            board_rows = self._format_board_rows(node.board)
            self._board_panel.update(f"Board\n{board_rows}")

        # Render actions as buttons
        if self._options_container:
            self._options_container.remove_children()
            buttons = []
            for i, k in enumerate(options, 1):
                classes = ["option-button"]
                key_lower = k.lower()
                if "fold" in key_lower:
                    classes.append("option-fold")
                elif "call" in key_lower:
                    classes.append("option-call")
                elif "check" in key_lower:
                    classes.append("option-check")
                else:
                    classes.append("option-value")
                btn = Button(f"{i}. {k}", id=f"opt-{i - 1}", classes=" ".join(classes))
                buttons.append(btn)
            if buttons:
                self._options_container.mount(*buttons)

    def show_step_feedback(self, _node: Node, chosen: Option, best: Option) -> None:
        correct = chosen.key == best.key
        ev_loss = best.ev - chosen.ev
        lines = ["[b]Decision Grade[/]"]
        if correct:
            lines.append(f"[green]Optimal[/] — {chosen.key} (EV {chosen.ev:.2f} bb)")
        else:
            lines.append(f"[red]{ev_loss:.2f} bb[/] behind optimal.")
            lines.append(f"Your action: {chosen.key} (EV {chosen.ev:.2f} bb)")
            lines.append(f"Best action: {best.key} (EV {best.ev:.2f} bb)")
        if chosen.why:
            lines.append(f"• Your reasoning: {chosen.why}")
        if not correct and best.why:
            lines.append(f"• Better reasoning: {best.why}")
        if getattr(chosen, "ends_hand", False):
            lines.append("[dim]Hand ends on this action.[/]")
        if self._feedback_panel:
            self._feedback_panel.update("\n".join(lines))

    def show_summary(self, records: list[dict[str, Any]]) -> None:
        if not records:
            if self._feedback_panel:
                self._feedback_panel.update("No hands answered.")
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
        if self._feedback_panel:
            self._feedback_panel.update(msg)

    # --- Session lifecycle helpers ---
    def _queue_restart_when_idle(self) -> None:
        def _try_start() -> None:
            if self._engine_thread and self._engine_thread.is_alive():
                self.call_later(_try_start)
                return
            if not self._pending_restart:
                return
            self._pending_restart = False
            self._start_engine_session()

        self.call_later(_try_start)

    def _request_restart(self) -> None:
        if self._engine_thread and self._engine_thread.is_alive():
            if self._pending_restart:
                return
            self._pending_restart = True
            self._idle_after_stop = False
            if hasattr(self, "_presenter"):
                self._presenter.cancel_session()
            self._queue_restart_when_idle()
            return

        self._pending_restart = False
        self._idle_after_stop = False
        self._start_engine_session()

    def _show_idle_prompt(self) -> None:
        if self._headline_label:
            self._headline_label.update("Session stopped — press Start Fresh Hand to resume")
        if self._meta_panel:
            self._meta_panel.update("")
        if self._options_container:
            self._options_container.remove_children()
        if self._feedback_panel:
            self._feedback_panel.update("[dim]Session ended at your request.[/]")

    # --- UI events ---
    @on(Button.Pressed, "#btn-new")
    def _on_new(self) -> None:
        self._request_restart()

    @on(Button.Pressed, "#btn-end")
    def _on_end(self) -> None:
        if self._engine_thread and self._engine_thread.is_alive():
            self._pending_restart = False
            self._idle_after_stop = True
            if self._feedback_panel:
                self._feedback_panel.update("[dim]Ending session…[/]")
            if hasattr(self, "_presenter"):
                self._presenter.cancel_session()
        else:
            self._show_idle_prompt()

    @on(Button.Pressed, "#btn-quit")
    def _on_quit(self) -> None:
        # Signal a quit to any pending prompt; -1 means end session
        if hasattr(self, "_presenter"):
            self._presenter.cancel_session()
        self.exit()

    def action_new_session(self) -> None:
        self._request_restart()

    def action_end_session(self) -> None:
        self._on_end()

    def action_quit_app(self) -> None:
        self._on_quit()

    @on(Button.Pressed)
    def _on_option_pressed(self, event: Button.Pressed) -> None:
        if not event.button.id or not event.button.id.startswith("opt-"):
            return
        idx = int(event.button.id.split("-", 1)[1])
        self._presenter.set_choice(idx)


def run_textual(hands: int = 1, mc_trials: int = 120, solver_csv: str | None = None) -> None:
    app = TrainerApp(hands=hands, mc_trials=mc_trials, solver_csv=solver_csv)
    app.run()
