from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Grid, Horizontal
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Button, Footer, Header, Label, Static

from ..core.engine_core import run_core
from ..core.interfaces import EpisodeGenerator, OptionProvider, Presenter
from ..core.models import Option
from ..core.scoring import summarize_records
from ..dynamic.cards import canonical_hand_abbrev, format_card_ascii
from ..dynamic.generator import Node, generate_episode
from ..dynamic.policy import options_for, resolve_for
from ..solver.oracle import CompositeOptionProvider, CSVStrategyOracle

_ACTION_PALETTE: dict[str, dict[str, str]] = {
    "fold": {
        "bg": "#f2d6d9",
        "border": "#dfbcc1",
        "text": "#684249",
        "hover": "#ebcacf",
    },
    "passive": {
        "bg": "#eef2fb",
        "border": "#d9e0f2",
        "text": "#3f4a63",
        "hover": "#e5e9f6",
    },
    "aggressive": {
        "bg": "#ece6f5",
        "border": "#d6cde8",
        "text": "#4d3d66",
        "hover": "#e3dced",
    },
}

_ACTION_CLASS_MAP: dict[str, str] = {
    "option-fold": "fold",
    "option-call": "passive",
    "option-check": "passive",
    "option-bet": "aggressive",
    "option-raise": "aggressive",
}


def _build_action_css() -> str:
    lines: list[str] = []
    for class_name, palette_key in _ACTION_CLASS_MAP.items():
        palette = _ACTION_PALETTE[palette_key]
        lines.append(
            "    .{cls} {{ background: {bg}; border: 1px solid {border}; color: {text}; }}".format(
                cls=class_name,
                bg=palette["bg"],
                border=palette["border"],
                text=palette["text"],
            )
        )
        lines.append(
            "    .{cls}:hover {{ background: {hover}; }}".format(
                cls=class_name,
                hover=palette["hover"],
            )
        )
    return "\n".join(lines) + "\n"


_BASE_CSS = """
    Screen {
        layout: vertical;
        background: #f4f6fb;
        color: #1b233d;
    }
    Header {
        background: #fdfdff;
        color: #141c35;
        border-bottom: 1px solid #d6def3;
    }
    Footer {
        background: #fdfdff;
        color: #4a5678;
        border-top: 1px solid #d6def3;
    }
    .section {
        padding: 1.5 2;
        background: #ffffff;
        border: 1px solid #d9e2f5;
        margin: 0 0 1 0;
    }
    #info {
        width: 100%;
        background: #fdfdff;
        border: 1px solid #d2dcf5;
        padding: 2 3;
        margin: 1 0 1 0;
    }
    #title {
        text-align: center;
        color: #111a33;
        margin: 0 0 0.5 0;
    }
    #tagline {
        text-align: center;
        color: #46557b;
        margin: 0 0 1 0;
    }
    #session-status {
        text-align: center;
        padding: 0.5 2;
        margin: 0 0 1 0;
        background: #edf2ff;
        border: 1px solid #c8d9ff;
        color: #203261;
    }
    #headline-row, #meta-row, #cards-row, #board-row {
        width: 100%;
        justify-content: center;
    }
    .headline-col { width: 100%; }
    #headline {
        padding: 0.2 2;
        background: #e7edff;
        border: 1px solid #c7d6ff;
        color: #1b2d55;
        text-align: center;
        min-width: 24;
    }
    .meta-panel {
        width: 100%;
        background: #f6f8ff;
        border: 1px solid #d3dcf6;
        padding: 1 2;
        color: #2d3b62;
    }
    .card-panel {
        width: 100%;
        padding: 1 2;
        font-family: monospace;
        background: #f1f4fb;
        border: 1px dashed #c3cde3;
        color: #1b2d55;
    }
    #options {
        layout: grid;
        grid-columns: 1fr;
        grid-gutter: 0 1;
        width: 100%;
        max-width: 50;
        margin: 0 auto;
    }
    #options-block {
        border-top: 1px solid #d7dfea;
        margin: 1.2 0 0 0;
        padding: 1.2 0 0 0;
    }
    .options-title {
        margin: 0 0 0.6 0;
        color: #2d3b62;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.85em;
    }
    #controls {
        padding: 0;
    }
    #primary-controls {
        column-gap: 1.5;
        justify-content: center;
    }
    #end-session-container {
        display: flex;
        justify-content: center;
        padding: 0.4 0;
    }
    #end-session-container Button {
        max-width: 24;
    }
    Button {
        width: 100%;
        min-height: 1.8;
        padding: 0.35 1.25;
        margin: 0 0 0.3 0;
        background: #ffffff;
        color: #1b2d55;
        border: 1px solid #d1daf3;
        transition: background 0.2s ease, border 0.2s ease, color 0.2s ease;
        text-align: left;
    }
    Button:hover { background: #eef3ff; border: 1px solid #b8c7f2; }
    Button:focus { border: 1px solid #5b76f8; }
    #controls Button {
        width: auto;
        min-width: 16;
        text-align: center;
        margin: 0;
    }
    .option-button { background: #ffffff; border: 1px solid #d1daf3; color: #1b2d55; }
    .option-button:hover { background: #eef3ff; }
    .option-button:focus { border: 1px solid #5b76f8; }
"""

_POST_ACTION_CSS = """
    #btn-new { background: #2f6bff; border: 1px solid #2a5de0; color: #ffffff; text-align: center; }
    #btn-new:hover { background: #2657d1; }
    #btn-end { background: #f3f6ff; border: 1px solid #ccd8ff; color: #253260; text-align: center; }
    #btn-end:hover { background: #e6ecff; }
    #btn-quit { background: #f8fafc; border: 1px solid #d9e0f3; color: #2d3655; text-align: center; }
    #btn-quit:hover { background: #eef2f9; }
    Label { text-align: left; width: 100%; color: #1b233d; }
    Static { color: #2d3b62; }
    #board { white-space: pre-wrap; background: #f1f4fb; border: 1px dashed #c3cde3; color: #1b2d55; }
    #feedback {
        background: #ffffff;
        border: 1px solid #d9e0f3;
        padding: 1 2;
        color: #2d3b62;
        min-height: 5;
    }
"""
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
    TITLE = "GTO Trainer"
    SUB_TITLE = "Solver-calibrated drills for every street"
    BINDINGS = [
        ("ctrl+n", "new_session", "Start Fresh Hand"),
        ("escape", "end_session", "End Session"),
        ("ctrl+q", "quit_app", "Quit"),
    ]
    CSS = _BASE_CSS + _build_action_css() + _POST_ACTION_CSS
    _PREPARING_FRAMES: tuple[str, ...] = ("|", "/", "-", "\\")
    _PREPARING_INTERVAL: float = 0.18

    # Reactive state for headline / meta
    headline: reactive[str | None] = reactive(None)
    meta: reactive[str | None] = reactive(None)
    _presenter: _TextualPresenter
    _engine_thread: threading.Thread | None = None
    _config: AppConfig
    CARD_STYLES = {
        0: "#1f2740",  # spades – deep indigo
        1: "#d24a5f",  # hearts – punchy crimson
        2: "#2d6fe6",  # diamonds – vibrant cobalt
        3: "#2c9a6d",  # clubs – lively jade
    }

    # Cached widget references populated on mount to avoid hot-path lookups
    _title_label: Label | None = None
    _headline_label: Label | None = None
    _tagline_panel: Static | None = None
    _status_panel: Static | None = None
    _meta_panel: Static | None = None
    _hand_panel: Static | None = None
    _board_panel: Static | None = None
    _options_container: Grid | None = None
    _feedback_panel: Static | None = None
    _pending_restart: bool = False
    _idle_after_stop: bool = False
    _preparing_timer: Timer | None = None
    _preparing_frame_index: int = 0
    _preparing_hint: str | None = None
    _current_hand_index: int = 0
    _total_hands: int = 0
    _decisions_played: int = 0
    _best_hits: int = 0
    _total_ev_lost: float = 0.0

    def __init__(self, *, hands: int = 1, mc_trials: int = 120, solver_csv: str | None = None) -> None:
        super().__init__()
        self._config = AppConfig(hands=hands, mc_trials=mc_trials, solver_csv=solver_csv)
        self._pending_restart = False
        self._preparing_timer = None
        self._preparing_frame_index = 0
        self._preparing_hint = None
        self._current_hand_index = 0
        self._total_hands = 0
        self._decisions_played = 0
        self._best_hits = 0
        self._total_ev_lost = 0.0

    # --- Compose UI ---
    def _format_preparing_text(self) -> str:
        frame = self._PREPARING_FRAMES[self._preparing_frame_index]
        lines = [
            f"[b #1b2d55]Preparing a fresh hand {frame}[/]",
            "[dim]Shuffling combos and loading dynamics…[/]",
        ]
        if self._preparing_hint:
            lines.append(self._preparing_hint)
        return "\n".join(lines)

    def _start_preparing_animation(self) -> None:
        if not self._status_panel:
            return
        self._stop_preparing_animation()
        self._preparing_frame_index = 0
        self._status_panel.update(self._format_preparing_text())
        self._preparing_timer = self.set_interval(
            self._PREPARING_INTERVAL,
            self._tick_preparing_animation,
        )

    def _stop_preparing_animation(self) -> None:
        if self._preparing_timer:
            self._preparing_timer.stop()
            self._preparing_timer = None
        self._preparing_hint = None

    def _tick_preparing_animation(self, timer: Timer) -> None:  # noqa: ARG002
        if not self._status_panel:
            return
        self._preparing_frame_index = (self._preparing_frame_index + 1) % len(self._PREPARING_FRAMES)
        self._status_panel.update(self._format_preparing_text())

    def _apply_preparing_placeholders(self) -> None:
        if self._hand_panel:
            self._hand_panel.update("[b #1b2d55]Hero[/]: [dim]-- --[/] [dim](awaiting cards)[/]")
        if self._board_panel:
            self._board_panel.update(f"[b #1b2d55]Board[/]\n{self._format_board_rows([])}")
        if self._meta_panel:
            self._meta_panel.update("[dim]Crunching equities and sizing context…[/]")
        if self._options_container:
            self._options_container.remove_children()

    def _hand_progress_fragment(self) -> str:
        if not self._total_hands:
            return ""
        index = max(1, min(self._current_hand_index or 1, self._total_hands))
        return f"[b #2f6bff]Hand {index}/{self._total_hands}[/]"

    def _session_perf_fragment(self) -> str | None:
        if self._decisions_played <= 0:
            return None
        accuracy_pct = (100.0 * self._best_hits) / self._decisions_played if self._decisions_played else 0.0
        ev_delta = -self._total_ev_lost
        return (
            f"[#2d3b62]ΔEV {ev_delta:+.2f} bb[/]"
            f"  [dim]{accuracy_pct:.0f}% hits ({self._best_hits}/{self._decisions_played})[/]"
        )

    def _headline_for_state(
        self,
        *,
        street: str | None = None,
        info: str | None = None,
        stats: str | None = None,
    ) -> str:
        fragments: list[str] = []
        progress = self._hand_progress_fragment()
        if progress:
            fragments.append(progress)
        if street:
            fragments.append(f"[b #1b2d55]{street}[/]")
        if info:
            fragments.append(info)
        if stats:
            fragments.append(stats)
        if not fragments:
            return "[b #1b2d55]GTO Trainer[/]"
        return "  [dim]•[/]  ".join(fragments)

    def _build_headline(self, node: Node) -> str:
        street = node.street.title()
        return self._headline_for_state(street=street, stats=self._session_perf_fragment())

    def compose(self) -> ComposeResult:  # type: ignore[override]
        yield Header(show_clock=False)
        with Container(classes="section", id="info"):
            yield Label("GTO Trainer", id="title")
            yield Static("Sharpen your instincts with solver-backed drills.", id="tagline")
            yield Static("We're dealing your first scenario…", id="session-status")
            with Horizontal(id="headline-row"):
                yield Label("", id="headline", classes="headline-col")
            with Horizontal(id="meta-row"):
                yield Static("", id="meta", classes="meta-panel")
            with Horizontal(id="cards-row"):
                yield Static("", id="hand", classes="card-panel")
            with Horizontal(id="board-row"):
                yield Static("", id="board", classes="card-panel")
            with Container(id="options-block"):
                yield Static("Choose your action:", classes="options-title")
                yield Grid(id="options")
        with Container(classes="section"):
            yield Label("Feedback:")
            yield Static("", id="feedback")
        with Container(classes="section", id="controls"):
            yield Horizontal(
                Button("Start Fresh Hand", id="btn-new", variant="success"),
                Button("Quit", id="btn-quit", variant="error"),
                id="primary-controls",
            )
        with Container(classes="section", id="end-session-container"):
            yield Button("End Session", id="btn-end", variant="warning")
        yield Footer()

    # --- Engine control ---
    def on_mount(self) -> None:  # type: ignore[override]
        self._title_label = self.query_one("#title", Label)
        self._headline_label = self.query_one("#headline", Label)
        self._tagline_panel = self.query_one("#tagline", Static)
        self._status_panel = self.query_one("#session-status", Static)
        self._meta_panel = self.query_one("#meta", Static)
        self._hand_panel = self.query_one("#hand", Static)
        self._board_panel = self.query_one("#board", Static)
        self._options_container = self.query_one("#options", Grid)
        self._feedback_panel = self.query_one("#feedback", Static)

        if self._title_label:
            self._title_label.update("[b #111a33]GTO Trainer[/]")
        if self._tagline_panel:
            self._tagline_panel.update("[#3c4f86]Sharpen your instincts with solver-backed drills.[/]")
        if self._status_panel:
            self._status_panel.update(
                "[b #1b2d55]Preparing a fresh hand[/]\n[dim]Hang tight—simulations are spinning up.[/]"
            )
        if self._meta_panel:
            self._meta_panel.update("[dim]We'll surface pot details the moment action begins.[/]")
        if self._hand_panel:
            self._hand_panel.update("[b #1b2d55]Hero[/]: [dim]-- --[/] [dim](awaiting cards)[/]")
        if self._board_panel:
            self._board_panel.update("[b #1b2d55]Board[/]\n[dim]-- -- --\n-- --[/]")
        if self._feedback_panel:
            self._feedback_panel.update(
                "[dim]Decision feedback, EV gaps, and coaching notes will land here after each choice.[/]"
            )

        self._presenter = _TextualPresenter(self)
        # Start first session automatically
        self._start_engine_session()

    def _start_engine_session(self) -> None:
        if self._engine_thread and self._engine_thread.is_alive():
            return
        if self._feedback_panel:
            self._feedback_panel.update("")
        self._apply_preparing_placeholders()
        self._start_preparing_animation()
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
            self.call_from_thread(self._stop_preparing_animation)
            if self._idle_after_stop:
                self._idle_after_stop = False
                self.call_from_thread(self._show_idle_prompt)

    # --- Presenter-driven UI updates ---
    def show_session_start(self, total_hands: int) -> None:
        self._total_hands = total_hands
        self._current_hand_index = 0
        self._decisions_played = 0
        self._best_hits = 0
        self._total_ev_lost = 0.0
        self._apply_preparing_placeholders()
        if self._headline_label:
            plural = "s" if total_hands != 1 else ""
            info = f"[#2d3b62]{total_hands} hand{plural} queued[/]"
            self._headline_label.update(
                self._headline_for_state(info=info, stats=self._session_perf_fragment())
            )
        if self._meta_panel:
            self._meta_panel.update("")
        plural = "s" if total_hands != 1 else ""
        self._preparing_hint = f"[dim]{total_hands} hand{plural} queued for play.[/]"
        self._start_preparing_animation()

    def show_hand_start(self, hand_index: int, total_hands: int) -> None:
        self._current_hand_index = hand_index
        self._total_hands = total_hands
        self._apply_preparing_placeholders()
        if self._headline_label:
            self._headline_label.update(
                self._headline_for_state(
                    info="[#2d3b62]Generating scenario…[/]",
                    stats=self._session_perf_fragment(),
                )
            )
        if self._feedback_panel:
            self._feedback_panel.update("")
        if self._options_container:
            self._options_container.remove_children()
        self._preparing_hint = (
            f"[dim]Decision {hand_index}/{total_hands} — building tree and equities…[/]"
        )
        self._start_preparing_animation()

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
        self._stop_preparing_animation()
        if self._headline_label:
            self._headline_label.update(self._build_headline(node))

        P = float(node.pot_bb)
        spr = (node.effective_bb / P) if P > 0 else float("inf")
        bet = node.context.get("bet")
        if isinstance(bet, (int, float)):
            pct = 100.0 * float(bet) / max(1e-9, P)
        if self._meta_panel:
            styled_meta = []
            styled_meta.extend(
                [
                    f"[b #1b2d55]Pot[/]: {P:.2f} bb [dim](SPR {spr:.1f})[/]",
                    f"[b #1b2d55]Effective[/]: {node.effective_bb:.1f} bb",
                ]
            )
            if isinstance(bet, (int, float)):
                styled_meta.append(f"[b #1b2d55]Facing[/]: {float(bet):.2f} bb ({pct:.0f}% pot)")
            self._meta_panel.update("\n".join(styled_meta))

        hand_str = self._format_cards_colored(node.hero_cards)
        if self._hand_panel:
            self._hand_panel.update(
                f"[b #1b2d55]Hero[/]: {hand_str} [dim]({canonical_hand_abbrev(node.hero_cards)})[/]"
            )
        if self._board_panel:
            board_rows = self._format_board_rows(node.board)
            self._board_panel.update(f"[b #1b2d55]Board[/]\n{board_rows}")

        if self._status_panel:
            self._status_panel.update(
                f"[b #1b2d55]{node.street.title()} spotlight[/]\n[dim]Select the line that preserves edge.[/]"
            )

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
                elif any(term in key_lower for term in ("raise", "3-bet", "4-bet", "jam", "all-in")):
                    classes.append("option-raise")
                elif "bet" in key_lower:
                    classes.append("option-bet")
                else:
                    classes.append("option-bet")
                btn_label = k
                btn = Button(btn_label, id=f"opt-{i - 1}", classes=" ".join(classes))
                buttons.append(btn)
            if buttons:
                self._options_container.mount(*buttons)

    def show_step_feedback(self, _node: Node, chosen: Option, best: Option) -> None:
        correct = chosen.key == best.key
        ev_loss = best.ev - chosen.ev
        self._decisions_played += 1
        if correct:
            self._best_hits += 1
        self._total_ev_lost += max(0.0, ev_loss)
        lines = ["[b #1b2d55]Decision grade[/]"]
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
        if self._status_panel:
            tag = "[green]Nice read[/]" if correct else "[b #9f3b56]Learn point[/]"
            if getattr(chosen, "ends_hand", False):
                if self._total_hands and self._current_hand_index >= self._total_hands:
                    follow_up = "[dim]Hands complete — preparing your session summary…[/]"
                else:
                    follow_up = "[dim]Review the feedback below before the next hand.[/]"
            else:
                follow_up = "[dim]Review the feedback below before the next decision.[/]"
            self._status_panel.update(f"{tag}\n{follow_up}")
        if self._headline_label:
            self._headline_label.update(
                self._headline_for_state(
                    street=_node.street.title(),
                    stats=self._session_perf_fragment(),
                )
            )

    def show_summary(self, records: list[dict[str, Any]]) -> None:
        self._stop_preparing_animation()
        if self._headline_label:
            self._headline_label.update(
                self._headline_for_state(
                    info="[#2d3b62]Session summary[/]",
                    stats=self._session_perf_fragment(),
                )
            )
        if not records:
            if self._feedback_panel:
                self._feedback_panel.update("No hands answered.")
            if self._status_panel:
                self._status_panel.update(
                    "[b #1b2d55]Session wrapped[/]\n[dim]Start a fresh hand when you're ready.[/]"
                )
            return
        stats = summarize_records(records)
        total_ev_chosen = stats.total_ev_chosen
        total_ev_best = stats.total_ev_best
        total_ev_lost = stats.total_ev_lost
        avg_ev_lost = stats.avg_ev_lost
        hits = stats.hits
        hands_answered = stats.hands
        score_pct = stats.score_pct
        avg_loss_pct = stats.avg_loss_pct
        msg = (
            f"[b #1b2d55]Session summary[/]\n"
            f"Total EV lost: {total_ev_lost:.2f} bb\n"
            f"Avg EV lost/decision: {avg_ev_lost:.2f} bb\n"
            f"Avg EV lost (% pot): {avg_loss_pct:.2f}%\n"
            f"Hands answered: {hands_answered}\n"
            f"Best choices hit: {hits} ({(100.0 * hits / len(records)):.0f}%)\n"
            f"Total EV (chosen): {total_ev_chosen:.2f} bb\n"
            f"Total EV (best): {total_ev_best:.2f} bb\n"
            f"Score (0–100): {score_pct:.0f}"
        )
        if self._feedback_panel:
            self._feedback_panel.update(msg)
        if self._status_panel:
            self._status_panel.update(
                "[b #1b2d55]Great work[/]\n[dim]Check the summary, then dive back in for another run.[/]"
            )

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
        self._stop_preparing_animation()
        if self._headline_label:
            self._headline_label.update("Session stopped — press Start Fresh Hand to resume")
        if self._meta_panel:
            self._meta_panel.update("")
        if self._options_container:
            self._options_container.remove_children()
        if self._feedback_panel:
            self._feedback_panel.update("[dim]Session ended at your request.[/]")
        if self._status_panel:
            self._status_panel.update(
                "[b #1b2d55]Session on pause[/]\n[dim]Restart when you're ready to keep sharpening.[/]"
            )

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
