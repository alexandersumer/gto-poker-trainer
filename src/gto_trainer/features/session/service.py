from __future__ import annotations

import random
import secrets
import string
import threading
from dataclasses import dataclass, field, replace
from typing import Any

from ...core.formatting import format_option_label
from ...core.models import Option
from ...core.scoring import SummaryStats, summarize_records
from ...dynamic.cards import format_card_ascii
from ...dynamic.generator import Episode, Node, available_rival_styles
from ...dynamic.policy import options_for, resolve_for
from ...dynamic.seating import SeatRotation
from .engine import SessionEngine
from .schemas import (
    ActionSnapshot,
    ChoiceResult,
    FeedbackPayload,
    NodePayload,
    NodeResponse,
    OptionPayload,
    SummaryPayload,
)

__all__ = [
    "SessionConfig",
    "SessionManager",
    "SessionState",
    "_ensure_active_node",
    "_ensure_options",
    "_summary_payload",
]


def _card_strings(cards: list[int]) -> list[str]:
    return [format_card_ascii(card, upper=True) for card in cards]


@dataclass(frozen=True)
class SessionConfig:
    """Configuration for a training session."""

    hands: int
    mc_trials: int
    seed: int | None = None
    rival_style: str = "balanced"


@dataclass
class SessionState:
    config: SessionConfig
    episodes: list[Episode]
    engine: SessionEngine
    hand_index: int = 0
    current_index: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)
    cached_options: dict[int, list[Option]] = field(default_factory=dict)


class SessionManager:
    """Owns session lifecycle independent of the presentation layer."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def create_session(self, config: SessionConfig) -> str:
        seed = config.seed or secrets.SystemRandom().getrandbits(32)
        rng = random.Random(seed)
        rotation = SeatRotation()
        style = (config.rival_style or "balanced").strip().lower()
        if style not in available_rival_styles():
            style = "balanced"
        engine = SessionEngine(rng=rng, rotation=rotation, rival_style=style)
        first_episode = engine.build_episode(0)
        session_id = _sid()
        normalized_config = SessionConfig(
            hands=max(1, config.hands),
            mc_trials=max(10, config.mc_trials),
            seed=seed,
            rival_style=style,
        )
        state = SessionState(
            config=normalized_config,
            episodes=[first_episode],
            engine=engine,
        )
        with self._lock:
            self._sessions[session_id] = state
        return session_id

    def get_node(self, session_id: str) -> NodeResponse:
        with self._lock:
            state = self._require_session(session_id)
            node = _ensure_active_node(state)
            if node is None:
                return NodeResponse(done=True, summary=_summary_payload(state.records))
            payload = _node_payload(state, node)
            options = _ensure_options(state, node)
            return NodeResponse(done=False, node=payload, options=_option_payloads(node, options))

    def choose(self, session_id: str, choice_index: int) -> ChoiceResult:
        with self._lock:
            state = self._require_session(session_id)
            node = _ensure_active_node(state)
            if node is None:
                raise ValueError("session already complete")
            options = _ensure_options(state, node)
            if not (0 <= choice_index < len(options)):
                raise ValueError("choice index out of range")
            chosen = options[choice_index]
            best = options[_best_index(options)]
            worst = min(options, key=lambda opt: opt.ev)
            resolution = resolve_for(node, chosen, state.engine.rng)
            chosen_feedback = replace(chosen)
            if resolution.note:
                chosen_feedback.resolution_note = resolution.note
            if resolution.hand_ended:
                chosen_feedback.ends_hand = True
            record = {
                "street": node.street,
                "chosen_key": chosen.key,
                "chosen_ev": chosen.ev,
                "best_key": best.key,
                "best_ev": best.ev,
                "worst_ev": worst.ev,
                "room_ev": max(1e-9, best.ev - worst.ev),
                "ev_loss": best.ev - chosen.ev,
                "hand_ended": getattr(chosen_feedback, "ends_hand", False),
                "resolution_note": chosen_feedback.resolution_note,
                "hand_index": state.hand_index,
                "pot_bb": float(getattr(node, "pot_bb", 0.0)),
            }
            state.records.append(record)
            ends = getattr(chosen_feedback, "ends_hand", False)
            episode = state.episodes[state.hand_index]
            state.current_index = len(episode.nodes) if ends else state.current_index + 1
            state.cached_options.clear()
            next_node = _ensure_active_node(state)
            next_payload = (
                NodeResponse(done=True, summary=_summary_payload(state.records))
                if next_node is None
                else NodeResponse(
                    done=False,
                    node=_node_payload(state, next_node),
                    options=_option_payloads(next_node, _ensure_options(state, next_node)),
                )
            )

        feedback = FeedbackPayload(
            correct=chosen.key == best.key,
            ev_loss=record["ev_loss"],
            chosen=_snapshot(node, chosen_feedback),
            best=_snapshot(node, best),
            ended=ends,
        )
        return ChoiceResult(feedback=feedback, next_payload=next_payload)

    def summary(self, session_id: str) -> SummaryPayload:
        with self._lock:
            state = self._require_session(session_id)
            return _summary_payload(state.records)

    def _require_session(self, session_id: str) -> SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            raise KeyError(f"session '{session_id}' not found")
        return state


def _sid(length: int = 10) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _ensure_episode(state: SessionState) -> None:
    if state.hand_index >= len(state.episodes):
        next_index = len(state.episodes)
        state.episodes.append(state.engine.build_episode(next_index))


def _ensure_active_node(state: SessionState) -> Node | None:
    while True:
        _ensure_episode(state)
        episode = state.episodes[state.hand_index]
        if state.current_index < len(episode.nodes):
            return episode.nodes[state.current_index]
        if state.hand_index + 1 >= state.config.hands:
            return None
        state.hand_index += 1
        state.current_index = 0


def _ensure_options(state: SessionState, node: Node) -> list[Option]:
    cache_key = id(node)
    cached = state.cached_options.get(cache_key)
    if cached is None:
        options = options_for(node, state.engine.rng, state.config.mc_trials)
        state.cached_options[cache_key] = options
        cached = options
    return [replace(opt) for opt in cached]


def _node_payload(state: SessionState, node: Node) -> NodePayload:
    return NodePayload(
        street=node.street,
        description=node.description,
        pot_bb=node.pot_bb,
        effective_bb=node.effective_bb,
        hero_cards=_card_strings(node.hero_cards),
        board_cards=_card_strings(node.board),
        actor=node.actor,
        hand_no=state.hand_index + 1,
        total_hands=state.config.hands,
    )


def _best_index(opts: list[Option]) -> int:
    return max(range(len(opts)), key=lambda idx: opts[idx].ev)


def _option_payloads(node: Node, options: list[Option]) -> list[OptionPayload]:
    return [
        OptionPayload(
            key=opt.key,
            label=format_option_label(node, opt),
            ev=opt.ev,
            why=opt.why,
            ends_hand=getattr(opt, "ends_hand", False),
            gto_freq=getattr(opt, "gto_freq", None),
        )
        for opt in options
    ]


def _snapshot(node: Node, option: Option) -> ActionSnapshot:
    return ActionSnapshot(
        key=option.key,
        label=format_option_label(node, option),
        ev=option.ev,
        why=option.why,
        gto_freq=getattr(option, "gto_freq", None),
        resolution_note=getattr(option, "resolution_note", None),
    )


def _summary_payload(records: list[dict[str, Any]]) -> SummaryPayload:
    stats: SummaryStats = summarize_records(records)
    return SummaryPayload(
        hands=stats.hands,
        decisions=stats.decisions,
        hits=stats.hits,
        ev_lost=stats.total_ev_lost,
        score=stats.score_pct,
    )
