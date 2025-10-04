from __future__ import annotations

import logging
import random
import secrets
import string
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from ...core.ev import effective_option_ev
from ...core.formatting import format_option_label
from ...core.models import Option
from ...core.scoring import SummaryStats, decision_accuracy, summarize_records
from ...dynamic.cards import format_card_ascii
from ...dynamic.generator import Episode, Node, available_rival_styles
from ...dynamic.policy import options_for, resolve_for
from ...dynamic.seating import SeatRotation
from .concurrency import run_blocking
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

logger = logging.getLogger(__name__)


def _card_strings(cards: list[int]) -> list[str]:
    return [format_card_ascii(card, upper=True) for card in cards]


def _view_context(node: Node) -> dict[str, Any]:
    """Expose a sanitised subset of the engine context for the UI."""

    raw = getattr(node, "context", {}) or {}
    context: dict[str, Any] = {}

    facing = raw.get("facing")
    if isinstance(facing, str) and facing.strip():
        context["facing"] = facing.strip().lower()

    bet = raw.get("bet")
    if isinstance(bet, (int, float)):
        context["bet"] = float(bet)

    open_size = raw.get("open_size")
    if isinstance(open_size, (int, float)):
        context["open_size"] = float(open_size)

    hero_seat = raw.get("hero_seat")
    if isinstance(hero_seat, str) and hero_seat.strip():
        context["hero_seat"] = hero_seat.strip().upper()

    rival_seat = raw.get("rival_seat")
    if isinstance(rival_seat, str) and rival_seat.strip():
        context["rival_seat"] = rival_seat.strip().upper()

    actor_seat = node.actor.strip().upper() if isinstance(node.actor, str) else ""
    if actor_seat:
        context["actor_seat"] = actor_seat
    if actor_seat and "hero_seat" in context:
        context["actor_role"] = "hero" if actor_seat == context["hero_seat"] else "rival"

    return context


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
    total_ev_lost: float = 0.0
    accuracy_points: float = 0.0
    decisions: int = 0


class SessionManager:
    """Owns session lifecycle independent of the presentation layer."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def create_session(self, config: SessionConfig) -> str:
        seed = config.seed if config.seed is not None else secrets.SystemRandom().getrandbits(32)
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

    async def create_session_async(self, config: SessionConfig) -> str:
        return await run_blocking(self.create_session, config)

    def get_node(self, session_id: str) -> NodeResponse:
        with self._lock:
            state = self._require_session(session_id)
            node = _ensure_active_node(state)
            if node is None:
                return NodeResponse(done=True, summary=_summary_payload(state.records))
            payload = _node_payload(state, node)
            options = _ensure_options(state, node)
            return NodeResponse(done=False, node=payload, options=_option_payloads(node, options))

    async def get_node_async(self, session_id: str) -> NodeResponse:
        return await run_blocking(self.get_node, session_id)

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
            worst = min(options, key=lambda opt: _effective_ev(opt))
            resolution = resolve_for(node, chosen, state.engine.rng)
            chosen_feedback = replace(chosen)
            if resolution.note:
                chosen_feedback.resolution_note = resolution.note
            if resolution.hand_ended:
                chosen_feedback.ends_hand = True
            chosen_ev_eff = _effective_ev(chosen)
            best_ev_eff = _effective_ev(best)
            worst_ev_eff = _effective_ev(worst)
            chosen_out_flag = _out_of_policy(chosen)
            best_out_flag = _out_of_policy(best)
            ev_gap = best_ev_eff - chosen_ev_eff
            if chosen_out_flag is True and best_out_flag is not True and ev_gap < 0.0:
                ev_loss = -ev_gap
            else:
                ev_loss = ev_gap
            record = {
                "street": node.street,
                "chosen_key": chosen.key,
                "chosen_ev": chosen_ev_eff,
                "best_key": best.key,
                "best_ev": best_ev_eff,
                "worst_ev": worst_ev_eff,
                "room_ev": max(1e-9, best_ev_eff - worst_ev_eff),
                "ev_loss": ev_loss,
                "chosen_cfr_ev": chosen.ev,
                "best_cfr_ev": best.ev,
                "hand_ended": getattr(chosen_feedback, "ends_hand", False),
                "resolution_note": chosen_feedback.resolution_note,
                "hand_index": state.hand_index,
                "pot_bb": float(getattr(node, "pot_bb", 0.0)),
                "chosen_out_of_policy": chosen_out_flag,
                "best_out_of_policy": best_out_flag,
                "chosen_freq": getattr(chosen, "gto_freq", None),
                "best_freq": getattr(best, "gto_freq", None),
            }
            state.records.append(record)
            accuracy_credit = decision_accuracy(record)
            state.total_ev_lost += record["ev_loss"]
            state.accuracy_points += accuracy_credit
            state.decisions += 1
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
            accuracy=accuracy_credit,
            cumulative_ev_lost=state.total_ev_lost,
            cumulative_accuracy=state.accuracy_points,
            decisions=state.decisions,
        )
        return ChoiceResult(feedback=feedback, next_payload=next_payload)

    async def choose_async(self, session_id: str, choice_index: int) -> ChoiceResult:
        return await run_blocking(self.choose, session_id, choice_index)

    def drive_session(
        self,
        session_id: str,
        chooser: Callable[[Node, Sequence[Option], random.Random], int],
        *,
        cleanup: bool = False,
    ) -> list[dict[str, Any]]:
        """Play out a session by delegating option selection to ``chooser``.

        Returns a copy of the recorded hands for downstream analysis.  The
        ``chooser`` callback receives the active node, its available options and
        the session RNG, and must return the index of the chosen option.
        """

        while True:
            with self._lock:
                state = self._require_session(session_id)
                node = _ensure_active_node(state)
                if node is None:
                    records = [dict(record) for record in state.records]
                    if cleanup:
                        self._sessions.pop(session_id, None)
                    logger.debug("drive_session completed", extra={"session_id": session_id, "records": len(records)})
                    return records
                options = list(_ensure_options(state, node))
                rng = state.engine.rng

            choice_index = chooser(node, tuple(options), rng)
            self.choose(session_id, choice_index)

    def summary(self, session_id: str) -> SummaryPayload:
        with self._lock:
            state = self._require_session(session_id)
            return _summary_payload(state.records)

    async def summary_async(self, session_id: str) -> SummaryPayload:
        return await run_blocking(self.summary, session_id)

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


def _effective_ev(option: Option) -> float:
    return effective_option_ev(option)


def _node_payload(state: SessionState, node: Node) -> NodePayload:
    context = _view_context(node)
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
        context=context or None,
    )


_POLICY_FREQ_EPSILON = 1e-3


def _policy_weight(option: Option) -> float | None:
    freq = getattr(option, "gto_freq", None)
    value: float | None
    try:
        value = float(freq) if freq is not None else None
    except (TypeError, ValueError):
        value = None

    if value is None:
        meta = getattr(option, "meta", None)
        if isinstance(meta, Mapping):
            mix = meta.get("solver_mix")
            if isinstance(mix, Mapping) and mix:
                weights: list[float] = []
                for raw in mix.values():
                    try:
                        weights.append(float(raw))
                    except (TypeError, ValueError):
                        continue
                if weights:
                    value = max(weights)

    if value is None:
        return None
    if value != value or value < 0.0:  # NaN or negative
        return None
    return value


def _best_index(opts: list[Option]) -> int:
    """Return the index of the best in-policy action, preferring higher EV and frequency.

    When in-policy actions exist (gto_freq > 0.1%), only those are considered.
    Actions with explicit out_of_policy flags are always excluded from best selection.
    Falls back to all non-flagged actions only when solver frequency data is missing.
    """
    if not opts:
        raise ValueError("options list cannot be empty")

    policy_weights = [_policy_weight(opt) for opt in opts]

    # First, filter out explicitly flagged out-of-policy actions
    not_explicitly_oop = [
        idx for idx, opt in enumerate(opts)
        if not (isinstance(getattr(opt, "meta", None), Mapping) and getattr(opt, "meta").get("out_of_policy") is True)
    ]

    # Among remaining actions, prefer those with in-policy frequencies
    eligible = [
        idx for idx in not_explicitly_oop
        if policy_weights[idx] and policy_weights[idx] > _POLICY_FREQ_EPSILON
    ]

    # Determine comparison set: in-policy actions if available, else all non-flagged actions
    if eligible:
        target_indices = eligible
    elif not_explicitly_oop:
        logger.debug(
            "No in-policy frequencies found; using all non-flagged actions",
            extra={"policy_weights": policy_weights, "option_keys": [opt.key for opt in opts]},
        )
        target_indices = not_explicitly_oop
    else:
        # All actions are explicitly out-of-policy; fall back to all (shouldn't happen)
        logger.warning(
            "All actions flagged as out-of-policy; falling back to all actions",
            extra={"option_keys": [opt.key for opt in opts]},
        )
        target_indices = list(range(len(opts)))

    def _key(idx: int) -> tuple[float, float]:
        ev = _effective_ev(opts[idx])
        weight = policy_weights[idx] if policy_weights[idx] is not None else -1.0
        return ev, weight

    return max(target_indices, key=_key)


def _out_of_policy(option: Option) -> bool | None:
    meta = getattr(option, "meta", None)
    if isinstance(meta, Mapping):
        flag = meta.get("out_of_policy")
        if isinstance(flag, bool):
            return flag
    freq = _policy_weight(option)
    if freq is None:
        return None
    return freq <= _POLICY_FREQ_EPSILON


def _option_payloads(node: Node, options: list[Option]) -> list[OptionPayload]:
    return [
        OptionPayload(
            key=opt.key,
            label=format_option_label(node, opt),
            ev=_effective_ev(opt),
            why=opt.why,
            ends_hand=getattr(opt, "ends_hand", False),
            gto_freq=getattr(opt, "gto_freq", None),
            out_of_policy=_out_of_policy(opt),
        )
        for opt in options
    ]


def _snapshot(node: Node, option: Option) -> ActionSnapshot:
    return ActionSnapshot(
        key=option.key,
        label=format_option_label(node, option),
        ev=_effective_ev(option),
        why=option.why,
        gto_freq=getattr(option, "gto_freq", None),
        out_of_policy=_out_of_policy(option),
        resolution_note=getattr(option, "resolution_note", None),
    )


def _summary_payload(records: list[dict[str, Any]]) -> SummaryPayload:
    stats: SummaryStats = summarize_records(records)
    accuracy_pct = stats.accuracy_pct
    return SummaryPayload(
        hands=stats.hands,
        decisions=stats.decisions,
        hits=stats.hits,
        ev_lost=stats.total_ev_lost,
        score=stats.score_pct,
        avg_ev_lost=stats.avg_ev_lost,
        avg_loss_pct=stats.avg_loss_pct,
        accuracy_pct=accuracy_pct,
        accuracy_points=stats.accuracy_points,
    )
