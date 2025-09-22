from __future__ import annotations

import math
import random
import secrets
import string
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

from ...core.formatting import format_option_label
from ...core.models import Option
from ...core.scoring import SummaryStats, summarize_records
from ...dynamic.cards import format_card_ascii
from ...dynamic.generator import Episode, Node, available_rival_styles
from ...dynamic.policy import options_for, resolve_for
from ...dynamic.seating import SeatRotation
from .concurrency import run_blocking
from .engine import SessionEngine
from .schemas import (
    ActionSnapshot,
    ChoiceResult,
    DecisionContract,
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


_STATE_HERO_NO_BET = "your_turn_no_bet"
_STATE_HERO_FACING_BET = "your_turn_facing_bet"
_STATE_OPPONENT = "opponent_turn"
_STATE_LOCKED = "locked"

_ACTION_ORDER: dict[str, list[str]] = {
    _STATE_HERO_NO_BET: ["check", "bet", "raise", "jam"],
    _STATE_HERO_FACING_BET: ["fold", "call", "raise", "jam"],
    _STATE_OPPONENT: [],
    _STATE_LOCKED: [],
}

_SIZE_PROMPTS: dict[str, str | None] = {
    _STATE_HERO_NO_BET: "Bet size",
    _STATE_HERO_FACING_BET: "Raise to",
    _STATE_OPPONENT: None,
    _STATE_LOCKED: None,
}


def _normalize_action_token(raw: str) -> str:
    token = (raw or "").strip().lower()
    if not token:
        return ""
    token = token.replace("3-bet", "3bet")
    if token in {"all-in", "allin", "jam", "shove"}:
        return "jam"
    if token.startswith("allin") or token.startswith("shove") or token.startswith("jam"):
        return "jam"
    if token.startswith("3bet"):
        return "raise"
    if token.startswith("raise"):
        return "raise"
    if token.startswith("bet"):
        return "bet"
    if token.startswith("call"):
        return "call"
    if token.startswith("check"):
        return "check"
    if token.startswith("fold"):
        return "fold"
    return token.split(" ", 1)[0]


def _canonical_action(option: Option) -> str:
    meta = option.meta or {}
    meta_action = _normalize_action_token(str(meta.get("action", "")))
    if meta_action:
        return meta_action
    return _normalize_action_token(option.key or option.why or "")


def _extract_numeric(meta: dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            if value > 0:
                return float(value)
    return None


def _detect_facing_bet(node: Node, options: Sequence[Option]) -> float:
    context = getattr(node, "context", {}) or {}
    bet_value = context.get("bet")
    if isinstance(bet_value, (int, float)) and bet_value > 0:
        return float(bet_value)

    candidates: list[float] = []
    for option in options:
        action = _canonical_action(option)
        meta = option.meta or {}
        if action in {"call", "fold", "raise", "jam"}:
            amount = _extract_numeric(meta, ("call_cost", "rival_bet", "rival_raise"))
            if amount is not None:
                candidates.append(amount)
    if candidates:
        return max(candidates)

    hand_state = context.get("hand_state") or {}
    hero_contrib = hand_state.get("hero_contrib")
    rival_contrib = hand_state.get("rival_contrib")
    try:
        hero_val = float(hero_contrib)
        rival_val = float(rival_contrib)
        diff = rival_val - hero_val
        if diff > 0:
            return diff
    except (TypeError, ValueError):
        pass
    return 0.0


def _resolve_betting_state(node: Node, options: Sequence[Option]) -> tuple[str, float]:
    if not options:
        return _STATE_LOCKED, 0.0
    context = getattr(node, "context", {}) or {}
    facing_token = str(context.get("facing") or "").strip().lower()
    facing_amount = _detect_facing_bet(node, options)
    if facing_token in {"bet", "lead"} or facing_amount > 0.0:
        return _STATE_HERO_FACING_BET, facing_amount
    if facing_token == "opponent" or node.actor != context.get("hero_seat", node.actor):
        return _STATE_OPPONENT, 0.0
    return _STATE_HERO_NO_BET, 0.0


def _filter_options_for_state(options: Sequence[Option], state: str) -> list[Option]:
    allowed_order = _ACTION_ORDER.get(state)
    if not allowed_order:
        return list(options)
    allowed = set(allowed_order)
    filtered = [opt for opt in options if _canonical_action(opt) in allowed]
    return filtered if filtered else list(options)


def _format_bb(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "0.00bb"
    return f"{value:.2f}bb"


def _derive_contract(state: SessionState, node: Node, options: Sequence[Option]) -> DecisionContract:
    betting_state, facing_amount = _resolve_betting_state(node, options)
    options = options or []
    pot_before = float(node.pot_bb) if isinstance(node.pot_bb, (int, float)) else 0.0
    pot_after_call = pot_before + facing_amount if facing_amount > 0 else pot_before
    opponent_seat = state.episodes[state.hand_index].rival_seat if state.episodes else ""

    if betting_state == _STATE_HERO_FACING_BET and facing_amount > 0:
        status_label = f"Your turn — facing {_format_bb(facing_amount)} into {_format_bb(pot_after_call)}"
        status_detail = f"Pot before action {_format_bb(pot_before)}"
    elif betting_state == _STATE_HERO_NO_BET:
        status_label = "Your turn — no bet"
        status_detail = f"Pot {_format_bb(pot_before)}"
    elif betting_state == _STATE_OPPONENT:
        status_label = f"{opponent_seat or 'Opponent'} to act"
        status_detail = f"Pot {_format_bb(pot_before)}"
    else:
        status_label = "Decision locked"
        status_detail = f"Pot {_format_bb(pot_before)}"

    size_prompt = _SIZE_PROMPTS.get(betting_state)
    legal_actions = list(_ACTION_ORDER.get(betting_state, []))

    contract = DecisionContract(
        state=betting_state,
        status_label=status_label,
        status_detail=status_detail,
        acting=node.actor,
        opponent=str(opponent_seat) if opponent_seat else None,
        facing_bet=facing_amount or None,
        pot_before=pot_before,
        pot_after_call=pot_after_call,
        size_prompt=size_prompt,
        legal_actions=legal_actions,
    )
    return contract


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

    async def create_session_async(self, config: SessionConfig) -> str:
        return await run_blocking(self.create_session, config)

    def get_node(self, session_id: str) -> NodeResponse:
        with self._lock:
            state = self._require_session(session_id)
            node = _ensure_active_node(state)
            if node is None:
                return NodeResponse(done=True, summary=_summary_payload(state.records))
            options = _ensure_options(state, node)
            payload = _node_payload(state, node, options)
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
            if next_node is None:
                next_payload = NodeResponse(done=True, summary=_summary_payload(state.records))
            else:
                next_options = _ensure_options(state, next_node)
                next_payload = NodeResponse(
                    done=False,
                    node=_node_payload(state, next_node, next_options),
                    options=_option_payloads(next_node, next_options),
                )

        feedback = FeedbackPayload(
            correct=chosen.key == best.key,
            ev_loss=record["ev_loss"],
            chosen=_snapshot(node, chosen_feedback),
            best=_snapshot(node, best),
            ended=ends,
        )
        return ChoiceResult(feedback=feedback, next_payload=next_payload)

    async def choose_async(self, session_id: str, choice_index: int) -> ChoiceResult:
        return await run_blocking(self.choose, session_id, choice_index)

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
        raw_options = options_for(node, state.engine.rng, state.config.mc_trials)
        betting_state, _ = _resolve_betting_state(node, raw_options)
        filtered = _filter_options_for_state(raw_options, betting_state)
        state.cached_options[cache_key] = filtered
        cached = filtered
    return [replace(opt) for opt in cached]


def _node_payload(state: SessionState, node: Node, options: Sequence[Option]) -> NodePayload:
    contract = _derive_contract(state, node, options)
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
        contract=contract,
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
