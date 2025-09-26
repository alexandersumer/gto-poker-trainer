from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ActionSnapshot",
    "ChoiceResult",
    "FeedbackPayload",
    "NodePayload",
    "NodeResponse",
    "OptionPayload",
    "SummaryPayload",
]


class _APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)


class OptionPayload(_APIModel):
    key: str
    label: str
    ev: float
    why: str
    ends_hand: bool
    gto_freq: float | None = None
    meta: dict[str, Any] | None = None


class ActionSnapshot(_APIModel):
    key: str
    label: str
    ev: float
    why: str
    gto_freq: float | None = None
    resolution_note: str | None = None
    meta: dict[str, Any] | None = None


class NodePayload(_APIModel):
    street: str
    description: str
    pot_bb: float
    effective_bb: float
    hero_cards: list[str]
    board_cards: list[str]
    actor: str
    hand_no: int
    total_hands: int
    context: dict[str, Any] | None = None


class SummaryPayload(_APIModel):
    hands: int
    decisions: int
    hits: int
    ev_lost: float
    score: float
    avg_ev_lost: float
    avg_loss_pct: float
    accuracy_pct: float
    accuracy_points: float


class FeedbackPayload(_APIModel):
    correct: bool
    ev_loss: float
    accuracy: float
    cumulative_ev_lost: float
    cumulative_accuracy: float
    decisions: int
    chosen: ActionSnapshot
    best: ActionSnapshot
    ended: bool


class NodeResponse(_APIModel):
    done: bool
    node: NodePayload | None = None
    options: list[OptionPayload] | None = None
    summary: SummaryPayload | None = None


class ChoiceResult(_APIModel):
    feedback: FeedbackPayload
    next_payload: NodeResponse = Field(..., alias="next")
