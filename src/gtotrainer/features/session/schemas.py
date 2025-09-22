from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ActionSnapshot",
    "DecisionContract",
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


class DecisionContract(_APIModel):
    state: Literal["your_turn_no_bet", "your_turn_facing_bet", "opponent_turn", "locked"]
    status_label: str
    status_detail: str | None = None
    acting: str
    opponent: str | None = None
    facing_bet: float | None = None
    pot_before: float | None = None
    pot_after_call: float | None = None
    size_prompt: str | None = None
    legal_actions: list[str] = Field(default_factory=list)


class ActionSnapshot(_APIModel):
    key: str
    label: str
    ev: float
    why: str
    gto_freq: float | None = None
    resolution_note: str | None = None


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
    contract: DecisionContract | None = None


class SummaryPayload(_APIModel):
    hands: int
    decisions: int
    hits: int
    ev_lost: float
    score: float


class FeedbackPayload(_APIModel):
    correct: bool
    ev_loss: float
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
