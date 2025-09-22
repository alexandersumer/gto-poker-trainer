from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ActionSnapshot",
    "ChoiceResult",
    "ActionEVBreakdown",
    "NodeAnalysisPayload",
    "FeedbackPayload",
    "NodePayload",
    "NodeResponse",
    "OptionAnalysisPayload",
    "OptionPayload",
    "SummaryPayload",
    "VillainRangeEntry",
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


class ActionSnapshot(_APIModel):
    key: str
    label: str
    ev: float
    why: str
    gto_freq: float | None = None
    resolution_note: str | None = None


class VillainRangeEntry(_APIModel):
    combo: str
    weight: float | None = None


class ActionEVBreakdown(_APIModel):
    fold_pct: float | None = None
    continue_pct: float | None = None
    fold_term: float | None = None
    continue_term: float | None = None
    hero_equity_vs_continue: float | None = None
    pot_before: float | None = None
    pot_if_called: float | None = None
    hero_invest: float | None = None
    villain_invest: float | None = None
    villain_continue_total: int | None = None
    villain_continue_sample: list[VillainRangeEntry] | None = None


class OptionAnalysisPayload(_APIModel):
    key: str
    label: str
    ev: float
    ev_delta: float
    is_best: bool
    breakdown: ActionEVBreakdown | None = None


class NodeAnalysisPayload(_APIModel):
    best_key: str | None
    options: list[OptionAnalysisPayload]


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
    analysis: NodeAnalysisPayload | None = None
    summary: SummaryPayload | None = None


class ChoiceResult(_APIModel):
    feedback: FeedbackPayload
    next_payload: NodeResponse = Field(..., alias="next")
