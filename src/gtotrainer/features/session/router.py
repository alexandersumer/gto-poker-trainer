from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, model_validator

from ...dynamic.generator import available_rival_styles
from .schemas import ChoiceResult, NodePayload, NodeResponse, SummaryPayload
from .service import SessionConfig, SessionManager

__all__ = ["ChoiceRequest", "CreateSessionRequest", "create_session_routers"]

_HX_HEADER = "HX-Request"

_SUIT_CLASS = {
    "S": "s",
    "H": "h",
    "D": "d",
    "C": "c",
}

_SUIT_SYMBOL = {
    "S": "♠",
    "H": "♥",
    "D": "♦",
    "C": "♣",
}


class CreateSessionRequest(BaseModel):
    hands: int | None = None
    mc: int | None = None
    rival_style: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: dict[str, object]) -> dict[str, object]:
        if not isinstance(data, dict):
            return data
        cleaned: dict[str, object] = dict(data)
        for field in ("hands", "mc"):
            value = cleaned.get(field)
            if value in (None, ""):
                cleaned[field] = None
                continue
            if isinstance(value, str):
                try:
                    cleaned[field] = int(value)
                except ValueError:
                    cleaned[field] = None
        return cleaned

    @model_validator(mode="after")
    def _normalize(self) -> CreateSessionRequest:
        hands = self.hands if self.hands is not None else 1
        if hands < 1:
            hands = 1
        mc = self.mc if self.mc is not None else 120
        if mc < 40:
            mc = 40
        styles = available_rival_styles()
        style = (self.rival_style or "balanced").strip().lower()
        if style not in styles:
            style = "balanced"
        self.hands = hands
        self.mc = mc
        self.rival_style = style
        return self


class ChoiceRequest(BaseModel):
    choice: int


class _SessionController:
    def __init__(self, manager: SessionManager, templates: Jinja2Templates) -> None:
        self.manager = manager
        self.templates = templates

    # ------------------------------------------------------------------ helpers
    def _is_hx(self, request: Request) -> bool:
        return request.headers.get(_HX_HEADER, "").lower() == "true"

    def _json_response(self, data: dict[str, object]) -> JSONResponse:
        response = JSONResponse(data)
        response.headers.setdefault("Vary", _HX_HEADER)
        return response

    def _template_response(
        self,
        request: Request,
        template: str,
        context: dict[str, object],
        *,
        trigger: dict[str, str] | None = None,
    ) -> Response:
        headers: dict[str, str] = {"Vary": _HX_HEADER}
        if trigger:
            headers["HX-Trigger"] = json.dumps(trigger)
        return self.templates.TemplateResponse(
            request,
            template,
            {**context, "request": request},
            headers=headers,
        )

    def _card_token(self, raw: str) -> dict[str, str]:
        token = (raw or "").strip().upper()
        if not token:
            return {"rank": "?", "css": "s", "symbol": "♠"}
        rank = token[:-1]
        suit = token[-1]
        if len(token) == 2:
            rank = token[0]
            suit = token[1]
        css = _SUIT_CLASS.get(suit, "s")
        symbol = _SUIT_SYMBOL.get(suit, "♠")
        return {"rank": rank, "css": css, "symbol": symbol}

    def _card_tokens(self, cards: list[str] | None) -> list[dict[str, str]]:
        return [self._card_token(card) for card in (cards or [])]

    def _summary_fragment(self, request: Request, summary: SummaryPayload) -> Response:
        return self._template_response(request, "session/summary.html", {"summary": summary})

    def _node_fragment(
        self,
        request: Request,
        payload: NodeResponse,
        *,
        session_id: str | None = None,
    ) -> Response:
        if payload.done:
            summary = payload.summary or SummaryPayload(hands=0, decisions=0, hits=0, ev_lost=0.0, score=0.0)
            return self._summary_fragment(request, summary)
        node = payload.node or NodePayload(
            street="preflop",
            description="",
            pot_bb=0.0,
            effective_bb=0.0,
            hero_cards=[],
            board_cards=[],
            actor="",
            hand_no=0,
            total_hands=0,
            context=None,
        )
        options = payload.options or []
        trigger = {"sessionCreated": session_id} if session_id else None
        return self._template_response(
            request,
            "session/node.html",
            {
                "node": node,
                "options": options,
                "hero_cards": self._card_tokens(list(node.hero_cards)),
                "board_cards": self._card_tokens(list(node.board_cards)),
            },
            trigger=trigger,
        )

    def _choice_fragment(self, request: Request, sid: str, result: ChoiceResult) -> Response:
        payload = result.next_payload
        node = payload.node
        context: dict[str, object] = {
            "feedback": result.feedback,
            "node": node,
            "options": payload.options or [],
            "hero_cards": self._card_tokens(list(node.hero_cards)) if node else [],
            "board_cards": self._card_tokens(list(node.board_cards)) if node else [],
            "summary": payload.summary,
        }
        return self._template_response(request, "session/choice.html", context, trigger={"sessionUpdated": sid})

    # ------------------------------------------------------------------ actions
    async def create(self, request: Request, body: CreateSessionRequest) -> Response:
        session_id = await self.manager.create_session_async(
            SessionConfig(hands=body.hands, mc_trials=body.mc, rival_style=body.rival_style)
        )
        if self._is_hx(request):
            payload = await self.manager.get_node_async(session_id)
            return self._node_fragment(request, payload, session_id=session_id)
        return self._json_response({"session": session_id})

    async def node(self, request: Request, sid: str) -> Response:
        try:
            payload = await self.manager.get_node_async(sid)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        if self._is_hx(request):
            return self._node_fragment(request, payload)
        return self._json_response(payload.to_dict())

    async def choose(self, request: Request, sid: str, body: ChoiceRequest) -> Response:
        try:
            result = await self.manager.choose_async(sid, body.choice)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if self._is_hx(request):
            return self._choice_fragment(request, sid, result)
        return self._json_response(result.to_dict())

    async def summary(self, request: Request, sid: str) -> Response:
        try:
            summary = await self.manager.summary_async(sid)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        if self._is_hx(request):
            return self._summary_fragment(request, summary)
        return self._json_response(summary.to_dict())


def create_session_routers(
    manager: SessionManager,
    templates: Jinja2Templates,
) -> tuple[APIRouter, APIRouter]:
    controller = _SessionController(manager, templates)

    router_v1 = APIRouter(prefix="/api/v1/session", tags=["session"])
    router_legacy = APIRouter(prefix="/api/session", tags=["session-legacy"])

    @router_v1.post("")
    async def create_session(request: Request, body: CreateSessionRequest) -> Response:
        return await controller.create(request, body)

    @router_legacy.post("")
    async def create_session_legacy(request: Request, body: CreateSessionRequest) -> Response:
        return await controller.create(request, body)

    @router_v1.get("/{sid}/node")
    async def get_node(request: Request, sid: str) -> Response:
        return await controller.node(request, sid)

    @router_legacy.get("/{sid}/node")
    async def get_node_legacy(request: Request, sid: str) -> Response:
        return await controller.node(request, sid)

    @router_v1.post("/{sid}/choose")
    async def post_choice(request: Request, sid: str, body: ChoiceRequest) -> Response:
        return await controller.choose(request, sid, body)

    @router_legacy.post("/{sid}/choose")
    async def post_choice_legacy(request: Request, sid: str, body: ChoiceRequest) -> Response:
        return await controller.choose(request, sid, body)

    @router_v1.get("/{sid}/summary")
    async def get_summary(request: Request, sid: str) -> Response:
        return await controller.summary(request, sid)

    @router_legacy.get("/{sid}/summary")
    async def get_summary_legacy(request: Request, sid: str) -> Response:
        return await controller.summary(request, sid)

    return router_v1, router_legacy
