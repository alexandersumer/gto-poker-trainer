from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, model_validator

from ..application import SessionConfig, SessionManager
from ..dynamic.generator import available_rival_styles
from ..services.session_v1 import (
    ChoiceResult,
    NodePayload,
    NodeResponse,
    SummaryPayload,
)


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


app = FastAPI(title="GTO Trainer")
_manager = SessionManager()
_api_v1 = APIRouter(prefix="/api/v1/session", tags=["session"])
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

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


def _is_hx(request: Request) -> bool:
    return request.headers.get(_HX_HEADER, "").lower() == "true"


def _card_token(raw: str) -> dict[str, str]:
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


def _card_tokens(cards: list[str] | None) -> list[dict[str, str]]:
    return [_card_token(card) for card in (cards or [])]


def _json_response(data: dict[str, object]) -> JSONResponse:
    response = JSONResponse(data)
    response.headers.setdefault("Vary", _HX_HEADER)
    return response


def _template_response(
    request: Request, template: str, context: dict[str, object], *, trigger: dict[str, str] | None = None
) -> Response:
    headers: dict[str, str] = {"Vary": _HX_HEADER}
    if trigger:
        headers["HX-Trigger"] = json.dumps(trigger)
    return templates.TemplateResponse(
        request,
        template,
        {**context, "request": request},
        headers=headers,
    )


def _render_node_fragment(request: Request, payload: NodeResponse, *, session_id: str | None = None) -> Response:
    if payload.done:
        summary = payload.summary or SummaryPayload(hands=0, decisions=0, hits=0, ev_lost=0.0, score=0.0)
        return _render_summary_fragment(request, summary)
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
    )
    options = payload.options or []
    trigger = {"sessionCreated": session_id} if session_id else None
    return _template_response(
        request,
        "session/node.html",
        {
            "node": node,
            "options": options,
            "hero_cards": _card_tokens(list(node.hero_cards)),
            "board_cards": _card_tokens(list(node.board_cards)),
        },
        trigger=trigger,
    )


def _render_summary_fragment(request: Request, summary: SummaryPayload) -> Response:
    return _template_response(request, "session/summary.html", {"summary": summary})


def _render_choice_fragment(request: Request, sid: str, result: ChoiceResult) -> Response:
    payload = result.next_payload
    node = payload.node
    options = payload.options or []
    context: dict[str, object] = {
        "feedback": result.feedback,
        "node": node,
        "options": options,
        "hero_cards": _card_tokens(list(node.hero_cards)) if node else [],
        "board_cards": _card_tokens(list(node.board_cards)) if node else [],
        "summary": payload.summary,
    }
    return _template_response(request, "session/choice.html", context, trigger={"sessionUpdated": sid})


def _create_session(body: CreateSessionRequest) -> dict[str, str]:
    session_id = _manager.create_session(
        SessionConfig(
            hands=body.hands,
            mc_trials=body.mc,
            rival_style=body.rival_style,
        )
    )
    return {"session": session_id}


def _session_node_payload(sid: str) -> NodeResponse:
    try:
        return _manager.get_node(sid)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


def _session_choice_result(sid: str, body: ChoiceRequest) -> ChoiceResult:
    try:
        return _manager.choose(sid, body.choice)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _session_summary_payload(sid: str) -> SummaryPayload:
    try:
        return _manager.summary(sid)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    try:
        from importlib.resources import files

        data_dir = files("gto_trainer.data")
        html = (data_dir / "web" / "index.html").read_text(encoding="utf-8")
        return html
    except Exception as exc:  # pragma: no cover - packaging edge
        return f"<html><body><h1>GTO Trainer</h1><p>Failed to load UI: {exc}</p></body></html>"


@_api_v1.post("")
def create_session_v1(request: Request, body: CreateSessionRequest) -> Response:
    data = _create_session(body)
    if _is_hx(request):
        payload = _session_node_payload(data["session"])
        return _render_node_fragment(request, payload, session_id=data["session"])
    return _json_response(data)


@app.post("/api/session")
def create_session(request: Request, body: CreateSessionRequest) -> Response:
    data = _create_session(body)
    if _is_hx(request):
        payload = _session_node_payload(data["session"])
        return _render_node_fragment(request, payload, session_id=data["session"])
    return _json_response(data)


@_api_v1.get("/{sid}/node")
def get_node_v1(request: Request, sid: str) -> Response:
    payload = _session_node_payload(sid)
    if _is_hx(request):
        return _render_node_fragment(request, payload)
    return _json_response(payload.to_dict())


@app.get("/api/session/{sid}/node")
def get_node(request: Request, sid: str) -> Response:
    payload = _session_node_payload(sid)
    if _is_hx(request):
        return _render_node_fragment(request, payload)
    return _json_response(payload.to_dict())


@_api_v1.post("/{sid}/choose")
def post_choice_v1(request: Request, sid: str, body: ChoiceRequest) -> Response:
    result = _session_choice_result(sid, body)
    if _is_hx(request):
        return _render_choice_fragment(request, sid, result)
    return _json_response(result.to_dict())


@app.post("/api/session/{sid}/choose")
def post_choice(request: Request, sid: str, body: ChoiceRequest) -> Response:
    result = _session_choice_result(sid, body)
    if _is_hx(request):
        return _render_choice_fragment(request, sid, result)
    return _json_response(result.to_dict())


@_api_v1.get("/{sid}/summary")
def get_summary_v1(request: Request, sid: str) -> Response:
    summary = _session_summary_payload(sid)
    if _is_hx(request):
        return _render_summary_fragment(request, summary)
    return _json_response(summary.to_dict())


app.include_router(_api_v1)


@app.get("/api/session/{sid}/summary")
def get_summary(request: Request, sid: str) -> Response:
    summary = _session_summary_payload(sid)
    if _is_hx(request):
        return _render_summary_fragment(request, summary)
    return _json_response(summary.to_dict())


def _custom_openapi() -> dict[str, object]:  # pragma: no cover - exercised via docs
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi  # type: ignore[assignment]
app.openapi_schema = None


def main() -> None:  # pragma: no cover - runner
    import uvicorn

    host = os.environ.get("BIND", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, factory=False)


if __name__ == "__main__":  # pragma: no cover
    main()
