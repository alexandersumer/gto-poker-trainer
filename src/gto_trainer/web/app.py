from __future__ import annotations

import os

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, model_validator

from ..application import SessionConfig, SessionManager
from ..dynamic.generator import available_rival_styles


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


def _create_session(body: CreateSessionRequest) -> dict[str, str]:
    session_id = _manager.create_session(
        SessionConfig(
            hands=body.hands,
            mc_trials=body.mc,
            rival_style=body.rival_style,
        )
    )
    return {"session": session_id}


def _session_node(sid: str) -> dict[str, object]:
    try:
        payload = _manager.get_node(sid)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return payload.to_dict()


def _session_choice(sid: str, body: ChoiceRequest) -> dict[str, object]:
    try:
        result = _manager.choose(sid, body.choice)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return result.to_dict()


def _session_summary(sid: str) -> dict[str, object]:
    try:
        summary = _manager.summary(sid)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return summary.to_dict()


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
def create_session_v1(body: CreateSessionRequest) -> JSONResponse:
    return JSONResponse(_create_session(body))


@app.post("/api/session")
def create_session(body: CreateSessionRequest) -> JSONResponse:
    return JSONResponse(_create_session(body))


@_api_v1.get("/{sid}/node")
def get_node_v1(sid: str) -> JSONResponse:
    return JSONResponse(_session_node(sid))


@app.get("/api/session/{sid}/node")
def get_node(sid: str) -> JSONResponse:
    return JSONResponse(_session_node(sid))


@_api_v1.post("/{sid}/choose")
def post_choice_v1(sid: str, body: ChoiceRequest) -> JSONResponse:
    return JSONResponse(_session_choice(sid, body))


@app.post("/api/session/{sid}/choose")
def post_choice(sid: str, body: ChoiceRequest) -> JSONResponse:
    return JSONResponse(_session_choice(sid, body))


@_api_v1.get("/{sid}/summary")
def get_summary_v1(sid: str) -> JSONResponse:
    return JSONResponse(_session_summary(sid))


app.include_router(_api_v1)


@app.get("/api/session/{sid}/summary")
def get_summary(sid: str) -> JSONResponse:
    return JSONResponse(_session_summary(sid))


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
