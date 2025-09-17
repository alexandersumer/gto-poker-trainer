from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, model_validator

from ..application import SessionConfig, SessionManager


class CreateSessionRequest(BaseModel):
    hands: int | None = None
    mc: int | None = None

    @model_validator(mode="after")
    def _normalize(self) -> CreateSessionRequest:
        hands = self.hands if self.hands is not None else 1
        if hands < 1:
            hands = 1
        mc = self.mc if self.mc is not None else 120
        if mc < 40:
            mc = 40
        self.hands = hands
        self.mc = mc
        return self


class ChoiceRequest(BaseModel):
    choice: int


app = FastAPI(title="GTO Poker Trainer")
_manager = SessionManager()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    try:
        from importlib.resources import files

        data_dir = files("gto_poker_trainer_cli.data")
        html = (data_dir / "web" / "index.html").read_text(encoding="utf-8")
        return html
    except Exception as exc:  # pragma: no cover - packaging edge
        return f"<html><body><h1>GTO Poker Trainer</h1><p>Failed to load UI: {exc}</p></body></html>"


@app.post("/api/session")
def create_session(body: CreateSessionRequest) -> JSONResponse:
    session_id = _manager.create_session(SessionConfig(hands=body.hands, mc_trials=body.mc))
    return JSONResponse({"session": session_id})


@app.get("/api/session/{sid}/node")
def get_node(sid: str) -> JSONResponse:
    try:
        payload = _manager.get_node(sid)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return JSONResponse(payload.to_dict())


@app.post("/api/session/{sid}/choose")
def post_choice(sid: str, body: ChoiceRequest) -> JSONResponse:
    try:
        result = _manager.choose(sid, body.choice)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(result.to_dict())


@app.get("/api/session/{sid}/summary")
def get_summary(sid: str) -> JSONResponse:
    try:
        summary = _manager.summary(sid)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return JSONResponse(summary.to_dict())


def main() -> None:  # pragma: no cover - runner
    import uvicorn

    host = os.environ.get("BIND", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("gto_poker_trainer_cli.web.app:app", host=host, port=port, factory=False)


if __name__ == "__main__":  # pragma: no cover
    main()
