from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from gtotrainer.features.session import SessionManager
from gtotrainer.features.session.router import create_session_routers


def _client() -> tuple[TestClient, SessionManager]:
    manager = SessionManager()
    project_root = Path(__file__).resolve().parents[1]
    templates = Jinja2Templates(directory=str(project_root / "src" / "gtotrainer" / "web" / "templates"))
    router_v1, _ = create_session_routers(manager, templates)

    app = FastAPI()
    app.include_router(router_v1)
    return TestClient(app), manager


def _play_session(client: TestClient, sid: str) -> None:
    while True:
        node_resp = client.get(f"/api/v1/session/{sid}/node")
        node_payload = node_resp.json()
        if node_payload["done"]:
            break
        client.post(
            f"/api/v1/session/{sid}/choose",
            json={"choice": 0},
        )


def test_create_session_normalizes_style_and_returns_json_summary() -> None:
    client, manager = _client()

    response = client.post("/api/v1/session", json={"hands": 1, "mc": 12, "rival_style": "Loose"})
    data = response.json()
    sid = data["session"]

    # Ensure style normalized and mc minimum enforced.
    state = manager._sessions[sid]
    assert state.config.mc_trials == 40  # router raises low mc values
    assert state.engine.rival_style == "balanced"

    _play_session(client, sid)

    summary_json = client.get(f"/api/v1/session/{sid}/summary").json()
    assert set(summary_json) >= {"avg_ev_lost", "avg_loss_pct", "accuracy_pct", "score", "hands", "decisions"}


def test_create_session_hx_response_includes_trigger_header() -> None:
    client, _ = _client()
    response = client.post(
        "/api/v1/session",
        headers={"HX-Request": "true"},
        json={"hands": 1, "mc": 50},
    )
    assert response.status_code == 200
    trigger_header = response.headers.get("HX-Trigger")
    assert trigger_header is not None
    payload = json.loads(trigger_header)
    assert "sessionCreated" in payload
    assert "hx-node" in response.text
