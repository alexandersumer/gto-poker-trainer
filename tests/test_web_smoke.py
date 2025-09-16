from __future__ import annotations

from fastapi.testclient import TestClient

from gto_poker_trainer_cli.web.app import app


def test_web_endpoints_session_flow():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    r = client.get("/")
    assert r.status_code == 200
    assert "GTO Poker Trainer" in r.text

    # Create a session for 2 hands
    r = client.post("/api/session", json={"hands": 2, "mc": 60})
    assert r.status_code == 200
    sid = r.json()["session"]

    # Play through nodes, always choosing option 0; ensure summary works at the end
    steps = 0
    while True:
        r = client.get(f"/api/session/{sid}/node")
        assert r.status_code == 200
        data = r.json()
        if data.get("done"):
            break
        assert data["node"]["hand_no"] in (1, 2)
        # choose first option
        r2 = client.post(f"/api/session/{sid}/choose", json={"choice": 0})
        assert r2.status_code == 200
        steps += 1
        # hard stop if something loops
        assert steps < 20

    r = client.get(f"/api/session/{sid}/summary")
    assert r.status_code == 200
    js = r.json()
    assert js["hands"] >= 1

