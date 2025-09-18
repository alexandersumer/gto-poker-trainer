from __future__ import annotations

from fastapi.testclient import TestClient

from gto_trainer.web.app import app


def test_web_endpoints_session_flow():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    r = client.get("/")
    assert r.status_code == 200
    assert "GTO Trainer" in r.text
    assert "flex: 0 0 auto" in r.text

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
        node = data["node"]
        assert node["hand_no"] in (1, 2)
        hero_cards = node["hero_cards"]
        assert len(hero_cards) == 2
        assert all(isinstance(card, str) and len(card) == 2 for card in hero_cards)
        board_cards = node["board_cards"]
        assert all(isinstance(card, str) and len(card) == 2 for card in board_cards)
        assert data["options"], "options list should not be empty"
        for option in data["options"]:
            assert isinstance(option, dict), "options should be dict payloads"
            assert "label" in option and isinstance(option["label"], str)
        # choose first option
        r2 = client.post(f"/api/session/{sid}/choose", json={"choice": 0})
        assert r2.status_code == 200
        choice_payload = r2.json()
        assert "feedback" in choice_payload
        chosen_snapshot = choice_payload["feedback"]["chosen"]
        assert "label" in chosen_snapshot
        assert "next" in choice_payload
        nxt = choice_payload["next"]
        assert isinstance(nxt, dict)
        if not nxt.get("done"):
            assert "node" in nxt and nxt["node"]
            assert "options" in nxt and nxt["options"], "next payload should preload options"
        else:
            assert "summary" in nxt and isinstance(nxt["summary"], dict)
        steps += 1
        # hard stop if something loops
        assert steps < 20

    r = client.get(f"/api/session/{sid}/summary")
    assert r.status_code == 200
    js = r.json()
    assert js["hands"] >= 1


def test_create_session_with_missing_or_invalid_inputs():
    client = TestClient(app)
    # Simulate empty/invalid form fields posting nulls/strings
    r = client.post("/api/session", json={"hands": None, "mc": None})
    assert r.status_code == 200
    sid = r.json()["session"]

    node = client.get(f"/api/session/{sid}/node")
    assert node.status_code == 200
    node_body = node.json()
    assert node_body.get("node", {}).get("total_hands", 0) >= 1

    summary = client.get(f"/api/session/{sid}/summary")
    assert summary.status_code == 200
    payload = summary.json()
    assert payload["hands"] >= 0

    # Also ensure values are clamped when below minimums
    r2 = client.post("/api/session", json={"hands": 0, "mc": 10})
    assert r2.status_code == 200
    sid2 = r2.json()["session"]
    summary2 = client.get(f"/api/session/{sid2}/summary")
    assert summary2.status_code == 200
    assert summary2.json()["hands"] >= 0

    r3 = client.post("/api/session", json={"hands": "", "mc": ""})
    assert r3.status_code == 200
    sid3 = r3.json()["session"]
    node3 = client.get(f"/api/session/{sid3}/node")
    assert node3.status_code == 200
