from __future__ import annotations

import json

from fastapi.testclient import TestClient

from gtotrainer.web.app import app


def test_web_endpoints_session_flow():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    r = client.get("/")
    assert r.status_code == 200
    assert "gtotrainer" in r.text
    assert "card poker-card placeholder" in r.text
    assert "width: clamp(38px, 6.8vw, 56px)" in r.text
    assert "data-summary-score" in r.text

    # Create a session for 2 hands
    base = "/api/v1/session"
    r = client.post(base, json={"hands": 2, "mc": 60})
    assert r.status_code == 200
    sid = r.json()["session"]

    # Play through nodes, always choosing option 0; ensure summary works at the end
    steps = 0
    while True:
        r = client.get(f"{base}/{sid}/node")
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
        contract = node.get("contract")
        assert contract is not None
        assert data["options"], "options list should not be empty"
        option_keys = [opt["key"].lower() for opt in data["options"]]
        if contract["state"] == "your_turn_no_bet":
            assert any(key.startswith("check") for key in option_keys)
            assert all(not key.startswith("fold") for key in option_keys)
            assert all(not key.startswith("call") for key in option_keys)
        elif contract["state"] == "your_turn_facing_bet":
            assert any(key.startswith("call") for key in option_keys)
            assert any(key.startswith("fold") for key in option_keys)
            assert all(not key.startswith("check") for key in option_keys)
        for option in data["options"]:
            assert isinstance(option, dict), "options should be dict payloads"
            assert "label" in option and isinstance(option["label"], str)
            assert ".0%" not in option["label"], f"label retains trailing .0%: {option['label']}"
        # choose first option
        r2 = client.post(f"{base}/{sid}/choose", json={"choice": 0})
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

    r = client.get(f"{base}/{sid}/summary")
    assert r.status_code == 200
    js = r.json()
    assert js["hands"] >= 1


def test_create_session_with_missing_or_invalid_inputs():
    client = TestClient(app)
    # Simulate empty/invalid form fields posting nulls/strings
    base = "/api/v1/session"
    r = client.post(base, json={"hands": None, "mc": None})
    assert r.status_code == 200
    sid = r.json()["session"]

    node = client.get(f"{base}/{sid}/node")
    assert node.status_code == 200
    node_body = node.json()
    assert node_body.get("node", {}).get("total_hands", 0) >= 1
    assert "contract" in node_body.get("node", {})

    summary = client.get(f"{base}/{sid}/summary")
    assert summary.status_code == 200
    payload = summary.json()
    assert payload["hands"] >= 0

    # Also ensure values are clamped when below minimums
    r2 = client.post(base, json={"hands": 0, "mc": 10})
    assert r2.status_code == 200
    sid2 = r2.json()["session"]
    summary2 = client.get(f"{base}/{sid2}/summary")
    assert summary2.status_code == 200
    assert summary2.json()["hands"] >= 0

    r3 = client.post(base, json={"hands": "", "mc": ""})
    assert r3.status_code == 200
    sid3 = r3.json()["session"]
    node3 = client.get(f"{base}/{sid3}/node")
    assert node3.status_code == 200


def test_hx_requests_receive_html_fragments():
    client = TestClient(app)
    headers = {"HX-Request": "true"}

    r = client.post("/api/v1/session", json={"hands": 1, "mc": 50}, headers=headers)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "hx-node" in r.text
    trigger_header = r.headers.get("hx-trigger")
    assert trigger_header, "HX-Trigger header should contain session metadata"
    trigger = json.loads(trigger_header)
    sid = trigger.get("sessionCreated")
    assert sid, "sessionCreated trigger missing"

    node = client.get(f"/api/v1/session/{sid}/node", headers=headers)
    assert node.status_code == 200
    assert "hx-node" in node.text

    choice = client.post(
        f"/api/v1/session/{sid}/choose",
        json={"choice": 0},
        headers=headers,
    )
    assert choice.status_code == 200
    assert "hx-feedback" in choice.text

    summary = client.get(f"/api/v1/session/{sid}/summary", headers=headers)
    assert summary.status_code == 200
    assert "hx-summary" in summary.text


def test_legacy_routes_still_operate():
    """Ensure pre-versioned endpoints remain functional for backwards compatibility."""

    client = TestClient(app)
    r = client.post("/api/session", json={"hands": 1, "mc": 50})
    assert r.status_code == 200
    sid = r.json()["session"]

    node = client.get(f"/api/session/{sid}/node")
    assert node.status_code == 200
    summary = client.get(f"/api/session/{sid}/summary")
    assert summary.status_code == 200
