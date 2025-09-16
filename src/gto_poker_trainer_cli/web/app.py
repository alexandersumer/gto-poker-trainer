from __future__ import annotations

import os
import secrets
import string
import threading
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from ..core.models import Option
from ..dynamic.cards import format_card_ascii, format_cards_spaced
from ..dynamic.generator import Episode, Node, generate_episode
from ..dynamic.policy import options_for


def _sid(n: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))


@dataclass
class Session:
    hands: int
    mc_trials: int
    episodes: list[Episode] = field(default_factory=list)
    hand_index: int = 0  # 0-based index of current hand
    current_index: int = 0  # index into nodes of current episode
    records: list[dict[str, Any]] = field(default_factory=list)


SESSIONS: dict[str, Session] = {}
LOCK = threading.Lock()


def _best_index(opts: list[Option]) -> int:
    return max(range(len(opts)), key=lambda i: opts[i].ev)


def _format_node(node: Node, *, hand_no: int, total_hands: int) -> dict[str, Any]:
    return {
        "street": node.street,
        "description": node.description,
        "pot_bb": node.pot_bb,
        "effective_bb": node.effective_bb,
        "hero": format_cards_spaced(node.hero_cards),
        "board": " ".join(format_card_ascii(c) for c in node.board),
        "actor": node.actor,
        "hand_no": hand_no,
        "total_hands": total_hands,
    }


class CreateSessionRequest(BaseModel):
    hands: int = 1
    mc: int = 120


class ChoiceRequest(BaseModel):
    choice: int


app = FastAPI(title="GTO Poker Trainer")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # Serve a tiny single-file UI kept in package data
    try:
        from importlib.resources import files

        data_dir = files("gto_poker_trainer_cli.data")
        html = (data_dir / "web" / "index.html").read_text(encoding="utf-8")
        return html
    except Exception as exc:  # pragma: no cover - packaging edge
        return f"<html><body><h1>GTO Poker Trainer</h1><p>Failed to load UI: {exc}</p></body></html>"


@app.post("/api/session")
def create_session(body: CreateSessionRequest) -> JSONResponse:
    rng = secrets.SystemRandom()
    first = generate_episode(rng)
    sid = _sid()
    sess = Session(hands=max(1, body.hands), mc_trials=max(10, body.mc), episodes=[first])
    with LOCK:
        SESSIONS[sid] = sess
    return JSONResponse({"session": sid})


@app.get("/api/session/{sid}/node")
def get_node(sid: str) -> JSONResponse:
    with LOCK:
        sess = SESSIONS.get(sid)
        if not sess:
            raise HTTPException(404, "no such session")
        # If current hand finished, advance or create next hand
        while True:
            ep = sess.episodes[sess.hand_index]
            if sess.current_index < len(ep.nodes):
                node = ep.nodes[sess.current_index]
                break
            # Current episode done; move to next hand
            if sess.hand_index + 1 >= sess.hands:
                return JSONResponse({"done": True})
            sess.hand_index += 1
            sess.current_index = 0
            # Ensure we have an episode for the next hand
            if sess.hand_index >= len(sess.episodes):
                sess.episodes.append(generate_episode(secrets.SystemRandom()))
            ep = sess.episodes[sess.hand_index]
            node = ep.nodes[sess.current_index]
            break
    opts = options_for(node, secrets.SystemRandom(), sess.mc_trials)
    return JSONResponse(
        {
            "done": False,
            "node": _format_node(node, hand_no=sess.hand_index + 1, total_hands=sess.hands),
            "options": [o.key for o in opts],
        }
    )


@app.post("/api/session/{sid}/choose")
def post_choice(sid: str, body: ChoiceRequest) -> JSONResponse:
    with LOCK:
        sess = SESSIONS.get(sid)
        if not sess:
            raise HTTPException(404, "no such session")
        # Determine current node
        ep = sess.episodes[sess.hand_index]
        if sess.current_index >= len(ep.nodes):
            return JSONResponse({"done": True})
        node = ep.nodes[sess.current_index]

    opts = options_for(node, secrets.SystemRandom(), sess.mc_trials)
    if not (0 <= body.choice < len(opts)):
        raise HTTPException(400, "invalid choice index")
    chosen = opts[body.choice]
    best = opts[_best_index(opts)]
    record = {
        "street": node.street,
        "chosen_key": chosen.key,
        "chosen_ev": chosen.ev,
        "best_key": best.key,
        "best_ev": best.ev,
        "ev_loss": best.ev - chosen.ev,
    }
    with LOCK:
        sess.records.append(record)
        ends = getattr(chosen, "ends_hand", False)
        sess.current_index += 1 if not ends else len(ep.nodes)

    return JSONResponse(
        {
            "feedback": {
                "correct": chosen.key == best.key,
                "ev_loss": record["ev_loss"],
                "chosen": {"key": chosen.key, "ev": chosen.ev, "why": chosen.why},
                "best": {"key": best.key, "ev": best.ev, "why": best.why},
                "ended": ends,
            }
        }
    )


@app.get("/api/session/{sid}/summary")
def get_summary(sid: str) -> JSONResponse:
    with LOCK:
        sess = SESSIONS.get(sid)
        if not sess:
            raise HTTPException(404, "no such session")
        recs = list(sess.records)
    if not recs:
        return JSONResponse({"hands": 0, "hits": 0, "score": 0.0})
    total_ev_best = sum(r["best_ev"] for r in recs)
    total_ev_chosen = sum(r["chosen_ev"] for r in recs)
    total_ev_lost = total_ev_best - total_ev_chosen
    hits = sum(1 for r in recs if r["chosen_key"] == r["best_key"])
    room = sum(max(1e-9, r["best_ev"] - min(0.0, r["chosen_ev"])) for r in recs)
    score_pct = 100.0 * max(0.0, 1.0 - (total_ev_lost / room)) if room > 1e-9 else 100.0
    return JSONResponse(
        {
            "hands": len(recs),
            "hits": hits,
            "ev_lost": total_ev_lost,
            "score": score_pct,
        }
    )


def main() -> None:  # pragma: no cover - runner
    import uvicorn

    host = os.environ.get("BIND", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("gto_poker_trainer_cli.web.app:app", host=host, port=port, factory=False)


if __name__ == "__main__":  # pragma: no cover
    main()
