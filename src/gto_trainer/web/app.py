from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..features.session import SessionManager
from ..features.session.router import create_session_routers

app = FastAPI(title="GTO Trainer")
_templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
_manager = SessionManager()

router_v1, router_legacy = create_session_routers(_manager, _templates)
app.include_router(router_v1)
app.include_router(router_legacy)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    try:
        data_dir = resources.files("gto_trainer.data")
        return (data_dir / "web" / "index.html").read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - packaging edge
        return f"<html><body><h1>GTO Trainer</h1><p>Failed to load UI: {exc}</p></body></html>"


def main() -> None:  # pragma: no cover - runner
    import uvicorn

    host = os.environ.get("BIND", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, factory=False)


if __name__ == "__main__":  # pragma: no cover
    main()
