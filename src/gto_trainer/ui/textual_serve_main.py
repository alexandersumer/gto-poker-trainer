from __future__ import annotations

import argparse
import os


def main() -> None:
    try:
        from textual_serve.server import Server
    except Exception as exc:  # pragma: no cover - optional path
        print(
            "textual-serve is not installed. Install extra 'tui' (pip install '.[tui]') or run: \n"
            "  pip install textual-serve\n"
            f"Error: {exc}"
        )
        raise SystemExit(2) from None

    p = argparse.ArgumentParser(
        prog="gto-trainer-serve",
        description="Serve the Textual UI over HTTP (Textual Web)",
    )
    p.add_argument("--host", default=os.environ.get("BIND", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    p.add_argument("--hands", type=int, default=int(os.environ.get("HANDS", "1")))
    p.add_argument("--mc", type=int, default=int(os.environ.get("MC", "200")))
    args = p.parse_args()

    # Serve the Textual app via shell command; one process per browser session
    # Avoid relying on console scripts (PATH issues in some containers)
    cmd = f"python -m gto_trainer.ui.textual_main --hands {args.hands} --mc {args.mc}"
    public_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not public_url:
        host = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
        if host:
            public_url = f"https://{host}"
    server = Server(command=cmd, host=args.host, port=args.port, title="GTO Trainer", public_url=public_url)
    print(f"Serving at http://{args.host}:{args.port} …")
    # textual-serve uses .serve(); keep default (debug=False)
    server.serve()


if __name__ == "__main__":
    main()
