from __future__ import annotations

import argparse


def main() -> None:
    try:
        from textual_serve.server import Server
    except Exception as exc:  # pragma: no cover - optional path
        print(
            "textual-serve is not installed. Install extra 'tui' (pip install '.[tui]') or run: \n"
            "  pip install textual-serve\n"
            f"Error: {exc}"
        )
        raise SystemExit(2)

    p = argparse.ArgumentParser(
        prog="gto-poker-trainer-serve",
        description="Serve the Textual UI over HTTP (Textual Web)",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--hands", type=int, default=1)
    p.add_argument("--mc", type=int, default=200)
    args = p.parse_args()

    # Serve the console script command; textual-serve runs the process per session
    cmd = f"gto-poker-trainer-textual --hands {args.hands} --mc {args.mc}"
    server = Server(apps={"trainer": cmd}, host=args.host, port=args.port)
    print(f"Serving at http://{args.host}:{args.port} …")
    server.run()


if __name__ == "__main__":
    main()

