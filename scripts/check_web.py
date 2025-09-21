from __future__ import annotations

import contextlib
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.request


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    port = int(os.environ.get("PORT", str(free_port())))

    try:
        import textual_serve.server  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency missing
        print(f"textual-serve is required for this check: {exc}", file=sys.stderr)
        return 3

    python_cmd = shlex.quote(sys.executable)
    command = f"{python_cmd} -m gto_trainer.ui.textual_main --hands 1 --mc 5"

    launcher = "\n".join(
        [
            "from textual_serve.server import Server",
            "server = Server(",
            f"    command={command!r},",
            "    host='127.0.0.1',",
            f"    port={port},",
            "    title='GTO Trainer',",
            "    public_url=None,",
            ")",
            "server.serve()",
        ]
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", launcher],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    base = f"http://127.0.0.1:{port}/"
    ok = False
    try:
        for _ in range(100):
            if proc.poll() is not None:
                break
            time.sleep(0.1)
            try:
                with urllib.request.urlopen(base, timeout=1) as resp:
                    html = resp.read(4096).decode("utf-8", "ignore")
                    if "<html" in html.lower():
                        ok = True
                        break
            except Exception:
                continue
        if ok:
            return 0
        output = proc.stdout.read() if proc.stdout else ""
        if output:
            sys.stderr.write(output[-2000:])
        return 2
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=2)
        if proc.poll() is None:
            proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
