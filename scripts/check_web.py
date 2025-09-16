from __future__ import annotations

import os
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
    cmd = [
        sys.executable,
        "-m",
        "textual",
        "serve",
        "python -m gto_poker_trainer_cli.ui.textual_main --hands 1 --mc 5",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Wait for server to accept connections
        base = f"http://127.0.0.1:{port}/"
        ok = False
        for _ in range(100):
            time.sleep(0.1)
            try:
                with urllib.request.urlopen(base, timeout=1) as resp:
                    html = resp.read(4096).decode("utf-8", "ignore")
                    if "<html" in html.lower():
                        ok = True
                        break
            except Exception:
                continue
        if not ok:
            # dump a bit of logs to help diagnose
            try:
                output = proc.stdout.read() if proc.stdout else ""
                sys.stderr.write(output[-2000:])
            except Exception:
                pass
            return 2
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()
            proc.wait(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
