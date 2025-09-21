#!/usr/bin/env python3
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

IMAGE_NAME = os.environ.get("RENDER_CHECK_IMAGE", "gtotrainer:render-check")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    kwargs = {
        "check": True,
        "text": True,
    }
    if capture:
        kwargs["stdout"] = subprocess.PIPE
    return subprocess.run(cmd, **kwargs)


def build_image() -> None:
    print(f"[render-check] Building Docker image '{IMAGE_NAME}'…", flush=True)
    _run(["docker", "build", "-t", IMAGE_NAME, "."])


def run_container(port: int) -> str:
    print(f"[render-check] Starting container on port {port}…", flush=True)
    cp = _run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "-p",
            f"{port}:8000",
            IMAGE_NAME,
        ],
        capture=True,
    )
    container_id = (cp.stdout or "").strip()
    if not container_id:
        raise RuntimeError("docker run did not return a container id")
    return container_id


def wait_for_health(port: int, timeout: float = 30.0) -> None:
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + timeout
    errors = (urllib.error.URLError, ConnectionError, OSError)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/healthz", timeout=1) as resp:
                if resp.status == 200:
                    print("[render-check] Health endpoint reachable.")
                    return
        except errors:
            time.sleep(0.5)
    raise TimeoutError("Timed out waiting for /healthz")


def dump_logs(container_id: str) -> None:
    try:
        print("[render-check] Container logs:", flush=True)
        _run(["docker", "logs", container_id])
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[render-check] Failed to fetch logs: {exc}", file=sys.stderr)


def stop_container(container_id: str) -> None:
    try:
        _run(["docker", "stop", container_id])
    except subprocess.CalledProcessError as exc:  # pragma: no cover - container already gone
        print(f"[render-check] docker stop failed: {exc}", file=sys.stderr)


def main() -> int:
    port = int(os.environ.get("RENDER_CHECK_PORT", _free_port()))
    build_image()
    container_id = ""
    try:
        container_id = run_container(port)
        wait_for_health(port)
        print("[render-check] Render deployment smoke test passed.")
        return 0
    except Exception as exc:
        print(f"[render-check] Failure: {exc}", file=sys.stderr)
        if container_id:
            dump_logs(container_id)
        return 1
    finally:
        if container_id:
            stop_container(container_id)


if __name__ == "__main__":
    raise SystemExit(main())
