import asyncio
import os
import signal
import subprocess
import time
from contextlib import suppress

import httpx
import pytest
from tenacity import retry, stop_after_delay, wait_fixed

BINARY = os.environ.get("GTO_TRAINER_BIN", "cargo")
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
SERVER_URL = f"{SERVER_HOST}:{SERVER_PORT}"


class ServerProcess:
    def __init__(self):
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        if BINARY == "cargo":
            cmd = [
                "cargo",
                "run",
                "--",
                "serve",
                "--addr",
                SERVER_URL,
            ]
        else:
            cmd = [BINARY, "serve", "--addr", SERVER_URL]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self) -> None:
        if not self.process:
            return
        with suppress(ProcessLookupError):
            self.process.send_signal(signal.SIGINT)
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                self.process.kill()
        self.process = None

    def iter_stdout(self):
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            yield line


@pytest.fixture(scope="session")
def server():
    proc = ServerProcess()
    proc.start()
    try:
        wait_for_healthcheck()
        yield SERVER_URL
    finally:
        proc.stop()


@retry(stop=stop_after_delay(30), wait=wait_fixed(0.5))
def wait_for_healthcheck() -> None:
    response = httpx.get(f"http://{SERVER_URL}/healthz", timeout=1.0)
    response.raise_for_status()


@pytest.mark.asyncio
async def test_session_flow(server):
    async with httpx.AsyncClient(base_url=f"http://{server}") as client:
        resp = await client.post(
            "/api/sessions",
            json={"hands": 1, "mc_samples": 50, "rival_style": "balanced"},
            timeout=10.0,
        )
        resp.raise_for_status()
        state = resp.json()
        assert state["status"] == "awaiting_input"
        session_id = state["session_id"]
        action = state["node"]["action_options"][0]["action"]

        resp = await client.post(
            f"/api/sessions/{session_id}/actions",
            json={"action": action},
            timeout=10.0,
        )
        resp.raise_for_status()
        updated = resp.json()
        assert updated["node"]["street"] in {"flop", "turn", "river", "terminal"}


@pytest.mark.asyncio
async def test_web_assets_serve(server):
    async with httpx.AsyncClient(base_url=f"http://{server}") as client:
        resp = await client.get("/")
        resp.raise_for_status()
        assert "GTO Trainer" in resp.text

        resp = await client.get("/healthz")
        resp.raise_for_status()
        assert resp.text.strip() == "ok"
