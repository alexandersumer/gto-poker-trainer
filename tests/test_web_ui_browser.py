from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest

sync_module = pytest.importorskip("playwright.sync_api")
sync_playwright = sync_module.sync_playwright


@contextlib.contextmanager
def _run_server() -> Iterator[str]:
    """Start the FastAPI server on localhost and yield its base URL."""

    proc = subprocess.Popen(  # noqa: S603 - trusted command
        [sys.executable, "-m", "gto_trainer.web.app"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        start = time.time()
        ready = False
        while time.time() - start < 30:
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                # Mirror logs to help diagnose CI flakes.
                sys.stdout.write(line)
            if "Application startup complete" in line:
                ready = True
                break
        if not ready:
            raise RuntimeError("web server failed to start")
        yield "http://127.0.0.1:8000"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_browser_flow_shows_initial_hand() -> None:
    console_errors: list[str] = []
    page_errors: list[str] = []

    with _run_server() as base_url, sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        page.on(
            "console",
            lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
        )
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        page.goto(base_url, wait_until="domcontentloaded")
        page.click("#btn-start")
        page.wait_for_selector("#hand .card", timeout=15_000)
        page.wait_for_selector("#hand .card:not(.placeholder)", timeout=15_000)

        # The UI should swap out the placeholder copy with actual cards.
        hand_html = page.inner_html("#hand").strip().lower()
        assert "placeholder" not in hand_html
        assert "--" not in hand_html

        # Ensure an action is presented so the drill can continue.
        page.wait_for_selector("#action-area .action-button", timeout=15_000)

        status = page.inner_text("#header-status").strip().lower()
        assert "preparing" not in status

        assert not console_errors, f"Console errors captured: {console_errors}"
        assert not page_errors, f"Page errors captured: {page_errors}"

        browser.close()
