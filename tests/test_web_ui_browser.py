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
        assert "rival" not in status
        assert "(" not in status
        assert status.count("hand") <= 1

        assert not console_errors, f"Console errors captured: {console_errors}"
        assert not page_errors, f"Page errors captured: {page_errors}"

        browser.close()


def test_card_markup_helpers_render() -> None:
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

        markup = page.evaluate(
            """
            () => {
              const el = document.createElement('div');
              if (!window.setTextWithCards) {
                throw new Error('setTextWithCards is not defined');
              }
              window.setTextWithCards(el, 'As Kd');
              return el.innerHTML;
            }
            """
        )

        assert "inline-card" in markup
        assert "inline-card--s" in markup
        assert "inline-card--d" in markup

        assert not console_errors, f"Console errors captured: {console_errors}"
        assert not page_errors, f"Page errors captured: {page_errors}"

        browser.close()


def test_start_button_state_transitions() -> None:
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

        page.wait_for_function(
            "() => document.querySelector('#btn-start').textContent.toLowerCase().includes('starting')",
            timeout=10_000,
        )
        page.wait_for_selector("#hand .card:not(.placeholder)", timeout=15_000)
        page.wait_for_function(
            "() => !document.querySelector('#btn-start').textContent.toLowerCase().includes('starting')",
            timeout=15_000,
        )

        label = page.eval_on_selector("#btn-start", "el => (el.textContent || '').trim().toLowerCase()")
        assert label == "start"
        assert not console_errors, f"Console errors captured: {console_errors}"
        assert not page_errors, f"Page errors captured: {page_errors}"

        browser.close()


def test_feedback_drawer_after_action() -> None:
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
        page.wait_for_selector("#hand .card:not(.placeholder)", timeout=15_000)
        page.wait_for_selector("#action-area .action-button", timeout=15_000)
        page.click("#action-area .action-button")

        page.wait_for_function(
            "() => !document.querySelector('#feedback-banner').classList.contains('hidden')",
            timeout=15_000,
        )

        page.click("#feedback-detail")
        page.wait_for_function(
            "() => !document.querySelector('#feedback-drawer').classList.contains('hidden')",
            timeout=10_000,
        )

        drawer_html = page.inner_html("#feedback-drawer-body").lower()
        assert "gto recommendation" in drawer_html
        assert "drawer-row" in drawer_html

        assert not console_errors, f"Console errors captured: {console_errors}"
        assert not page_errors, f"Page errors captured: {page_errors}"

        browser.close()


def test_end_session_shows_summary() -> None:
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
        page.wait_for_selector("#hand .card:not(.placeholder)", timeout=15_000)

        page.click("#btn-end")
        page.wait_for_function(
            "() => !document.querySelector('#panel-summary').classList.contains('hidden')",
            timeout=15_000,
        )

        summary_html = page.inner_html("#panel-summary").lower()
        assert "session summary" in summary_html
        assert "data-summary-score" in summary_html

        assert not console_errors, f"Console errors captured: {console_errors}"
        assert not page_errors, f"Page errors captured: {page_errors}"

        browser.close()


def test_theme_toggle_cycles_dark_mode() -> None:
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

        page.click("#theme-toggle")
        page.wait_for_function(
            "() => document.querySelector('#theme-menu') && document.querySelector('#theme-menu').hidden === false",
            timeout=5_000,
        )
        page.click(".theme-menu__item[data-theme-choice='dark']")

        page.wait_for_function(
            "() => document.documentElement.dataset.theme === 'dark'",
            timeout=10_000,
        )

        assert page.evaluate("() => document.documentElement.dataset.theme") == "dark"

        assert not console_errors, f"Console errors captured: {console_errors}"
        assert not page_errors, f"Page errors captured: {page_errors}"

        browser.close()
