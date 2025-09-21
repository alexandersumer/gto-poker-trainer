from __future__ import annotations

import re
from pathlib import Path


def _load_web_index() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "src" / "gtotrainer" / "data" / "web" / "index.html").read_text(encoding="utf-8")


def test_web_bundle_has_no_external_http_assets() -> None:
    html = _load_web_index()
    links = re.findall(r"(?:src|href)\s*=\s*\"([^\"]+)\"", html, flags=re.IGNORECASE)

    disallowed = [
        url for url in links if url.lower().startswith(("http://", "https://")) and not url.lower().startswith("data:")
    ]

    assert not disallowed, f"Unexpected external asset references: {disallowed}"


def test_web_bundle_contains_expected_utility_classes() -> None:
    html = _load_web_index()
    required = {
        ".grid",
        ".grid-cols-1",
        ".sm\\:grid-cols-2",
        ".gap-3",
        ".rounded-lg",
        ".rounded-xl",
    }
    for token in required:
        assert token in html, f"Missing CSS utility token: {token}"
