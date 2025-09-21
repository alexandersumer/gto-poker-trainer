from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RENDER_YAML = REPO_ROOT / "render.yaml"


def test_render_yaml_declares_dockerfile_path() -> None:
    assert RENDER_YAML.exists(), "render.yaml must exist for Render deployments"
    contents = RENDER_YAML.read_text(encoding="utf-8")
    marker = "dockerfilePath:"
    assert marker in contents, "render.yaml should set dockerfilePath so Render finds the Dockerfile"
    path_fragment = contents.split(marker, 1)[1].splitlines()[0]
    cleaned = path_fragment.strip().strip("\"'")
    assert cleaned, "dockerfilePath entry should not be empty"
    dockerfile = (REPO_ROOT / cleaned).resolve()
    assert dockerfile.exists(), f"Declared Dockerfile '{cleaned}' not found"
