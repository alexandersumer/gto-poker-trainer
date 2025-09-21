from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RENDER_YAML = REPO_ROOT / "render.yaml"
DEPLOY_DOCKERFILE = REPO_ROOT / "deploy" / "Dockerfile"
ROOT_DOCKERFILE = REPO_ROOT / "Dockerfile"


def _extract(contents: str, key: str) -> str:
    assert key in contents, f"render.yaml must define {key}"
    fragment = contents.split(key, 1)[1].splitlines()[0]
    value = fragment.strip().strip("\"'")
    assert value, f"{key} entry should not be empty"
    return value


def test_render_yaml_docker_settings() -> None:
    assert RENDER_YAML.exists(), "render.yaml must exist for Render deployments"
    contents = RENDER_YAML.read_text(encoding="utf-8")

    dockerfile_path = _extract(contents, "dockerfilePath:")
    docker_context = _extract(contents, "dockerContext:")
    root_dir = _extract(contents, "rootDir:")

    dockerfile = (REPO_ROOT / dockerfile_path).resolve()
    assert dockerfile.exists(), f"Declared Dockerfile '{dockerfile_path}' not found"
    assert (REPO_ROOT / docker_context).resolve().exists(), "dockerContext path must exist"
    assert (REPO_ROOT / root_dir).resolve().exists(), "rootDir path must exist"

    assert DEPLOY_DOCKERFILE.exists(), "deploy/Dockerfile should exist to support alternative contexts"
    assert ROOT_DOCKERFILE.exists(), "Root Dockerfile must exist"
    assert DEPLOY_DOCKERFILE.read_text(encoding="utf-8") == ROOT_DOCKERFILE.read_text(encoding="utf-8"), (
        "deploy/Dockerfile should stay in sync with the root Dockerfile"
    )
