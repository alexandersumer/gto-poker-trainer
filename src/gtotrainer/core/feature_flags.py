"""Lightweight feature flag helpers.

The trainer needs to toggle upcoming algorithmic experiments on and off while
we benchmark them.  We keep the implementation minimal to avoid introducing a
new dependency: flags are exposed through an environment variable and can be
temporarily overridden in tests via a context manager.

Usage::

    from gtotrainer.core import feature_flags

    if feature_flags.is_enabled("solver.high_precision_cfr"):
        ...

The environment variable ``GTOTRAINER_FEATURES`` accepts a comma-separated
list of flag names.  Flag names are case-insensitive.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Final

_ENV_VAR: Final = "GTOTRAINER_FEATURES"


def _normalise(flag: str) -> str:
    return flag.strip().lower()


def _parse_env(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {_normalise(entry) for entry in raw.split(",") if entry.strip()}


_OVERRIDE_STACK: list[tuple[set[str], set[str]]] = []


def _current_overrides() -> tuple[set[str], set[str]]:
    enabled: set[str] = set()
    disabled: set[str] = set()
    for en, dis in _OVERRIDE_STACK:
        enabled.update(en)
        disabled.update(dis)
    return enabled, disabled


def is_enabled(flag: str) -> bool:
    """Return True when *flag* is enabled via env var or overrides."""

    key = _normalise(flag)
    enabled, disabled = _current_overrides()
    if key in disabled:
        return False
    if key in enabled:
        return True
    env_flags = _parse_env(os.getenv(_ENV_VAR))
    return key in env_flags


@contextmanager
def override(*, enable: Iterable[str] | None = None, disable: Iterable[str] | None = None):
    """Temporarily override flag state within the context.

    ``enable`` and ``disable`` accept iterables of flag names.  Overrides are
    stacked, so nested contexts behave predictably.
    """

    enabled = {_normalise(flag) for flag in (enable or ())}
    disabled = {_normalise(flag) for flag in (disable or ())}
    _OVERRIDE_STACK.append((enabled, disabled))
    try:
        yield
    finally:
        _OVERRIDE_STACK.pop()


def set_env_flags(flags: Iterable[str]) -> None:
    """Convenience helper used in scripts/tests to set the env list."""

    os.environ[_ENV_VAR] = ",".join(sorted({_normalise(flag) for flag in flags}))
