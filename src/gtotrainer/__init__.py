from __future__ import annotations

import sys as _sys

# Enforce the project’s target runtime for consistency.
if _sys.version_info[:3] != (3, 13, 7):
    raise RuntimeError(f"gtotrainer requires Python 3.13.7 exactly; detected {_sys.version.split()[0]}")

__all__: list[str] = []
