from __future__ import annotations

import sys as _sys

# Enforce the project’s target runtime for consistency.
if _sys.version_info[:3] != (3, 12, 11):
    raise RuntimeError(f"gto-trainer requires Python 3.12.11 exactly; detected {_sys.version.split()[0]}")

__all__: list[str] = []
