from __future__ import annotations

import sys as _sys

# Enforce the project’s target runtime for consistency.
_major, _minor, _patch = _sys.version_info[:3]
if _major != 3 or _minor != 13 or _patch < 5:
    raise RuntimeError(
        f"gto-trainer requires Python 3.13.5 or newer within the 3.13 line; detected {_sys.version.split()[0]}"
    )

__all__: list[str] = []
