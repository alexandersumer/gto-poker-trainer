from __future__ import annotations

import sys as _sys

if _sys.version_info[:3] != (3, 12, 11):
    raise RuntimeError(
        f"gto-poker-trainer-cli requires Python 3.12.11 exactly; detected {_sys.version.split()[0]}"
    )

__all__: list[str] = []
