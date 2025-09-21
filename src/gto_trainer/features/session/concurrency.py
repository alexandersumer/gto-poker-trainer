from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

_MAX_WORKERS = max(1, min(32, os.cpu_count() or 1))
_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="gto-session")


async def run_blocking(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    loop = asyncio.get_running_loop()
    bound = partial(func, *args, **kwargs)
    return await loop.run_in_executor(_EXECUTOR, bound)
