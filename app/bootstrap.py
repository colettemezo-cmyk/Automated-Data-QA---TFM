"""Insert the `app/` package root on sys.path (cwd-independent imports)."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_app_on_path(caller_file: str | Path) -> Path:
    """Walk up from `caller_file` until `app/` (with `common/`) is found."""
    p = Path(caller_file).resolve()
    for parent in p.parents:
        if parent.name == "app" and (parent / "common").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError(f"Could not locate app/ package root from {caller_file}")
