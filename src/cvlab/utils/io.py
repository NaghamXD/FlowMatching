"""Generic filesystem helpers shared by the feature cache and results store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create `path` (and parents) if missing, and return it as a `Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
