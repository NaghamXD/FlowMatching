"""Manifest schema for a cached feature split. A manifest is the source of truth
for whether a cache file can be trusted — filenames alone are never trusted."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class FeatureManifest:
    dataset: str
    split: str
    encoder: str
    checkpoint_id: str
    transform_description: str
    dtype: str
    shape: tuple[int, int]
    code_version: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureManifest":
        d = dict(data)
        d["shape"] = tuple(d["shape"])
        return cls(**d)


def manifest_hash(manifest: FeatureManifest) -> str:
    """Stable content hash of a manifest, for embedding in a run record's
    `feature_cache_manifest_hashes` without duplicating the full manifest there."""
    canonical = json.dumps(manifest.to_dict(), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def get_code_version(repo_root: Path | None = None) -> str:
    """Best-effort git identifier for the current source tree: `<commit>[-dirty]`,
    or 'unknown' if this project isn't itself a git repository (e.g. it's nested
    inside an unrelated enclosing repo, or git isn't installed)."""
    root = repo_root or _PROJECT_ROOT
    try:
        toplevel = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if Path(toplevel).resolve() != root.resolve():
            return "unknown"
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = (
            subprocess.run(["git", "-C", str(root), "diff", "--quiet"]).returncode != 0
        )
        return f"{commit}-dirty" if dirty else commit
    except Exception:
        return "unknown"
