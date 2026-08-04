"""Feature cache: tensor files + manifest on disk, keyed by (dataset, split, encoder).
`is_cache_valid` is the only thing that decides whether a cache hit is trustworthy —
callers never infer validity from a file merely existing."""

from __future__ import annotations

from pathlib import Path

import torch

from cvlab.features.manifest import FeatureManifest
from cvlab.utils.io import ensure_dir, read_json, write_json


def cache_dir(cache_root: str | Path, dataset: str, split: str, encoder: str) -> Path:
    return Path(cache_root) / dataset / split / encoder


def _manifest_path(directory: Path) -> Path:
    return directory / "manifest.json"


def _tensors_path(directory: Path) -> Path:
    return directory / "features.pt"


def is_cache_valid(
    cache_root: str | Path,
    dataset: str,
    split: str,
    encoder: str,
    expected: FeatureManifest,
) -> bool:
    directory = cache_dir(cache_root, dataset, split, encoder)
    manifest_path = _manifest_path(directory)
    tensors_path = _tensors_path(directory)
    if not manifest_path.exists() or not tensors_path.exists():
        return False
    try:
        stored = FeatureManifest.from_dict(read_json(manifest_path))
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return stored == expected


def save(
    cache_root: str | Path,
    dataset: str,
    split: str,
    encoder: str,
    Z: torch.Tensor,
    y: torch.Tensor,
    manifest: FeatureManifest,
) -> None:
    directory = ensure_dir(cache_dir(cache_root, dataset, split, encoder))
    torch.save({"Z": Z, "y": y}, _tensors_path(directory))
    write_json(_manifest_path(directory), manifest.to_dict())


def load(
    cache_root: str | Path, dataset: str, split: str, encoder: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load cached (Z, y) for a (dataset, split, encoder). Raises FileNotFoundError
    if nothing is cached yet — run `extract-features` first."""
    directory = cache_dir(cache_root, dataset, split, encoder)
    tensors_path = _tensors_path(directory)
    if not tensors_path.exists():
        raise FileNotFoundError(
            f"No cached features at {tensors_path}. Run extract-features for "
            f"(dataset={dataset!r}, split={split!r}, encoder={encoder!r}) first."
        )
    blob = torch.load(tensors_path)
    return blob["Z"], blob["y"]


def load_manifest(
    cache_root: str | Path, dataset: str, split: str, encoder: str
) -> FeatureManifest:
    """Load just the manifest for a cached (dataset, split, encoder) -- used by the
    sweep runner to embed manifest hashes in a run record without re-reading tensors."""
    manifest_path = _manifest_path(cache_dir(cache_root, dataset, split, encoder))
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")
    return FeatureManifest.from_dict(read_json(manifest_path))
