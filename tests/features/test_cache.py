from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cvlab.features.cache import is_cache_valid, load, save
from cvlab.features.manifest import FeatureManifest


def _manifest(**overrides) -> FeatureManifest:
    fields = dict(
        dataset="dtd",
        split="train",
        encoder="resnet18",
        checkpoint_id="IMAGENET1K_V1",
        transform_description="tag=v1",
        dtype="float32",
        shape=(5, 4),
        code_version="unknown",
    )
    fields.update(overrides)
    return FeatureManifest(**fields)


def test_save_then_load_round_trips_tensors_exactly(tmp_path: Path) -> None:
    Z = torch.randn(5, 4)
    y = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
    manifest = _manifest()

    save(tmp_path, "dtd", "train", "resnet18", Z, y, manifest)
    Z2, y2 = load(tmp_path, "dtd", "train", "resnet18")

    assert torch.equal(Z, Z2)
    assert torch.equal(y, y2)


def test_load_missing_cache_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "dtd", "train", "resnet18")


def test_is_cache_valid_false_when_nothing_cached(tmp_path: Path) -> None:
    assert not is_cache_valid(tmp_path, "dtd", "train", "resnet18", _manifest())


def test_is_cache_valid_true_for_matching_manifest(tmp_path: Path) -> None:
    manifest = _manifest()
    save(tmp_path, "dtd", "train", "resnet18", torch.randn(5, 4), torch.zeros(5, dtype=torch.long), manifest)
    assert is_cache_valid(tmp_path, "dtd", "train", "resnet18", manifest)


def test_is_cache_valid_false_for_mismatched_manifest(tmp_path: Path) -> None:
    original = _manifest()
    save(tmp_path, "dtd", "train", "resnet18", torch.randn(5, 4), torch.zeros(5, dtype=torch.long), original)
    changed = _manifest(transform_description="tag=v2")
    assert not is_cache_valid(tmp_path, "dtd", "train", "resnet18", changed)
