"""M4 test: exactly K per class; same seed -> same indices; different seeds ->
different indices; all indices valid positions in the split; a class with fewer
than K available raises a clear error rather than silently undersampling."""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from cvlab.sampling.fewshot import balanced_kshot


def _synthetic_labels(num_classes: int = 10, per_class: int = 20) -> torch.Tensor:
    # Interleaved (not grouped) so index-order alone can't accidentally satisfy balance.
    labels = torch.arange(num_classes).repeat(per_class)
    perm = torch.randperm(labels.numel(), generator=torch.Generator().manual_seed(123))
    return labels[perm]


def test_exactly_k_per_class() -> None:
    labels = _synthetic_labels(num_classes=10, per_class=20)
    idx = balanced_kshot(labels, k=5, seed=0)
    counts = Counter(labels[idx].tolist())
    assert set(counts.keys()) == set(range(10))
    assert all(c == 5 for c in counts.values())
    assert idx.numel() == 50


def test_same_seed_gives_same_indices() -> None:
    labels = _synthetic_labels()
    idx1 = balanced_kshot(labels, k=5, seed=42)
    idx2 = balanced_kshot(labels, k=5, seed=42)
    assert torch.equal(idx1, idx2)


def test_different_seeds_give_different_indices() -> None:
    labels = _synthetic_labels()
    idx1 = balanced_kshot(labels, k=5, seed=0)
    idx2 = balanced_kshot(labels, k=5, seed=1)
    assert not torch.equal(idx1, idx2)


def test_all_indices_are_valid_positions() -> None:
    labels = _synthetic_labels(num_classes=10, per_class=20)
    idx = balanced_kshot(labels, k=10, seed=7)
    assert idx.min().item() >= 0
    assert idx.max().item() < labels.numel()
    assert idx.numel() == len(set(idx.tolist())), "indices must be unique"


def test_class_with_fewer_than_k_raises_clear_error() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])  # class 0 has only 3 examples
    with pytest.raises(ValueError, match="class 0"):
        balanced_kshot(labels, k=5, seed=0)


def test_non_positive_k_raises() -> None:
    labels = _synthetic_labels()
    with pytest.raises(ValueError):
        balanced_kshot(labels, k=0, seed=0)


def test_indices_index_into_original_labels_correctly() -> None:
    labels = _synthetic_labels(num_classes=5, per_class=8)
    idx = balanced_kshot(labels, k=3, seed=0)
    # Selected indices must map back to exactly the classes claimed, via the
    # original (unpermuted) label tensor -- not some internal reordering.
    selected_labels = labels[idx]
    for cls in range(5):
        assert (selected_labels == cls).sum().item() == 3


def test_real_dtd_train_split_is_balanced_at_k5() -> None:
    pytest.importorskip("torchvision")
    from cvlab.data.dtd import DTDDataset

    dtd_root = "data/dtd"
    import os

    if not os.path.isdir(dtd_root):
        pytest.skip("DTD not downloaded in this environment")

    train = DTDDataset(root=dtd_root, split="train", partition=1)
    # Read labels directly rather than through __getitem__, which would decode
    # all 1880 images just to discard the pixels.
    labels = torch.tensor(train._dataset._labels)
    idx = balanced_kshot(labels, k=5, seed=0)
    counts = Counter(labels[idx].tolist())
    assert len(counts) == 47
    assert all(c == 5 for c in counts.values())
