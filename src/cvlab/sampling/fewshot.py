"""Deterministic balanced K-shot index selection. The linear probe and the
image-prototype head both call `balanced_kshot` so they see identical subsets
for a given (K, seed) — see decision 4 (one seed controls all stochastic
behavior in a run, including this selection)."""

from __future__ import annotations

import torch


def balanced_kshot(labels: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """Select exactly `k` indices per class from `labels`, deterministically for a
    given seed. Returns a 1-D sorted LongTensor of indices into `labels` — never
    the underlying data itself.

    Raises ValueError if any class has fewer than `k` available examples. Silently
    returning fewer than `k` for that class would break the "exactly K per class"
    balance guarantee every downstream head relies on.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    labels = labels.to(torch.long)
    generator = torch.Generator().manual_seed(seed)

    selected: list[torch.Tensor] = []
    for cls in torch.unique(labels).tolist():
        class_indices = torch.nonzero(labels == cls, as_tuple=True)[0]
        if class_indices.numel() < k:
            raise ValueError(
                f"class {cls} has only {class_indices.numel()} examples, fewer than k={k}"
            )
        perm = torch.randperm(class_indices.numel(), generator=generator)
        selected.append(class_indices[perm[:k]])

    return torch.cat(selected).sort().values
