"""Head-agnostic metrics computed from logits + labels. Every head (ImagePrototype,
LinearProbe, and any Stage-2 head satisfying the Head interface) is evaluated the
same way via argmax -- decision 12: scores themselves are not comparable across
heads, so nothing here reads raw score magnitudes."""

from __future__ import annotations

import torch


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def per_class_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Per-class top-1 accuracy, shape [num_classes]. A class with zero test
    examples gets NaN (undefined, not zero) rather than being silently dropped."""
    preds = logits.argmax(dim=1)
    accs = torch.full((num_classes,), float("nan"), dtype=torch.float64)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            accs[c] = (preds[mask] == c).float().mean()
    return accs


def mean_per_class_accuracy(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Decision 6: computed and stored alongside top-1 since Flowers-102's test
    split is class-imbalanced, but top-1 remains the headline metric."""
    accs = per_class_accuracy(logits, labels, num_classes)
    valid = accs[~torch.isnan(accs)]
    return valid.mean().item()


def confusion_matrix(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Row-normalized confusion matrix: rows are true classes, columns predicted
    classes. Rows for a class with at least one example sum to exactly 1; a class
    with zero examples gets an all-zero row rather than dividing by zero."""
    preds = logits.argmax(dim=1)
    flat_index = labels.to(torch.long) * num_classes + preds.to(torch.long)
    counts = torch.bincount(flat_index, minlength=num_classes * num_classes)
    counts = counts.reshape(num_classes, num_classes).to(torch.float64)
    row_sums = counts.sum(dim=1, keepdim=True)
    return counts / row_sums.clamp(min=1)
