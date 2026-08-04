from __future__ import annotations

import pytest
import torch

from cvlab.config.schema import LinearProbeConfig
from cvlab.heads.linear import LinearProbe


def _separable_data(num_classes: int, per_class: int, dim: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, dim, generator=generator) * 8.0
    Z, y = [], []
    for cls in range(num_classes):
        Z.append(centers[cls] + 0.1 * torch.randn(per_class, dim, generator=generator))
        y.extend([cls] * per_class)
    return torch.cat(Z), torch.tensor(y, dtype=torch.long)


def _separable_train_val_split(num_classes: int, per_class_train: int, per_class_val: int, dim: int, seed: int):
    """Train and val share the same class centers (only the noise differs) --
    otherwise they'd represent unrelated classification problems."""
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, dim, generator=generator) * 8.0
    Z_train, y_train, Z_val, y_val = [], [], [], []
    for cls in range(num_classes):
        Z_train.append(centers[cls] + 0.1 * torch.randn(per_class_train, dim, generator=generator))
        y_train.extend([cls] * per_class_train)
        Z_val.append(centers[cls] + 0.1 * torch.randn(per_class_val, dim, generator=generator))
        y_val.extend([cls] * per_class_val)
    return (
        torch.cat(Z_train),
        torch.tensor(y_train, dtype=torch.long),
        torch.cat(Z_val),
        torch.tensor(y_val, dtype=torch.long),
    )


def test_fit_requires_val_tensors() -> None:
    Z, y = _separable_data(4, 5, 8, seed=0)
    head = LinearProbe(num_classes=4, feature_dim=8, config=LinearProbeConfig(max_epochs=5), seed=0)
    with pytest.raises(ValueError):
        head.fit(Z, y)


def test_logits_raises_before_fit() -> None:
    head = LinearProbe(num_classes=4, feature_dim=8, config=LinearProbeConfig(max_epochs=5), seed=0)
    with pytest.raises(RuntimeError):
        head.logits(torch.randn(3, 8))


def test_fit_then_logits_end_to_end() -> None:
    Z_train, y_train, Z_val, y_val = _separable_train_val_split(
        num_classes=5, per_class_train=10, per_class_val=4, dim=16, seed=1
    )
    config = LinearProbeConfig(lr=1e-2, weight_decay=0.0, max_epochs=150)
    head = LinearProbe(num_classes=5, feature_dim=16, config=config, seed=1)
    head.fit(Z_train, y_train, Z_val, y_val)

    assert head.training_result is not None
    assert len(head.training_result.history) == 150

    logits = head.logits(Z_train)
    assert logits.shape == (Z_train.shape[0], 5)
    preds = logits.argmax(dim=1)
    accuracy = (preds == y_train).float().mean().item()
    assert accuracy > 0.9  # well-separated synthetic clusters
