"""M5 test list (verbatim from the spec):
- prototypes are unit-norm; shape [C, D]
- classifying the training features used to build a full-split prototype is
  clearly above chance
- a synthetic dataset of well-separated Gaussian clusters is classified at 100%
- top-1 accuracy is identical for temperature in {1.0, 10.0, 100.0}
- two `fit` calls on the same subset produce bit-identical prototypes
"""

from __future__ import annotations

import torch

from cvlab.heads.image_prototype import ImagePrototype


def _make_balanced_features(
    num_classes: int, per_class: int, dim: int, seed: int, separation: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian blobs, one per class, centered at well-separated random points."""
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, dim, generator=generator) * separation
    Z = []
    y = []
    for cls in range(num_classes):
        Z.append(centers[cls] + 0.01 * torch.randn(per_class, dim, generator=generator))
        y.extend([cls] * per_class)
    return torch.cat(Z), torch.tensor(y, dtype=torch.long)


def test_prototypes_are_unit_norm_and_correct_shape() -> None:
    Z, y = _make_balanced_features(num_classes=6, per_class=10, dim=16, seed=0, separation=5.0)
    head = ImagePrototype()
    head.fit(Z, y)
    assert head.prototypes.shape == (6, 16)
    norms = head.prototypes.norm(dim=1)
    assert torch.allclose(norms, torch.ones(6), atol=1e-5)


def test_classifying_training_features_is_above_chance() -> None:
    num_classes = 10
    Z, y = _make_balanced_features(
        num_classes=num_classes, per_class=20, dim=32, seed=1, separation=3.0
    )
    head = ImagePrototype()
    head.fit(Z, y)
    preds = head.logits(Z).argmax(dim=1)
    accuracy = (preds == y).float().mean().item()
    chance = 1.0 / num_classes
    assert accuracy > chance * 3, f"accuracy {accuracy} not clearly above chance {chance}"


def test_well_separated_gaussian_clusters_classified_perfectly() -> None:
    num_classes = 8
    Z, y = _make_balanced_features(
        num_classes=num_classes, per_class=15, dim=32, seed=2, separation=50.0
    )
    head = ImagePrototype()
    head.fit(Z, y)
    preds = head.logits(Z).argmax(dim=1)
    assert (preds == y).all()


def test_top1_accuracy_invariant_to_temperature() -> None:
    Z, y = _make_balanced_features(num_classes=7, per_class=12, dim=24, seed=3, separation=2.0)
    head = ImagePrototype()
    head.fit(Z, y)

    predictions = {}
    for temperature in (1.0, 10.0, 100.0):
        head.temperature = temperature
        predictions[temperature] = head.logits(Z).argmax(dim=1)

    base = predictions[1.0]
    for temperature, preds in predictions.items():
        assert torch.equal(preds, base), f"predictions changed at temperature={temperature}"
        accuracy = (preds == y).float().mean().item()
        assert accuracy == (base == y).float().mean().item()


def test_two_fit_calls_on_same_subset_are_bit_identical() -> None:
    Z, y = _make_balanced_features(num_classes=5, per_class=8, dim=16, seed=4, separation=4.0)

    head1 = ImagePrototype()
    head1.fit(Z, y)
    head2 = ImagePrototype()
    head2.fit(Z, y)

    assert torch.equal(head1.prototypes, head2.prototypes)


def test_logits_and_prototypes_raise_before_fit() -> None:
    import pytest

    head = ImagePrototype()
    with pytest.raises(RuntimeError):
        head.logits(torch.randn(3, 4))
    with pytest.raises(RuntimeError):
        _ = head.prototypes
