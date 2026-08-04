"""M7 test: top-1 matches a hand-computed value on a toy logits tensor;
confusion-matrix rows sum to 1."""

from __future__ import annotations

import math

import torch

from cvlab.evaluation.metrics import (
    confusion_matrix,
    mean_per_class_accuracy,
    per_class_accuracy,
    top1_accuracy,
)

# logits -> preds = [0, 1, 1]; labels = [0, 1, 0] -> correct = [True, True, False]
_LOGITS = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])
_LABELS = torch.tensor([0, 1, 0])


def test_top1_matches_hand_computed_value() -> None:
    acc = top1_accuracy(_LOGITS, _LABELS)
    assert math.isclose(acc, 2 / 3, rel_tol=1e-6)


def test_per_class_accuracy_hand_computed() -> None:
    # class 0: true idx [0, 2], preds [0, 1] -> 1/2 correct
    # class 1: true idx [1], pred [1] -> 1/1 correct
    accs = per_class_accuracy(_LOGITS, _LABELS, num_classes=2)
    assert math.isclose(accs[0].item(), 0.5, rel_tol=1e-6)
    assert math.isclose(accs[1].item(), 1.0, rel_tol=1e-6)


def test_mean_per_class_accuracy_hand_computed() -> None:
    # mean(0.5, 1.0) = 0.75, distinct from top1 (2/3) -- guards against the two
    # metrics accidentally computing the same formula.
    mpca = mean_per_class_accuracy(_LOGITS, _LABELS, num_classes=2)
    assert math.isclose(mpca, 0.75, rel_tol=1e-6)
    assert not math.isclose(mpca, top1_accuracy(_LOGITS, _LABELS), rel_tol=1e-6)


def test_per_class_accuracy_nan_for_unseen_class() -> None:
    accs = per_class_accuracy(_LOGITS, _LABELS, num_classes=3)
    assert math.isnan(accs[2].item())


def test_confusion_matrix_rows_sum_to_one() -> None:
    num_classes = 2
    cm = confusion_matrix(_LOGITS, _LABELS, num_classes=num_classes)
    assert cm.shape == (num_classes, num_classes)
    row_sums = cm.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(num_classes, dtype=torch.float64))


def test_confusion_matrix_hand_computed_values() -> None:
    # true=0,pred=0 (x1); true=1,pred=1 (x1); true=0,pred=1 (x1)
    cm = confusion_matrix(_LOGITS, _LABELS, num_classes=2)
    expected = torch.tensor([[0.5, 0.5], [0.0, 1.0]], dtype=torch.float64)
    assert torch.allclose(cm, expected)


def test_confusion_matrix_zero_row_for_unseen_class_not_nan() -> None:
    cm = confusion_matrix(_LOGITS, _LABELS, num_classes=3)
    assert torch.equal(cm[2], torch.zeros(3, dtype=torch.float64))
    assert not torch.isnan(cm).any()


def test_top1_accuracy_on_larger_random_batch_matches_manual_loop() -> None:
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(50, 5, generator=generator)
    labels = torch.randint(0, 5, (50,), generator=generator)
    preds = logits.argmax(dim=1)
    manual = sum(int(p == y) for p, y in zip(preds.tolist(), labels.tolist())) / 50
    assert math.isclose(top1_accuracy(logits, labels), manual, rel_tol=1e-6)
