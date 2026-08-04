"""M11 test: rows sum to 1 within 1e-6; the cluster-growing selector returns a
known cluster on a synthetic block-structured matrix; the selector returns
exactly n distinct valid class indices; the masked-diagonal version's vmax
equals the true maximum off-diagonal value."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cvlab.reporting.confusion import (
    assert_row_sums_valid,
    grow_confusion_cluster,
    hierarchical_leaf_order,
    load_confusion_matrix,
    max_off_diagonal,
)
from cvlab.results.schema import RunRecord


def test_assert_row_sums_valid_passes_for_correct_matrix() -> None:
    M = torch.tensor([[0.5, 0.5], [0.0, 1.0]], dtype=torch.float64)
    assert_row_sums_valid(M)


def test_assert_row_sums_valid_accepts_zero_row_for_absent_class() -> None:
    M = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    assert_row_sums_valid(M)  # class 1 had zero test examples -- valid, not an error


def test_assert_row_sums_valid_raises_on_bad_row() -> None:
    M = torch.tensor([[0.5, 0.3], [0.0, 1.0]], dtype=torch.float64)
    with pytest.raises(AssertionError):
        assert_row_sums_valid(M)


def test_load_confusion_matrix_rows_sum_to_one_within_tolerance(tmp_path: Path) -> None:
    preds = torch.tensor([0, 1, 1, 0, 2, 2, 2])
    labels = torch.tensor([0, 1, 0, 0, 2, 2, 1])
    predictions_path = tmp_path / "predictions.pt"
    torch.save({"preds": preds, "labels": labels}, predictions_path)
    record = RunRecord(
        run_id="x", config={}, git_commit="unknown", feature_cache_manifest_hashes={},
        test_top1=0.5, mean_per_class_accuracy=0.5, train_size=100, num_classes=3,
        best_epoch=1, epoch_curves=[], predictions_path=str(predictions_path),
    )
    M = load_confusion_matrix(record)
    assert torch.allclose(M.sum(dim=1), torch.ones(3, dtype=torch.float64), atol=1e-6)
    assert_row_sums_valid(M)  # must not raise


def _block_confused_matrix() -> torch.Tensor:
    """8 classes; {0, 1, 2} are heavily mutually confused, {3..7} are near-perfect."""
    n = 8
    M = torch.eye(n, dtype=torch.float64) * 0.9
    M[0, 0] = M[1, 1] = M[2, 2] = 0.5
    M[0, 1] = M[1, 0] = 0.3
    M[0, 2] = M[2, 0] = 0.2
    M[1, 2] = M[2, 1] = 0.2
    return M


def test_grow_confusion_cluster_finds_known_cluster() -> None:
    M = _block_confused_matrix()
    selected = grow_confusion_cluster(M, 3)
    assert set(selected) == {0, 1, 2}
    assert len(selected) == 3


def test_grow_confusion_cluster_returns_n_distinct_valid_indices() -> None:
    generator = torch.Generator().manual_seed(0)
    M = torch.rand(20, 20, generator=generator, dtype=torch.float32).to(torch.float64)
    selected = grow_confusion_cluster(M, 10)
    assert len(selected) == 10
    assert len(set(selected)) == 10
    assert all(0 <= i < 20 for i in selected)


def test_grow_confusion_cluster_rejects_n_larger_than_matrix() -> None:
    M = torch.eye(3, dtype=torch.float64)
    with pytest.raises(ValueError):
        grow_confusion_cluster(M, 5)


def test_max_off_diagonal_matches_hand_computed_value() -> None:
    M = torch.tensor([[0.9, 0.05, 0.05], [0.3, 0.6, 0.1], [0.02, 0.02, 0.96]], dtype=torch.float64)
    assert max_off_diagonal(M) == pytest.approx(0.3)


def test_hierarchical_leaf_order_is_a_valid_permutation() -> None:
    M = _block_confused_matrix()
    order = hierarchical_leaf_order(M)
    assert sorted(order) == list(range(8))


def test_hierarchical_leaf_order_deterministic_across_calls() -> None:
    M = _block_confused_matrix()
    assert hierarchical_leaf_order(M) == hierarchical_leaf_order(M)
