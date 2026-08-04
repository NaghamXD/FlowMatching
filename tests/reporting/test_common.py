"""M9 test: images_per_class is computed from the actual split sizes, and the
decision-15 assertion fires if Flowers is not exactly 10/class."""

from __future__ import annotations

import pytest

from cvlab.reporting.common import images_per_class
from cvlab.reporting.loader import RunCell
from cvlab.results.schema import RunRecord


def _record(train_size: int, num_classes: int) -> RunRecord:
    return RunRecord(
        run_id="x",
        config={},
        git_commit="unknown",
        feature_cache_manifest_hashes={},
        test_top1=0.5,
        mean_per_class_accuracy=0.5,
        train_size=train_size,
        num_classes=num_classes,
        best_epoch=None,
        epoch_curves=[],
        predictions_path="x",
    )


def test_images_per_class_for_k5_and_k10_is_k_itself() -> None:
    # Deliberately mismatched train_size/num_classes to prove K=5/10 never touch them.
    cell = RunCell("dtd", "resnet18", "linear_probe", 5, (_record(999, 3),))
    assert images_per_class(cell) == 5
    cell10 = RunCell("dtd", "resnet18", "linear_probe", 10, (_record(999, 3),))
    assert images_per_class(cell10) == 10


def test_images_per_class_full_dtd_matches_decision_15() -> None:
    cell = RunCell("dtd", "resnet18", "linear_probe", "full", (_record(1880, 47),))
    assert images_per_class(cell) == 40


def test_images_per_class_full_flowers_matches_decision_15() -> None:
    cell = RunCell("flowers102", "resnet18", "linear_probe", "full", (_record(1020, 102),))
    assert images_per_class(cell) == 10


def test_decision_15_assertion_fires_on_flowers_mismatch() -> None:
    # Simulates a torchvision version where Flowers no longer has exactly 10/class.
    cell = RunCell("flowers102", "resnet18", "linear_probe", "full", (_record(1122, 102),))
    with pytest.raises(AssertionError, match="decision 15"):
        images_per_class(cell)


def test_decision_15_assertion_fires_on_dtd_mismatch() -> None:
    # 2000 // 47 = 42, not the expected 40 -- a genuine mismatch (1900 // 47 would
    # coincidentally still floor to 40, so it wouldn't actually exercise the check).
    cell = RunCell("dtd", "resnet18", "linear_probe", "full", (_record(2000, 47),))
    with pytest.raises(AssertionError, match="decision 15"):
        images_per_class(cell)
