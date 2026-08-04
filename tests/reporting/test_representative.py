"""M10 test: representative selection returns the median-accuracy seed on a
synthetic set of three runs with known accuracies, and breaks a deliberate tie
to the lowest seed."""

from __future__ import annotations

import pytest

from cvlab.reporting.representative import select_representative_run
from cvlab.results.schema import RunRecord


def _record(seed: int, top1: float) -> RunRecord:
    return RunRecord(
        run_id=f"run_seed{seed}",
        config={"seed": seed},
        git_commit="unknown",
        feature_cache_manifest_hashes={},
        test_top1=top1,
        mean_per_class_accuracy=top1,
        train_size=470,
        num_classes=47,
        best_epoch=10,
        epoch_curves=[],
        predictions_path="x",
    )


def test_median_accuracy_run_is_selected() -> None:
    records = [_record(0, 0.70), _record(1, 0.50), _record(2, 0.60)]
    chosen = select_representative_run(records)
    assert chosen.config["seed"] == 2  # 0.60 is the median of {0.50, 0.60, 0.70}


def test_deliberate_tie_at_median_breaks_to_lowest_seed() -> None:
    # sorted values: [0.60, 0.60, 0.70] -> median value 0.60, tied between seed 1 and seed 2.
    records = [_record(0, 0.70), _record(1, 0.60), _record(2, 0.60)]
    chosen = select_representative_run(records)
    assert chosen.config["seed"] == 1


def test_tie_break_is_independent_of_input_order() -> None:
    records = [_record(2, 0.60), _record(0, 0.70), _record(1, 0.60)]
    chosen = select_representative_run(records)
    assert chosen.config["seed"] == 1


def test_requires_odd_count() -> None:
    with pytest.raises(ValueError):
        select_representative_run([_record(0, 0.5), _record(1, 0.6)])
