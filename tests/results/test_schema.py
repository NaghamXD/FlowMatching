"""M7 test: a run record round-trips through JSON without loss."""

from __future__ import annotations

import json

from cvlab.results.schema import RunRecord


def _sample_record() -> RunRecord:
    return RunRecord(
        run_id="dtd_resnet18_linear_probe_k5_seed0",
        config={
            "dataset": {"name": "dtd", "root": "data/dtd", "partition": 1},
            "encoder": {"name": "resnet18", "checkpoint_id": "IMAGENET1K_V1", "feature_dim": 512},
            "method": {"name": "linear_probe"},
            "k": 5,
            "seed": 0,
        },
        git_commit="unknown",
        feature_cache_manifest_hashes={"train": "abc123", "val": "def456", "test": "ghi789"},
        test_top1=0.6335,
        mean_per_class_accuracy=0.61,
        best_epoch=23,
        epoch_curves=[
            {"epoch": 1, "step": 30, "train_loss": 2.1, "train_acc": 0.3, "val_loss": 2.0, "val_acc": 0.35},
            {"epoch": 2, "step": 60, "train_loss": 1.8, "train_acc": 0.4, "val_loss": 1.9, "val_acc": 0.4},
        ],
        predictions_path="runs/dtd_resnet18_linear_probe_k5_seed0/predictions.pt",
    )


def test_round_trips_through_dict_without_loss() -> None:
    record = _sample_record()
    assert RunRecord.from_dict(record.to_dict()) == record


def test_round_trips_through_actual_json_serialization() -> None:
    record = _sample_record()
    serialized = json.dumps(record.to_dict())
    restored = RunRecord.from_dict(json.loads(serialized))
    assert restored == record
    assert restored.epoch_curves == record.epoch_curves
    assert restored.feature_cache_manifest_hashes == record.feature_cache_manifest_hashes


def test_prototype_run_has_none_best_epoch_and_empty_curves() -> None:
    record = RunRecord(
        run_id="dtd_resnet18_image_prototype_full",
        config={"method": {"name": "image_prototype"}},
        git_commit="unknown",
        feature_cache_manifest_hashes={"train": "abc", "test": "ghi"},
        test_top1=0.5878,
        mean_per_class_accuracy=0.58,
        best_epoch=None,
        epoch_curves=[],
        predictions_path="runs/dtd_resnet18_image_prototype_full/predictions.pt",
    )
    serialized = json.dumps(record.to_dict())
    restored = RunRecord.from_dict(json.loads(serialized))
    assert restored.best_epoch is None
    assert restored.epoch_curves == []
    assert restored == record
