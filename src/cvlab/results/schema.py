"""Run-record schema: one JSON record per run, containing exactly what a later
reporting module would need (curves, predictions, confusion-matrix inputs) --
full config, git commit, feature-cache manifest hashes, test top-1, per-epoch
curves, and a path to saved predictions. Reporting reads only this; it never
re-runs experiments."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    config: dict[str, Any]
    git_commit: str
    feature_cache_manifest_hashes: dict[str, str]
    test_top1: float
    mean_per_class_accuracy: float
    # Full official train-split size and class count (not the K-shot subset size) --
    # lets reporting compute/assert images-per-class at K="full" without reading
    # cached features (decision 15 in the reporting spec).
    train_size: int
    num_classes: int
    # None for training-free heads (ImagePrototype); an epoch index for LinearProbe.
    best_epoch: int | None
    # Empty for training-free heads; one dict per epoch (see EpochRecord) for LinearProbe.
    epoch_curves: list[dict[str, Any]]
    predictions_path: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        return cls(**data)
