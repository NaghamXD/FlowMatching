"""Shared fixtures for reporting tests: a synthetic, schema-correct 48-run
results store, built without running any real experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
import torch

from cvlab.results.schema import RunRecord
from cvlab.results.store import save
from cvlab.runner.experiment import RunSpec
from cvlab.runner.sweep import enumerate_stage1_runs
from cvlab.utils.io import ensure_dir

# Matches the real official splits (decision 15): DTD 1880/47=40, Flowers 1020/102=10.
TRAIN_SIZE = {"dtd": 1880, "flowers102": 1020}
NUM_CLASSES = {"dtd": 47, "flowers102": 102}
SYNTHETIC_MAX_EPOCHS = 200


def _default_top1(spec: RunSpec, index: int) -> float:
    return 0.5 + 0.01 * (index % 5)


def _synthetic_epoch_curves(seed: int, max_epochs: int = SYNTHETIC_MAX_EPOCHS) -> list[dict]:
    """A plausible, deterministic loss/accuracy curve -- decaying train/val loss,
    rising val accuracy -- so M10's training-curves figure has real data to draw."""
    return [
        {
            "epoch": epoch,
            "step": epoch * 5,
            "train_loss": 2.0 / epoch + seed * 0.001,
            "train_acc": min(1.0, epoch / max_epochs),
            "val_loss": 2.0 / epoch + 0.5 * epoch / max_epochs,
            "val_acc": min(0.9, epoch / max_epochs * 0.9),
        }
        for epoch in range(1, max_epochs + 1)
    ]


def _synthetic_predictions(
    num_classes: int, seed: int, per_class: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """A plausible, imperfect classifier's predictions -- mostly correct, with a
    deterministic slice of off-by-one confusions -- so M11's confusion-matrix
    artifacts have real (non-degenerate) structure to draw."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(num_classes).repeat_interleave(per_class)
    preds = labels.clone()
    noise_count = max(1, len(labels) // 6)
    noise_idx = torch.randperm(len(labels), generator=generator)[:noise_count]
    preds[noise_idx] = (labels[noise_idx] + 1) % num_classes
    return preds, labels


def populate_synthetic_store(
    runs_root: str | Path, top1_fn: Callable[[RunSpec], float] | None = None
) -> None:
    specs = enumerate_stage1_runs()
    for i, spec in enumerate(specs):
        top1 = top1_fn(spec) if top1_fn else _default_top1(spec, i)
        is_linear_probe = spec.method == "linear_probe"
        num_classes = NUM_CLASSES[spec.dataset]
        predictions_path = Path(runs_root) / spec.run_id / "predictions.pt"
        record = RunRecord(
            run_id=spec.run_id,
            config={
                "dataset": {"name": spec.dataset},
                "encoder": {"name": spec.encoder},
                "method": {"name": spec.method},
                "k": spec.k,
                "seed": spec.seed,
            },
            git_commit="unknown",
            feature_cache_manifest_hashes={"train": "x", "test": "y"},
            test_top1=top1,
            mean_per_class_accuracy=max(0.0, top1 - 0.02),
            train_size=TRAIN_SIZE[spec.dataset],
            num_classes=num_classes,
            best_epoch=42 if is_linear_probe else None,
            epoch_curves=_synthetic_epoch_curves(spec.seed) if is_linear_probe else [],
            predictions_path=str(predictions_path),
        )
        save(runs_root, record)

        ensure_dir(predictions_path.parent)
        preds, labels = _synthetic_predictions(num_classes, seed=i)
        torch.save({"preds": preds, "labels": labels}, predictions_path)


@pytest.fixture
def populate_store() -> Callable[..., None]:
    return populate_synthetic_store


@pytest.fixture
def synthetic_runs_root(tmp_path: Path) -> Path:
    runs_root = tmp_path / "runs"
    populate_synthetic_store(runs_root)
    return runs_root
