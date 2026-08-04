"""Orchestrates one Stage 1 run end-to-end: load cached features -> (optionally)
K-shot subset the train split -> fit a head -> evaluate on test -> save predictions
+ a RunRecord. This is the one place that wires every other package together for
a single run; `sweep.py` just calls `run_single` once per enumerated spec."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from cvlab.config.schema import DatasetConfig, EncoderConfig, MethodConfig, RunConfig
from cvlab.evaluation.metrics import mean_per_class_accuracy, top1_accuracy
from cvlab.features.cache import load as load_features
from cvlab.features.cache import load_manifest
from cvlab.features.manifest import manifest_hash
from cvlab.heads.image_prototype import ImagePrototype
from cvlab.heads.linear import LinearProbe
from cvlab.results.schema import RunRecord
from cvlab.results.store import exists
from cvlab.results.store import load as load_record
from cvlab.results.store import run_dir
from cvlab.results.store import save as save_record
from cvlab.sampling.fewshot import balanced_kshot
from cvlab.utils.io import ensure_dir
from cvlab.utils.logging import get_logger

logger = get_logger("runner.experiment")


@dataclasses.dataclass(frozen=True)
class RunSpec:
    dataset: str
    encoder: str
    method: str
    k: int | str
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.dataset}_{self.encoder}_{self.method}_k{self.k}_seed{self.seed}"


def run_single(
    spec: RunSpec,
    dataset_cfg: DatasetConfig,
    encoder_cfg: EncoderConfig,
    method_cfg: MethodConfig,
    cache_root: str | Path,
    runs_root: str | Path,
) -> RunRecord:
    """Run one (dataset, encoder, method, K, seed) combination. Resumable: if a
    record for this run_id already exists, it's returned unchanged -- a no-op,
    so an interrupted sweep can always be re-invoked safely."""
    if exists(runs_root, spec.run_id):
        logger.info("run %s already recorded -- skipping (no-op)", spec.run_id)
        return load_record(runs_root, spec.run_id)

    Z_train_full, y_train_full = load_features(cache_root, spec.dataset, "train", spec.encoder)
    Z_test, y_test = load_features(cache_root, spec.dataset, "test", spec.encoder)
    num_classes = max(int(y_train_full.max()), int(y_test.max())) + 1

    if spec.k == "full":
        Z_train, y_train = Z_train_full, y_train_full
    else:
        idx = balanced_kshot(y_train_full, int(spec.k), spec.seed)
        Z_train, y_train = Z_train_full[idx], y_train_full[idx]

    train_manifest = load_manifest(cache_root, spec.dataset, "train", spec.encoder)
    test_manifest = load_manifest(cache_root, spec.dataset, "test", spec.encoder)
    manifest_hashes = {
        "train": manifest_hash(train_manifest),
        "test": manifest_hash(test_manifest),
    }

    if spec.method == "linear_probe":
        # Flagged caveat (DECISIONS.md): the *full* official val split is always
        # used for model selection, even at K=5/K=10 -- it is never subset.
        Z_val, y_val = load_features(cache_root, spec.dataset, "val", spec.encoder)
        manifest_hashes["val"] = manifest_hash(
            load_manifest(cache_root, spec.dataset, "val", spec.encoder)
        )
        head = LinearProbe(
            num_classes=num_classes,
            feature_dim=encoder_cfg.feature_dim,
            config=method_cfg.linear_probe,
            seed=spec.seed,
        )
        head.fit(Z_train, y_train, Z_val, y_val)
        assert head.training_result is not None
        best_epoch: int | None = head.training_result.best_epoch
        epoch_curves = [dataclasses.asdict(r) for r in head.training_result.history]
    elif spec.method == "image_prototype":
        head = ImagePrototype(temperature=method_cfg.image_prototype.temperature)
        head.fit(Z_train, y_train)
        best_epoch = None
        epoch_curves = []
    else:
        raise ValueError(f"Unknown method {spec.method!r}")

    logits = head.logits(Z_test)
    preds = logits.argmax(dim=1)

    predictions_path = run_dir(runs_root, spec.run_id) / "predictions.pt"
    ensure_dir(predictions_path.parent)
    torch.save({"preds": preds, "labels": y_test}, predictions_path)

    run_config = RunConfig(
        dataset=dataset_cfg,
        encoder=encoder_cfg,
        method=method_cfg,
        k=spec.k,
        seed=spec.seed,
    )

    record = RunRecord(
        run_id=spec.run_id,
        config=dataclasses.asdict(run_config),
        git_commit=train_manifest.code_version,
        feature_cache_manifest_hashes=manifest_hashes,
        test_top1=top1_accuracy(logits, y_test),
        mean_per_class_accuracy=mean_per_class_accuracy(logits, y_test, num_classes),
        best_epoch=best_epoch,
        epoch_curves=epoch_curves,
        predictions_path=str(predictions_path),
    )
    save_record(runs_root, record)
    logger.info("run %s -> test_top1=%.4f", spec.run_id, record.test_top1)
    return record
