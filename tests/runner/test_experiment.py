from __future__ import annotations

from pathlib import Path

import torch

from cvlab.config.schema import DatasetConfig, EncoderConfig, LinearProbeConfig, MethodConfig
from cvlab.features.cache import save as cache_save
from cvlab.features.manifest import FeatureManifest
from cvlab.heads.image_prototype import ImagePrototype
from cvlab.heads.linear import LinearProbe
from cvlab.runner.experiment import RunSpec, run_single


def _synthetic_features(num_classes: int, per_class: int, dim: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, dim, generator=generator) * 5.0
    Z, y = [], []
    for cls in range(num_classes):
        Z.append(centers[cls] + 0.1 * torch.randn(per_class, dim, generator=generator))
        y.extend([cls] * per_class)
    return torch.cat(Z), torch.tensor(y, dtype=torch.long)


def _write_cache(cache_root: Path, dataset: str, encoder: str, split: str, Z: torch.Tensor, y: torch.Tensor) -> None:
    manifest = FeatureManifest(
        dataset=dataset,
        split=split,
        encoder=encoder,
        checkpoint_id="fake-ckpt",
        transform_description="fake-transform",
        dtype="float32",
        shape=tuple(Z.shape),
        code_version="unknown",
    )
    cache_save(cache_root, dataset, split, encoder, Z, y, manifest)


def test_run_id_format() -> None:
    spec = RunSpec(dataset="dtd", encoder="resnet18", method="linear_probe", k=5, seed=1)
    assert spec.run_id == "dtd_resnet18_linear_probe_k5_seed1"
    spec_full = RunSpec(dataset="dtd", encoder="resnet18", method="image_prototype", k="full", seed=0)
    assert spec_full.run_id == "dtd_resnet18_image_prototype_kfull_seed0"


def test_run_single_image_prototype_end_to_end_and_resumable(tmp_path: Path, monkeypatch) -> None:
    cache_root = tmp_path / "cache"
    runs_root = tmp_path / "runs"
    dim, num_classes = 8, 4

    Z_train, y_train = _synthetic_features(num_classes, 12, dim, seed=0)
    Z_test, y_test = _synthetic_features(num_classes, 6, dim, seed=1)
    _write_cache(cache_root, "dtd", "resnet18", "train", Z_train, y_train)
    _write_cache(cache_root, "dtd", "resnet18", "test", Z_test, y_test)

    dataset_cfg = DatasetConfig(name="dtd", root="unused")
    encoder_cfg = EncoderConfig(name="resnet18", checkpoint_id="fake", feature_dim=dim, input_resolution=224)
    method_cfg = MethodConfig(name="image_prototype")
    spec = RunSpec(dataset="dtd", encoder="resnet18", method="image_prototype", k="full", seed=0)

    calls = {"n": 0}
    original_fit = ImagePrototype.fit

    def counting_fit(self, *a, **kw):
        calls["n"] += 1
        return original_fit(self, *a, **kw)

    monkeypatch.setattr(ImagePrototype, "fit", counting_fit)

    record1 = run_single(spec, dataset_cfg, encoder_cfg, method_cfg, cache_root, runs_root)
    assert calls["n"] == 1
    assert record1.best_epoch is None
    assert record1.epoch_curves == []
    assert 0.0 <= record1.test_top1 <= 1.0
    assert record1.config["k"] == "full"
    assert record1.config["dataset"]["name"] == "dtd"
    assert set(record1.feature_cache_manifest_hashes) == {"train", "test"}

    # Resumable: calling again must not re-fit, and must return the identical record.
    record2 = run_single(spec, dataset_cfg, encoder_cfg, method_cfg, cache_root, runs_root)
    assert calls["n"] == 1
    assert record1 == record2


def test_run_single_linear_probe_uses_full_val_split_and_records_curves(tmp_path: Path, monkeypatch) -> None:
    cache_root = tmp_path / "cache"
    runs_root = tmp_path / "runs"
    dim, num_classes = 8, 4

    Z_train, y_train = _synthetic_features(num_classes, 12, dim, seed=0)
    Z_val, y_val = _synthetic_features(num_classes, 5, dim, seed=1)
    Z_test, y_test = _synthetic_features(num_classes, 5, dim, seed=2)
    _write_cache(cache_root, "dtd", "resnet18", "train", Z_train, y_train)
    _write_cache(cache_root, "dtd", "resnet18", "val", Z_val, y_val)
    _write_cache(cache_root, "dtd", "resnet18", "test", Z_test, y_test)

    dataset_cfg = DatasetConfig(name="dtd", root="unused")
    encoder_cfg = EncoderConfig(name="resnet18", checkpoint_id="fake", feature_dim=dim, input_resolution=224)
    method_cfg = MethodConfig(name="linear_probe", linear_probe=LinearProbeConfig(max_epochs=5))
    spec = RunSpec(dataset="dtd", encoder="resnet18", method="linear_probe", k=5, seed=0)

    seen_val_sizes = []
    original_fit = LinearProbe.fit

    def spying_fit(self, Z_train_, y_train_, Z_val_=None, y_val_=None):
        seen_val_sizes.append(Z_val_.shape[0] if Z_val_ is not None else None)
        assert Z_train_.shape[0] == 5 * num_classes  # K=5, balanced across 4 classes
        return original_fit(self, Z_train_, y_train_, Z_val_, y_val_)

    monkeypatch.setattr(LinearProbe, "fit", spying_fit)

    record = run_single(spec, dataset_cfg, encoder_cfg, method_cfg, cache_root, runs_root)

    # Full val split (5 * num_classes = 20), not a K-shot subset of it.
    assert seen_val_sizes == [5 * num_classes]
    assert record.best_epoch is not None
    assert len(record.epoch_curves) == 5
    assert "val" in record.feature_cache_manifest_hashes
