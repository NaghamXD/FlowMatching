"""M3 test: extract -> reload -> tensors identical; re-running an unchanged config
is a no-op; changing the transform in the config invalidates the cache.

Uses a synthetic dataset + encoder (not real DTD/ResNet18) so this stays fast and
tests only the cache mechanics that M3 owns — encoder/data correctness is already
covered in tests/encoders and tests/data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import PIL.Image
import pytest
import torch

import cvlab.features.extract as extract_mod
from cvlab.config.schema import DatasetConfig, EncoderConfig
from cvlab.features.cache import load


class _FakeTransform:
    """A stand-in preprocessing transform whose repr is controllable, so tests can
    simulate 'the transform changed' without depending on real encoder internals."""

    def __init__(self, resolution: int, tag: str) -> None:
        self._resolution = resolution
        self._tag = tag

    def __call__(self, image: PIL.Image.Image) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.float32).reshape(-1, 3)
        channel_means = torch.from_numpy(arr.mean(axis=0))
        return channel_means.view(3, 1, 1).expand(3, self._resolution, self._resolution).clone()

    def __repr__(self) -> str:
        return f"FakeTransform(resolution={self._resolution}, tag={self._tag!r})"


class _FakeEncoder:
    feature_dim = 3

    def __init__(self, resolution: int = 4, tag: str = "v1") -> None:
        self.calls = 0
        self.transform = _FakeTransform(resolution, tag)

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return batch.mean(dim=(2, 3))


class _FakeDataset:
    def __init__(self, n: int = 7, num_classes: int = 3) -> None:
        rng = np.random.RandomState(0)
        self._images = [
            PIL.Image.fromarray(rng.randint(0, 256, size=(4, 4, 3), dtype=np.uint8))
            for _ in range(n)
        ]
        self._labels = [i % num_classes for i in range(n)]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[PIL.Image.Image, int]:
        return self._images[idx], self._labels[idx]


def _expected_channel_means(dataset: _FakeDataset) -> torch.Tensor:
    rows = [np.asarray(img, dtype=np.float32).reshape(-1, 3).mean(axis=0) for img in dataset._images]
    return torch.from_numpy(np.stack(rows))


@pytest.fixture
def configs() -> tuple[DatasetConfig, EncoderConfig]:
    dataset_cfg = DatasetConfig(name="dtd", root="unused")
    encoder_cfg = EncoderConfig(
        name="resnet18", checkpoint_id="fake-ckpt", feature_dim=3, input_resolution=4
    )
    return dataset_cfg, encoder_cfg


def test_extract_then_reload_matches_expected_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configs: tuple[DatasetConfig, EncoderConfig]
) -> None:
    dataset_cfg, encoder_cfg = configs
    dataset = _FakeDataset(n=7)
    encoder = _FakeEncoder(resolution=4, tag="v1")

    monkeypatch.setattr(extract_mod, "load_dataset", lambda cfg, split: dataset)
    monkeypatch.setattr(extract_mod, "load_encoder", lambda cfg, device="cpu": encoder)

    extract_mod.extract_split(dataset_cfg, encoder_cfg, "train", tmp_path, batch_size=3)

    Z, y = load(tmp_path, "dtd", "train", "resnet18")
    assert torch.allclose(Z, _expected_channel_means(dataset), atol=1e-4)
    assert y.tolist() == dataset._labels
    assert encoder.calls > 0


def test_rerunning_unchanged_config_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configs: tuple[DatasetConfig, EncoderConfig]
) -> None:
    dataset_cfg, encoder_cfg = configs
    dataset = _FakeDataset(n=7)
    encoder = _FakeEncoder(resolution=4, tag="v1")

    monkeypatch.setattr(extract_mod, "load_dataset", lambda cfg, split: dataset)
    monkeypatch.setattr(extract_mod, "load_encoder", lambda cfg, device="cpu": encoder)

    extract_mod.extract_split(dataset_cfg, encoder_cfg, "train", tmp_path, batch_size=3)
    calls_after_first_run = encoder.calls
    assert calls_after_first_run > 0

    extract_mod.extract_split(dataset_cfg, encoder_cfg, "train", tmp_path, batch_size=3)
    assert encoder.calls == calls_after_first_run, "unchanged config re-ran the encoder"


def test_changing_transform_invalidates_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configs: tuple[DatasetConfig, EncoderConfig]
) -> None:
    dataset_cfg, encoder_cfg = configs
    dataset = _FakeDataset(n=7)
    encoder_v1 = _FakeEncoder(resolution=4, tag="v1")

    monkeypatch.setattr(extract_mod, "load_dataset", lambda cfg, split: dataset)
    monkeypatch.setattr(extract_mod, "load_encoder", lambda cfg, device="cpu": encoder_v1)
    extract_mod.extract_split(dataset_cfg, encoder_cfg, "train", tmp_path, batch_size=3)
    assert encoder_v1.calls > 0

    encoder_v2 = _FakeEncoder(resolution=4, tag="v2")  # simulates a transform config change
    monkeypatch.setattr(extract_mod, "load_encoder", lambda cfg, device="cpu": encoder_v2)
    extract_mod.extract_split(dataset_cfg, encoder_cfg, "train", tmp_path, batch_size=3)
    assert encoder_v2.calls > 0, "changed transform did not trigger re-extraction"
