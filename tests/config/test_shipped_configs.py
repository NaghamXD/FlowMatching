"""Every shipped YAML fragment under configs/ must load into its dataclass without error,
so schema drift is caught here rather than at sweep time."""

from __future__ import annotations

from pathlib import Path

import pytest

from cvlab.config.loader import load_dataclass, load_yaml
from cvlab.config.schema import DatasetConfig, EncoderConfig, MethodConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"


@pytest.mark.parametrize("name", ["dtd", "flowers102"])
def test_dataset_configs_load(name: str) -> None:
    cfg = load_dataclass(DatasetConfig, load_yaml(CONFIGS / "datasets" / f"{name}.yaml"))
    assert cfg.name == name


@pytest.mark.parametrize("name", ["resnet18", "dinov2_vits14"])
def test_encoder_configs_load(name: str) -> None:
    cfg = load_dataclass(EncoderConfig, load_yaml(CONFIGS / "encoders" / f"{name}.yaml"))
    assert cfg.name == name
    assert cfg.feature_dim > 0


@pytest.mark.parametrize("name", ["linear_probe", "image_prototype"])
def test_method_configs_load(name: str) -> None:
    cfg = load_dataclass(MethodConfig, load_yaml(CONFIGS / "methods" / f"{name}.yaml"))
    assert cfg.name == name
