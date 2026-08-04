from __future__ import annotations

import pytest

from cvlab.config.schema import (
    DatasetConfig,
    EncoderConfig,
    ImagePrototypeConfig,
    LinearProbeConfig,
    MethodConfig,
    RunConfig,
    parse_k_setting,
)


def test_dataset_config_rejects_non_partition_1_dtd() -> None:
    with pytest.raises(ValueError):
        DatasetConfig(name="dtd", root="data/dtd", partition=2)


def test_dataset_config_flowers_ignores_partition() -> None:
    cfg = DatasetConfig(name="flowers102", root="data/flowers102", partition=3)
    assert cfg.name == "flowers102"


@pytest.mark.parametrize("k", [1, 5, 10, "full"])
def test_parse_k_setting_accepts_valid_values(k) -> None:
    assert parse_k_setting(k) == k


@pytest.mark.parametrize("k", [0, -1, "5", "partial", True, 3.5])
def test_parse_k_setting_rejects_invalid_values(k) -> None:
    with pytest.raises(ValueError):
        parse_k_setting(k)


def test_method_config_defaults_are_independently_typed() -> None:
    cfg = MethodConfig(name="linear_probe")
    assert isinstance(cfg.linear_probe, LinearProbeConfig)
    assert isinstance(cfg.image_prototype, ImagePrototypeConfig)
    assert cfg.linear_probe.patience is None
    assert cfg.image_prototype.temperature == 1.0


def test_run_config_round_trips_and_validates_k() -> None:
    run = RunConfig(
        dataset=DatasetConfig(name="dtd", root="data/dtd"),
        encoder=EncoderConfig(
            name="resnet18", checkpoint_id="IMAGENET1K_V1", feature_dim=512, input_resolution=224
        ),
        method=MethodConfig(name="linear_probe"),
        k=5,
        seed=0,
    )
    assert run.k == 5
    assert run.seed == 0

    with pytest.raises(ValueError):
        RunConfig(
            dataset=DatasetConfig(name="dtd", root="data/dtd"),
            encoder=EncoderConfig(
                name="resnet18",
                checkpoint_id="IMAGENET1K_V1",
                feature_dim=512,
                input_resolution=224,
            ),
            method=MethodConfig(name="linear_probe"),
            k="partial",
            seed=0,
        )
