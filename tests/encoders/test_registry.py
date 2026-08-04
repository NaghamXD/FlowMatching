from __future__ import annotations

import pytest

from cvlab.config.schema import EncoderConfig
from cvlab.encoders.dinov2 import DINOv2ViTS14Encoder
from cvlab.encoders.registry import load_encoder
from cvlab.encoders.resnet import ResNet18Encoder


def test_registry_dispatches_resnet18() -> None:
    cfg = EncoderConfig(
        name="resnet18", checkpoint_id="IMAGENET1K_V1", feature_dim=512, input_resolution=224
    )
    enc = load_encoder(cfg)
    assert isinstance(enc, ResNet18Encoder)
    assert enc.feature_dim == cfg.feature_dim


def test_registry_dispatches_dinov2() -> None:
    cfg = EncoderConfig(
        name="dinov2_vits14",
        checkpoint_id="facebookresearch/dinov2:dinov2_vits14",
        feature_dim=384,
        input_resolution=224,
    )
    enc = load_encoder(cfg)
    assert isinstance(enc, DINOv2ViTS14Encoder)
    assert enc.feature_dim == cfg.feature_dim


def test_registry_rejects_unknown_encoder() -> None:
    cfg = EncoderConfig(
        name="resnet18", checkpoint_id="x", feature_dim=512, input_resolution=224
    )
    object.__setattr__(cfg, "name", "not_a_real_encoder")
    with pytest.raises(ValueError):
        load_encoder(cfg)
