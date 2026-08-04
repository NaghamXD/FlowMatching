"""Registry of frozen encoder backbones, keyed by name."""

from __future__ import annotations

import torch

from cvlab.config.schema import EncoderConfig
from cvlab.encoders.base import Encoder
from cvlab.encoders.dinov2 import DINOv2ViTS14Encoder
from cvlab.encoders.resnet import ResNet18Encoder


def load_encoder(config: EncoderConfig, device: str | torch.device = "cpu") -> Encoder:
    if config.name == "resnet18":
        return ResNet18Encoder(device=device)
    if config.name == "dinov2_vits14":
        return DINOv2ViTS14Encoder(device=device, input_resolution=config.input_resolution)
    raise ValueError(f"Unknown encoder {config.name!r}")
