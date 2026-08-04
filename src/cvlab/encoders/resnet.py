"""Frozen ImageNet-1K ResNet-18 encoder. Extracts the 512-d penultimate
(global-average-pooled) representation by replacing `fc` with identity."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNet18Encoder:
    feature_dim = 512

    def __init__(self, device: str | torch.device = "cpu") -> None:
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1
        model = tvm.resnet18(weights=weights)
        model.fc = nn.Identity()
        model.requires_grad_(False)
        model.eval()
        self._device = torch.device(device)
        self._model = model.to(self._device)
        self.transform = weights.transforms()

    @torch.no_grad()
    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        return self._model(batch.to(self._device)).cpu()
