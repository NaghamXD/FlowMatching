"""Base interface for frozen backbones. This package is the only one that touches
raw pixel values — every encoder exposes its own preprocessing transform, a
batched `encode_image`, and its output `feature_dim`."""

from __future__ import annotations

from typing import Callable, Protocol

import PIL.Image
import torch


class Encoder(Protocol):
    feature_dim: int
    transform: Callable[[PIL.Image.Image], torch.Tensor]

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor: ...
