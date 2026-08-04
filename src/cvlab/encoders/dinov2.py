"""Frozen DINOv2 ViT-S/14 encoder (no register tokens). Extracts the final,
post-LayerNorm CLS-token representation only — never concatenated with the
mean-pooled patch tokens. See DECISIONS.md decision 9 for checkpoint and
resolution provenance."""

from __future__ import annotations

import torch
import torchvision.transforms as T

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv2ViTS14Encoder:
    feature_dim = 384

    def __init__(
        self, device: str | torch.device = "cpu", input_resolution: int = 224
    ) -> None:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model.requires_grad_(False)
        model.eval()
        self._device = torch.device(device)
        self._model = model.to(self._device)
        self.transform = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(input_resolution),
                T.ToTensor(),
                T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    @torch.no_grad()
    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        features = self._model.forward_features(batch.to(self._device))
        return features["x_norm_clstoken"].cpu()
