"""M2 test: output shape matches declared feature_dim, every parameter is frozen,
and two forward passes on the same batch are bit-identical."""

from __future__ import annotations

import PIL.Image
import pytest
import torch

from cvlab.encoders.dinov2 import DINOv2ViTS14Encoder


@pytest.fixture(scope="module")
def encoder() -> DINOv2ViTS14Encoder:
    return DINOv2ViTS14Encoder()


def test_output_shape_matches_feature_dim(encoder: DINOv2ViTS14Encoder) -> None:
    batch = torch.rand(4, 3, 224, 224)
    out = encoder.encode_image(batch)
    assert out.shape == (4, encoder.feature_dim)
    assert encoder.feature_dim == 384


def test_all_parameters_frozen(encoder: DINOv2ViTS14Encoder) -> None:
    assert all(not p.requires_grad for p in encoder._model.parameters())


def test_two_forward_passes_are_bit_identical(encoder: DINOv2ViTS14Encoder) -> None:
    batch = torch.rand(4, 3, 224, 224)
    out1 = encoder.encode_image(batch)
    out2 = encoder.encode_image(batch)
    assert torch.equal(out1, out2)


def test_transform_applies_to_pil_image(encoder: DINOv2ViTS14Encoder) -> None:
    image = PIL.Image.new("RGB", (300, 400), color=(128, 64, 32))
    tensor = encoder.transform(image)
    assert tensor.shape == (3, 224, 224)
    batch = tensor.unsqueeze(0)
    out = encoder.encode_image(batch)
    assert out.shape == (1, 384)


def test_no_register_tokens(encoder: DINOv2ViTS14Encoder) -> None:
    batch = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        features = encoder._model.forward_features(batch)
    assert features["x_norm_regtokens"].shape == (1, 0, 384)
