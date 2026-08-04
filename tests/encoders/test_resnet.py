"""M2 test: output shape matches declared feature_dim, every parameter is frozen,
and two forward passes on the same batch are bit-identical."""

from __future__ import annotations

import PIL.Image
import pytest
import torch

from cvlab.encoders.resnet import ResNet18Encoder


@pytest.fixture(scope="module")
def encoder() -> ResNet18Encoder:
    return ResNet18Encoder()


def test_output_shape_matches_feature_dim(encoder: ResNet18Encoder) -> None:
    batch = torch.rand(4, 3, 224, 224)
    out = encoder.encode_image(batch)
    assert out.shape == (4, encoder.feature_dim)
    assert encoder.feature_dim == 512


def test_all_parameters_frozen(encoder: ResNet18Encoder) -> None:
    assert all(not p.requires_grad for p in encoder._model.parameters())


def test_two_forward_passes_are_bit_identical(encoder: ResNet18Encoder) -> None:
    batch = torch.rand(4, 3, 224, 224)
    out1 = encoder.encode_image(batch)
    out2 = encoder.encode_image(batch)
    assert torch.equal(out1, out2)


def test_transform_applies_to_pil_image(encoder: ResNet18Encoder) -> None:
    image = PIL.Image.new("RGB", (300, 400), color=(128, 64, 32))
    tensor = encoder.transform(image)
    assert tensor.shape == (3, 224, 224)
    batch = tensor.unsqueeze(0)
    out = encoder.encode_image(batch)
    assert out.shape == (1, 512)
