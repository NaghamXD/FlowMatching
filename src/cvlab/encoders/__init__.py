"""Registry of frozen backbones. The only package that touches pixels."""

from cvlab.encoders.base import Encoder
from cvlab.encoders.registry import load_encoder

__all__ = ["Encoder", "load_encoder"]
