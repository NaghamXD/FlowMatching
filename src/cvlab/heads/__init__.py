"""Classifier heads behind one `fit`/`logits` interface, wide enough for a Stage-2
flow-matching head to satisfy it unchanged."""

from cvlab.heads.base import Head
from cvlab.heads.image_prototype import ImagePrototype
from cvlab.heads.linear import LinearProbe

__all__ = ["Head", "ImagePrototype", "LinearProbe"]
