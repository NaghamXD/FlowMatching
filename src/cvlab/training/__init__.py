"""Optimization loop for parametric heads only (prototype heads bypass this entirely)."""

from cvlab.training.curves import EpochRecord, TrainingResult, tie_break_key
from cvlab.training.trainer import train_linear_probe

__all__ = ["EpochRecord", "TrainingResult", "tie_break_key", "train_linear_probe"]
