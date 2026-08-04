"""Head-agnostic metrics from logits + labels, plus aggregation over seeds."""

from cvlab.evaluation.aggregate import Aggregate, aggregate
from cvlab.evaluation.metrics import (
    confusion_matrix,
    mean_per_class_accuracy,
    per_class_accuracy,
    top1_accuracy,
)

__all__ = [
    "Aggregate",
    "aggregate",
    "confusion_matrix",
    "mean_per_class_accuracy",
    "per_class_accuracy",
    "top1_accuracy",
]
