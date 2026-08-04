"""Trained linear-probe head: s = W z + b, softmax cross-entropy. Only W, b are
trained. Decision 3: features are not L2-normalized here, for either encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from cvlab.config.schema import LinearProbeConfig
from cvlab.training.curves import TrainingResult
from cvlab.training.trainer import train_linear_probe


class LinearProbe:
    def __init__(
        self, num_classes: int, feature_dim: int, config: LinearProbeConfig, seed: int
    ) -> None:
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.config = config
        self.seed = seed
        self._model: nn.Linear | None = None
        self.training_result: TrainingResult | None = None

    def fit(
        self,
        Z_train: torch.Tensor,
        y_train: torch.Tensor,
        Z_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
    ) -> None:
        if Z_val is None or y_val is None:
            raise ValueError(
                "LinearProbe.fit requires Z_val/y_val: checkpoint selection is "
                "always by best validation accuracy (decision 5)"
            )
        self._model, self.training_result = train_linear_probe(
            Z_train,
            y_train,
            Z_val,
            y_val,
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            config=self.config,
            seed=self.seed,
        )

    def logits(self, Z: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("call fit() before logits()")
        self._model.eval()
        with torch.no_grad():
            return self._model(Z)
