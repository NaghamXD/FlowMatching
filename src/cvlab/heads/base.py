"""Common classifier-head interface. Deliberately wide enough that a Stage-2
flow-matching head can satisfy it unchanged: `fit` on cached features (with
optional val tensors for model selection), `logits` for evaluation."""

from __future__ import annotations

from typing import Protocol

import torch


class Head(Protocol):
    def fit(
        self,
        Z_train: torch.Tensor,
        y_train: torch.Tensor,
        Z_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
    ) -> None: ...

    def logits(self, Z: torch.Tensor) -> torch.Tensor: ...
