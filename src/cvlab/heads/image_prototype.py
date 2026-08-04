"""Image-derived class-prototype head.

Despite being informally called "k-means" in discussion, this is NOT k-means
clustering -- it's a deterministic class-mean prototype:

    mu_c = normalize( mean_{i in S_c} ( normalize(z_i) ) )
    y_hat = argmax_c cos(z, mu_c)

No optimizer, no gradients, no randomness: two `fit` calls on the same subset
produce bit-identical prototypes. See DECISIONS.md decisions 11-12 for why
`temperature` is a Stage-2 seam only and must not be tuned or compared across
heads in Stage 1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class ImagePrototype:
    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature
        self._prototypes: torch.Tensor | None = None

    def fit(
        self,
        Z_train: torch.Tensor,
        y_train: torch.Tensor,
        Z_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
    ) -> None:
        # y_train is guaranteed 0-indexed and contiguous by the data layer, and
        # balanced_kshot never drops a class, so every row 0..num_classes-1 below
        # is populated from at least one example.
        num_classes = int(y_train.max().item()) + 1
        z_norm = F.normalize(Z_train, dim=1)
        prototypes = torch.zeros(num_classes, Z_train.shape[1], dtype=Z_train.dtype)
        for cls in range(num_classes):
            prototypes[cls] = z_norm[y_train == cls].mean(dim=0)
        self._prototypes = F.normalize(prototypes, dim=1)

    def logits(self, Z: torch.Tensor) -> torch.Tensor:
        if self._prototypes is None:
            raise RuntimeError("call fit() before logits()")
        z_norm = F.normalize(Z, dim=1)
        return self.temperature * (z_norm @ self._prototypes.T)

    @property
    def prototypes(self) -> torch.Tensor:
        if self._prototypes is None:
            raise RuntimeError("call fit() before accessing prototypes")
        return self._prototypes
