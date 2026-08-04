"""Deliverable 5 (M12) projection: PCA is the primary, deterministic projection
(decision 22) -- linear, so a projected prototype genuinely sits at the
projected centroid of its class and inter-cluster distances are meaningful.
t-SNE is explicitly optional/secondary in the spec and is not implemented here."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


def joint_pca_projection(
    test_features: torch.Tensor, prototypes: torch.Tensor, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L2-normalizes test features (decision 21 -- prototypes are already
    unit-norm by construction, see heads.image_prototype), stacks
    [test_features; prototypes] into one [N + C, D] matrix, and calls
    `fit_transform` ONCE on the stacked matrix (never fit-then-transform
    separately -- that would not be a joint fit). Returns
    (test_2d, proto_2d, explained_variance_ratio[:2])."""
    normalized_test = F.normalize(test_features, dim=1)
    stacked = torch.cat([normalized_test, prototypes], dim=0).numpy()

    pca = PCA(n_components=2, random_state=random_state)
    projected = pca.fit_transform(stacked)

    n_test = test_features.shape[0]
    test_2d = projected[:n_test]
    proto_2d = projected[n_test:]
    return test_2d, proto_2d, pca.explained_variance_ratio_[:2]
