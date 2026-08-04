"""M12 test: the joint projection is fit on a matrix with exactly N + n_classes
rows; prototypes are unit-norm before projection (decision 21: only test
features get normalized here -- prototypes already are, by construction)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from cvlab.reporting.projection import joint_pca_projection


def test_joint_pca_fits_on_exactly_n_plus_c_rows(monkeypatch) -> None:
    captured = {}
    original_fit_transform = PCA.fit_transform

    def spy(self, X, *a, **kw):
        captured["shape"] = X.shape
        return original_fit_transform(self, X, *a, **kw)

    monkeypatch.setattr(PCA, "fit_transform", spy)

    test_features = torch.randn(20, 8)
    prototypes = F.normalize(torch.randn(4, 8), dim=1)
    joint_pca_projection(test_features, prototypes)

    assert captured["shape"] == (24, 8)


def test_projection_output_row_counts_match_inputs() -> None:
    test_features = torch.randn(15, 6)
    prototypes = F.normalize(torch.randn(5, 6), dim=1)
    test_2d, proto_2d, explained = joint_pca_projection(test_features, prototypes)
    assert test_2d.shape == (15, 2)
    assert proto_2d.shape == (5, 2)
    assert explained.shape == (2,)


def test_test_features_and_prototypes_are_unit_norm_at_fit_time(monkeypatch) -> None:
    captured = {}
    original_fit_transform = PCA.fit_transform

    def spy(self, X, *a, **kw):
        captured["X"] = X.copy()
        return original_fit_transform(self, X, *a, **kw)

    monkeypatch.setattr(PCA, "fit_transform", spy)

    test_features = torch.randn(10, 6) * 5.0  # deliberately large norm
    prototypes = F.normalize(torch.randn(4, 6), dim=1)  # already unit-norm, as ImagePrototype produces
    joint_pca_projection(test_features, prototypes)

    X = captured["X"]
    test_norms = np.linalg.norm(X[:10], axis=1)
    proto_norms = np.linalg.norm(X[10:], axis=1)
    assert np.allclose(test_norms, 1.0, atol=1e-5)
    assert np.allclose(proto_norms, 1.0, atol=1e-5)


def test_projection_is_deterministic_for_fixed_random_state() -> None:
    test_features = torch.randn(10, 6)
    prototypes = F.normalize(torch.randn(4, 6), dim=1)
    a = joint_pca_projection(test_features, prototypes, random_state=0)
    b = joint_pca_projection(test_features, prototypes, random_state=0)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
