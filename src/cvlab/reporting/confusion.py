"""Deliverable 4 (M11) core: builds the row-normalized confusion matrix for a
run from its saved predictions (results store only -- decision 18: the
full-split linear probe, seed 0), plus the reusable cluster-growing selector
that M12 also depends on."""

from __future__ import annotations

import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from cvlab.evaluation.metrics import confusion_matrix_from_preds
from cvlab.results.schema import RunRecord


def load_confusion_matrix(record: RunRecord) -> torch.Tensor:
    """Row-normalized confusion matrix from a run's saved predictions --
    reporting's only source for this, never re-run inference."""
    blob = torch.load(record.predictions_path)
    return confusion_matrix_from_preds(blob["preds"], blob["labels"], record.num_classes)


def assert_row_sums_valid(M: torch.Tensor, tol: float = 1e-6) -> None:
    """Every row must sum to ~1 (a class with test examples) or ~0 (a class with
    none) -- never anything else, and never NaN."""
    if torch.isnan(M).any():
        raise AssertionError("confusion matrix contains NaN -- a row was divided by zero")
    row_sums = M.sum(dim=1)
    valid = torch.isclose(row_sums, torch.ones_like(row_sums), atol=tol) | torch.isclose(
        row_sums, torch.zeros_like(row_sums), atol=tol
    )
    if not valid.all():
        bad_rows = torch.nonzero(~valid).flatten().tolist()
        raise AssertionError(f"confusion matrix rows {bad_rows} sum to neither ~1 nor ~0: {row_sums[~valid].tolist()}")


def max_off_diagonal(M: torch.Tensor) -> float:
    """The true maximum off-diagonal cell value -- decision 19's `vmax` for the
    diagonal-masked confusion-matrix figure."""
    n = M.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    return M[mask].max().item()


def grow_confusion_cluster(M: torch.Tensor, n: int) -> list[int]:
    """Selects `n` class indices carrying the most mutual-confusion mass, per the
    reporting spec's seed-pair-then-greedy-growth rule:

        S = (M + M.T) / 2 ; fill_diagonal(S, 0)
        seed_pair = argmax over (i, j) of S
        selected = {i, j}
        while len(selected) < n:
            add argmax_k not-in selected of sum over s in selected of S[k, s]

    This grows a mutually-confused *cluster*, not just a collection of
    independently low-accuracy classes -- reused by M12 for feature-viz class
    selection (decision 20)."""
    num_classes = M.shape[0]
    if n > num_classes:
        raise ValueError(f"cannot select {n} classes from a {num_classes}-class matrix")

    S = ((M + M.T) / 2).clone()
    S.fill_diagonal_(0)

    flat_index = int(torch.argmax(S).item())
    i, j = divmod(flat_index, num_classes)
    selected = [i, j]

    while len(selected) < n:
        candidates = [k for k in range(num_classes) if k not in selected]
        scores = [sum(S[k, s].item() for s in selected) for k in candidates]
        best = candidates[int(torch.tensor(scores).argmax().item())]
        selected.append(best)

    return selected


def hierarchical_leaf_order(M: torch.Tensor) -> list[int]:
    """Dendrogram leaf order from hierarchical clustering of the symmetrized
    confusion matrix -- pulls mutually-confused classes adjacent to each other
    so error structure reads as visible off-diagonal blocks instead of
    scattered dots, replacing an alphabetical ordering."""
    S = ((M + M.T) / 2).numpy().astype(np.float64)
    max_val = S.max()
    distance = 1.0 - S / max_val if max_val > 0 else np.ones_like(S)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method="average")
    return dendrogram(Z, no_plot=True)["leaves"]
