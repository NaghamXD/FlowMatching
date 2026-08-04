"""Deliverable 4 (M11): the layered confusion-matrix artifact set -- (a) full
reordered matrix, plain and diagonal-masked (decision 19), (b) zoomed
submatrix of the most mutually-confused classes, (c) top-N confusion pairs
table, (d) per-class recall distribution. All four come from the same
underlying row-normalized matrix per combination (decision 18: full-split
linear probe, seed 0)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import torch

from cvlab.reporting.common import COMBINATIONS
from cvlab.reporting.confusion import (
    assert_row_sums_valid,
    grow_confusion_cluster,
    hierarchical_leaf_order,
    load_confusion_matrix,
    max_off_diagonal,
)
from cvlab.reporting.figures import save_figure
from cvlab.reporting.labels import get_class_names
from cvlab.reporting.loader import RunCell
from cvlab.results.schema import RunRecord
from cvlab.utils.io import ensure_dir

ZOOM_CLASS_COUNT = 14  # within the spec's 12-15 range
TOP_N_PAIRS = 15


def _select_full_split_seed0(cell: RunCell) -> RunRecord:
    matches = [r for r in cell.records if r.config["seed"] == 0]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one seed=0 record in cell, found {len(matches)}")
    return matches[0]


def get_full_split_seed0_confusion(
    cells: dict[tuple[str, str, str, int | str], RunCell], dataset: str, encoder: str
) -> tuple[RunRecord, torch.Tensor]:
    """Decision 18's setting, reused verbatim by M12 for class selection."""
    cell = cells[(dataset, encoder, "linear_probe", "full")]
    record = _select_full_split_seed0(cell)
    M = load_confusion_matrix(record)
    assert_row_sums_valid(M)
    return record, M


def plot_full_matrix(M: torch.Tensor, class_names: tuple[str, ...], name_prefix: str, out_dir: str | Path) -> list[Path]:
    order = hierarchical_leaf_order(M)
    reordered = M[order][:, order].numpy()
    reordered_names = [class_names[i] for i in order]
    n = len(order)
    suppress_labels = n > 60  # per spec: suppress on the 102-class version, keep small-font on 47

    def _style(ax: plt.Axes) -> None:
        if suppress_labels:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xticks(range(n))
            ax.set_xticklabels(reordered_names, rotation=90, fontsize=5)
            ax.set_yticks(range(n))
            ax.set_yticklabels(reordered_names, fontsize=5)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")

    fig, ax = plt.subplots(figsize=(9, 9) if suppress_labels else (12, 12))
    im = ax.imshow(reordered, vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Fraction of true class (row-normalized)", shrink=0.8)
    _style(ax)
    ax.set_title(f"{name_prefix}: full confusion matrix (hierarchically clustered order)", fontsize=11, wrap=True)
    fig.tight_layout()
    plain_paths = save_figure(fig, out_dir, f"{name_prefix}_confusion_full")
    plt.close(fig)

    diag_mask = np.eye(n, dtype=bool)
    off_diag_max = max_off_diagonal(torch.from_numpy(reordered))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    masked_data = ma.array(reordered, mask=diag_mask)

    fig2, ax2 = plt.subplots(figsize=(9, 9) if suppress_labels else (12, 12))
    im2 = ax2.imshow(masked_data, vmin=0, vmax=off_diag_max, cmap=cmap)
    fig2.colorbar(im2, ax=ax2, label=f"Fraction of true class (off-diagonal scale, vmax={off_diag_max:.3f})", shrink=0.8)
    _style(ax2)
    ax2.set_title(f"{name_prefix}: confusion matrix, diagonal masked", fontsize=11, wrap=True)
    fig2.tight_layout()
    masked_paths = save_figure(fig2, out_dir, f"{name_prefix}_confusion_full_masked")
    plt.close(fig2)

    return [*plain_paths, *masked_paths]


def plot_zoomed_submatrix(
    M: torch.Tensor, class_names: tuple[str, ...], name_prefix: str, out_dir: str | Path
) -> list[Path]:
    """This is "the figure you actually present" per the spec, so decision 19's
    off-diagonal-visibility treatment applies here too, not just to the full
    matrix: plain [0, 1] scale, plus a diagonal-masked version rescaled to the
    submatrix's own max off-diagonal value (its cells are a subset of the full
    matrix's, so the full matrix's off-diagonal max would often still wash this
    one out)."""
    selected = grow_confusion_cluster(M, ZOOM_CLASS_COUNT)
    sub_t = M[selected][:, selected]
    sub = sub_t.numpy()
    names = [class_names[i] for i in selected]
    n = len(selected)

    def _style(ax: plt.Axes) -> None:
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(sub, vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Fraction of true class (row-normalized)")
    _style(ax)
    ax.set_title(f"{name_prefix}: {n} most mutually-confused classes", fontsize=11, wrap=True)
    fig.tight_layout()
    plain_paths = save_figure(fig, out_dir, f"{name_prefix}_confusion_zoomed")
    plt.close(fig)

    off_diag_max = max_off_diagonal(sub_t)
    diag_mask = np.eye(n, dtype=bool)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    masked_data = ma.array(sub, mask=diag_mask)

    fig2, ax2 = plt.subplots(figsize=(10, 8.5))
    im2 = ax2.imshow(masked_data, vmin=0, vmax=off_diag_max, cmap=cmap)
    fig2.colorbar(im2, ax=ax2, label=f"Fraction of true class (off-diagonal scale, vmax={off_diag_max:.3f})")
    _style(ax2)
    ax2.set_title(f"{name_prefix}: {n} most mutually-confused classes, diagonal masked", fontsize=11, wrap=True)
    fig2.tight_layout()
    masked_paths = save_figure(fig2, out_dir, f"{name_prefix}_confusion_zoomed_masked")
    plt.close(fig2)

    return [*plain_paths, *masked_paths]


def top_confusion_pairs(M: torch.Tensor, class_names: tuple[str, ...], n: int = TOP_N_PAIRS) -> list[dict]:
    num_classes = M.shape[0]
    off_diag = M.clone()
    off_diag.fill_diagonal_(-1.0)  # exclude the diagonal from the top-k search
    top_values, top_indices = torch.topk(off_diag.flatten(), n)
    rows = []
    for value, idx in zip(top_values.tolist(), top_indices.tolist()):
        i, j = divmod(idx, num_classes)
        rows.append(
            {
                "true_class": class_names[i],
                "predicted_class": class_names[j],
                "rate": value,
                "true_class_recall": M[i, i].item(),
            }
        )
    return rows


def render_top_pairs_markdown(rows: list[dict], name_prefix: str) -> str:
    lines = [
        f"# {name_prefix}: top {len(rows)} confusion pairs",
        "",
        "| True class | Predicted class | Rate | True-class recall |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['true_class']} | {r['predicted_class']} | {r['rate'] * 100:.2f}% | "
            f"{r['true_class_recall'] * 100:.2f}% |"
        )
    return "\n".join(lines) + "\n"


def plot_recall_distribution(
    M: torch.Tensor, class_names: tuple[str, ...], name_prefix: str, out_dir: str | Path
) -> list[Path]:
    diag = torch.diagonal(M)
    order = torch.argsort(diag, descending=True)
    sorted_names = [class_names[i] for i in order.tolist()]
    sorted_values = diag[order].numpy()
    n = len(sorted_values)

    colors = ["#0072B2"] * n
    for i in range(min(5, n)):
        colors[i] = "#009E73"
    for i in range(max(0, n - 5), n):
        colors[i] = "#D55E00"

    fig, ax = plt.subplots(figsize=(max(8, n * 0.12), 5))
    ax.bar(range(n), sorted_values, color=colors, width=1.0)
    ax.set_xlabel("Classes (sorted by recall)")
    ax.set_ylabel("Per-class recall")
    ax.set_title(f"{name_prefix}: per-class recall distribution (5 best, 5 worst labeled)", fontsize=11, wrap=True)
    ax.set_xticks([])
    # Headroom above the tallest bar so the rotated class-name labels never
    # collide with the title.
    ax.set_ylim(0, float(sorted_values.max()) * 1.3)
    for i in list(range(min(5, n))) + list(range(max(0, n - 5), n)):
        ax.annotate(sorted_names[i], (i, sorted_values[i]), rotation=90, fontsize=6, ha="center", va="bottom")
    fig.tight_layout()
    paths = save_figure(fig, out_dir, f"{name_prefix}_recall_distribution")
    plt.close(fig)
    return list(paths)


def build_confusion_artifacts(
    cells: dict[tuple[str, str, str, int | str], RunCell],
    out_dir: str | Path,
    configs_dir: str | Path = "configs",
) -> list[Path]:
    out_dir = ensure_dir(out_dir)
    written: list[Path] = []
    for dataset, encoder, panel_id in COMBINATIONS:
        record, M = get_full_split_seed0_confusion(cells, dataset, encoder)
        class_names = get_class_names(dataset, configs_dir)
        name_prefix = f"deliverable4_{panel_id}_{dataset}_{encoder}"

        written += plot_full_matrix(M, class_names, name_prefix, out_dir)
        written += plot_zoomed_submatrix(M, class_names, name_prefix, out_dir)
        written += plot_recall_distribution(M, class_names, name_prefix, out_dir)

        pairs = top_confusion_pairs(M, class_names)
        table_path = Path(out_dir) / f"{name_prefix}_top_pairs.md"
        table_path.write_text(render_top_pairs_markdown(pairs, name_prefix))
        written.append(table_path)

    return written
