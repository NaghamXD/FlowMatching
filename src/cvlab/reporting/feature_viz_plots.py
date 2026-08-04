"""Deliverable 5 (M12) figures: 5A (DTD, ResNet-18 and DINOv2 side by side),
5B (Flowers-102, ResNet-18), and optional 5C (DTD/ResNet-18, 5-shot vs
full-split prototypes connected by a line segment per class)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cvlab.features.cache import load as load_features
from cvlab.reporting.common import CLASS_VIZ_PALETTE, DATASET_LABEL, ENCODER_LABEL
from cvlab.reporting.confusion_plots import get_full_split_seed0_confusion
from cvlab.reporting.feature_viz import (
    FLOWERS_MAX_TEST_PER_CLASS,
    FLOWERS_SAMPLE_SEED,
    compute_and_save_class_selection,
    compute_prototypes,
    load_class_selection,
    select_test_indices,
    select_visualization_classes,
)
from cvlab.reporting.figures import save_figure
from cvlab.reporting.labels import get_class_names
from cvlab.reporting.loader import RunCell
from cvlab.reporting.projection import joint_pca_projection

_TEST_POINT_KW = dict(s=10, alpha=0.35, linewidths=0)
_PROTO_KW = dict(s=260, marker="*", edgecolors="black", linewidths=1.2, zorder=5)


def _scatter_panel(
    ax: plt.Axes,
    test_2d: np.ndarray,
    test_labels: torch.Tensor,
    proto_2d: np.ndarray,
    class_indices: list[int],
    class_names: list[str],
    colors: list[str],
) -> None:
    for c, color in zip(class_indices, colors):
        mask = (test_labels == c).numpy()
        ax.scatter(test_2d[mask, 0], test_2d[mask, 1], color=color, **_TEST_POINT_KW)
    # Confused-cluster prototypes often project close together (that's the point
    # of the visualization), which would otherwise stack their labels illegibly
    # on top of each other -- alternate offset/alignment per index to spread
    # them out. Not full collision avoidance, but a real, cheap improvement.
    for i, ((x, y), name, color) in enumerate(zip(proto_2d, class_names, colors)):
        ax.scatter([x], [y], color=color, **_PROTO_KW)
        dx = 8 if i % 2 == 0 else -8
        dy = 8 if (i // 2) % 2 == 0 else -14
        ha = "left" if i % 2 == 0 else "right"
        ax.annotate(
            name, (x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=8, weight="bold", ha=ha, va="bottom",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def plot_figure_5a(
    cells: dict[tuple[str, str, str, int | str], RunCell],
    cache_root: str | Path,
    reports_dir: str | Path,
    configs_dir: str | Path = "configs",
) -> tuple[list[Path], str]:
    compute_and_save_class_selection(cells, reports_dir, configs_dir)
    selection = load_class_selection(reports_dir)
    class_indices, class_names, colors = (
        selection["class_indices"], selection["class_names"], selection["colors"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    caption_lines = [
        "Deliverable 5A: DTD test features projected with PCA, full-split prototypes overlaid.",
        f"Classes ({len(class_indices)}): 6 from the ResNet-18 confusion cluster (decision 20) + "
        f"4 highest-recall control classes: {', '.join(class_names)}.",
        "Qualitative visualization only -- not a measure of classification performance.",
    ]
    for ax, encoder in zip(axes, ("resnet18", "dinov2_vits14")):
        Z_test, y_test = load_features(cache_root, "dtd", "test", encoder)
        test_idx = select_test_indices(y_test, class_indices, max_per_class=None, seed=0)
        prototypes = compute_prototypes(cache_root, "dtd", encoder, k="full")
        proto_subset = prototypes[class_indices]

        test_2d, proto_2d, explained = joint_pca_projection(Z_test[test_idx], proto_subset)
        _scatter_panel(ax, test_2d, y_test[test_idx], proto_2d, class_indices, class_names, colors)
        ax.set_title(
            f"{ENCODER_LABEL[encoder]} (PC1 {explained[0] * 100:.1f}%, PC2 {explained[1] * 100:.1f}%)"
        )
        caption_lines.append(
            f"{ENCODER_LABEL[encoder]}: {test_idx.numel()} test points, "
            f"explained variance PC1={explained[0]:.4f}, PC2={explained[1]:.4f}."
        )

    fig.suptitle("Deliverable 5A: DTD -- ResNet-18 vs. DINOv2 ViT-S/14 (same classes, same colors)")
    fig.tight_layout()
    paths = save_figure(fig, reports_dir, "deliverable5a_dtd_feature_viz")
    plt.close(fig)
    return list(paths), "\n".join(caption_lines) + "\n"


def plot_figure_5b(
    cells: dict[tuple[str, str, str, int | str], RunCell],
    cache_root: str | Path,
    reports_dir: str | Path,
    configs_dir: str | Path = "configs",
) -> tuple[list[Path], str]:
    _record, M = get_full_split_seed0_confusion(cells, "flowers102", "resnet18")
    selection = select_visualization_classes(M)
    class_indices = selection["all"]
    class_names_all = get_class_names("flowers102", configs_dir)
    class_names = [class_names_all[i] for i in class_indices]
    colors = list(CLASS_VIZ_PALETTE[: len(class_indices)])

    Z_test, y_test = load_features(cache_root, "flowers102", "test", "resnet18")
    test_idx = select_test_indices(
        y_test, class_indices, max_per_class=FLOWERS_MAX_TEST_PER_CLASS, seed=FLOWERS_SAMPLE_SEED
    )
    prototypes = compute_prototypes(cache_root, "flowers102", "resnet18", k="full")
    proto_subset = prototypes[class_indices]

    test_2d, proto_2d, explained = joint_pca_projection(Z_test[test_idx], proto_subset)

    fig, ax = plt.subplots(figsize=(8, 7))
    _scatter_panel(ax, test_2d, y_test[test_idx], proto_2d, class_indices, class_names, colors)
    ax.set_title(
        f"{ENCODER_LABEL['resnet18']} (PC1 {explained[0] * 100:.1f}%, PC2 {explained[1] * 100:.1f}%)"
    )
    fig.suptitle(f"Deliverable 5B: {DATASET_LABEL['flowers102']} test features + full-split prototypes (PCA)")
    fig.tight_layout()
    paths = save_figure(fig, reports_dir, "deliverable5b_flowers102_feature_viz")
    plt.close(fig)

    caption = (
        "Deliverable 5B: Flowers-102 test features projected with PCA, full-split prototypes overlaid.\n"
        f"Classes ({len(class_indices)}): 6 from the ResNet-18 confusion cluster + 4 highest-recall "
        f"control classes: {', '.join(class_names)}.\n"
        f"Test points capped at {FLOWERS_MAX_TEST_PER_CLASS}/class (seed={FLOWERS_SAMPLE_SEED}) since "
        "the Flowers-102 test split is class-imbalanced.\n"
        f"Explained variance PC1={explained[0]:.4f}, PC2={explained[1]:.4f}. "
        "Qualitative visualization only -- not a measure of classification performance.\n"
    )
    return list(paths), caption


def plot_figure_5c(
    cells: dict[tuple[str, str, str, int | str], RunCell],
    cache_root: str | Path,
    reports_dir: str | Path,
    configs_dir: str | Path = "configs",
) -> tuple[list[Path], str]:
    """Optional: same dataset/encoder/classes/test-features as one panel of 5A
    (DTD/ResNet-18), with 5-shot (seed 0) and full-split prototypes overlaid and
    connected -- the segment length is the K=5 prototype's estimation error."""
    selection = load_class_selection(reports_dir)
    class_indices, class_names, colors = (
        selection["class_indices"], selection["class_names"], selection["colors"]
    )

    Z_test, y_test = load_features(cache_root, "dtd", "test", "resnet18")
    test_idx = select_test_indices(y_test, class_indices, max_per_class=None, seed=0)

    full_prototypes = compute_prototypes(cache_root, "dtd", "resnet18", k="full")
    five_shot_prototypes = compute_prototypes(cache_root, "dtd", "resnet18", k=5, seed=0)

    n_test = test_idx.numel()
    stacked_test = Z_test[test_idx]
    combined_prototypes = torch.cat(
        [five_shot_prototypes[class_indices], full_prototypes[class_indices]], dim=0
    )
    test_2d, proto_2d, explained = joint_pca_projection(stacked_test, combined_prototypes)
    five_shot_2d = proto_2d[: len(class_indices)]
    full_2d = proto_2d[len(class_indices) :]

    fig, ax = plt.subplots(figsize=(9, 8))
    for c, color in zip(class_indices, colors):
        mask = (y_test[test_idx] == c).numpy()
        ax.scatter(test_2d[mask, 0], test_2d[mask, 1], color=color, **_TEST_POINT_KW)
    for i, (name, color) in enumerate(zip(class_names, colors)):
        x0, y0 = five_shot_2d[i]
        x1, y1 = full_2d[i]
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.2, alpha=0.8, zorder=4)
        ax.scatter([x0], [y0], color=color, marker="o", s=140, edgecolors="black", linewidths=1.0, zorder=5)
        ax.scatter([x1], [y1], color=color, marker="*", s=260, edgecolors="black", linewidths=1.2, zorder=5)
        dx = 8 if i % 2 == 0 else -8
        dy = 8 if (i // 2) % 2 == 0 else -14
        ha = "left" if i % 2 == 0 else "right"
        ax.annotate(
            name, (x1, y1), xytext=(dx, dy), textcoords="offset points",
            fontsize=8, weight="bold", ha=ha, va="bottom",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"DTD / ResNet-18: 5-shot (circle) -> full-split (star) prototypes "
        f"(PC1 {explained[0] * 100:.1f}%, PC2 {explained[1] * 100:.1f}%)"
    )
    fig.suptitle("Deliverable 5C (optional): why accuracy improves with K")
    fig.tight_layout()
    paths = save_figure(fig, reports_dir, "deliverable5c_dtd_kshot_vs_full_prototypes")
    plt.close(fig)

    caption = (
        "Deliverable 5C (optional): DTD/ResNet-18 test features (same classes as 5A) with both the "
        "5-shot (seed 0, circle marker) and full-split (star marker) image prototypes overlaid, "
        "connected by a line segment per class -- the segment length is the 5-shot prototype's "
        "estimation error relative to the full-split centroid.\n"
        f"Explained variance PC1={explained[0]:.4f}, PC2={explained[1]:.4f}. "
        "Qualitative visualization only -- not a measure of classification performance.\n"
    )
    return list(paths), caption
