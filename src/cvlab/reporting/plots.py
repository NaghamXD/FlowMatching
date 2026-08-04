"""Deliverable 2 (M9): accuracy vs. training-set size, one figure, two panels.

Categorical x-axis at K=5/10/full: DTD's "full" is 40 images/class, Flowers-102's
is 10 (decision 15), so a numeric axis would place the two datasets' final points
at different x-positions and invite a false visual comparison between them."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cvlab.evaluation.aggregate import aggregate
from cvlab.reporting.common import ENCODER_COLOR, ENCODER_LABEL, K_ORDER, METHOD_LABEL, images_per_class
from cvlab.reporting.figures import save_figure
from cvlab.reporting.loader import RunCell

_METHOD_STYLE = {
    "linear_probe": {"linestyle": "-", "marker": "o"},
    "image_prototype": {"linestyle": "--", "marker": "s"},
}
_X_POSITIONS = tuple(range(len(K_ORDER)))


def _series(
    cells: dict[tuple[str, str, str, int | str], RunCell], dataset: str, encoder: str, method: str
) -> tuple[list[float], list[float | None]]:
    means: list[float] = []
    stds: list[float | None] = []
    for k in K_ORDER:
        cell = cells[(dataset, encoder, method, k)]
        agg = aggregate([r.test_top1 for r in cell.records])
        means.append(agg.mean)
        stds.append(agg.std if agg.n > 1 else None)
    return means, stds


def _x_tick_labels(
    cells: dict[tuple[str, str, str, int | str], RunCell], dataset: str, encoder: str
) -> list[str]:
    return [str(images_per_class(cells[(dataset, encoder, "linear_probe", k)])) for k in K_ORDER]


def _plot_series(ax, means: list[float], stds: list[float | None], color: str, label: str, style: dict) -> None:
    ax.plot(_X_POSITIONS, means, color=color, label=label, **style)
    for x, m, s in zip(_X_POSITIONS, means, stds):
        if s is not None:
            ax.errorbar([x], [m], yerr=[s], color=color, capsize=3, linestyle="none")


def plot_accuracy_vs_k(
    cells: dict[tuple[str, str, str, int | str], RunCell], out_dir: str | Path
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    fig, (ax_dtd, ax_flowers) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for encoder in ("resnet18", "dinov2_vits14"):
        for method in ("linear_probe", "image_prototype"):
            means, stds = _series(cells, "dtd", encoder, method)
            _plot_series(
                ax_dtd, means, stds, ENCODER_COLOR[encoder],
                f"{ENCODER_LABEL[encoder]} / {METHOD_LABEL[method]}", _METHOD_STYLE[method],
            )
    ax_dtd.set_xticks(list(_X_POSITIONS))
    ax_dtd.set_xticklabels(_x_tick_labels(cells, "dtd", "resnet18"))
    ax_dtd.set_xlabel("Images per class")
    ax_dtd.set_ylabel("Test top-1 accuracy")
    ax_dtd.set_title("DTD")
    ax_dtd.legend(fontsize=8, loc="upper left")

    for method in ("linear_probe", "image_prototype"):
        means, stds = _series(cells, "flowers102", "resnet18", method)
        _plot_series(
            ax_flowers, means, stds, ENCODER_COLOR["resnet18"],
            f"{ENCODER_LABEL['resnet18']} / {METHOD_LABEL[method]}", _METHOD_STYLE[method],
        )
    ax_flowers.set_xticks(list(_X_POSITIONS))
    ax_flowers.set_xticklabels(_x_tick_labels(cells, "flowers102", "resnet18"))
    ax_flowers.set_xlabel("Images per class")
    ax_flowers.set_title("Flowers-102")
    ax_flowers.legend(fontsize=8, loc="lower right")

    fig.suptitle("Deliverable 2: Test accuracy vs. training-set size")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    # Figure-level caption (not an in-axes annotation) so it never collides with
    # the legend or data -- decision 15's note about Flowers-102's split.
    fig.text(
        0.5, 0.01,
        "Flowers-102: K=10 and full use identical training data "
        "(official split has exactly 10 images/class).",
        ha="center", fontsize=8, style="italic",
    )

    paths = save_figure(fig, out_dir, "deliverable2_accuracy_vs_k")
    plt.close(fig)
    return paths
