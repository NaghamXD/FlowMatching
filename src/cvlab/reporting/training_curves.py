"""Deliverable 3 (M10): training and validation loss curves for the linear
probe at K=10, one representative run per (dataset, encoder) combination
(decision 16) -- three panels, C1/C2/C3.

Train and val loss share one axes/scale: the vertical gap between them is the
overfitting signal itself, and twin axes would destroy it. Full 200-epoch
x-range (decision 13: no early stopping), so post-peak divergence is visible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cvlab.reporting.common import COMBINATIONS, DATASET_LABEL, ENCODER_LABEL
from cvlab.reporting.figures import save_figure
from cvlab.reporting.loader import RunCell
from cvlab.reporting.representative import select_representative_run
from cvlab.results.schema import RunRecord

_TRAIN_COLOR = "#0072B2"
_VAL_COLOR = "#D55E00"
_BACKGROUND_ALPHA = 0.18


def _curve(record: RunRecord, key: str) -> tuple[list[int], list[float]]:
    epochs = [e["epoch"] for e in record.epoch_curves]
    values = [e[key] for e in record.epoch_curves]
    return epochs, values


def _plot_panel(ax_loss: plt.Axes, ax_acc: plt.Axes, cell: RunCell, representative: RunRecord) -> None:
    for other in cell.records:
        if other.run_id == representative.run_id:
            continue
        epochs, train_loss = _curve(other, "train_loss")
        _, val_loss = _curve(other, "val_loss")
        _, val_acc = _curve(other, "val_acc")
        ax_loss.plot(epochs, train_loss, color=_TRAIN_COLOR, alpha=_BACKGROUND_ALPHA, linewidth=1)
        ax_loss.plot(epochs, val_loss, color=_VAL_COLOR, alpha=_BACKGROUND_ALPHA, linewidth=1)
        ax_acc.plot(epochs, val_acc, color=_VAL_COLOR, alpha=_BACKGROUND_ALPHA, linewidth=1)

    epochs, train_loss = _curve(representative, "train_loss")
    _, val_loss = _curve(representative, "val_loss")
    _, val_acc = _curve(representative, "val_acc")
    max_epoch = max(epochs)

    ax_loss.plot(epochs, train_loss, color=_TRAIN_COLOR, linewidth=1.8, label="Train loss")
    ax_loss.plot(epochs, val_loss, color=_VAL_COLOR, linewidth=1.8, label="Val loss")
    ax_loss.axvline(representative.best_epoch, color="0.3", linestyle=":", linewidth=1)
    ax_loss.text(
        representative.best_epoch, 0.97, f"best_epoch={representative.best_epoch}",
        transform=ax_loss.get_xaxis_transform(), rotation=90, va="top", ha="right",
        fontsize=7, color="0.3",
    )
    ax_loss.set_xlim(1, max_epoch)

    ax_acc.plot(epochs, val_acc, color=_VAL_COLOR, linewidth=1.8, label="Val accuracy")
    ax_acc.axvline(representative.best_epoch, color="0.3", linestyle=":", linewidth=1)
    ax_acc.set_xlim(1, max_epoch)


def select_all_representatives(
    cells: dict[tuple[str, str, str, int | str], RunCell],
) -> dict[str, RunRecord]:
    """One representative run per C1/C2/C3, at K=10, linear probe."""
    representatives = {}
    for dataset, encoder, panel_id in COMBINATIONS:
        cell = cells[(dataset, encoder, "linear_probe", 10)]
        representatives[panel_id] = select_representative_run(cell.records)
    return representatives


def plot_training_curves(
    cells: dict[tuple[str, str, str, int | str], RunCell], out_dir: str | Path
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex="col")

    for col, (dataset, encoder, panel_id) in enumerate(COMBINATIONS):
        cell = cells[(dataset, encoder, "linear_probe", 10)]
        representative = select_representative_run(cell.records)

        ax_loss, ax_acc = axes[0, col], axes[1, col]
        _plot_panel(ax_loss, ax_acc, cell, representative)

        ax_loss.set_title(f"{panel_id}: {ENCODER_LABEL[encoder]} / {DATASET_LABEL[dataset]}")
        ax_acc.set_xlabel("Epoch")
        if col == 0:
            ax_loss.set_ylabel("Loss")
            ax_acc.set_ylabel("Val accuracy")
            ax_loss.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Deliverable 3: Train/val loss and val accuracy (K=10, representative seed per decision 16)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    return save_figure(fig, out_dir, "deliverable3_training_curves")


def render_training_curves_captions(representatives: dict[str, RunRecord]) -> str:
    lines = ["# Deliverable 3: training curve captions", ""]
    for dataset, encoder, panel_id in COMBINATIONS:
        r = representatives[panel_id]
        final_train_loss = r.epoch_curves[-1]["train_loss"]
        best_val_loss = min(e["val_loss"] for e in r.epoch_curves)
        lines.append(
            f"{panel_id}: {DATASET_LABEL[dataset]} / {ENCODER_LABEL[encoder]}, K=10. "
            f"Seed {r.config['seed']} shown (median test top-1 among seeds {{0,1,2}}; "
            f"ties break to the lowest seed number -- decision 16). "
            f"best_epoch={r.best_epoch}, final train_loss={final_train_loss:.4f}, "
            f"best val_loss={best_val_loss:.4f}, test top-1={r.test_top1:.4f}."
        )
    return "\n".join(lines) + "\n"
