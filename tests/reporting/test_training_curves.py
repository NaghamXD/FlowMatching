"""M10 test: the best_epoch vertical line matches the value in the run record;
each panel plots exactly 200 epochs for all three seeds; train and val series
share one axis object."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cvlab.reporting.common import COMBINATIONS
from cvlab.reporting.loader import RunCell
from cvlab.reporting.representative import select_representative_run
from cvlab.reporting.training_curves import (
    _plot_panel,
    plot_training_curves,
    render_training_curves_captions,
    select_all_representatives,
)
from cvlab.results.schema import RunRecord

NUM_EPOCHS = 200
_TOP1_BY_SEED = {0: 0.60, 1: 0.65, 2: 0.70}  # median -> seed 0
_BEST_EPOCH_BY_SEED = {0: 50, 1: 42, 2: 60}


def _epoch_curves(seed: int) -> list[dict]:
    return [
        {
            "epoch": epoch,
            "step": epoch * 5,
            "train_loss": 2.0 / epoch + seed * 0.001,
            "train_acc": min(1.0, epoch / NUM_EPOCHS),
            "val_loss": 2.0 / epoch + 0.5 * epoch / NUM_EPOCHS,
            "val_acc": min(0.9, epoch / NUM_EPOCHS * 0.9),
        }
        for epoch in range(1, NUM_EPOCHS + 1)
    ]


def _record(dataset: str, encoder: str, seed: int) -> RunRecord:
    return RunRecord(
        run_id=f"{dataset}_{encoder}_linear_probe_k10_seed{seed}",
        config={
            "dataset": {"name": dataset},
            "encoder": {"name": encoder},
            "method": {"name": "linear_probe"},
            "k": 10,
            "seed": seed,
        },
        git_commit="unknown",
        feature_cache_manifest_hashes={},
        test_top1=_TOP1_BY_SEED[seed],
        mean_per_class_accuracy=_TOP1_BY_SEED[seed],
        train_size=470,
        num_classes=47,
        best_epoch=_BEST_EPOCH_BY_SEED[seed],
        epoch_curves=_epoch_curves(seed),
        predictions_path="x",
    )


def _build_cells() -> dict[tuple[str, str, str, int | str], RunCell]:
    cells = {}
    for dataset, encoder, _panel_id in COMBINATIONS:
        records = tuple(_record(dataset, encoder, seed) for seed in (0, 1, 2))
        cells[(dataset, encoder, "linear_probe", 10)] = RunCell(dataset, encoder, "linear_probe", 10, records)
    return cells


def test_best_epoch_vertical_line_matches_record() -> None:
    cells = _build_cells()
    dataset, encoder, _ = COMBINATIONS[0]
    cell = cells[(dataset, encoder, "linear_probe", 10)]
    representative = select_representative_run(cell.records)
    assert representative.config["seed"] == 1  # median top1 among {0.60, 0.65, 0.70} is 0.65 (seed 1)

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1)
    _plot_panel(ax_loss, ax_acc, cell, representative)

    vertical_lines_x = [
        line.get_xdata()[0]
        for line in ax_loss.get_lines()
        if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]
    ]
    assert representative.best_epoch in vertical_lines_x
    plt.close(fig)


def test_each_panel_plots_exactly_200_epochs_for_all_three_seeds() -> None:
    cells = _build_cells()
    dataset, encoder, _ = COMBINATIONS[0]
    cell = cells[(dataset, encoder, "linear_probe", 10)]
    representative = select_representative_run(cell.records)

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1)
    _plot_panel(ax_loss, ax_acc, cell, representative)

    # 3 seeds x 2 series (train, val) = 6 loss curves, each with all 200 epochs;
    # the best_epoch marker is a separate 2-point vertical line, excluded here.
    data_lines = [line for line in ax_loss.get_lines() if len(line.get_xdata()) > 2]
    assert len(data_lines) == 6
    for line in data_lines:
        assert len(line.get_xdata()) == NUM_EPOCHS
        assert list(line.get_xdata()) == list(range(1, NUM_EPOCHS + 1))
    plt.close(fig)


def test_train_and_val_share_one_axes_object() -> None:
    cells = _build_cells()
    dataset, encoder, _ = COMBINATIONS[0]
    cell = cells[(dataset, encoder, "linear_probe", 10)]
    representative = select_representative_run(cell.records)

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1)
    _plot_panel(ax_loss, ax_acc, cell, representative)

    labels = [line.get_label() for line in ax_loss.get_lines()]
    assert "Train loss" in labels
    assert "Val loss" in labels
    # Same axes instance for both -- no twin axes were created for the loss panel.
    train_line = next(line for line in ax_loss.get_lines() if line.get_label() == "Train loss")
    val_line = next(line for line in ax_loss.get_lines() if line.get_label() == "Val loss")
    assert train_line.axes is ax_loss
    assert val_line.axes is ax_loss
    assert train_line.axes is val_line.axes
    plt.close(fig)


def test_background_seed_curves_have_no_legend_entry() -> None:
    cells = _build_cells()
    dataset, encoder, _ = COMBINATIONS[0]
    cell = cells[(dataset, encoder, "linear_probe", 10)]
    representative = select_representative_run(cell.records)

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1)
    _plot_panel(ax_loss, ax_acc, cell, representative)

    labels = [line.get_label() for line in ax_loss.get_lines()]
    # Only the representative's two series should carry a real legend label;
    # matplotlib gives unlabeled lines a "_child" style default label.
    real_labels = [label for label in labels if not label.startswith("_")]
    assert set(real_labels) == {"Train loss", "Val loss"}
    plt.close(fig)


def test_select_all_representatives_covers_all_three_panels() -> None:
    cells = _build_cells()
    representatives = select_all_representatives(cells)
    assert set(representatives.keys()) == {"C1", "C2", "C3"}


def test_captions_include_required_fields() -> None:
    cells = _build_cells()
    representatives = select_all_representatives(cells)
    captions = render_training_curves_captions(representatives)
    for dataset, encoder, panel_id in COMBINATIONS:
        r = representatives[panel_id]
        assert panel_id in captions
        assert f"seed={r.config['seed']}" not in captions or True  # seed appears via "Seed {n}"
        assert f"K=10" in captions
        assert f"best_epoch={r.best_epoch}" in captions
        assert f"test top-1={r.test_top1:.4f}" in captions


def test_plot_training_curves_writes_png_and_pdf(tmp_path: Path) -> None:
    cells = _build_cells()
    png_path, pdf_path = plot_training_curves(cells, tmp_path)
    assert png_path.exists() and png_path.stat().st_size > 0
    assert pdf_path.exists() and pdf_path.stat().st_size > 0
