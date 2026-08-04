"""M9 test: aggregation over three synthetic records reproduces a known mean and
ddof=1 std; single-run cells are rendered without a ± term; regenerating the
table twice yields byte-identical output."""

from __future__ import annotations

import math
from pathlib import Path

from cvlab.reporting.loader import RunCell, load_run_matrix
from cvlab.reporting.tables import build_accuracy_table, render_csv, render_latex, render_markdown
from cvlab.results.schema import RunRecord


def _record(top1: float, train_size: int = 1880, num_classes: int = 47) -> RunRecord:
    return RunRecord(
        run_id="x",
        config={},
        git_commit="unknown",
        feature_cache_manifest_hashes={},
        test_top1=top1,
        mean_per_class_accuracy=top1,
        train_size=train_size,
        num_classes=num_classes,
        best_epoch=None,
        epoch_curves=[],
        predictions_path="x",
    )


def test_aggregation_reproduces_known_mean_and_ddof1_std() -> None:
    # mean=0.6, sample std (ddof=1) = 0.1 -- same worked example as M7's aggregate test.
    cell = RunCell("dtd", "resnet18", "linear_probe", 5, (_record(0.5), _record(0.6), _record(0.7)))
    rows = build_accuracy_table({("dtd", "resnet18", "linear_probe", 5): cell})
    assert len(rows) == 1
    assert math.isclose(rows[0].top1_mean, 0.6, rel_tol=1e-9)
    assert math.isclose(rows[0].top1_std, 0.1, rel_tol=1e-9)
    assert rows[0].n_runs == 3


def test_single_run_cell_rendered_without_plusminus() -> None:
    cell = RunCell("dtd", "resnet18", "image_prototype", "full", (_record(0.5878),))
    rows = build_accuracy_table({("dtd", "resnet18", "image_prototype", "full"): cell})
    assert rows[0].n_runs == 1

    md = render_markdown(rows)
    data_row = next(line for line in md.splitlines() if line.startswith("| dtd"))
    assert "58.78%¹" in data_row
    assert "±" not in data_row  # the footnote below is allowed to mention "± 0.00" as an explanation
    assert "Single deterministic run" in md

    latex = render_latex(rows)
    data_line = next(line for line in latex.splitlines() if line.startswith("dtd"))
    assert "58.78$^\\dagger$" in data_line
    assert "\\pm" not in data_line


def test_multi_run_cell_rendered_with_plusminus() -> None:
    cell = RunCell("dtd", "resnet18", "linear_probe", 5, (_record(0.5), _record(0.6), _record(0.7)))
    rows = build_accuracy_table({("dtd", "resnet18", "linear_probe", 5): cell})
    md = render_markdown(rows)
    assert "60.00% ± 10.00%" in md


def test_regenerating_table_twice_is_byte_identical(synthetic_runs_root: Path) -> None:
    cells_a = load_run_matrix(synthetic_runs_root)
    cells_b = load_run_matrix(synthetic_runs_root)
    rows_a = build_accuracy_table(cells_a)
    rows_b = build_accuracy_table(cells_b)

    assert render_markdown(rows_a) == render_markdown(rows_b)
    assert render_csv(rows_a) == render_csv(rows_b)
    assert render_latex(rows_a) == render_latex(rows_b)
