"""M11 test: a synthetic matrix with a known planted confusion pair has that
pair ranked first in the top-N table; end-to-end artifact generation runs
against the real 3-combination grid."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cvlab.reporting.common import COMBINATIONS
from cvlab.reporting.confusion_plots import (
    build_confusion_artifacts,
    get_full_split_seed0_confusion,
    top_confusion_pairs,
)
from cvlab.reporting.loader import RunCell
from cvlab.results.schema import RunRecord
from cvlab.results.store import save as save_record


def test_top_confusion_pairs_ranks_planted_pair_first() -> None:
    n = 5
    M = torch.eye(n, dtype=torch.float64) * 0.9
    M[2, 4] = 0.35  # planted, deliberately larger than every other off-diagonal cell
    M[0, 1] = 0.05
    M[3, 0] = 0.03
    names = tuple(f"class{i}" for i in range(n))

    pairs = top_confusion_pairs(M, names, n=3)

    assert pairs[0]["true_class"] == "class2"
    assert pairs[0]["predicted_class"] == "class4"
    assert pairs[0]["rate"] == pytest.approx(0.35)
    assert pairs[0]["true_class_recall"] == pytest.approx(M[2, 2].item())


def test_top_confusion_pairs_excludes_diagonal() -> None:
    M = torch.eye(4, dtype=torch.float64)  # perfect classifier -- no off-diagonal mass at all
    names = tuple(f"c{i}" for i in range(4))
    pairs = top_confusion_pairs(M, names, n=3)
    for p in pairs:
        assert p["rate"] == pytest.approx(0.0)


def _synthetic_predictions(tmp_path: Path, name: str, num_classes: int, per_class: int, seed: int) -> Path:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(num_classes).repeat_interleave(per_class)
    # A biased-but-imperfect classifier: mostly correct, with a few confusions.
    preds = labels.clone()
    noise_idx = torch.randperm(len(labels), generator=generator)[: len(labels) // 6]
    preds[noise_idx] = (labels[noise_idx] + 1) % num_classes

    path = tmp_path / f"{name}_predictions.pt"
    torch.save({"preds": preds, "labels": labels}, path)
    return path


def _build_full_split_cells(tmp_path: Path, num_classes: int = 16) -> dict[tuple[str, str, str, int | str], RunCell]:
    cells = {}
    for dataset, encoder, panel_id in COMBINATIONS:
        records = []
        for seed in (0, 1, 2):
            preds_path = _synthetic_predictions(
                tmp_path, f"{dataset}_{encoder}_seed{seed}", num_classes, per_class=10, seed=seed
            )
            records.append(
                RunRecord(
                    run_id=f"{dataset}_{encoder}_linear_probe_kfull_seed{seed}",
                    config={
                        "dataset": {"name": dataset}, "encoder": {"name": encoder},
                        "method": {"name": "linear_probe"}, "k": "full", "seed": seed,
                    },
                    git_commit="unknown",
                    feature_cache_manifest_hashes={},
                    test_top1=0.8,
                    mean_per_class_accuracy=0.8,
                    train_size=num_classes * 40,
                    num_classes=num_classes,
                    best_epoch=50,
                    epoch_curves=[],
                    predictions_path=str(preds_path),
                )
            )
        cells[(dataset, encoder, "linear_probe", "full")] = RunCell(
            dataset, encoder, "linear_probe", "full", tuple(records)
        )
    return cells


def test_get_full_split_seed0_confusion_picks_seed_zero(tmp_path: Path) -> None:
    cells = _build_full_split_cells(tmp_path)
    dataset, encoder, _ = COMBINATIONS[0]
    record, M = get_full_split_seed0_confusion(cells, dataset, encoder)
    assert record.config["seed"] == 0
    assert M.shape == (16, 16)
    assert torch.allclose(M.sum(dim=1), torch.ones(16, dtype=torch.float64), atol=1e-6)


def test_build_confusion_artifacts_writes_all_four_artifact_types_per_combination(tmp_path: Path) -> None:
    cells = _build_full_split_cells(tmp_path)
    out_dir = tmp_path / "reports"
    written = build_confusion_artifacts(cells, out_dir)

    for dataset, encoder, panel_id in COMBINATIONS:
        prefix = f"deliverable4_{panel_id}_{dataset}_{encoder}"
        names = {p.name for p in written if p.name.startswith(prefix)}
        assert any(n.endswith("_confusion_full.png") for n in names)
        assert any(n.endswith("_confusion_full_masked.png") for n in names)
        assert any(n.endswith("_confusion_zoomed.png") for n in names)
        assert any(n.endswith("_recall_distribution.png") for n in names)
        assert any(n.endswith("_top_pairs.md") for n in names)

    for p in written:
        assert p.exists() and p.stat().st_size > 0
