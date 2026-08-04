"""M12 test: the DTD class list is derived from the ResNet-18 matrix and is
byte-identical between the two panels of 5A; the color mapping is identical
across panels; the selected class list and colors round-trip through the
config file; the pinned subsample seed reproduces the same Flowers test
indices across two runs."""

from __future__ import annotations

from pathlib import Path

import torch

from cvlab.reporting.common import CLASS_VIZ_PALETTE
from cvlab.reporting.feature_viz import (
    compute_and_save_class_selection,
    load_class_selection,
    select_test_indices,
    select_visualization_classes,
)
from cvlab.reporting.loader import RunCell
from cvlab.results.schema import RunRecord
from cvlab.results.store import save as save_record


def _predictions_with_confused_group(
    num_classes: int, confused_group: list[int], per_class: int = 20, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real, controllable mutual confusion within `confused_group`; every other
    class predicted perfectly -- lets a test assert exactly which group a
    selector should find."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(num_classes).repeat_interleave(per_class)
    preds = labels.clone()
    for i, label in enumerate(labels.tolist()):
        if label in confused_group and torch.rand(1, generator=generator).item() < 0.6:
            others = [c for c in confused_group if c != label]
            preds[i] = others[torch.randint(0, len(others), (1,), generator=generator).item()]
    return preds, labels


def _record_with_predictions(tmp_path: Path, name: str, preds: torch.Tensor, labels: torch.Tensor, seed: int) -> RunRecord:
    num_classes = int(labels.max().item()) + 1
    path = tmp_path / f"{name}_predictions.pt"
    torch.save({"preds": preds, "labels": labels}, path)
    return RunRecord(
        run_id=name,
        config={"dataset": {}, "encoder": {}, "method": {"name": "linear_probe"}, "k": "full", "seed": seed},
        git_commit="unknown",
        feature_cache_manifest_hashes={},
        test_top1=0.8,
        mean_per_class_accuracy=0.8,
        train_size=num_classes * 40,
        num_classes=num_classes,
        best_epoch=50,
        epoch_curves=[],
        predictions_path=str(path),
    )


def _cells_with_distinct_confusions(tmp_path: Path, num_classes: int = 16) -> dict:
    """DTD/resnet18 confuses classes {0,1,2}; DTD/dinov2_vits14 confuses a
    completely different group {8,9,10}; flowers102/resnet18 is generic."""
    cells = {}
    groups = {
        ("dtd", "resnet18"): [0, 1, 2],
        ("dtd", "dinov2_vits14"): [8, 9, 10],
        ("flowers102", "resnet18"): [4, 5, 6],
    }
    for (dataset, encoder), group in groups.items():
        records = []
        for seed in (0, 1, 2):
            preds, labels = _predictions_with_confused_group(num_classes, group, seed=seed)
            records.append(_record_with_predictions(tmp_path, f"{dataset}_{encoder}_seed{seed}", preds, labels, seed))
        cells[(dataset, encoder, "linear_probe", "full")] = RunCell(
            dataset, encoder, "linear_probe", "full", tuple(records)
        )
    return cells


def test_class_selection_driven_by_resnet18_not_dinov2_matrix(tmp_path: Path) -> None:
    cells = _cells_with_distinct_confusions(tmp_path)
    path = compute_and_save_class_selection(cells, tmp_path)
    selection = load_class_selection(tmp_path)

    # Must reflect DTD/ResNet-18's confused group {0,1,2} (the selector grows to
    # exactly N_CONFUSED=6 regardless -- see M11's DECISIONS.md finding -- so the
    # true 3-class cluster is a subset, padded with the next-best candidates).
    # DINOv2's completely disjoint confused group {8,9,10} must not appear.
    confused = set(selection["confused_indices"])
    assert {0, 1, 2}.issubset(confused)
    assert not {8, 9, 10} & confused
    assert selection["driving_encoder"] == "resnet18"
    assert path.name == "deliverable5_class_selection.json"


def test_class_selection_round_trips_through_config_file(tmp_path: Path) -> None:
    cells = _cells_with_distinct_confusions(tmp_path)
    compute_and_save_class_selection(cells, tmp_path)
    selection = load_class_selection(tmp_path)

    assert len(selection["class_indices"]) == 10
    assert len(selection["class_names"]) == 10
    assert len(selection["colors"]) == 10
    assert selection["colors"] == list(CLASS_VIZ_PALETTE[:10])
    # No overlap between the confused and control groups.
    assert set(selection["confused_indices"]).isdisjoint(selection["control_indices"])


def test_color_mapping_is_pinned_and_identical_across_repeated_loads(tmp_path: Path) -> None:
    cells = _cells_with_distinct_confusions(tmp_path)
    compute_and_save_class_selection(cells, tmp_path)
    selection_a = load_class_selection(tmp_path)
    selection_b = load_class_selection(tmp_path)
    assert selection_a["colors"] == selection_b["colors"]
    assert selection_a["class_indices"] == selection_b["class_indices"]


def test_select_visualization_classes_composition() -> None:
    # 6 confused (block cluster among 0,1,2 growth) + 4 control (highest recall).
    M = torch.eye(10, dtype=torch.float64) * 0.9
    M[0, 0] = M[1, 1] = M[2, 2] = 0.5
    M[0, 1] = M[1, 0] = 0.3
    M[0, 2] = M[2, 0] = 0.2
    M[1, 2] = M[2, 1] = 0.2
    selection = select_visualization_classes(M)
    assert len(selection["confused"]) == 6
    assert len(selection["control"]) == 4
    assert len(selection["all"]) == 10
    assert len(set(selection["all"])) == 10  # no duplicate class appears in both groups


def test_flowers_subsample_seed_reproduces_same_indices() -> None:
    generator = torch.Generator().manual_seed(0)
    y_test = torch.randint(0, 20, (2000,), generator=generator)
    class_indices = [3, 7, 11]

    idx_a = select_test_indices(y_test, class_indices, max_per_class=50, seed=0)
    idx_b = select_test_indices(y_test, class_indices, max_per_class=50, seed=0)
    assert torch.equal(idx_a, idx_b)

    idx_different_seed = select_test_indices(y_test, class_indices, max_per_class=50, seed=1)
    assert not torch.equal(idx_a, idx_different_seed)


def test_select_test_indices_uncapped_includes_every_matching_example() -> None:
    y_test = torch.tensor([0, 1, 0, 2, 1, 0, 1])
    idx = select_test_indices(y_test, [0, 1], max_per_class=None, seed=0)
    assert sorted(idx.tolist()) == [0, 1, 2, 4, 5, 6]


def test_select_test_indices_respects_cap() -> None:
    y_test = torch.arange(100) % 3
    idx = select_test_indices(y_test, [0, 1, 2], max_per_class=5, seed=0)
    assert idx.numel() == 15
    for c in (0, 1, 2):
        assert (y_test[idx] == c).sum().item() == 5
