"""Deliverable 5 (M12): joint PCA projections of test features + class
prototypes. This is the one place reporting is explicitly allowed to read
cached features directly rather than only the results store (the reporting
spec's hard constraint restated: "the one exception is cached features, which
the visualization deliverable legitimately needs to load")."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from cvlab.features.cache import load as load_features
from cvlab.heads.image_prototype import ImagePrototype
from cvlab.reporting.common import CLASS_VIZ_PALETTE, DATASET_LABEL, ENCODER_LABEL
from cvlab.reporting.confusion import grow_confusion_cluster
from cvlab.reporting.confusion_plots import get_full_split_seed0_confusion
from cvlab.reporting.figures import save_figure
from cvlab.reporting.labels import get_class_names
from cvlab.reporting.loader import RunCell
from cvlab.sampling.fewshot import balanced_kshot

N_CONFUSED = 6
N_CONTROL = 4
FLOWERS_MAX_TEST_PER_CLASS = 50
FLOWERS_SAMPLE_SEED = 0
# Decision 20: DTD's selection is driven by the ResNet-18 matrix specifically,
# not DINOv2's -- selecting on the weaker encoder's failures and then showing
# the stronger encoder separating those same classes is an unbiased result.
DRIVING_DATASET = "dtd"
DRIVING_ENCODER = "resnet18"

_SELECTION_FILENAME = "deliverable5_class_selection.json"


def select_visualization_classes(M: torch.Tensor) -> dict[str, list[int]]:
    """6 classes from the confusion cluster (the ones the model mixes up) + 4
    classes with the highest per-class recall (well-separated visual control)."""
    confused = grow_confusion_cluster(M, N_CONFUSED)
    diag = torch.diagonal(M)
    order = torch.argsort(diag, descending=True).tolist()
    control = [i for i in order if i not in confused][:N_CONTROL]
    return {"confused": confused, "control": control, "all": confused + control}


def compute_and_save_class_selection(
    cells: dict[tuple[str, str, str, int | str], RunCell],
    reports_dir: str | Path,
    configs_dir: str | Path = "configs",
) -> Path:
    """Computes the deliverable-5 class selection ONCE and writes it to a config
    file -- decision 20: the class list and color mapping are inputs to the
    figures, not values recomputed at plot time."""
    _record, M = get_full_split_seed0_confusion(cells, DRIVING_DATASET, DRIVING_ENCODER)
    selection = select_visualization_classes(M)
    class_names = get_class_names(DRIVING_DATASET, configs_dir)

    payload = {
        "dataset": DRIVING_DATASET,
        "driving_encoder": DRIVING_ENCODER,
        "confused_indices": selection["confused"],
        "control_indices": selection["control"],
        "class_indices": selection["all"],
        "class_names": [class_names[i] for i in selection["all"]],
        "colors": list(CLASS_VIZ_PALETTE[: len(selection["all"])]),
    }
    path = Path(reports_dir) / _SELECTION_FILENAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_class_selection(reports_dir: str | Path) -> dict:
    path = Path(reports_dir) / _SELECTION_FILENAME
    return json.loads(path.read_text())


def compute_prototypes(
    cache_root: str | Path, dataset: str, encoder: str, k: int | str = "full", seed: int = 0
) -> torch.Tensor:
    """Recomputes class prototypes directly from cached train features -- cheap
    and deterministic (heads.image_prototype has no randomness for a fixed
    subset), so this is not "re-running an experiment," just re-deriving a
    pure function of already-cached data, exactly as decision 4's exception
    for cached features permits."""
    Z_train, y_train = load_features(cache_root, dataset, "train", encoder)
    if k != "full":
        idx = balanced_kshot(y_train, int(k), seed)
        Z_train, y_train = Z_train[idx], y_train[idx]
    head = ImagePrototype()
    head.fit(Z_train, y_train)
    return head.prototypes


def select_test_indices(
    y_test: torch.Tensor, class_indices: list[int], max_per_class: int | None, seed: int
) -> torch.Tensor:
    """All test examples per class (DTD: 40/class), or a pinned-seed sample
    capped at `max_per_class` (Flowers-102: its test split is imbalanced, so an
    uncapped plot would be dominated by whichever selected class happens to be
    large)."""
    generator = torch.Generator().manual_seed(seed)
    chunks = []
    for c in class_indices:
        idx = torch.nonzero(y_test == c, as_tuple=True)[0]
        if max_per_class is not None and idx.numel() > max_per_class:
            perm = torch.randperm(idx.numel(), generator=generator)[:max_per_class]
            idx = idx[perm].sort().values
        chunks.append(idx)
    return torch.cat(chunks)
