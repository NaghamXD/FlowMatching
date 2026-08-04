"""Shared constants and helpers used across the Stage 1 report: canonical K
ordering, method/encoder labels, and the images-per-class computation that
decision 15 requires to be *asserted* against the actual recorded split sizes,
not assumed."""

from __future__ import annotations

from cvlab.reporting.loader import RunCell

K_ORDER: tuple[int | str, ...] = (5, 10, "full")
METHOD_ORDER = {"linear_probe": 0, "image_prototype": 1}
METHOD_LABEL = {"linear_probe": "Linear Probe", "image_prototype": "Image Prototype"}
ENCODER_LABEL = {"resnet18": "ResNet-18", "dinov2_vits14": "DINOv2 ViT-S/14"}
DATASET_LABEL = {"dtd": "DTD", "flowers102": "Flowers-102"}

# The three dataset x encoder combinations (C1/C2/C3), reused across deliverables
# 3 (training curves) and 5 (feature visualizations).
COMBINATIONS = (("dtd", "resnet18", "C1"), ("dtd", "dinov2_vits14", "C2"), ("flowers102", "resnet18", "C3"))

# Colorblind-safe (Okabe-Ito) qualitative colors, reused wherever an encoder needs
# a stable color across figures.
ENCODER_COLOR = {"resnet18": "#0072B2", "dinov2_vits14": "#D55E00"}

# Colorblind-safe categorical palette (Paul Tol's "Muted" qualitative scheme,
# https://sronpersonalpages.nl/~pault/ -- 9 hues plus a 10th distinguishable
# accent), for the up-to-10 classes shown in deliverable 5's feature-viz figures.
CLASS_VIZ_PALETTE = (
    "#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE",
    "#882255", "#44AA99", "#999933", "#AA4499", "#6699CC",
)

# Decision 15: the official Flowers-102 train split is exactly 10 images/class
# (1020 / 102); DTD's is 40 (1880 / 47). Not a rounding artifact -- the official
# split. Asserted here against the actual recorded split sizes, not assumed, so
# a torchvision version change that breaks this assumption fails loudly instead
# of silently mis-reporting every K=10-vs-full comparison.
EXPECTED_FULL_IMAGES_PER_CLASS = {"dtd": 40, "flowers102": 10}


def images_per_class(cell: RunCell) -> int:
    """K itself, for K in {5, 10}. For K="full", `train_size // num_classes`,
    asserted against the expected value pinned in decision 15."""
    if cell.k != "full":
        return int(cell.k)

    record = cell.records[0]
    computed = record.train_size // record.num_classes
    expected = EXPECTED_FULL_IMAGES_PER_CLASS.get(cell.dataset)
    if expected is not None and computed != expected:
        raise AssertionError(
            f"decision 15 violated for dataset={cell.dataset!r}: full-split train "
            f"has {record.train_size} images / {record.num_classes} classes = "
            f"{computed} images/class, expected {expected}. Every K=10-vs-full "
            f"claim for this dataset needs revisiting before the report can be "
            f"trusted -- not silently proceeding."
        )
    return computed
