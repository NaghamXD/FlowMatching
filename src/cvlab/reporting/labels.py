"""Class-name lookup for DTD and Flowers-102, needed to label confusion-matrix
axes and feature-visualization legends with real names instead of bare indices.

Boundary note on the "reporting reads only the results store" constraint: this
reads static category-name metadata (Flowers-102's shipped name list; DTD's
`.classes`, derived from its label index files only -- no image is opened, no
pixel is touched). It is not re-running an experiment, re-extracting a feature,
or touching the test split's images, so it falls outside what the constraint
prohibits -- but it's still a second data source alongside the results store,
flagged here rather than left implicit.
"""

from __future__ import annotations

from pathlib import Path

from cvlab.config.loader import load_dataclass, load_yaml
from cvlab.config.schema import DatasetConfig
from cvlab.data.dtd import DTDDataset
from cvlab.data.flowers102 import load_flowers102_classnames

_CACHE: dict[str, tuple[str, ...]] = {}


def get_class_names(dataset: str, configs_dir: str | Path = "configs") -> tuple[str, ...]:
    if dataset in _CACHE:
        return _CACHE[dataset]

    if dataset == "flowers102":
        names = load_flowers102_classnames()
    elif dataset == "dtd":
        dataset_cfg = load_dataclass(
            DatasetConfig, load_yaml(Path(configs_dir) / "datasets" / "dtd.yaml")
        )
        # partition=1, split="train" is enough to enumerate all 47 class names --
        # DTD's classes are read from the label index file, not from any image.
        names = DTDDataset(root=dataset_cfg.root, split="train", partition=dataset_cfg.partition).metadata.class_names
    else:
        raise ValueError(f"Unknown dataset {dataset!r}")

    _CACHE[dataset] = names
    return names
