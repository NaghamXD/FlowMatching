from __future__ import annotations

from pathlib import Path

from cvlab.reporting.common import COMBINATIONS
from cvlab.reporting.report import make_report

_FIXED_NAMES = {
    "accuracy_table.md",
    "accuracy_table.csv",
    "accuracy_table.tex",
    "accuracy_table_mean_per_class.md",
    "accuracy_table_mean_per_class.csv",
    "accuracy_table_mean_per_class.tex",
    "deliverable2_accuracy_vs_k.png",
    "deliverable2_accuracy_vs_k.pdf",
    "deliverable3_training_curves.png",
    "deliverable3_training_curves.pdf",
    "deliverable3_training_curves_captions.txt",
    "deliverable5a_dtd_feature_viz.png",
    "deliverable5a_dtd_feature_viz.pdf",
    "deliverable5a_dtd_feature_viz_caption.txt",
    "deliverable5b_flowers102_feature_viz.png",
    "deliverable5b_flowers102_feature_viz.pdf",
    "deliverable5b_flowers102_feature_viz_caption.txt",
    "deliverable5c_dtd_kshot_vs_full_prototypes.png",
    "deliverable5c_dtd_kshot_vs_full_prototypes.pdf",
    "deliverable5c_dtd_kshot_vs_full_prototypes_caption.txt",
    "deliverable5_class_selection.json",
}

_PER_COMBINATION_SUFFIXES = (
    "_confusion_full.png",
    "_confusion_full.pdf",
    "_confusion_full_masked.png",
    "_confusion_full_masked.pdf",
    "_confusion_zoomed.png",
    "_confusion_zoomed.pdf",
    "_confusion_zoomed_masked.png",
    "_confusion_zoomed_masked.pdf",
    "_recall_distribution.png",
    "_recall_distribution.pdf",
    "_top_pairs.md",
)


def _expected_names() -> set[str]:
    names = set(_FIXED_NAMES)
    for dataset, encoder, panel_id in COMBINATIONS:
        prefix = f"deliverable4_{panel_id}_{dataset}_{encoder}"
        names.update(prefix + suffix for suffix in _PER_COMBINATION_SUFFIXES)
    return names


# M12 needs real cached features (the results-store-only synthetic fixture has
# none) -- the fixture's num_classes/train_size are deliberately set to match
# the real datasets (conftest.py), so pointing at the project's real feature
# cache is consistent, not a mismatch. Skipped if that cache isn't populated.
import pytest

from cvlab.features.cache import cache_dir

_REAL_CACHE_ROOT = "cache/features"


def _real_cache_populated() -> bool:
    return all(
        (cache_dir(_REAL_CACHE_ROOT, dataset, split, encoder) / "features.pt").exists()
        for dataset, encoder, _ in COMBINATIONS
        for split in ("train", "test")
    )


pytestmark = pytest.mark.skipif(
    not _real_cache_populated(), reason="real feature cache not populated in this environment"
)


def test_make_report_writes_expected_files(synthetic_runs_root: Path, tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    written = make_report(synthetic_runs_root, reports_root, cache_root=_REAL_CACHE_ROOT)

    assert {p.name for p in written} == _expected_names()
    for p in written:
        assert p.exists()
        assert p.stat().st_size > 0


def test_make_report_text_and_figure_artifacts_are_byte_identical_on_rerun(
    synthetic_runs_root: Path, tmp_path: Path
) -> None:
    reports_root = tmp_path / "reports"
    written_1 = make_report(synthetic_runs_root, reports_root, cache_root=_REAL_CACHE_ROOT)
    contents_1 = {p.name: p.read_bytes() for p in written_1}

    written_2 = make_report(synthetic_runs_root, reports_root, cache_root=_REAL_CACHE_ROOT)
    contents_2 = {p.name: p.read_bytes() for p in written_2}

    assert contents_1 == contents_2
